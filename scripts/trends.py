"""Statistical trend tests and forecasts for the agent corpus.

Usage: python trends.py <full.csv export> <out_dir>

Unit of analysis: ISO weeks (Mon-start). Weeks with collection gaps are dropped
(papers: < 300 announcements; news: < 1,500 stories).
Trend model: quasi-binomial GLM, logit(share) = a + b * months, scale from Pearson chi2.
Multiple testing: Benjamini-Hochberg FDR within each family of tests.
Forecasts: the same trend model projected to April 2027, with 95% CIs for the trend
(delta method on the logit scale). Backtest: fit Nov 2025 - Jun 2026, predict monthly
shares Jul - Oct 2026, compare MAE with a naive forecast (mean of the last 3 fitted months).
"""
import re, sys, json, itertools
import numpy as np, pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from scipy import stats

SRC, OUT = sys.argv[1:3]
T0 = pd.Timestamp('2025-11-01')
HORIZON = pd.Timestamp('2027-04-15')
HORIZON_SHORT = pd.Timestamp('2027-01-15')

f = pd.read_csv(SRC, parse_dates=['published_at'])
f = f[f.published_at < '2028-01-01'].copy()
f['descr'] = f.description.fillna('').str.replace(r'^arXiv:\S+ Announce Type: \S+\s*Abstract:\s*', '', regex=True)
f['announce'] = f.description.fillna('').str.extract(r'Announce Type: (\S+)')[0]
f['txt'] = (f.title.fillna('') + ' . ' + f.descr.str.slice(0, 1500)).str.lower()
AG = r'\bagent(?:s|ic)?\b|\bmulti-?agents?\b|\bagentic\b'
EX = r'\bice\b|immigration|\bfbi\b|federal agents?|border patrol|secret service|travel agents?|real estate|free agents?|nerve agent|agent orange|double agent|undercover|deport|shooting|sports agent|talent agent|\bdea\b|\bcbp\b|homeland'
ST = r'\bai\b|agentic|\bllms?\b|chatbot|language model|openai|anthropic|claude|gemini|copilot'
f['agent'] = f.txt.str.contains(AG) & ~(f.txt.str.contains(EX) & ~f.txt.str.contains(ST))
f['week'] = f.published_at.dt.to_period('W-SUN').dt.start_time
f['t'] = (f.published_at - T0).dt.days / 30.44
f['month'] = f.published_at.dt.to_period('M').astype(str)

R = f[f.feed_type == 'research'].copy()
N = f[f.feed_type == 'news'].copy()
wk_R = R.groupby('week').size(); good_R = wk_R[wk_R >= 300].index
wk_N = N.groupby('week').size(); good_N = wk_N[wk_N >= 1500].index
R = R[R.week.isin(good_R)]; N = N[N.week.isin(good_N)]
print('weeks kept: papers', len(good_R), 'of', len(wk_R), '| news', len(good_N), 'of', len(wk_N))
RA = R[R.agent].copy(); NA = N[N.agent].copy()

def week_t(w):  # week midpoint in months since T0
    return ((w + pd.Timedelta(days=3)) - T0).days / 30.44

def fit(k, n, wt):
    """quasi-binomial logistic trend. k, n arrays per week; wt month index."""
    k = np.asarray(k, float); n = np.asarray(n, float); x = np.asarray(wt, float)
    X = sm.add_constant(x)
    m = sm.GLM(np.column_stack([k, n - k]), X, family=sm.families.Binomial()).fit()
    # quasi-binomial: Pearson dispersion computed by hand, floored at 1
    mu = m.predict(X)
    chi2 = np.sum((k - n * mu) ** 2 / np.maximum(n * mu * (1 - mu), 1e-12))
    phi = max(1.0, chi2 / (len(k) - 2))
    return _Q(m.params, m.cov_params() * phi, phi)

class _Q:
    def __init__(self, params, cov, phi):
        self.params = np.asarray(params); self._cov = np.asarray(cov); self.scale = phi
        self.bse = np.sqrt(np.diag(self._cov))
        z = self.params / self.bse
        self.pvalues = 2 * stats.norm.sf(np.abs(z))
    def cov_params(self):
        return self._cov

def trend(df_flag, base, label_fn=None):
    """df_flag: boolean Series aligned to base rows. Returns dict with slope etc."""
    g = pd.DataFrame({'week': base.week, 'hit': df_flag.astype(int)}).groupby('week').hit.agg(['sum', 'size'])
    wt = [week_t(w) for w in g.index]
    m = fit(g['sum'], g['size'], wt)
    b, se = m.params[1], m.bse[1]; p = m.pvalues[1]
    # projected share at horizon and at end of data, with 95% CI on logit scale
    def proj(tm):
        xv = np.array([1, tm]); eta = xv @ m.params; v = xv @ m.cov_params() @ xv
        lo, hi = eta - 1.96 * np.sqrt(v), eta + 1.96 * np.sqrt(v)
        s = lambda z: 100 / (1 + np.exp(-z))
        return s(eta), s(lo), s(hi)
    t_end = week_t(base.week.max()); t_h = (HORIZON - T0).days / 30.44
    now, now_lo, now_hi = proj(t_end); fut, fut_lo, fut_hi = proj(t_h)
    s3, s3_lo, s3_hi = proj((HORIZON_SHORT - T0).days / 30.44)
    # Mann-Kendall-style check: Kendall tau on monthly shares
    mo = pd.DataFrame({'m': base.month, 'hit': df_flag.astype(int)}).groupby('m').hit.mean()
    tau, ptau = stats.kendalltau(np.arange(len(mo)), mo.values)
    return dict(slope_logodds_per_month=b, se=se, p=p, growth_pct_per_month=100 * (np.exp(b) - 1),
                share_all_pct=100 * df_flag.mean(), docs=int(df_flag.sum()),
                fitted_now_pct=now, proj_apr2027_pct=fut, proj_lo=fut_lo, proj_hi=fut_hi, proj_jan2027_pct=s3, proj_jan_lo=s3_lo, proj_jan_hi=s3_hi, intercept=m.params[0],
                kendall_tau=tau, kendall_p=ptau, dispersion=m.scale,
                monthly_pct=[round(100 * v, 2) for v in mo.values])

def bh(rows, key='p'):
    if not rows: return rows
    q = multipletests([r[key] for r in rows], method='fdr_bh')[1]
    for r, qq in zip(rows, q): r['q'] = qq
    return rows

def backtest(flag, base, cut='2026-07-01'):
    """Fit on weeks before cut; forecast monthly shares after; MAE vs naive."""
    b = base.assign(hit=flag.astype(int))
    tr = b[b.published_at < cut]; te = b[b.published_at >= cut]
    g = tr.groupby('week').hit.agg(['sum', 'size'])
    m = fit(g['sum'], g['size'], [week_t(w) for w in g.index])
    te_m = te.groupby('month').agg(share=('hit', 'mean'), t=('t', 'mean'))
    pred = 1 / (1 + np.exp(-(m.params[0] + m.params[1] * te_m.t)))
    tr_m = tr.groupby('month').hit.mean()
    naive = tr_m.iloc[-3:].mean()
    ens = (pred + naive) / 2
    return (float(np.mean(np.abs(pred - te_m.share))) * 100, float(np.mean(np.abs(naive - te_m.share))) * 100,
            float(np.mean(np.abs(ens - te_m.share))) * 100)

res = {}
# ---------- 1. overall agent share
res['overall'] = {'papers': trend(R.agent, R), 'news': trend(N.agent, N)}
res['overall_backtest'] = {'papers': backtest(R.agent, R), 'news': backtest(N.agent, N)}
for k, v in res['overall'].items():
    print(k, {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items() if kk != 'monthly_pct'})
print('backtest MAE (model, naive):', res['overall_backtest'])

# ---------- 2. topics (within agent papers)
TOP = {'Benchmarks & evaluation': r'benchmark|\bevaluation\b|leaderboard', 'RL training': r'reinforcement learning|\brl\b|grpo|policy optimization|\brlvr\b',
 'Memory & long-horizon': r'\bmemory\b|long[- ]horizon|long-term', 'Tool use': r'tool[- ]use|tool[- ]calling|function[- ]calling|tool-augmented|tool learning|\btools?\b',
 'Multi-agent systems': r'multi-?agent', 'Safety & security': r'safety|secur|prompt injection|jailbreak|\battacks?\b|misalign|guardrail|red[- ]team',
 'Planning': r'\bplanning\b|\bplanner\b', 'Coding & software': r'swe-bench|software engineering|coding agents?|code agents?|repositor|program repair|\bcoding\b',
 'Embodied & robotics': r'embodied|robot|vision-language-action|\bvla\b', 'GUI, web & computer use': r'\bgui\b|computer[- ]use|web agents?|browser|web navigation|mobile agents?|\bos agents?|screen',
 'Self-improving agents': r'self-evolv|self-improv|self-reflect|evolving agents?', 'Deep research & science': r'deep research|scientific discovery|research agents?|ai scientist|literature review',
 'MCP & protocols': r'model context protocol|\bmcp\b|agent-to-agent|\ba2a\b'}
rows = []
for k, p in TOP.items():
    fl = RA.txt.str.contains(p)
    r = trend(fl, RA); r['name'] = k; r['backtest'] = backtest(fl, RA); rows.append(r)
res['topics'] = bh(rows)

# ---------- 3. vocabulary: unigrams + bigrams in agent paper title+abstract
STOP = set("""a an the of for and in on with to via from by is as at towards toward agent agents agentic llm llms language models model large using
based through into are can we does not when what how across over study this that these those our their its it be or vs you they which while such
both each than then also more most less only other into within without under between after before about via upon however thus here there where
show shows shown propose proposes proposed present presents introduce introduces novel new approach approaches method methods framework frameworks
paper work results result performance task tasks existing first two three one use used uses achieve achieves significantly further across has have
been were was do while may well even often still across among per like many much several various different multiple key specific given due ai
data learning system systems et al e.g i.e state-of-the-art sota experiments experimental extensive demonstrate demonstrates""".split())
def grams(t):
    w = re.findall(r'[a-z][a-z0-9\-]+', t)
    g = {x for x in w if x not in STOP and len(x) > 2}
    g |= {a + ' ' + b for a, b in zip(w, w[1:]) if a not in STOP and b not in STOP and len(a) > 2 and len(b) > 2}
    return g
docs = [grams(t) for t in RA.txt]
df_ = pd.Series(itertools.chain.from_iterable(docs)).value_counts()
cand = df_[df_ >= 40].index.tolist()
print('candidate terms', len(cand))
week_arr = RA.week.values
gw = pd.Series(1, index=RA.week).groupby(level=0).size()
wts = np.array([week_t(w) for w in gw.index])
pos = {w: i for i, w in enumerate(gw.index)}
widx = np.array([pos[w] for w in week_arr])
T1 = RA.published_at < '2026-05-01'
term_rows = []
for term in cand:
    hit = np.fromiter((term in d for d in docs), bool, len(docs))
    k = np.bincount(widx, weights=hit, minlength=len(gw))
    try:
        m = fit(k, gw.values, wts)
    except Exception:
        continue
    b, p = m.params[1], m.pvalues[1]
    xv = np.array([1, (HORIZON - T0).days / 30.44]); eta = xv @ m.params; v = xv @ m.cov_params() @ xv
    xs = np.array([1, (HORIZON_SHORT - T0).days / 30.44]); es = xs @ m.params; vs = xs @ m.cov_params() @ xs
    sg = lambda z: 100 / (1 + np.exp(-z))
    a1 = hit[T1.values].mean(); a2 = hit[~T1.values].mean()
    term_rows.append(dict(term=term, docs=int(hit.sum()), share_pct=100 * hit.mean(), first_half_pct=100 * a1, second_half_pct=100 * a2,
                          slope=b, p=p, growth_pct_per_month=100 * (np.exp(b) - 1),
                          proj_apr2027_pct=100 / (1 + np.exp(-eta)), proj_lo=100 / (1 + np.exp(-(eta - 1.96 * np.sqrt(v)))), proj_hi=100 / (1 + np.exp(-(eta + 1.96 * np.sqrt(v)))),
                          proj_jan2027_pct=sg(es), proj_jan_lo=sg(es - 1.96 * np.sqrt(vs)), proj_jan_hi=sg(es + 1.96 * np.sqrt(vs)), fitted_now_pct=sg(m.params[0] + m.params[1] * wts.max()), dispersion=m.scale))
term_rows = bh(term_rows)
tr = pd.DataFrame(term_rows)
sig = tr[tr.q < 0.05]
print('terms tested', len(tr), '| significant at q<0.05:', len(sig), 'rising', int((sig.slope > 0).sum()), 'fading', int((sig.slope < 0).sum()))
tr.to_csv(f'{OUT}/term_trends.csv', index=False)

# ---------- 4. base models and benchmarks named in agent papers (and all papers)
MODELS = {'Qwen': r'\bqwen', 'GPT-4o / GPT-4': r'\bgpt-?4(?:o|\b|-turbo)', 'GPT-5': r'\bgpt-?5', 'Claude': r'\bclaude', 'Gemini': r'\bgemini',
          'Llama': r'\bllama', 'DeepSeek': r'deepseek', 'Mistral': r'\bmistral|\bmixtral', 'Gemma': r'\bgemma', 'Kimi': r'\bkimi', 'GLM': r'\bglm-?\d', 'o-series (o1/o3/o4)': r'\bo[134](?:-mini|-pro)?\b(?![-\.]\d)'}
BENCH = {'SWE-bench': r'swe-?bench', 'Terminal-Bench': r'terminal-?bench', 'OSWorld': r'osworld', 'WebArena': r'webarena', 'GAIA': r'\bgaia\b',
         'tau-bench': r'tau-?bench|τ-bench|\\tau.?-bench|tau\^?2-bench', 'BrowseComp': r'browsecomp', "Humanity's Last Exam": r"humanity'?s last exam", 'ALFWorld': r'alfworld',
         'WebShop': r'webshop', 'MLE-bench': r'mle-?bench', 'BFCL': r'\bbfcl\b', 'AndroidWorld': r'androidworld', 'Mind2Web': r'mind2web', 'HotpotQA': r'hotpotqa'}
for name, D in [('models', MODELS), ('benchmarks', BENCH)]:
    rows = []
    for k, p in D.items():
        fl = RA.txt.str.contains(p)
        if fl.sum() < 15: continue
        r = trend(fl, RA); r['name'] = k; rows.append(r)
    res[name] = bh(rows)

# ---------- 5. arXiv category slices: agent share within each feed
rows = []
for c, g in R.groupby('feed_name'):
    r = trend(g.agent, g); r['name'] = c.replace(' updates on arXiv.org', '').replace(' arXiv.org', ''); rows.append(r)
res['categories'] = bh(rows)

# ---------- 6. news: source groups, themes, companies
GEN = ['NY Times', 'NYT > Technology', 'Bloomberg Technology', 'Bloomberg AI', 'International Business Times', 'Business Insider', 'Business Latest', 'AI | The Guardian',
       'Globalnews CA', 'Rest of World - Latest Stories', 'Scientific American', 'New Scientist', 'AI - The Conversation', 'Quanta Magazine', 'MIT Technology Review',
       'MIT Technology Review - Computing', 'IEEE Spectrum', 'Sifted', 'Crunchbase News', 'Tech.eu', 'Digital Journal', 'Futurism', 'Gizmodo']
TECH = ['TechCrunch', 'Techmeme', 'TechSpot', 'ZDNet AI', 'Ars Technica - All content', 'Wired', 'The Verge -  All Posts', 'THE DECODER', 'Silicon Republic', 'The Next Web',
        'AI | TechRepublic', 'gHacks Technology News', 'The Register', '404 Media', 'EE Times', 'AI - SiliconANGLE', 'SiliconANGLE', 'InfoQ blog', 'aibusiness', 'Analytics India',
        'MarkTechPost', 'AIWire', 'AI News', 'AI News | VentureBeat', 'Tech Monitor', 'The New Stack', 'BigDATAwire', 'Dataconomy', 'AI-Tech Park', 'AI KnowTechie', 'The Stack']
DEV = ['AI Agents Subreddit', 'Hacker news', 'DEV Community', 'Product Hunt', 'Towards AI - Medium', 'HackerNoon - ai', 'Stack Overflow Blog', 'Analytics Vidhya', 'KDnuggets', 'R-bloggers']
N['group'] = np.where(N.feed_name.isin(GEN), 'General & business press', np.where(N.feed_name.isin(TECH), 'Tech press', np.where(N.feed_name.isin(DEV), 'Developer community', 'Vendors, newsletters & video')))
rows = []
for gname, g in N.groupby('group'):
    r = trend(g.agent, g); r['name'] = gname; r['stories'] = len(g); rows.append(r)
res['news_groups'] = bh(rows)
NA = N[N.agent].copy()
ENT = {'Anthropic / Claude': r'anthropic|claude', 'OpenAI': r'openai|chatgpt|\bgpt-?\d|codex', 'Google / Gemini': r'google|gemini|deepmind', 'Meta': r'\bmeta\b|llama',
       'Microsoft / Copilot': r'microsoft|copilot', 'Amazon / AWS': r'amazon|\baws\b|bedrock', 'Nvidia': r'nvidia', 'OpenClaw': r'openclaw|clawdbot|moltbot|moltbook'}
NT = {'Enterprise adoption': r'enterprise|business|customer|workflow|productivity', 'Coding agents': r'coding|developer|code\b|software engineer',
      'Security & risk': r'secur|prompt injection|hack|vulnerab|breach|malware|risk', 'Regulation & policy': r'regulat|law\b|policy|congress|\beu\b|govern',
      'Jobs & workforce': r'\bjobs?\b|layoff|workforce|worker|employ|hiring', 'Funding & deals': r'raises|funding|valuation|acqui|\$\d+\s?(?:m|b|million|billion)|series [a-d]',
      'Shopping & payments': r'shopping|commerce|checkout|payment|purchase', 'Browsers & computer use': r'browser|computer use|operator\b|comet\b',
      'Incidents & rogue agents': r'rogue|incident|went rogue|deleted|wiped|out of control|misbehav'}
for name, D in [('news_companies', ENT), ('news_themes', NT)]:
    rows = []
    for k, p in D.items():
        fl = NA.txt.str.contains(p); r = trend(fl, NA); r['name'] = k; rows.append(r)
    res[name] = bh(rows)

# ---------- 7. topic co-occurrence shift (agent papers): lift H1 vs H2
codes = {k: RA.txt.str.contains(p).values for k, p in TOP.items()}
h1 = (RA.published_at < '2026-05-01').values
co = []
for a, b in itertools.combinations(TOP, 2):
    def lift(mask):
        A, B = codes[a][mask], codes[b][mask]
        pa, pb, pab = A.mean(), B.mean(), (A & B).mean()
        return pab / (pa * pb) if pa * pb > 0 else np.nan, int((A & B).sum())
    l1, n1 = lift(h1); l2, n2 = lift(~h1)
    # Fisher-style test of change in joint share: two-proportion z on co-occurrence
    tbl = [[(codes[a][h1] & codes[b][h1]).sum(), h1.sum() - (codes[a][h1] & codes[b][h1]).sum()],
           [(codes[a][~h1] & codes[b][~h1]).sum(), (~h1).sum() - (codes[a][~h1] & codes[b][~h1]).sum()]]
    _, pchi, _, _ = stats.chi2_contingency(tbl)
    co.append(dict(a=a, b=b, lift_h1=l1, lift_h2=l2, joint_h1_pct=100 * n1 / h1.sum(), joint_h2_pct=100 * n2 / (~h1).sum(), p=pchi))
res['cooccurrence'] = bh(co)

# ---------- 8. backtest summary across topics
bt = np.array([r['backtest'] for r in res['topics']])
res['backtest_summary'] = dict(topics_model_mae=float(bt[:, 0].mean()), topics_naive_mae=float(bt[:, 1].mean()), topics_ens_mae=float(bt[:, 2].mean()),
                              model_wins=int((bt[:, 0] < bt[:, 1]).sum()), ens_wins=int((bt[:, 2] < bt[:, 1]).sum()), n=len(bt))
print('topic backtest', res['backtest_summary'])

def clean(o):
    if isinstance(o, dict): return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [clean(v) for v in o]
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, np.bool_): return bool(o)
    return o
json.dump(clean(res), open(f'{OUT}/trends.json', 'w'), indent=1)
for sec in ['topics', 'models', 'benchmarks', 'categories', 'news_groups', 'news_companies', 'news_themes']:
    print('\n==', sec)
    for r in sorted(res[sec], key=lambda r: -r['slope_logodds_per_month']):
        print(f"  {r['name'][:28]:28s} share {r['share_all_pct']:5.1f}%  growth {r['growth_pct_per_month']:+6.1f}%/mo  q={r['q']:.3g}  tau={r['kendall_tau']:+.2f}  now {r['fitted_now_pct']:5.1f}% -> Apr27 {r['proj_apr2027_pct']:5.1f}% [{r['proj_lo']:.1f},{r['proj_hi']:.1f}]" + (f"  bt {r['backtest'][0]:.2f}/{r['backtest'][1]:.2f}/{r['backtest'][2]:.2f}" if 'backtest' in r else '') + f"  phi={r['dispersion']:.1f}")
print('\n== co-occurrence biggest lift changes (q<0.05)')
for r in sorted([c for c in res['cooccurrence'] if c['q'] < 0.05], key=lambda c: -(c['joint_h2_pct'] - c['joint_h1_pct']))[:12]:
    print(f"  {r['a']} x {r['b']}: joint {r['joint_h1_pct']:.1f}% -> {r['joint_h2_pct']:.1f}%  lift {r['lift_h1']:.2f} -> {r['lift_h2']:.2f} q={r['q']:.2g}")
