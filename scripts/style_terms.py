"""Writing-style index across all papers, and stats for selected content terms in agent papers.
Usage: python style_terms.py <full.csv> <term_trends.csv> <out_dir>"""
import sys, re, json, numpy as np, pandas as pd
sys.path.insert(0, __import__('os').path.dirname(__file__))
from scipy import stats
import statsmodels.api as sm
SRC, TT, OUT = sys.argv[1:4]
T0 = pd.Timestamp('2025-11-01')
f = pd.read_csv(SRC, parse_dates=['published_at'])
R = f[(f.feed_type == 'research')].copy()
R['descr'] = R.description.fillna('').str.replace(r'^arXiv:\S+ Announce Type: \S+\s*Abstract:\s*', '', regex=True)
R['announce'] = R.description.fillna('').str.extract(r'Announce Type: (\S+)')[0]
R['txt'] = (R.title.fillna('') + ' . ' + R.descr).str.lower()
R['week'] = R.published_at.dt.to_period('W-SUN').dt.start_time
R['month'] = R.published_at.dt.to_period('M').astype(str)
wk = R.groupby('week').size(); R = R[R.week.isin(wk[wk >= 300].index)]
# Word lists: "LLM-era stock vocabulary" drawn from the excess-vocabulary literature (Kobak et al. 2024; Liang et al. 2024)
STOCK = ['delve', 'delves', 'leverag', 'comprehensive', 'crucial', 'pivotal', 'intricate', 'meticulous', 'showcas', 'underscor', 'notabl', 'facilitat', 'enhanc', 'highlight', 'seamless', 'robustness and', 'landscape', 'realm', 'innovative', 'invaluable']
PLAIN = ['keeps', 'writes', 'reads', 'stays', 'leaves', 'unchanged', 'same', 'every', 'because', 'whether', 'frozen', 'exactly', 'none', 'fixed']
def idx(words):
    pat = r'\b(?:' + '|'.join(re.escape(w) for w in words) + r')'
    return R.txt.str.count(pat) / (R.txt.str.split().str.len() / 100)  # per 100 words
R['stock'] = idx(STOCK); R['plain'] = idx(PLAIN)
def wtrend(col, D):
    g = D.groupby('week')[col].mean(); t = np.array([((w + pd.Timedelta(days=3)) - T0).days / 30.44 for w in g.index])
    m = sm.OLS(g.values, sm.add_constant(t)).fit(cov_type='HAC', cov_kwds={'maxlags': 4})
    return dict(slope_per_month=m.params[1], p=m.pvalues[1], start=m.params[0], monthly=D.groupby('month')[col].mean().round(4).to_dict())
out = {}
for label, D in [('all', R), ('new', R[R.announce == 'new']), ('replace', R[R.announce.isin(['replace', 'replace-cross'])])]:
    out[label] = {c: wtrend(c, D) for c in ['stock', 'plain']}
    print(label, len(D), {c: (round(out[label][c]['slope_per_month'], 4), f"{out[label][c]['p']:.2g}") for c in ['stock', 'plain']})
    print('   stock', out[label]['stock']['monthly']); print('   plain', out[label]['plain']['monthly'])
# selected content terms
tt = pd.read_csv(TT)
CONTENT_UP = ['harness', 'frozen', 'reusable', 'skill', 'skills', 'deterministic', 'budget', 'evidence', 'on-policy distillation', 'self-distillation', 'qwen3', 'gate', 'verifier', 'verifiable', 'rubric', 'trajectories', 'terminal', 'long-horizon', 'context engineering', 'sandbox', 'permissions', 'audit', 'traces', 'credit assignment', 'recursive', 'compaction', 'world model', 'coding agents', 'coding agent', 'claude code', 'codex']
CONTENT_DOWN = ['test-time scaling', 'multi-agent collaboration', 'reasoning capabilities', 'multimodal reasoning', 'llm-powered', 'human-like', 'context-aware', 'step-by-step', 'dialogues', 'chain-of-thought', 'role-playing', 'prompt engineering', 'retrieval-augmented', 'multi-agent framework', 'multi-agent system', 'collaborative', 'synthetic', 'zero-shot', 'gpt-4o', 'embodied', 'navigation', 'question answering']
sel = tt[tt.term.isin(CONTENT_UP + CONTENT_DOWN)].copy()
sel['set'] = np.where(sel.term.isin(CONTENT_UP), 'up', 'down')
pd.set_option('display.width', 250)
print(sel.sort_values('slope', ascending=False)[['term', 'set', 'docs', 'first_half_pct', 'second_half_pct', 'growth_pct_per_month', 'proj_apr2027_pct', 'proj_lo', 'proj_hi', 'q']].round(3).to_string(index=False))
sel.to_csv(f'{OUT}/content_terms.csv', index=False)
json.dump(out, open(f'{OUT}/style.json', 'w'), indent=1, default=float)
