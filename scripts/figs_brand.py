"""Quome-branded figures for the paper.
Usage: python figs_brand.py <full.csv> <stats_dir> <fonts_dir> <out_dir>"""
import sys, json, glob, datetime as dt
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.dates as md
from matplotlib import font_manager as fm
import statsmodels.api as sm
from adjustText import adjust_text

SRC, ST, FONTS, OUT = sys.argv[1:5]
for f in glob.glob(f'{FONTS}/*.ttf'): fm.fontManager.addfont(f)
VIOLET, PURPLE, BLUE, VOID, DEEP, SILVER, MUTED, INK = '#8c7ff2', '#7c5ceb', '#3985ff', '#0a0a10', '#1a1a24', '#b4b4c0', '#71717a', '#1a1a24'
FADE = '#9a9aa6'; GRID = '#ececf3'; LAV = '#ece9fd'
plt.rcParams.update({'font.family': 'Source Sans 3', 'font.size': 9, 'axes.titlesize': 10, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left',
                     'axes.labelsize': 8.5, 'axes.labelcolor': '#3f3f46', 'xtick.color': '#52525b', 'ytick.color': '#52525b', 'axes.edgecolor': '#c9c9d3',
                     'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.04,
                     'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.7, 'axes.axisbelow': True, 'text.color': INK})
TITLE = dict(fontfamily='Sora', fontweight='bold', fontsize=10, color=INK)
T = json.load(open(f'{ST}/trends.json')); STY = json.load(open(f'{ST}/style.json'))
CT = pd.read_csv(f'{ST}/content_terms.csv')
T0 = pd.Timestamp('2025-11-01')
months = [f'2025-{m:02d}' for m in (11, 12)] + [f'2026-{m:02d}' for m in range(1, 11)]
mdate = [dt.date(int(m[:4]), int(m[5:]), 15) for m in months]
def save(fig, name): fig.savefig(f'{OUT}/{name}.pdf'); plt.close(fig)
def gci(r):
    b, se = r['slope_logodds_per_month'], r['se']
    return 100 * (np.exp(b) - 1), 100 * (np.exp(b - 1.96 * se) - 1), 100 * (np.exp(b + 1.96 * se) - 1)

# ---------- F1 hero: share + trend + forecast fan
f = pd.read_csv(SRC, usecols=['title', 'description', 'feed_type', 'published_at'], parse_dates=['published_at'])
f = f[f.published_at < '2028-01-01']
f['txt'] = (f.title.fillna('') + ' . ' + f.description.fillna('').str.replace(r'^arXiv:\S+ Announce Type: \S+\s*Abstract:\s*', '', regex=True).str.slice(0, 1500)).str.lower()
AG = r'\bagent(?:s|ic)?\b|\bmulti-?agents?\b|\bagentic\b'
EX = r'\bice\b|immigration|\bfbi\b|federal agents?|border patrol|secret service|travel agents?|real estate|free agents?|nerve agent|agent orange|double agent|undercover|deport|shooting|sports agent|talent agent|\bdea\b|\bcbp\b|homeland'
STR = r'\bai\b|agentic|\bllms?\b|chatbot|language model|openai|anthropic|claude|gemini|copilot'
f['agent'] = f.txt.str.contains(AG) & ~(f.txt.str.contains(EX) & ~f.txt.str.contains(STR))
f['week'] = f.published_at.dt.to_period('W-SUN').dt.start_time
fig, ax = plt.subplots(figsize=(7.0, 3.3))
for ft, col, lab, minw in [('research', VIOLET, 'arXiv papers', 300), ('news', BLUE, 'News stories', 1500)]:
    g = f[f.feed_type == ft].groupby('week').agent.agg(['sum', 'size']); g = g[g['size'] >= minw]
    t = np.array([((w + pd.Timedelta(days=3)) - T0).days / 30.44 for w in g.index]); X = sm.add_constant(t)
    m = sm.GLM(np.column_stack([g['sum'], g['size'] - g['sum']]), X, family=sm.families.Binomial()).fit()
    mu = m.predict(X); phi = max(1, np.sum((g['sum'] - g['size'] * mu) ** 2 / (g['size'] * mu * (1 - mu))) / (len(g) - 2))
    cov = np.asarray(m.cov_params()) * phi
    wd = [w + pd.Timedelta(days=3) for w in g.index]
    ax.scatter(wd, 100 * g['sum'] / g['size'], s=7, color=col, alpha=.28, lw=0)
    tt = np.linspace(t.min(), (pd.Timestamp('2027-04-15') - T0).days / 30.44, 200)
    XX = sm.add_constant(tt); eta = XX @ np.asarray(m.params); se = np.sqrt(np.einsum('ij,jk,ik->i', XX, cov, XX))
    s = lambda z: 100 / (1 + np.exp(-z))
    dd = [T0 + pd.Timedelta(days=x * 30.44) for x in tt]
    cut = pd.Timestamp('2026-10-26'); obs = np.array([d <= cut for d in dd])
    ax.plot(np.array(dd)[obs], s(eta)[obs], color=col, lw=2.2, label=lab, solid_capstyle='round')
    ax.plot(np.array(dd)[~obs], s(eta)[~obs], color=col, lw=2, ls=(0, (3, 2)))
    ax.fill_between(np.array(dd)[~obs], s(eta - 1.96 * se)[~obs], s(eta + 1.96 * se)[~obs], color=col, alpha=.14, lw=0)
    end = s(eta)[-1]
    ax.annotate(f'{end:.1f}%', (dd[-1], end), xytext=(4, 0), textcoords='offset points', va='center', fontsize=8.5, fontweight='bold', color=col)
ax.axvspan(pd.Timestamp('2026-10-26'), pd.Timestamp('2027-04-15'), color=LAV, alpha=.55, lw=0, zorder=0)
ax.text(pd.Timestamp('2027-01-20'), 3.2, 'FORECAST', ha='center', fontsize=7.5, color=PURPLE, fontfamily='Sora', fontweight='bold')
ax.set_ylim(0, 25); ax.set_ylabel('Share mentioning agents (%)')
ax.set_xlim(pd.Timestamp('2025-11-01'), pd.Timestamp('2027-05-25'))
ax.xaxis.set_major_locator(md.MonthLocator(bymonth=[1, 4, 7, 10])); ax.xaxis.set_major_formatter(md.DateFormatter('%b\n%Y'))
ax.legend(loc='upper left', ncol=2, fontsize=8.5)
save(fig, 'b1_hero_forecast')

# ---------- F2 topic forest
tp = sorted(T['topics'], key=lambda r: r['slope_logodds_per_month'])
fig, ax = plt.subplots(figsize=(6.4, 4.0))
for i, r in enumerate(tp):
    g, lo, hi = gci(r); sig = r['q'] < 0.05; col = VIOLET if g > 0 else MUTED
    ax.plot([lo, hi], [i, i], color=col, lw=2.2 if sig else 1.2, alpha=1 if sig else .55, solid_capstyle='round')
    ax.scatter([g], [i], s=18 + r['share_all_pct'] * 4, color=col if sig else 'white', edgecolor=col, lw=1.4, zorder=3)
    ax.text(max(hi, 0) + .6, i, f"{g:+.1f}%/mo" + ('' if sig else '  n.s.'), va='center', fontsize=7.5, color='#3f3f46' if sig else FADE)
ax.axvline(0, color='#a1a1aa', lw=.9)
ax.set_yticks(range(len(tp))); ax.set_yticklabels([r['name'] for r in tp]); ax.grid(axis='y', visible=False)
ax.set_xlabel('Monthly change in odds that an agent paper carries the topic (%, 95% CI)'); ax.set_xlim(-12, 19)
save(fig, 'b2_topic_forest')

# ---------- F3 lifecycle map (content terms + topics)
pts = []
for _, r in CT.iterrows():
    pts.append((r.term, r.fitted_now_pct, r.growth_pct_per_month, r.q < 0.05, 'term'))
for r in T['topics']:
    pts.append((r['name'], r['fitted_now_pct'], r['growth_pct_per_month'], r['q'] < 0.05, 'topic'))
fig, ax = plt.subplots(figsize=(7.2, 5.4))
ax.set_xscale('log'); ax.set_xlim(0.13, 80); ax.set_ylim(-34, 60)
ax.axhline(0, color='#a1a1aa', lw=.9); ax.axvline(5, color='#d4d4dc', lw=.9, ls='--')
for (x0, x1, y0, y1, lab, ha, xa, ya) in [(0.15, 5, 0, 58, 'EMERGING', 'left', 0.145, 57), (5, 70, 0, 58, 'MAINSTREAMING', 'right', 75, 57),
                                          (0.15, 5, -22, 0, 'FADING NICHE', 'left', 0.145, -31.5), (5, 70, -22, 0, 'DECLINING CORE', 'right', 75, -31.5)]:
    ax.text(xa, ya, lab, ha=ha, va='center', fontsize=7.5, color=PURPLE if y0 >= 0 else MUTED, fontfamily='Sora', fontweight='bold', alpha=.85)
texts = []
for name, x, y, sig, kind in pts:
    col = VIOLET if y > 0 else MUTED
    if kind == 'topic': col = BLUE if y > 0 else '#52525b'
    ax.scatter([x], [y], s=34 if kind == 'topic' else 20, marker='s' if kind == 'topic' else 'o', color=col if sig else 'white', edgecolor=col, lw=1.1, zorder=3)
    KEEP_DOWN = {'test-time scaling', 'chain-of-thought', 'multi-agent collaboration', 'reasoning capabilities', 'collaborative', 'dialogues', 'embodied'}
    if kind == 'topic' or (sig and (y > 0 or name in KEEP_DOWN)):
        texts.append(ax.text(x, y, name, fontsize=6.8 if kind == 'term' else 7.2, color='#27272a', fontweight='bold' if kind == 'topic' else 'normal'))
adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle='-', color='#c4c4cc', lw=.5), expand=(1.35, 1.7), force_text=(0.6, 0.9), max_move=40, iter_lim=2000)
ax.set_xlabel('Current share of agent papers (%, fitted, log scale)'); ax.set_ylabel('Growth (% per month, odds)')
ax.set_xticks([0.2, 0.5, 1, 2, 5, 10, 20, 50]); ax.set_xticklabels(['0.2', '0.5', '1', '2', '5', '10', '20', '50'])
ax.scatter([], [], marker='s', color=BLUE, label='Topic code'); ax.scatter([], [], marker='o', color=VIOLET, label='Term (filled: q < 0.05)')
ax.legend(loc='upper center', ncol=2, fontsize=7.5, bbox_to_anchor=(.5, 1.06))
save(fig, 'b3_lifecycle')

# ---------- F4 base models
MS = {r['name']: r for r in T['models']}
show = [('Qwen', VIOLET, 2.2), ('GPT-5', BLUE, 1.8), ('Claude', PURPLE, 1.8), ('Gemini', '#a1a1aa', 1.2), ('GPT-4o / GPT-4', '#71717a', 1.2), ('Llama', '#c4c4cc', 1.2), ('DeepSeek', '#d4d4d8', 1.2)]
fig, ax = plt.subplots(figsize=(6.6, 3.1))
labs = []
for name, col, lw in show:
    y = MS[name]['monthly_pct']; ax.plot(mdate, y, color=col, lw=lw, marker='o', ms=2.4, solid_capstyle='round')
    labs.append([y[-1], name, col])
labs.sort(); prev = -9
for v, name, col in labs:
    yy = max(v, prev + 0.62); prev = yy
    g = MS[name]['growth_pct_per_month']
    ax.text(mdate[-1] + dt.timedelta(days=9), yy, f"{name}  {g:+.0f}%/mo", va='center', fontsize=7.6, color=col if col not in ('#c4c4cc', '#d4d4d8') else '#71717a', fontweight='bold' if lw > 1.5 else 'normal')
ax.set_ylabel('Agent papers naming the model (%)'); ax.set_xlim(mdate[0] - dt.timedelta(days=8), mdate[-1] + dt.timedelta(days=95))
ax.xaxis.set_major_locator(md.MonthLocator(bymonth=[11, 2, 5, 8])); ax.xaxis.set_major_formatter(md.DateFormatter('%b\n%Y'))
save(fig, 'b4_models')

# ---------- F5 benchmarks forest
bm = sorted(T['benchmarks'], key=lambda r: r['slope_logodds_per_month'])
fig, ax = plt.subplots(figsize=(6.0, 3.6))
for i, r in enumerate(bm):
    g, lo, hi = gci(r); sig = r['q'] < 0.05; col = VIOLET if g > 0 else MUTED
    ax.plot([lo, hi], [i, i], color=col, lw=2 if sig else 1.1, alpha=1 if sig else .5, solid_capstyle='round')
    ax.scatter([g], [i], s=26, color=col if sig else 'white', edgecolor=col, lw=1.3, zorder=3)
    ax.text(110, i, f"{r['docs']}", va='center', ha='right', fontsize=7.3, color=MUTED)
ax.text(110, len(bm) - 0.2, 'papers', ha='right', va='bottom', fontsize=7, color=MUTED)
ax.axvline(0, color='#a1a1aa', lw=.9); ax.set_yticks(range(len(bm))); ax.set_yticklabels([r['name'] for r in bm]); ax.grid(axis='y', visible=False)
ax.set_xlim(-45, 112); ax.set_xlabel('Monthly growth in share of agent papers naming the benchmark (%, 95% CI). Filled: q < 0.05')
save(fig, 'b5_benchmarks')

# ---------- F6 style crossover
fig, ax = plt.subplots(figsize=(6.4, 2.9))
st = STY['all']; sv = [st['stock']['monthly'][m] for m in months]; pv = [st['plain']['monthly'][m] for m in months]
ax.fill_between(mdate, sv, color=MUTED, alpha=.10, lw=0); ax.fill_between(mdate, pv, color=VIOLET, alpha=.12, lw=0)
ax.plot(mdate, sv, color=MUTED, lw=2.2, marker='o', ms=3); ax.plot(mdate, pv, color=VIOLET, lw=2.4, marker='o', ms=3)
ax.text(mdate[0], sv[0] + .04, '“leverage, comprehensive, crucial, enhance, highlight …”', fontsize=7.6, color='#52525b', style='italic')
ax.text(mdate[-1], pv[-1] + .045, '“keeps, same, every, because, frozen …”', fontsize=7.6, color=PURPLE, style='italic', ha='right')
ax.set_ylabel('Occurrences per 100 words'); ax.set_ylim(0, .72)
ax.xaxis.set_major_locator(md.MonthLocator(bymonth=[11, 2, 5, 8])); ax.xaxis.set_major_formatter(md.DateFormatter('%b\n%Y'))
save(fig, 'b6_style')

# ---------- F7 news source groups, indexed to Nov 2025
NG = {r['name']: r for r in T['news_groups']}
fig, ax = plt.subplots(figsize=(6.4, 3.0))
ends = []
for name, col, lw in [('General & business press', VIOLET, 2.4), ('Tech press', BLUE, 2), ('Developer community', PURPLE, 2), ('Vendors, newsletters & video', '#a1a1aa', 1.5)]:
    y = np.array(NG[name]['monthly_pct']); base = y[:2].mean(); idx = y / base
    ax.plot(mdate, idx, color=col, lw=lw, marker='o', ms=2.6)
    ends.append([idx[-1], name, col, y[-1]])
ends.sort(); prev = -9
for v, name, col, last in ends:
    yy = max(v, prev + 0.55); prev = yy
    ax.text(mdate[-1] + dt.timedelta(days=8), yy, f"{name}  ({last:.1f}% of stories)", va='center', fontsize=7.2, color=col if col != '#a1a1aa' else MUTED)
ax.axhline(1, color='#a1a1aa', lw=.8, ls=':')
ax.spines['bottom'].set_bounds(md.date2num(mdate[0] - dt.timedelta(days=14)), md.date2num(mdate[-1] + dt.timedelta(days=16)))
ax.set_ylabel('Agent share, indexed (Nov–Dec 2025 = 1)'); ax.set_xlim(mdate[0] - dt.timedelta(days=8), mdate[-1] + dt.timedelta(days=150))
ax.set_xticks([dt.date(2025, 11, 1), dt.date(2026, 2, 1), dt.date(2026, 5, 1), dt.date(2026, 8, 1), dt.date(2026, 11, 1)]); ax.xaxis.set_major_formatter(md.DateFormatter('%b\n%Y'))
save(fig, 'b7_news_groups')

# ---------- F8 news themes and companies forest (two panels)
fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.3), gridspec_kw=dict(wspace=0.85))
for ax, key, title in [(axs[0], 'news_themes', 'Themes'), (axs[1], 'news_companies', 'Companies')]:
    rows = sorted(T[key], key=lambda r: r['slope_logodds_per_month'])
    for i, r in enumerate(rows):
        g, lo, hi = gci(r); sig = r['q'] < 0.05; col = BLUE if g > 0 else MUTED
        ax.plot([lo, hi], [i, i], color=col, lw=2 if sig else 1.1, alpha=1 if sig else .5)
        ax.scatter([g], [i], s=16 + r['share_all_pct'] * 3, color=col if sig else 'white', edgecolor=col, lw=1.2, zorder=3)
    ax.axvline(0, color='#a1a1aa', lw=.9); ax.set_yticks(range(len(rows))); ax.set_yticklabels([r['name'] for r in rows], fontsize=7.8); ax.grid(axis='y', visible=False)
    ax.set_title(title, **TITLE); ax.set_xlabel('Growth, % per month (95% CI)', fontsize=7.8)
save(fig, 'b8_news_forest')

# ---------- F9 topic heatmap (agent papers), brand ramp
from matplotlib.colors import LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list('q', ['#ffffff', '#ece9fd', '#c9c1fa', VIOLET, PURPLE, '#3b2a8f'])
tp2 = sorted(T['topics'], key=lambda r: -r['slope_logodds_per_month'])
M = np.array([r['monthly_pct'] for r in tp2])
rel = M / M.mean(axis=1, keepdims=True)
fig, ax = plt.subplots(figsize=(6.8, 3.6))
im = ax.imshow(rel, aspect='auto', cmap=cmap, vmin=0.4, vmax=1.6)
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        ax.text(j, i, f'{M[i, j]:.0f}', ha='center', va='center', fontsize=6.2, color='white' if rel[i, j] > 1.25 else '#3f3f46')
ax.set_yticks(range(len(tp2))); ax.set_yticklabels([r['name'] for r in tp2], fontsize=7.8)
ax.set_xticks(range(12)); ax.set_xticklabels([dt.date(int(m[:4]), int(m[5:]), 1).strftime('%b') for m in months], fontsize=7.5)
ax.grid(False); [s.set_visible(False) for s in ax.spines.values()]
cb = fig.colorbar(im, ax=ax, fraction=.03, pad=.015); cb.set_label('Relative to topic’s yearly mean', fontsize=7.2); cb.outline.set_visible(False); cb.ax.tick_params(labelsize=6.5)
save(fig, 'b9_topic_heat')
print('figures written')
