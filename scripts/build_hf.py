"""Build the Hugging Face dataset files from the agents-blog export.
Usage: python build_hf.py <full.csv> <articles_truncated.csv> <out_dir>
Labels use title + first 1500 chars of the abstract/summary, matching the paper's analysis."""
import pandas as pd, re, sys, json
full, trunc, OUT = sys.argv[1:4]
f = pd.read_csv(full, parse_dates=['published_at'])
f = f[(f.published_at < '2028-01-01')]
f['descr'] = f.description.fillna('').str.replace(r'^arXiv:\S+ Announce Type: \S+\s*Abstract:\s*', '', regex=True)
f['announce_type'] = f.description.fillna('').str.extract(r'Announce Type: (\S+)')[0]
f['txt'] = (f.title.fillna('') + ' . ' + f.descr.str.slice(0, 1500)).str.lower()
AG = r'\bagent(?:s|ic)?\b|\bmulti-?agents?\b|\bagentic\b'
EX = r'\bice\b|immigration|\bfbi\b|federal agents?|border patrol|secret service|travel agents?|real estate|free agents?|nerve agent|agent orange|double agent|undercover|deport|shooting|sports agent|talent agent|\bdea\b|\bcbp\b|homeland'
ST = r'\bai\b|agentic|\bllms?\b|chatbot|language model|openai|anthropic|claude|gemini|copilot'
f['is_agent'] = f.txt.str.contains(AG) & ~(f.txt.str.contains(EX) & ~f.txt.str.contains(ST))
f['agent_in_title'] = f.title.fillna('').str.lower().str.contains(AG)
TOP = {'benchmarks_evaluation': r'benchmark|\bevaluation\b|leaderboard', 'rl_training': r'reinforcement learning|\brl\b|grpo|policy optimization|\brlvr\b',
 'memory_long_horizon': r'\bmemory\b|long[- ]horizon|long-term', 'tool_use': r'tool[- ]use|tool[- ]calling|function[- ]calling|tool-augmented|tool learning|\btools?\b',
 'multi_agent': r'multi-?agent', 'safety_security': r'safety|secur|prompt injection|jailbreak|\battacks?\b|misalign|guardrail|red[- ]team',
 'planning': r'\bplanning\b|\bplanner\b', 'coding_software': r'swe-bench|software engineering|coding agents?|code agents?|repositor|program repair|\bcoding\b',
 'embodied_robotics': r'embodied|robot|vision-language-action|\bvla\b', 'gui_web_computer_use': r'\bgui\b|computer[- ]use|web agents?|browser|web navigation|mobile agents?|\bos agents?|screen',
 'self_improving': r'self-evolv|self-improv|self-reflect|evolving agents?', 'deep_research_science': r'deep research|scientific discovery|research agents?|ai scientist|literature review',
 'mcp_protocols': r'model context protocol|\bmcp\b|agent-to-agent|\ba2a\b'}
R = f[f.feed_type == 'research'].copy()
R['topics'] = [[k for k, p in TOP.items() if a and re.search(p, t)] for t, a in zip(R.txt, R.is_agent)]
R['arxiv_id'] = R.url.str.extract(r'arxiv\.org/abs/([^v\s]+)')[0]
R['arxiv_feed'] = R.feed_name.str.replace(' updates on arXiv.org', '').str.replace(' arXiv.org', '')
papers = R[['arxiv_id', 'title', 'descr', 'arxiv_feed', 'announce_type', 'published_at', 'url', 'is_agent', 'agent_in_title', 'topics']].rename(columns={'descr': 'abstract', 'published_at': 'announced_at'})
papers['title'] = papers.title.str.replace(r'\s+', ' ', regex=True).str.strip()
papers = papers.sort_values('announced_at').reset_index(drop=True)
papers.to_parquet(f'{OUT}/papers.parquet', index=False)
ENT = {'Anthropic / Claude': r'anthropic|claude', 'OpenAI': r'openai|chatgpt|\bgpt-?\d|codex', 'Google / Gemini': r'google|gemini|deepmind', 'Meta': r'\bmeta\b|llama', 'Microsoft / Copilot': r'microsoft|copilot', 'Amazon / AWS': r'amazon|\baws\b|bedrock', 'Nvidia': r'nvidia', 'OpenClaw': r'openclaw|clawdbot|moltbot|moltbook'}
NT = {'Enterprise adoption': r'enterprise|business|customer|workflow|productivity', 'Coding agents': r'coding|developer|code\b|software engineer', 'Security & risk': r'secur|prompt injection|hack|vulnerab|breach|malware|risk', 'Regulation & policy': r'regulat|law\b|policy|congress|\beu\b|govern', 'Jobs & workforce': r'\bjobs?\b|layoff|workforce|worker|employ|hiring', 'Funding & deals': r'raises|funding|valuation|acqui|\$\d+\s?(?:m|b|million|billion)|series [a-d]', 'Shopping & payments': r'shopping|commerce|checkout|payment|purchase', 'Browsers & computer use': r'browser|computer use|operator\b|comet\b'}
N = f[(f.feed_type == 'news') & f.is_agent].copy()
N['companies'] = [[k for k, p in ENT.items() if re.search(p, t)] for t in N.txt]
N['themes'] = [[k for k, p in NT.items() if re.search(p, t)] for t in N.txt]
news = N[['title', 'feed_name', 'published_at', 'url', 'companies', 'themes']].rename(columns={'title': 'headline', 'feed_name': 'source'})
news['headline'] = news.headline.str.replace(r'\s+', ' ', regex=True).str.strip()
news = news.sort_values('published_at').reset_index(drop=True)
news.to_parquet(f'{OUT}/agent_news.parquet', index=False)
# monthly
f['month'] = f.published_at.dt.to_period('M').astype(str)
mo = f.groupby(['month', 'feed_type']).agg(items=('is_agent', 'size'), agent_items=('is_agent', 'sum')).reset_index()
mo['agent_share_pct'] = (100 * mo.agent_items / mo['items']).round(2)
mo = mo.rename(columns={'feed_type': 'corpus'}).replace({'corpus': {'research': 'papers'}})
mo.to_parquet(f'{OUT}/monthly.parquet', index=False)
print('papers', len(papers), int(papers.is_agent.sum()), '| news', len(news), '| all news', int((f.feed_type == 'news').sum()))
print(mo.pivot(index='month', columns='corpus', values='agent_share_pct'))
tc = pd.Series([t for ts in papers.topics for t in ts]).value_counts(); print(tc.to_dict())
