"""Rebuild the hand-validation sample (60 flagged papers, 60 flagged news; seed 42) and attach the labels."""
import pandas as pd, sys
src, out = sys.argv[1:3]
df = pd.read_csv(src, parse_dates=['published_at'])
df['txt'] = (df.title.fillna('') + ' . ' + df.descr.fillna('')).str.lower()
AG = r'\bagent(?:s|ic)?\b|\bmulti-?agents?\b|\bagentic\b'
EX = r'\bice\b|immigration|\bfbi\b|federal agents?|border patrol|secret service|travel agents?|real estate|free agents?|nerve agent|agent orange|double agent|undercover|deport|shooting|sports agent|talent agent|\bdea\b|\bcbp\b|homeland'
ST = r'\bai\b|agentic|\bllms?\b|chatbot|language model|openai|anthropic|claude|gemini|copilot'
df['agent'] = df.txt.str.contains(AG) & ~(df.txt.str.contains(EX) & ~df.txt.str.contains(ST))
P = df[(df.feed_type == 'research') & df.agent].sample(60, random_state=42)
N = df[(df.feed_type == 'news') & df.agent].sample(60, random_state=42)
# Paper labels: A = LLM/foundation-model agent, B = RL/control/robotics agent, C = incidental use of the word
pl = 'B A A A C B B A A A A A C A A C C A A B B A C A A A A A B A A B A A A A A A C A C C B A A B A A C A B A A A A A B A B A'.split()
# News labels: R = primarily about AI agents, T = AI-related but agents tangential, X = not AI
nl = 'R R R R R R R R T T T T R T R R R R R R R T R T R R R R R R R R R R R R R R R R T R R R T R R R T R R R R R R R R R R R'.split()
assert len(pl) == 60 and len(nl) == 60
names = {'A': 'llm_agent', 'B': 'rl_control_agent', 'C': 'incidental', 'R': 'ai_agent_focus', 'T': 'ai_tangential', 'X': 'not_ai'}
rows = [dict(corpus='papers', text=r.title, url=r.url, label=names[l]) for (_, r), l in zip(P.iterrows(), pl)]
rows += [dict(corpus='news', text=r.title, url=r.url, label=names[l]) for (_, r), l in zip(N.iterrows(), nl)]
v = pd.DataFrame(rows); v.to_parquet(out, index=False)
print(v.groupby(['corpus', 'label']).size())
