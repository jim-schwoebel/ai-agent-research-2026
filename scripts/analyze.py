import pandas as pd, re, json, sys
from collections import Counter
S=sys.argv[1]
df=pd.read_csv(f"{S}/data/articles.csv", parse_dates=['published_at'])
df=df[df.feed_type.isin(['research','news'])].copy()
df['m']=df.published_at.dt.to_period('M').astype(str)
df['txt']=(df.title.fillna('')+' . '+df.descr.fillna('')).str.lower()
df['tl']=df.title.fillna('').str.lower()
AG=r'\bagent(?:s|ic)?\b|\bmulti-?agents?\b|\bagentic\b'
EX=r'\bice\b|immigration|\bfbi\b|federal agents?|border patrol|secret service|travel agents?|real estate|free agents?|nerve agent|agent orange|double agent|undercover|deport|shooting|sports agent|talent agent|\bdea\b|\bcbp\b|homeland'
STRONG=r'\bai\b|agentic|\bllms?\b|chatbot|language model|openai|anthropic|claude|gemini|copilot'
df['agent']=df.txt.str.contains(AG,regex=True) & ~(df.txt.str.contains(EX,regex=True) & ~df.txt.str.contains(STRONG,regex=True))
df['agent_title']=df.tl.str.contains(AG,regex=True)
TOP={
 'Multi-agent systems':r'multi-?agent',
 'Tool use & function calling':r'tool[- ]use|tool[- ]calling|function[- ]calling|tool-augmented|tool learning|\btools?\b',
 'MCP & agent protocols':r'model context protocol|\bmcp\b|agent-to-agent|\ba2a\b',
 'GUI, web & computer-use':r'\bgui\b|computer[- ]use|web agents?|browser|web navigation|mobile agents?|\bos agents?|screen',
 'Coding & software agents':r'swe-bench|software engineering|coding agents?|code agents?|repositor|program repair|\bcoding\b',
 'Memory & long-horizon':r'\bmemory\b|long[- ]horizon|long-term',
 'Planning':r'\bplanning\b|\bplanner\b',
 'RL training of agents':r'reinforcement learning|\brl\b|grpo|policy optimization|\brlvr\b',
 'Benchmarks & evaluation':r'benchmark|\bevaluation\b|leaderboard',
 'Safety & security':r'safety|secur|prompt injection|jailbreak|\battacks?\b|misalign|guardrail|red[- ]team',
 'Embodied & robotics':r'embodied|robot|vision-language-action|\bvla\b',
 'Deep research & science':r'deep research|scientific discovery|research agents?|ai scientist|literature review',
 'Self-improving agents':r'self-evolv|self-improv|self-reflect|evolving agents?',
}
months=sorted(df.m.unique())
out={'months':months}
cnt=df.groupby(['m','feed_type']).size().unstack(fill_value=0)
out['volume']={t:[int(cnt.loc[m,t]) if m in cnt.index else 0 for m in months] for t in ['research','news']}
ag=df.groupby(['m','feed_type']).agent.mean().unstack()
agt=df.groupby(['m','feed_type']).agent_title.mean().unstack()
agn=df[df.agent].groupby(['m','feed_type']).size().unstack(fill_value=0)
out['agent_share']={t:[round(100*ag.loc[m,t],2) for m in months] for t in ['research','news']}
out['agent_title_share']={t:[round(100*agt.loc[m,t],2) for m in months] for t in ['research','news']}
out['agent_count']={t:[int(agn.loc[m,t]) for m in months] for t in ['research','news']}
R=df[df.feed_type=='research']; RA=R[R.agent]
out['totals']={'papers':len(R),'news':int((df.feed_type=='news').sum()),'agent_papers':len(RA),'agent_news':int(df[(df.feed_type=='news')&df.agent].shape[0]),
 'agent_title_papers':int(R.agent_title.sum()),'first':str(df.published_at.min().date()),'last':str(df.published_at.max().date()),
 'research_first':str(R.published_at.min().date()),'research_last':str(R.published_at.max().date())}
cat=R.groupby(['m','feed_name']).agent.mean().unstack()
out['cat_agent_share']={c:[round(100*cat.loc[m,c],2) for m in months] for c in cat.columns}
out['cat_totals']=R.feed_name.value_counts().to_dict()
out['cat_agent_totals']=RA.feed_name.value_counts().to_dict()
sub={}
for k,p in TOP.items():
  RA_k=RA.txt.str.contains(p,regex=True)
  s=RA_k.groupby(RA.m).mean()
  sub[k]={'share':[round(100*s.get(m,0),1) for m in months],'total':int(RA_k.sum()),'pct':round(100*RA_k.mean(),1)}
out['subtopics']=sub
# trending bigrams/phrases in agent paper titles: H1 vs H2
def grams(t):
  w=re.findall(r'[a-z][a-z0-9\-]+',t); st={'a','an','the','of','for','and','in','on','with','to','via','from','by','is','as','at','towards','toward','agent','agents','agentic','llm','llms','language','models','model','large','using','based','through','into','are','can','we'}
  g=[x for x in w if x not in st and len(x)>2]
  return set(g)|set(' '.join(p) for p in zip(w,w[1:]) if p[0] not in st and p[1] not in st)
h1=RA[RA.m<='2026-04']; h2=RA[RA.m>='2026-05']
c1=Counter(); c2=Counter()
for t in h1.tl: c1.update(grams(t))
for t in h2.tl: c2.update(grams(t))
n1,n2=len(h1),len(h2)
rows=[]
for g in set(c1)|set(c2):
  a,b=c1[g],c2[g]
  if a+b<40: continue
  r=((b+1)/n2)/((a+1)/n1); rows.append((g,a,b,round(r,2)))
rows.sort(key=lambda x:-x[3]); out['rising']=rows[:40]; out['falling']=sorted(rows,key=lambda x:x[3])[:25]
out['top_terms_all']=Counter({g:c1[g]+c2[g] for g in set(c1)|set(c2)}).most_common(60)
out['h']=[n1,n2]
# news
N=df[(df.feed_type=='news')&df.agent]
out['news_sources']=N.feed_name.value_counts().head(15).to_dict()
ENT={'OpenAI':r'openai|chatgpt|\bgpt-?\d|codex','Anthropic / Claude':r'anthropic|claude','Google / Gemini':r'google|gemini|deepmind','Microsoft / Copilot':r'microsoft|copilot','Meta':r'\bmeta\b|llama','Nvidia':r'nvidia','Amazon / AWS':r'amazon|\baws\b|bedrock','Salesforce':r'salesforce|agentforce','Apple':r'\bapple\b|siri','xAI / Grok':r'\bxai\b|grok','DeepSeek / Qwen':r'deepseek|qwen|alibaba','OpenClaw':r'openclaw|clawdbot|moltbot|moltbook','Perplexity':r'perplexity','Cursor':r'\bcursor\b'}
es={}
for k,p in ENT.items():
  h=N.txt.str.contains(p,regex=True); s=h.groupby(N.m).mean()
  es[k]={'share':[round(100*s.get(m,0),1) for m in months],'total':int(h.sum())}
out['news_entities']=es
NT={'Security & risk':r'secur|prompt injection|hack|vulnerab|breach|malware|risk','Jobs & workforce':r'\bjobs?\b|layoff|workforce|worker|employ|hiring','Enterprise adoption':r'enterprise|business|customer|workflow|productivity','Funding & deals':r'raises|funding|valuation|acqui|\$\d+\s?(m|b|million|billion)|series [a-d]','Coding agents':r'coding|developer|code\b|software engineer','Shopping & payments':r'shopping|commerce|checkout|payment|purchase','Browsers & computer-use':r'browser|computer use|operator\b|comet\b','Regulation & policy':r'regulat|law\b|policy|congress|\beu\b|govern'}
nt={}
for k,p in NT.items():
  h=N.txt.str.contains(p,regex=True); s=h.groupby(N.m).mean()
  nt[k]={'share':[round(100*s.get(m,0),1) for m in months],'total':int(h.sum())}
out['news_themes']=nt
# rising news terms
n1d=N[N.m<='2026-04']; n2d=N[N.m>='2026-05']; d1=Counter(); d2=Counter()
for t in n1d.tl: d1.update(grams(t))
for t in n2d.tl: d2.update(grams(t))
rr=[]
for g in set(d1)|set(d2):
  a,b=d1[g],d2[g]
  if a+b<30: continue
  rr.append((g,a,b,round(((b+1)/len(n2d))/((a+1)/len(n1d)),2)))
rr.sort(key=lambda x:-x[3]); out['news_rising']=rr[:30]
# example papers per month: titles with agent in title, sample 6 per month deterministic
ex={}
for m in months:
  t=R[(R.m==m)&R.agent_title]
  ex[m]=t.sample(min(8,len(t)),random_state=1)[['title','url','feed_name']].values.tolist() if len(t) else []
out['examples']=ex
json.dump(out,open(f"{S}/data/analysis.json",'w'),indent=1,default=str)
print(json.dumps({k:out[k] for k in ['totals','agent_share','agent_title_share','agent_count','volume','h']},indent=0))
for k,v in sub.items(): print(k,v['pct'],v['share'])
print('RISING',out['rising'][:40]); print('FALLING',out['falling'][:25])
print('CAT',out['cat_agent_totals'],out['cat_totals'])
for k,v in es.items(): print(k,v['total'],v['share'])
for k,v in nt.items(): print(k,v['total'],v['share'])
print('NEWSRISING',out['news_rising']); print(out['news_sources'])
