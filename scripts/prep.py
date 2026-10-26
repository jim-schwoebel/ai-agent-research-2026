import pandas as pd, re, json, sys
from collections import Counter
S=sys.argv[1]; O=f"{S}/out"
df=pd.read_csv(f"{S}/data/articles.csv", parse_dates=['published_at'])
df=df[df.feed_type.isin(['research','news'])].copy()
df['m']=df.published_at.dt.to_period('M').astype(str)
df['txt']=(df.title.fillna('')+' . '+df.descr.fillna('')).str.lower(); df['tl']=df.title.fillna('').str.lower()
AG=r'\bagent(?:s|ic)?\b|\bmulti-?agents?\b|\bagentic\b'
EX=r'\bice\b|immigration|\bfbi\b|federal agents?|border patrol|secret service|travel agents?|real estate|free agents?|nerve agent|agent orange|double agent|undercover|deport|shooting|sports agent|talent agent|\bdea\b|\bcbp\b|homeland'
STRONG=r'\bai\b|agentic|\bllms?\b|chatbot|language model|openai|anthropic|claude|gemini|copilot'
df['agent']=df.txt.str.contains(AG) & ~(df.txt.str.contains(EX) & ~df.txt.str.contains(STRONG))
df['agent_title']=df.tl.str.contains(AG)
months=sorted(df.m.unique()); Q1=months[:3]; Q4=months[-3:]
R=df[df.feed_type=='research']; N=df[df.feed_type=='news']; RA=R[R.agent]; NA=N[N.agent]
def w(name,rows): json.dump(rows,open(f"{O}/{name}.json",'w'),indent=0,ensure_ascii=False); print(name,len(rows))
r1=lambda x:round(float(x),1); r2=lambda x:round(float(x),2)
mo=[]
for m in months:
  r=R[R.m==m]; n=N[N.m==m]
  mo.append(dict(month=m,papers=len(r),news=len(n),agent_papers=int(r.agent.sum()),agent_news=int(n.agent.sum()),
    agent_paper_pct=r2(100*r.agent.mean()),agent_news_pct=r2(100*n.agent.mean()),agent_title_paper_pct=r2(100*r.agent_title.mean())))
w('monthly',mo)
q=lambda d,col,ms: 100*d[d.m.isin(ms)][col].mean()
summ=[dict(papers=len(R),news=len(N),agent_papers=int(RA.shape[0]),agent_news=int(NA.shape[0]),
  agent_paper_pct=r1(100*R.agent.mean()),agent_news_pct=r1(100*N.agent.mean()),
  paper_pct_first_q=r1(q(R,'agent',Q1)),paper_pct_last_q=r1(q(R,'agent',Q4)),news_pct_first_q=r1(q(N,'agent',Q1)),news_pct_last_q=r1(q(N,'agent',Q4)),
  paper_first=str(R.published_at.min().date()),paper_last=str(R.published_at.max().date()),news_first=str(N.published_at.min().date()),news_last=str(N.published_at.max().date()),
  first_q='Nov 2025 – Jan 2026',last_q='Aug – Oct 2026',arxiv_feeds=R.feed_name.nunique(),news_feeds=N.feed_name.nunique())]
w('summary',summ)
TOP={'Benchmarks & evaluation':r'benchmark|\bevaluation\b|leaderboard','RL training of agents':r'reinforcement learning|\brl\b|grpo|policy optimization|\brlvr\b',
 'Memory & long-horizon':r'\bmemory\b|long[- ]horizon|long-term','Tool use & function calling':r'tool[- ]use|tool[- ]calling|function[- ]calling|tool-augmented|tool learning|\btools?\b',
 'Multi-agent systems':r'multi-?agent','Safety & security':r'safety|secur|prompt injection|jailbreak|\battacks?\b|misalign|guardrail|red[- ]team',
 'Planning':r'\bplanning\b|\bplanner\b','Coding & software agents':r'swe-bench|software engineering|coding agents?|code agents?|repositor|program repair|\bcoding\b',
 'Embodied & robotics':r'embodied|robot|vision-language-action|\bvla\b','GUI, web & computer use':r'\bgui\b|computer[- ]use|web agents?|browser|web navigation|mobile agents?|\bos agents?|screen',
 'Self-improving agents':r'self-evolv|self-improv|self-reflect|evolving agents?','Deep research & science':r'deep research|scientific discovery|research agents?|ai scientist|literature review',
 'MCP & agent protocols':r'model context protocol|\bmcp\b|agent-to-agent|\ba2a\b'}
pt=[];pc=[]
for k,p in TOP.items():
  h=RA.txt.str.contains(p)
  for m in months:
    sel=RA.m==m; pt.append(dict(month=m,topic=k,pct=r1(100*h[sel].mean()),papers=int(h[sel].sum())))
  a=100*h[RA.m.isin(Q1)].mean(); b=100*h[RA.m.isin(Q4)].mean()
  pc.append(dict(topic=k,papers=int(h.sum()),pct=r1(100*h.mean()),first_q_pct=r1(a),last_q_pct=r1(b),change_pts=r1(b-a)))
w('paper_topics',pt); w('paper_topic_change',sorted(pc,key=lambda x:-x['change_pts']))
TERMS={'Agent harness':r'\bharness','Agent skills':r'\bskills?\b','World models':r'world models?','Long-horizon':r'long[- ]horizon','Multi-agent':r'multi-?agent','GUI & computer use':r'\bgui\b|computer[- ]use'}
tm=[]
for k,p in TERMS.items():
  h=R.tl.str.contains(p)&R.txt.str.contains(AG) if k in('Agent harness','Agent skills') else R.tl.str.contains(p)
  for m in months:
    sel=R.m==m; tm.append(dict(month=m,term=k,titles=int(h[sel].sum()),per_10k=r1(1e4*h[sel].mean())))
w('title_terms',tm)
STOP=set('a an the of for and in on with to via from by is as at towards toward agent agents agentic llm llms language models model large using based through into are can we does not when what how across over study state around same four weeks found says exclusive why more less beyond without under between your their its our new all one two do it be or vs you that this they he she his her who just now get got make makes made first after before up out about than i my me ai'.split())
def grams(t):
  w_=re.findall(r'[a-z][a-z0-9\-\.]+',t); g=[x for x in w_ if x not in STOP and len(x)>2]
  return set(g)|set(' '.join(p) for p in zip(w_,w_[1:]) if p[0] not in STOP and p[1] not in STOP and len(p[0])>2 and len(p[1])>2)
rt=[]
for corpus,D,minc in [('Papers',RA,40),('News',NA,30)]:
  h1=D[D.m<='2026-04']; h2=D[D.m>='2026-05']; c1=Counter(); c2=Counter()
  for t in h1.tl: c1.update(grams(t))
  for t in h2.tl: c2.update(grams(t))
  rows=[]
  for g in set(c1)|set(c2):
    a,b=c1[g],c2[g]
    if a+b<minc: continue
    rows.append(dict(corpus=corpus,term=g,first_half=a,second_half=b,ratio=r2(((b+1)/len(h2))/((a+1)/len(h1)))))
  # drop bigram whose words duplicate an included unigram with same counts
  idx={x['term']:x for x in rows}
  for t in list(idx):
    if t+'s' in idx and ' ' not in t:
      a=idx.pop(t); b=idx[t+'s']; b['first_half']+=a['first_half']; b['second_half']+=a['second_half']
      b['ratio']=r2(((b['second_half']+1)/len(h2))/((b['first_half']+1)/len(h1)))
  for t in list(idx):
    if t not in idx: continue
    if t.endswith('.') or any(t!=o and ' ' in o and t in o.split() and idx[o]['second_half']>=0.8*idx[t]['second_half'] for o in idx): idx.pop(t)
  rows=list(idx.values())
  for t in [x['term'] for x in rows]:
    pass
  rows=[x for x in rows if not (x['term'] in ('hack',) and 'hacked' in idx)]
  rows.sort(key=lambda x:-x['ratio']); up=[x for x in rows if x['ratio']>1][:14]
  dn=sorted([x for x in rows if x['ratio']<1],key=lambda x:x['ratio'])[:8]
  for x in up: x['direction']='Rising'
  for x in dn: x['direction']='Falling'
  rt+=up+(dn if corpus=='Papers' else [])
w('rising_terms',rt)
cats=[]
for c,g in R.groupby('feed_name'):
  cats.append(dict(category=c.replace(' updates on arXiv.org','').replace(' arXiv.org',''),papers=len(g),agent_papers=int(g.agent.sum()),agent_pct=r1(100*g.agent.mean()),first_q_pct=r1(q(g,'agent',Q1)),last_q_pct=r1(q(g,'agent',Q4))))
w('categories',sorted(cats,key=lambda x:-x['agent_pct']))
ENT={'Anthropic / Claude':r'anthropic|claude','OpenAI':r'openai|chatgpt|\bgpt-?\d|codex','Google / Gemini':r'google|gemini|deepmind','Meta':r'\bmeta\b|llama','Microsoft / Copilot':r'microsoft|copilot','Amazon / AWS':r'amazon|\baws\b|bedrock','Nvidia':r'nvidia','OpenClaw':r'openclaw|clawdbot|moltbot|moltbook'}
NT={'Enterprise adoption':r'enterprise|business|customer|workflow|productivity','Coding agents':r'coding|developer|code\b|software engineer','Security & risk':r'secur|prompt injection|hack|vulnerab|breach|malware|risk','Regulation & policy':r'regulat|law\b|policy|congress|\beu\b|govern','Jobs & workforce':r'\bjobs?\b|layoff|workforce|worker|employ|hiring','Funding & deals':r'raises|funding|valuation|acqui|\$\d+\s?(?:m|b|million|billion)|series [a-d]','Shopping & payments':r'shopping|commerce|checkout|payment|purchase','Browsers & computer use':r'browser|computer use|operator\b|comet\b'}
for nm,D in [('news_companies',ENT),('news_themes',NT)]:
  rows=[];tot=[]
  for k,p in D.items():
    h=NA.txt.str.contains(p)
    for m in months:
      sel=NA.m==m; rows.append(dict(month=m,name=k,pct=r1(100*h[sel].mean()),stories=int(h[sel].sum())))
    a=100*h[NA.m.isin(Q1)].mean(); b=100*h[NA.m.isin(Q4)].mean()
    tot.append(dict(name=k,stories=int(h.sum()),pct=r1(100*h.mean()),first_q_pct=r1(a),last_q_pct=r1(b),change_pts=r1(b-a)))
  w(nm,rows); w(nm+'_totals',sorted(tot,key=lambda x:-x['stories']))
w('news_sources',[dict(source=k,stories=int(v)) for k,v in NA.feed_name.value_counts().head(12).items()])
EV=[('2025-11-24','Claude Opus 4.5',r'^Claude Opus 4.5$'),('2025-12-13','Skills spread',r'skills framework'),('2026-01-25','Clawdbot / OpenClaw',r'Clawdbot, an open-source'),('2026-02-26','“Agents actually work”',r'Karpathy says programming'),
 ('2026-03-18','Rogue agents at Meta',r'Meta is having trouble with rogue'),('2026-04-07','Claude Mythos preview',r'debuts preview of powerful new AI model Mythos'),('2026-06-07','“Chat is dead”',r'chat is dead'),
 ('2026-08-26','Agents hack Hugging Face',r'why OpenAI agents hacked Hugging Face'),('2026-09-03','GPT-6 Astra',r'GPT-6 Astra is the first'),('2026-09-10','Meta Muse at No. 2',r'Muse is now the No. 2'),('2026-10-09','1,000 parallel agents',r'orchestrate up to 1,000')]
ev=[]
for d,l,p in EV:
  h=N[N.title.str.contains(p,regex=True,na=False)].sort_values('published_at').iloc[0]
  ev.append(dict(date=str(h.published_at.date()),label=l,headline=h.title,source=h.feed_name.replace(' -  All Posts','').replace(' - All content',''),url=h.url))
w('events',ev)
EXQ={'Agent harness':r'\bharness','Agent skills':r'\bskills?\b','Coding agents':r'coding agents?|swe-bench|software engineering agents?','Computer use':r'\bgui\b|computer[- ]use','Long-horizon':r'long[- ]horizon','Agent security':r'prompt injection|jailbreak|attack|secur'}
ex=[]
for k,p in EXQ.items():
  h=RA[RA.tl.str.contains(p)&RA.agent_title&(RA.m>='2026-08')].sort_values('published_at',ascending=False)
  for _,r in h.head(4).iterrows(): ex.append(dict(theme=k,date=str(r.published_at.date()),title=re.sub(r'\s+',' ',r.title).replace("\\'","'"),category=r.feed_name.replace(' updates on arXiv.org','').replace(' arXiv.org',''),url=r.url))
w('example_papers',ex)
