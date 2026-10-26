import pandas as pd, re, sys, textwrap
S=sys.argv[1]; which=sys.argv[2]
df=pd.read_csv(f"{S}/data/articles.csv", parse_dates=['published_at'])
df['tl']=df.title.fillna('').str.lower(); df['txt']=(df.tl+' '+df.descr.fillna('').str.lower())
AG=r'\bagent(?:s|ic)?\b|\bmulti-?agents?\b'
R=df[(df.feed_type=='research')&df.txt.str.contains(AG)]
TH={'harness':r'\bharness','skills':r'\bskills?\b','coding':r'coding agents?|swe-bench|software engineering agent','memory':r'\bmemory\b|long[- ]horizon','security':r'prompt injection|jailbreak|attack|unsafe|safety','multi_early':r'multi-?agent','eval':r'\bauditing|\bmeasuring|evidence|verifiab|reliab','selfevolve':r'self-evolv|recursive|self-improv'}
def show(d,n,seed=7):
  for _,r in d.sample(min(n,len(d)),random_state=seed).sort_values('published_at').iterrows():
    print(f"[{r.published_at.date()}] {r.title}\n   {textwrap.shorten(str(r.descr),520)}\n   {r.url}")
if which=='papers':
  for k,p in TH.items():
    d=R[R.tl.str.contains(p)]
    if k=='multi_early': d=d[d.published_at<'2026-02-01']
    print('=====',k,len(d)); show(d,9)
else:
  N=df[(df.feed_type=='news')&df.txt.str.contains(AG)]
  NT={'rogue_hack':r'rogue|hacked|went rogue|deleted|incident','regulation':r'regulat|\blaw\b|bill|policy|liabil','openclaw':r'openclaw|moltbot|moltbook|clawdbot','jobs':r'\bjobs?\b|layoff|workers|workforce','enterprise':r'enterprise|employees|workflow','reddit':r'.'}
  for k,p in NT.items():
    d=N[N.tl.str.contains(p)]
    if k=='reddit': d=N[N.feed_name=='AI Agents Subreddit']
    else: d=d[d.feed_name!='AI Agents Subreddit']
    print('=====',k,len(d))
    for _,r in d.sample(min(14,len(d)),random_state=3).sort_values('published_at').iterrows(): print(f"[{r.published_at.date()}] {r.feed_name} | {r.title[:170]}")
