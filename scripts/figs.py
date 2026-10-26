import json, sys, datetime as dt
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.dates as md, numpy as np
O=sys.argv[1]; F=sys.argv[2]
L=lambda n: json.load(open(f"{O}/{n}.json"))
plt.rcParams.update({'font.family':'serif','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'axes.titlesize':9.5,'axes.labelsize':9,'legend.frameon':False,'savefig.bbox':'tight','axes.edgecolor':'#555','xtick.color':'#333','ytick.color':'#333'})
C1,C2,MU='#1f4e79','#c0504d','#9a9a9a'
mo=L('monthly'); ev=L('events')
mid=lambda m: dt.date(int(m[:4]),int(m[5:]),15)
x=[mid(r['month']) for r in mo]
fig,ax=plt.subplots(figsize=(6.5,2.9))
ax.plot(x,[r['agent_paper_pct'] for r in mo],'-o',color=C1,ms=3,lw=1.6,label='arXiv papers')
ax.plot(x,[r['agent_news_pct'] for r in mo],'-s',color=C2,ms=3,lw=1.6,label='News stories')
ax.set_ylim(0,19); ax.set_ylabel('Share mentioning agents (%)')
for i,e in enumerate(ev):
  d=dt.date.fromisoformat(e['date']); ax.axvline(d,color='#bbb',lw=.6,ls=':',zorder=0)
  ax.text(d,18.2-(i%2)*1.2,str(i+1),ha='center',va='center',fontsize=6.5,bbox=dict(boxstyle='circle,pad=0.15',fc='white',ec='#666',lw=.5))
ax.xaxis.set_major_formatter(md.DateFormatter('%b\n%Y')); ax.xaxis.set_major_locator(md.MonthLocator(bymonth=[11,1,3,5,7,9]))
ax.legend(loc='lower right',ncol=2); ax.grid(axis='y',color='#eee',lw=.6)
fig.savefig(f"{F}/fig1_share.pdf"); plt.close(fig)
# fig2 dumbbell
pc=L('paper_topic_change')[::-1]
fig,ax=plt.subplots(figsize=(6.0,3.6)); y=np.arange(len(pc))
for i,r in enumerate(pc):
  ax.plot([r['first_q_pct'],r['last_q_pct']],[i,i],color=MU,lw=1.4,zorder=1)
  ax.text(max(r['first_q_pct'],r['last_q_pct'])+1.2,i,f"{r['change_pts']:+.1f}",va='center',fontsize=7.5)
ax.scatter([r['first_q_pct'] for r in pc],y,facecolor='white',edgecolor=MU,s=22,zorder=2,label='Nov 2025–Jan 2026')
ax.scatter([r['last_q_pct'] for r in pc],y,color=C1,s=24,zorder=3,label='Aug–Oct 2026')
ax.set_yticks(y); ax.set_yticklabels([r['topic'] for r in pc]); ax.set_xlabel('Share of agent papers (%)'); ax.set_xlim(0,60)
ax.legend(loc='lower right'); ax.grid(axis='x',color='#eee',lw=.6)
fig.savefig(f"{F}/fig2_topics.pdf"); plt.close(fig)
# fig3 terms small multiples
tt=L('title_terms'); terms=['Agent harness','Agent skills','Long-horizon','World models','GUI & computer use','Multi-agent']
fig,axs=plt.subplots(2,3,figsize=(6.5,3.3),sharex=True)
for a,t in zip(axs.flat,terms):
  p=[r for r in tt if r['term']==t]; a.bar(range(len(p)),[r['per_10k'] for r in p],color=C1 if t!='Multi-agent' else C2,width=.75)
  a.set_title(t); a.set_xticks([0,4,8,11]); a.set_xticklabels(['Nov','Mar','Jul','Oct'])
  a.grid(axis='y',color='#eee',lw=.6); a.set_axisbelow(True)
for a in axs[:,0]: a.set_ylabel('Titles per 10k papers')
fig.tight_layout(); fig.savefig(f"{F}/fig3_terms.pdf"); plt.close(fig)
# fig4 news heatmap themes + companies
nm=L('news_monthly'); nt=L('news_totals'); months=[r['month'] for r in mo]
fig,axs=plt.subplots(1,2,figsize=(6.8,2.9))
for a,g in zip(axs,['Theme','Company']):
  names=[r['name'] for r in nt if r['group']==g]
  M=np.array([[next(r['pct'] for r in nm if r['group']==g and r['name']==n and r['month']==m) for m in months] for n in names])
  im=a.imshow(M,aspect='auto',cmap='Blues' if g=='Theme' else 'Reds',vmin=0)
  a.set_yticks(range(len(names))); a.set_yticklabels(names,fontsize=7.5)
  a.set_xticks(range(0,12,2)); a.set_xticklabels([dt.date(int(m[:4]),int(m[5:]),1).strftime('%b\n%y') for m in months[::2]],fontsize=7)
  a.set_title(("Themes" if g=="Theme" else "Companies")+" in agent news (% of month)"); 
  for s in a.spines.values(): s.set_visible(False)
  cb=fig.colorbar(im,ax=a,fraction=.045,pad=.02); cb.ax.tick_params(labelsize=6.5); cb.outline.set_visible(False)
fig.tight_layout(); fig.savefig(f"{F}/fig4_news.pdf"); plt.close(fig)
print('ok')
