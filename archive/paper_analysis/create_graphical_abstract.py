from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
base=Path(__file__).resolve().parents[1]
fig,ax=plt.subplots(figsize=(12,5)); ax.set_xlim(0,12); ax.set_ylim(0,5); ax.axis('off')
boxes=[(.25,1.65,2.35,1.8,'Waste-mask\ngasification\nlogged measurements','#edf3fa'),
 (3.1,1.65,2.45,1.8,'Past 30 rows\n11 sensors +\n4 gas histories','#e6f1ed'),
 (6.05,1.65,2.5,1.8,'RNN / LSTM / GRU\nfixed settings\n10 seeds each','#fff1df'),
 (9.05,1.65,2.65,1.8,'CO / CO2 / CH4 / H2\n1 / 3 / 5 / 10 s\nfuture logged values','#edf3fa')]
for x,y,w,h,text,color in boxes:
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.05',facecolor=color,edgecolor='#42627f',linewidth=1.5)); ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=13)
for a,b in zip(boxes[:-1],boxes[1:]): ax.annotate('',xy=(b[0]-.1,2.55),xytext=(a[0]+a[2]+.1,2.55),arrowprops={'arrowstyle':'->','lw':2,'color':'#42627f'})
ax.text(6,4.4,'Past-only forecasting from measured histories',ha='center',fontsize=19,weight='bold')
ax.text(6,.95,'Fixed file split: 10 train / 4 validation / 2 test; common targets; train-only scaling',ha='center',fontsize=12)
ax.text(6,.5,'Two test records; no target-time inputs. At 10 s, recurrent H2 RMSE did not improve on persistence.',ha='center',fontsize=11)
ax.text(6,.15,'For CO2, persistence had lower mean RMSE than every recurrent configuration at every lead.',ha='center',fontsize=10)
fig.tight_layout()
for ext in ['png','pdf','svg']: fig.savefig(base/f'figures/Graphical_abstract.{ext}',dpi=200,bbox_inches='tight')
print('Created scope-consistent graphical abstract.')
