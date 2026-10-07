"""Publication study design with authenticated simulation fields and vector schematic.

Produces 180 mm PDF/SVG with embedded original field rasters and 600 dpi PNG.
"""
from pathlib import Path
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Polygon, FancyArrowPatch

INK='#202C38'; MUTED='#52616F'; LINE='#C5CED5'
COLORS={'ae':'#0072B2','pc':'#E69F00','rae':'#009E73','rpc':'#CC79A7','pod':'#595959'}

def main(out, visuals=None, base_name="Figure_1_study_design"):
    import json, hashlib
    arrays = meta = None
    if visuals is not None:
        root = Path(visuals)
        meta = json.loads((root / "VISUAL_EXPORT.json").read_text())
        if hashlib.sha256((root / "visuals.npz").read_bytes()).hexdigest() != meta["visuals_sha256"]:
            raise ValueError("Visual export hash mismatch")
        arrays = np.load(root / "visuals.npz", allow_pickle=False)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,
       'svg.fonttype':'none','svg.hashsalt':'active-matter-figure1-schematic-v2',
       'pdf.fonttype':42,'mathtext.fontset':'dejavusans'})
    fig=plt.figure(figsize=(180/25.4,170/25.4),facecolor='white')
    ax=fig.add_axes([0,0,1,1]); ax.set(xlim=(0,180),ylim=(0,170)); ax.axis('off')
    def txt(x,y,s,size=8,weight='normal',color=INK,ha='left',va='center',**kw):
        return ax.text(x,y,s,fontsize=size,fontweight=weight,color=color,ha=ha,va=va,**kw)
    def arrow(x1,y1,x2,y2,color=MUTED,lw=.9,style='-|>',scale=9):
        ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle=style,mutation_scale=scale,
                                    linewidth=lw,color=color,shrinkA=0,shrinkB=0))
    def heading(letter,x,y,title):
        txt(x,y,letter,11,'bold'); txt(x+7,y,title,9,'bold')
    def grid(x,y,w=13,h=15,color=COLORS['ae'],alpha=1):
        ax.add_patch(Rectangle((x,y),w,h,facecolor='white',edgecolor=color,lw=.8,alpha=alpha))
        for v in np.linspace(0,w,5)[1:-1]: ax.plot([x+v,x+v],[y,y+h],color=color,lw=.4,alpha=.4)
        for v in np.linspace(0,h,5)[1:-1]: ax.plot([x,x+w],[y+v,y+v],color=color,lw=.4,alpha=.4)

    if arrays is None:
        raise ValueError("Authentic visual export is required for this figure")
    heading('a',5,164,'Coupled active-fluid state')
    selected=meta['dataset_selected'][1]
    alpha,zeta=selected['parameters']
    txt(175,164,rf'$\alpha={alpha:g},\;\zeta={zeta:g},\;t={selected.get("time",10):g}$',7.3,ha='right',color=MUTED)
    views=arrays['dataset_views'][1]
    specs=[(0,'Concentration',r'$10^4(c-1)$','viridis'),
           (1,'Speed',r'$|\mathbf{v}|$','magma'),
           (3,'Local order',r'$S$','cividis'),
           (4,'Strain magnitude',r'$\|\mathbf{E}\|_{\mathrm{F}}$','viridis')]
    for j,(k,title,quantity,cmap) in enumerate(specs):
        x=6+44*j
        txt(x+17,152,title,8.2,'bold',ha='center')
        imax=fig.add_axes([x/180,115/170,34/180,34/170])
        values=views[k].astype(float)
        if k==0: values=1e4*(values-1)
        lo,hi=(0,1) if k==3 else (float(values.min()),float(values.max()))
        if hi<=lo: hi=lo+1e-8
        im=imax.imshow(values.T,origin='lower',extent=[0,meta['length'][0],0,meta['length'][1]],
                       cmap=cmap,vmin=lo,vmax=hi,interpolation='nearest')
        imax.set_xticks([]); imax.set_yticks([])
        for spine in imax.spines.values(): spine.set_linewidth(.4); spine.set_color(LINE)
        cbax=fig.add_axes([x/180,109/170,34/180,1.7/170])
        cb=fig.colorbar(im,cax=cbax,orientation='horizontal')
        from matplotlib.ticker import MaxNLocator, ScalarFormatter
        cb.ax.xaxis.set_major_locator(MaxNLocator(3))
        cb.ax.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        cb.ax.tick_params(labelsize=6,length=2,pad=1)
        cb.outline.set_linewidth(.4)
        txt(x+17,101,quantity,7.2,ha='center')
    txt(90,158,'11 stored channels · periodic domain L = 10 · reference snapshot, frame 40',7.1,ha='center',color=MUTED)

    heading('b',5,94,'Matched compression and physical decoding')
    for dx,dy in [(0,0),(2,2),(4,4)]: grid(9+dx,68+dy,12,15,alpha=.9)
    txt(17,63,'Field state',7.3,ha='center')
    txt(17,58,r'$\mathbf{x}_t$',9,ha='center')
    arrow(29,77,42,77)
    enc=Polygon([(45,67),(45,87),(67,81),(67,73)],facecolor='#E8F2F8',edgecolor=COLORS['ae'],lw=.9)
    ax.add_patch(enc)
    for i in range(1,4):
        x=45+i*5.5; h=20-i*3
        ax.plot([x,x],[77-h/2,77+h/2],color=COLORS['ae'],alpha=.55,lw=.65)
    txt(56,63,'Encoder',7.3,ha='center')
    arrow(70,77,81,77)
    ax.add_patch(Rectangle((85,68),20,18,facecolor='#F7F3F8',edgecolor=COLORS['rpc'],lw=.9))
    for y in [71,75,79,83]:
        for x in [90,95,100]: ax.add_patch(Circle((x,y),.65,color=COLORS['rpc'],alpha=.9))
    txt(95,63,'Latent state',7.3,ha='center'); txt(95,58,r'$\mathbf{z}_t\in\mathbb{R}^{d}$',9,ha='center')
    arrow(109,77,120,77)
    dec=Polygon([(124,73),(124,81),(146,87),(146,67)],facecolor='#E8F2F8',edgecolor=COLORS['ae'],lw=.9)
    ax.add_patch(dec)
    for i in range(1,4):
        x=124+i*5.5; h=8+i*3
        ax.plot([x,x],[77-h/2,77+h/2],color=COLORS['ae'],alpha=.55,lw=.65)
    txt(135,63,'Decoder',7.3,ha='center')
    arrow(149,77,159,77)
    grid(163,69,12,16,COLORS['rae'])
    txt(169,63,'Reconstruction',7.3,ha='center'); txt(169,58,r'$\hat{\mathbf{x}}_t$',9,ha='center')
    txt(56,58,r'$f_\theta$',9,ha='center'); txt(135,58,r'$g_\phi$ + optional hard map',7.3,ha='center')
    txt(90,51,'d = 128, 256, 512   ·   three fitted neural seeds   ·   weighted POD baseline',7.3,ha='center',color=MUTED)
    ax.plot([5,175],[46,46],color=LINE,lw=.55)
    ax.plot([87,87],[9,41],color=LINE,lw=.55)

    heading('c',5,40,'Constraint controls')
    txt(40,33,'Soft penalties',7.5,'bold',ha='center')
    txt(38,27,'Off',7,ha='center'); txt(66,27,'On',7,ha='center')
    txt(8,15,'Hard map',7.3,'bold',ha='center',rotation=90); txt(20,19,'Off',7,ha='center'); txt(20,11,'On',7,ha='center')
    for x,y,name,key in [(27,15.5,'AE','ae'),(55,15.5,'PC-AE','pc'),(27,7.5,'R-AE','rae'),(55,7.5,'R-PC-AE','rpc')]:
        ax.add_patch(Rectangle((x,y),24,7,facecolor=COLORS[key],alpha=.10,edgecolor='none'))
        ax.add_patch(Rectangle((x,y),.8,7,facecolor=COLORS[key],edgecolor='none'))
        txt(x+12,y+3.5,name,7.6,'bold',ha='center')

    heading('d',94,40,'Separate downstream tests')
    for y,name,details in [(30,'Snapshot fidelity',r'$R_{\rm rec}$, $Q$; admissibility'),
                           (20,'Physical information','Latent probes vs parameters + time'),
                           (10,'Recursive forecasting',r'4 history frames → 10 / 77 leads; $F$, $Q$')]:
        ax.add_patch(Circle((97,y),1.2,facecolor=COLORS['rpc'],edgecolor='none'))
        txt(101,y+2,name,7.7,'bold'); txt(101,y-2,details,7.1,color=MUTED)
    # Protocol metadata, not simulated evidence.
    txt(90,2.5,'Training: fit   ·   Validation: select   ·   Frozen test: evaluate   |   Real fields + study schematic',6.6,ha='center',color=MUTED)
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    for ext in ['svg','pdf','png']:
        meta={'Creator':'Active Matter publication schematic','CreationDate':None,'ModDate':None} if ext=='pdf' else ({'Date':None} if ext=='svg' else None)
        fig.savefig(out/f'{base_name}.{ext}',dpi=600,metadata=meta)
    plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--out',default='.'); p.add_argument('--visuals',required=True)
    args=p.parse_args(); main(args.out,args.visuals)
