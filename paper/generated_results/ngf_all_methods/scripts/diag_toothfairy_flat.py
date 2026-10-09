import sys, numpy as np, torch, nibabel as nib
R = __import__("pathlib").Path(__file__).resolve().parents[4]; sys.path.insert(0, str(R / "benchmark/02_tasks/brain_ms/open-ms/7_analysis_open-ms/texture_analysis_lvl_1/scripts"))
from compute_ngf_texture import ngf_map
from compute_texture_metrics_openms import erode
import glob
f=sorted(glob.glob(str(R / "benchmark/02_tasks/mandible_healthy/toothfairy2/1_BIDS_toothfairy2/maxillofacial-toothfairy2/sub-*/anat/*_ct.nii.gz")))[300]
a=torch.from_numpy(nib.load(f).get_fdata(dtype=np.float32)); print(f.split('/')[-1],a.shape,"min/max/p1/p50/p99",[float(x) for x in (a.min(),a.max(),*torch.quantile(a.flatten()[::5],torch.tensor([.01,.5,.99])))])
print("n unique values", len(torch.unique(a)), "most common value frac", float((a==torch.mode(a.flatten()).values).float().mean()), "mode",float(torch.mode(a.flatten()).values))
s=(a-a.mean())/a.std(); fg=a>0.1*torch.quantile(a.flatten()[::5],.99); m=erode(fg,3)
print("fg frac",float(fg.float().mean()),"eroded",float(m.float().mean()))
sim,g2=ngf_map(s,s.clone()); v=g2[m]; sv=sim[m]
print("identity mean",float(sv.mean()))
for lo,hi in [(0,1e-12),(1e-12,1e-9),(1e-9,1e-7),(1e-7,1e-5),(1e-5,1e-3),(1e-3,1e9)]:
    k=(v>=lo)&(v<hi); print(f"g2 in [{lo:g},{hi:g}): frac {float(k.float().mean()):.4f}  mean ngf {float(sv[k].mean()) if k.any() else float('nan'):.3f}")
