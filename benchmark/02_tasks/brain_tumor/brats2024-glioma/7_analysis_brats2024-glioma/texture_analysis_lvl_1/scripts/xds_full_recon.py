import sys; sys.path.insert(0,"/project/aip-jcohen/paulh/mri_synthesis_project/benchmark/02_tasks/brain_tumor/brats2024-glioma/7_analysis_brats2024-glioma/texture_analysis_lvl_1/scripts")
import xds_full_common as C, numpy as np
MAN=C.manifest()
specs=C.dev_specs(); print("dev cells",len(specs), {f:sum(s['fam']==f for s in specs) for f in set(s['fam'] for s in specs)})
for k in C.dev_feature_keys()+C.TEST_KEYS:
    cs=MAN[k][1]()[:C.CAP]; print("MAN",k,len(MAN[k][1]()),[c[0] for c in cs[:2]])
R=C.load_rad(); print("rad rows",R.shape); 
import collections
print({k:int((R.index.get_level_values(0)==k).sum()) for k in C.dev_feature_keys()+C.TEST_KEYS})
for s in specs:
    d,m=C.targets(s)
    if s['fam']=='brats': print(s['name'],m['n_full'],m['n_csv'],round(m['full_mean']-m['json_delta'],3)); continue
    k=s['eval_key']; mc=set(c[0] for c in MAN[k][1]()[:C.CAP]); ev=set(d)
    pk={C.pkey(c,k) for c in mc}
    print(s['name'],"n_tgt",m['n_full'],"mean",round(m['full_mean'],2),"json",round(m['json_delta'],2),"ids",sorted(ev)[:2],"overlap_exact",len(mc&ev),"overlap_pk",len({C.pkey(c,k) for c in ev}&pk),"radcov",len([c for c in mc if (k,c) in R.index]))
