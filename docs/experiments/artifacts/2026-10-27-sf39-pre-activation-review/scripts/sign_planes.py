import csv, sys, os
R=sys.argv[1]
def load(d):
    with open(os.path.join(R,d,'realizations','r_0000','moments_planes.csv')) as f:
        rows=[r for r in csv.DictReader(l for l in f if not l.startswith('#'))]
    return rows
for eps in (1,2):
    for s in (6101,6102,6103):
        ref=load(f'F_e{eps}_s{s}_g256')
        for g in (32,64,128):
            a=load(f'F_e{eps}_s{s}_g{g}')
            neg=0;n=0;rels=[]
            for ra,rr in zip(a,ref):
                vT=float(ra['var_x2'])+float(ra['var_x3']); vR=float(rr['var_x2'])+float(rr['var_x3'])
                n+=1; neg+=vT<vR; rels.append((vT-vR)/vR)
            print(f'eps={eps} seed={s} lh={g//2:3d} sigmaT2<ref {neg:2d}/{n} rel first/last {rels[0]:+.2e}/{rels[-1]:+.2e}')
