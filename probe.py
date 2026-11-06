import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_predict, KFold
from train import load, raw
X, cp, fens = load(["ds/pilot_hub","ds/hub_a","ds/hub_b"]); y=np.tanh(cp/400)
act=np.unique(X.indices); Xa=np.sqrt(X[:,act].toarray())
groups=np.load("ds/groups_hub.npy"); inp=np.isin(act,groups.ravel())
cnt = (Xa>0).mean(0)
kf=KFold(5,shuffle=True,random_state=0)
def ev(name, model, Z):
    p=cross_val_predict(model,Z,y,cv=kf); print(f"{name:<40} corr={np.corrcoef(p,y)[0,1]:.3f} acc={np.mean(np.sign(p)==np.sign(y)):.1%}", flush=True)
R=raw(fens)
ev("total spikes only", RidgeCV(), Xa.sum(1,keepdims=True))
ev("material diff only (raw piece counts)", RidgeCV(), np.stack([R[:,i*64:(i+1)*64].sum(1) for i in range(12)],1))
for k in [50,200,800]:
    ev(f"fly: scale+PCA{k}+ridge", make_pipeline(StandardScaler(),PCA(k,random_state=0),RidgeCV(np.logspace(-2,5,15))), Xa)
keep = cnt>0.2
ev(f"fly: reliable neurons only ({keep.sum()})", make_pipeline(StandardScaler(),RidgeCV(np.logspace(-2,5,15))), Xa[:,keep])
ev(f"fly downstream reliable ({(keep&~inp).sum()})", make_pipeline(StandardScaler(),RidgeCV(np.logspace(-2,5,15))), Xa[:,keep&~inp])
