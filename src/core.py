
"""core.py - dependency-free ML primitives (pure NumPy)."""
import numpy as np
def train_test_split(X,y,test_size=0.2,seed=42):
    X,y=np.asarray(X,float),np.asarray(y); rng=np.random.default_rng(seed)
    idx=rng.permutation(len(X)); n=max(1,int(len(X)*test_size))
    return X[idx[n:]],X[idx[:n]],y[idx[n:]],y[idx[:n]]
class Standardizer:
    def fit(self,X):
        X=np.asarray(X,float); self.mu_=X.mean(0); self.sd_=X.std(0)+1e-8; return self
    def transform(self,X): return (np.asarray(X,float)-self.mu_)/self.sd_
    def fit_transform(self,X): return self.fit(X).transform(X)
def sigmoid(z): return 1.0/(1.0+np.exp(-np.clip(z,-35,35)))
class LogisticRegression:
    def __init__(self,lr=0.2,epochs=400,l2=1e-3,seed=0): self.lr,self.epochs,self.l2,self.seed=lr,epochs,l2,seed
    def fit(self,X,y):
        X=np.asarray(X,float); y=np.asarray(y,float); n,d=X.shape; rng=np.random.default_rng(self.seed)
        self.w_=rng.normal(0,0.01,d); self.b_=0.0
        pos=max(y.sum(),1.0); neg=max((1-y).sum(),1.0); sw=np.where(y==1,n/(2*pos),n/(2*neg))
        for _ in range(self.epochs):
            p=sigmoid(X@self.w_+self.b_); err=(p-y)*sw
            self.w_-=self.lr*(X.T@err/n+self.l2*self.w_); self.b_-=self.lr*err.mean()
        return self
    def predict_proba(self,X): return sigmoid(np.asarray(X,float)@self.w_+self.b_)
    def predict(self,X,t=0.5): return (self.predict_proba(X)>=t).astype(int)
class RidgeRegression:
    def __init__(self,alpha=1.0): self.alpha=alpha
    def fit(self,X,y):
        X=np.asarray(X,float); y=np.asarray(y,float); Xb=np.hstack([np.ones((len(X),1)),X])
        A=Xb.T@Xb+self.alpha*np.eye(Xb.shape[1]); A[0,0]-=self.alpha
        self.coef_=np.linalg.solve(A,Xb.T@y); return self
    def predict(self,X): return np.hstack([np.ones((len(X),1)),np.asarray(X,float)])@self.coef_
def roc_auc_score(y,s):
    y=np.asarray(y); s=np.asarray(s,float); npos=(y==1).sum(); nneg=(y==0).sum()
    if npos==0 or nneg==0: return float("nan")
    order=np.argsort(s); ranks=np.empty(len(s)); ranks[order]=np.arange(1,len(s)+1)
    return float((ranks[y==1].sum()-npos*(npos+1)/2)/(npos*nneg))
def accuracy_score(y,p): return float((np.asarray(y)==np.asarray(p)).mean())
def f1_score(y,p):
    y,p=np.asarray(y),np.asarray(p); tp=int(((p==1)&(y==1)).sum()); fp=int(((p==1)&(y==0)).sum()); fn=int(((p==0)&(y==1)).sum())
    pr=tp/(tp+fp) if tp+fp else 0.0; rc=tp/(tp+fn) if tp+fn else 0.0
    return float(2*pr*rc/(pr+rc)) if pr+rc else 0.0
def rmse(y,p): return float(np.sqrt(np.mean((np.asarray(y,float)-np.asarray(p,float))**2)))
def mape(y,p):
    y=np.asarray(y,float); p=np.asarray(p,float); m=np.abs(y)>1e-8
    return float(np.mean(np.abs((y[m]-p[m])/y[m]))*100)
