"""A genuine NumPy-trained 2-hidden-layer neural network (12->32->16->1).
No fake model: includes forward pass, analytical backprop, Adam, scaling and save.
"""
import json
import numpy as np
from pathlib import Path
from data.make_data import FEATURES

ROOT=Path(__file__).resolve().parents[1]
MODEL_PATH=ROOT/'models'/'player_model.npz'

def sigmoid(x):
    z=np.clip(x,-30,30)
    return 1/(1+np.exp(-z))

class MLP:
    def __init__(self, input_dim=len(FEATURES), seed=42):
        rng=np.random.default_rng(seed)
        self.params={}
        sizes=(input_dim,32,16,1)
        for i in range(1,len(sizes)):
            self.params[f'W{i}']=rng.normal(0,np.sqrt(2/sizes[i-1]),(sizes[i-1],sizes[i]))
            self.params[f'b{i}']=np.zeros((1,sizes[i]))
        self.mean=np.zeros(input_dim);self.std=np.ones(input_dim)

    def fit_scaler(self,x):
        self.mean=x.mean(axis=0); self.std=np.maximum(x.std(axis=0),1e-8)

    def scale(self,x):return (x-self.mean)/self.std

    def forward(self,x,cache=False):
        z1=x@self.params['W1']+self.params['b1'];a1=np.maximum(z1,0)
        z2=a1@self.params['W2']+self.params['b2'];a2=np.maximum(z2,0)
        z3=a2@self.params['W3']+self.params['b3'];p=sigmoid(z3)
        if cache:return p,(x,z1,a1,z2,a2)
        return p

    def predict_proba(self,x):
        x=np.asarray(x,dtype=float)
        if x.ndim==1:x=x[None,:]
        return self.forward(self.scale(x)).ravel()

    def train(self,x,y,epochs=105,lr=0.007,batch_size=128,seed=42):
        self.fit_scaler(x);xx=self.scale(x)
        y=np.asarray(y,dtype=float).reshape(-1,1)
        rng=np.random.default_rng(seed)
        m={k:np.zeros_like(v) for k,v in self.params.items()}
        v={k:np.zeros_like(p) for k,p in self.params.items()}
        history=[]; t=0
        for epoch in range(epochs):
            perm=rng.permutation(len(y))
            for ids in np.array_split(perm,max(1,int(np.ceil(len(y)/batch_size)))):
                xb=xx[ids];yb=y[ids];n=len(ids)
                pred,(inp,z1,a1,z2,a2)=self.forward(xb,True)
                dz3=(pred-yb)/n
                grad={'W3':a2.T@dz3,'b3':dz3.sum(axis=0,keepdims=True)}
                dz2=(dz3@self.params['W3'].T)*(z2>0)
                grad.update({'W2':a1.T@dz2,'b2':dz2.sum(axis=0,keepdims=True)})
                dz1=(dz2@self.params['W2'].T)*(z1>0)
                grad.update({'W1':inp.T@dz1,'b1':dz1.sum(axis=0,keepdims=True)})
                t+=1
                for k in self.params:
                    m[k]=0.9*m[k]+0.1*grad[k];v[k]=0.999*v[k]+0.001*grad[k]*grad[k]
                    self.params[k]-=lr*(m[k]/(1-0.9**t))/(np.sqrt(v[k]/(1-0.999**t))+1e-8)
            p=np.clip(self.forward(xx),1e-7,1-1e-7)
            loss=float(np.mean(-y*np.log(p)-(1-y)*np.log(1-p)))
            history.append(loss)
        return history

    def save(self,path=MODEL_PATH):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        np.savez(path,mean=self.mean,std=self.std,features=np.array(FEATURES),**self.params)

    @classmethod
    def load(cls,path=MODEL_PATH):
        with np.load(path,allow_pickle=False) as data:
            if tuple(data['features'].tolist())!=FEATURES:raise ValueError('Feature schema mismatch')
            model=cls()
            model.mean=data['mean'];model.std=data['std']
            for k in model.params:model.params[k]=data[k]
        return model


class LogisticBaseline:
    def __init__(self):
        self.mean=None;self.std=None;self.w=None

    def train(self,x,y,epochs=450,lr=0.075):
        self.mean=x.mean(axis=0);self.std=np.maximum(x.std(axis=0),1e-8)
        X=np.column_stack((np.ones(len(x)),(x-self.mean)/self.std))
        y=np.asarray(y)
        self.w=np.zeros(X.shape[1]);n=len(y)
        for _ in range(epochs):
            p=sigmoid(X@self.w)
            grad=X.T@(p-y)/n+np.r_[0,0.002*self.w[1:]]
            self.w-=lr*grad

    def predict_proba(self,x):
        x=np.asarray(x,dtype=float)
        if x.ndim==1:x=x[None,:]
        X=np.column_stack((np.ones(len(x)),(x-self.mean)/self.std))
        return sigmoid(X@self.w)
