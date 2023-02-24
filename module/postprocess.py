import os
import copy
import pickle
import numpy as np

class RandomWalk(object):
    def __init__(self,nt_prob,weight_matrix):
        self.w=weight_matrix
        self.nt_prob=nt_prob
        
    def walk(self,a=0.4,end=10e-6):
        p0=self.nt_prob[:,-1].astype(float)
        delta,restart,pt,count=1,p0*a,p0,0
        while(delta>end):
            count+=1
            transfer=pt.dot(self.w)*(1-a)
            pt_=transfer+restart
            delta=np.sum(np.abs(pt_-pt))
            pt=pt_
        return np.column_stack((self.nt_prob[:,:-1],pt_))
      
    def multi_walk(self,a,end):
        # pre=list(map(lambda x:self.walk(self.nt_prob[x[1],:],self.w_dict[x[0]],a,end),self.rna_index.items()))
        # return np.vstack(pre)
        pass

def get_maskmap(distmap,ref,cut):
    def mask(array,ref,cut):
        assert len(array)==len(ref)
        index=np.where(ref<=cut)[0]
        array[index,:]=-1.
        array[:,index]=-1.
        return array
    return mask(distmap,ref,cut)

def get_maskmap2(distmap,maskres):
    tmp=distmap
    tmp[maskres,:]=-1
    tmp[:,maskres]=-1
    maskmap=tmp
    return maskmap

def get_distmap(distmap,cut,geodesic=False,ref=None,geodesic_cut=0.1,maskres=None):
    def keep_mask(maparray):
        for i in range(maparray.shape[0]):
            if (maparray[i]==0).all():
                maparray[i,i]=1
        return maparray
    
    with open(distmap,'rb') as f:
        distmap=pickle.load(f)
    if geodesic:
        distmap=get_maskmap(distmap,ref,geodesic_cut)
    if maskres!=None:
        distmap=get_maskmap2(distmap,maskres)
    tmp=distmap
    tmp[np.where(tmp<0)]=np.inf
    tmp[np.diag_indices_from(tmp)]=0 #self loop or NOT
    if cut:
        tmp=cut-tmp
    else:
        tmp=np.max(tmp,axis=0)-tmp
    tmp[np.where(tmp<0)]=0
    # tmp=keep_mask(tmp)
    outmap=tmp/np.sum(tmp,axis=0)
    return outmap

def postprocess(nt_prob,distmap,rsa,**kwargs):
    w=get_distmap(distmap,cut=kwargs['Rdist'],geodesic=True,ref=rsa,geodesic_cut=kwargs['surfacersa'],)
    rw=RandomWalk(nt_prob,w)
    return rw.walk(kwargs['a'],kwargs['end'])
