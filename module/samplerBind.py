import pickle
import random
import itertools
import numpy as np
from collections import namedtuple

import torch
import dgl


keypoint=namedtuple('keypoint',
                    ['bind','sequence','res','pdb','contact','feature'])

class Mode:
    modelist=['train','validate','test']
    
    def __init__(self,datasetsfile,datatype,fold=None):
        with open(datasetsfile,'rb') as f:
            self.dataset=pickle.load(f)[datatype.split('_')[0]]
        if fold!=None:
            self.dataset=self.dataset[int(fold)]
        self.mode=None
        
    def set_mode(self,mode):
        if mode not in self.modelist:
            raise ValueError('not accepted mode')
        self.mode=mode


class Sampler(Mode):
    def __init__(self,datadir,datatype,fold=None):
        with open(datadir+datatype+'.pkl','rb') as f:
            self.data=pickle.load(f)
        
        super().__init__(datadir+'datasets.pkl',datatype,fold=fold)
        # self.dataset=self.oridataset[datatype.split('_')[0]]
        
        self.samples_from_mode=dict.fromkeys(self.modelist)
        self.sampleindex_from_mode=dict.fromkeys(self.modelist)
        self.batchindex_from_mode=dict.fromkeys(self.modelist,0)
        
        self._train_samples_index=None
    
    def sample(self,batch_size=1,sample_weight=None):
        chains=np.array(self.dataset[self.mode])
        no_of_chains=len(chains)
        
        if sample_weight is not None:
            sample_weight.sort(key=lambda x:list(chains).index(x[0]))
            sample_weight=np.array([x[1] for x in sample_weight],dtype=float)
            sample_weight=sample_weight/np.sum(sample_weight)
        
        if not self.sampleindex_from_mode[self.mode]:
            self.sampleindex_from_mode[self.mode]=self._divide_to_batch(no_of_chains,batch_size,sample_weight)
        if self.batchindex_from_mode[self.mode]>=len(self.sampleindex_from_mode[self.mode]):
            self.batchindex_from_mode[self.mode]=0
            self.sampleindex_from_mode[self.mode]=self._divide_to_batch(no_of_chains,batch_size,sample_weight)
        
        sampleindexlist=self.sampleindex_from_mode[self.mode]
        batchindex=self.batchindex_from_mode[self.mode]
        self.batchindex_from_mode[self.mode]+=1
        
        sampleindex=sampleindexlist[batchindex]
        select_chains=chains[sampleindex]
        
        out_features=np.array(list(map(lambda x:self.data[x].feature,select_chains)),dtype=float)
        out_adj=np.array(list(map(lambda x:self.data[x].contact,select_chains)))
        out_targets=np.vstack(list(map(lambda x:self.data[x].bind.reshape(-1,1),
                                       select_chains)))
        out_tags=np.vstack(list(
                                map(lambda x:np.array(list(map(lambda y:[x+'_'+str(y)],self.data[x].res))),
                                    select_chains)
                                ))
        if batch_size==1:
            out_features=out_features.squeeze(0)
            out_adj=out_adj.squeeze(0)
        
        return out_features,\
               out_adj,\
               out_targets.astype(int),\
               out_tags.astype(str)
    
    def _divide_to_batch(self,length,batch_size,p=None):
        tmplist=list(range(length))
        if p is not None:
            tmplist=list(np.random.choice(tmplist,len(tmplist),replace=True,p=p))
        while length%batch_size!=0:
            tmplist.append(random.choice(tmplist))
            length+=1
        random.shuffle(tmplist)
        return list(map(lambda x:tmplist[x:x+batch_size],range(0,len(tmplist),batch_size)))
    
    def _get_features_targets_tags(self,batch_size,chain=False):
        if chain:
            no_of_samples=len(self.dataset[self.mode])
        else:
            no_of_samples=len(self.samples_from_mode[self.mode][0])
        
        if no_of_samples%batch_size==0:
            n_step=int(no_of_samples/batch_size)
        else:
            n_step=int(no_of_samples/batch_size)+1
        inputs_and_targets=[]
        alltargets=[]
        alltags=[]
        for _ in range(n_step):
            inputs,adj,targets,tags=self.sample(batch_size)
            inputs_and_targets.append((inputs,adj,targets))
            alltargets.append(targets)
            alltags.append(tags)
        return (inputs_and_targets,
                np.vstack(tuple(alltargets)),
                np.vstack(tuple(alltags)))
    
    def get_validation_set(self,batch_size):
        self.set_mode('validate')
        return self._get_features_targets_tags(batch_size)
    
    def get_test_set(self,batch_size):
        self.set_mode('test')
        return self._get_features_targets_tags(batch_size)
    
    def get_train_set(self,batch_size):
        self.set_mode('train')
        self.batchindex_from_mode[self.mode]=0
        self.sampleindex_from_mode[self.mode]=None
        out=self._get_features_targets_tags(batch_size)
        self.batchindex_from_mode[self.mode]=0
        self.sampleindex_from_mode[self.mode]=None
        return out


class SamplerBind(Sampler):
    def __init__(self,datadir,datatype,fold):
        super().__init__(datadir,datatype,fold)
        
    def sample(self,batch_size=1,sample_weight=None):
        # if sample_weight is not None:
        #     rnatags=dict(map(lambda x:(x[0],int(x[1])),np.loadtxt('./RNAtag.txt',dtype=str)))
        #     sample_weight=list(map(lambda x:[x[0],rnatags[x[0]]],sample_weight))
        
        old_return=list(super().sample(batch_size=batch_size,sample_weight=sample_weight))
        rna='_'.join(old_return[3][0,0].split('_')[:2])
        res_index={str(j):i for i,j in enumerate(self.data[rna].res)}
        g=dgl.heterograph(data_dict={('res','res2res','res'):
                     (list(map(lambda x:res_index[x.split('_')[1]],old_return[1][:,0])),
                      list(map(lambda x:res_index[x.split('_')[1]],old_return[1][:,1])))
                               })
        g=dgl.add_self_loop(g)  #include self-loop
        old_return[1]=g
        return tuple(old_return)
    
    def _get_features_targets_tags(self,batch_size,chain=True):
        return super()._get_features_targets_tags(batch_size,chain)


class SamplerEGAT(SamplerBind):
    def __init__(self,datadir,datatype,fold=None,normalize=None,distcut=None):
        super().__init__(datadir,datatype,fold)
        
        with open(datadir+'/nonormdistancemap.pkl','rb') as f:
            self.distmap=pickle.load(f)
        with open(datadir+'/anglemap.pkl','rb') as f:
            self.anglemap=pickle.load(f)
        
        if normalize:
            self._normalize(normalize)
        
        self.distcut=distcut
    
    def sample(self,batch_size=1,sample_weight=None):
        old_return=list(Sampler.sample(self,batch_size,sample_weight))
        rna='_'.join(old_return[3][0,0].split('_')[:2])
        res_length=len(self.data[rna].res)
        res_index=list(itertools.product(range(res_length),range(res_length)))
        g=dgl.graph(
                      ([x[0] for x in res_index],
                      [x[1] for x in res_index])
                                )
        anglemap=self.anglemap[rna]
        nonormdistmap=self.distmap[rna]
        normdistmap=(nonormdistmap-np.mean(nonormdistmap))/np.std(nonormdistmap)
        g.edata['edistangle']=torch.Tensor([(normdistmap[x[0],x[1]],anglemap[x[0],x[1]]) for x in res_index])
        g.edata['ex']=g.edata.pop('edistangle')
        g.edata['edist']=torch.Tensor([[nonormdistmap[x[0],x[1]]] for x in res_index])
        sg_dist=dgl.edge_subgraph(g,g.edata['edist'].flatten()<self.distcut,preserve_nodes=True)
        old_return[1]=sg_dist
        new_return=tuple(old_return)
        return new_return
    
    def _normalize(self,columns):#columns->list
        trainfeat=np.vstack([self.data[x].feature[:,columns].astype(float) for x in self.dataset['train']])
        mean=np.mean(trainfeat,axis=0)
        std=np.std(trainfeat,axis=0)
        for key in self.data:
            old_tuple=self.data[key]
            feat=self.data[key].feature.astype(float)
            feat[:,columns]=(feat[:,columns]-mean)/std
            new_tuple=old_tuple._replace(feature=feat)
            self.data[key]=new_tuple
