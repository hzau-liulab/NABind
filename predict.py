import dgl
import torch
import itertools
import numpy as np
from collections import OrderedDict
from module.modelsBind import EGATM
from module.postprocess import postprocess
from Feature.feature import get_seq_str_feature,get_template_feature
from utils.PDBfuc import PDB
from utils.FEAfuc import RSA

import pickle
from collections import namedtuple
keypoint=namedtuple('keypoint',
                    ['bind','sequence','res','pdb','contact','feature'])

def sample(pdb,config):
    feat,nonormdistmap,anglemap=get_seq_str_feature(pdb,**config)
    # with open('./demo/test_DL_data.pkl','rb') as f:
    #     feat=pickle.load(f)['test'].feature
    # with open('./demo/nonormdistancemap.pkl','rb') as f:
    #     nonormdistmap=pickle.load(f)
    # with open('./demo/anglemap.pkl','rb') as f:
    #     anglemap=pickle.load(f)
    res_length=len(feat)
    res_index=list(itertools.product(range(res_length),range(res_length)))
    g=dgl.graph(([x[0] for x in res_index],[x[1] for x in res_index]))
    normdistmap=(nonormdistmap-np.mean(nonormdistmap))/np.std(nonormdistmap)
    normanglemap=(anglemap-np.mean(anglemap))/np.std(anglemap)
    g.edata['ex']=torch.Tensor([(normdistmap[x[0],x[1]],normanglemap[x[0],x[1]]) for x in res_index])
    g.edata['edist']=torch.Tensor([[nonormdistmap[x[0],x[1]]] for x in res_index])
    sg=dgl.edge_subgraph(g,g.edata['edist'].flatten()<config['radius'],preserve_nodes=True)
    sg.edata.pop('edist')
    return sg,torch.Tensor(feat)

def load_model_from_state_dict(state_dict, model):
    model_keys = model.state_dict().keys()
    state_dict_keys = state_dict.keys()

    new_state_dict = OrderedDict()
    for (k1, k2) in zip(model_keys, state_dict_keys):
        value = state_dict[k2]
        if k1 == k2:
            new_state_dict[k2] = value
        elif ('module' in k1 and k2 in k1) \
                or ('module' in k2 and k1 in k2):
            new_state_dict[k1] = value
        else:
            raise ValueError("Model state dict keys do not match "
                             "the keys specified in `state_dict` input. "
                             "Cannot load state into the model:\n\n"
                             "\tExpected keys:\n\t{0}\n\n"
                             "\tKeys in the input state dict:\n\t{1}\n".format(
                                 model_keys, state_dict_keys))
    model.load_state_dict(new_state_dict)
    return model

def DLmodule(config):
    g,h=sample(config['pdb'],config)
    
    def get_proba(g,h,checkpoint_resume):
        model=EGATM()
        checkpoint=torch.load(checkpoint_resume,map_location=lambda storage, location: storage)
        model=load_model_from_state_dict(checkpoint["state_dict"],model)
        model.eval()
        pre=model(h,g)
        proba=pre.data.cpu().numpy()
        return 1/(1+np.exp(-proba))

    proba=np.column_stack([get_proba(g,h,'{}/{}_{}_{}.tar'.format(config['modeldir'],config['type'],config['structure'],i)) for i in range(5)])
    return proba

def TLmodule(config):
    feat=get_template_feature(config['pdb'],**config)
    # with open('./demo/test_TL_data.pkl','rb') as f:
    #     feat=pickle.load(f)
    
    def get_proba(feat,clf):
        return clf.predict_proba(feat)[:,[-1]]
    
    with open('{}/{}_{}.pkl'.format(config['modeldir'],config['type'],config['structure']),'rb') as f:
        clfs=pickle.load(f)
    proba=np.column_stack([get_proba(feat,clf) for clf in clfs])
    return proba

def Mer(config):
    feat=np.hstack((DLmodule(config),TLmodule(config)))
    with open('{}/{}_{}_.pkl'.format(config['modeldir'],config['type'],config['structure']),'rb') as f:
        clf=pickle.load(f)
    return clf.predict_proba(feat)[:,[-1]]

def PostProcess(config):
    proba=Mer(config)
    # proba=np.load('./Merpre.npy')
    pdb=PDB(config['pdb'])
    nt_proba=np.column_stack(([[x[0],x[2]] for x in pdb.xulie],proba))
    rsa=RSA(pdbfile=config['pdb'],**config).res_array()
    result=postprocess(nt_proba,'{}/nonormdistancemap.pkl'.format(config['outdir']),
                rsa,**config)
    return result

def predict(config):
    def get_binary(proba):
        cutoff=config['cutoff_{}_{}'.format(config['type'],config['structure'])]
        return np.where(proba>cutoff,1,0)
        
    result=PostProcess(config)
    binary=get_binary(result[:,-1].astype(float)).astype(int)
    return np.column_stack((result,binary))

def main(config):
    np.savetxt('{}/result.txt'.format(config['outdir']),
               predict(config),
               fmt='%s',
               delimiter='\t',
               header='\t'.join(['AA','RES','Proba','Binary']))
    

if __name__=='__main__':
    import argparse
    
    parse=argparse.ArgumentParser()
    parse.add_argument('--pdb',type=str,default='./demo/6chv_D.pdb',help='input pdb')
    parse.add_argument('--outdir',type=str,default='./demo/',help='output directory')
    parse.add_argument('--type',type=str,default='DNA',help='predict DNA- or RNA-binding residues (DNA/RNA)')
    parse.add_argument('--structure',type=str,default='native',help='choose models trained on native or predicted structures (native/predicted)')
    args=parse.parse_args()
    
    import json
    
    with open('./config/config.json','r') as f:
        config=json.load(f)
    config['pdb']=args.pdb
    config['type']=args.type
    config['structure']=args.structure
    config['outdir']=args.outdir
    
    main(config)
    # print(config)
