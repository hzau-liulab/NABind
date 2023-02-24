import os
import pickle
from collections import namedtuple
import numpy as np
from .ESM import esmmsa1b,esmif
from .sequence import get_pssm_hhm
from .structure import RicciCurvature,MultifractalDim,MathMorphologyPocket,MultiDistance,UltrafastShape,Topo

keypoint=namedtuple('keypoint',
                    ['bind','sequence','res','pdb','contact','feature'])

def get_seq_str_feature(pdb,**kwargs):
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)),'standard.pkl'),'rb') as f:
        standard=pickle.load(f)
    
    def get_str_feature(pdb):
        topo=Topo(pdb)
        contact=topo.contact(5)
        distmap=topo.distmap
        anglemap=topo.angle_map()
        res=[x[-1] for x in topo.xulie]
        rc=RicciCurvature(contact)
        orc=rc.ollivier()
        frc=rc.forman()
        mfd=MultifractalDim(contact)
        slope=mfd.slope()
        graphfeat=np.array([[orc[x],frc[x],slope[x]] for x in res],dtype=float)
        mmp=MathMorphologyPocket(pdb,ghecomexe=kwargs['ghecom'])
        pockets=mmp.res_array()[:,1:].astype(float)
        mtltidist=MultiDistance(pdb).res_array()
        usr=UltrafastShape(pdb).res_array()
        ofmp=np.column_stack((graphfeat,pockets))
        ofmp=(ofmp-standard[kwargs['type']][kwargs['structure']][0])/standard[kwargs['type']][kwargs['structure']][1]
        return distmap,anglemap,np.column_stack((ofmp,mtltidist,usr))
        
    seqfeat=get_pssm_hhm(pdb,out=kwargs['outdir'],pssmdb=kwargs['pssmdb'],hhmdb=kwargs['hhmdb'],pssmexe=kwargs['pssm'],hhmexe=kwargs['hhm'])
    seqesm=esmmsa1b('{}/hhm.a3m'.format(kwargs['outdir'])).get_embedding()
    
    stresm=esmif(pdb).get_embedding()
    # with open('{}/esmif.pkl'.format(kwargs['outdir']),'rb') as f:
    #     stresm=pickle.load(f)
    
    distmap,anglemap,strfeat=get_str_feature(pdb)
    feature=np.hstack((strfeat,seqfeat,seqesm,stresm))
    with open('{}/nonormdistancemap.pkl'.format(kwargs['outdir']),'wb') as f:
        pickle.dump(distmap,f)
    with open('{}/anglemap.pkl'.format(kwargs['outdir']),'wb') as f:
        pickle.dump(anglemap,f)
    featuredict={'test':keypoint(bind=None,sequence=None,res=None,pdb=None,contact=None,feature=feature)}
    with open('{}/test_DL_data.pkl'.format(kwargs['outdir']),'wb') as f:
        pickle.dump(featuredict,f)
    return feature,distmap,anglemap

from .template import search,align_top_templates,TemplateFeat

def get_template_feature(pdb,**kwargs):
    pdb_tmscores_indentitys=search(pdb,
                                    '{}/{}/receptor/'.format(kwargs['templatedb'],kwargs['type']),
                                    id_cut=kwargs['idt_for_q_and_k'],
                                    tm_cut=0.3)
    with open('{}/pdb_tmscore_idt.pkl'.format(kwargs['outdir']),'wb') as f:
        pickle.dump(pdb_tmscores_indentitys,f)
    align_top_templates('{}/pdb_tmscore_idt.pkl'.format(kwargs['outdir']),
                        '{}/{}/record.pkl'.format(kwargs['templatedb'],kwargs['type']),
                        outpath=kwargs['outdir'],
                        database='{}/{}/receptor/'.format(kwargs['templatedb'],kwargs['type']),
                        q=pdb,
                        idcut=kwargs['idt_for_ks'],
                        templates=kwargs['templates'],
                        )
    feature=TemplateFeat(kwargs['outdir'],
                          '{}/{}/receptor/'.format(kwargs['templatedb'],kwargs['type']),
                          '{}/{}/binddict.pkl'.format(kwargs['templatedb'],kwargs['type']),
                          pdb,
                          '{}/{}/record.pkl'.format(kwargs['templatedb'],kwargs['type'],),
                          kwargs['templatedb'].replace('NUC','ligand'),
                          )
    with open('{}/test_TL_data.pkl'.format(kwargs['outdir']),'wb') as f:
        pickle.dump(feature,f)
    return feature

if __name__=='__main__':
    import json
    with open('../config/config.json','r') as f:
        config=json.load(f)
    config['type']='DNA'
    config['structure']='native'
    get_seq_str_feature('../demo/6chv_D.pdb',**config)
    # get_template_feature('../demo/6chv_D.pdb',**config)
    