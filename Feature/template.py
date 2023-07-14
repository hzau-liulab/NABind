import os
import pickle
import joblib
import subprocess
import json
with open(os.path.join(os.path.dirname(os.path.realpath(__file__)),'../config/config.json'),'r') as f:
    globalpara=json.load(f)

def NWalign(x,y):
    def get_identity(o):
        return float(o.split('\n')[5].split()[2])
    
    command=' '.join([globalpara['nwalign'],x,y,'1'])
    out=subprocess.run(command,shell=True,capture_output=True,check=True,text=True)
    return y,get_identity(out.stdout)

def TMalign(x,y,idt=None):
    def get_TMscore(o):
        tm1,tm2=o.split('\n')[13:15]
        return float(tm1.split()[1]),float(tm2.split()[1])
    
    command=' '.join([globalpara['tmalign'],x,y])
    out=subprocess.run(command,shell=True,capture_output=True,check=True,text=True)
    return os.path.basename(y).replace('.pdb',''),get_TMscore(out.stdout),idt

def search(pdb,database,id_cut=0.30,tm_cut=0.3):
    p=joblib.Parallel(n_jobs=-1)
    pdb_identitys=p(joblib.delayed(NWalign)(pdb,database+y) for y in os.listdir(database))
    pdb_identitys=list(filter(lambda x:x[1]<=id_cut,pdb_identitys))
    
    p=joblib.Parallel(n_jobs=-1)
    pdb_tmscores_indentitys=p(joblib.delayed(TMalign)(pdb,y,idt) for y,idt in pdb_identitys)
    pdb_tmscores_indentitys=list(filter(lambda x:x[1][1]>=tm_cut,pdb_tmscores_indentitys))
    return pdb_tmscores_indentitys

from cdhit_reader import read_cdhit

def generate_fasta(record_dict,keys,wpath):
    fasta=list()
    for k in keys:
        fasta.append('>'+k)
        fasta.append(record_dict[k])
    wpath=os.path.join(wpath,'fasta.fasta')
    with open(wpath,'w') as f:
        for l in fasta:
            f.write('{}\n'.format(l))
    return wpath

def excute_cdhit(fasta,out,idcut):
    def get_wordsize(idcut):
        if idcut<0.5:
            w=2
        elif idcut<0.6:
            w=3
        elif idcut<0.7:
            w=4
        else:
            w=5
        return w
    
    command=' '.join([globalpara['cdhit'],
                      '-i',fasta,'-o',out,'-c',str(idcut),'-T 0 -M 0',
                      '-n',str(get_wordsize(idcut))])
    subprocess.run(command,shell=True)

def cdhit_parse(clstr):
    clstrdict=dict() #key->cls vaule->seq name list
    for cluster in read_cdhit(clstr):
        clstrdict[cluster.name]=[x.name for x in cluster.sequences]
    return clstrdict

def diversity_select(pdb_tmscore_idt_dict,record_dict,wpath,idcut,templates):
    def get_ref(pdb_tmscore_idt_dict,candidate):
        candidate.sort(key=lambda x:pdb_tmscore_idt_dict[x][0][1],reverse=True)
        ref=candidate.pop(0)
        return ref,candidate
    
    def base(pdb_tmscore_idt_dict,clstrdict,templates,selected=list()):
        clusters=[key for key,value in clstrdict.items() if value]
        if len(clusters)>=(templates-len(selected)):
            refs=[get_ref(pdb_tmscore_idt_dict,clstrdict[x])[0] for x in clusters]
            refs.sort(key=lambda x:pdb_tmscore_idt_dict[x][0][1],reverse=True)
            selected.extend(refs[:templates-len(selected)])
        return selected
    
    def select(pdb_tmscore_idt_dict,clstrdict,templates,selected=list()):
        selected=base(pdb_tmscore_idt_dict,clstrdict,templates,selected=selected)
        if len(selected)==templates:
            return selected
        for key in clstrdict:
            if not clstrdict[key]:
                continue
            candidate=clstrdict[key]
            ref,candidate=get_ref(pdb_tmscore_idt_dict,candidate)
            selected.append(ref)
            clstrdict[key]=candidate
        return select(pdb_tmscore_idt_dict,clstrdict,templates,selected)
    
    wpath=generate_fasta(record_dict,keys=pdb_tmscore_idt_dict.keys(),wpath=wpath)
    excute_cdhit(wpath,wpath.replace('.fasta','.out'),idcut)
    clstrdict=cdhit_parse(wpath.replace('.fasta','.out.clstr'))
    if len(sum(list(clstrdict.values()),[]))<templates:
        print('DEPRECATED\t{}\twith total templates {}'.format(wpath.split('/')[-2],len(sum(list(clstrdict.values()),[]))))
        return []
    return select(pdb_tmscore_idt_dict,clstrdict,templates)

def get_top_templates(pdb_tmscore_idt,record,wpath,idcut,templates):
    with open(pdb_tmscore_idt,'rb') as f:
        pdb_tmscore_idt=pickle.load(f)
    pdb_tmscore_idt_dict=dict(zip([x[0] for x in pdb_tmscore_idt],[x[1:] for x in pdb_tmscore_idt]))
    with open(record,'rb') as f:
        record=pickle.load(f)
    record_dict=dict(zip([''.join(x.split('\t')[:2]) for x in record],
                          [x.split('\t')[19].strip() for x in record]))
    
    if int(idcut)==1:
        pdb_tmscore_idt.sort(key=lambda x:x[1][1],reverse=True)
        toptemplates=[x[0] for x in pdb_tmscore_idt[:templates]]
    else:
        toptemplates=diversity_select(pdb_tmscore_idt_dict,record_dict,
                                      wpath=wpath,idcut=idcut,templates=templates)
    return toptemplates,pdb_tmscore_idt_dict

def alignment(q,k,database,outpath):
    outpath=os.path.join(outpath,k)
    os.makedirs(outpath,exist_ok=True)
    command=' '.join([
                      globalpara['tmalign'],
                      q,os.path.join(database,k)+'.pdb',
                      '-o',outpath+'/sup',
                      '>',outpath+'/log'])
    subprocess.run(command,shell=True)

def align_top_templates(pdb_tmscore_idt,record,outpath,database,q,idcut,templates=20):
    os.makedirs(outpath,exist_ok=True)
    record,recorddict=get_top_templates(pdb_tmscore_idt,record,
                                        wpath=outpath,
                                        idcut=idcut,templates=templates)
    with open(outpath+'/pdb_tmscore_idt.pkl','wb') as f:
        pickle.dump(recorddict,f)
    [alignment(q,k,database,outpath) for k in record]

import numpy as np

def BindingMap(tmalignout,bindtag):
    def tm_deal(tmalignout):
        x=tmalignout.split('\n')
        tm1=x[18].strip()
        tm2=x[20].strip()
        seq1=[i for i in range(len(tm1)) if tm1[i] != '-']
        seq2=[i for i in range(len(tm2)) if tm2[i] != '-']
        seq1={j:i for i,j in enumerate(seq1)} #insert index to noinsert index
        seq2={j:i for i,j in enumerate(seq2)}
        maps=[i for i in range(len(tm1)) if (tm1[i] != '-' and tm2[i] != '-')]
        maps={seq2[i]:seq1[i] for i in maps}
        return len(seq1),maps,(tm1.replace('-',''),tm2.replace('-',''))
    
    length,maps,(seq1,seq2)=tm_deal(tmalignout)
    bind=list(bindtag)
    index=[maps[x] for x in bind if x in maps]
    feat=np.zeros(length,dtype=int)
    feat[index]=1
    return feat,(length,maps),(seq1,seq2)

from collections import defaultdict
from utils.Bindingfuc import Binding

def get_ligandpdb(record):
    def get_key(x):
        return ''.join(x.split('\t')[:2])
    
    def get_value(x):
        xl=x.split('\t')
        return '_'.join([xl[0],xl[4],xl[5],xl[6]])
    
    with open(record,'rb') as f:
        record=pickle.load(f)
    liganddict=defaultdict(list)
    for x in record:
        liganddict[get_key(x)].append(get_value(x))
    return liganddict

def get_distbind_feat(tarpdb,liganddict,ligand_database,dist_cut):
    def generate_ligand(key,liganddict,ligand_database):
        ligands=liganddict[key]
        ligandpdb=list()
        for l in ligands:
            try:
                with open(os.path.join(ligand_database,l+'.pdb'),'r') as f:
                    ligandpdb+=f.readlines()
            except:
                with open(os.path.join(ligand_database.replace('ligand','BioLiP_updated_set/ligand'),l+'.pdb'),'r') as f:
                    ligandpdb+=f.readlines()
        return ligandpdb
    
    ligandpdb=generate_ligand(tarpdb.split('/')[-2],liganddict,ligand_database)
    bind=Binding(tarpdb,ligandpdb,dist_cut)
    return bind.res_array()[:,[1]].astype(float)

from utils.PDBfuc import PDB
from scipy.spatial.distance import cdist

def DistanceMap(pdb1,pdb2,maps,length):
    #maps: pdb2->pdb1
    def get_coord(coorddict,res,atomtype='CA'):
        if res in coorddict:
            if atomtype in coorddict[res]:
                coord=coorddict[res][atomtype]
            elif 'CB' in coorddict[res]:
                coord=coorddict[res]['CB']
            else:
                coord=np.mean(list(coorddict[res].values()),axis=0)
        else:
            raise ValueError('Not exists {}'.format(res))
        return np.array(coord)
    
    def get_coords(pdb1,pdb2):
        pdb1=PDB(pdb1)
        pdb2=PDB(pdb2)
        coords1=[get_coord(pdb1.coord,res) for _,_,res in pdb1.xulie]
        coords2=[get_coord(pdb2.coord,res) for _,_,res in pdb2.xulie]
        return coords1,coords2
    
    feat=np.full(length,-1,dtype=float)
    coords1,coords2=get_coords(pdb1,pdb2)
    index_dist=[[i,cdist([coords1[i]],[coords2[j]],metric='euclidean')[0][0]] 
                for j,i in maps.items()]
    feat[[x[0] for x in index_dist]]=[x[1] for x in index_dist]
    return feat

from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP

def SSMap(pdb1,pdb2,maps,length):
    def ss2index(ss):
        if ss in ['H','G','I']:
            return 0 #helix
        if ss in ['E','B']:
            return 1 #strand
        if ss in ['T','S','L']:
            return 2 #coil
        if ss in ['-']:
            return 2
    
    def get_ss(pdb1,pdb2):
        def ss(structure,dssp):
            out=list()
            for k_ref in structure.get_residues():
                k_ref=k_ref.full_id[2:]
                if k_ref in dssp:
                    out.append(dssp[k_ref][2])
                else:
                    out.append('-')
            return out
        
        def check(structure):
            #exclued some all CA items
            if len(list(structure.get_residues()))==len(list(structure.get_atoms())):
                return False
            else:
                return True
        
        p=PDBParser()
        structure1=p.get_structure('pdb1',pdb1)
        structure2=p.get_structure('pdb2',pdb2)
        check_stas=[check(structure1),check(structure2)]
        if any(x==False for x in check_stas):
            ss1=['-',]*len(list(structure1.get_residues()))
            ss2=['-',]*len(list(structure2.get_residues()))
            print('NO DSSP: {}  {}'.format(pdb1,pdb2))
        else:
            dssp1=DSSP(structure1[0],pdb1,dssp=globalpara['dssp'])
            dssp2=DSSP(structure2[0],pdb2,dssp=globalpara['dssp'])
            ss1=ss(structure1,dssp1)
            ss2=ss(structure2,dssp2)
        ss1=[ss2index(x) for x in ss1]
        ss2=[ss2index(x) for x in ss2]
        return ss1,ss2
    
    def get_energy(ss1,ss2):
        ss=[ss1,ss2]
        if all(x==0 for x in ss):
            out=2
        elif all(x==1 for x in ss):
            out=2
        elif all(x==2 for x in ss):
            out=1
        elif all(x in ss for x in [0,1]):
            out=-0.5
        elif all(x in ss for x in [0,2]):
            out=0.
        elif all(x in ss for x in [1,2]):
            out=0.5
        else:
            raise ValueError('check ss')
        return out
    feat=np.full(length,-1,dtype=float)
    ss1,ss2=get_ss(pdb1,pdb2)
    index_energy=[[i,get_energy(ss1[i],ss2[j])] 
                  for j,i in maps.items()]
    feat[[x[0] for x in index_energy]]=[x[1] for x in index_energy]
    return feat

from Bio.SubsMat import MatrixInfo

def SequenceMap(seq1,seq2,maps,length):
    def get_matrix():
        return MatrixInfo.blosum62
    
    def get_score(res1,res2,matrix):
        if (res1,res2) in matrix:
            return matrix[(res1,res2)]
        elif (res2,res1) in matrix:
            return matrix[(res2,res1)]
        else:
            return 0 #to deal some HETATM restype
        
    feat=np.full(length,-4,dtype=int)
    matrix=get_matrix()
    index_score=[[i,get_score(seq1[i],seq2[j],matrix)] 
                  for j,i in maps.items()]
    feat[[x[0] for x in index_score]]=[x[1] for x in index_score]
    return feat

from threading import Thread

class MyThread(Thread):
    def __init__(self, func, args):
        Thread.__init__(self)
        self.func = func
        self.args = args
        self.result = None

    def run(self):
        self.result = self.func(*self.args)

    def getResult(self):
        return self.result

def TemplateFeat(pdbdir,database,binddict,oripdb,record,ligand_database,tmscore_idt_dict=None):
    # templates: 20 default
    with open(binddict,'rb') as f:
        binddict=pickle.load(f)
    
    def get_res_feat(tmalignout,bindtag,oripdb,suppdb,templatepdb):
        _,(length,maps),(seq1,seq2)=BindingMap(tmalignout,bindtag)
        bindfeat=get_distbind_feat(suppdb,get_ligandpdb(record),ligand_database,dist_cut=4.5)
        
        t1=MyThread(SequenceMap,(seq1,seq2,maps,length))
        t2=MyThread(SSMap,(oripdb,templatepdb,maps,length))
        t3=MyThread(DistanceMap,(suppdb,templatepdb,maps,length))
        t1.start()
        t2.start()
        t3.start()
        t1.join()
        t2.join()
        t3.join()
        feat=np.column_stack([bindfeat,t1.getResult(),
                                  t2.getResult(),t3.getResult()])
        
        # print(oripdb,templatepdb)
        # seqfeat=SequenceMap(seq1,seq2,maps,length)
        # ssfeat=SSMap(oripdb,templatepdb,maps,length)
        # distfeat=DistanceMap(suppdb,templatepdb,maps,length)
        # feat=np.column_stack((bindfeat,seqfeat,ssfeat,distfeat))
        
        return feat
    
    def get_seq_feat(tmscore,idt,length):
        return np.column_stack(([tmscore,]*length,[idt,]*length))
    
    with open(pdbdir+'/pdb_tmscore_idt.pkl','rb') as f:
        tmscore_idt_dict=pickle.load(f)
    
    def get_feat(name,pdbdir,database,binddict,oripdb,tmscore_idt_dict):
        with open(os.path.join(pdbdir,name,'log'),'r') as f:
            tmalignout=f.read()
        suppdb=os.path.join(pdbdir,name,'sup.pdb')
        templatepdb=os.path.join(database,name+'.pdb')
        bindtag=binddict[name]
        resfeat=get_res_feat(tmalignout,bindtag,oripdb,suppdb,templatepdb)
        seqfeat=get_seq_feat(tmscore_idt_dict[name][0][1],tmscore_idt_dict[name][1],resfeat.shape[0])
        feat=np.hstack((resfeat,seqfeat))
        return feat
    
    feat=[get_feat(name,pdbdir,database,binddict,oripdb,tmscore_idt_dict) 
          for name in os.listdir(pdbdir) 
          if os.path.isdir(os.path.join(pdbdir,name))]
    feat=np.hstack(feat).astype(float)
    
    def feat_sort(array):
        index=np.arange(0,array.shape[1],6)
        arraylist=[array[:,x:x+6] for x in index]
        tmscorelist=[array[0,x+4] for x in index]
        index=np.argsort(tmscorelist)[::-1] #sorted max to min
        arraylist=np.hstack([arraylist[x] for x in index])
        return arraylist
    
    sorted_feat=feat_sort(feat)
    cols=sorted_feat.shape[1]
    if cols<120:
        paded_cols=120-cols
        sorted_feat=np.pad(sorted_feat,((0,0),(0,paded_cols)),mode='constant')
    
    return sorted_feat
