import numpy as np
import itertools
import re
import os
from utils.PDBfuc import PDB
 
class SEQfeature(PDB):
    
    def __init__(self,pdbfile):
        super().__init__(pdbfile)
        
        self.res_restype=dict(zip([x[-1] for x in self.xulie],[self.het_to_atom[x[0]] for x in self.xulie]))
    
    def get_trinucleotides_occurrence(self,windowsize):
        """
        windowsize: int
        return: dict {res:[occurrence,]*64,......}
        """
        tlist=['A','G','C','U']
        trip=list(map(lambda x:''.join(x),itertools.product(tlist,tlist,tlist)))
        return self.get_occurrence(mode='NA',windowsize=windowsize,target=trip)
    
    def get_occurrence(self,mode,windowsize,target):
        """
        mode: NA/protein
        windowsize: int
        target: str (eg. AC/IN/UU) / list
        return: dict {res:[occurrence,occurrence,...],...}
        """
        res_window=self.get_sequence_with_window(mode,windowsize)
        if isinstance(target,str):
            target=[target]
        
        for key in res_window:
            res_window[key]=[res_window[key].count(x) for x in target]
        return res_window
    
    def get_sequence_with_window(self,mode,windowsize):
        """
        mode: NA/protein
        windowsize: int
        return: dict {res:sequence,res:sequence,.......}
        """
        if mode not in ['NA','protein']:
            raise ValueError('not accepted mode')
        self.res_restype['NA']='-'
        
        res_window=self.get_res_seq_windows(windowsize=windowsize)
        for key in res_window:
            if mode=='NA':
                res_window[key]=''.join(list(map(lambda x:self.res_restype[x],res_window[key])))
            elif mode=='protein':
                res_window[key]=''.join(list(map(lambda x:self.three_to_one[self.res_restype[x]],res_window[key])))
        return res_window
    
    def pssm(self,pssmfile,windowsize=0,normlize=True):
        """
        pssmfile: str (file)
        windowsize: int
        return: dict {res:[fea,fea,fea,..],.......}
        """
        with open(pssmfile,'r') as f:
            flist=f.readlines()
        flist=list(filter(lambda x:re.match('\s+\d+',x),flist))
        fdict=dict(zip([x[-1] for x in self.xulie],[x.split()[2:22] for x in flist]))
        fdict['NA']=[0.]*20
        res_window=self.get_res_seq_windows(windowsize=windowsize)
        for key in res_window:
            res_window[key]=np.array(sum(list(map(lambda x:fdict[x],res_window[key])),[])).astype(np.float)
        if normlize:
            for key in res_window:
                res_window[key]=1/(1+np.exp(-res_window[key]))
        return res_window
    
    def hhm(self,hhmfile,windowsize=0,normlize=True):
        """
        hmmfile: str (file)
        windowsize: int
        NOTE: normlize is True in general
        return: dict {res:[fea,fea,fea,...],.......}
        """
        with open(hhmfile,'r') as f:
            flist=f.readlines()
        tag=list(filter(lambda x:re.match('#',x),flist))[0]
        index=flist.index(tag)+5
        hmmlist=[]
        while not re.match('//',flist[index]):
            hmmlist.append([-1000. if x=='*' else float(x) for x in flist[index].split()[2:-1]]\
                +[-1000. if x=='*' else float(x) for x in flist[index+1].split()])
            index+=3
        hmmdict=dict(zip([x[-1] for x in self.xulie],hmmlist))
        hmmdict['NA']=[-1000.]*30
        res_window=self.get_res_seq_windows(windowsize=windowsize)
        for key in res_window:
            res_window[key]=np.array(sum(list(map(lambda x:hmmdict[x],res_window[key])),[])).astype(np.float)
        if normlize:
            for key in res_window:
                tmp=2**(-0.001*res_window[key])
                tmp[np.where(tmp==2.)]=0.
                res_window[key]=tmp
        return res_window

def pssm(query,out,db,pssmexe):
    os.system('{} -query {} -db {} -num_iterations 3 -evalue 0.001 -num_threads 4 -out {} -out_ascii_pssm {}'.format(
        pssmexe,query,db,out,out+'.pssm'))

def hhm(query,out,db,hhmexe):
    os.system('{} -i {} -d {} -v 1 -cpu 4 -oa3m {}.a3m -ohhm {}.hhm'.format(
        hhmexe,query,db,out,out))

def get_pssm_hhm(pdb,out,pssmdb,hhmdb,pssmexe,hhmexe):
    seqfeat=SEQfeature(pdb)
    sequence=seqfeat.get_sequence()
    fasta='{}/fasta.txt'.format(out)
    with open(fasta,'w') as f:
        f.write('>{}\n'.format(pdb))
        f.write(''.join(sequence))
    pssm(fasta,'{}/pssm'.format(out),pssmdb,pssmexe)
    hhm(fasta,'{}/hhm'.format(out),hhmdb,hhmexe)
    pssmdict=seqfeat.pssm('{}/pssm.pssm'.format(out),)
    hhmdict=seqfeat.hhm('{}/hhm.hhm'.format(out),)
    results=list()
    for _,_,res in seqfeat.xulie:
        results.append(list(pssmdict[res])+list(hhmdict[res]))
    return np.array(results,dtype=float)
