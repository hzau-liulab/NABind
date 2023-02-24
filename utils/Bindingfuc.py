from .PDBfuc import PDB

import itertools
import numpy as np


class Contact(PDB):
    """
    NOTE: tarpdb and ligandpdb should to be single chain for now
    """
    def __init__(self,tarpdb=None,ligandpdb=None,dist_cutoff=4.5):
        self.dist_cutoff=dist_cutoff
        self.tarpdb=PDB(tarpdb)
        self.ligandpdb=PDB(ligandpdb)
        
        self.contactdict=dict()
        
        if all(x!=None for x in (tarpdb,ligandpdb)):
            self._cal_contact()
        
        pass
    
    def _contact(self):
        for tarres,ligandres in itertools.product(self.tarpdb.res_index,self.ligandpdb.res_index):
            for taratom,ligandatom in itertools.product(self.tarpdb.coord[tarres],self.ligandpdb.coord[ligandres]):
                if self._cal_distance(self.tarpdb.coord[tarres][taratom],self.ligandpdb.coord[ligandres][ligandatom])<=self.dist_cutoff:
                    self.contactdict.setdefault((tarres,ligandres),[]).append((taratom,ligandatom))
    
    def _cal_distance(self,vec1,vec2):
        return super(Contact,self).cal_distance(vec1,vec2)
    
    def get_binding_res(self):
        """
        return binding res for tarpdb
        """
        self._cal_contact()
        return list(map(lambda x:x[0],self.contactdict))
    
    def get_contact_res(self):
        """
        return contact res pairs
        col 1 => tar res
        col 2 => ligand res
        NOTE: no ordered
        """
        self._cal_contact()
        return np.array(list(map(lambda x:list(x),self.contactdict)))
    
    def get_binding_res_atoms(self,mode='tar'):
        """
        get binding res atoms for tarpdb (mode=='tar') or for ligandpdb (mode=='ligand')
        """
        # TODO write 
        pass
    
    def get_contact_res_atoms(self):
        """
        return dict => {(res1,res2):[(atom1,atom2),..],....}
        """
        self._cal_contact()
        return self.contactdict
    
    def _cal_contact(self):
        if not self.contactdict:
            self._contact()


class Binding(object):
    def __init__(self,tarpdb,ligandpdb,dist_cutoff):
        """
        tarpdb: single chain pdb for binding res retrive, str(file)
        ligandpdb: single or more chains pdb, str(file)
        dist_cutoff: int/float
        """
        self.tarpdb=tarpdb
        if isinstance(ligandpdb,str):
            with open(ligandpdb,'r') as f:
                flist=f.readlines()
        else:
            flist=ligandpdb
        self.dist_cutoff=dist_cutoff
        
        ligand_chains=set(list(map(lambda x:x[1],PDB(ligandpdb,autofix=False).xulie)))
        self.ligandpdblists=list()
        for chain in ligand_chains:
            self.ligandpdblists.append(list(filter(lambda x:x[21:22]==chain,flist)))
    
    def get_binding_res(self):
        """
        return: list of binding res
        """
        bindingres=list()
        for ligandpdb in self.ligandpdblists:
            contact=Contact(self.tarpdb,ligandpdb,dist_cutoff=self.dist_cutoff)
            bindingres.extend(contact.get_binding_res())
        bindingres=set(bindingres)
        return list(bindingres)
    
    def res_array(self):
        """
        return: strarray col.1=>res col.2=>1/0
        """
        bindingres=self.get_binding_res()
        array=list()
        for res_type,chain,res in PDB(self.tarpdb).xulie:
            if res in bindingres:
                array.append([res,'1'])
            else:
                array.append([res,'0'])
        return np.array(array,dtype=str)
    

class MultiBinding(object):
    def __init__(self,tarpdb,ligandpdb,dist_cutoff):
        """
        tarpdb: multiple chain pdb
        ligandpdb: multiple chain pdb
        dist_cutoff: float
        """
        with open(tarpdb,'r') as f:
            flist=f.readlines()
        tar_chains=set(list(map(lambda x:x[1],PDB(tarpdb).xulie)))
        self.tarpdblists=list()
        for chain in tar_chains:
            self.tarpdblists.append([chain,list(filter(lambda x:x[21:22]==chain,flist))])
        
        self.ligandpdb=ligandpdb
        self.dist_cutoff=dist_cutoff
    
    def get_binding_res(self):
        """
        return: dict {chain:[res,res,.....]}
        """
        bindingdict=dict()
        for chain,pdb in self.tarpdblists:
            binding=Binding(pdb,self.ligandpdb,self.dist_cutoff)
            bindingdict[chain]=binding.get_binding_res()
        return bindingdict
