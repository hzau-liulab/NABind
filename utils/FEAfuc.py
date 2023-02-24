from .PDBfuc import MPDB
import numpy as np
import os

class RSA(MPDB):
    
    asadict={'A':400,'G':400,'C':350,'U':350,'T':350,'DA':400,'DG':400,'DC':350,'DT':350,'DU':350,
            'ALA':106,'ARG':248,'ASN':157,'ASP':163,'CYS':135,'GLN':198,'GLU':194,'GLY':84,'HIS':184,'ILE':169,
            'LEU':164,'LYS':205,'MET':188,'PHE':197,'PRO':136,'SER':130,'THR':142,'TRP':227,'TYR':222,'VAL':142,
            }
    
    def __init__(self,asafile=None,rsafile=None,pdbfile=None,prob=1.5,**kwargs):
        self.asafile=asafile
        self.rsafile=rsafile
        self.pdbfile=pdbfile
        self.naccessexe=kwargs['naccess']
        self.out_dir=kwargs['outdir']
        if self.asafile==None:
            self.asafile=os.path.join(os.path.dirname(os.path.realpath(self.pdbfile)),os.path.basename(self.pdbfile).split('.')[0]+'.asa')
            if not os.path.exists(self.asafile):
                self.naccess(prob=prob)
        
        super(RSA,self).__init__(pdbfile)
        
        self.res_atom_asa=dict()
        
    def naccess(self,prob=1.5):
        """
        perform naccess algorithm
        """
        old_dir=os.getcwd()
        os.chdir(self.out_dir)
        os.system(self.naccessexe+' -p '+str(prob)+' -h '+self.pdbfile)
        self.asafile=os.path.join(os.path.dirname(os.path.realpath(self.pdbfile)),os.path.basename(self.pdbfile).split('.')[0]+'.asa')
        os.chdir(old_dir)
    
    def residue(self,relative=True):
        """
        residue level asa
        relative: to calculate relative asa
        return dict => {res1:asa,res2:asa,......}
        """
        if not self.res_atom_asa:
            self.atom()
        
        res_asa={}
        for res in self.res_atom_asa:
            res_asa[res]=np.sum(np.vstack(tuple(np.array(x[1]).reshape(1,-1) for x in self.res_atom_asa[res].items())),axis=0)
        
        if relative:
            for key in res_asa:
                res_asa[key]=res_asa[key]/self.asadict[self.het_to_atom[self.xulie[self.res_index[key]][0]]]
        return res_asa
    
    def res_array(self,relative=True):
        """
        return ndarray float
        """
        res_asa=self.residue(relative=relative)
        return np.array([res_asa[x[2]] for x in self.xulie],dtype=float)
    
    def atom(self):
        """
        atom level asa, asafile needed
        return dict => {res1:{atomtype1:asa,atomtype2:,asa},.......}
        """
        res_atom_asa=dict()
        with open(self.asafile,'r') as f:
            for line in f.readlines():
                res_atom_asa.setdefault(line[21:22]+'_'+line[22:28].strip(),{})
                res_atom_asa[line[21:22]+'_'+line[22:28].strip()][line[12:17].strip()]=float(line[54:62].strip())
        
        self.res_atom_asa=self._check(res_atom_asa,mode='atom')
        return self.res_atom_asa
    
    def atom_array(self):
        """
        return: ndarray (number of atoms*1)
        """
        if not self.res_atom_asa:
            self.atom()
        return np.array(list(map(lambda x:self.res_atom_asa[x[0]][x[1]],self.res_atom))).reshape(-1,1)
    
    def _check(self,dictin,mode='atom'):
        """
        to check if the residues(atoms) in asafile is equal to the pdbfile
        if not equal, print the missing info and add value -1 to the missings
        mode => atom or residue
        return dict
        """
        for res in self.coord:
            if res not in dictin:
                if mode=='residue':
                    # print('missing residue {}'.format(res))
                    dictin[res]=-1.
                elif mode=='atom':
                    # print('missing all atoms in residue {}'.format(res))
                    dictin[res]=dict.fromkeys(self.coord[res].keys(),-1.)
            else:
                if mode=='atom':
                    for atom in self.coord[res]:
                        if atom not in dictin[res]:
                            # print('missing atom {} in residue {}'.format(atom,res))
                            dictin[res][atom]=-1.
        return dictin

