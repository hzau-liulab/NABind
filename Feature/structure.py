import os
import numpy as np
import networkx as nx
import scipy.stats as stats
from collections import Counter
from GraphRicciCurvature.OllivierRicci import OllivierRicci
from GraphRicciCurvature.FormanRicci import FormanRicci
from utils.PDBfuc import MPDB

class MathMorphologyPocket(object):
    def __init__(self,pdbfile=None,ghecomresfile=None,ghecomexe=None):
        self.ghecomres=ghecomresfile
        self.ghecomexe=ghecomexe
        self.ghecom(pdbfile)

    def ghecom(self,pdbfile=None):
        """
        excute ghecom program
        """
        if pdbfile is not None:
            resout=''.join([pdbfile,'.txt'])
            os.system(self.ghecomexe+' -M M -atmhet B -hetpep2atm F -ipdb '+pdbfile+' -ores '+resout)
            self.ghecomres=resout
        
    def descriptor(self,ghecomresfile=None):
        """
        return: strarray col.1=>res col.2=>shellAcc col.3=>Rinacc col.4=>pocketness
        """
        ghecomres=ghecomresfile if ghecomresfile is not None else self.ghecomres
        out=np.loadtxt(ghecomres,skiprows=43,usecols=(0,3,4,7),dtype=str)
        return out
    
    def res_array(self,ghecomresfile=None):
        return self.descriptor(ghecomresfile)

class Graph(object):
    def __init__(self,graph):
        """
        graph: edges (list/file)
        """
        if isinstance(graph,str):
            with open(graph,'r') as f:
                edges=list(map(lambda x:x.strip().split(),f.readlines()))
        elif isinstance(graph,(list,np.ndarray)):
            edges=list(graph)
        else:
            raise ValueError('check edges in')
        
        self.g=nx.Graph()
        node_unique=set(sum(edges,[]))
        nodedict={j:i for i,j in enumerate(node_unique)}
        self.index_node={j:i for i,j in nodedict.items()}
        edges=list(map(lambda x:[nodedict[y] for y in x],edges))
        self.g.add_edges_from(edges)

class RicciCurvature(Graph):
    def __init__(self,graph):
        """
        graph: edges (list/file)
        """
        super(RicciCurvature,self).__init__(graph)
        
        self.ORC=dict()
        self.FRC=dict()
        
    def ollivier(self,mode='sum'):
        """
        mode: str (sum/mean)
        return: dict
        """
        orc=OllivierRicci(self.g)
        orc.compute_ricci_curvature()
        g=orc.G
        nodes=list(g.nodes)
        aggregate=np.sum if mode=='sum' else np.mean
        for n in nodes:
            curvature=list(map(lambda x:g[n][x]['ricciCurvature'],g[n]))
            curvature=aggregate(curvature)
            self.ORC[self.index_node[n]]=curvature
        return self.ORC
    
    def forman(self,mode='sum'):
        """
        mode: str (sum/mean)
        return: dict
        """
        frc=FormanRicci(self.g)
        frc.compute_ricci_curvature()
        g=frc.G
        nodes=list(g.nodes)
        aggregate=np.sum if mode=='sum' else np.mean
        for n in nodes:
            curvature=list(map(lambda x:g[n][x]['formanCurvature'],g[n]))
            curvature=aggregate(curvature)
            self.FRC[self.index_node[n]]=curvature
        return self.FRC

class GaussianNet(object):
    """
    Gaussian Network Model (GNM)
    """
    def __init__(self,):
        pass


class MultifractalDim(Graph):
    def __init__(self,graph):
        """
        graph: edges (list/file)
        """
        super(MultifractalDim,self).__init__(graph)
        
        self.MFD=dict()
    
    def slope(self,weight=None):
        self.MFD=dict(map(lambda x:(self.index_node[x],self._slope(x,weight=weight)),self.g.nodes))
        return self.MFD
    
    def _slope(self,node,weight=None):
        m=nx.single_source_shortest_path_length if weight is None else nx.single_source_dijkstra_path_length
        spl=m(self.g,node)
        grow=[y for x,y in spl.items() if x!=node]
        grow.sort()
        l_ml=[[x,y] for x,y in Counter(grow).items()]
        if len(l_ml)<2:
            slope=0
        else:
            l=np.log([x for x,y in l_ml])
            ml=np.log(np.cumsum([y for x,y in l_ml]))
            slope,intercept,r_value,p_value,std_err=stats.linregress(l,ml)
        return slope

class MultiDistance(MPDB):
    """
    Calculate multi-distance based feature
    Reference: Fast protein structure comparison through effective representation learning with contrastive graph neural networks
    """
    def __init__(self,pdbfile,M=3,**kwargs):
        """
        pdbfile: file
        M: int, default==3
        """
        super(MultiDistance,self).__init__(pdbfile)
        
        self.M=M
        self.res_coord=self.get_res_coord_dict(mode='average')
        
        self.refpoints=sum(list(map(self._get_reference_points,range(self.M))),[])
        
    def _get_reference_points(self,m):
        """
        m: int
        split into 2**m
        """
        def _average_coords(resindex):
            ress=[self.xulie[x][-1] for x in resindex]
            coords=[self.res_coord[x] for x in ress]
            return np.mean(coords,axis=0)
        
        N=list(range(len(self.xulie)))
        Nsegment=np.array_split(N,2**m)
        refpoints=list(map(_average_coords,Nsegment))
        return refpoints
    
    def multidist(self,normalize=True):
        """
        return: ndarray (length of res*feat dim (2**M-1))
        """
        res_coordlist=[self.res_coord[x[-1]] for x in self.xulie]
        
        def resdist(refpoint):
            return list(map(lambda x:self.cal_distance(x,refpoint),res_coordlist))
        out=np.column_stack(tuple(map(resdist,self.refpoints)))
        if normalize:
            out=(out-np.mean(out,axis=0))/np.std(out,axis=0)
        return out
    
    def res_array(self,normalize=True):
        return self.multidist(normalize=normalize)

class UltrafastShape(MPDB):
    """
    USR ultrafast shape recognition
    Reference: 
    """
    def __init__(self,pdbfile,mode='average',**kwargs):
        """
        pdbfile: file
        mode: average/atom name
        """
        super(UltrafastShape,self).__init__(pdbfile)
        
        self.mode=mode
        
    def _USRfeat(self,distmap,normalize=True):
        """
        return: ndarray (length of res*3)
        """
        # distmap=self.distance_map(mode=self.mode)
        meandist=np.mean(distmap,axis=1)
        maxdist_resdict=dict(zip(range(len(distmap)),np.argmax(distmap,axis=1)))
        USR=list(map(lambda x:[meandist[x],
                                   meandist[maxdist_resdict[x]],
                                   meandist[maxdist_resdict[maxdist_resdict[x]]]],
                         range(len(distmap))))
        USR=np.array(USR,dtype=float)
        if normalize:
            USR=(USR-np.mean(USR,axis=0))/np.std(USR,axis=0)
        return USR
    
    def atom_array(self,normalize=True):
        """
        return: ndarray (number of atoms*3)
        """
        distmap=self._atom_distance().values
        return self._USRfeat(distmap,normalize=normalize)
    
    def res_array(self,normalize=True):
        """
        return: ndarray (number of res*3)
        """
        distmap=self.distance_map(mode=self.mode)
        return self._USRfeat(distmap,normalize=normalize)


class CircularVariance(MPDB):
    """
    Reference: A new method for mapping macromolecular topography
    """
    def __init__(self,pdbfile,r,**kwargs):
        """
        pdbfile: file
        r: int/float/list
        """
        super(CircularVariance,self).__init__(pdbfile)
        
        if isinstance(r,(int,float)):
            self.r=[r]
        elif isinstance(r,(list,tuple)):
            self.r=r
        else:
            raise ValueError('not accepted r')
        
        self.atom_dist_df=self._atom_distance()
        self.atom_atom_vector=self._atom_atom_vector()
        
        
    def _atom_atom_vector(self):
        """
        return: dict {atom:{atom:vector},......}
        """
        atoms=self.atom_dist_df.index.values
        atom_atom_vector=dict()
        for atom1 in atoms:
            atom_atom_vector[atom1]=dict(map(lambda x:(x,
                                                       np.array(self.coord['_'.join(atom1.split('_')[:-1])][atom1.split('_')[-1]],dtype=float)-np.array(self.coord['_'.join(x.split('_')[:-1])][x.split('_')[-1]],dtype=float)),
                                             atoms))
        return atom_atom_vector
    
    def _CVatom(self,r):
        """
        r: float
        return: dict {atom:cv,......}
        """
        def cvcalculate(atom,relative_atoms):
            vectors=np.array([self.atom_atom_vector[atom][x] for x in relative_atoms if x!=atom],dtype=float)
            norms=np.linalg.norm(vectors,axis=1).reshape(-1,1)
            normvectors=vectors/norms
            return 1-np.linalg.norm(np.sum(normvectors,axis=0))/len(vectors)
        
        atoms=self.atom_dist_df.index.values
        cvatomdict=dict()
        for atom in atoms:
            dists=self.atom_dist_df.loc[atom].values
            relative_atoms=atoms[np.where(dists<r)]
            cvatomdict[atom]=cvcalculate(atom,relative_atoms)
        return cvatomdict
    
    def _CVatom_array(self,r):
        """
        r: float
        return: ndarray (length of atoms*1)
        """
        cvatomdict=self._CVatom(r)
        return np.array(list(map(lambda x:cvatomdict['_'.join(x)],self.res_atom))).reshape(-1,1)
    
    def _CVres(self,r):
        """
        r: float
        return: ndarray (length of res*1)
        """
        cvatomdict=self._CVatom(r)
        
        def cvcalculate(res):
            res_atoms=filter(lambda x:'_'.join(x.split('_')[:-1])==res,cvatomdict)
            return np.mean([cvatomdict[x] for x in res_atoms])
        
        cvres=np.array([cvcalculate(x) for _,_,x in self.xulie],dtype=float).reshape(-1,1)
        return cvres
    
    def CVatom(self):
        """
        return: ndarray (length of atoms*length of r)
        """
        return np.hstack(list(map(self._CVatom_array,self.r)))
    
    def CVres(self):
        """
        return: ndarray (length of res*length of r)
        """
        return np.hstack(list(map(self._CVres,self.r)))
    
    def atom_array(self):
        return self.CVatom()
    
    def res_array(self):
        return self.CVres()

class Topo(MPDB):
    
    def __init__(self,pdbfile=None):
        super(Topo,self).__init__(pdbfile)
        self.contactlist=list()
        
    def contact(self,distcut,mode='min'):
        """
        mode: min,average,atom name
        self.contactlist: [(res1,res2),(res9,res2),......]
        """
        contactmap=self.contact_map(distcut=distcut,mode=mode)
        ijindex=np.nonzero(contactmap)
        for i,j in zip(ijindex[0],ijindex[1]):
            self.contactlist.append([self.index_res[i],self.index_res[j]])
        return self.contactlist
