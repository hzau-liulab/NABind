import torch
import esm

import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from typing import List, Tuple, Optional, Dict, NamedTuple, Union, Callable
import string
from scipy.spatial.distance import cdist
from Bio import SeqIO

torch.set_grad_enabled(False)

class esmmsa1b(object):
    deletekeys = dict.fromkeys(string.ascii_lowercase)
    deletekeys["."] = None
    deletekeys["*"] = None
    translation = str.maketrans(deletekeys)
    
    def __init__(self,msa):#str->file
        self.msa=self.read_msa(msa)
    
    def remove_insertions(self,sequence: str) -> str:
        return sequence.translate(self.translation)

    def read_msa(self,filename: str) -> List[Tuple[str, str]]:
        return [(record.description, self.remove_insertions(str(record.seq))) for record in SeqIO.parse(filename, "fasta")]

    def greedy_select(self,msa: List[Tuple[str, str]], num_seqs: int, mode: str = "max") -> List[Tuple[str, str]]:
        assert mode in ("max", "min")
        
        if len(msa) <= num_seqs:
            return msa
        
        array = np.array([list(seq) for _, seq in msa], dtype=np.bytes_).view(np.uint8)

        optfunc = np.argmax if mode == "max" else np.argmin
        all_indices = np.arange(len(msa))
        indices = [0]
        pairwise_distances = np.zeros((0, len(msa)))
        for _ in range(num_seqs - 1):
            dist = cdist(array[indices[-1:]], array, "hamming")
            pairwise_distances = np.concatenate([pairwise_distances, dist])
            shifted_distance = np.delete(pairwise_distances, indices, axis=1).mean(0)
            shifted_index = optfunc(shifted_distance)
            index = np.delete(all_indices, indices)[shifted_index]
            indices.append(index)
        indices = sorted(indices)
        return [msa[idx] for idx in indices]
    
    def get_embedding(self,num_seqs=128):
        msa_transformer, msa_transformer_alphabet = esm.pretrained.esm_msa1b_t12_100M_UR50S()
        msa_transformer = msa_transformer.eval().cuda()
        msa_transformer_batch_converter = msa_transformer_alphabet.get_batch_converter()
        inputs = self.greedy_select(self.msa, num_seqs=num_seqs)
        msa_transformer_batch_labels, msa_transformer_batch_strs, msa_transformer_batch_tokens = msa_transformer_batch_converter([inputs])
        msa_transformer_batch_tokens = msa_transformer_batch_tokens.to(next(msa_transformer.parameters()).device)
        if msa_transformer_batch_tokens.shape[2]>1024:
            index=msa_transformer_batch_tokens.shape[2]-1024
            msa_transformer_batch_tokens1=msa_transformer_batch_tokens[:,:,:1024]
            msa_transformer_batch_tokens2=msa_transformer_batch_tokens[:,:,-1024:]
            results1=msa_transformer(msa_transformer_batch_tokens1, repr_layers=[12], return_contacts=False)['representations'][12]
            results2=msa_transformer(msa_transformer_batch_tokens2, repr_layers=[12], return_contacts=False)['representations'][12]
            results=torch.cat((results1[:,:,:index,:],
                              (results1[:,:,index:,:]+results2[:,:,:-index,:])/2,
                                results2[:,:,-index:,:]),dim=2)
        else:
            results=msa_transformer(msa_transformer_batch_tokens, repr_layers=[12], return_contacts=False)['representations'][12]
        results=torch.mean(results[0],dim=0)[1:,:]
        results=results.cpu().numpy()
        return results

import esm.inverse_folding
from utils.PDBfuc import PDB

class esmif(object):
    def __init__(self,pdb):#str->file
        self.pdb=PDB(pdb)
    
    def GetSeq(self,pdb):
        return ''.join([pdb.three_to_one[pdb.het_to_atom[x[0]]] for x in pdb.xulie])

    def GetCoords(self,pdb):
        coorddict=pdb.coord
        def fuc(res,x):
            if x in coorddict[res]:
                return list(coorddict[res][x])
            else:
                return [float('inf'),float('inf'),float('inf')]
        coords=list()
        for _,_,res in pdb.xulie:
            coords.append([fuc(res,x) for x in ['N','CA','C']])
        return np.array(coords,dtype=np.float32)
    
    def get_embedding(self,):
        model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
        model = model.eval()
        coords=self.GetCoords(self.pdb)
        result=esm.inverse_folding.util.get_encoder_output(model, alphabet, coords)
        result=result.detach().numpy()
        return result

if __name__=='__main__':
    import pickle
    feat=esmif('../demo/6chv_D.pdb').get_embedding()
    print(feat.shape)
    with open('../demo/esmif.pkl','wb') as f:
        pickle.dump(feat,f)