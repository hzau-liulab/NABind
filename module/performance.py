from Evafuc import Eva_by_chain
import numpy as np
import os
from scipy.special import softmax

class PerformanceMetrics:
    def __init__(self,outdir,
                 metrics=['cutoff','recall','precision','tnr',
                          'acc','mcc','auc','prc']):
        self.metrics_list=metrics
        self.metrics={}
        for key in metrics:
            self.metrics[key]=None
        self.count=0
        self.prediction=np.array([])
        self.outdir=outdir+'/prediction/'
        
    def update(self,prediction,targets,tags,pair=False):
        prediction=1/(1+np.exp(-prediction))#sigmoid
        # inputs=np.unique(np.column_stack((tags,targets,prediction)),
        #                  axis=0)
        # prediction=softmax(prediction,axis=1)
        inputs=np.column_stack((tags,targets,prediction))
        self.prediction=inputs
        
        os.makedirs(self.outdir,exist_ok=True)
        np.savetxt(self.outdir+'pre'+str(self.count)+'.txt',
                    inputs,fmt='%s',delimiter='\t')
        self.count+=1
        
        self.eva,self.auc,self.prc=Eva_by_chain(inputs[:,[0,1,2]],pair=pair)
        for key,value in zip(self.metrics_list,self._max_eva(self.eva)+[self.auc,self.prc]):
            self.metrics[key]=float(value)
        return self.metrics
    
    def _max_eva(self,eva):
        mcclist=eva[:,-1].tolist()
        index=mcclist.index(max(mcclist))
        return [index/100]+eva[index,:].tolist()
    
    def write_feature_scores_to_file(self,outfile):
        np.savetxt(outfile,
                   np.column_stack((np.arange(0,1.01,0.01),self.eva)),
                   fmt='%.4f',delimiter='\t',
                   footer=str(self.auc)+'\t'+str(self.prc),
                   comments='')
    
    def write_predictions_to_file(self,outfile):
        np.savetxt(outfile,self.prediction,fmt='%s',delimiter='\t')