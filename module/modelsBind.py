import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.function as fn
from dgl.nn import GraphConv,GATConv
from dgl.nn import EGATConv

from .EGRET import *



class EGATM(nn.Module):
    config={
    'attention_node':False,
    'attention_node_and_edge':True,
    'aggregate_with_node_and_edge':False,
    'aggregate_with_node':True,
    'use_bias':True,
    'in_dim':1346,
    'hidden_dim':1346,
    'edge_dim':2,
    'hidden_edge_dim':2,
    'heads':1,
    'layer':1,
    'concatenate_learned_and_ori':True,
    'multihead_mean':True,
    'fully_connect_layer':2,
    'fully_connect_dims':[2692,1346],
    'dropout':True,
    'dropout_ratio':0.5,
    'ReLU':True,
    }
    
    def __init__(self):
        super().__init__()
        
        if self.config['attention_node']:
            self.egat1=GATConv(in_feats=self.config['in_dim'],
                             out_feats=self.config['hidden_dim'],
                             num_heads=self.config['heads'],
                             bias=self.config['use_bias']
                             )
            if self.config['layer']>1:
                self.egat=[GATConv(in_feats=self.config['in_dim'],
                                 out_feats=self.config['hidden_dim'],
                                 num_heads=self.config['heads'],
                                 bias=self.config['use_bias']
                                 ) for i in range(self.config['layer'])]
        elif self.config['attention_node_and_edge']:
            if self.config['aggregate_with_node']:
                self.egat1=EGATConv(in_node_feats=self.config['in_dim'],
                                    in_edge_feats=self.config['edge_dim'],
                                    out_node_feats=self.config['hidden_dim'],
                                    out_edge_feats=self.config['hidden_edge_dim'],
                                    num_heads=self.config['heads'],
                                    bias=self.config['use_bias'],
                                    )
                if self.config['layer']>1:
                    self.egat=[EGATConv(in_node_feats=self.config['in_dim'],
                                        in_edge_feats=self.config['edge_dim'],
                                        out_node_feats=self.config['hidden_dim'],
                                        out_edge_feats=self.config['hidden_edge_dim'],
                                        num_heads=self.config['heads'],
                                        bias=self.config['use_bias'],
                                        ) for i in range(self.config['layer'])]
            elif self.config['aggregate_with_node_and_edge']:
                self.egat1=EGRETLayer(in_dim=self.config['in_dim'],
                                      out_dim=self.config['hidden_dim'],
                                      edge_dim=self.config['edge_dim'],
                                      use_bias=self.config['use_bias'],
                                      )
                if self.config['layer']>1:
                    self.egat=[EGRETLayer(in_dim=self.config['in_dim'],
                                          out_dim=self.config['hidden_dim'],
                                          edge_dim=self.config['edge_dim'],
                                          use_bias=self.config['use_bias'],
                                          ) for i in range(self.config['layer'])]
            else:
                raise ValueError('model config error')
        else:
            raise ValueError('model config error')
        
        if self.config['concatenate_learned_and_ori']:
            if not self.config['dropout']:
                self.config['dropout_ratio']=0.
            self.fc=nn.Sequential(
                nn.Linear(self.config['fully_connect_dims'][0],self.config['fully_connect_dims'][1]),
                nn.ReLU(inplace=True),
                nn.Dropout(p=self.config['dropout_ratio']),
                nn.Linear(self.config['fully_connect_dims'][1],1),
                # nn.Sigmoid(),
                )
        else:
            if not self.config['dropout']:
                self.config['dropout_ratio']=0.
            self.fc=nn.Sequential(
                nn.Linear(self.config['fully_connect_dims'][0]/2,self.config['fully_connect_dims'][1]/2),
                nn.ReLU(inplace=True),
                nn.Dropout(p=self.config['dropout_ratio']),
                nn.Linear(self.config['fully_connect_dims'][1]/2,1),
                # nn.Sigmoid(),
                )
    
    def multilayer(self,ms,g,h,eh):
        for m in ms:
            if self.config['attention_node_and_edge'] and self.config['aggregate_with_node']:
                h,_=m(g,h,eh)
            else:
                h,_=m(g,h)
        return h,_
    
    def forward(self,h,g):
        eh=g.edata['ex']
        if hasattr(self,'egat'):
            h1,_=self.multilayer(self.egat,g,h,eh)
        else:
            h1,_=self.egat1(g,h,eh)
        if self.config['concatenate_learned_and_ori']:
            if self.config['multihead_mean']:
                x=torch.cat((h,torch.mean(h1,dim=1),),dim=1)
        else:
            if self.config['multihead_mean']:
                x=torch.mean(h1,dim=1)
        x=self.fc(x)
        return x
