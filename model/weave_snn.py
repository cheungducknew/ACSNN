import torch
import torch.nn as nn
from dgllife.model import WeavePredictor


class Weave_SNN(nn.Module):

    def __init__(self, args):
        super(Weave_SNN, self).__init__()

        # Weave parameters
        node_feat_size = 32  # Same as ACGCN_MMP and AttentiveFP_MMP
        edge_feat_size = 6   # From model_utils.py, bond_features returns 6 features
        num_gnn_layers = 2   # Default value from WeavePredictor
        gnn_hidden_feats = 128  # Same as ACGCN_MMP's hidden_feats[0]
        graph_feats = 128    # Same as ACGCN_MMP's hidden_feats[0]
        gaussian_expand = False  # Setting to False for consistency
        out_size = 128*2     # Same as ACGCN_MMP and AttentiveFP_MMP

        # Create Weave predictor for both smiles1 and smiles2
        self.weave = WeavePredictor(
            node_in_feats=node_feat_size,
            edge_in_feats=edge_feat_size,
            num_gnn_layers=num_gnn_layers,
            gnn_hidden_feats=gnn_hidden_feats,
            graph_feats=graph_feats,
            gaussian_expand=gaussian_expand,
            n_tasks=out_size
        )

        # Same MLP layers as ACGCN_SNN
        self.fc_layer2 = nn.Sequential(nn.Linear(128*2, 128*4),
                                       nn.BatchNorm1d(128*4),
                                       nn.ReLU(inplace=True),
                                       nn.Dropout(args['DROP_OUT']))

        self.out_layer = nn.Linear(128*4, 1)

    def forward(self, batch_smiles1, batch_smiles2):
        # Get node features from batch_smiles1 and batch_smiles2
        node_feats1 = batch_smiles1.ndata['x'].float()
        edge_feats1 = batch_smiles1.edata['y'].float() if 'y' in batch_smiles1.edata else None
        
        node_feats2 = batch_smiles2.ndata['x'].float()
        edge_feats2 = batch_smiles2.edata['y'].float() if 'y' in batch_smiles2.edata else None
        
        # Encode both smiles using Weave
        smiles1 = self.weave(batch_smiles1, node_feats1, edge_feats1)
        smiles2 = self.weave(batch_smiles2, node_feats2, edge_feats2)
        
        # Use difference instead of concatenation (SNN architecture)
        diff = smiles1 - smiles2

        out = self.fc_layer2(diff)
        out = self.out_layer(out)
        # regression: do NOT apply sigmoid
        out = out.squeeze(-1)
        
        return out