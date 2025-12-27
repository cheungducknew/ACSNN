import torch
import torch.nn as nn
from dgllife.model import AttentiveFPPredictor


class AttentiveFP_MMP(nn.Module):

    def __init__(self, args):
        super(AttentiveFP_MMP, self).__init__()

        # AttentiveFP parameters
        node_feat_size = 32  # Same as ACGCN_MMP
        edge_feat_size = 6   # From model_utils.py, bond_features returns 6 features
        num_layers = 2       # Default value from dgllife
        num_timesteps = 2    # Default value from dgllife
        graph_feat_size = 128  # Same as ACGCN_MMP's hidden_feats[0]
        out_size = 128*2     # Same as ACGCN_MMP
        dropout = args['DROP_OUT']

        # Create AttentiveFP predictor for both smiles1 and smiles2
        self.attentivefp = AttentiveFPPredictor(
            node_feat_size=node_feat_size,
            edge_feat_size=edge_feat_size,
            num_layers=num_layers,
            num_timesteps=num_timesteps,
            graph_feat_size=graph_feat_size,
            n_tasks=out_size,
            dropout=dropout
        )

        # Same MLP layers as ACGCN_MMP
        self.fc_layer2 = nn.Sequential(nn.Linear(128*4, 128*8),
                                       nn.BatchNorm1d(128*8),
                                       nn.ReLU(inplace=True),
                                       nn.Dropout(args['DROP_OUT']))

        self.out_layer = nn.Linear(128*8, 1)

    def forward(self, batch_smiles1, batch_smiles2):
        # Get node features from batch_smiles1 and batch_smiles2
        node_feats1 = batch_smiles1.ndata['x'].float()
        edge_feats1 = batch_smiles1.edata['y'].float() if 'y' in batch_smiles1.edata else None
        
        node_feats2 = batch_smiles2.ndata['x'].float()
        edge_feats2 = batch_smiles2.edata['y'].float() if 'y' in batch_smiles2.edata else None
        
        # Encode both smiles using AttentiveFP
        smiles1 = self.attentivefp(batch_smiles1, node_feats1, edge_feats1)
        smiles2 = self.attentivefp(batch_smiles2, node_feats2, edge_feats2)
        
        # Same concatenation and MLP layers as ACGCN_MMP
        out = torch.cat((smiles1, smiles2), axis=1)
        out = self.fc_layer2(out)
        out = self.out_layer(out)
        # regression: do NOT apply sigmoid
        out = out.squeeze(-1)
        
        return out