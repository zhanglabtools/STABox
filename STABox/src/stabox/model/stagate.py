from scanpy import AnnData
import torch
from ._mixin import BaseModelMixin
from torch import nn
import torch.nn.functional as F
from typing import List
from tqdm import tqdm
from ._utils import STAGateModule
import numpy as np
import scanpy as sc
import scipy.sparse as sp
import os


class STAGATE(BaseModelMixin):
    SUPPORTED_TASKS = ["tissue_structure_annotation", "spatial_embedding", "enhanced_gene_expression",
                       "SVG_identification"]
    METHOD_NAME = "STAGATE"
    """
    STAGATE for identifying spatial domain.
    model_dir: The directory to save the model.
    in_features: The number of input features.
    hidden_dims: The list of hidden dimensions.
    n_models: The number of models to train.

    """
    def __init__(self,
                 model_dir: str,
                 in_features: int,
                 hidden_dims: List[int],
                 n_models: int = 5,
                 **kwargs):
        super().__init__(model_dir, in_features, hidden_dims, **kwargs)
        self._check_validity()
        self.n_models = n_models
        self.train_status = []
        self.model = None
        self.model_name = None
    
    def _check_validity(self):
        assert set(self.SUPPORTED_TASKS).issubset(set(BaseModelMixin.SUPPORTED_TASKS)) and len(self.SUPPORTED_TASKS) > 0

    def train(self, adata, hidden_dims=[512, 30], n_epochs=1000, lr=0.001, key_added='STAGATE',
                  gradient_clipping=5., weight_decay=0.0001, verbose=True,
                  random_seed=0, save_loss=False, save_reconstrction=False,
                  device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),**kwargs):
        seed = random_seed
        import random
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

        adata.X = sp.csr_matrix(adata.X)

        if 'highly_variable' in adata.var.columns:
            adata_Vars = adata[:, adata.var['highly_variable']]
        else:
            adata_Vars = adata

        if verbose:
            print('Size of Input: ', adata_Vars.shape)
        if 'Spatial_Net' not in adata.uns.keys():
            raise ValueError("Spatial_Net is not existed! Run Cal_Spatial_Net first!")
        
        data = self.prepare_data(adata_Vars)
        # print("self.in_features", self.in_features)
        # print("self.hidden_dims", self.hidden_dims)
        # print()
        model = STAGateModule(self.in_features, self.hidden_dims).to(device)
        data = data.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        loss_list = []
        for epoch in tqdm(range(1, n_epochs + 1)):
            model.train()
            optimizer.zero_grad()
            z, out = model(data.x, data.edge_index)
            loss = F.mse_loss(data.x, out)
            loss_list.append(loss.item())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
            optimizer.step()

        model.eval()
        z, out = model(data.x, data.edge_index)

        STAGATE_rep = z.to('cpu').detach().numpy()
        adata.obsm[key_added] = STAGATE_rep
        self.model = model

        if save_loss:
            adata.uns['STAGATE_loss'] = loss
        if save_reconstrction:
            ReX = out.to('cpu').detach().numpy()
            ReX[ReX < 0] = 0
            adata.layers['STAGATE_ReX'] = ReX
        return adata
    
    def save(self, path, **kwargs):
        self.model_name='STAGATE'
        model_path = os.path.join(path, self.model_name + '_model.pth')
        torch.save(self.model.state_dict(), model_path)

    def load(self, path, **kwargs):
        self.model_name='STAGATE'
        model_path = os.path.join(path, self.model_name + '_model.pth')
        model = self.model.load_state_dict(torch.load(model_path))
        return model