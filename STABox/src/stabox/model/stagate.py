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
import time
import random
from torch_geometric.loader import ClusterData, ClusterLoader

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
    
    def train_subgraph(self, adata, hidden_dims=[512, 30], n_epochs=1000, lr=0.001, key_added='STAGATE',
                  gradient_clipping=5., weight_decay=0.0001, verbose=True,
                  random_seed=0, save_loss=False, save_reconstrction=False,
                  device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),num_parts=128,batch_size=16,num_workers=0,**kwargs):
        # Set random seeds for reproducibility
        seed = random_seed
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Ensure sparse matrix format
        adata.X = sp.csr_matrix(adata.X)

        # Filter data for highly variable genes if available
        if 'highly_variable' in adata.var.columns:
            adata_Vars = adata[:, adata.var['highly_variable']]
        else:
            adata_Vars = adata

        if verbose:
            print('Size of Input: ', adata_Vars.shape)

        # Ensure that Spatial_Net is calculated before running
        if 'Spatial_Net' not in adata.uns.keys():
            raise ValueError("Spatial_Net does not exist! Run Cal_Spatial_Net first!")

        # Prepare the data for training
        data = self.prepare_data(adata_Vars)

        # Initialize the model
        model = STAGateModule(self.in_features, hidden_dims).to(device)

        # Reset CUDA peak memory stats and log initial memory usage
        torch.cuda.reset_peak_memory_stats()
        print(f"Initial Peak Memory: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")

        # Print the number of model parameters
        print('Number of parameters: ', sum(p.numel() for p in model.parameters()))

        # Create clusters (subgraphs)
        cluster_data = ClusterData(data, num_parts)
        train_loader = ClusterLoader(cluster_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

        # Ensure the total number of nodes in the subgraphs equals the number of nodes in the full graph
        total_num_nodes, total_num_edges = 0, 0
        for step, sub_data in enumerate(train_loader):
            print(f"Step {step+1}")
            print(f"Subgraph Node Count: {sub_data.num_nodes}, Subgraph Edge Count: {sub_data.num_edges}")
            total_num_nodes += sub_data.num_nodes
            total_num_edges += sub_data.num_edges

        print(f"Total Nodes: {total_num_nodes}, Full Graph Nodes: {data.num_nodes}, Total Edges: {total_num_edges}, Full Graph Edges: {data.num_edges}")
        assert total_num_nodes == data.num_nodes, "The number of nodes in subgraphs is not equal to the number of nodes in the full graph!"

        # Set up the optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        start_time=time.time()
        for epoch in tqdm(range(1, n_epochs + 1)):
            model.train()
            for step,sub_data in enumerate(train_loader):
                sub_data = sub_data.to(device)
                optimizer.zero_grad()
                z, out = model(sub_data.x, sub_data.edge_index)
                # Calculate loss and backpropagate
                loss = F.mse_loss(sub_data.x, out)
                loss.backward()
                # Apply gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                # Update model parameters
                optimizer.step()

        # Set the model to evaluation mode and move it to CPU for final inference
        model.eval()
        model = model.to('cpu')

        # Perform inference on the full graph (CPU)
        with torch.no_grad():
            data = data.to('cpu')
            z, out = model(data.x, data.edge_index)

        # Print the final peak memory usage
        print(f"Final Peak Memory: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")

        # Save the model outputs and other results to adata
        STAGATE_rep = z.to('cpu').detach().numpy()
        adata.obsm[key_added] = STAGATE_rep
        self.model = model

        # Save loss if requested
        if save_loss:
            adata.uns['STAGATE_loss'] = loss.item()

        # Save reconstruction if requested
        if save_reconstrction:
            ReX = out.to('cpu').detach().numpy()
            ReX[ReX < 0] = 0  # Ensure non-negative values for reconstruction
            adata.layers['STAGATE_ReX'] = ReX

        return adata