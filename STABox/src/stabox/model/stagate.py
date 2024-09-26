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

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
# import STAGATE_pyG
from torch_geometric.loader import DataLoader
from torch.utils.data.distributed import DistributedSampler
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
            if epoch % 100 == 0:
                print(f"Epoch {epoch}/{n_epochs}, Loss: {loss.item():.4f},Batch:{len(data)}")
            loss_list.append(loss.item())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
            optimizer.step()

        model.eval()
        z, out = model(data.x, data.edge_index)
        print('test loss',F.mse_loss(data.x, out))
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
    
    def setup(self, rank, world_size):
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group("nccl", rank=rank, world_size=world_size)

    def cleanup(self):
        dist.destroy_process_group()
        
    def _train_ddp_process(self, rank, world_size, data_list, hidden_dims, n_epochs, lr, key_added,
                           gradient_clipping, weight_decay, verbose,
                           random_seed):
        self.setup(rank, world_size)
        device = torch.device(f'cuda:{rank}')
        print(f"Training on rank {rank}, device {device}")
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)


        data_list = [d.to(device) for d in data_list]
        
        # Create DataLoader with DistributedSampler
        sampler = DistributedSampler(data_list, num_replicas=world_size, rank=rank, shuffle=True)
        loader = DataLoader(data_list, batch_size=64, sampler=sampler)
        # Create model and wrap it with DDP
        model = STAGATE_pyG.STAGATE(hidden_dims=[data_list[0].x.shape[1]] + hidden_dims).to(device)
        model = DDP(model, device_ids=[rank],find_unused_parameters=True)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        for epoch in range(1, n_epochs + 1):
            model.train()
            sampler.set_epoch(epoch)
            for batch in loader:
                optimizer.zero_grad()
                _, out = model(batch.x, batch.edge_index)
                loss = F.mse_loss(batch.x, out)
                loss.backward()
                if rank == 0 and epoch % 100 == 0:
                    print(f"Epoch {epoch}/{n_epochs}, Loss: {loss.item():.4f},Batch:{len(batch)}")
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                optimizer.step()

        # Save the model state dict only in the main process (rank 0)
        if rank == 0:
            model_state_dict = model.module.state_dict()
            torch.save(model_state_dict, 'trained_model.pth')
        self.cleanup()
        return None

    def train_ddp(self, adata, hidden_dims=[512, 30], n_epochs=1000, lr=0.0001, key_added='STAGATE',
                  gradient_clipping=1., weight_decay=0.0005, verbose=True,
                  random_seed=0, **kwargs):
        world_size = torch.cuda.device_count()
        print(f"Training on {world_size} GPUs")
        
        # Prepare data
        data_list, data = self.prepare_batch_data(
            adata, 
            kwargs.get('num_batch_x', 2), 
            kwargs.get('num_batch_y', 3), 
            kwargs.get('rad_cutoff', 50), 
            verbose=False
        )
        
        mp.spawn(self._train_ddp_process, 
                 args=(world_size, data_list, hidden_dims, n_epochs, lr, key_added,
                       gradient_clipping, weight_decay, verbose,
                       random_seed),
                 nprocs=world_size,
                 join=True)
        # After training, perform inference on a single GPU
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model = STAGATE_pyG.STAGATE(hidden_dims=[data.x.shape[1]] + hidden_dims).to(device)
        
        # Load the trained model state dict
        model.load_state_dict(torch.load('trained_model.pth'))
        
        model.eval()
        data = data.to(device)
        with torch.no_grad():
            z, _ = model(data.x, data.edge_index)
            print('test loss',F.mse_loss(data.x, _))

        STAGATE_rep = z.cpu().numpy()
        adata.obsm[key_added] = STAGATE_rep

        return adata
    
    def prepare_batch_data(self, adata, num_batch_x, num_batch_y, rad_cutoff, verbose):
        adata.obs['X'] = np.asarray(adata.obsm['spatial'][:, 0])
        adata.obs['Y'] = np.asarray(adata.obsm['spatial'][:, 1])
        Batch_list = STAGATE_pyG.Batch_Data(
            adata, 
            num_batch_x=num_batch_x, 
            num_batch_y=num_batch_y,
            spatial_key=['X', 'Y'], 
            plot_Stats=True
        )

        for temp_adata in Batch_list:
            STAGATE_pyG.Cal_Spatial_Net(temp_adata, rad_cutoff=rad_cutoff, verbose=verbose)

        data_list = [STAGATE_pyG.Transfer_pytorch_Data(adata) for adata in Batch_list]

        STAGATE_pyG.Cal_Spatial_Net(adata, rad_cutoff=50, verbose=verbose)
        data = STAGATE_pyG.Transfer_pytorch_Data(adata)
        return data_list, data

    def train_batch(self, adata, hidden_dims=[512, 30], n_epochs=1000, lr=0.001, key_added='STAGATE',
                    gradient_clipping=5., weight_decay=0.0001, verbose=True,
                    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
                    num_batch_x=2, num_batch_y=3, rad_cutoff=50, **kwargs):

        data_list, data = self.prepare_batch_data(adata, num_batch_x, num_batch_y, rad_cutoff, verbose)

        data_list = [data.to(device) for data in data_list]
        data.to(device)

        loader = DataLoader(data_list, batch_size=8, shuffle=True)

        model = STAGATE_pyG.STAGATE(hidden_dims=[data_list[0].x.shape[1]] + hidden_dims).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        for epoch in tqdm(range(1, n_epochs+1)):
            for batch in loader:
                model.train()
                optimizer.zero_grad()
                z, out = model(batch.x, batch.edge_index)
                loss = F.mse_loss(batch.x, out)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                optimizer.step()
            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}/{n_epochs}, Loss: {loss.item():.4f}")

        model.eval()
        z, out = model(data.x, data.edge_index)

        STAGATE_rep = z.to('cpu').detach().numpy()
        adata.obsm[key_added] = STAGATE_rep
        return adata