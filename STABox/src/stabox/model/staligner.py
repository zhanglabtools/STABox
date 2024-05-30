import os

import torch
from ._mixin import BaseModelMixin
from ._utils import GATConv
from torch import nn
import torch.nn.functional as F
from typing import List
import numpy as np
import pandas as pd
from tqdm import tqdm

from ._utils import STAGateModule, create_dictionary_mnn, Transfer_pytorch_Data


import torch
import torch.backends.cudnn as cudnn

cudnn.deterministic = True
cudnn.benchmark = True
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


class STAligner(BaseModelMixin):
    SUPPORTED_TASKS = ["tissue_structure_annotation", "spatial_embedding", "enhanced_gene_expression",
                       "3D_reconstruction"]
    METHOD_NAME = "STAligner"

    def __init__(self,
                 model_dir: str,
                 in_features: int,
                 hidden_dims: List[int],
                 n_models: int = 5,
                 **kwargs):
        super().__init__(model_dir, in_features, hidden_dims, **kwargs)
        self.model = None
        self._check_validity()
        self.n_models = n_models
        self.train_status = []
        self.model = None
        self.model_name = None

    def _check_validity(self):
        assert set(self.SUPPORTED_TASKS).issubset(set(BaseModelMixin.SUPPORTED_TASKS)) and len(self.SUPPORTED_TASKS) > 0

    def train(self, adata, hidden_dims=[512, 30], n_epochs=1000, lr=0.001, key_added='STAligner',
                    gradient_clipping=5., weight_decay=0.0001, margin=1.0, verbose=False,
                    random_seed=666, iter_comb=None, knn_neigh=100,
                    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):

        seed = random_seed
        import random
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

        section_ids = np.array(adata.obs['batch_name'].unique())
        edgeList = adata.uns['edgeList']

        if type(adata.X) == np.ndarray:
            data = Data(edge_index=torch.LongTensor(np.array([edgeList[0], edgeList[1]])),
                        prune_edge_index=torch.LongTensor(np.array([])),
                        x=torch.FloatTensor(adata.X))
        else:
            data = Data(edge_index=torch.LongTensor(np.array([edgeList[0], edgeList[1]])),
                        prune_edge_index=torch.LongTensor(np.array([])),
                        x=torch.FloatTensor(adata.X.todense()))
        data = data.to(device)

        model = STAGateModule(data.x.shape[1], self.hidden_dims).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        if verbose:
            print(model)

        print('Pretrain with STAGATE...')
        for epoch in tqdm(range(0, 500)):
            model.train()
            optimizer.zero_grad()
            z, out = model(data.x, data.edge_index)

            loss = F.mse_loss(data.x, out)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()

        with torch.no_grad():
            z, _ = model(data.x, data.edge_index)
        adata.obsm['STAGATE'] = z.cpu().detach().numpy()

        print('Train with STAligner...')
        for epoch in tqdm(range(500, n_epochs)):
            if epoch % 100 == 0 or epoch == 500:
                if verbose:
                    print('Update spot triplets at epoch ' + str(epoch))
                adata.obsm['STAGATE'] = z.cpu().detach().numpy()

                mnn_dict = create_dictionary_mnn(adata, use_rep='STAGATE', batch_name='batch_name', k=knn_neigh,
                                                 iter_comb=iter_comb, verbose=0)

                anchor_ind = []
                positive_ind = []
                negative_ind = []
                for batch_pair in mnn_dict.keys():  # pairwise compare for multiple batches
                    batchname_list = adata.obs['batch_name'][mnn_dict[batch_pair].keys()]

                    cellname_by_batch_dict = dict()
                    for batch_id in range(len(section_ids)):
                        cellname_by_batch_dict[section_ids[batch_id]] = adata.obs_names[
                            adata.obs['batch_name'] == section_ids[batch_id]].values

                    anchor_list = []
                    positive_list = []
                    negative_list = []
                    for anchor in mnn_dict[batch_pair].keys():
                        anchor_list.append(anchor)
                        positive_spot = mnn_dict[batch_pair][anchor][0]  # select the first positive spot
                        positive_list.append(positive_spot)
                        section_size = len(cellname_by_batch_dict[batchname_list[anchor]])
                        negative_list.append(
                            cellname_by_batch_dict[batchname_list[anchor]][np.random.randint(section_size)])

                    batch_as_dict = dict(zip(list(adata.obs_names), range(0, adata.shape[0])))
                    anchor_ind = np.append(anchor_ind, list(map(lambda _: batch_as_dict[_], anchor_list)))
                    positive_ind = np.append(positive_ind, list(map(lambda _: batch_as_dict[_], positive_list)))
                    negative_ind = np.append(negative_ind, list(map(lambda _: batch_as_dict[_], negative_list)))

            model.train()
            optimizer.zero_grad()
            z, out = model(data.x, data.edge_index)
            mse_loss = F.mse_loss(data.x, out)

            anchor_arr = z[anchor_ind,]
            positive_arr = z[positive_ind,]
            negative_arr = z[negative_ind,]

            triplet_loss = torch.nn.TripletMarginLoss(margin=margin, p=2, reduction='mean')
            tri_output = triplet_loss(anchor_arr, positive_arr, negative_arr)

            loss = mse_loss + tri_output
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
            optimizer.step()

        #
        model.eval()
        adata.obsm[key_added] = z.cpu().detach().numpy()
        self.model = model
        return adata

    def save(self, path, **kwargs):
        self.model_name='STAligner'
        model_path = os.path.join(path, self.model_name + '_model.pth')
        torch.save(self.model.state_dict(), model_path)

    def load(self, path, **kwargs):
        self.model_name='STAligner'
        model_path = os.path.join(path, self.model_name + '_model.pth')
        model = self.model.load_state_dict(torch.load(model_path))
        return model