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

    def train_minibatch(self, adata_concat, hidden_dims=[512, 30], n_epochs=500, lr=0.001,
                                  key_added='STAligner', MNN_pair=5, batch_size_list=None,
                                  gradient_clipping=5., weight_decay=0.0001, margin=1,
                                  verbose=True, step_nan=True, random_seed=666, iter_comb=None, knn_neigh=50,
                                  alpha=0.5, num_neighbors=[-1], device=torch.device(
                'cuda:0' if torch.cuda.is_available() else 'cpu'), all_MNN=False):
        """\
        Train graph attention auto-encoder and use spot triplets across slices to perform batch correction in the embedding space.

        Parameters
        ----------
        adata_concat
            AnnData object of scanpy package.
        hidden_dims
            The dimension of the encoder.
        n_epochs
            Number of total epochs in training.
        lr
            Learning rate for AdamOptimizer.
        key_added
            The latent embeddings are saved in adata.obsm[key_added].
        gradient_clipping
            Gradient Clipping.
        weight_decay
            Weight decay for AdamOptimizer.
        margin
            Margin is used in triplet loss to enforce the distance between positive and negative pairs.
            Larger values result in more aggressive correction.
        iter_comb
            For multiple slices integration, we perform iterative pairwise integration. iter_comb is used to specify the order of integration.
            For example, (0, 1) means slice 0 will be algined with slice 1 as reference.
            If iter_comb is not evaluated, then the two slices that are sequentially adjacent are integrated by default
        knn_neigh
            The number of nearest neighbors when constructing MNNs. If knn_neigh>1, points in one slice may have multiple MNN points in another slice.
        alpha
            alpha controls the relative contributions of Lrec loss and Ltri loss
        device
            See torch.device.
        num_neighbors
            Neighborhood Sampling.
        MNN_pair
            The number of MNN_pairs used in each pair of slices when calculating triplet loss.
        step_nan
            After taking minibatch for nodes in each graph, if there is no MNN pair when integrating subgraphs of the two slices, the two subgraph data are
            considered to be not available for bootstrap pairing. At this point, this epoch should be skipped.
        batch_size_list
            Each element corresponds to the number of center nodes selected in each slice. The minibatch_size for each slice is recommended to be chosen as 5% to           15% of the number of cells.
            If batch_size_list is not evaluated, then the default minibatch_size is 2048 per slice.

        Returns
        -------
        AnnData
        """

        # seed_everything()
        global MNN_df_list
        seed = random_seed
        import random
        import anndata as ad
        import numpy as np

        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

        categories = adata_concat.obs['slice_name'].cat.categories
        sorted_counts = adata_concat.obs['slice_name'].value_counts().reindex(categories)

        slice_cell_num_list = sorted_counts.tolist()

        section_ids = adata_concat.obs['slice_name'].cat.categories.tolist()

        if iter_comb == None:
            # If iter_comb is not evaluated, then the two slices that are sequentially adjacent are integrated by default
            iter_comb = [(i, i + 1) for i in range(len(section_ids) - 1)]
        if batch_size_list == None:
            # If batch_size_list is not evaluated, then the default minibatch_size is 2048 per slice.
            batch_size_list = [2048 for _ in range(len(section_ids))]

        adj_list = adata_concat.uns['edge_list']

        # concat the different slices, then take the minibatch and put it in all_train_loader
        all_train_loader = []

        for comb in iter_comb:

            max_ind = 0
            edge_list = []
            edge_comb_list_0 = []
            edge_comb_list_1 = []

            i = comb[0]
            num_i = sum(slice_cell_num_list[:i])
            num_j = sum(slice_cell_num_list[:i + 1])

            adata_i = adata_concat[num_i:num_j, ]

            j = comb[1]
            num_i = sum(slice_cell_num_list[:j])
            num_j = sum(slice_cell_num_list[:j + 1])
            adata_j = adata_concat[num_i:num_j, ]

            for (adata, k) in zip([adata_i, adata_j], [i, j]):
                edge_list.append(np.nonzero(adj_list[k]))

                # Adjust node numbering not to repeat in different graphs
                if max_ind > 0:
                    edge_list[-1] = (edge_list[-1][0] + max_ind + 1, edge_list[-1][1] + max_ind + 1)
                    max_ind = np.max(edge_list[-1])
                else:
                    max_ind = edge_list[-1][0].max()
                edge_comb_list_0 = np.append(edge_comb_list_0, edge_list[-1][0])
                edge_comb_list_1 = np.append(edge_comb_list_1, edge_list[-1][1])

            edge_comb_list = (edge_comb_list_0.astype(int), edge_comb_list_1.astype(int))

            adata_comb = ad.concat([adata_i, adata_j], label="slice_name",
                                   keys=[section_ids[i], section_ids[j]])

            adata_comb.uns['edgeList'] = edge_comb_list

            comb_section_ids = np.array(adata_comb.obs['batch_name'].unique())

            from torch_geometric.data import Data, Batch
            from torch_geometric.loader import DataLoader

            comb_data = Data(
                edge_index=torch.LongTensor(np.array([adata_comb.uns['edgeList'][0], adata_comb.uns['edgeList'][1]])),
                x=torch.FloatTensor(adata_comb.X.todense()), n_id=torch.arange(adata_comb.shape[0]))

            comb_data = comb_data.to(device)

            from torch_geometric.loader import NeighborLoader

            train_loader1 = NeighborLoader(comb_data, input_nodes=None, num_neighbors=num_neighbors, shuffle=True,
                                           batch_size=batch_size_list[i] + batch_size_list[j])

            all_train_loader.append(train_loader1)

        test_loaders_list = []

        for i in range(len(slice_cell_num_list)):
            num_i = sum(slice_cell_num_list[:i])
            num_j = sum(slice_cell_num_list[:i + 1])

            adata_i = adata_concat[num_i:num_j, ]

            adata_i.uns['edgeList'] = np.nonzero(adj_list[i])

            from torch_geometric.data import Data, Batch
            from torch_geometric.loader import NeighborLoader

            graph_adata_i = Data(
                edge_index=torch.LongTensor(np.array([adata_i.uns['edgeList'][0], adata_i.uns['edgeList'][1]])),
                x=torch.FloatTensor(adata_i.X.todense()), n_id=torch.arange(adata_i.shape[0]))

            graph_adata_i = graph_adata_i.to(device)

            graph_adata_i = graph_adata_i.to(device)

            test_loader = NeighborLoader(graph_adata_i, input_nodes=None, num_neighbors=num_neighbors, shuffle=False,
                                         batch_size=batch_size_list[i])

            test_loaders_list.append(test_loader)

        # model = STAligner(
        #     hidden_dims=[adata_concat.n_vars, hidden_dims[0], hidden_dims[1]]).to(device)
        model = STAGateModule(adata_concat.X.shape[1], self.hidden_dims).to(device)
        # optimizer = torch.optim.Adam(model.parameters(), lr=lr,
        #                              weight_decay=weight_decay, eps=1e-4)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                     weight_decay=weight_decay)
        if verbose:
            print(model)

        print('Pretrain with STAGATE...' + str(iter_comb))
        STAGATE_loss_list = []

        for epoch in tqdm(range(0, n_epochs // 2)):
            '''Implemented to minibatch two slices and merge them together'''
            for comb in iter_comb:
                i = iter_comb.index(comb)
                for batch in all_train_loader[i]:
                    # STAGATE
                    model.train()
                    optimizer.zero_grad()

                    z, out = model(batch.x, batch.edge_index)

                    loss = F.mse_loss(batch.x[:batch.batch_size],
                                      out[:batch.batch_size])

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                    optimizer.step()
                    # STAGATE_loss_list.append(loss.item())

        with torch.no_grad():
            z_list = []

            for test_loader in test_loaders_list:
                for batch in test_loader:
                    z, out = model(batch.x, batch.edge_index)
                    z_list.append(z.cpu().detach().numpy()[:batch.batch_size])
            adata_concat.obsm['STAGATE'] = np.concatenate(z_list, axis=0)

        print('Train with STAligner...' + str(iter_comb))
        loss_list = []
        empty = 0
        for epoch in tqdm(range(0, n_epochs // 2)):
            epoch_all_empty = []
            if epoch % 100 == 0:  ## and epoch > 0:
                if verbose:
                    print('Update spot triplets at epoch ' + str(epoch))

                with torch.no_grad():
                    z_list = []

                    for test_loader in test_loaders_list:
                        for batch in test_loader:
                            z, out = model(batch.x, batch.edge_index)
                            z_list.append(z.cpu().detach().numpy()[:batch.batch_size])
                    adata_concat.obsm['STAGATE'] = np.concatenate(z_list, axis=0)

                # If knn_neigh>1, points in one slice may have multiple MNN points in another slice.
                # not all points have MNN achors
                mnn_dict = create_dictionary_mnn(adata_concat, use_rep='STAGATE',
                                                 batch_name='batch_name',
                                                 k=knn_neigh,
                                                 iter_comb=iter_comb, verbose=0)

                MNN_df_list = []
                num_key = 0
                for batch_pair in mnn_dict.keys():  # pairwise compare for multiple batches

                    anchor_ind = []
                    positive_ind = []
                    negative_ind = []

                    batch_list = adata_concat.obs['batch_name'][
                        mnn_dict[batch_pair].keys()]

                    cellname_by_batch_dict = dict()

                    i = iter_comb[num_key][0]
                    j = iter_comb[num_key][1]

                    num_i = sum(slice_cell_num_list[:i])
                    num_j = sum(slice_cell_num_list[:i + 1])

                    adata_i = adata_concat[num_i:num_j, ]

                    num_i = sum(slice_cell_num_list[:j])
                    num_j = sum(slice_cell_num_list[:j + 1])
                    adata_j = adata_concat[num_i:num_j, ]

                    adata_comb = ad.concat([adata_i, adata_j], label="slice_name",
                                           keys=[section_ids[i], section_ids[j]])

                    comb_section_ids = np.array(adata_comb.obs['batch_name'].unique())

                    for batch_id in range(len(comb_section_ids)):
                        cellname_by_batch_dict[comb_section_ids[batch_id]] = \
                            adata_comb.obs_names[
                                adata_comb.obs['batch_name'] == comb_section_ids[
                                    batch_id]].values

                    anchor_list = []
                    positive_list = []
                    negative_list = []
                    for anchor in mnn_dict[batch_pair].keys():
                        i = len(mnn_dict[batch_pair][anchor])

                        if all_MNN == True:
                            i = i
                        else:
                            i = min(i, MNN_pair)

                        for j in torch.arange(i):
                            anchor_list.append(anchor)

                            positive_spot = mnn_dict[batch_pair][anchor][j]
                            positive_list.append(positive_spot)
                            section_size = len(cellname_by_batch_dict[batch_list[anchor]])
                            negative_list.append(cellname_by_batch_dict[batch_list[anchor]][
                                                     np.random.randint(section_size)])

                    batch_as_dict = dict(
                        zip(list(adata_comb.obs_names), range(0, adata_comb.shape[0])))

                    anchor_ind = np.append(anchor_ind, list(
                        map(lambda _: batch_as_dict[_], anchor_list)))
                    positive_ind = np.append(positive_ind, list(
                        map(lambda _: batch_as_dict[_], positive_list)))
                    negative_ind = np.append(negative_ind, list(
                        map(lambda _: batch_as_dict[_], negative_list)))

                    anchor_ind = np.asarray(anchor_ind).astype(np.int64)
                    positive_ind = np.asarray(positive_ind).astype(np.int64)
                    negative_ind = np.asarray(negative_ind).astype(np.int64)

                    comb_MNN_df = pd.DataFrame([anchor_ind, positive_ind, negative_ind]).T
                    MNN_df_list.append(comb_MNN_df)
                    num_key = num_key + 1

            for comb in iter_comb:
                i = iter_comb.index(comb)
                for batch in all_train_loader[i]:

                    # STAGATE
                    model.train()
                    optimizer.zero_grad()

                    z, out = model(batch.x, batch.edge_index)

                    mse_loss = F.mse_loss(batch.x[:batch.batch_size],
                                          out[:batch.batch_size])

                    MNN_df = MNN_df_list[i]

                    MNN_batched = MNN_df[MNN_df.loc[:, 0].isin(
                        batch.n_id.cpu().numpy()[:batch.batch_size])
                                         & MNN_df.loc[:, 1].isin(
                        batch.n_id.cpu().numpy()[:batch.batch_size])
                                         & MNN_df.loc[:, 2].isin(
                        batch.n_id.cpu().numpy()[:batch.batch_size])]

                    if MNN_batched.empty:
                        empty = empty + 1
                        if step_nan:
                            continue

                    global_2_local = dict(
                        zip(list(batch.n_id.cpu().numpy()), range(0, batch.x.shape[0])))

                    MNN_a = list(
                        map(lambda _: global_2_local[_], MNN_batched.iloc[:, 0]))
                    MNN_p = list(
                        map(lambda _: global_2_local[_], MNN_batched.iloc[:, 1]))
                    MNN_n = list(
                        map(lambda _: global_2_local[_], MNN_batched.iloc[:, 2]))

                    anchor_arr = z[MNN_a,]
                    positive_arr = z[MNN_p,]
                    negative_arr = z[MNN_n,]

                    triplet_loss = torch.nn.TripletMarginLoss(margin=margin, p=2,
                                                              reduction='mean')
                    tri_output = triplet_loss(anchor_arr, positive_arr, negative_arr)

                    loss = alpha * mse_loss + (1 - alpha) * tri_output
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                   gradient_clipping)
                    optimizer.step()

                    loss_list.append(loss.item())

        model.eval()

        with torch.no_grad():
            z_list = []
            out_list = []

            for test_loader in test_loaders_list:
                for batch in test_loader:
                    z, out = model(batch.x, batch.edge_index)
                    z_list.append(z.cpu().detach().numpy()[:batch.batch_size])
                    out_list.append(out.cpu().detach().numpy()[:batch.batch_size])

            adata_concat.obsm[key_added] = np.concatenate(z_list, axis=0)
            out = np.concatenate(out_list, axis=0)

            ReX = pd.DataFrame(out, index=adata_concat.obs_names,
                               columns=adata_concat.var_names)
            ReX[ReX < 0] = 0
        adata_concat.layers['STAligner_ReX'] = ReX

        adata_concat.uns['edge_list'] = []

        return adata_concat

    def save(self, path, **kwargs):
        self.model_name='STAligner'
        model_path = os.path.join(path, self.model_name + '_model.pth')
        torch.save(self.model.state_dict(), model_path)

    def load(self, path, **kwargs):
        self.model_name='STAligner'
        model_path = os.path.join(path, self.model_name + '_model.pth')
        model = self.model.load_state_dict(torch.load(model_path))
        return model