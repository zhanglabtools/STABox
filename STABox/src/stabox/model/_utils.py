"""
Utility functions for model
"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment
from scipy.cluster import hierarchy
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
from torch_geometric.nn.conv import MessagePassing
from torch_sparse import SparseTensor, set_diag
from typing import Union, Tuple, Optional
from torch_geometric.typing import (OptPairTensor, Adj, Size, NoneType,
                                    OptTensor)
from torch.nn import Parameter
from torch import Tensor
from typing import Union, Tuple, Optional
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Data
import scipy.sparse as sp
import pandas as pd
import sklearn.neighbors
import scanpy as sc
from sklearn.neighbors import NearestNeighbors
from annoy import AnnoyIndex
import itertools
import networkx as nx
import hnswlib
from torch_geometric.nn.inits import glorot


class GATConv(MessagePassing):
    r"""The graph attentional operator from the `"Graph Attention Networks"
    <https://arxiv.org/abs/1710.10903>`_ paper
    .. math::
        \mathbf{x}^{\prime}_i = \alpha_{i,i}\mathbf{\Theta}\mathbf{x}_{i} +
        \sum_{j \in \mathcal{N}(i)} \alpha_{i,j}\mathbf{\Theta}\mathbf{x}_{j},
    where the attention coefficients :math:`\alpha_{i,j}` are computed as
    .. math::
        \alpha_{i,j} =
        \frac{
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_j]
        \right)\right)}
        {\sum_{k \in \mathcal{N}(i) \cup \{ i \}}
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_k]
        \right)\right)}.
    Args:
        in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
            derive the size from the first input(s) to the forward method.
            A tuple corresponds to the sizes of source and target
            dimensionalities.
        out_channels (int): Size of each output sample.
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        concat (bool, optional): If set to :obj:`False`, the multi-head
            attentions are averaged instead of concatenated.
            (default: :obj:`True`)
        negative_slope (float, optional): LeakyReLU angle of the negative
            slope. (default: :obj:`0.2`)
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    _alpha: OptTensor

    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, heads: int = 1, concat: bool = True,
                 negative_slope: float = 0.2, dropout: float = 0.0,
                 add_self_loops: bool = True, bias: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(GATConv, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.lin_src = nn.Parameter(torch.zeros(size=(in_channels, out_channels)))
        nn.init.xavier_normal_(self.lin_src.data, gain=1.414)
        self.lin_dst = self.lin_src
        # The learnable parameters to compute attention coefficients:
        self.att_src = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_dst = Parameter(torch.Tensor(1, heads, out_channels))
        nn.init.xavier_normal_(self.att_src.data, gain=1.414)
        nn.init.xavier_normal_(self.att_dst.data, gain=1.414)
        self._alpha = None
        self.attentions = None

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                size: Size = None, return_attention_weights=None, attention=True, tied_attention=None):
        r"""
        Args:
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """
        H, C = self.heads, self.out_channels

        # We first transform the input node features. If a tuple is passed, we
        # transform source and target node features via separate weights:
        if isinstance(x, Tensor):
            assert x.dim() == 2, "Static graphs not supported in 'GATConv'"
            # x_src = x_dst = self.lin_src(x).view(-1, H, C)
            x_src = x_dst = torch.mm(x, self.lin_src).view(-1, H, C)
        else:  # Tuple of source and target node features:
            x_src, x_dst = x
            assert x_src.dim() == 2, "Static graphs not supported in 'GATConv'"
            x_src = self.lin_src(x_src).view(-1, H, C)
            if x_dst is not None:
                x_dst = self.lin_dst(x_dst).view(-1, H, C)

        x = (x_src, x_dst)

        if not attention:
            return x[0].mean(dim=1)
            # return x[0].view(-1, self.heads * self.out_channels)

        if tied_attention == None:
            # Next, we compute node-level attention coefficients, both for source
            # and target nodes (if present):
            alpha_src = (x_src * self.att_src).sum(dim=-1)
            alpha_dst = None if x_dst is None else (x_dst * self.att_dst).sum(-1)
            alpha = (alpha_src, alpha_dst)
            self.attentions = alpha
        else:
            alpha = tied_attention

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                # We only want to add self-loops for nodes that appear both as
                # source and target nodes:
                num_nodes = x_src.size(0)
                if x_dst is not None:
                    num_nodes = min(num_nodes, x_dst.size(0))
                num_nodes = min(size) if size is not None else num_nodes
                edge_index, _ = remove_self_loops(edge_index)
                edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                edge_index = set_diag(edge_index)

        # propagate_type: (x: OptPairTensor, alpha: OptPairTensor)
        out = self.propagate(edge_index, x=x, alpha=alpha, size=size)

        alpha = self._alpha
        assert alpha is not None
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        # if self.bias is not None:
        #     out += self.bias

        if isinstance(return_attention_weights, bool):
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

    def message(self, x_j: Tensor, alpha_j: Tensor, alpha_i: OptTensor,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        # Given egel-level attention coefficients for source and target nodes,
        # we simply need to sum them up to "emulate" concatenation:
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha  # Save for later use.
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.unsqueeze(-1)

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)


class STAGateModule(nn.Module):
    def __init__(self, in_features, hidden_dims):
        super(STAGateModule, self).__init__()
        [num_hidden, out_dim] = hidden_dims
        # print("hidden_dims", hidden_dims)
        # print("type(hidden_dims)", type(hidden_dims))
        self.conv1 = GATConv(in_features, num_hidden, heads=1, concat=False,
                             dropout=0, add_self_loops=False, bias=False)
        self.conv2 = GATConv(num_hidden, out_dim, heads=1, concat=False,
                             dropout=0, add_self_loops=False, bias=False)
        self.conv3 = GATConv(out_dim, num_hidden, heads=1, concat=False,
                             dropout=0, add_self_loops=False, bias=False)
        self.conv4 = GATConv(num_hidden, in_features, heads=1, concat=False,
                             dropout=0, add_self_loops=False, bias=False)

    def forward(self, features, edge_index):
        h1 = F.elu(self.conv1(features, edge_index))
        h2 = self.conv2(h1, edge_index, attention=False)
        self.conv3.lin_src.data = self.conv2.lin_src.transpose(0, 1)
        self.conv3.lin_dst.data = self.conv2.lin_dst.transpose(0, 1)
        self.conv4.lin_src.data = self.conv1.lin_src.transpose(0, 1)
        self.conv4.lin_dst.data = self.conv1.lin_dst.transpose(0, 1)
        h3 = F.elu(self.conv3(h2, edge_index, attention=True,
                              tied_attention=self.conv1.attentions))
        h4 = self.conv4(h3, edge_index, attention=False)
        return h2, h4


class StackMLPModule(nn.Module):
    name = "StackMLP"
    def __init__(self, in_features, n_classes, hidden_dims=[30, 40, 30], activation="relu"):
        super(StackMLPModule, self).__init__()
        self.in_features = in_features
        self.n_classes = n_classes
        self.classifier = nn.ModuleList()
        self.act = nn.ReLU() if activation == "relu" else nn.LeakyReLU()
        mlp_dims = [in_features] + hidden_dims + [n_classes]
        for ind in range(len(mlp_dims) - 1):
            self.classifier.append(nn.Linear(mlp_dims[ind], mlp_dims[ind + 1]))

    def forward(self, x):
        for layer in self.classifier:
            x = layer(x)
            if layer != self.classifier[-1]:
                x = self.act(x)
        score = F.softmax(x, dim=0)
        return {"last_layer": x, "score": score}



def convert_labels(labels):
    """
    convert labels to 0,1, 2, ...
    :param labels:
    :return:
    """
    label_dict = dict()
    for i, label in enumerate(np.unique(labels)):
        label_dict[label] = i
    new_labels = np.zeros_like(labels)
    for i, label in enumerate(labels):
        new_labels[i] = label_dict[label]
    return new_labels


def compute_consensus_matrix(clustering_results):
    """
    Compute the consensus matrix from M times clustering results.

    Parameters:
    -- clustering_results: numpy array of shape (M, n)
        M times clustering results, where M is the number of times clustering was performed
        and n is the number of data points or elements in the clustering results.

    Returns:
    -- consensus_matrix: numpy array of shape (n, n)
        Consensus matrix, where n is the number of data points or elements in the clustering results.
    """
    M, n = clustering_results.shape

    # Compute dissimilarity matrix between clustering results using cdist
    dissimilarity_matrix = distance.cdist(clustering_results, clustering_results, metric='hamming')

    # Compute consensus matrix using linear sum assignment
    row_ind, col_ind = linear_sum_assignment(dissimilarity_matrix)
    consensus_matrix = np.zeros((n, n))
    for i, j in zip(row_ind, col_ind):
        consensus_matrix += (clustering_results[i][:, np.newaxis] == clustering_results[j])

    # Divide the consensus matrix by the number of comparisons to obtain the consensus percentages
    consensus_matrix /= M

    return consensus_matrix


def plot_clustered_consensus_matrix(cmat, n_clusters, method="average", resolution=0.5,
                                    figsize=(5, 5)):
    n_samples = cmat.shape[0]
    linkage_matrix = hierarchy.linkage(cmat, method='average', metric='euclidean')
    cluster_labels = hierarchy.fcluster(linkage_matrix, n_clusters, criterion='maxclust')
    visualization_clusters = hierarchy.fcluster(linkage_matrix, int(n_samples * resolution), criterion='maxclust')
    sorted_indices = np.argsort(visualization_clusters)
    sorted_cmat = cmat[sorted_indices][:, sorted_indices]
    figure, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(sorted_cmat, cmap='rocket', interpolation='nearest')
    return figure, cluster_labels


def consensus_clustering(labels_list, n_clusters, save_dir, name="cluster_labels.npy", plot=True):
    """
    Consensus clustering
    :param n_clusters:
    :param name:
    :param plot:
    :return:
    """
    import time
    st = time.time()
    cons_mat = compute_consensus_matrix(labels_list)
    print("Compute consensus matrix: {:.2f}".format(time.time() - st))
    st = time.time()
    if plot:
        figure, consensus_labels = plot_clustered_consensus_matrix(cons_mat, n_clusters)
        # figure.savefig(os.path.join(self.save_dir, "consensus_clustering.png"))
        figure.savefig(os.path.join('D:\\Users\\lqlu\\download', "consensus_clustering.png"))
        print("plot consensus map: {:.2f}".format(time.time() - st))
    else:
        linkage_matrix = hierarchy.linkage(cons_mat, method='average', metric='euclidean')
        consensus_labels = hierarchy.fcluster(linkage_matrix, n_clusters, criterion='maxclust')
    consensus_labels = convert_labels(consensus_labels)
    np.save(os.path.join(save_dir, "consensus"), consensus_labels)


def Transfer_pytorch_Data(adata):
    G_df = adata.uns['Spatial_Net'].copy()
    cells = np.array(adata.obs_names)
    cells_id_tran = dict(zip(cells, range(cells.shape[0])))
    G_df['Cell1'] = G_df['Cell1'].map(cells_id_tran)
    G_df['Cell2'] = G_df['Cell2'].map(cells_id_tran)

    G = sp.coo_matrix((np.ones(G_df.shape[0]), (G_df['Cell1'], G_df['Cell2'])), shape=(adata.n_obs, adata.n_obs))
    G = G + sp.eye(G.shape[0])

    edgeList = np.nonzero(G)
    if type(adata.X) == np.ndarray:
        data = Data(edge_index=torch.LongTensor(np.array(
            [edgeList[0], edgeList[1]])), x=torch.FloatTensor(adata.X))  # .todense()
    else:
        data = Data(edge_index=torch.LongTensor(np.array(
            [edgeList[0], edgeList[1]])), x=torch.FloatTensor(adata.X.todense()))  # .todense()
    return data
    
def Cal_Spatial_Net(adata, rad_cutoff=None, k_cutoff=None, model='Radius', verbose=True):
    """
    Construct the spatial neighbor networks.

    Parameters
    ----------
    adata
        AnnData object of scanpy package.
    rad_cutoff
        radius cutoff when model='Radius'
    k_cutoff
        The number of nearest neighbors when model='KNN'
    model
        The network construction model. When model=='Radius', the spot is connected to spots whose distance is less than rad_cutoff. When model=='KNN', the spot is connected to its first k_cutoff nearest neighbors.
    
    Returns
    -------
    The spatial networks are saved in adata.uns['Spatial_Net']
    """

    assert(model in ['Radius', 'KNN'])
    if verbose:
        print('------Calculating spatial graph...')
    coor = pd.DataFrame(adata.obsm['spatial'])
    coor.index = adata.obs.index
    coor.columns = ['imagerow', 'imagecol']

    if model == 'Radius':
        nbrs = sklearn.neighbors.NearestNeighbors(radius=rad_cutoff).fit(coor)
        distances, indices = nbrs.radius_neighbors(coor, return_distance=True)
        KNN_list = []
        for it in range(indices.shape[0]):
            KNN_list.append(pd.DataFrame(zip([it]*indices[it].shape[0], indices[it], distances[it])))
    
    if model == 'KNN':
        nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=k_cutoff+1).fit(coor)
        distances, indices = nbrs.kneighbors(coor)
        KNN_list = []
        for it in range(indices.shape[0]):
            KNN_list.append(pd.DataFrame(zip([it]*indices.shape[1],indices[it,:], distances[it,:])))

    KNN_df = pd.concat(KNN_list)
    KNN_df.columns = ['Cell1', 'Cell2', 'Distance']

    Spatial_Net = KNN_df.copy()
    Spatial_Net = Spatial_Net.loc[Spatial_Net['Distance']>0,]
    id_cell_trans = dict(zip(range(coor.shape[0]), np.array(coor.index), ))
    Spatial_Net['Cell1'] = Spatial_Net['Cell1'].map(id_cell_trans)
    Spatial_Net['Cell2'] = Spatial_Net['Cell2'].map(id_cell_trans)
    if verbose:
        print('The graph contains %d edges, %d cells.' %(Spatial_Net.shape[0], adata.n_obs))
        print('%.4f neighbors per cell on average.' %(Spatial_Net.shape[0]/adata.n_obs))

    adata.uns['Spatial_Net'] = Spatial_Net

def Stats_Spatial_Net(adata):
    import matplotlib.pyplot as plt
    Num_edge = adata.uns['Spatial_Net']['Cell1'].shape[0]
    Mean_edge = Num_edge/adata.shape[0]
    plot_df = pd.value_counts(pd.value_counts(adata.uns['Spatial_Net']['Cell1']))
    plot_df = plot_df/adata.shape[0]
    fig, ax = plt.subplots(figsize=[3,2])
    plt.ylabel('Percentage')
    plt.xlabel('')
    plt.title('Number of Neighbors (Mean=%.2f)'%Mean_edge)
    ax.bar(plot_df.index, plot_df)


def mclust_R(representation, n_clusters, r_seed=2022, model_name="EEE"):
    """
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    """
    np.random.seed(r_seed)
    import rpy2.robjects as ro
    from rpy2.robjects import numpy2ri
    numpy2ri.activate()
    ro.r.library("mclust")
    r_random_seed = ro.r['set.seed']
    r_random_seed(r_seed)
    rmclust = ro.r['Mclust']
    res = rmclust(representation, n_clusters, model_name)
    mclust_res = np.array(res[-2])
    numpy2ri.deactivate()
    #  close R session
    return mclust_res.astype('int')


def louvain(representation, resolution=1.0, r_seed=2022):
    """
    Run louvain clustering on the data_module
    """
    adata = sc.AnnData(representation)
    sc.pp.neighbors(adata, use_rep='X')
    sc.tl.louvain(adata, resolution=resolution, random_state=r_seed)
    return  adata.obs["louvain"].to_numpy().astype("int")


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
from sklearn import metrics


def generator_from_index(adata, batch_name, celltype_name=None, mask_batch=None, Y=None, k=20, label_ratio=0.8,
                         k_to_m_ratio=0.75, batch_size=32, search_k=-1,
                         save_on_disk=True, approx=True, verbose=1):
    print('version 0.0.2. 09:00, 12/01/2020')

    # Calculate MNNs by pairwise comparison between batches

    cells = adata.obs_names

    if (verbose > 0):
        print("Calculating MNNs...")

    mnn_dict = create_dictionary_mnn(adata, batch_name=batch_name, k=k, save_on_disk=save_on_disk, approx=approx,
                                     verbose=verbose)

    if (verbose > 0):
        print(str(len(mnn_dict)) + " cells defined as MNNs")

    if celltype_name is None:
        label_dict = dict()
    else:

        if (verbose > 0):
            print('Generating supervised positive pairs...')

        label_dict_original = create_dictionary_label(adata, celltype_name=celltype_name, batch_name=batch_name,
                                                      mask_batch=mask_batch, k=k, verbose=verbose)
        num_label = round(label_ratio * len(label_dict_original))

        cells_for_label = np.random.choice(list(label_dict_original.keys()), num_label, replace=False)

        label_dict = {key: value for key, value in label_dict_original.items() if key in cells_for_label}

        if (verbose > 0):
            print(str(len(label_dict.keys())) + " cells defined as supervision triplets")

        print(len(set(mnn_dict.keys()) & set(label_dict.keys())))

    if k_to_m_ratio == 0.0:
        knn_dict = dict()
    else:
        num_k = round(k_to_m_ratio * len(mnn_dict))
        # Calculate KNNs for subset of residual cells
        # 除MNN节点之外的所有节点
        cells_for_knn = list(set(cells) - (set(list(label_dict.keys())) | set(list(mnn_dict.keys()))))
        if (len(cells_for_knn) > num_k):  # 如果剩余的节点过多，就只选择num_k个
            cells_for_knn = np.random.choice(cells_for_knn, num_k, replace=False)

        if (verbose > 0):
            print("Calculating KNNs...")

        cdata = adata[cells_for_knn]
        knn_dict = create_dictionary_knn(cdata, cells_for_knn, k=k, save_on_disk=save_on_disk, approx=approx)
        if (verbose > 0):
            print(str(len(cells_for_knn)) + " cells defined as KNNs")

    final_dict = merge_dict(mnn_dict, label_dict)
    final_dict.update(knn_dict)

    cells_for_train = list(final_dict.keys())
    print('Total cells for training:' + str(len(cells_for_train)))

    ddata = adata[cells_for_train]

    # Reorder triplet list according to cells
    if (verbose > 0):
        print("Reorder")
    names_as_dict = dict(zip(list(adata.obs_names), range(0, adata.shape[0])))  # 建立adata.obs_names与顺序数字编号的对应关系

    def get_indices2(name):
        return ([names_as_dict[x] for x in final_dict[name]])

    triplet_list = list(map(get_indices2, cells_for_train))  # 把用于训练的细胞单独提出来到这个list，细胞重新顺序编号，用于找anchor

    batch_list = ddata.obs[batch_name]  # 用于训练的ddata的obs_names还是原始的编号
    batch_indices = []
    for i in batch_list.unique():  # 把三个批次的细胞分别提取出来
        batch_indices.append(list(np.where(batch_list == i)[0]))  # 但是这里的编号就被重新设定了

    batch_as_dict = dict(zip(list(batch_list.unique()), range(0, len(batch_list.unique()))))
    tmp = map(lambda _: batch_as_dict[_], batch_list)
    batch_list = list(tmp)

    if Y is None:
        return KnnTripletGenerator(X=ddata.obsm["X_pca"], X1=adata.obsm['X_pca'], dictionary=triplet_list,
                                   batch_list=batch_list, batch_indices=batch_indices, batch_size=batch_size)

    else:
        tmp = dict(zip(cells, Y))
        Y_new = [tmp[x] for x in cells_for_train]
        Y_new = le.fit_transform(Y_new)
        return LabeledKnnTripletGenerator(X=ddata.obsm["X_pca"], X1=adata.obsm['X_pca'], Y=Y_new,
                                          dictionary=triplet_list,
                                          batch_list=batch_list, batch_indices=batch_indices, batch_size=batch_size)


def merge_dict(x, y):
    for k, v in x.items():
        if k in y.keys():
            y[k] += v
        else:
            y[k] = v
    return y


def create_dictionary_mnn(adata, use_rep, batch_name, k=50, save_on_disk=True, approx=True, verbose=1, iter_comb=None):
    cell_names = adata.obs_names

    batch_list = adata.obs[batch_name]
    datasets = []
    datasets_pcs = []
    cells = []
    for i in batch_list.unique():
        datasets.append(adata[batch_list == i])
        datasets_pcs.append(adata[batch_list == i].obsm[use_rep])
        cells.append(cell_names[batch_list == i])

    batch_name_df = pd.DataFrame(np.array(batch_list.unique()))
    mnns = dict()

    # if len(cells) > 2:
    if iter_comb is None:
        iter_comb = list(itertools.combinations(range(len(cells)), 2))
    # for comb in list(itertools.combinations(range(len(cells)), 2)): # 返回多个批次所有可能的组合
    for comb in iter_comb:
        i = comb[0]
        j = comb[1]
        key_name1 = batch_name_df.loc[comb[0]].values[0] + "_" + batch_name_df.loc[comb[1]].values[0]
        mnns[key_name1] = {}

        if (verbose > 0):
            print('Processing datasets {}'.format((i, j)))

        new = list(cells[j])
        ref = list(cells[i])

        ds1 = adata[new].obsm[use_rep]
        ds2 = adata[ref].obsm[use_rep]
        names1 = new
        names2 = ref
        # 如果K>1，则MNN点就很有可能出现1:n,即一对多的情况
        match = mnn(ds1, ds2, names1, names2, knn=k, save_on_disk=save_on_disk, approx=approx)

        G = nx.Graph()
        G.add_edges_from(match)
        node_names = np.array(G.nodes)
        anchors = list(node_names)
        adj = nx.adjacency_matrix(G)  # src 和 dst 中的points拼起来作为矩阵的列或行，这个矩阵是对称的
        # https://blog.csdn.net/Snowmyth/article/details/121280577?spm=1001.2101.3001.6661.1&utm_medium=distribute.pc_relevant_t0.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-1.pc_relevant_paycolumn_v3&depth_1-utm_source=distribute.pc_relevant_t0.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-1.pc_relevant_paycolumn_v3&utm_relevant_index=1
        # indptr提示的是非零数在稀疏矩阵中的位置信息。indices是具体的连接边的一个节点的编号。
        # https://www.csdn.net/tags/NtzaQgysMjgyNjEtYmxvZwO0O0OO0O0O.html
        tmp = np.split(adj.indices, adj.indptr[1:-1])  # 把一个数组从左到右按顺序切分;

        for i in range(0, len(anchors)):  # 把src 和 dst 中所有的points都包含到字典的key中了
            key = anchors[i]
            i = tmp[i]
            names = list(node_names[i])
            # mnns这里是个字典，多个切片时，由于key是相同的
            # 最后一个切片的mnn会把前面的mnn都覆盖掉，导致最后一个切片的mnn特别多！
            mnns[key_name1][key] = names
    return (mnns)


def create_dictionary_knn(adata, use_rep, cell_subset, k=50, save_on_disk=True, approx=True):
    # cell_names = adata.obs_names

    dataset = adata[cell_subset]
    pcs = dataset.obsm[use_rep]

    def get_names(ind):
        return np.array(cell_subset)[ind]

    if approx:
        dim = pcs.shape[1]
        num_elements = pcs.shape[0]
        p = hnswlib.Index(space='l2', dim=dim)
        p.init_index(max_elements=num_elements, ef_construction=100, M=16)
        p.set_ef(10)
        p.add_items(pcs)
        ind, distances = p.knn_query(pcs, k=k)
        ind = ind[1:]  # remove self-point

        cell_subset = np.array(cell_subset)
        names = list(map(lambda x: cell_subset[x], ind))
        knns = dict(zip(cell_subset, names))

    else:
        nn_ = NearestNeighbors(n_neighbors=k, p=2)
        nn_.fit(pcs)
        ind = nn_.kneighbors(pcs, return_distance=False)
        ind = ind[1:]  # remove self-point

        names = list(map(lambda x: cell_subset[x], ind))
        knns = dict(zip(cell_subset, names))

    return (knns)


def validate_sparse_labels(Y):
    if not zero_indexed(Y):
        raise ValueError('Ensure that your labels are zero-indexed')
    if not consecutive_indexed(Y):
        raise ValueError('Ensure that your labels are indexed consecutively')


def zero_indexed(Y):
    if min(abs(Y)) != 0:
        return False
    return True


def consecutive_indexed(Y):
    """ Assumes that Y is zero-indexed. """
    n_classes = len(np.unique(Y[Y != np.array(-1)]))
    if max(Y) >= n_classes:
        return False
    return True


def nn_approx(ds1, ds2, names1, names2, knn=50, pos_knn=None):
    dim = ds2.shape[1]
    num_elements = ds2.shape[0]
    p = hnswlib.Index(space='l2', dim=dim)
    p.init_index(max_elements=num_elements, ef_construction=100, M=16)
    p.set_ef(10)
    p.add_items(ds2)
    ind, distances = p.knn_query(ds1, k=knn)
    match = set()
    for a, b in zip(range(ds1.shape[0]), ind):
        for b_i in b:
            match.add((names1[a], names2[b_i]))
    return match


def Staligner_nn(ds1, ds2, names1, names2, knn=50, metric_p=2):
    # Find nearest neighbors of first dataset.
    nn_ = NearestNeighbors(knn, p=metric_p)
    nn_.fit(ds2)
    ind = nn_.kneighbors(ds1, return_distance=False)

    match = set()
    for a, b in zip(range(ds1.shape[0]), ind):
        for b_i in b:
            match.add((names1[a], names2[b_i]))

    return match


def nn_annoy(ds1, ds2, names1, names2, knn=20, metric='euclidean', n_trees=50, save_on_disk=True):
    """ Assumes that Y is zero-indexed. """
    # Build index.
    a = AnnoyIndex(ds2.shape[1], metric=metric)
    if (save_on_disk):
        a.on_disk_build('annoy.index')
    for i in range(ds2.shape[0]):
        a.add_item(i, ds2[i, :])
    a.build(n_trees)

    # Search index.
    ind = []
    for i in range(ds1.shape[0]):
        ind.append(a.get_nns_by_vector(ds1[i, :], knn, search_k=-1))
    ind = np.array(ind)

    # Match.
    match = set()
    for a, b in zip(range(ds1.shape[0]), ind):
        for b_i in b:
            match.add((names1[a], names2[b_i]))

    return match


def mnn(ds1, ds2, names1, names2, knn=20, save_on_disk=True, approx=True, pos_knn1=None, pos_knn2=None):
    # Find nearest neighbors in first direction.
    if approx:  # 输出KNN pair; match1: (names1中节点，names2中节点), 大小为ds1.shape[0]*knn
        match1 = nn_approx(ds1, ds2, names1, names2, knn=knn, pos_knn=pos_knn1)  # , save_on_disk = save_on_disk)
        # Find nearest neighbors in second direction.
        match2 = nn_approx(ds2, ds1, names2, names1, knn=knn, pos_knn=pos_knn2)  # , save_on_disk = save_on_disk)
    else:
        match1 = Staligner_nn(ds1, ds2, names1, names2, knn=knn)
        match2 = Staligner_nn(ds2, ds1, names2, names1, knn=knn)
    # Compute mutual nearest neighbors.
    mutual = match1 & set([(b, a) for a, b in match2])

    return mutual


def Transfer_pytorch_Data(adata):
    G_df = adata.uns['Spatial_Net'].copy()
    cells = np.array(adata.obs_names)
    cells_id_tran = dict(zip(cells, range(cells.shape[0])))
    G_df['Cell1'] = G_df['Cell1'].map(cells_id_tran)
    G_df['Cell2'] = G_df['Cell2'].map(cells_id_tran)

    G = sp.coo_matrix((np.ones(G_df.shape[0]), (G_df['Cell1'], G_df['Cell2'])), shape=(adata.n_obs, adata.n_obs))
    G = G + sp.eye(G.shape[0])

    edgeList = np.nonzero(G)
    if type(adata.X) == np.ndarray:
        data = Data(edge_index=torch.LongTensor(np.array(
            [edgeList[0], edgeList[1]])), x=torch.FloatTensor(adata.X))  # .todense()
    else:
        data = Data(edge_index=torch.LongTensor(np.array(
            [edgeList[0], edgeList[1]])), x=torch.FloatTensor(adata.X.todense()))  # .todense()
    return data