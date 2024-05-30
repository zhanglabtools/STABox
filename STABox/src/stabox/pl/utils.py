import pandas as pd
import numpy as np
import sklearn.neighbors
import scipy.sparse as sp
import seaborn as sns
import matplotlib.pyplot as plt
import os
import torch
import yaml
from torch_geometric.data import Data
import scanpy as sc
import sklearn
import itertools
import hnswlib
from sklearn.neighbors import NearestNeighbors
from intervaltree import IntervalTree
import operator

from scipy.sparse import issparse
from annoy import AnnoyIndex
from multiprocessing import Process, cpu_count, Queue
from collections import namedtuple
from operator import attrgetter
from tqdm import tqdm
import time
import itertools
import networkx as nx


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


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
    ## 遍历ind中每一行的每个KNN节点，判断当前点的空间相近KNN和跨批次的KNN有多少是重叠的, 去掉重叠过少的点
    match = set()
    for a, b in zip(range(ds1.shape[0]), ind):
        # knn_used = np.asarray([len(set(pos_knn[b[i]]).intersection(set(b))) for i in range(knn)]) > 0
        # for b_i in b[knn_used]:
        #     match.add((names1[a], names2[b_i]))
        # for b_i in b:
        #     if len(set(pos_knn[b_i]).intersection(set(b))) > 0: #去掉没有重叠的点
        #         match.add((names1[a], names2[b_i]))
        for b_i in b:
            match.add((names1[a], names2[b_i]))
    return match


def nn(ds1, ds2, names1, names2, knn=50, metric_p=2):
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
        match1 = nn(ds1, ds2, names1, names2, knn=knn)
        match2 = nn(ds2, ds1, names2, names1, knn=knn)
    # Compute mutual nearest neighbors.
    mutual = match1 & set([(b, a) for a, b in match2])

    return mutual


def Batch_Data(adata, num_batch_x, num_batch_y, spatial_key=['X', 'Y'], plot_Stats=False):
    Sp_df = adata.obs.loc[:, spatial_key].copy()
    Sp_df = np.array(Sp_df)
    batch_x_coor = [np.percentile(Sp_df[:, 0], (1/num_batch_x)*x*100) for x in range(num_batch_x+1)]
    batch_y_coor = [np.percentile(Sp_df[:, 1], (1/num_batch_y)*x*100) for x in range(num_batch_y+1)]

    Batch_list = []
    for it_x in range(num_batch_x):
        for it_y in range(num_batch_y):
            min_x = batch_x_coor[it_x]
            max_x = batch_x_coor[it_x+1]
            min_y = batch_y_coor[it_y]
            max_y = batch_y_coor[it_y+1]
            temp_adata = adata.copy()
            temp_adata = temp_adata[temp_adata.obs[spatial_key[0]].map(lambda x: min_x <= x <= max_x)]
            temp_adata = temp_adata[temp_adata.obs[spatial_key[1]].map(lambda y: min_y <= y <= max_y)]
            Batch_list.append(temp_adata)
    if plot_Stats:
        f, ax = plt.subplots(figsize=(1, 3))
        plot_df = pd.DataFrame([x.shape[0] for x in Batch_list], columns=['#spot/batch'])
        sns.boxplot(y='#spot/batch', data=plot_df, ax=ax)
        sns.stripplot(y='#spot/batch', data=plot_df, ax=ax, color='red', size=5)
    return Batch_list


def Cal_Spatial_Net(adata, rad_cutoff=None, k_cutoff=None, model='Radius', verbose=True):
    """\
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

    assert (model in ['Radius', 'KNN'])
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
            KNN_list.append(pd.DataFrame(zip([it] * indices[it].shape[0], indices[it], distances[it])))

    if model == 'KNN':
        nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=k_cutoff + 1).fit(coor)
        distances, indices = nbrs.kneighbors(coor)
        KNN_list = []
        for it in range(indices.shape[0]):
            KNN_list.append(pd.DataFrame(zip([it] * indices.shape[1], indices[it, :], distances[it, :])))

    KNN_df = pd.concat(KNN_list)
    KNN_df.columns = ['Cell1', 'Cell2', 'Distance']

    Spatial_Net = KNN_df.copy()
    Spatial_Net = Spatial_Net.loc[Spatial_Net['Distance'] > 0,]
    id_cell_trans = dict(zip(range(coor.shape[0]), np.array(coor.index), ))
    Spatial_Net['Cell1'] = Spatial_Net['Cell1'].map(id_cell_trans)
    Spatial_Net['Cell2'] = Spatial_Net['Cell2'].map(id_cell_trans)
    if verbose:
        print('The graph contains %d edges, %d cells.' % (Spatial_Net.shape[0], adata.n_obs))
        print('%.4f neighbors per cell on average.' % (Spatial_Net.shape[0] / adata.n_obs))

    adata.uns['Spatial_Net'] = Spatial_Net


def Stats_Spatial_Net(adata, rad_cutoff=None, k_cutoff=None, model='Radius', verbose=True):
    """\
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


def Cal_Spatial_Net_3D(adata, rad_cutoff_2D, rad_cutoff_Zaxis,
                       key_section='Section_id', section_order=None, verbose=True):
    """\
    Construct the spatial neighbor networks.

    Parameters
    ----------
    adata
        AnnData object of scanpy package.
    rad_cutoff_2D
        radius cutoff for 2D SNN construction.
    rad_cutoff_Zaxis
        radius cutoff for 2D SNN construction for consturcting SNNs between adjacent sections.
    key_section
        The columns names of section_ID in adata.obs.
    section_order
        The order of sections. The SNNs between adjacent sections are constructed according to this order.

    Returns
    -------
    The 3D spatial networks are saved in adata.uns['Spatial_Net'].
    """
    adata.uns['Spatial_Net_2D'] = pd.DataFrame()
    adata.uns['Spatial_Net_Zaxis'] = pd.DataFrame()
    num_section = np.unique(adata.obs[key_section]).shape[0]
    if verbose:
        print('Radius used for 2D SNN:', rad_cutoff_2D)
        print('Radius used for SNN between sections:', rad_cutoff_Zaxis)
    for temp_section in np.unique(adata.obs[key_section]):
        if verbose:
            print('------Calculating 2D SNN of section ', temp_section)
        temp_adata = adata[adata.obs[key_section] == temp_section,]
        Cal_Spatial_Net(
            temp_adata, rad_cutoff=rad_cutoff_2D, verbose=False)
        temp_adata.uns['Spatial_Net']['SNN'] = temp_section
        if verbose:
            print('This graph contains %d edges, %d cells.' %
                  (temp_adata.uns['Spatial_Net'].shape[0], temp_adata.n_obs))
            print('%.4f neighbors per cell on average.' %
                  (temp_adata.uns['Spatial_Net'].shape[0] / temp_adata.n_obs))
        adata.uns['Spatial_Net_2D'] = pd.concat(
            [adata.uns['Spatial_Net_2D'], temp_adata.uns['Spatial_Net']])
    for it in range(num_section - 1):
        section_1 = section_order[it]
        section_2 = section_order[it + 1]
        if verbose:
            print('------Calculating SNN between adjacent section %s and %s.' %
                  (section_1, section_2))
        Z_Net_ID = section_1 + '-' + section_2
        temp_adata = adata[adata.obs[key_section].isin(
            [section_1, section_2]),]
        Cal_Spatial_Net(
            temp_adata, rad_cutoff=rad_cutoff_Zaxis, verbose=False)
        spot_section_trans = dict(
            zip(temp_adata.obs.index, temp_adata.obs[key_section]))
        temp_adata.uns['Spatial_Net']['Section_id_1'] = temp_adata.uns['Spatial_Net']['Cell1'].map(
            spot_section_trans)
        temp_adata.uns['Spatial_Net']['Section_id_2'] = temp_adata.uns['Spatial_Net']['Cell2'].map(
            spot_section_trans)
        used_edge = temp_adata.uns['Spatial_Net'].apply(
            lambda x: x['Section_id_1'] != x['Section_id_2'], axis=1)
        temp_adata.uns['Spatial_Net'] = temp_adata.uns['Spatial_Net'].loc[used_edge,]
        temp_adata.uns['Spatial_Net'] = temp_adata.uns['Spatial_Net'].loc[:, [
                                                                                 'Cell1', 'Cell2', 'Distance']]
        temp_adata.uns['Spatial_Net']['SNN'] = Z_Net_ID
        if verbose:
            print('This graph contains %d edges, %d cells.' %
                  (temp_adata.uns['Spatial_Net'].shape[0], temp_adata.n_obs))
            print('%.4f neighbors per cell on average.' %
                  (temp_adata.uns['Spatial_Net'].shape[0] / temp_adata.n_obs))
        adata.uns['Spatial_Net_Zaxis'] = pd.concat(
            [adata.uns['Spatial_Net_Zaxis'], temp_adata.uns['Spatial_Net']])
    adata.uns['Spatial_Net'] = pd.concat(
        [adata.uns['Spatial_Net_2D'], adata.uns['Spatial_Net_Zaxis']])
    if verbose:
        print('3D SNN contains %d edges, %d cells.' %
              (adata.uns['Spatial_Net'].shape[0], adata.n_obs))
        print('%.4f neighbors per cell on average.' %
              (adata.uns['Spatial_Net'].shape[0] / adata.n_obs))

def match_cluster_labels(true_labels, est_labels):
    true_labels_arr = np.array(list(true_labels))
    est_labels_arr = np.array(list(est_labels))
    org_cat = list(np.sort(list(pd.unique(true_labels))))
    est_cat = list(np.sort(list(pd.unique(est_labels))))
    B = nx.Graph()
    B.add_nodes_from([i + 1 for i in range(len(org_cat))], bipartite=0)
    B.add_nodes_from([-j - 1 for j in range(len(est_cat))], bipartite=1)
    for i in range(len(org_cat)):
        for j in range(len(est_cat)):
            weight = np.sum((true_labels_arr == org_cat[i]) * (est_labels_arr == est_cat[j]))
            B.add_edge(i + 1, -j - 1, weight=-weight)
    match = nx.algorithms.bipartite.matching.minimum_weight_full_matching(B)
    #     match = minimum_weight_full_matching(B)
    if len(org_cat) >= len(est_cat):
        return np.array([match[-est_cat.index(c) - 1] - 1 for c in est_labels_arr])
    else:
        unmatched = [c for c in est_cat if not (-est_cat.index(c) - 1) in match.keys()]
        l = []
        for c in est_labels_arr:
            if (-est_cat.index(c) - 1) in match:
                l.append(match[-est_cat.index(c) - 1] - 1)
            else:
                l.append(len(org_cat) + unmatched.index(c))
        return np.array(l)


def Cal_Spatial_Net_new(adata, rad_cutoff=None, k_cutoff=None, max_neigh=50, model='Radius', verbose=True):
    """\
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

    assert (model in ['Radius', 'KNN'])
    if verbose:
        print('------Calculating spatial graph...')
    coor = pd.DataFrame(adata.obsm['spatial'])
    coor.index = adata.obs.index
    coor.columns = ['imagerow', 'imagecol']

    nbrs = sklearn.neighbors.NearestNeighbors(
        n_neighbors=max_neigh + 1, algorithm='ball_tree').fit(coor)
    distances, indices = nbrs.kneighbors(coor)
    if model == 'KNN':
        indices = indices[:, 1:k_cutoff + 1]
        distances = distances[:, 1:k_cutoff + 1]
    if model == 'Radius':
        indices = indices[:, 1:]
        distances = distances[:, 1:]

    KNN_list = []
    for it in range(indices.shape[0]):
        KNN_list.append(pd.DataFrame(zip([it] * indices.shape[1], indices[it, :], distances[it, :])))
    KNN_df = pd.concat(KNN_list)
    KNN_df.columns = ['Cell1', 'Cell2', 'Distance']

    Spatial_Net = KNN_df.copy()
    if model == 'Radius':
        Spatial_Net = KNN_df.loc[KNN_df['Distance'] < rad_cutoff,]
    id_cell_trans = dict(zip(range(coor.shape[0]), np.array(coor.index), ))
    Spatial_Net['Cell1'] = Spatial_Net['Cell1'].map(id_cell_trans)
    Spatial_Net['Cell2'] = Spatial_Net['Cell2'].map(id_cell_trans)
    # self_loops = pd.DataFrame(zip(Spatial_Net['Cell1'].unique(), Spatial_Net['Cell1'].unique(),
    #                  [0] * len((Spatial_Net['Cell1'].unique())))) ###add self loops
    # self_loops.columns = ['Cell1', 'Cell2', 'Distance']
    # Spatial_Net = pd.concat([Spatial_Net, self_loops], axis=0)

    if verbose:
        print('The graph contains %d edges, %d cells.' % (Spatial_Net.shape[0], adata.n_obs))
        print('%.4f neighbors per cell on average.' % (Spatial_Net.shape[0] / adata.n_obs))
    adata.uns['Spatial_Net'] = Spatial_Net

    #########
    if type(adata.X) == np.ndarray:
        X = pd.DataFrame(adata.X[:, ], index=adata.obs.index, columns=adata.var.index)
    else:
        X = pd.DataFrame(adata.X.toarray()[:, ], index=adata.obs.index, columns=adata.var.index)
    cells = np.array(X.index)
    cells_id_tran = dict(zip(cells, range(cells.shape[0])))
    if 'Spatial_Net' not in adata.uns.keys():
        raise ValueError("Spatial_Net is not existed! Run Cal_Spatial_Net first!")

    Spatial_Net = adata.uns['Spatial_Net']
    G_df = Spatial_Net.copy()
    G_df['Cell1'] = G_df['Cell1'].map(cells_id_tran)
    G_df['Cell2'] = G_df['Cell2'].map(cells_id_tran)
    G = sp.coo_matrix((np.ones(G_df.shape[0]), (G_df['Cell1'], G_df['Cell2'])), shape=(adata.n_obs, adata.n_obs))
    G = G + sp.eye(G.shape[0])  # self-loop
    adata.uns['adj'] = G


def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='STAGATE', random_seed=666):
    """\
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    """

    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")

    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']

    res = rmclust(adata.obsm[used_obsm], num_cluster, modelNames)
    mclust_res = np.array(res[-2])

    adata.obs['mclust'] = mclust_res
    adata.obs['mclust'] = adata.obs['mclust'].astype('int')
    adata.obs['mclust'] = adata.obs['mclust'].astype('category')
    return adata


import scipy.sparse as sp


def prepare_graph_data(adj):
    # adapted from preprocess_adj_bias
    num_nodes = adj.shape[0]
    # adj = adj + sp.eye(num_nodes)# self-loop  ##new !!
    # data =  adj.tocoo().data
    # adj[adj > 0.0] = 1.0
    if not sp.isspmatrix_coo(adj):
        adj = adj.tocoo()
    adj = adj.astype(np.float32)
    indices = np.vstack((adj.col, adj.row)).transpose()

    # adj = normalize(adj, norm="l1")

    return (adj, indices, adj.data, adj.shape)


def prune_spatial_Net(Graph_df, label):
    print('------Pruning the graph...')
    print('%d edges before pruning.' % Graph_df.shape[0])
    pro_labels_dict = dict(zip(list(label.index), label))
    Graph_df['Cell1_label'] = Graph_df['Cell1'].map(pro_labels_dict)
    Graph_df['Cell2_label'] = Graph_df['Cell2'].map(pro_labels_dict)
    Graph_df = Graph_df.loc[Graph_df['Cell1_label'] == Graph_df['Cell2_label'],]
    print('%d edges after pruning.' % Graph_df.shape[0])
    return Graph_df


# https://github.com/ClayFlannigan/icp
def best_fit_transform(A, B):
    '''
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
      A: Nxm numpy array of corresponding points
      B: Nxm numpy array of corresponding points
    Returns:
      T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
      R: mxm rotation matrix
      t: mx1 translation vector
    '''

    # assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
        Vt[m - 1, :] *= -1
        R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_B.T - np.dot(R, centroid_A.T)

    # homogeneous transformation
    T = np.identity(m + 1)
    T[:m, :m] = R
    T[:m, m] = t

    return T, R, t

def ICP_align(adata_concat, adata_target, adata_ref, slice_target, slice_ref, landmark_domain, plot_align=False):
    ### find MNN pairs in the landmark domain with knn=1
    adata_slice1 = adata_target[adata_target.obs['louvain'].isin(landmark_domain)]
    adata_slice2 = adata_ref[adata_ref.obs['louvain'].isin(landmark_domain)]

    batch_pair = adata_concat[
        adata_concat.obs['batch_name'].isin([slice_target, slice_ref]) & adata_concat.obs['louvain'].isin(
            landmark_domain)]
    mnn_dict = create_dictionary_mnn(batch_pair, use_rep='STAligner', batch_name='batch_name', k=1, iter_comb=None,
                                     verbose=0)
    adata_1 = batch_pair[batch_pair.obs['batch_name'] == slice_target]
    adata_2 = batch_pair[batch_pair.obs['batch_name'] == slice_ref]

    anchor_list = []
    positive_list = []
    for batch_pair_name in mnn_dict.keys():
        for anchor in mnn_dict[batch_pair_name].keys():
            positive_spot = mnn_dict[batch_pair_name][anchor][0]
            ### anchor should only in the ref slice, pos only in the target slice
            if anchor in adata_1.obs_names and positive_spot in adata_2.obs_names:
                anchor_list.append(anchor)
                positive_list.append(positive_spot)

    batch_as_dict = dict(zip(list(adata_concat.obs_names), range(0, adata_concat.shape[0])))
    anchor_ind = list(map(lambda _: batch_as_dict[_], anchor_list))
    positive_ind = list(map(lambda _: batch_as_dict[_], positive_list))
    anchor_arr = adata_concat.obsm['STAligner'][anchor_ind,]
    positive_arr = adata_concat.obsm['STAligner'][positive_ind,]
    dist_list = [np.sqrt(np.sum(np.square(anchor_arr[ii, :] - positive_arr[ii, :]))) for ii in
                 range(anchor_arr.shape[0])]

    key_points_src = np.array(anchor_list)[dist_list < np.percentile(dist_list, 50)]  ## remove remote outliers
    key_points_dst = np.array(positive_list)[dist_list < np.percentile(dist_list, 50)]
    # print(len(anchor_list), len(key_points_src))

    coor_src = adata_slice1.obsm["spatial"]  ## to_be_aligned
    coor_dst = adata_slice2.obsm["spatial"]  ## reference_points

    ## index number
    MNN_ind_src = [list(adata_1.obs_names).index(key_points_src[ii]) for ii in range(len(key_points_src))]
    MNN_ind_dst = [list(adata_2.obs_names).index(key_points_dst[ii]) for ii in range(len(key_points_dst))]

    ####### ICP alignment
    init_pose = None
    max_iterations = 100
    tolerance = 0.001

    coor_used = coor_src  ## Batch_list[1][Batch_list[1].obs['annotation']==2].obsm["spatial"]
    coor_all = adata_target.obsm["spatial"].copy()
    coor_used = np.concatenate([coor_used, np.expand_dims(np.ones(coor_used.shape[0]), axis=1)], axis=1).T
    coor_all = np.concatenate([coor_all, np.expand_dims(np.ones(coor_all.shape[0]), axis=1)], axis=1).T
    A = coor_src  ## to_be_aligned
    B = coor_dst  ## reference_points

    m = A.shape[1]  # get number of dimensions

    # make points homogeneous, copy them to maintain the originals
    src = np.ones((m + 1, A.shape[0]))
    dst = np.ones((m + 1, B.shape[0]))
    src[:m, :] = np.copy(A.T)
    dst[:m, :] = np.copy(B.T)

    # apply the initial pose estimation
    if init_pose is not None:
        src = np.dot(init_pose, src)
    prev_error = 0

    for ii in range(max_iterations + 1):
        p1 = src[:m, MNN_ind_src].T
        p2 = dst[:m, MNN_ind_dst].T
        T, _, _ = best_fit_transform(src[:m, MNN_ind_src].T,
                                     dst[:m, MNN_ind_dst].T)  ## compute the transformation matrix based on MNNs
        import math
        distances = np.mean([math.sqrt(((p1[kk, 0] - p2[kk, 0]) ** 2) + ((p1[kk, 1] - p2[kk, 1]) ** 2))
                             for kk in range(len(p1))])

        # update the current source
        src = np.dot(T, src)
        coor_used = np.dot(T, coor_used)
        coor_all = np.dot(T, coor_all)

        # check error
        mean_error = np.mean(distances)
        # print(mean_error)
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    aligned_points = coor_used.T  # MNNs in the landmark_domain
    aligned_points_all = coor_all.T  # all points in the slice

    if plot_align:
        import matplotlib.pyplot as plt
        plt.rcParams["figure.figsize"] = (3, 3)
        fig, ax = plt.subplots(1, 2, figsize=(8, 3), gridspec_kw={'wspace': 0.5, 'hspace': 0.1})
        ax[0].scatter(adata_slice2.obsm["spatial"][:, 0], adata_slice2.obsm["spatial"][:, 1],
                      c="blue", cmap=plt.cm.binary_r, s=1)
        ax[0].set_title('Reference ' + slice_ref, size=14)
        ax[1].scatter(aligned_points[:, 0], aligned_points[:, 1],
                      c="blue", cmap=plt.cm.binary_r, s=1)
        ax[1].set_title('Target ' + slice_target, size=14)

        plt.axis("equal")
        # plt.axis("off")
        plt.show()

    # adata_target.obsm["spatial"] = aligned_points_all[:,:2]
    return aligned_points_all[:, :2]


# https://scikit-learn.org/dev/modules/generated/sklearn.neighbors.NearestNeighbors.html#sklearn.neighbors.NearestNeighbors
def nearest_neighbor(src, dst):
    '''
    Find the nearest (Euclidean) neighbor in dst for each point in src
    Input:
        src: Nxm array of points
        dst: Nxm array of points
    Output:
        distances: Euclidean distances of the nearest neighbor
        indices: dst indices of the nearest neighbor
    '''

    # assert src.shape == dst.shape
    neigh = sklearn.neighbors.NearestNeighbors(n_neighbors=1)
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()

def parse_args(yaml_file):
    with open(yaml_file, 'r') as stream:
        try:
            cfg = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return cfg

def select_svgs(smap, domain_id, domain_labels, alpha=1.5):
    """
    Select spatial domain SVGs (spatially variable genes)
    """
    scores = np.linalg.norm(smap.iloc[domain_labels==domain_id, :], axis=0)
    mu, std = np.mean(scores), np.std(scores)
    return smap.columns[scores > mu + alpha * std].tolist()