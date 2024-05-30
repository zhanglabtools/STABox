import pandas as pd
import numpy as np
from scipy.spatial import cKDTree


def _cal_spatial_net(spatial_df, rad_cutoff=None, k_cutoff=None, verbose=True):
    """
    Construct the spatial neighbor networks from the spatial coordinates dataframe.

    Parameters
    ----------
    spatial_df: pd.DataFrame
        The spatial coordinates dataframe with index cell names and columns for coordinates.
    rad_cutoff: float
        The radius cutoff when model="Radius"
    k_cutoff: int
        The number of nearest neighbors when model="KNN"
    verbose: bool
        Whether to print the information of the spatial network
    """
    assert rad_cutoff is not None or k_cutoff is not None, "Either rad_cutoff or k_cutoff must be provided"
    assert rad_cutoff is None or k_cutoff is None, "Only one of rad_cutoff and k_cutoff must be provided"
    if verbose:
        print('------Calculating spatial graph...')

    coor = spatial_df.values
    tree = cKDTree(coor)
    if rad_cutoff is not None:
        indices = tree.query_ball_point(coor, r=rad_cutoff)
    else:
        distances, indices = tree.query(coor, k=k_cutoff + 1)

    # construct the spatial df
    spatial_net_data = []
    for i, neighbors in enumerate(indices):
        cell1 = spatial_df.index[i]
        for j in neighbors:
            if i != j:  # Avoid self-loop
                cell2 = spatial_df.index[j]
                if rad_cutoff is not None:
                    distance = np.linalg.norm(coor[i] - coor[j])
                else:
                    distance = distances[i, j]
                spatial_net_data.append((cell1, cell2, distance))

    spatial_net = pd.DataFrame(spatial_net_data, columns=['Cell1', 'Cell2', 'Distance'])
    # add the edge type information "within"
    spatial_net['EdgeType'] = 'within'
    if verbose:
        print('------Spatial graph calculated.')
        print('The graph contains %d edges, %d cells, %.4f neighbors per cell on average.' % (
            spatial_net.shape[0], spatial_df.shape[0], spatial_net.shape[0] / spatial_df.shape[0]))
    return spatial_net


def cal_spatial_net2D(adata, rad_cutoff=None, k_cutoff=None, use_obsm="spatial",
                      add_key="Spatial_Net", verbose=True):
    spatial_df = pd.DataFrame(adata.obsm[use_obsm])
    spatial_df.index = adata.obs.index
    spatial_net = _cal_spatial_net(spatial_df, rad_cutoff=rad_cutoff, k_cutoff=k_cutoff,
                                   verbose=verbose)
    adata.uns[add_key] = spatial_net
    return adata


def _cal_spatial_bipartite(spatial_df1, spatial_df2, rad_cutoff=None, k_cutoff=None, verbose=True):
    """
    Construct the spatial neighbor across two spatial coordinates dataframe.

    Parameters
    ----------
    spatial_df1: pd.DataFrame
        The spatial coordinates dataframe with index cell names and columns for coordinates.
    spatial_df2: pd.DataFrame
        The spatial coordinates dataframe with index cell names and columns for coordinates.
    model: str
        "Radius" or "KNN"
    rad_cutoff: float
        The radius cutoff when model="Radius"
    k_cutoff: int
        The number of nearest neighbors when model="KNN"
    verbose: bool
        Whether to print the information of the spatial network
    """
    # only of rad_cutoff and k_cutoff must be provided
    assert rad_cutoff is not None or k_cutoff is not None, "Either rad_cutoff or k_cutoff must be provided"
    assert rad_cutoff is None or k_cutoff is None, "Only one of rad_cutoff and k_cutoff must be provided"
    if verbose:
        print('------Calculating spatial bipartite graph...')

    coor1 = spatial_df1.values
    coor2 = spatial_df2.values

    tree1 = cKDTree(coor1)
    tree2 = cKDTree(coor2)

    if rad_cutoff is not None:
        indices = tree1.query_ball_tree(tree2, r=rad_cutoff)  # indices is a list of lists in spatial_df2
    else:
        distances, indices = tree2.query(coor1, k=k_cutoff + 1)

    # construct the spatial bipartite df
    spatial_bipartite_data = []
    for i, neighbors in enumerate(indices):
        cell1 = spatial_df1.index[i]
        for j in neighbors:
            cell2 = spatial_df2.index[j]
            distance = np.linalg.norm(coor1[i] - coor2[j])
            spatial_bipartite_data.append((cell1, cell2, distance))

    spatial_bipartite = pd.DataFrame(spatial_bipartite_data, columns=['Cell1', 'Cell2', 'Distance'])
    # add the edge type iinformation "across"
    spatial_bipartite['EdgeType'] = 'across'
    if verbose:
        print('------Spatial bipartite graph calculated.')
        print('The graph contains %d edges, %d cells, %.4f neighbors per cell on average.' \
              % (spatial_bipartite.shape[0], spatial_df1.shape[0] + spatial_df2.shape[0],
                 spatial_bipartite.shape[0] / (spatial_df1.shape[0])))
    return spatial_bipartite


def cal_spatial_net3D(adata, batch_id=None, iter_comb=None, rad_cutoff=None, k_cutoff=None,
                      z_rad_cutoff=None, z_k_cutoff=None, use_obsm="spatial", add_key="Spatial_Net",
                      verbose=True):
    """
    Calculate the spatial network for 3D data.
    First, calculate the spatial network for each layer.
    Then, calculate the spatial bipartite network between layers.
    Finally, combine the two networks.
    """
    assert batch_id is not None, "batch_id must be provided"
    # assert iter_comb is not None, "iter_comb must be provided"
    batch_list = adata.obs[batch_id].unique()
    # iter_comb must be a list of tuples with length 2 and each tuple contains two batch ids belonging to the batch_list
    if iter_comb is not None:
        assert all([len(comb) == 2 and comb[0] in batch_list and comb[1] in batch_list for comb in iter_comb]), \
            "`iter_comb` must be a list of tuples with length 2 and each tuple contains two batch ids belonging to the " \
            "batch_list"
    if verbose:
        print("------Calculating spatial network for each batch...")
    spatial_net_list = []
    for batch in batch_list:
        if verbose:
            print(f"Calculating spatial network for batch {batch}...")
        adata_batch = adata[adata.obs[batch_id] == batch]
        spatial_df = pd.DataFrame(adata_batch.obsm[use_obsm])
        spatial_df.index = adata_batch.obs.index
        spatial_net = _cal_spatial_net(spatial_df, rad_cutoff=rad_cutoff, k_cutoff=k_cutoff, verbose=verbose)
        spatial_net_list.append(spatial_net)
    # construct the spatial bipartite network
    if verbose:
        print("------Calculating spatial bipartite network...")
    spatial_bipartite_list = []
    if iter_comb is not None:
        for comb in iter_comb:
            if verbose:
                print(f"Calculating spatial bipartite network for {comb}...")
            adata1 = adata[adata.obs[batch_id] == comb[0]]
            adata2 = adata[adata.obs[batch_id] == comb[1]]
            spatial_df1 = pd.DataFrame(adata1.obsm[use_obsm])
            spatial_df1.index = adata1.obs.index
            spatial_df2 = pd.DataFrame(adata2.obsm[use_obsm])
            spatial_df2.index = adata2.obs.index
            spatial_bipartite = _cal_spatial_bipartite(spatial_df1, spatial_df2, rad_cutoff=z_rad_cutoff,
                                                       k_cutoff=z_k_cutoff, verbose=verbose)
            spatial_bipartite_list.append(spatial_bipartite)
    # combine the spatial network and spatial bipartite network
        spatial_net = pd.concat(spatial_net_list + spatial_bipartite_list, axis=0)
    else:
        spatial_net = pd.concat(spatial_net_list, axis=0)
    if verbose:
        print("------Spatial network calculated.")
    adata.uns[add_key] = pd.DataFrame(spatial_net)
    return adata