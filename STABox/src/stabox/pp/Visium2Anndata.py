import scanpy as sc
import numpy as np
import pandas as pd
import os
import glob


def single_10Xvisium(path: str, label=False):
    data_file = glob.glob(path + '/*.h5')
    truth_file = glob.glob(path + '/*.txt')
    truth_file_csv = glob.glob(path + '/*.csv')
    data_name = data_file[0].rsplit("\\", 1)[-1]
    adata = sc.read_visium(path=path, count_file=data_name)
    adata.var_names_make_unique()

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    if len(truth_file) > 0 and label:
        Ann_df = pd.read_csv(os.path.join(path, truth_file[0].rsplit('\\', 1)[-1]), sep='\t',
                             header=None, index_col=0)
        Ann_df.columns = ['Ground Truth']
        adata.obs['GroundTruth'] = Ann_df.loc[adata.obs_names, 'Ground Truth']

        adata.obs.GroundTruth = adata.obs.GroundTruth.astype(str)
        adata.obs['GroundTruth'] = adata.obs['GroundTruth']
    if len(truth_file_csv) > 0 and label:
        Ann_df = pd.read_csv(os.path.join(path, truth_file_csv[0].rsplit('\\', 1)[-1]), sep=',',
                             index_col=0)
        Ann_df.columns = ['Ground Truth']
        adata.obs['GroundTruth'] = Ann_df.loc[adata.obs_names, 'Ground Truth']
    return adata