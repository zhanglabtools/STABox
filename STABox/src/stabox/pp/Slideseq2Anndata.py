import scanpy as sc
import numpy as np
import pandas as pd
import os
import glob


def Slideseq_tsv_txt_file_to_h5ad(path):
    try:
        used_barcodes = glob.glob(path + '/*.txt')
        data_file = glob.glob(path + '/*.tsv')
        file_one = data_file[0].rsplit("\\", 1)[-1]
        file_two = data_file[1].rsplit("\\", 1)[-1]
        fsize_one = os.path.getsize(os.path.join(path, file_one))
        fsize_two = os.path.getsize(os.path.join(path, file_two))

        if fsize_one > fsize_two:
            data_name = file_one
            location_file_name = file_two
        else:
            data_name = file_two
            location_file_name = file_one

        counts = pd.read_csv(os.path.join(path, data_name), sep='\t', index_col=0)
        print(counts.shape)
        coor_df = pd.read_csv(os.path.join(path, location_file_name), sep='\t')
        print(coor_df.shape)

        counts.columns = ['Spot_' + str(x) for x in counts.columns]
        coor_df.index = coor_df['label'].map(lambda x: 'Spot_' + str(x))
        coor_df = coor_df.loc[:, ['x', 'y']]

        adata = sc.AnnData(counts.T)
        adata.var_names_make_unique()

        coor_df = coor_df.loc[adata.obs_names, ['y', 'x']]
        adata.obsm["spatial"] = coor_df.to_numpy()
        if len(used_barcodes) != 0:
            used_barcode = pd.read_csv(used_barcodes[0], sep='\t', header=None)
            used_barcode = used_barcode[0]
            adata = adata[used_barcode,]

        sc.pp.calculate_qc_metrics(adata, inplace=True)
        sc.pp.filter_genes(adata, min_cells=50)
        sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        adata.write_h5ad(path + 'adata.h5ad')

    except:
        print("Make sure your file exist!")


def Slideseq_txt_file_to_h5ad(path: str, label):
    data_file = glob.glob(path + '/*.txt')
    file_one = data_file[0].rsplit("\\", 1)[-1]
    file_two = data_file[1].rsplit("\\", 1)[-1]
    fsize_one = os.path.getsize(os.path.join(path, file_one))
    fsize_two = os.path.getsize(os.path.join(path, file_two))

    if fsize_one > fsize_two:
        exprefile = file_one
        coorfile = file_two
    else:
        exprefile = file_two
        coorfile = file_one

    data = pd.read_csv(os.path.join(path, exprefile), sep='\t')
    coor_df = pd.read_csv(os.path.join(path, coorfile), sep='\t')
    adata = sc.AnnData(data)

    adata.obsm["spatial"] = coor_df.to_numpy()
    adata.write_h5ad(path + 'adata.h5ad')
    return adata


def Slideseq_txt_csv_file_to_h5ad(path):

    count_file = glob.glob(path + '/*.txt')
    location_file = glob.glob(path + '/*.csv')

    counts = pd.read_csv(count_file[0], sep='\t', index_col=0)
    coor_df = pd.read_csv(location_file[0], sep=',', index_col=0)
    adata = sc.AnnData(counts.T)
    adata.var_names_make_unique()
    coor_df = coor_df.loc[adata.obs_names, ['xcoord', 'ycoord']]
    adata.obsm["spatial"] = coor_df.to_numpy()

    if os.path.exists(path + '/used_barcodes.txt'):
        used_barcode = pd.read_csv(path + '/used_barcodes.txt', sep='\t', header=None)
        used_barcode = used_barcode[0]
        adata = adata[used_barcode,]

    sc.pp.calculate_qc_metrics(adata, inplace=True)
    sc.pp.filter_genes(adata, min_cells=50)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.write_h5ad(path + 'adata.h5ad')
    return adata
