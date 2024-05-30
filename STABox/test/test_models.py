import sys
import torch
import scanpy as sc
import os
os.environ["R_HOME"] = r"C:\\Program Files\\R\\R-4.3.0"
os.environ["PATH"]   = r"C:\\Program Files\\R\\R-4.3.0\\bin\x64" + ";" + os.environ["PATH"]
os.environ["R_USER"] = "C:\\Users\\001\\anaconda3\\envs\\STADIffusion\\Lib\\site-packages\\rpy2"
# add the path of the package to sys.path
sys.path.append("./src")
# set R path
from stabox.model import STAMarker
from stabox.model import STAGATE
from stabox.model._utils import Cal_Spatial_Net, Stats_Spatial_Net


adata = sc.read_h5ad("./datasets/DLPFC_151507.h5ad")

# STAGATE test
Cal_Spatial_Net(adata, rad_cutoff=150)
Stats_Spatial_Net(adata)
STAGATE = STAGATE(model_dir="./result/DLPFC_151507",
                  in_features=3000, hidden_dims=[512, 30])
adata = STAGATE.train(adata)

# STAMarker test
stamarker = STAMarker(model_dir="./result/DLPFC_151507",
                      in_features=3000, hidden_dims=[512, 30],
                      n_models=3, device=torch.device("cuda:0"))
stamarker.train(adata, lr=1e-4, n_epochs=10, gradient_clip=5.0, use_net="Spatial_Net",
                resume=False, plot_consensus=True, n_clusters=7)
stamarker.predict(adata, use_net="Spatial_Net")
output = stamarker.select_spatially_variable_genes(adata, use_smap="smap", alpha=1.5)

# STAligner test
