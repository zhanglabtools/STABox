import os
import torch
import numpy as np
import scipy.sparse as sp
from torch_geometric.data import Data
import torch.nn.functional as F
from typing import Any, List
from scanpy import AnnData
import tqdm
from ._utils import Transfer_pytorch_Data


class BaseModelMixin:
    METHOD_NAME = "BaseModel"
    SUPPORTED_TASKS = ["tissue_structure_annotation", "spatial_embedding", "enhanced_gene_expression",
                       "3D_reconstruction", "SVG_identification"]

    def __init__(self,
                 model_dir: str,
                 in_features: int,
                 hidden_dims: List[int],
                 device: str = 'cpu',
                 **kwargs):
        self.model_dir = model_dir
        self.model = None
        self.in_features = in_features
        self.hidden_dims = hidden_dims
        self.device = device
        # check if the model directory exists
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

    def _check_validity(self):
        """
        Check if the arguments are valid else raise exceptions.
        """
        pass

    def prepare_data(self, adata: AnnData,
                        use_rep=None,
                        use_spatial='spatial',
                        use_net='Spatial_Net',
                        **kwargs):
        G_df = adata.uns[use_net].copy()
        cells = np.array(adata.obs_names)
        cells_id_tran = dict(zip(cells, range(cells.shape[0])))
        G_df['Cell1'] = G_df['Cell1'].map(cells_id_tran)
        G_df['Cell2'] = G_df['Cell2'].map(cells_id_tran)
        G = sp.coo_matrix((np.ones(G_df.shape[0]), (G_df['Cell1'], G_df['Cell2'])), shape=(adata.n_obs, adata.n_obs))
        G = G + sp.eye(G.shape[0])
        if use_rep is not None:
            x = adata.obsm[use_rep]
        else:
            x = adata.X
        if sp.issparse(x):
            x = x.todense()
        spatial = adata.obsm[use_spatial]
        edge_list = np.nonzero(G)
        data = Data(edge_index=torch.LongTensor(np.array([edge_list[0], edge_list[1]])),
                    x=torch.FloatTensor(x))
        return data.to(self.device)

    def save(self, name, **kwargs):
        model_path = os.path.join(self.model_dir, self.model_name + '_' + name + '.pth')
        torch.save(self.model.state_dict(), model_path)

    def load(self, name, **kwargs):
        model_path = os.path.join(self.model_dir, self.model_name + '_' + name + '.pth')
        self.model.load_state_dict(torch.load(model_path))

    

    def train(self, adata, lr=1e-4, n_epochs=500, gradient_clip=5.0, methods=None, **kwargs):
        """
        Train the model on the given AnnData object and return the trained model.
        AnnData object should be properly preprocessed.
        For example, if the model use GNN model, AnnData object should have constructed graphs.
        :param kwargs:
        :return:
        """
        pass

    def predict(self, adata, add_key=None, **kwargs):
        """
        Predict given the AnnData object and return the annotated AnnData object.
        AnnData object should be properly preprocessed.
        For example, if for STAGate model, AnnData object should have constructed graphs and will be used to compute
        the latent embeddings of the spots and add to `adata.obsm[add_key].
        :param adata: AnnData object to predict.
        :add_key: If not None, add the predicted result to the AnnData object with the given key.
        :return: Annotated AnnData object.
        """
        pass

