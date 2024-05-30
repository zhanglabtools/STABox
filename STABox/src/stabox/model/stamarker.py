import os
import glob
import torch
import numpy as np
from tqdm import tqdm
from stabox.model._utils import STAGateModule, StackMLPModule, convert_labels, compute_consensus_matrix, plot_clustered_consensus_matrix
from scipy.cluster import hierarchy
from ._mixin import BaseModelMixin
from torch import nn
import torch.nn.functional as F
from typing import List
from torch.autograd import Variable
import pandas as pd

STAGE1 = "Stage1: Autoencoders trained"
STAGE2 = "Stage2: Consensus labels generated"
STAGE3 = "Stage3: Classifiers trained"


class STAGATECls(nn.Module):
    def __init__(self,
                 satage: STAGateModule = None,
                 classifier: StackMLPModule = None):
        super().__init__()
        self.stagate = satage
        self.classifier = classifier

    def forward(self, x, edge_index, mode="classifier"):
        z, x_recon = self.stagate(x, edge_index)
        z = torch.clone(z)
        if mode == "classifier":
            return z, self.classifier(z)
        elif mode == "reconstruction":
            return z, x_recon
        else:
            raise NotImplementedError

    def get_saliency_map(self, x, edge_index, target_index="max"):
        """
        Get saliency map by backpropagation.
        :param x: input tensors
        :param edge_index: graph edge index
        :param target_index:  target index to compute final scores
        :param save:
        :return: gradients
        """
        x_var = Variable(x, requires_grad=True)
        _, output = self.forward(x_var, edge_index, mode="classifier")
        scores = output["last_layer"]
        if target_index == "max":
            target_score_indices = Variable(torch.argmax(scores, 1))
        elif isinstance(target_index, int):
            target_score_indices = Variable(torch.ones(scores.shape[0], dtype=torch.int64) * target_index)
        else:
            raise NotImplementedError
        target_scores = scores.gather(1, target_score_indices.view(-1, 1)).squeeze()
        loss = torch.sum(target_scores)
        loss.backward()
        gradients = x_var.grad.data
        return gradients, scores


class STAMarker(BaseModelMixin):
    SUPPORTED_TASKS = ["tissue_structure_annotation", "spatial_embedding", "enhanced_gene_expression",
                       "SVG_identification"]
    METHOD_NAME = "STAMarker"

    """
    STAMarker for identifying spatial domain-specific spatially variable genes.
    model_dir: The directory to save the model.
    in_features: The number of input features.
    hidden_dims: The list of hidden dimensions.
    n_models: The number of models to train.


    Ref:  https://doi.org/10.1093/nar/gkad801
    """

    def __init__(self,
                 model_dir: str,
                 in_features: int,
                 hidden_dims: List[int],
                 n_models: int = 5,
                 device: torch.device = torch.device("cpu"),
                 **kwargs):
        super().__init__(model_dir, in_features, hidden_dims, **kwargs)
        self._check_validity()
        self.n_models = n_models
        self.latent_dim = hidden_dims[-1]
        self.train_status = []
        self.device = device
        self.model = None
        self.model_dir = model_dir

    def _check_validity(self):
        assert set(self.SUPPORTED_TASKS).issubset(set(BaseModelMixin.SUPPORTED_TASKS)) and len(self.SUPPORTED_TASKS) > 0

    def train(self, adata, lr=1e-4, n_epochs=500, gradient_clip=5.0,
              cluster_method="mclust",
              n_clusters=7,
              resolution=0.8,
              cluster_seed=2023,
              clf_lr=1e-3,
              clf_n_epochs=300,
              verbose=True,
              use_net='spatial_net',
              resume=True,
              plot_consensus=False,
              **kwargs):
        # -- check if the autoencoders dir exists
        if self.METHOD_NAME not in adata.uns:
            adata.uns["STAMarker"] = dict()
        if not os.path.exists(os.path.join(self.model_dir, "autoencoders")):
            os.makedirs(os.path.join(self.model_dir, "autoencoders"))
        # check if the classifier dir exists
        if not os.path.exists(os.path.join(self.model_dir, "classifiers")):
            os.makedirs(os.path.join(self.model_dir, "classifiers"))
        if resume:
            # check if there are trained autoencoders
            autoencoder_paths = glob.glob(os.path.join(self.model_dir, "autoencoders", "*.pth"))
            if len(autoencoder_paths) == self.n_models:
                self.train_status.append(STAGE1)
            # check if there are consensus clustering labels `consensus_labels.npy`
            if "consensus_labels" in adata.uns[self.METHOD_NAME].keys():
                self.train_status.append(STAGE2)
            if STAGE1 in self.train_status and STAGE2 in self.train_status:
                # check if there are trained classifiers
                classifier_paths = glob.glob(os.path.join(self.model_dir, "classifiers", "*.pth"))
                if len(classifier_paths) == self.n_models:
                    self.train_status.append(STAGE3)
            # sort the train_status by the order of stages
            self.train_status = sorted(self.train_status, key=lambda x: [STAGE1, STAGE2, STAGE3].index(x))
        else:
            self.train_status = []
        print("Starting training at status {}...".format(self.train_status))
        data = self.prepare_data(adata, use_net=use_net)
        if len(self.train_status) == 0:
            if verbose:
                print("------Stage 1: Autoencoders training...")
            if not resume:
                model_index = 0
            else:
                model_index = len(glob.glob(os.path.join(self.model_dir, "autoencoders", "*.pth")))
            pbar = tqdm(total=self.n_models)
            for model_index in range(model_index, self.n_models):
                autoencoder = self._train_autoencoder(adata, lr=lr, n_epochs=n_epochs, gradient_clip=gradient_clip,
                                                      model_index=model_index, pbar=pbar, data=data)
                pbar.update(1)
                torch.save(autoencoder,
                           os.path.join(self.model_dir, "autoencoders", "autoencoder_%d.pth" % model_index))
            self.train_status.append(STAGE1)
        # -- Stage 2: Consensus labels generation
        if len(self.train_status) == 1:
            if verbose:
                print("------Stage 2: Consensus labels generation...")
            pbar = tqdm(total=self.n_models + 1)
            cluster_res_list = []
            for model_index in range(self.n_models):
                rep = self.get_rep(data, model_index)
                if cluster_method == "mclust":
                    from ._utils import mclust_R
                    cluster_res = mclust_R(rep, n_clusters=n_clusters, r_seed=cluster_seed)
                    cluster_res = convert_labels(cluster_res)
                elif cluster_method == "louvain":
                    from ._utils import louvain
                    cluster_res = louvain(rep, resolution=resolution, r_seed=cluster_seed)
                else:
                    raise NotImplementedError("Unknown cluster method: {}".format(cluster_method))
                # convert cluster_res to 0 to n_clusters-1
                cluster_res_list.append(cluster_res)
                pbar.update(1)
            # save cluster_res_list to adata
            adata.uns[self.METHOD_NAME]["cluster_res"] = cluster_res_list
            # consensus clustering
            consensus_labels = self.consensus_clustering(cluster_res_list, n_clusters=n_clusters, plot=plot_consensus)
            # save consensus labels
            adata.uns[self.METHOD_NAME]["consensus_labels"] = consensus_labels
            self.train_status.append(STAGE2)
        # -- Stage 3: Classifiers training
        if len(self.train_status) == 2:
            if verbose:
                print("------Stage 3: Classifiers training...")
            pbar = tqdm(total=self.n_models)
            for model_index in range(self.n_models):
                rep = self.get_rep(data, model_index)
                consensus_labels = adata.uns[self.METHOD_NAME]["consensus_labels"]
                classifier = self._train_classifier(rep, model_index, n_clusters=n_clusters, seed=cluster_seed,
                                                    consensus_labels=consensus_labels, pbar=pbar)
                torch.save(classifier,
                           os.path.join(self.model_dir, "classifiers", "classifier_%d.pth" % model_index))
                pbar.update(1)
            self.train_status.append(STAGE3)

    def get_rep(self, data, model_index):
        autoencoder = torch.load(
            os.path.join(self.model_dir, "autoencoders", "autoencoder_%d.pth" % model_index))
        autoencoder.eval()
        with torch.no_grad():
            rep, x_hat = autoencoder(data.x, data.edge_index)
        rep = rep.cpu().numpy()
        return rep

    def _train_autoencoder(self, adata, lr=1e-4, n_epochs=500, gradient_clip=5.0, data=None, seed=None,
                           model_index=None, pbar=None,
                           **kwargs):
        autoencoder = STAGateModule(self.in_features, self.hidden_dims).to(self.device)
        self.model = autoencoder
        optimizer = torch.optim.Adam(autoencoder.parameters(), lr=lr)
        if data is None:
            data = self.prepare_data(adata)
        if seed is not None:
            torch.manual_seed(seed)
        for epoch in range(n_epochs):
            autoencoder.train()
            optimizer.zero_grad()
            rep, x_hat = autoencoder(data.x, data.edge_index)
            loss = F.mse_loss(data.x, x_hat)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(autoencoder.parameters(), gradient_clip)
            optimizer.step()
            if pbar is not None:
                pbar.set_description(
                    "Train autoencoder %d Epoch: %d/%d, Loss: %.4f" % (model_index, epoch, n_epochs, loss.item()))
        return autoencoder

    def _train_classifier(self,
                          rep: np.ndarray,
                          model_index: int,
                          n_clusters: int = None,
                          consensus_labels: np.ndarray = None,
                          lr=1e-3,
                          n_epochs=300,
                          seed=None,
                          pbar=None,
                          **kwargs):
        classifier = StackMLPModule(self.latent_dim, n_clusters).to(self.device)
        data = torch.from_numpy(rep).float().to(self.device)
        optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)
        target_labels = torch.from_numpy(consensus_labels).long().to(self.device)
        if seed is not None:
            torch.manual_seed(seed)
        for epoch in range(n_epochs):
            classifier.train()
            optimizer.zero_grad()
            logits = classifier(data)
            loss = F.cross_entropy(logits["score"], target_labels)
            loss.backward()
            optimizer.step()
            if pbar is not None:
                pbar.set_description(
                    "Train classifier %d Epoch: %d/%d, Loss: %.4f" % (model_index, epoch, n_epochs, loss.item()))
        return classifier

    def consensus_clustering(self, labels_list, n_clusters, plot=False):
        labels_list = np.vstack(labels_list)
        import time
        st = time.time()
        cons_mat = compute_consensus_matrix(labels_list)
        print("Compute consensus matrix. Elapsed: {:.2f}\n".format(time.time() - st))
        st = time.time()
        if plot:
            figure, consensus_labels = plot_clustered_consensus_matrix(cons_mat, n_clusters)
            figure.savefig(os.path.join(self.model_dir,
                                        "consensus_clustering_{}_clusters.png".format(n_clusters)), dpi=300)
            print("Plot consensus map. Elapsed: {:.2f}\n".format(time.time() - st))
            # delete the figure
            del figure
        else:
            linkage_matrix = hierarchy.linkage(cons_mat, method='average', metric='euclidean')
            consensus_labels = hierarchy.fcluster(linkage_matrix, n_clusters, criterion='maxclust')
        consensus_labels = convert_labels(consensus_labels)
        return consensus_labels

    def predict(self, adata, add_key="smap", use_net="spatial_net",
                target_index="max", reduction="mean", return_all_smaps=False,
                **kwargs):
        assert(len(self.train_status) == 3)
        #  check the number of autoencoders and classifiers == n_models
        if len(glob.glob(os.path.join(self.model_dir, "autoencoders", "*.pth"))) != self.n_models:
            raise ValueError("The number of autoencoders is not equal to n_models!. Please retrain the model.")
        pbar = tqdm(total=self.n_models)
        # get data
        data = self.prepare_data(adata, use_net=use_net)
        smaps = []
        smaps_red = 0
        # get data
        for model_index in range(self.n_models):
            # load autoencoder
            autoencoder = torch.load(os.path.join(self.model_dir, "autoencoders", "autoencoder_%d.pth" % model_index))
            autoencoder.eval()
            # load classifier
            classifier = torch.load(os.path.join(self.model_dir, "classifiers", "classifier_%d.pth" % model_index))
            classifier.eval()
            # get stacked model
            model = STAGATECls(autoencoder, classifier).to(self.device)
            model.eval()
            # get smap
            smap, _ = model.get_saliency_map(data.x, data.edge_index, target_index="max")
            smap = smap.detach().cpu().numpy()
            if return_all_smaps:
                smaps.append(smap)
            if reduction == "mean":
                smaps_red += smap
            else:
                raise NotImplementedError("Unknown reduction method: {}".format(reduction))
            pbar.set_description("Compute sailency map model %d" % (model_index + 1))
            pbar.update(1)
        pbar.close()
        if reduction == "mean":
            smaps_red = smaps_red / self.n_models
        else:
            raise NotImplementedError("Unknown reduction method: {}".format(reduction))
        adata.obsm[add_key] = smaps_red
        if return_all_smaps:
            return smaps

    def select_spatially_variable_genes(self, adata, use_smap="smap", alpha=1.5, transform="log"):
        """
        Select spatially variable genes based on the saliency map.
        :param adata: AnnData object after training and prediction
        :param use_smap: used saliency map in adata.obsm
        :param alpha: threshold to select genes
        :param transform: log or None
        :return: dictionary of output
            sailency_scores: (n_genes, n_spatial_domains), sailency scores of all genes
            gene_df: (n_genes, n_spatial_domains), boolean matrix of selected genes
            gene_list: selected genes list combined from all spatial domains
        """
        smap = adata.obsm[use_smap]
        labels = adata.uns[self.METHOD_NAME]["consensus_labels"]
        unique_labels = np.unique(labels)
        scores = pd.DataFrame(index=adata.var.index)
        for label in unique_labels:
            scores["spatial domain {}".format(label)] =np.linalg.norm(smap[labels == label, :], axis=0)
        if transform == "log":
            scores = np.log(scores)
        genes_df = scores.apply(lambda x: x > x.mean() + alpha * x.std(), axis=0)
        gene_list = genes_df.index[genes_df.sum(axis=1) > 0].tolist()
        output = {"saliency_scores": scores, "gene_list": gene_list, "gene_df": genes_df}
        return output
