import random
import re
import subprocess
import warnings
import sys

# from concurrent.futures import ThreadPoolExecutor
# from typing import Callable
# from tqdm import tqdm
import matplotlib
import tkinter as tk
import cv2
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from scipy.sparse import issparse, csr_matrix
import anndata
import yaml

sys.path.append("..")
plt.rc("font", family="Arial")

from upsetplot import plot, from_contents
import gseapy as gp
from gseapy import barplot, dotplot

matplotlib.use("Agg")
warnings.filterwarnings("ignore")
from datetime import datetime
import ttkbootstrap as ttk
from ttkbootstrap.style import Bootstyle
from tkinter.filedialog import askdirectory
from ttkbootstrap.dialogs import Messagebox
from ttkbootstrap.constants import *
from ttkbootstrap import Style
from pathlib import Path
import threading
import queue
import shutil

from tkinter import colorchooser, filedialog
import glob
from . import color_panel
import pandas as pd
import numpy as np
import scanpy as sc
import os
from sklearn.metrics.cluster import adjusted_rand_score
from ..model import STAGATE
from sklearn.cluster import KMeans
import scipy.sparse as sp
import scipy.linalg
import torch
import anndata as ad
from ..pl.utils import (
    Cal_Spatial_Net,
    Stats_Spatial_Net,
    Cal_Spatial_Net_3D,
    mclust_R,
    Cal_Spatial_Net_new,
    parse_args,
    select_svgs,
)
from ..model import STAligner
from ..model import STAMarker
from ..extension.STAGE import STAGE
from ..module_3D.DLPFC_Data import webcache_main, webServer
from ..extension import SpaGCN as spg
from ..extension.SEDR.graph_func import graph_construction
from ..extension.SEDR.utils_func import adata_preprocess
from ..extension.SEDR.SEDR_train import SEDR_Train
from ..extension.SEDR.SEDR_parameter import params, res_search_fixed_clus
from ..dataset.SODB import SODB
from ..dataset.SpatiaIDB import SpatialDB
import json
import tkinter
import tkinter.ttk as ttks
from tkinter import messagebox

seed = 666
import torch.backends.cudnn as cudnn

cudnn.deterministic = True
cudnn.benchmark = True
import random

random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
global methods
global data_type
global image_files
image_files = {
    "properties-dark": "icons8_settings_24px.png",
    "properties-light": "icons8_settings_24px_2.png",
    "add-to-backup-dark": "icons8_add_folder_24px.png",
    "add-to-backup-light": "icons8_add_book_24px.png",
    "stop-backup-dark": "icons8_cancel_24px.png",
    "stop-backup-light": "icons8_cancel_24px_1.png",
    "play": "icons8_play_24px_1.png",
    "refresh": "icons8_refresh_24px_1.png",
    "stop-dark": "icons8_stop_24px.png",
    "stop-light": "icons8_stop_24px_1.png",
    "opened-folder": "icons8_opened_folder_24px.png",
    "data_add": "icons8_opened_folder_24px_1.png",
    "logo": "backup.png",
}
methods = [
    "STAGATE",
    "STAligner",
    "STAMarker",
    "STAGE",
    "SpaGCN",
    "SEDR",
    "SCANPY",
    "STAMapper",
    "STALocator",
]
data_type = [
    "10x",
    "Slide-seqV2",
    "Stereo-seq",
    "MERFISH",
    "Slide-seq",
    "ST",
    "STARmap",
    "HDST",
    "H5AD-files",
    "Multi-files",
]

PATH = Path(__file__).parent / "assets"
Raw_PATH = Path(__file__).parent

current_file_path = os.path.abspath(__file__)
test_file_path = os.path.dirname(current_file_path)
running_path = os.path.dirname(test_file_path)
print(f"running_path={running_path}")
print(f"current_file_path={current_file_path}")
print(f"test_file_path={test_file_path}")


class Tooltip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tooltip = None

        widget.bind("<Enter>", self.show_tooltip)
        widget.bind("<Leave>", self.hide_tooltip)

    def show_tooltip(self, event):
        x = y = 0
        x, y, _, _ = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 25

        self.tooltip = tkinter.Toplevel(self.widget)
        self.tooltip.wm_overrideredirect(True)
        self.tooltip.wm_geometry(f"+{x}+{y}")

        label = ttks.Label(self.tooltip, text=self.text)
        label.pack()

    def hide_tooltip(self, event):
        if self.tooltip:
            self.tooltip.destroy()
            self.tooltip = None


class STABox(ttk.Frame):

    def __init__(self, master=None, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        self.master = master
        self.pack(fill=BOTH, expand=YES)
        self.color_reset = None
        self.gene_color_type = None
        self.result_flag = False
        self.model_train_flag = False
        self.cluster_flag = False

        self.StartTime = None
        self.data_load_flag = False
        self.path_load_flag = False
        self.multi_files = None
        self.file_path = None

        self.rad_cutoff_value = None
        self.alpha_value = None
        self.cluster_value = None
        self.genes = None
        self.detail_info = None
        self.functions_choose = None
        self.adjust_value = None
        self.show_gene_name = None
        self.show_gene_num = None

        self.data_type = None
        self.upload_file_path = None
        self.label_files_exit = False
        self.spot_size = None
        self.louvain_res = None
        self.kmeans_num = None
        self.Mcluster_num = None
        self.train_epoch_num = None
        self.cla_epoch_num = None

        self.result_queue = queue.Queue()
        self.flag_queue = queue.Queue()
        self.method_flag = ""
        self.trained_file_path = None

        self.photoimages = []
        self.GUI()

    def GUI(self):
        # imgpath = 'VIEW/assets'
        imgpath = test_file_path + "/assets"
        for key, val in image_files.items():
            _path = imgpath + "/" + val
            self.photoimages.append(ttk.PhotoImage(name=key, file=_path))

        buttonbar = ttk.Frame(self, style="primary.TFrame")
        buttonbar.pack(fill=X, pady=1, side=TOP)
        sty = ttk.Style()
        sty.configure("my.TButton", font="Arial")

        self.downloadbtn = ttk.Button(
            master=buttonbar,
            text="Download",
            image="data_add",
            compound=LEFT,
            style="my.TButton",
            command=self.data_download_GUI,
        )
        self.downloadbtn.pack(side=LEFT, ipadx=5, ipady=5, padx=0, pady=1)

        self.backup = ttk.Button(
            master=buttonbar,
            text="Load",
            image="add-to-backup-light",
            compound=LEFT,
            style="my.TButton",
            command=self.Data_Preprocess_thread,  # self.Data_Preprocess_thread
        )
        self.backup.pack(side=LEFT, ipadx=5, ipady=5, padx=(1, 0), pady=1)

        btn = ttk.Button(
            master=buttonbar,
            text="Preprocess",
            image="refresh",
            compound=LEFT,
            style="my.TButton",
            command=self.Preprocess,
        )
        btn.pack(side=LEFT, ipadx=5, ipady=5, padx=0, pady=1)

        btn = ttk.Button(
            master=buttonbar,
            text="Run",
            image="play",
            compound=LEFT,
            style="my.TButton",
            command=self.Data_process,
        )
        btn.pack(side=LEFT, ipadx=5, ipady=5, padx=0, pady=1)

        btn = ttk.Button(
            master=buttonbar,
            text="Restart",
            image="stop-light",
            compound=LEFT,
            style="my.TButton",
            command=self.Restart,
        )
        btn.pack(side=LEFT, ipadx=5, ipady=5, padx=0, pady=1)

        btn = ttk.Button(
            master=buttonbar,
            text="Settings",
            image="properties-light",
            compound=LEFT,
            style="my.TButton",
            command=self.settings,
        )
        btn.pack(side=LEFT, ipadx=5, ipady=5, padx=0, pady=1)

        self.left_panel = ttk.Frame(self, style="bg.TFrame")
        self.left_panel.pack(side=LEFT, fill=Y)

        self.bus_cf = CollapsingFrame(self.left_panel)
        self.bus_cf.pack(fill=X, pady=1)

        self.file_path_frm = ttk.Frame(self.bus_cf, padding=10)
        self.file_path_frm.columnconfigure(1, weight=2)
        self.bus_cf.add(
            child=self.file_path_frm,
            font=("Arial", 10),
            title="Load h5ad files",
            bootstyle=SECONDARY,
        )

        self.path_load_flag = False
        style = Style()
        style.configure("TCheckbutton", font=("Arial", 10))
        style.configure("TButton", font=("Arial", 10))
        style.configure("TRadiobutton", font=("Arial", 10))
        style.configure("TCombobox", font=("Arial", 10))
        self.radio_frame = ttk.Frame(self.file_path_frm)
        self.radio_frame.pack(side=TOP)

        def on_radio_select(value):
            if self.path_load_flag:
                self.Single_file_moduel.destroy()
                self.path_load_flag = False
            if value == "single-h5ad":
                self.Single_file_moduel = ttk.Frame(self.file_path_frm)
                self.Single_file_moduel.pack(after=self.radio_frame)
                self.file_entry = ttk.Entry(
                    self.Single_file_moduel, textvariable="folder-path", font="Arial"
                )
                self.file_entry.pack(side=LEFT, fill=X, expand=YES)
                self.file_entry.insert(END, "select your datas here!")

                def set_data_type_choose(event):
                    print("Current choose:　{}".format(self.data_updata.get()))
                    self.data_type = self.data_updata.get()

                # select_data_type = ttk.StringVar()
                self.data_updata = ttk.Combobox(
                    master=self.Single_file_moduel,
                    textvariable="select_data_type",
                    font=("Arial", 10),
                    values=data_type,
                    height=8,
                    width=6,
                    state="normal",
                    cursor="plus",
                )
                self.data_updata.pack(side=LEFT)
                self.setvar("select_data_type", data_type[0])
                # self.data_updata.current(0)

                self.data_updata.bind("<<ComboboxSelected>>", set_data_type_choose)

                def file_btn_getdirectory():
                    self.path_load_flag = False
                    self.get_directory()

                self.file_btn = ttk.Button(
                    master=self.Single_file_moduel,
                    image="opened-folder",
                    bootstyle=(LINK, SECONDARY),
                    command=file_btn_getdirectory,
                )
                self.file_btn.pack(side=RIGHT)
                self.path_load_flag = True
            else:
                self.Single_file_moduel = ttk.Frame(self.file_path_frm)
                self.Single_file_moduel.pack(after=self.radio_frame)

                file_entry = ttk.Entry(
                    self.Single_file_moduel, textvariable="folder-path"
                )
                file_entry.pack(side=LEFT, fill=X, expand=YES)

                def Select_files():
                    self.multi_files = filedialog.askdirectory()
                    file_entry.delete(0, tk.END)
                    file_entry.insert(0, self.multi_files)
                    self.path_load_flag = False
                    self.Select_files()

                file_choose_btn = ttk.Button(
                    master=self.Single_file_moduel,
                    text="Choose",
                    width=6,
                    command=Select_files,
                )
                file_choose_btn.pack(side=RIGHT)
                self.path_load_flag = True

        radio_var = tk.StringVar()
        radio1 = ttk.Radiobutton(
            self.radio_frame,
            text="Analysis single h5ad file",
            style="TCheckbutton",
            variable=radio_var,
            value="single-h5ad",
            command=lambda: on_radio_select("single-h5ad"),
        )
        radio1.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        radio2 = ttk.Radiobutton(
            self.radio_frame,
            text="Analysis Multi-h5ad files",
            style="TCheckbutton",
            variable=radio_var,
            value="multi h5ad",
            command=lambda: on_radio_select("Multi-h5ad"),
        )
        radio2.grid(row=1, column=0, padx=5, pady=5, sticky="w")

        self.bus_frm = ttk.Frame(self.bus_cf, padding=10)
        self.bus_frm.columnconfigure(1, weight=2)
        self.bus_cf.add(
            child=self.bus_frm,
            font=("Arial", 10),
            title="Select Methods",
            bootstyle=SECONDARY,
        )
        self.frames = ttk.Frame(self.bus_frm)
        self.frames.pack(side=TOP, anchor=W)

        def set_value_before_choose():
            print("Methods：", select_text_data.get())
            new_select_data = []
            for i in methods:
                if select_text_data.get() in i:
                    new_select_data.append(i)
            self.select_box_obj["value"] = new_select_data

        select_text_data = ttk.StringVar()
        self.select_box_obj = ttk.Combobox(
            master=self.frames,
            textvariable=select_text_data,
            font=("Arial", 10),
            values=methods,
            height=8,
            width=20,
            state="normal",
            cursor="plus",
            postcommand=set_value_before_choose,
        )
        self.select_box_obj.pack(side=LEFT, padx=20, anchor=W)
        self.select_box_obj.current(0)

        def cutoff_num_check():
            self.rad_cutoff_value = self.rad_cutoff.get()
            print(self.rad_cutoff_value)

        def alpha_num_check():
            self.alpha_value = self.alpha.get()
            print(self.alpha_value)

        def cluster_num_check():
            self.cluster_value = self.cluster.get()
            print(self.cluster_value)

        def gene_name_check():
            self.genes = self.gene_name.get()
            print(self.genes)

        def functions_info_check():
            self.functions_choose = self.function_box.get()
            print(self.functions_choose)

        def detail_info_check():
            self.detail_info = self.detail_info_box.get()
            print(self.detail_info)

        def adjust_num_check():
            self.adjust_value = self.adjust.get()
            print(self.adjust_value)

        def class_epoch_choose():
            self.cla_epoch_num = self.class_epoch_box.get()
            print(self.cla_epoch_num)

        def train_epoch_choose():
            self.train_epoch_num = self.train_epoch_box.get()
            print(self.train_epoch_num)

        def referance_adjust():
            try:
                self.p_frame = tk.Frame(master=self.bus_frm)
                self.p_frame.pack(
                    fill=ttk.BOTH, expand=True, side=BOTTOM, anchor=W, padx=20, pady=10
                )
                self.label = ttk.Label(
                    self.p_frame, text="Parameter setting: ", width=25
                )
                self.label.pack(side=TOP, anchor=W, pady=10)
                self.para_frame = tk.Frame(
                    master=self.p_frame,
                    highlightbackground="black",
                    highlightthickness=1,
                )
                self.para_frame.pack(side=BOTTOM, anchor=W, pady=10)
                style = Style()
                style.configure("TCheckbutton", font=("Arial", 10))
                style.configure("TButton", font=("Arial", 10))

                self.btn_states = False
                if self.method_flag == "STAGATE":
                    self.rad_cutoff_value = int(150)
                    self.alpha_value = int(3000)
                    self.cluster_value = int(1000)
                    self.genes = float(0.001)
                    self.detail_info = "Deciphering spatial domains"
                    self.rad_cutoff_info = ttk.Label(
                        master=self.para_frame, text="Neighbor distance", font="Arial"
                    )
                    self.rad_cutoff_info.grid(
                        row=1, column=0, padx=20, pady=10, sticky="nsew"
                    )

                    self.rad_cutoff = ttk.Entry(
                        master=self.para_frame,
                        textvariable="rad_cutoff",
                        validate="focusout",
                        validatecommand=cutoff_num_check,
                        width=20,
                    )
                    self.rad_cutoff.grid(row=1, column=1, padx=20, pady=10)
                    self.setvar("rad_cutoff", int(150))
                    self.alpha_info = ttk.Label(
                        master=self.para_frame, text="Features dimension ", font="Arial"
                    )
                    self.alpha_info.grid(
                        row=2, column=0, padx=20, pady=10, sticky="nsew"
                    )
                    self.alpha = ttk.Entry(
                        master=self.para_frame,
                        textvariable="features",
                        validate="focusout",
                        validatecommand=alpha_num_check,
                        width=20,
                    )
                    self.alpha.grid(row=2, column=1, padx=20, pady=10)
                    self.setvar("features", int(3000))
                    self.cluster_info = ttk.Label(
                        master=self.para_frame, text="Train epochs", font="Arial"
                    )
                    self.cluster_info.grid(
                        row=3, column=0, padx=20, pady=10, sticky="nsew"
                    )
                    self.cluster = ttk.Entry(
                        master=self.para_frame,
                        textvariable="epochs",
                        validate="focusout",
                        validatecommand=cluster_num_check,
                        width=20,
                    )
                    self.cluster.grid(row=3, column=1, padx=20, pady=10)
                    self.setvar("epochs", int(1000))
                    self.genes_info = ttk.Label(
                        master=self.para_frame, text="Learning rate", font="Arial"
                    )
                    self.genes_info.grid(
                        row=4, column=0, padx=20, pady=10, sticky="nsew"
                    )
                    self.gene_name = ttk.Entry(
                        master=self.para_frame,
                        textvariable="learning rate",
                        validate="focusout",
                        validatecommand=gene_name_check,
                        width=20,
                    )
                    self.gene_name.grid(row=4, column=1, padx=20, pady=10)
                    self.setvar("learning rate", float(0.001))
                    self.detail_infos = ttk.Label(
                        master=self.para_frame, text="Details", font="Arial"
                    )
                    self.detail_infos.grid(
                        row=5, column=0, padx=20, pady=10, sticky="nsew"
                    )
                    self.detail_info_box = ttk.Entry(
                        master=self.para_frame,
                        textvariable="detail_info",
                        validate="focusout",
                        validatecommand=detail_info_check,
                        width=20,
                    )
                    self.detail_info_box.grid(row=5, column=1, padx=20, pady=10)
                    self.setvar("detail_info", "Deciphering spatial domains")
                    Tooltip(
                        self.rad_cutoff,
                        "Distance threshold between center spot and neighbors: 10X-150",
                    )
                    Tooltip(self.alpha, "Dimensions of embedding genes: 3000")
                    Tooltip(self.cluster, "Training epochs number: 1000")
                    Tooltip(self.gene_name, "Learning rate: 0.001")
                    Tooltip(self.detail_info_box, "Editable information: STAGATE")

                    method_btn.configure(state=DISABLED)

                    self.information = "Deep learning"
                    self.functions = "Deciphering spatial domains"
                    self.Run_time = "3-5 mins"
                    self.Datatime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    style_head = ttk.Style()
                    style_head.configure("Treeview.Heading", font="Arial")
                    self.tv.insert(
                        "",
                        END,
                        values=(
                            self.method_flag,
                            self.information,
                            self.functions,
                            self.Run_time,
                            self.Datatime,
                        ),
                    )

                elif self.method_flag == "STAligner":
                    self.rad_cutoff_value = int(150)
                    self.adjust_value = int(1)
                    self.cluster_value = int(1000)
                    self.alpha_value = float(0.001)
                    self.detail_info = (
                        "Alignment and integration of spatially transcriptomes"
                    )
                    self.rad_cutoff_info = ttk.Label(
                        master=self.para_frame, text="Neighbor distance", font="Arial"
                    )
                    self.rad_cutoff_info.grid(
                        row=1, column=0, padx=20, pady=10, sticky="nsew"
                    )
                    self.rad_cutoff = ttk.Entry(
                        master=self.para_frame,
                        textvariable="rad_cutoff",
                        validate="focusout",
                        validatecommand=cutoff_num_check,
                        width=20,
                    )
                    self.rad_cutoff.grid(row=1, column=1, padx=20, pady=10)
                    self.setvar("rad_cutoff", int(150))

                    self.cluster_info = ttk.Label(
                        master=self.para_frame, text="Training epochs", font="Arial"
                    )
                    self.cluster_info.grid(
                        row=2, column=0, padx=20, pady=10, sticky="nsew"
                    )
                    self.cluster = ttk.Entry(
                        master=self.para_frame,
                        textvariable="n_epochs",
                        validate="focusout",
                        validatecommand=cluster_num_check,
                        width=20,
                    )
                    self.cluster.grid(row=2, column=1, padx=20, pady=10)
                    self.setvar("n_epochs", int(1000))

                    self.adjust_info = ttk.Label(
                        master=self.para_frame, text="Slice margin", font="Arial"
                    )
                    self.adjust_info.grid(
                        row=3, column=0, padx=20, pady=10, sticky="nsew"
                    )
                    self.adjust = ttk.Entry(
                        master=self.para_frame,
                        textvariable="margin",
                        validate="focusout",
                        validatecommand=adjust_num_check,
                        width=20,
                    )
                    self.adjust.grid(row=3, column=1, padx=20, pady=10)
                    self.setvar("margin", int(1))

                    self.alpha_info = ttk.Label(
                        master=self.para_frame, text="Learning rate", font="Arial"
                    )
                    self.alpha_info.grid(
                        row=4, column=0, padx=20, pady=10, sticky="nsew"
                    )
                    self.alpha = ttk.Entry(
                        master=self.para_frame,
                        textvariable="learning rate",
                        validate="focusout",
                        validatecommand=alpha_num_check,
                        width=20,
                    )
                    self.alpha.grid(row=4, column=1, padx=20, pady=10)
                    self.setvar("learning rate", float(0.001))

                    self.detail_infos = ttk.Label(
                        master=self.para_frame, text="Details", font="Arial"
                    )
                    self.detail_infos.grid(
                        row=5, column=0, padx=20, pady=10, sticky="nsew"
                    )
                    self.detail_info_box = ttk.Entry(
                        master=self.para_frame,
                        textvariable="detail_info",
                        validate="focusout",
                        validatecommand=detail_info_check,
                        width=20,
                    )
                    self.detail_info_box.grid(row=5, column=1, padx=20, pady=10)
                    self.setvar(
                        "detail_info",
                        "Alignment and integration of spatially transcriptomes",
                    )

                    Tooltip(
                        self.rad_cutoff,
                        "Distance threshold between center spot and neighbors: 10X-150",
                    )
                    Tooltip(self.cluster, "Training epochs number: 1000")
                    Tooltip(self.adjust, "Slice alignment hyperparameter: 1")
                    Tooltip(self.alpha, "learning rate: 0.001")
                    Tooltip(self.detail_info_box, "Editable information: STAligner")

                    method_btn.configure(state=DISABLED)

                    self.information = "Deep learning"
                    self.functions = (
                        "Alignment and integration of spatially transcriptomes"
                    )
                    self.Run_time = "5-10 mins"
                    self.Authods = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    self.tv.insert(
                        "",
                        END,
                        values=(
                            self.method_flag,
                            self.information,
                            self.functions,
                            self.Run_time,
                            self.Authods,
                        ),
                    )

                elif self.method_flag == "STAMarker":
                    self.rad_cutoff_value = int(150)
                    self.alpha_value = int(5)
                    self.cluster_value = int(7)
                    self.train_epoch_num = int(500)
                    self.cla_epoch_num = int(500)
                    self.detail_info = "Deciphering spatial domains SVGs"
                    self.rad_cutoff_info = ttk.Label(
                        master=self.para_frame, text="Neighbor distance", font="Arial"
                    )
                    self.rad_cutoff_info.grid(
                        row=1, column=0, padx=20, pady=10, sticky="nsew"
                    )

                    self.rad_cutoff = ttk.Entry(
                        master=self.para_frame,
                        textvariable="rad_cutoff",
                        validate="focusout",
                        validatecommand=cutoff_num_check,
                        width=20,
                    )
                    self.rad_cutoff.grid(row=1, column=1, padx=20, pady=10)
                    self.setvar("rad_cutoff", int(150))

                    self.alpha_info = ttk.Label(
                        master=self.para_frame, text="Autoencoder number", font="Arial"
                    )
                    self.alpha_info.grid(
                        row=2, column=0, padx=20, pady=10, sticky="nsew"
                    )
                    self.alpha = ttk.Entry(
                        master=self.para_frame,
                        textvariable="autoencoder",
                        validate="focusout",
                        validatecommand=alpha_num_check,
                        width=20,
                    )
                    self.alpha.grid(row=2, column=1, padx=20, pady=10)
                    self.setvar("autoencoder", int(5))

                    self.cluster_info = ttk.Label(
                        master=self.para_frame, text="Cluster number", font="Arial"
                    )
                    self.cluster_info.grid(
                        row=3, column=0, padx=20, pady=10, sticky="nsew"
                    )

                    self.cluster = ttk.Entry(
                        master=self.para_frame,
                        textvariable="cluster",
                        validate="focusout",
                        validatecommand=cluster_num_check,
                        width=20,
                    )
                    self.cluster.grid(row=3, column=1, padx=20, pady=10)
                    self.setvar("cluster", int(7))

                    self.train_epoch_info = ttk.Label(
                        master=self.para_frame, text="Training epochs", font="Arial"
                    )
                    self.train_epoch_info.grid(
                        row=4, column=0, padx=20, pady=10, sticky="nsew"
                    )

                    self.train_epoch_box = ttk.Entry(
                        master=self.para_frame,
                        textvariable="epochs",
                        validate="focusout",
                        validatecommand=train_epoch_choose,
                        width=20,
                    )
                    self.train_epoch_box.grid(row=4, column=1, padx=20, pady=10)
                    self.setvar("epochs", 500)

                    self.class_epoch_infos = ttk.Label(
                        master=self.para_frame, text="Classifiers epochs", font="Arial"
                    )
                    self.class_epoch_infos.grid(
                        row=5, column=0, padx=20, pady=10, sticky="nsew"
                    )

                    self.class_epoch_box = ttk.Entry(
                        master=self.para_frame,
                        textvariable="classifiers",
                        validate="focusout",
                        validatecommand=class_epoch_choose,
                        width=20,
                    )
                    self.class_epoch_box.grid(row=5, column=1, padx=20, pady=10)
                    self.setvar("classifiers", 500)

                    self.detail_infos = ttk.Label(
                        master=self.para_frame, text="Details", font="Arial"
                    )
                    self.detail_infos.grid(
                        row=6, column=0, padx=20, pady=10, sticky="nsew"
                    )

                    self.detail_info_box = ttk.Entry(
                        master=self.para_frame,
                        textvariable="detail_info",
                        validate="focusout",
                        validatecommand=detail_info_check,
                        width=20,
                    )
                    self.detail_info_box.grid(row=6, column=1, padx=20, pady=10)
                    self.setvar("detail_info", "Deciphering spatial domains SVGs")

                    Tooltip(
                        self.rad_cutoff,
                        "Distance threshold between center spot and neighbors: 10X-150",
                    )
                    Tooltip(self.alpha, "Autoencoder number: 5")
                    Tooltip(self.cluster, "Number of cluster labels: 7")
                    Tooltip(self.train_epoch_box, "training epochs: 500")
                    Tooltip(self.class_epoch_box, "classifiers epochs: 500")
                    Tooltip(self.detail_info_box, "Editable information: STAMarker")

                    method_btn.configure(state=DISABLED)

                    self.information = "Deep learning"
                    self.functions = "Deciphering spatial domains SVGs"
                    self.Run_time = "10-30 mins"
                    self.Datatime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    style_head = ttk.Style()
                    style_head.configure("Treeview.Heading", font="Arial")
                    self.tv.insert(
                        "",
                        END,
                        values=(
                            self.method_flag,
                            self.information,
                            self.functions,
                            self.Run_time,
                            self.Datatime,
                        ),
                    )

                elif self.method_flag == "STAGE":
                    self.alpha_value = float(0.5)
                    self.genes = int(2000)
                    self.functions_choose = "recovery"
                    self.detail_info = "Gene information enhancement and recovery"
                    self.alpha_info = ttk.Label(
                        master=self.para_frame, text="Downsampling ratio", font="Arial"
                    )
                    self.alpha_info.grid(
                        row=1, column=0, padx=20, pady=10, sticky="nsew"
                    )

                    self.alpha = ttk.Entry(
                        master=self.para_frame,
                        textvariable="alpha",
                        validate="focusout",
                        validatecommand=alpha_num_check,
                        width=20,
                    )
                    self.alpha.grid(row=1, column=1, padx=20, pady=10)
                    self.setvar("alpha", float(0.5))

                    self.genes_info = ttk.Label(
                        master=self.para_frame, text="Training epochs", font="Arial"
                    )
                    self.genes_info.grid(
                        row=2, column=0, padx=20, pady=10, sticky="nsew"
                    )

                    self.gene_name = ttk.Entry(
                        master=self.para_frame,
                        textvariable="train_epoch",
                        validate="focusout",
                        validatecommand=gene_name_check,
                        width=20,
                    )
                    self.gene_name.grid(row=2, column=1, padx=20, pady=10)
                    self.setvar("train_epoch", 2000)

                    self.function_infos = ttk.Label(
                        master=self.para_frame, text="Functions select", font="Arial"
                    )
                    self.function_infos.grid(
                        row=3, column=0, padx=20, pady=10, sticky="nsew"
                    )

                    self.function_box = ttk.Entry(
                        master=self.para_frame,
                        textvariable="function_infos",
                        validate="focusout",
                        validatecommand=functions_info_check,
                        width=20,
                    )
                    self.function_box.grid(row=3, column=1, padx=20, pady=10)
                    self.setvar("function_infos", "recovery")

                    self.detail_infos = ttk.Label(
                        master=self.para_frame, text="Details", font="Arial"
                    )
                    self.detail_infos.grid(
                        row=4, column=0, padx=20, pady=10, sticky="nsew"
                    )

                    self.detail_info_box = ttk.Entry(
                        master=self.para_frame,
                        textvariable="detail_info",
                        validate="focusout",
                        validatecommand=detail_info_check,
                        width=20,
                    )
                    self.detail_info_box.grid(row=4, column=1, padx=20, pady=10)
                    self.setvar(
                        "detail_info", "Gene information enhancement and recovery"
                    )

                    Tooltip(self.alpha, "Algorithm hyperparameter: 0.5/[0-1]")
                    Tooltip(self.gene_name, "Training epoch number: 2000")
                    Tooltip(
                        self.function_box,
                        "Functions choose: generation, recovery, 3d_model",
                    )
                    Tooltip(self.detail_info_box, "Editable information: STAGE")

                    method_btn.configure(state=DISABLED)

                    self.information = (
                        "Supervised learning-gene information enhancement"
                    )
                    self.functions = (
                        "Gene expression prediction and tissue segmentation"
                    )
                    self.Run_time = "10-40 mins"
                    self.Datatime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    style_head = ttk.Style()
                    style_head.configure("Treeview.Heading", font="Arial")
                    self.tv.insert(
                        "",
                        END,
                        values=(
                            self.method_flag,
                            self.information,
                            self.functions,
                            self.Run_time,
                            self.Datatime,
                        ),
                    )
                elif self.method_flag == "SpaGCN":
                    self.rad_cutoff_value = int(49)
                    self.alpha_value = int(200)
                    self.cluster_value = int(7)
                    self.genes = float(0.5)
                    self.detail_info = "Deciphering spatial domains"
                    self.rad_cutoff_info = ttk.Label(
                        master=self.para_frame, text="Training weight", font="Arial"
                    )
                    self.rad_cutoff_info.grid(
                        row=1, column=0, padx=20, pady=10, sticky="nsew"
                    )

                    self.rad_cutoff = ttk.Entry(
                        master=self.para_frame,
                        textvariable="rad_cutoff",
                        validate="focusout",
                        validatecommand=cutoff_num_check,
                        width=20,
                    )
                    self.rad_cutoff.grid(row=1, column=1, padx=20, pady=10)
                    self.setvar("rad_cutoff", int(49))

                    self.alpha_info = ttk.Label(
                        master=self.para_frame, text="Training epochs", font="Arial"
                    )
                    self.alpha_info.grid(
                        row=2, column=0, padx=20, pady=10, sticky="nsew"
                    )
                    self.alpha = ttk.Entry(
                        master=self.para_frame,
                        textvariable="alpha",
                        validate="focusout",
                        validatecommand=alpha_num_check,
                        width=20,
                    )
                    self.alpha.grid(row=2, column=1, padx=20, pady=10)
                    self.setvar("alpha", int(200))

                    self.cluster_info = ttk.Label(
                        master=self.para_frame, text="Cluster number", font="Arial"
                    )
                    self.cluster_info.grid(
                        row=3, column=0, padx=20, pady=10, sticky="nsew"
                    )

                    self.cluster = ttk.Entry(
                        master=self.para_frame,
                        textvariable="cluster",
                        validate="focusout",
                        validatecommand=cluster_num_check,
                        width=20,
                    )
                    self.cluster.grid(row=3, column=1, padx=20, pady=10)
                    self.setvar("cluster", int(7))

                    self.genes_info = ttk.Label(
                        master=self.para_frame, text="Training percentage", font="Arial"
                    )
                    self.genes_info.grid(
                        row=4, column=0, padx=20, pady=10, sticky="nsew"
                    )
                    self.gene_name = ttk.Entry(
                        master=self.para_frame,
                        textvariable="gene_name",
                        validate="focusout",
                        validatecommand=gene_name_check,
                        width=20,
                    )
                    self.gene_name.grid(
                        row=4, column=1, padx=20, pady=10, sticky="nsew"
                    )
                    self.setvar("gene_name", float(0.5))

                    self.detail_infos = ttk.Label(
                        master=self.para_frame, text="Details", font="Arial"
                    )
                    self.detail_infos.grid(
                        row=5, column=0, padx=20, pady=10, sticky="nsew"
                    )
                    self.detail_info_box = ttk.Entry(
                        master=self.para_frame,
                        textvariable="detail_info",
                        validate="focusout",
                        validatecommand=detail_info_check,
                        width=20,
                    )
                    self.detail_info_box.grid(
                        row=5, column=1, padx=20, pady=10, sticky="nsew"
                    )
                    self.setvar("detail_info", "Deciphering spatial domains")

                    Tooltip(
                        self.rad_cutoff, "Image feature extraction parameters: 49(10X)"
                    )
                    Tooltip(self.alpha, "Algorithm training epoch: 200")
                    Tooltip(self.cluster, "Number of cluster labels: 7")
                    Tooltip(self.gene_name, "image use rate: 0.5")
                    Tooltip(self.detail_info_box, "Editable information: SpaGCN")

                    method_btn.configure(state=DISABLED)

                    self.information = "SpaGCN Deep learning"
                    self.functions = "Deciphering spatial domains"
                    self.Run_time = "3-5 mins"
                    self.Datatime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    style_head = ttk.Style()
                    style_head.configure("Treeview.Heading", font="Arial")
                    self.tv.insert(
                        "",
                        END,
                        values=(
                            self.method_flag,
                            self.information,
                            self.functions,
                            self.Run_time,
                            self.Datatime,
                        ),
                    )
                elif self.method_flag == "SEDR":
                    self.rad_cutoff_value = int(10)
                    self.alpha_value = float(0.1)
                    self.cluster_value = int(200)
                    self.genes = float(0.2)
                    self.detail_info = "Deciphering spatial domains"

                    self.rad_cutoff_info = ttk.Label(
                        master=self.para_frame, text="Graph weight", font="Arial"
                    )
                    self.rad_cutoff_info.grid(
                        row=1, column=0, padx=20, pady=10, sticky="nsew"
                    )

                    self.rad_cutoff = ttk.Entry(
                        master=self.para_frame,
                        textvariable="rad_cutoff",
                        validate="focusout",
                        validatecommand=cutoff_num_check,
                        width=20,
                    )
                    self.rad_cutoff.grid(row=1, column=1, padx=20, pady=10)
                    self.setvar("rad_cutoff", int(10))

                    self.alpha_info = ttk.Label(
                        master=self.para_frame, text="Training parameter", font="Arial"
                    )
                    self.alpha_info.grid(
                        row=2, column=0, padx=20, pady=10, sticky="nsew"
                    )
                    self.alpha = ttk.Entry(
                        master=self.para_frame,
                        textvariable="alpha",
                        validate="focusout",
                        validatecommand=alpha_num_check,
                        width=20,
                    )
                    self.alpha.grid(row=2, column=1, padx=20, pady=10)
                    self.setvar("alpha", float(0.1))

                    self.cluster_info = ttk.Label(
                        master=self.para_frame, text="n_cluster", font="Arial"
                    )
                    self.cluster_info.grid(row=3, column=0, padx=20, pady=10)
                    self.cluster = ttk.Entry(
                        master=self.para_frame,
                        textvariable="cluster",
                        validate="focusout",
                        validatecommand=cluster_num_check,
                        width=20,
                    )
                    self.cluster.grid(row=3, column=1, padx=20, pady=10)
                    self.setvar("cluster", int(200))

                    self.genes_info = ttk.Label(
                        master=self.para_frame, text="Dropout rate", font="Arial"
                    )
                    self.genes_info.grid(
                        row=4, column=0, padx=20, pady=10, sticky="nsew"
                    )
                    self.gene_name = ttk.Entry(
                        master=self.para_frame,
                        textvariable="gene_name",
                        validate="focusout",
                        validatecommand=gene_name_check,
                        width=20,
                    )
                    self.gene_name.grid(row=4, column=1, padx=20, pady=10)
                    self.setvar("gene_name", float(0.2))

                    self.detail_infos = ttk.Label(
                        master=self.para_frame, text="Details", font="Arial"
                    )
                    self.detail_infos.grid(row=5, column=0, padx=20, pady=10)

                    self.detail_info_box = ttk.Entry(
                        master=self.para_frame,
                        textvariable="detail_info",
                        validate="focusout",
                        validatecommand=detail_info_check,
                        width=20,
                    )
                    self.detail_info_box.grid(row=5, column=1, padx=20, pady=10)
                    self.setvar("detail_info", "Deciphering spatial domains")

                    Tooltip(self.rad_cutoff, "parameter k in spatial graph: 10X-10")
                    Tooltip(self.alpha, "Weight of GCN loss: 0.1")
                    Tooltip(self.cluster, "Training epochs: 200")
                    Tooltip(self.gene_name, "Dropout rate.: 0.2")
                    Tooltip(self.detail_info_box, "Editable information: SEDR")

                    method_btn.configure(state=DISABLED)

                    self.information = "Deep learning"
                    self.functions = "Deciphering spatial domains"
                    self.Run_time = "3-5 mins"
                    self.Datatime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    style_head = ttk.Style()
                    style_head.configure("Treeview.Heading", font="Arial")
                    self.tv.insert(
                        "",
                        END,
                        values=(
                            self.method_flag,
                            self.information,
                            self.functions,
                            self.Run_time,
                            self.Datatime,
                        ),
                    )
                elif self.method_flag == "SCANPY":
                    self.rad_cutoff_value = int(20)
                    self.cluster_value = str(3)
                    self.detail_info = "Deciphering spatial domains"
                    self.rad_cutoff_info = ttk.Label(
                        master=self.para_frame, text="PCA-dim", font="Arial"
                    )
                    self.rad_cutoff_info.grid(
                        row=1, column=0, padx=20, pady=10, sticky="nsew"
                    )

                    self.rad_cutoff = ttk.Entry(
                        master=self.para_frame,
                        textvariable="rad_cutoff",
                        validate="focusout",
                        validatecommand=cutoff_num_check,
                        width=20,
                    )
                    self.rad_cutoff.grid(row=1, column=1, padx=20, pady=10)
                    self.setvar("rad_cutoff", int(20))

                    self.cluster_info = ttk.Label(
                        master=self.para_frame, text="Nth-cluster", font="Arial"
                    )
                    self.cluster_info.grid(
                        row=2, column=0, padx=20, pady=10, sticky="nsew"
                    )
                    self.cluster = ttk.Entry(
                        master=self.para_frame,
                        textvariable="cluster",
                        validate="focusout",
                        validatecommand=cluster_num_check,
                        width=20,
                    )
                    self.cluster.grid(row=2, column=1, padx=20, pady=10)
                    self.setvar("cluster", str(3))

                    self.detail_infos = ttk.Label(
                        master=self.para_frame, text="Details", font="Arial"
                    )
                    self.detail_infos.grid(
                        row=3, column=0, padx=20, pady=10, sticky="nsew"
                    )
                    self.detail_info_box = ttk.Entry(
                        master=self.para_frame,
                        textvariable="detail_info",
                        validate="focusout",
                        validatecommand=detail_info_check,
                        width=20,
                    )
                    self.detail_info_box.grid(row=3, column=1, padx=20, pady=10)
                    self.setvar(
                        "detail_info",
                        "Analysis of single-cell and spatial transcriptome processes",
                    )
                    Tooltip(self.rad_cutoff, "dimensions of pca reduction: 10X-50")
                    Tooltip(
                        self.cluster, "variable gene difference detection cluster: 3"
                    )
                    Tooltip(self.detail_info_box, "Editable information: SCANPY")

                    method_btn.configure(state=DISABLED)

                    self.information = "Statistical data analysis"
                    self.functions = (
                        "Analysis of single-cell and spatial transcriptome processes"
                    )
                    self.Run_time = "3-5 mins"
                    self.Datatime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    style_head = ttk.Style()
                    style_head.configure("Treeview.Heading", font="Arial")
                    self.tv.insert(
                        "",
                        END,
                        values=(
                            self.method_flag,
                            self.information,
                            self.functions,
                            self.Run_time,
                            self.Datatime,
                        ),
                    )
                else:
                    Messagebox.ok(message="please input right Methods!")
                    self.btn_states = True

                self.result_setting_frame = ttk.Frame(
                    master=self.right_panel, height=20, borderwidth=2, relief="solid"
                )
                self.result_setting_frame.pack(after=self.info_Frame, fill=X)

                result_info_frame = ttk.Frame(master=self.result_setting_frame)
                result_info_frame.pack(side=TOP, anchor=W)
                result_info_label = ttk.Label(
                    master=result_info_frame,
                    text=f"{self.method_flag} Cluster setting: ",
                )
                result_info_label.pack(side=TOP, anchor=W, pady=10)

                result_frame = ttk.Frame(master=self.result_setting_frame)
                result_frame.pack()

                label = ttk.Label(result_frame, text="select file:")
                label.pack(side=LEFT)
                entry = ttk.Entry(result_frame, textvariable="load", width=10)
                entry.pack(side=LEFT)
                self.setvar("load", "select trained file from computer")

                load_file_btn = ttk.Button(
                    master=result_frame,
                    image="opened-folder",
                    bootstyle=(LINK, SECONDARY),
                    command=self.get_trained_file_path,
                )
                load_file_btn.pack(side=LEFT)

                if self.method_flag == "STAMarker":
                    label1 = ttk.Label(result_frame, text="Single SVGs name:")
                    label1.pack(side=LEFT)
                    entry1 = ttk.Entry(result_frame, textvariable="single-SVG", width=6)
                    entry1.pack(side=LEFT, padx=15)
                    self.setvar("single-SVG", "CCDC18")

                    label2 = ttk.Label(result_frame, text="Multi-SVGs number:")
                    label2.pack(side=LEFT)
                    entry2 = ttk.Entry(result_frame, textvariable="multi-SVGs", width=6)
                    entry2.pack(side=LEFT, padx=15)
                    self.setvar("multi-SVGs", 4)

                    label4 = ttk.Label(result_frame, text="spot_size:")
                    label4.pack(side=LEFT)
                    entry4 = ttk.Entry(result_frame, textvariable="spot_size", width=6)
                    entry4.pack(side=LEFT, padx=15)
                    self.setvar("spot_size", 100)

                    cluster_btn = ttk.Button(
                        result_frame,
                        text="Confirm",
                        command=self.Cluster_analysis_thread,
                    )
                    cluster_btn.pack(side=LEFT, padx=6)

                    button1 = ttk.Button(
                        result_frame, text="Show", command=self.STAMarker_show
                    )
                    button1.pack(side=RIGHT, padx=6)

                    button2 = ttk.Button(
                        result_frame, text="Reset", command=self.remove_images
                    )
                    button2.pack(side=RIGHT)

                    self.spot_size = int(entry4.get())
                    self.show_gene_name = entry1.get()
                    self.show_gene_num = int(entry2.get())

                    Tooltip(entry1, "Show single SVGs name: CCDC18")
                    Tooltip(entry2, "Show Multi-SVGs, input number of genes: 4")
                    Tooltip(entry4, "image spot size: 100")
                elif self.method_flag == "STAGE":
                    label1 = ttk.Label(result_frame, text="Single gene name:")
                    label1.pack(side=LEFT)
                    entry1 = ttk.Entry(result_frame, textvariable="gene", width=6)
                    entry1.pack(side=LEFT, padx=15)
                    self.setvar("gene", "Pcp4")

                    label4 = ttk.Label(result_frame, text="spot_size:")
                    label4.pack(side=LEFT)
                    entry4 = ttk.Entry(result_frame, textvariable="Spot size", width=6)
                    entry4.pack(side=LEFT, padx=15)
                    self.setvar("Spot size", 100)

                    cluster_btn = ttk.Button(
                        result_frame,
                        text="Confirm",
                        command=self.Cluster_analysis_thread,
                    )
                    cluster_btn.pack(side=LEFT, padx=6)

                    button1 = ttk.Button(
                        result_frame, text="Show", command=self.STAGE_show
                    )
                    button1.pack(side=RIGHT, padx=6)

                    button2 = ttk.Button(
                        result_frame, text="Reset", command=self.remove_images
                    )
                    button2.pack(side=RIGHT)

                    self.spot_size = int(entry4.get())
                    self.show_gene_name = entry1.get()

                    Tooltip(entry1, "Show single SVGs name: Pcp4")
                    Tooltip(entry4, "image spot size: 100")
                else:
                    label1 = ttk.Label(result_frame, text="mcluster:")
                    label1.pack(side=LEFT)
                    entry1 = ttk.Entry(result_frame, textvariable="mcluster", width=6)
                    entry1.pack(side=LEFT, padx=15)
                    self.setvar("mcluster", 7)

                    label2 = ttk.Label(result_frame, text="louvain:")
                    label2.pack(side=LEFT)
                    entry2 = ttk.Entry(result_frame, textvariable="louvain", width=6)
                    entry2.pack(side=LEFT, padx=15)
                    self.setvar("louvain", 0.65)

                    if self.method_flag == "STAGATE":
                        label3 = ttk.Label(result_frame, text="kmeans:")
                        label3.pack(side=LEFT)
                        entry3 = ttk.Entry(result_frame, textvariable="kmeans", width=6)
                        entry3.pack(side=LEFT, padx=15)
                        self.setvar("kmeans", 7)
                        self.kmeans_num = int(entry3.get())
                        Tooltip(entry3, "Kmean cluster number: 7")

                        label4 = ttk.Label(result_frame, text="spot_size:")
                        label4.pack(side=LEFT)
                        entry4 = ttk.Entry(
                            result_frame, textvariable="spot_size", width=6
                        )
                        entry4.pack(side=LEFT, padx=15)
                        self.setvar("spot_size", 100)

                        cluster_btn = ttk.Button(
                            result_frame,
                            text="Confirm",
                            command=self.Cluster_analysis_thread,
                        )
                        cluster_btn.pack(side=LEFT, padx=6)

                        button1 = ttk.Button(
                            result_frame, text="Show", command=self.STAGATE_show
                        )
                        button1.pack(side=RIGHT, padx=6)

                    if self.method_flag == "SEDR":
                        label3 = ttk.Label(result_frame, text="kmeans:")
                        label3.pack(side=LEFT)
                        entry3 = ttk.Entry(result_frame, textvariable="kmeans", width=6)
                        entry3.pack(side=LEFT, padx=15)
                        self.setvar("kmeans", 7)
                        self.kmeans_num = int(entry3.get())
                        Tooltip(entry3, "Kmean cluster number: 7")

                        label4 = ttk.Label(result_frame, text="spot_size:")
                        label4.pack(side=LEFT)
                        entry4 = ttk.Entry(
                            result_frame, textvariable="spot_size", width=6
                        )
                        entry4.pack(side=LEFT, padx=15)
                        self.setvar("spot_size", 100)

                        cluster_btn = ttk.Button(
                            result_frame,
                            text="Confirm",
                            command=self.Cluster_analysis_thread,
                        )
                        cluster_btn.pack(side=LEFT, padx=6)
                        button1 = ttk.Button(
                            result_frame, text="Show", command=self.SEDR_show
                        )
                        button1.pack(side=RIGHT, padx=6)

                    if self.method_flag == "SCANPY":
                        label3 = ttk.Label(result_frame, text="kmeans:")
                        label3.pack(side=LEFT)
                        entry3 = ttk.Entry(result_frame, textvariable="kmeans", width=6)
                        entry3.pack(side=LEFT, padx=15)
                        self.setvar("kmeans", 7)
                        self.kmeans_num = int(entry3.get())
                        Tooltip(entry3, "Kmean cluster number: 7")

                        label4 = ttk.Label(result_frame, text="spot_size:")
                        label4.pack(side=LEFT)
                        entry4 = ttk.Entry(
                            result_frame, textvariable="spot_size", width=6
                        )
                        entry4.pack(side=LEFT, padx=15)
                        self.setvar("spot_size", 100)

                        cluster_btn = ttk.Button(
                            result_frame,
                            text="Confirm",
                            command=self.Cluster_analysis_thread,
                        )
                        cluster_btn.pack(side=LEFT, padx=6)
                        button1 = ttk.Button(
                            result_frame, text="Show", command=self.SCANPY_show
                        )
                        button1.pack(side=RIGHT, padx=6)

                    # label4 = ttk.Label(result_frame, text="spot_size:")
                    # label4.pack(side=LEFT)
                    # entry4 = ttk.Entry(result_frame, textvariable='spot_size', width=6)
                    # entry4.pack(side=LEFT, padx=15)
                    # self.setvar('spot_size', 100)

                    if self.method_flag == "STAligner":
                        label4 = ttk.Label(result_frame, text="spot_size:")
                        label4.pack(side=LEFT)
                        entry4 = ttk.Entry(
                            result_frame, textvariable="spot_size", width=6
                        )
                        entry4.pack(side=LEFT, padx=15)
                        self.setvar("spot_size", 100)

                        cluster_btn = ttk.Button(
                            result_frame,
                            text="Confirm",
                            command=self.Cluster_analysis_thread,
                        )
                        cluster_btn.pack(side=LEFT, padx=6)
                        button1 = ttk.Button(
                            result_frame, text="Show", command=self.STAligner_show
                        )
                        button1.pack(side=RIGHT, padx=6)

                    if self.method_flag == "SpaGCN":
                        label4 = ttk.Label(result_frame, text="spot_size:")
                        label4.pack(side=LEFT)
                        entry4 = ttk.Entry(
                            result_frame, textvariable="spot_size", width=6
                        )
                        entry4.pack(side=LEFT, padx=15)
                        self.setvar("spot_size", 100)

                        cluster_btn = ttk.Button(
                            result_frame,
                            text="Confirm",
                            command=self.Cluster_analysis_thread,
                        )
                        cluster_btn.pack(side=LEFT, padx=6)
                        button1 = ttk.Button(
                            result_frame, text="Show", command=self.SpaGCN_show
                        )
                        button1.pack(side=RIGHT, padx=6)

                    button2 = ttk.Button(
                        result_frame, text="Reset", command=self.remove_images
                    )
                    button2.pack(side=RIGHT)

                    self.louvain_res = float(entry2.get())
                    self.Mcluster_num = int(entry1.get())
                    self.spot_size = int(entry4.get())

                    Tooltip(entry1, "Mcluster number: 7")
                    Tooltip(entry2, "louvain cluster resolution: 0.65")
                    Tooltip(entry4, "image spot size: 100")
            except:
                self.btn_states = True
                Messagebox.show_warning(
                    title="Attention", message="Make sure your data had beed loaded!"
                )

        self.btn_states = True
        method_btn = ttk.Button(
            master=self.frames,
            text="Confirm",
            width=6,
            style="my.TButton",
            command=referance_adjust,
        )
        method_btn.pack(side=LEFT)
        self.method_flag = methods[0]

        def submit_result(event):
            if self.btn_states == False:
                self.para_frame.destroy()
                self.btn_states = True
                self.p_frame.destroy()
                self.result_setting_frame.destroy()
            method_btn.configure(state=ACTIVE)
            self.method_flag = self.select_box_obj.get()
            print("Current choose: {}".format(self.select_box_obj.get()))

        self.select_box_obj.bind("<<ComboboxSelected>>", submit_result)

        status_cf = CollapsingFrame(self.left_panel)
        status_cf.pack(fill=BOTH, pady=1)

        status_frm = ttk.Frame(status_cf, padding=10)
        status_frm.columnconfigure(1, weight=1)
        status_cf.add(child=status_frm, title="Running Status", bootstyle=SECONDARY)

        lbl = ttk.Label(master=status_frm, textvariable="prog-message")
        lbl.configure(font="Arial")
        lbl.grid(row=0, column=0, columnspan=2, sticky=W)
        self.setvar("prog-message", "choosing methods...")

        self.pb = ttk.Progressbar(
            master=status_frm, mode="determinate", length=100, orient=HORIZONTAL
        )
        self.pb.grid(row=1, column=0, columnspan=2, sticky=EW, pady=(10, 5))
        lbl = ttk.Label(status_frm, text="Start time:")
        lbl.configure(font="Arial")
        lbl.grid(row=2, column=0, sticky=W, pady=2)

        lbl = ttk.Label(status_frm, textvariable="prog-time-started")
        lbl.grid(row=2, column=1, columnspan=2, sticky=EW, padx=2, pady=5)

        self.setvar("prog-time-started", None)
        lbl = ttk.Label(status_frm, text="End time:")
        lbl.configure(font="Arial")
        lbl.grid(row=3, column=0, sticky=W, pady=2)
        lbl = ttk.Label(status_frm, textvariable="End-time")
        lbl.grid(row=3, column=1, columnspan=2, sticky=EW, pady=5, padx=2)
        self.setvar("End-time", None)

        lbl = ttk.Label(status_frm, text="Cost time:")
        lbl.configure(font="Arial")
        lbl.grid(row=4, column=0, sticky=W, pady=2)
        lbl = ttk.Label(status_frm, textvariable="total-time-cost")
        lbl.grid(row=4, column=1, columnspan=2, sticky=EW, pady=5, padx=2)
        self.setvar("total-time-cost", None)

        sep = ttk.Separator(status_frm, bootstyle=SECONDARY)
        sep.grid(row=5, column=0, columnspan=2, pady=10, sticky=EW)

        lbl = ttk.Label(status_frm, text="Now used method:")
        lbl.configure(font="Arial")
        lbl.grid(row=6, column=0, sticky=W, pady=5)
        lbl = ttk.Label(status_frm, textvariable="current-file-msg")
        lbl.configure(font="Arial")
        lbl.grid(row=6, column=1, columnspan=2, pady=5, sticky=EW, padx=5)
        self.setvar("current-file-msg", None)

        lbl = ttk.Label(self.left_panel, image="logo", style="bg.TLabel")
        lbl.pack(side="bottom")

        self.right_panel = ttk.Frame(self, padding=(2, 1))
        self.right_panel.pack(side=RIGHT, fill=BOTH, expand=YES)

        self.info_Frame = ttk.Frame(self.right_panel)
        self.info_Frame.pack(side=TOP, fill=X)

        self.ybar = ttk.Scrollbar(self.info_Frame)
        self.ybar.pack(side=RIGHT, fill=Y)

        self.tv = ttk.Treeview(
            self.info_Frame, show="headings", height=5, yscrollcommand=self.ybar.set
        )
        self.tv.configure(
            columns=(
                "Method",
                "information",
                "functions",
                "Estimated time cost",
                "datetime",
            )
        )
        style = ttk.Style()

        def fixed_map(option):
            return [
                elm
                for elm in style.map("Treeview", query_opt=option)
                if elm[:2] != ("!disabled", "!selected")
            ]

        style.map(
            "Treeview",
            foreground=fixed_map("foreground"),
            background=fixed_map("background"),
        )
        style.configure("Treeview.Heading", font="Arial")
        style.configure("Treeview", font="Arial")
        self.tv.column("Method", width=150, stretch=True)

        for col in ["information", "Estimated time cost", "datetime"]:
            self.tv.column(col, stretch=False)

        for col in self.tv["columns"]:
            self.tv.heading(col, text=col.title(), anchor=W)

        self.ybar.config(command=self.tv.yview)
        self.tv.pack(fill=X, pady=1)

        self.scroll_cf = CollapsingFrame(self.right_panel)
        self.scroll_cf.pack(fill=BOTH, expand=YES)

        self.output_container = ttk.Frame(self.scroll_cf, padding=1)
        self.scroll_cf.add(self.output_container, textvariable="scroll-message")

        self._value = (
            "Log: "
            + "STABox"
            + " is ready, the visualization results are displayed below......"
        )
        self.setvar("scroll-message", self._value)

        path = "../Renv_setting.yaml"
        if os.path.exists(path):
            with open(path, "r") as f:
                yamlData = yaml.safe_load(f)
            os.environ["R_HOME"] = yamlData["R_HOME"]
            os.environ["R_USER"] = yamlData["R_USER"]
        else:
            Messagebox.show_error(
                title="Warning", message="please setting R environ first!"
            )

    def Cluster_analysis(self):
        if self.method_flag == "STAGATE":
            if self.trained_file_path is not None:
                adata = sc.read(self.trained_file_path)
                sc.pp.neighbors(adata, use_rep="STAGATE")
                sc.tl.umap(adata)
                adata = mclust_R(
                    adata, used_obsm="STAGATE", num_cluster=self.Mcluster_num
                )
                sc.pp.pca(
                    adata, n_comps=50, use_highly_variable=True, svd_solver="arpack"
                )
                sc.pp.neighbors(adata, use_rep="X_pca")
                sc.tl.louvain(adata, resolution=self.louvain_res)
                kmeans = KMeans(n_clusters=self.Mcluster_num)
                kmeans.fit(adata.X)
                adata.obs["STAGATE_kmeans"] = kmeans.labels_
                adata.obs["STAGATE_kmeans"] = adata.obs["STAGATE_kmeans"].astype(
                    "category"
                )
                self.result_queue.put(adata)
                del adata

        elif self.method_flag == "STAligner":
            if self.trained_file_path is not None:
                adata_new = sc.read(self.trained_file_path)
                sc.tl.louvain(
                    adata_new,
                    random_state=666,
                    key_added="louvain",
                    resolution=float(self.louvain_res),
                )
                mclust_R(
                    adata_new, num_cluster=int(self.Mcluster_num), used_obsm="STAligner"
                )
                if self.label_files_exit:
                    # adata_list = sc.read(os.path.join(running_path, "result", "STAligner_GroundTruth_output.h5ad"))
                    adata_list = {}
                    key_add = []
                    print(self.multi_files)
                    for section_id in self.multi_files:
                        file_name = section_id.rsplit("\\", 1)[-1]
                        k = file_name.rsplit(".", 1)[-2]
                        key_add.append(k)
                        temp_adata = sc.read(section_id)
                        temp_adata.var_names_make_unique()
                        temp_adata.obs_names = [
                            x + "_" + k for x in temp_adata.obs_names
                        ]
                        adata_list[k] = temp_adata.copy()
                    # self.result_queue.put(adata_list)
                    # adata_list.write_h5ad(os.path.join(running_path, "result", "STAligner_GroundTruth_output.h5ad"))
                    # del temp_adata, adata_list
                    self.result_queue.put(adata_list)
                    del adata_list, temp_adata
                self.result_queue.put(adata_new)
                del adata_new

        elif self.method_flag == "STAMarker":
            if self.trained_file_path is not None:
                ann_data = sc.read(self.trained_file_path)
                self.result_queue.put(ann_data)
                del ann_data
        elif self.method_flag == "STAGE":
            if self.trained_file_path is not None:
                adata = sc.read(
                    os.path.join(running_path, "result", "STAGE_raw_output.h5ad")
                )
                adata_sample = sc.read(
                    os.path.join(
                        running_path, "result", "STAGE_adata_sample_output.h5ad"
                    )
                )
                adata_stage = sc.read(
                    os.path.join(
                        running_path, "result", "STAGE_adata_stage_output.h5ad"
                    )
                )
                self.result_queue.put(adata)
                self.result_queue.put(adata_sample)
                self.result_queue.put(adata_stage)
                del adata, adata_sample, adata_stage
        elif self.method_flag == "SpaGCN":
            if self.trained_file_path is not None:
                adata = sc.read(self.trained_file_path)
                sc.tl.louvain(adata, resolution=self.louvain_res)
                self.result_queue.put(adata)
                del adata
        elif self.method_flag == "SEDR":
            if self.trained_file_path is not None:
                try:
                    adata = sc.read(self.trained_file_path)
                    eval_cluster = int(self.Mcluster_num)
                    eval_resolution = res_search_fixed_clus(adata, eval_cluster)
                    sc.tl.leiden(
                        adata, key_added="SEDR_leiden", resolution=eval_resolution
                    )

                    kmeans = KMeans(n_clusters=self.kmeans_num)
                    kmeans.fit(adata.X)
                    adata.obs["SEDR_kmeans"] = kmeans.labels_
                    adata.obs["SEDR_kmeans"] = adata.obs["SEDR_kmeans"].astype("int")
                    adata.obs["SEDR_kmeans"] = adata.obs["SEDR_kmeans"].astype(
                        "category"
                    )

                    import rpy2.robjects as robjects

                    robjects.r.library("mclust")

                    import rpy2.robjects.numpy2ri

                    rpy2.robjects.numpy2ri.activate()

                    rmclust = robjects.r["Mclust"]
                    res2 = rmclust(adata.X, eval_cluster, "EEE")
                    mclust_res = np.array(res2[-2])

                    adata.obs["SEDR_mclust"] = mclust_res
                    # adata.obs['SEDR_mclust'] = mclust_R(adata_sedr.X, eval_cluster)
                    adata.obs["SEDR_mclust"] = adata.obs["SEDR_mclust"].astype("int")
                    adata.obs["SEDR_mclust"] = adata.obs["SEDR_mclust"].astype(
                        "category"
                    )

                    self.result_queue.put(adata)
                    del adata
                except:
                    raise "The data set cannot be analyzed by mcluster clustering method!!!"
        else:
            if self.trained_file_path is not None:
                adata = sc.read(self.trained_file_path)
                if self.label_files_exit:
                    adata = adata[adata.obs["GroundTruth"] == adata.obs["GroundTruth"],]
                else:
                    adata.obsm["spatial"] = adata.obsm["spatial"] * (-1)

                eval_cluster = int(self.Mcluster_num)
                eval_resolution = res_search_fixed_clus(adata, eval_cluster)
                sc.tl.leiden(
                    adata, key_added="SCANPY_leiden", resolution=eval_resolution
                )

                kmeans = KMeans(n_clusters=self.kmeans_num)
                kmeans.fit(adata.X)
                adata.obs["SCANPY_kmeans"] = kmeans.labels_
                adata.obs["SCANPY_kmeans"] = adata.obs["SCANPY_kmeans"].astype("int")
                adata.obs["SCANPY_kmeans"] = adata.obs["SCANPY_kmeans"].astype(
                    "category"
                )
                sc.tl.rank_genes_groups(adata, "leiden", inplace=True)
                self.result_queue.put(adata)
                del adata
        self.cluster_flag = True
        if self.file_path is None:
            self.file_path = self.trained_file_path
        self.pb.stop()
        self.pb["value"] = 100
        self.setvar("prog-message", "Cluster analysis run over!")

    def Cluster_analysis_thread(self):
        print(self.model_train_flag)
        # temp
        # self.model_train_flag = True
        if self.model_train_flag:
            # self.result_flag = True
            self.pb.start()
            self.setvar(
                "prog-message", f"{self.method_flag}: cluster analysis underway!!"
            )
            T = threading.Thread(target=self.Cluster_analysis)
            T.start()

    def data_download_GUI(self):
        self.path_load_flag = False
        datadown_gui = ttk.Toplevel(self.downloadbtn)
        datadown_gui.title("Data download")
        datadown_gui.geometry("1050x350")
        style = Style()
        style.configure("TCheckbutton", font=("Arial", 10))
        style.configure("TButton", font=("Arial", 10))
        style.configure("TLabel", font=("Arial", 10))
        style.configure("TRadiobutton", font=("Arial", 10))
        style.configure("TCombobox", font=("Arial", 10))
        database_select_info_frame = ttk.Frame(datadown_gui)
        database_select_info_frame.pack(side=TOP, anchor="nw")

        database_select_info = ttk.Label(
            database_select_info_frame, text="Select ST Database: "
        )
        database_select_info.pack(side=LEFT)

        database_select_frame = ttk.Frame(datadown_gui)
        database_select_frame.pack(anchor="center")
        sodb = SODB()
        spatialdb = SpatialDB()

        def on_radio_select(value):
            if self.path_load_flag:
                self.data_download_frame.destroy()
                self.path_load_flag = False
            if value == "SODB":
                self.data_download_frame = ttk.Frame(datadown_gui)
                self.data_download_frame.pack(after=database_select_frame)
                data_download_frame = ttk.Frame(self.data_download_frame)
                data_download_frame.pack(side=TOP)
                comfirm_btn_frame = ttk.Frame(self.data_download_frame)
                comfirm_btn_frame.pack(anchor="center")
                catagory = [
                    "Spatial Transcriptomics",
                    "Spatial Proteomics",
                    "Spatial Metabolomics",
                    "Spatial Genomics",
                    "Spatial MultiOmics",
                ]
                catagory_label = ttk.Label(data_download_frame, text="Data catagory:")
                catagory_label.grid(row=0, column=0)

                def set_data_type_download(event):
                    print("Current choose: {}".format(catagory_combobox.get()))
                    list_experiment_label = ttk.Label(
                        data_download_frame, text="Data experiment:"
                    )
                    list_experiment_label.grid(row=0, column=2)

                    def set_data_experiment_download(event):
                        print(
                            "Current choose: {}".format(list_experiment_combobox.get())
                        )
                        list_data_label = ttk.Label(
                            data_download_frame, text="Data type:"
                        )
                        list_data_label.grid(row=0, column=4)

                        def data_type_download(event):
                            print("Current choose: {}".format(list_data_combobox.get()))
                            if list_data_combobox.get() is not None:

                                def download_data():
                                    # adata = sodb.load_experiment(list_experiment_combobox.get(),list_data_combobox.get())
                                    scrollbar = ttk.Scrollbar(comfirm_btn_frame)
                                    scrollbar.pack(side=RIGHT, fill=Y)
                                    text_box = ttk.Text(
                                        comfirm_btn_frame, yscrollcommand=scrollbar.set
                                    )
                                    text_box.pack(after=btn, fill=BOTH, expand=True)
                                    # scrollbar = tk.Scrollbar(root)
                                    # scrollbar.pack(side=RIGHT, fill=Y)
                                    scrollbar.config(command=text_box.yview)

                                    text_box.insert(tk.END, f"Current database: SODB\n")
                                    text_box.insert(
                                        tk.END,
                                        f"Current data catagory: {catagory_combobox.get()}\n",
                                    )
                                    text_box.insert(
                                        tk.END,
                                        f"Current technology: {list_experiment_combobox.get()}\n",
                                    )
                                    text_box.insert(
                                        tk.END,
                                        f"Current data type: {list_data_combobox.get()}\n",
                                    )
                                    text_box.insert(
                                        tk.END,
                                        f"Current data saved path: {os.path.join(running_path, 'dataset', 'cache')}\n",
                                    )

                                    self.pb.start()
                                    self.setvar("prog-message", "Data start download!!")
                                    self.data_download_thread(
                                        sodb,
                                        "SODB",
                                        list_experiment_combobox.get(),
                                        list_data_combobox.get(),
                                    )

                                btn = ttk.Button(
                                    master=comfirm_btn_frame,
                                    text="Confirm",
                                    command=download_data,
                                )
                                btn.pack(side=TOP)

                        if list_experiment_combobox.get() is not None:
                            datas = sodb.list_experiment_by_dataset(
                                list_experiment_combobox.get()
                            )
                            list_data_combobox = ttk.Combobox(
                                master=data_download_frame,
                                textvariable="select_list_data",
                                font=("Arial", 10),
                                values=datas,
                                state="normal",
                                cursor="plus",
                            )
                            list_data_combobox.current(0)
                            list_data_combobox.grid(row=0, column=5)

                            list_data_combobox.bind(
                                "<<ComboboxSelected>>", data_type_download
                            )

                    if catagory_combobox.get() is not None:
                        data_list = sodb.list_dataset_by_category(
                            catagory_combobox.get()
                        )

                        list_experiment_combobox = ttk.Combobox(
                            master=data_download_frame,
                            textvariable="select_list_experiment",
                            font=("Arial", 10),
                            values=data_list,
                            state="normal",
                            cursor="plus",
                        )
                        list_experiment_combobox.current(0)
                        list_experiment_combobox.grid(row=0, column=3)

                        list_experiment_combobox.bind(
                            "<<ComboboxSelected>>", set_data_experiment_download
                        )

                catagory_combobox = ttk.Combobox(
                    master=data_download_frame,
                    textvariable="select_data_catagory",
                    font=("Arial", 10),
                    values=catagory,
                    state="normal",
                    cursor="plus",
                )
                catagory_combobox.current(0)
                catagory_combobox.grid(row=0, column=1)

                catagory_combobox.bind("<<ComboboxSelected>>", set_data_type_download)
                self.path_load_flag = True
            else:
                self.data_download_frame = ttk.Frame(datadown_gui)
                self.data_download_frame.pack(after=database_select_frame)
                data_download_frame = ttk.Frame(self.data_download_frame)
                data_download_frame.pack(side=TOP)
                comfirm_btn_frame = ttk.Frame(self.data_download_frame)
                comfirm_btn_frame.pack(anchor="center")
                catagory_label = ttk.Label(data_download_frame, text="Data catagory:")
                catagory_label.grid(row=0, column=0)

                def set_data_type_download(event):
                    print("Current choose: {}".format(catagory_combobox.get()))
                    list_experiment_label = ttk.Label(
                        data_download_frame, text="Data experiment:"
                    )
                    list_experiment_label.grid(row=0, column=2)

                    def set_data_experiment_download(event):
                        if list_experiment_combobox.get() is not None:
                            print(
                                "Current choose: {}".format(
                                    list_experiment_combobox.get()
                                )
                            )
                            if list_experiment_combobox.get() is not None:

                                def download_data():
                                    # spatialdb.download(list_experiment_combobox.get())
                                    scrollbar = ttk.Scrollbar(comfirm_btn_frame)
                                    scrollbar.pack(side=RIGHT, fill=Y)
                                    text_box = ttk.Text(
                                        comfirm_btn_frame, yscrollcommand=scrollbar.set
                                    )
                                    text_box.pack(
                                        after=btn, side=TOP, fill=BOTH, expand=True
                                    )
                                    scrollbar.config(command=text_box.yview)

                                    text_box.insert(
                                        tk.END, f"Current database: SpatialDB\n"
                                    )
                                    text_box.insert(
                                        tk.END,
                                        f"Current data catagory: {catagory_combobox.get()}\n",
                                    )
                                    text_box.insert(
                                        tk.END,
                                        f"Current data type: {list_experiment_combobox.get()}\n",
                                    )
                                    # text_box.insert(tk.END, f"Current data type: {list_data_combobox.get()}\n")
                                    text_box.insert(
                                        tk.END,
                                        f"Current data saved path: {os.path.join(running_path, 'dataset', 'cache')}\n",
                                    )

                                    self.pb.start()
                                    self.setvar("prog-message", "Data start download!!")
                                    self.data_download_thread(
                                        spatialdb,
                                        "SpatialDB",
                                        None,
                                        list_experiment_combobox.get(),
                                    )

                                btn = ttk.Button(
                                    master=comfirm_btn_frame,
                                    text="Confirm",
                                    command=download_data,
                                )
                                btn.pack(side=TOP)

                    if catagory_combobox.get() is not None:
                        data_list = spatialdb.get_download_data_info(
                            catagory_combobox.get()
                        )
                        data_list = [i.split(".", 1)[0] for i in data_list]

                        list_experiment_combobox = ttk.Combobox(
                            master=data_download_frame,
                            textvariable="select_list_experiment",
                            font=("Arial", 10),
                            values=data_list,
                            state="normal",
                            cursor="plus",
                        )
                        list_experiment_combobox.current(0)
                        list_experiment_combobox.grid(row=0, column=3)

                        list_experiment_combobox.bind(
                            "<<ComboboxSelected>>", set_data_experiment_download
                        )

                catagory_combobox = ttk.Combobox(
                    master=data_download_frame,
                    textvariable="select_data_catagory",
                    font=("Arial", 10),
                    values=spatialdb.get_download_data_type(),
                    state="normal",
                    cursor="plus",
                )
                catagory_combobox.current(0)
                catagory_combobox.grid(row=0, column=1)
                catagory_combobox.bind("<<ComboboxSelected>>", set_data_type_download)
                self.path_load_flag = True

        radio_var = tk.StringVar()
        radio1 = ttk.Radiobutton(
            database_select_frame,
            text="Download ST data from SODB",
            style="TCheckbutton",
            variable=radio_var,
            value="SODB",
            command=lambda: on_radio_select("SODB"),
        )
        radio1.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        radio2 = ttk.Radiobutton(
            database_select_frame,
            text="Download ST data from SpatialDB",
            style="TCheckbutton",
            variable=radio_var,
            value="SpatialDB",
            command=lambda: on_radio_select("SpatialDB"),
        )
        radio2.grid(row=1, column=0, padx=5, pady=5, sticky="w")

        def on_closing():
            self.path_load_flag = False
            datadown_gui.destroy()

        datadown_gui.protocol("WM_DELETE_WINDOW", on_closing)

    def data_download(self, db_class, db_flag, experiment_type, data_type):
        if db_flag == "SODB":
            db_class.load_experiment(experiment_type, data_type)
            self.setvar("prog-message", f"Data download complete!!")
            self.pb.stop()
            self.pb["value"] = 100
        else:
            db_class.download(data_type)
            self.setvar("prog-message", f"Data download complete!!")
            self.pb.stop()
            self.pb["value"] = 100

    def data_download_thread(self, db_class, db_flag, experiment_type, data_type):
        self.pb.start()
        self.setvar("prog-message", f"Download data from {db_flag}!!")
        T = threading.Thread(
            target=self.data_download,
            args=(db_class, db_flag, experiment_type, data_type),
        )
        T.start()

    def remove_images(self):
        if self.model_train_flag:
            if self.result_flag:
                self.figure_Frame.destroy()
                self.result_flag = False
                self.cluster_flag = False
        else:
            Messagebox.show_warning(
                title="Attention", message="Waiting for model training!!!"
            )

    def get_trained_file_path(self):
        self.update_idletasks()
        self.trained_file_path = filedialog.askopenfilename()
        print(f"trained_file_path is {self.trained_file_path}")
        self.setvar("load", self.trained_file_path)

    def Select_files(self):
        files_show = ttk.Toplevel(self.bus_frm)
        files_show.title("Files select")
        files_show.geometry("300x250")
        frame0 = tk.Frame(files_show)
        frame0.pack()
        path_label = tk.Label(frame0, text=f"Files in {self.multi_files}:  ")
        path_label.grid(row=0, column=0)

        frame1 = ttk.Frame(files_show)
        frame1.pack(side="bottom")

        file_list = os.listdir(self.multi_files)
        print(file_list)

        def getselect(item):
            print(item, "selected")

        def unselectall():
            for index, item in enumerate(list1):
                v[index].set("")

        def selectall():
            for index, item in enumerate(list1):
                v[index].set(item)

        def showselect():
            selected = [i.get() for i in v if i.get()]
            print(selected)
            self.multi_files = [os.path.join(self.multi_files, i) for i in selected]
            print(self.multi_files)
            files_show.destroy()
            tk.messagebox.showwarning(
                title="Attention", message=f"you have selected {len(selected)} file!"
            )
            self.path_load_flag = True

        canvas = tk.Canvas(files_show)
        canvas.pack(side="left", fill="both", expand=True)

        scrollbar = ttk.Scrollbar(files_show, orient="vertical", command=canvas.yview)
        scrollbar.pack(side="right", fill="y")
        canvas.configure(yscrollcommand=scrollbar.set, width=60, height=30)

        frame2 = ttk.Frame(canvas)
        canvas.create_window((0, 0), window=frame2, anchor="nw")

        opt = tk.IntVar()
        list1 = [i for i in file_list]
        v = []

        for index, item in enumerate(list1):
            v.append(tk.StringVar())
            ttk.Checkbutton(
                frame2,
                text=item,
                variable=v[-1],
                onvalue=item,
                offvalue="",
                command=lambda item=item: getselect(item),
            ).grid(row=index, column=0, sticky="w")

        seltone = ttk.Radiobutton(
            frame1, text="All", variable=opt, value=1, command=selectall
        )
        seltone.grid(row=0, column=0)
        selttwo = ttk.Radiobutton(
            frame1, text="Cancel", variable=opt, value=0, command=unselectall
        )
        selttwo.grid(row=0, column=1)
        btn = ttk.Button(frame1, text="Confirm", command=showselect)
        btn.grid(row=0, column=2)
        frame2.update_idletasks()
        canvas.config(scrollregion=canvas.bbox("all"))

    def Dataload_thread(self):
        T = threading.Thread(target=self.Dataload)
        T.setDaemon(True)
        T.start()

    def Dataload(self):
        if self.data_load_flag:
            adata = self.result_queue.get()
            # print('Dataload self.result_queue.qsize()', self.result_queue.qsize())
            # print('self.file_path', self.file_path)
            self.paned_window.destroy()
        self.Visium_data_process()

    def Data_Preprocess_thread(self):
        T = threading.Thread(target=self.load_data)
        T.setDaemon(True)
        T.start()

    def load_data(self):
        self.path_load_flag = False
        style = Style()
        style.configure("TCheckbutton", font=("Arial", 16))
        style.configure("TButton", font=("Arial", 16))
        self.Data_load_toplevel = ttk.Toplevel(self.backup)
        self.Data_load_toplevel.attributes("-topmost", "true")
        self.Data_load_toplevel.title("Data Loading")
        self.Data_load_toplevel.geometry("1000x450")
        self.radio_moduel = tk.Frame(self.Data_load_toplevel)
        self.radio_moduel.pack(side=TOP)

        def on_radio_select(value):
            if self.path_load_flag:
                self.load_moduel.destroy()
                self.path_load_flag = False
            if value == "txt_csv_to_h5ad":
                self.path_load_flag = True
                self.load_moduel = tk.Frame(self.Data_load_toplevel)
                self.load_moduel.pack(after=self.radio_moduel)
                file_label_one = ttk.Label(
                    self.load_moduel, text="Count txt file path:", font=("Arial", 16)
                )
                file_label_one.grid(row=0, column=0, padx=5, pady=5, sticky="w")

                file_entry_one = ttk.Entry(self.load_moduel)
                file_entry_one.grid(row=0, column=1, padx=5, pady=5)

                file_label_two = ttk.Label(
                    self.load_moduel, text="Location csv file path:", font=("Arial", 16)
                )
                file_label_two.grid(row=1, column=0, padx=5, pady=5, sticky="w")

                file_entry_two = ttk.Entry(self.load_moduel)
                file_entry_two.grid(row=1, column=1, padx=5, pady=5)

                def browse_folder_one():
                    folder_path = filedialog.askopenfilename()
                    file_entry_one.delete(0, tk.END)
                    file_entry_one.insert(0, folder_path)

                def browse_folder_two():
                    folder_path = filedialog.askopenfilename()
                    file_entry_two.delete(0, tk.END)
                    file_entry_two.insert(0, folder_path)

                confirm_button_one = ttk.Button(
                    self.load_moduel,
                    text="Choose",
                    style="TButton",
                    command=browse_folder_one,
                )
                confirm_button_one.grid(row=0, column=2, padx=5, pady=5)
                confirm_button_two = ttk.Button(
                    self.load_moduel,
                    text="Choose",
                    style="TButton",
                    command=browse_folder_two,
                )
                confirm_button_two.grid(row=1, column=2, padx=5, pady=5)

                def load_datas():
                    count = pd.read_csv(file_entry_one.get(), sep="\t", index_col=0)
                    location = pd.read_csv(file_entry_two.get(), index_col=0)
                    adata = sc.AnnData(count.T)
                    adata.var_names_make_unique()
                    coor_df = location.loc[adata.obs_names, ["xcoord", "ycoord"]]
                    adata.obsm["spatial"] = coor_df.to_numpy()
                    sc.pp.normalize_total(adata, target_sum=1e4)
                    sc.pp.log1p(adata)
                    sc.pp.calculate_qc_metrics(adata, inplace=True)
                    path = file_entry_one.get()
                    adata.write_h5ad(path.rsplit("/", 1)[-2] + "/new_adata.h5ad")
                    self.path_load_flag = False
                    self.Data_load_toplevel.destroy()
                    Messagebox.show_warning(
                        title="Attention", message="Files has been converted to h5ad!!!"
                    )

                confirm_button = ttk.Button(
                    self.load_moduel,
                    text="Confirm",
                    style="TButton",
                    command=load_datas,
                )
                confirm_button.grid(row=2, column=1, padx=5, pady=5)

            elif value == "h5_to_h5ad":
                self.path_load_flag = True
                self.load_moduel = tk.Frame(self.Data_load_toplevel)
                self.load_moduel.pack(after=self.radio_moduel)
                file_label_one = ttk.Label(
                    self.load_moduel, text="H5file folder Path:", font=("Arial", 16)
                )
                file_label_one.grid(row=0, column=0, padx=5, pady=5)

                file_entry = ttk.Entry(self.load_moduel)
                file_entry.grid(row=0, column=1, padx=5, pady=5)

                def browse_folder():
                    folder_path = filedialog.askdirectory()
                    file_entry.delete(0, tk.END)
                    file_entry.insert(0, folder_path)

                confirm_button_one = ttk.Button(
                    self.load_moduel,
                    text="Choose",
                    style="TButton",
                    command=browse_folder,
                )
                confirm_button_one.grid(row=0, column=2, padx=5, pady=5)

                def load_datas():
                    h5file = glob.glob(file_entry.get() + "/*.h5")
                    adata = sc.read_visium(
                        path=file_entry.get(), count_file=h5file[0].rsplit("\\", 1)[-1]
                    )
                    adata.var_names_make_unique()
                    sc.pp.normalize_total(adata, target_sum=1e4)
                    sc.pp.log1p(adata)
                    sc.pp.calculate_qc_metrics(adata, inplace=True)
                    adata.write_h5ad(file_entry.get() + "/new_adata.h5ad")
                    self.path_load_flag = False
                    self.Data_load_toplevel.destroy()
                    Messagebox.show_warning(
                        title="Attention", message="Files has been converted to h5ad!!!"
                    )

                confirm_button = ttk.Button(
                    self.load_moduel,
                    text="Confirm",
                    style="TButton",
                    command=load_datas,
                )
                confirm_button.grid(row=2, column=1, padx=5, pady=5)

            elif value == "txt_to_h5ad" or value == "tsv_to_h5ad":
                self.path_load_flag = True
                self.load_moduel = tk.Frame(self.Data_load_toplevel)
                self.load_moduel.pack(after=self.radio_moduel)
                file_label_one = ttk.Label(
                    self.load_moduel, text="Count.tsv file Path:", font=("Arial", 16)
                )
                file_label_one.grid(row=0, column=0, padx=5, pady=5, sticky="w")

                file_entry_one = ttk.Entry(self.load_moduel)
                file_entry_one.grid(row=0, column=1, padx=5, pady=5)

                file_label_two = ttk.Label(
                    self.load_moduel, text="Location.tsv file Path:", font=("Arial", 16)
                )
                file_label_two.grid(row=1, column=0, padx=5, pady=5, sticky="w")

                file_entry_two = ttk.Entry(self.load_moduel)
                file_entry_two.grid(row=1, column=1, padx=5, pady=5)

                def browse_folder_one():
                    folder_path = filedialog.askopenfilename()
                    file_entry_one.delete(0, tk.END)
                    file_entry_one.insert(0, folder_path)

                def browse_folder_two():
                    folder_path = filedialog.askopenfilename()
                    file_entry_two.delete(0, tk.END)
                    file_entry_two.insert(0, folder_path)

                confirm_button_one = ttk.Button(
                    self.load_moduel,
                    text="Choose",
                    style="TButton",
                    command=browse_folder_one,
                )
                confirm_button_one.grid(row=0, column=2, padx=5, pady=5)
                confirm_button_two = ttk.Button(
                    self.load_moduel,
                    text="Choose",
                    style="TButton",
                    command=browse_folder_two,
                )
                confirm_button_two.grid(row=1, column=2, padx=5, pady=5)

                def load_datas():
                    count = pd.read_csv(file_entry_one.get(), sep="\t")
                    location = pd.read_csv(file_entry_two.get(), sep="\t")
                    adata = sc.AnnData(count)
                    adata.var_names_make_unique()
                    adata.obsm["spatial"] = location.to_numpy()
                    sc.pp.normalize_total(adata, target_sum=1e4)
                    sc.pp.log1p(adata)
                    path = file_entry_one.get()
                    sc.pp.calculate_qc_metrics(adata, inplace=True)
                    adata.write_h5ad(path.rsplit("/", 1)[-2] + "/new_adata.h5ad")
                    self.path_load_flag = False
                    self.Data_load_toplevel.destroy()
                    Messagebox.show_warning(
                        title="Attention", message="Files has been converted to h5ad!!!"
                    )

                confirm_button = ttk.Button(
                    self.load_moduel,
                    text="Confirm",
                    style="TButton",
                    command=load_datas,
                )
                confirm_button.grid(row=2, column=1, padx=5, pady=5)

            elif value == "SpatialDB_file_to_h5ad":
                self.path_load_flag = True
                self.load_moduel = tk.Frame(self.Data_load_toplevel)
                self.load_moduel.pack(after=self.radio_moduel)
                file_label_one = ttk.Label(
                    self.load_moduel, text="Count.count file Path:", font=("Arial", 16)
                )
                file_label_one.grid(row=0, column=0, padx=5, pady=5, sticky="w")

                file_entry_one = ttk.Entry(self.load_moduel)
                file_entry_one.grid(row=0, column=1, padx=5, pady=5)

                file_label_two = ttk.Label(
                    self.load_moduel, text="Location.idx file Path:", font=("Arial", 16)
                )
                file_label_two.grid(row=1, column=0, padx=5, pady=5, sticky="w")

                file_entry_two = ttk.Entry(self.load_moduel)
                file_entry_two.grid(row=1, column=1, padx=5, pady=5)

                def browse_folder_one():
                    folder_path = filedialog.askopenfilename()
                    file_entry_one.delete(0, tk.END)
                    file_entry_one.insert(0, folder_path)

                def browse_folder_two():
                    folder_path = filedialog.askopenfilename()
                    file_entry_two.delete(0, tk.END)
                    file_entry_two.insert(0, folder_path)

                confirm_button_one = ttk.Button(
                    self.load_moduel,
                    text="Choose",
                    style="TButton",
                    command=browse_folder_one,
                )
                confirm_button_one.grid(row=0, column=2, padx=5, pady=5)
                confirm_button_two = ttk.Button(
                    self.load_moduel,
                    text="Choose",
                    style="TButton",
                    command=browse_folder_two,
                )
                confirm_button_two.grid(row=1, column=2, padx=5, pady=5)

                def load_datas():
                    count = pd.read_csv(file_entry_one.get(), sep=",", index_col=0)
                    count = count.T
                    path = file_entry_one.get()
                    with open(path.rsplit("/", 1)[-2] + "/gene_id.txt", "w") as f:
                        for item in list(count.columns):
                            f.write("%s\n" % item)
                    count.columns = count.iloc[0]
                    count = count[1:]
                    location = pd.read_csv(file_entry_two.get(), sep=",", index_col=0)
                    adata = sc.AnnData(count.values)
                    adata.obs_names = [i for i in list(count.index)]
                    adata.var_names = [i for i in list(count.columns)]
                    adata.obsm["spatial"] = location[["x", "y"]].values
                    # sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=2000)
                    sc.pp.normalize_total(adata, target_sum=1e4)
                    sc.pp.log1p(adata)
                    adata.write_h5ad(
                        path.rsplit("/", 1)[-2] + "/Transfor_new_data.h5ad"
                    )
                    self.path_load_flag = False
                    self.Data_load_toplevel.destroy()
                    Messagebox.show_warning(
                        title="Attention", message="Files has been converted to h5ad!!!"
                    )

                confirm_button = ttk.Button(
                    self.load_moduel,
                    text="Confirm",
                    style="TButton",
                    command=load_datas,
                )
                confirm_button.grid(row=2, column=1, padx=5, pady=5)
            else:
                pass

        radio_var = tk.StringVar()
        radio1 = ttk.Radiobutton(
            self.radio_moduel,
            text="Raw count txt and location csv to h5ad file",
            style="TCheckbutton",
            variable=radio_var,
            value="txt_csv_to_h5ad",
            command=lambda: on_radio_select("txt_csv_to_h5ad"),
        )
        radio1.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        radio2 = ttk.Radiobutton(
            self.radio_moduel,
            text="Raw h5file to h5ad file",
            style="TCheckbutton",
            variable=radio_var,
            value="h5_to_h5ad",
            command=lambda: on_radio_select("h5_to_h5ad"),
        )
        radio2.grid(row=1, column=0, padx=5, pady=5, sticky="w")
        radio3 = ttk.Radiobutton(
            self.radio_moduel,
            text="Raw count txt and location txt to h5ad file",
            style="TCheckbutton",
            variable=radio_var,
            value="txt_to_h5ad",
            command=lambda: on_radio_select("txt_to_h5ad"),
        )
        radio3.grid(row=2, column=0, padx=5, pady=5, sticky="w")
        radio4 = ttk.Radiobutton(
            self.radio_moduel,
            text="Raw count tsv and location tsv to h5ad file",
            style="TCheckbutton",
            variable=radio_var,
            value="tsv_to_h5ad",
            command=lambda: on_radio_select("tsv_to_h5ad"),
        )
        radio4.grid(row=3, column=0, padx=5, pady=5, sticky="w")

        radio5 = ttk.Radiobutton(
            self.radio_moduel,
            text="Raw count.count file and location.idx file to h5ad file",
            style="TCheckbutton",
            variable=radio_var,
            value="SpatialDB_file_to_h5ad",
            command=lambda: on_radio_select("SpatialDB_file_to_h5ad"),
        )
        radio5.grid(row=4, column=0, padx=5, pady=5, sticky="w")

        def on_closing():
            self.path_load_flag = False
            self.Data_load_toplevel.destroy()

        self.Data_load_toplevel.protocol("WM_DELETE_WINDOW", on_closing)

    def Restart(self):
        self.master.quit()
        self.master.destroy()
        python = sys.executable
        subprocess.Popen([python] + sys.argv)

    def Preprocess(self):
        self.path_load_flag = False
        self.Data_load_toplevel = ttk.Toplevel(self.backup)
        self.Data_load_toplevel.attributes("-topmost", "true")
        self.Data_load_toplevel.title("Data Preprocess")
        self.Data_load_toplevel.geometry("550x320")
        self.radio_moduel = tk.Frame(self.Data_load_toplevel)
        self.radio_moduel.pack(side=TOP)
        style = Style()
        style.configure("TCheckbutton", font=("Arial", 16))
        style.configure("TButton", font=("Arial", 16))

        def on_radio_select(value):
            if self.path_load_flag:
                self.load_moduel.destroy()
                self.path_load_flag = False
            if value == "add_location":
                self.path_load_flag = True
                self.load_moduel = tk.Frame(self.Data_load_toplevel)
                self.load_moduel.pack(after=self.radio_moduel)
                file_label_one = ttk.Label(
                    self.load_moduel, text="H5ad file path:", font=("Arial", 16)
                )  # 创建Label控件
                file_label_one.grid(row=0, column=0, padx=5, pady=5, sticky="w")

                file_entry_one = ttk.Entry(self.load_moduel)  # 创建Entry控件
                file_entry_one.grid(row=0, column=1, padx=5, pady=5)

                file_label_two = ttk.Label(
                    self.load_moduel, text="Location csv file Path:", font=("Arial", 16)
                )  # 创建Label控件
                file_label_two.grid(row=1, column=0, padx=5, pady=5, sticky="w")

                file_entry_two = ttk.Entry(self.load_moduel)  # 创建Entry控件
                file_entry_two.grid(row=1, column=1, padx=5, pady=5)

                file_label_three = ttk.Label(
                    self.load_moduel,
                    text="Filter highly variable genes:",
                    font=("Arial", 16),
                )  # 创建Label控件
                file_label_three.grid(row=2, column=0, padx=5, pady=5, sticky="w")

                file_entry_three = ttk.Entry(self.load_moduel)  # 创建Entry控件
                file_entry_three.grid(row=2, column=1, padx=5, pady=5)

                def browse_folder_one():
                    folder_path = filedialog.askopenfilename()
                    file_entry_one.delete(0, tk.END)
                    file_entry_one.insert(0, folder_path)

                def browse_folder_two():
                    folder_path = filedialog.askopenfilename()
                    file_entry_two.delete(0, tk.END)
                    file_entry_two.insert(0, folder_path)

                confirm_button_one = ttk.Button(
                    self.load_moduel,
                    text="Choose",
                    style="TButton",
                    command=browse_folder_one,
                )  # 创建确认按钮
                confirm_button_one.grid(row=0, column=2, padx=5, pady=5)
                confirm_button_two = ttk.Button(
                    self.load_moduel,
                    text="Choose",
                    style="TButton",
                    command=browse_folder_two,
                )  # 创建确认按钮
                confirm_button_two.grid(row=1, column=2, padx=5, pady=5)

                def load_datas():
                    adata = sc.read(file_entry_one.get(), sep="\t", index_col=0)
                    location = pd.read_csv(file_entry_two.get(), sep=",", index_col=0)
                    adata.var_names_make_unique()
                    coor_df = location.loc[adata.obs_names, ["x", "y"]]
                    adata.obsm["spatial"] = coor_df.to_numpy()
                    sc.pp.normalize_total(adata, target_sum=1e4)
                    sc.pp.log1p(adata)
                    sc.pp.calculate_qc_metrics(adata, inplace=True)
                    if file_entry_three.get():
                        sc.pp.highly_variable_genes(
                            adata,
                            flavor="seurat_v3",
                            n_top_genes=int(file_entry_three.get()),
                        )
                    if "highly_variable" in adata.var.columns:
                        adata = adata[:, adata.var["highly_variable"]]
                    path = file_entry_one.get()
                    adata.write_h5ad(path.rsplit("/", 1)[-2] + "/new_adata.h5ad")
                    self.path_load_flag = False
                    self.Data_load_toplevel.destroy()
                    Messagebox.show_warning(
                        title="Attention",
                        message="Information has been added into h5ad files!!!",
                    )

                confirm_button = ttk.Button(
                    self.load_moduel,
                    text="Confirm",
                    style="TButton",
                    command=load_datas,
                )
                confirm_button.grid(row=3, column=1, padx=5, pady=5)

            elif value == "add_GroundTurth":
                self.path_load_flag = True
                self.load_moduel = tk.Frame(self.Data_load_toplevel)
                self.load_moduel.pack(after=self.radio_moduel)
                file_label_one = ttk.Label(
                    self.load_moduel, text="H5ad file path:", font=("Arial", 16)
                )  # 创建Label控件
                file_label_one.grid(row=0, column=0, padx=5, pady=5, sticky="w")

                file_entry_one = ttk.Entry(self.load_moduel)
                file_entry_one.grid(row=0, column=1, padx=5, pady=5)

                file_label_two = ttk.Label(
                    self.load_moduel,
                    text="GroundTruth txt file Path:",
                    font=("Arial", 16),
                )  # 创建Label控件
                file_label_two.grid(row=1, column=0, padx=5, pady=5, sticky="w")

                file_entry_two = ttk.Entry(self.load_moduel)
                file_entry_two.grid(row=1, column=1, padx=5, pady=5)

                file_label_three = ttk.Label(
                    self.load_moduel,
                    text="Filter highly variable genes:",
                    font=("Arial", 16),
                )  # 创建Label控件
                file_label_three.grid(row=2, column=0, padx=5, pady=5, sticky="w")

                file_entry_three = ttk.Entry(self.load_moduel)
                file_entry_three.grid(row=2, column=1, padx=5, pady=5)

                def browse_folder_one():
                    folder_path = filedialog.askopenfilename()
                    file_entry_one.delete(0, tk.END)
                    file_entry_one.insert(0, folder_path)

                def browse_folder_two():
                    folder_path = filedialog.askopenfilename()
                    file_entry_two.delete(0, tk.END)
                    file_entry_two.insert(0, folder_path)

                confirm_button_one = ttk.Button(
                    self.load_moduel,
                    text="Choose",
                    style="TButton",
                    command=browse_folder_one,
                )  # 创建确认按钮
                confirm_button_one.grid(row=0, column=2, padx=5, pady=5)

                confirm_button_two = ttk.Button(
                    self.load_moduel,
                    text="Choose",
                    style="TButton",
                    command=browse_folder_two,
                )  # 创建确认按钮
                confirm_button_two.grid(row=1, column=2, padx=5, pady=5)

                def load_datas():
                    adata = sc.read(file_entry_one.get())
                    adata.var_names_make_unique()
                    Ann_df = pd.read_csv(
                        file_entry_two.get(), sep="\t", header=None, index_col=0
                    )
                    Ann_df.columns = ["Ground Truth"]
                    adata.obs["GroundTruth"] = Ann_df.loc[
                        adata.obs_names, "Ground Truth"
                    ]
                    sc.pp.normalize_total(adata, target_sum=1e4)
                    sc.pp.log1p(adata)
                    sc.pp.calculate_qc_metrics(adata, inplace=True)
                    if file_entry_three.get():
                        sc.pp.highly_variable_genes(
                            adata,
                            flavor="seurat_v3",
                            n_top_genes=int(file_entry_three.get()),
                        )
                    if "highly_variable" in adata.var.columns:
                        adata = adata[:, adata.var["highly_variable"]]
                    path = file_entry_one.get()
                    adata.write_h5ad(path.rsplit("/", 1)[-2] + "/new_adata.h5ad")
                    del adata
                    self.path_load_flag = False
                    self.Data_load_toplevel.destroy()
                    Messagebox.show_warning(
                        title="Attention",
                        message="Information has been added into h5ad files!!!",
                    )

                confirm_button = ttk.Button(
                    self.load_moduel,
                    text="Confirm",
                    style="TButton",
                    command=load_datas,
                )
                confirm_button.grid(row=3, column=1, padx=5, pady=5)
            else:
                self.path_load_flag = True
                self.load_moduel = tk.Frame(self.Data_load_toplevel)
                self.load_moduel.pack(after=self.radio_moduel)
                file_label_one = ttk.Label(
                    self.load_moduel, text="H5ad file path:", font=("Arial", 16)
                )  # 创建Label控件
                file_label_one.grid(row=0, column=0, padx=5, pady=5, sticky="w")

                file_entry_one = ttk.Entry(self.load_moduel)
                file_entry_one.grid(row=0, column=1, padx=5, pady=5)

                file_label_two = ttk.Label(
                    self.load_moduel,
                    text="Filter highly variable genes:",
                    font=("Arial", 16),
                )
                file_label_two.grid(row=1, column=0, padx=5, pady=5, sticky="w")

                file_entry_two = ttk.Entry(self.load_moduel)
                file_entry_two.grid(row=1, column=1, padx=5, pady=5)

                def browse_folder_one():
                    folder_path = filedialog.askopenfilename()
                    file_entry_one.delete(0, tk.END)
                    file_entry_one.insert(0, folder_path)

                confirm_button_one = ttk.Button(
                    self.load_moduel,
                    text="Choose",
                    style="TButton",
                    command=browse_folder_one,
                )
                confirm_button_one.grid(row=0, column=2, padx=5, pady=5)

                def load_datas():
                    adata = sc.read(file_entry_one.get())
                    adata.var_names_make_unique()
                    sc.pp.normalize_total(adata, target_sum=1e4)
                    sc.pp.log1p(adata)
                    sc.pp.calculate_qc_metrics(adata, inplace=True)
                    if file_entry_two.get():
                        sc.pp.highly_variable_genes(
                            adata,
                            flavor="seurat_v3",
                            n_top_genes=int(file_entry_two.get()),
                        )
                    if "highly_variable" in adata.var.columns:
                        adata = adata[:, adata.var["highly_variable"]]
                    path = file_entry_one.get()
                    adata.write_h5ad(path.rsplit("/", 1)[-2] + "/new_adata.h5ad")
                    self.path_load_flag = False
                    self.Data_load_toplevel.destroy()
                    Messagebox.show_warning(
                        title="Attention",
                        message="Information has been added into h5ad files!!!",
                    )

                confirm_button = ttk.Button(
                    self.load_moduel,
                    text="Confirm",
                    style="TButton",
                    command=load_datas,
                )
                confirm_button.grid(row=2, column=1, padx=5, pady=5)

        radio_var = tk.StringVar()
        radio1 = ttk.Radiobutton(
            self.radio_moduel,
            text="H5ad file add location csv information",
            style="TCheckbutton",
            variable=radio_var,
            value="add_location",
            command=lambda: on_radio_select("add_location"),
        )
        radio1.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        radio2 = ttk.Radiobutton(
            self.radio_moduel,
            text="H5ad file add GroundTurth txt information",
            style="TCheckbutton",
            variable=radio_var,
            value="add_GroundTurth",
            command=lambda: on_radio_select("add_GroundTurth"),
        )
        radio2.grid(row=1, column=0, padx=5, pady=5, sticky="w")
        # radio3 = ttk.Radiobutton(self.radio_moduel, text="Multiple h5ad conversions to STAligner input format",
        #                          style="TCheckbutton", variable=radio_var,
        #                          value="multiple_h5ad_files", command=lambda: on_radio_select("multiple_h5ad_files"))
        # radio3.grid(row=2, column=0, padx=5, pady=5, sticky="w")
        radio3 = ttk.Radiobutton(
            self.radio_moduel,
            text="Only filter top genes and normalization data",
            style="TCheckbutton",
            variable=radio_var,
            value="filter_top_n",
            command=lambda: on_radio_select("filter_top_n"),
        )
        radio3.grid(row=2, column=0, padx=5, pady=5, sticky="w")

        def on_closing():
            self.path_load_flag = False
            self.Data_load_toplevel.destroy()

        self.Data_load_toplevel.protocol("WM_DELETE_WINDOW", on_closing)

    def ShutdowmSoftware(self):
        sys.exit()

    def web_Server_Thread(self, data_save_path, ports=8050):
        T = threading.Thread(
            target=webServer,
            args=(
                data_save_path,
                ports,
            ),
        )
        T.setDaemon(True)
        T.start()

    def web_Thread(self, filepath, inh5data_, conf_file_, data_name):
        T = threading.Thread(
            target=webcache_main,
            args=(
                filepath,
                inh5data_,
                conf_file_,
                data_name,
            ),
        )
        T.setDaemon(True)
        T.start()

    # def STAGATE_cluster_Thread(self):
    #     T = threading.Thread(target=self.STAGATE_cluster, args=(filepath, inh5data_, conf_file_, data_name,))
    #     # T.setDaemon(True)
    #     T.start()
    #     pass

    def STAGATE_show(self):
        if self.cluster_flag:
            self.result_flag = True
            plt.rcParams["font.sans-serif"] = "Arial"
            self.gene_color_type = "viridis"
            if not self.data_type:
                self.data_type = "10x"

            def draw_images():
                i = 1
                os.chdir(Raw_PATH)
                from matplotlib import pyplot as plt

                plt.rcParams["figure.figsize"] = (3, 3)
                path = (
                    test_file_path
                    + "/figures/"
                    + self.method_flag
                    + "_"
                    + self.data_type
                )
                if self.label_files_exit:
                    sc.pl.spatial(
                        adata,
                        img_key="hires",
                        color="GroundTruth",
                        title="Ground Truth",
                        show=False,
                        save="STAGATE_" + str(i) + ".png",
                        spot_size=self.spot_size,
                    )
                    shutil.move(
                        test_file_path + "/figures/showSTAGATE_" + str(i) + ".png",
                        path + "/STAGATE_GroundTruth.png",
                    )

                    i = i + 1
                    obs_df = adata.obs.dropna()
                    ARI = adjusted_rand_score(obs_df["mclust"], obs_df["GroundTruth"])

                    plt.rcParams["figure.figsize"] = (3, 3)
                    sc.pl.umap(
                        adata,
                        color="mclust",
                        title=["STAGATE-mclust(ARI=%.2f)" % ARI],
                        show=False,
                        s=6,
                        save="STAGATE_" + str(i) + ".png",
                    )
                    shutil.move(
                        test_file_path + "/figures/umapSTAGATE_" + str(i) + ".png",
                        path + "/STAGATE_mclust_umap.png",
                    )

                    i = i + 1
                    sc.pl.spatial(
                        adata,
                        color="mclust",
                        title=["STAGATE-mclust(ARI=%.2f)" % ARI],
                        show=False,
                        spot_size=self.spot_size,
                        save="STAGATE_" + str(i) + ".png",
                    )
                    shutil.move(
                        test_file_path + "/figures/showSTAGATE_" + str(i) + ".png",
                        path + "/STAGATE_mclust.png",
                    )

                    i = i + 1
                    sc.pl.umap(
                        adata,
                        color="louvain",
                        title=["STAGATE-louvain"],
                        show=False,
                        s=6,
                        save="STAGATE_" + str(i) + ".png",
                    )
                    shutil.move(
                        test_file_path + "/figures/umapSTAGATE_" + str(i) + ".png",
                        path + "/STAGATE_louvain_umap.png",
                    )

                    i = i + 1
                    sc.pl.spatial(
                        adata,
                        color="louvain",
                        title=["STAGATE-louvain"],
                        show=False,
                        spot_size=self.spot_size,
                        save="STAGATE_" + str(i) + ".png",
                    )
                    shutil.move(
                        test_file_path + "/figures/showSTAGATE_" + str(i) + ".png",
                        path + "/STAGATE_louvain.png",
                    )

                    i = i + 1
                    sc.pl.umap(
                        adata,
                        color="STAGATE_kmeans",
                        title=["STAGATE-kmeans"],
                        show=False,
                        s=6,
                        save="STAGATE_" + str(i) + ".png",
                    )
                    shutil.move(
                        test_file_path + "/figures/umapSTAGATE_" + str(i) + ".png",
                        path + "/STAGATE_kmeans_umap.png",
                    )

                    i = i + 1
                    sc.pl.spatial(
                        adata,
                        color="STAGATE_kmeans",
                        title=["STAGATE-kmeans"],
                        show=False,
                        spot_size=self.spot_size,
                        save="STAGATE_" + str(i) + ".png",
                    )
                    shutil.move(
                        test_file_path + "/figures/showSTAGATE_" + str(i) + ".png",
                        path + "/STAGATE_kmeans.png",
                    )

                    adata.obs.GroundTruth = adata.obs.GroundTruth.astype(str)
                    i = i + 1
                    sc.tl.paga(adata, groups="GroundTruth")
                    plt.rcParams["figure.figsize"] = (4, 3)
                    sc.pl.paga(
                        adata,
                        color="GroundTruth",
                        title="PAGA",
                        show=False,
                        save="STAGATE_" + str(i) + ".png",
                    )
                    shutil.move(
                        test_file_path + "/figures/pagaSTAGATE_" + str(i) + ".png",
                        path + "/STAGATE_PAGA.png",
                    )
                else:
                    sc.pp.calculate_qc_metrics(adata, inplace=True)
                    sc.pl.spatial(
                        adata,
                        img_key="hires",
                        color="log1p_total_counts",
                        title="log1p_total_counts",
                        show=False,
                        spot_size=self.spot_size,
                        color_map=self.gene_color_type,
                        save="STAGATE_" + str(i) + ".png",
                    )
                    shutil.move(
                        test_file_path + "/figures/showSTAGATE_" + str(i) + ".png",
                        path + "/STAGATE_log1p_total_counts.png",
                    )

                    i = i + 1
                    sc.pl.umap(
                        adata,
                        color="mclust",
                        title=["STAGATE-Mclust"],
                        show=False,
                        s=6,
                        save="STAGATE_" + str(i) + ".png",
                    )
                    shutil.move(
                        test_file_path + "/figures/umapSTAGATE_" + str(i) + ".png",
                        path + "/STAGATE_mclust_umap.png",
                    )

                    i = i + 1
                    sc.pl.spatial(
                        adata,
                        color="mclust",
                        title=["STAGATE-Mclust"],
                        show=False,
                        spot_size=self.spot_size,
                        save="STAGATE_" + str(i) + ".png",
                    )
                    shutil.move(
                        test_file_path + "/figures/showSTAGATE_" + str(i) + ".png",
                        path + "/STAGATE_mclust.png",
                    )
                    i = i + 1
                    sc.pl.umap(
                        adata,
                        color="louvain",
                        title=["STAGATE-louvain"],
                        show=False,
                        s=6,
                        save="STAGATE_" + str(i) + ".png",
                    )
                    shutil.move(
                        test_file_path + "/figures/umapSTAGATE_" + str(i) + ".png",
                        path + "/STAGATE_louvain_umap.png",
                    )

                    i = i + 1
                    sc.pl.spatial(
                        adata,
                        color="louvain",
                        title=["STAGATE-louvain"],
                        show=False,
                        spot_size=self.spot_size,
                        save="STAGATE_" + str(i) + ".png",
                    )
                    shutil.move(
                        test_file_path + "/figures/showSTAGATE_" + str(i) + ".png",
                        path + "/STAGATE_louvain.png",
                    )

                    i = i + 1
                    sc.pl.umap(
                        adata,
                        color="STAGATE_kmeans",
                        title=["STAGATE-kmeans"],
                        show=False,
                        s=6,
                        save="STAGATE_" + str(i) + ".png",
                    )
                    shutil.move(
                        test_file_path + "/figures/umapSTAGATE_" + str(i) + ".png",
                        path + "/STAGATE_kmeans_umap.png",
                    )

                    i = i + 1
                    sc.pl.spatial(
                        adata,
                        color="STAGATE_kmeans",
                        title=["STAGATE-kmeans"],
                        show=False,
                        spot_size=self.spot_size,
                        save="STAGATE_" + str(i) + ".png",
                    )
                    shutil.move(
                        test_file_path + "/figures/showSTAGATE_" + str(i) + ".png",
                        path + "/STAGATE_kmeans.png",
                    )
                adata.write_h5ad(
                    self.file_path.rsplit("/", 1)[-2]
                    + "/"
                    + self.method_flag
                    + "_result.h5ad"
                )
                print(
                    f"Result images are saved in : {test_file_path}/figures",
                )
                print(f"Result data are saved in : ", self.file_path.rsplit("/", 1)[-2])

            def scoller():
                self.figure_ybar = ttk.Scrollbar(
                    self.figure_Frame, orient=VERTICAL, cursor="draft_small"
                )
                self.figure_ybar.pack(side=RIGHT, fill=Y)
                self.figure_ybar.config(command=self.canvas.yview)

                self.figure_xbar = ttk.Scrollbar(
                    self.figure_Frame, orient=HORIZONTAL, cursor="draft_small"
                )
                self.figure_xbar.pack(side=BOTTOM, fill=X)
                self.figure_xbar.config(command=self.canvas.xview)

                self.canvas.configure(yscrollcommand=self.figure_xbar.set)
                self.canvas.config(yscrollincrement=1)

                self.canvas.configure(xscrollcommand=self.figure_xbar.set)
                self.canvas.config(xscrollincrement=1)
                self.canvas.pack(side=LEFT, expand=YES, fill=BOTH, padx=0)  # anchor=NW
                self.canvas.create_window((0, 0), window=self.show_frame, anchor=N)
                self.show_frame.bind("<Configure>", onFrameconfigure)

            def onFrameconfigure(event):
                self.canvas.configure(
                    scrollregion=self.canvas.bbox("all"), width=1000, height=650
                )

            def choose_color():
                colorvalue = colorchooser.askcolor()
                color_list = []
                color = colorvalue[1]
                print(color)
                if (
                    "louvain_colors" in adata.uns
                    and "STAGATE_kmeans_colors" in adata.uns
                ):
                    color_lens = [
                        adata.uns["louvain_colors"],
                        adata.uns["mclust_colors"],
                        adata.uns["STAGATE_kmeans_colors"],
                    ]
                    max_len = max(color_lens, key=len)
                else:
                    max_len = adata.uns["mclust_colors"]
                if len(max_len) > self.Mcluster_num:
                    self.Mcluster_num = len(max_len)
                if color[1:3] == "ff":
                    color_list.append(color)
                    list = random.sample(color_panel.FF_color, self.Mcluster_num)
                    color_list = color_list + list
                elif color[1:3] == "cc":
                    color_list.append(color)
                    list = random.sample(color_panel.CC_color, self.Mcluster_num)
                    color_list = color_list + list
                elif color[1:3] == "99":
                    color_list.append(color)
                    list = random.sample(color_panel.NN_color, self.Mcluster_num)
                    color_list = color_list + list
                elif color[1:3] == "66":
                    color_list.append(color)
                    list = random.sample(color_panel.SS_color, self.Mcluster_num)
                    color_list = color_list + list
                elif color[1:3] == "33":
                    color_list.append(color)
                    list = random.sample(color_panel.TT_color, self.Mcluster_num)
                    color_list = color_list + list
                elif color[1:3] == "00":
                    color_list.append(color)
                    list = random.sample(color_panel.ZZ_color, self.Mcluster_num)
                    color_list = color_list + list
                else:
                    color_list.append(color)
                    list = random.sample(color_panel.random_color, self.Mcluster_num)
                    color_list = color_list + list

                print(color_list)
                if self.label_files_exit:
                    adata.uns["GroundTruth_colors"] = color_list[
                        : len(adata.uns["GroundTruth_colors"])
                    ]
                adata.uns["mclust_colors"] = color_list[
                    : len(adata.uns["mclust_colors"])
                ]
                if (
                    "louvain_colors" in adata.uns
                    and "STAGATE_kmeans_colors" in adata.uns
                ):
                    adata.uns["louvain_colors"] = color_list[
                        : len(adata.uns["louvain_colors"])
                    ]
                    adata.uns["STAGATE_kmeans_colors"] = color_list[
                        : len(adata.uns["STAGATE_kmeans_colors"])
                    ]
                self.color_reset = color_list
                self.gene_color_type = color_panel.five_gene_color[random.randint(0, 5)]
                os.chdir(Raw_PATH)
                draw_images()

                fig_path = (
                    test_file_path
                    + "/figures/"
                    + self.method_flag
                    + "_"
                    + self.data_type
                )
                figures = os.listdir(fig_path)
                global image_list
                image_list = []
                for i in range(len(figures)):
                    image_list.append(ttk.PhotoImage(file=fig_path + "/" + figures[i]))
                    self.result_images = ttk.Label(self.show_frame, image=image_list[i])
                    self.result_images.grid(row=i // 3, column=i % 3, sticky=W, pady=0)

            fig_path = (
                test_file_path + "/figures/" + self.method_flag + "_" + self.data_type
            )
            if not os.path.isdir(fig_path):
                os.mkdir(fig_path)

            adata = self.result_queue.get()
            self.result_queue.put(adata)
            # adata = mclust_R(adata, used_obsm='STAGATE', num_cluster=self.Mcluster_num)
            # sc.tl.louvain(adata, resolution=self.louvain_res)
            # kmeans = KMeans(n_clusters=self.Mcluster_num)
            # kmeans.fit(adata.X)
            # adata.obs['STAGATE_kmeans'] = kmeans.labels_
            # adata.obs['STAGATE_kmeans'] = adata.obs['STAGATE_kmeans'].astype('category')
            if self.label_files_exit:
                adata = adata[adata.obs["GroundTruth"] == adata.obs["GroundTruth"],]
            else:
                adata.obsm["spatial"] = adata.obsm["spatial"] * (-1)
            draw_images()
            self.figure_Frame = ttk.Frame(self.right_panel)
            self.figure_Frame.pack(side=TOP, fill=X, expand=YES)

            self.canvas = ttk.Canvas(self.figure_Frame)
            self.show_frame = ttk.Frame(self.canvas)
            scoller()

            self.set_frame = ttk.Frame(
                self.figure_Frame, borderwidth=2, relief="sunken"
            )
            self.set_frame.pack(side="right", expand=YES, anchor=N)
            self.set_frame_one = ttk.Frame(self.set_frame)
            self.set_frame_one.pack(side=TOP, expand=YES)
            self.set_frame_two = ttk.Frame(self.set_frame)
            self.set_frame_two.pack(side=TOP, expand=YES)

            self.Reset = ttk.Button(
                self.set_frame_one,
                text="Reset all colors",
                command=choose_color,
                width=12,
            )
            self.Reset.grid(row=0, column=0, sticky=W, pady=0, padx=0)

            self.domain_color_label = ttk.Label(
                self.set_frame_one, text="Reset domain color: ", width=20
            )
            self.domain_color_label.grid(row=1, column=0, sticky=W, pady=0)

            self.Reset_domain_color = ttk.Entry(self.set_frame_one, width=10)
            self.Reset_domain_color.grid(row=1, column=1, sticky=W, pady=0)
            Tooltip(
                self.Reset_domain_color,
                "Input value must be int type and in [1:cluster_name]: 2",
            )
            self.color_update = None

            def Reset_single_domain_color():
                from tkinter import colorchooser, filedialog

                colorvalue = colorchooser.askcolor()
                color = colorvalue[1]
                print(color)
                cluster = self.Reset_domain_color.get()
                print(cluster)
                adata.uns["mclust_colors"] = self.color_reset[
                    : len(adata.uns["mclust_colors"])
                ]
                adata.uns["mclust_colors"][int(cluster) - 1] = color
                if "louvain_colors" in adata.uns:
                    adata.uns["louvain_colors"] = self.color_reset[
                        : len(adata.uns["louvain_colors"])
                    ]
                    adata.uns["louvain_colors"][int(cluster) - 1] = color
                self.color_reset[int(cluster) - 1] = color
                self.color_update = self.color_reset
                print(f"adata.uns['mclust_colors'] = {adata.uns['mclust_colors']}")
                import matplotlib.pyplot as plt

                plt.rcParams["figure.figsize"] = (3, 3)
                i = 1
                path = (
                    test_file_path
                    + "/figures/"
                    + self.method_flag
                    + "_"
                    + self.data_type
                )

                if self.label_files_exit:
                    i = i + 1
                    obs_df = adata.obs.dropna()
                    ARI = adjusted_rand_score(obs_df["mclust"], obs_df["GroundTruth"])

                    plt.rcParams["figure.figsize"] = (3, 3)
                    sc.pl.umap(
                        adata,
                        color="mclust",
                        title=["STAGATE (ARI=%.2f)" % ARI],
                        show=False,
                        s=6,
                        save="STAGATE_" + str(i) + ".png",
                    )
                    shutil.move(
                        test_file_path + "/figures/umapSTAGATE_" + str(i) + ".png",
                        path + "/STAGATE_" + str(i) + ".png",
                    )
                    i = i + 1
                    sc.pl.spatial(
                        adata,
                        color="mclust",
                        title=["STAGATE (ARI=%.2f)" % ARI],
                        show=False,
                        spot_size=150,
                        save="STAGATE_" + str(i) + ".png",
                    )
                    shutil.move(
                        test_file_path + "/figures/showSTAGATE_" + str(i) + ".png",
                        path + "/STAGATE_" + str(i) + ".png",
                    )
                else:
                    i = i + 1
                    plt.rcParams["figure.figsize"] = (3, 3)
                    sc.pl.umap(
                        adata,
                        color="mclust",
                        title=["STAGATE"],
                        show=False,
                        s=6,
                        save="STAGATE_" + str(i) + ".png",
                    )
                    shutil.move(
                        test_file_path + "/figures/umapSTAGATE_" + str(i) + ".png",
                        path + "/STAGATE_" + str(i) + ".png",
                    )
                    i = i + 1
                    sc.pl.spatial(
                        adata,
                        color="mclust",
                        title=["STAGATE"],
                        show=False,
                        spot_size=150,
                        save="STAGATE_" + str(i) + ".png",
                    )
                    shutil.move(
                        test_file_path + "/figures/showSTAGATE_" + str(i) + ".png",
                        path + "/STAGATE_" + str(i) + ".png",
                    )
                    i = i + 1
                    plt.rcParams["figure.figsize"] = (3, 3)
                    sc.pl.umap(
                        adata,
                        color="louvain",
                        title=["STAGATE-louvain"],
                        show=False,
                        s=6,
                        save="STAGATE_" + str(i) + ".png",
                    )
                    shutil.move(
                        test_file_path + "/figures/umapSTAGATE_" + str(i) + ".png",
                        path + "/STAGATE_" + str(i) + ".png",
                    )

                    i = i + 1
                    sc.pl.spatial(
                        adata,
                        color="louvain",
                        title=["STAGATE-louvain"],
                        show=False,
                        spot_size=150,
                        save="STAGATE_" + str(i) + ".png",
                    )
                    shutil.move(
                        test_file_path + "/figures/showSTAGATE_" + str(i) + ".png",
                        path + "/STAGATE_" + str(i) + ".png",
                    )
                figures = os.listdir(fig_path)
                global image_list
                image_list = []
                for i in range(len(figures)):
                    img = Image.open(fig_path + "/" + figures[i])
                    print(img.size)
                    image_list.append(ttk.PhotoImage(file=fig_path + "/" + figures[i]))
                    self.result_images = ttk.Label(self.show_frame, image=image_list[i])
                    self.result_images.grid(
                        row=i // 3, column=i % 3, sticky=W, pady=0, padx=0
                    )
                    s = i

            self.Reset_domain_btn = ttk.Button(
                self.set_frame_one,
                width=10,
                text="Confirm",
                command=Reset_single_domain_color,
            )
            self.Reset_domain_btn.grid(row=1, column=2, sticky=W, pady=10)
            self.update_image_label = ttk.Label(
                self.set_frame_one, text="Reset image dpi: ", width=20
            )
            self.update_image_label.grid(row=2, column=0, sticky=W, pady=10)
            self.update_image_scale = ttk.Entry(self.set_frame_one, width=10)
            self.update_image_scale.grid(row=2, column=1, sticky=W, pady=10)
            Tooltip(
                self.update_image_scale, "Input value must be int type and >= 300: 300"
            )

            def ipdate_hd():
                dpi = self.update_image_scale.get()
                print(dpi)
                import matplotlib.pyplot as plt

                plt.rcParams["figure.figsize"] = (3, 3)
                sc.set_figure_params(dpi=dpi)
                draw_images()
                figures = os.listdir(fig_path)
                print(len(figures))

            self.Reset_domain_btn = ttk.Button(
                self.set_frame_one, width=10, text="Save", command=ipdate_hd
            )
            self.Reset_domain_btn.grid(row=2, column=2, sticky=W, pady=0)

            self.gene_visualization_label = ttk.Label(
                self.set_frame_one, text="Input gene name: ", width=20
            )
            self.gene_visualization_label.grid(row=3, column=0, sticky=W, pady=0)

            self.gene_visualization_entry = ttk.Entry(self.set_frame_one, width=10)
            self.gene_visualization_entry.grid(row=3, column=1, sticky=W, pady=0)
            Tooltip(self.gene_visualization_entry, "Gene name: NEFH/ATP2B4")

            def gene_visualization():
                gene = self.gene_visualization_entry.get()
                if gene not in adata.var_names:
                    print(f"{gene} is mot in adata!!Please input right gene name!")
                sc.pl.spatial(
                    adata,
                    img_key="hires",
                    color=gene,
                    title="$" + gene + "$",
                    spot_size=150,
                    show=False,
                    save=gene + ".png",
                )
                global img0
                photo = Image.open(test_file_path + "/figures/show" + gene + ".png")
                img0 = ImageTk.PhotoImage(photo)
                img1 = ttk.Label(self.set_frame_two, image=img0)
                img1.grid(row=0, column=0, sticky=W, pady=0)
                if os.path.exists(test_file_path + "/figures/show" + gene + ".png"):
                    os.remove(test_file_path + "/figures/show" + gene + ".png")
                    print("yes")
                else:
                    print("error！")

            self.gene_visualization_btn = ttk.Button(
                self.set_frame_one, width=10, command=gene_visualization, text="Show"
            )
            self.gene_visualization_btn.grid(row=3, column=2, sticky=W, pady=0)
            figures = os.listdir(fig_path)
            global image_list
            image_list = []
            for i in range(len(figures)):
                image_list.append(ttk.PhotoImage(file=fig_path + "/" + figures[i]))
                self.result_images = ttk.Label(self.show_frame, image=image_list[i])
                self.result_images.grid(row=i // 3, column=i % 3, sticky=W, pady=0)
                s = i

            self.Entry = ttk.Entry(self.set_frame_one, width=10)
            self.Entry.grid(row=0, column=1, sticky=W, pady=0)
            Tooltip(self.Entry, "Local port number: 8050")

            def VIEW_3D():
                import webbrowser

                self.web_Server_Thread("/DLPFC/webcache", int(self.Entry.get()))
                http = "http://127.0.0.1:" + self.Entry.get() + "/"
                # 'http://127.0.0.1:8050/'
                webbrowser.open(http)

            self.Reset = ttk.Button(
                self.set_frame_one, text="3D VIEW", command=VIEW_3D, width=10
            )
            self.Reset.grid(row=0, column=2, sticky=W, pady=0)
        else:
            Messagebox.show_warning(
                title="Attention", message="Waiting for model training!!!"
            )

    def get_directory(self):
        """Open dialogue to get directory and update variable"""
        self.update_idletasks()
        self.file_path = filedialog.askopenfilename()
        print(self.file_path)
        if self.file_path:
            self.setvar("folder-path", self.file_path)

        if self.path_load_flag:
            self.path_load_flag = False
            self.paned_window.destroy()

        if self.file_path:
            self.path_load_flag = True
            self.Dataload_thread()

    def Visium_data_process(self):
        if self.multi_files is not None and self.method_flag == "STAligner":
            Batch_list = []
            adj_list = []
            key_add = []
            for i in self.multi_files:
                file_name = i.rsplit("\\", 1)[-1]
                key_add.append(file_name.rsplit(".", 1)[-2])
                adata_ = sc.read_h5ad(i)
                adata_.X = csr_matrix(adata_.X)
                adata_.var_names_make_unique(join="++")
                sc.pp.filter_genes(adata_, min_cells=50)
                adata_.obs_names = [
                    x + "_" + file_name.rsplit(".", 1)[-2] for x in adata_.obs_names
                ]
                Cal_Spatial_Net_new(adata_, rad_cutoff=self.rad_cutoff_value)
                sc.pp.highly_variable_genes(
                    adata_, flavor="seurat_v3", n_top_genes=5000
                )
                if "highly_variable" in adata_.var.columns:
                    adata_ = adata_[:, adata_.var["highly_variable"]]
                sc.pp.normalize_total(adata_, target_sum=1e4)
                sc.pp.log1p(adata_)
                adj_list.append(adata_.uns["adj"])
                Batch_list.append(adata_)
            adata = ad.concat(Batch_list, label="slice_name", keys=key_add)
            adata.obs["batch_name"] = adata.obs["slice_name"].astype("category")
            adj_concat = np.asarray(adj_list[0].todense())
            for batch_id in range(1, len(self.multi_files)):
                adj_concat = scipy.linalg.block_diag(
                    adj_concat, np.asarray(adj_list[batch_id].todense())
                )
            adata.uns["edgeList"] = np.nonzero(adj_concat)
        else:
            adata = sc.read(self.file_path)
            adata.var_names_make_unique()
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)

        self.paned_window = tk.PanedWindow(
            self.right_panel, orient=tk.HORIZONTAL, sashrelief=tk.SUNKEN, height=100
        )
        self.paned_window.pack(fill=X, before=self.info_Frame)
        text_widget = ttk.Text(self.paned_window)
        self.paned_window.add(text_widget)
        max_lines = 16
        redirector = PrintRedirector(text_widget, max_lines)
        sys.stdout = redirector
        # list = []
        if self.data_type is not None:
            print(f"Sequencing data platform is : {self.data_type}")
        else:
            print(f"Sequencing data platform is : NO choose(default 10x)")
        # print(f"Data type is : {type(adata)}")
        print(f"Data size is : {adata.shape[0]}*{adata.shape[1]}")
        if "GroundTruth" in adata.obs.columns:
            print(f"Whether Ground Truth is in Data : YES")
        else:
            print(f"Whether Ground Truth is in Data : NO")
        flag = False
        for i in adata.obsm:
            if i in methods:
                flag = True
                print(f"Whether the data has been trained : YES, used {i}")
                break
        if not flag:
            print(f"Whether the data has been trained : NO")
        print(f"First 5 gene of data are : {list(adata.var_names[:5])}")
        print(f"First 5 cell/spot of data are : {list(adata.obs_names[:5])}")
        print(f"Data type is : {type(adata)}")
        # with open("Data_info.txt", "w") as file:
        #     # 遍历字符串列表，将每个字符串写入文件
        #     for line in list:
        #         file.write(line + "\n")
        # print(adata)
        sys.stdout = sys.__stdout__
        self.result_queue.put(adata)
        self.data_load_flag = True
        del adata

    def Run_STAGATE(self):
        for i in range(self.result_queue.qsize() - 1):
            adata_remove = self.result_queue.get()
            del adata_remove
        if not self.result_queue.empty():
            self.model_train_flag = False
            self.cluster_flag = False
            adata = self.result_queue.get()
            if "GroundTruth" in adata.obs.columns:
                # if 'GroundTruth' in adata.obs.columns or 'GroundTruth_colors' in adata.uns.columns:
                self.label_files_exit = True
            if "STAGATE" not in adata.obsm:
                data_save_path = running_path + "/result/"
                Cal_Spatial_Net(adata, float(self.rad_cutoff_value))
                Stats_Spatial_Net(adata)
                if adata.shape[1] < int(self.alpha_value):
                    self.alpha_value = adata.shape[1]
                stagate = STAGATE(
                    model_dir=data_save_path,
                    in_features=self.alpha_value,
                    hidden_dims=[512, 30],
                )
                print("self.alpha_value", self.alpha_value)
                print("self.cluster_value", self.cluster_value)
                print("self.genes", self.genes)
                print("self.rad_cutof", float(self.rad_cutoff.get()))
                print("self.genes", float(self.gene_name.get()))
                print("self.cluster_value", int(self.cluster.get()))
                adata = stagate.train(
                    adata, n_epochs=int(self.cluster_value), lr=float(self.genes)
                )
                sc.pp.neighbors(adata, use_rep="STAGATE")
                sc.tl.umap(adata)
                # adata = mclust_R(adata, used_obsm='STAGATE', num_cluster=self.cluster_value)
                # sc.tl.louvain(adata, resolution=self.alpha_value)

            data_save_path = test_file_path + "/DLPFC"
            if not os.path.exists(data_save_path):
                os.makedirs(data_save_path)
                adata.write_h5ad(os.path.join(data_save_path, "DLPFC.h5ad"))
                json_file = {
                    "Coordinate": "spatial",
                    "Annotatinos": ["mclust"],
                    "Meshes": {},
                    "mesh_coord": "fixed.json",
                    "Genes": ["all"],
                }
                fixed_file = {
                    "xmin": 0,
                    "ymin": 0,
                    "margin": 0,
                    "zmin": 0,
                    "binsize": 1,
                }
                json.dump(
                    json_file,
                    open(os.path.join(data_save_path, "DLPFC.json"), "w"),
                    indent=4,
                )
                json.dump(
                    fixed_file, open(os.path.join(data_save_path, "fixed.json"), "w")
                )
                self.web_Thread(
                    data_save_path, "DLPFC.h5ad", "DLPFC.json", "Single-DLPFC"
                )

            data_path = os.path.join(running_path, "result", "STAGATE_output.h5ad")
            adata.write_h5ad(data_path)
            print(f"File is saved in :{data_path}")
            # self.result_queue.put(adata)
            self.pb.stop()
            self.pb["value"] = 100
            self.setvar("prog-message", "STAGATE run over!")
            self.setvar("End-time", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            EndTime = datetime.now().replace(microsecond=0)
            self.setvar("total-time-cost", EndTime - self.StartTime)
            self.model_train_flag = True
            self.cluster_flag = True

    def STAGATE_Thread(self):
        T = threading.Thread(target=self.Run_STAGATE)
        T.setDaemon(True)
        T.start()

    def STAligner_show(self):
        if self.model_train_flag:
            self.result_flag = True
            key_add = []
            Batch_list = []
            adata_list = {}
            if self.result_queue.qsize() > 1:
                adata_list = self.result_queue.get()
                self.result_queue.put(adata_list)

            adata_new = self.result_queue.get()
            self.result_queue.put(adata_list)
            self.result_queue.put(adata_new)
            # sc.tl.louvain(adata_new, random_state=666, key_added="louvain",
            #               resolution=float(self.louvain_res))
            # mclust_R(adata_new, num_cluster=int(self.Mcluster_num), used_obsm='STAligner')
            for section_id in self.multi_files:
                file_name = section_id.rsplit("\\", 1)[-1]
                k = file_name.rsplit(".", 1)[-2]
                key_add.append(k)
            for section_id in key_add:
                Batch_list.append(adata_new[adata_new.obs["batch_name"] == section_id])

            def draw_images():
                os.chdir(Raw_PATH)
                from matplotlib import pyplot as plt

                plt.rcParams["font.sans-serif"] = "Arial"
                path = test_file_path + "/figures/" + self.method_flag + "_output/"
                if self.label_files_exit:
                    i = 1
                    k = 1
                    for section_id in key_add:
                        plt.rcParams["figure.figsize"] = (3, 3)
                        sc.pl.spatial(
                            adata_list[section_id],
                            img_key="hires",
                            color=["GroundTruth"],
                            title=section_id + "-Ground Truth",
                            show=False,
                            save="STAligner_" + str(k) + "_" + str(i) + ".png",
                        )
                        shutil.move(
                            test_file_path
                            + "/figures/show"
                            + "STAligner_"
                            + str(k)
                            + "_"
                            + str(i)
                            + ".png",
                            path
                            + "STAligner_"
                            + str(k)
                            + "_"
                            + str(i)
                            + "_"
                            + section_id
                            + "_GroundTruth.png",
                        )
                        i = i + 1

                    k = k + 1
                    i = 1
                    for j in range(len(self.multi_files)):
                        plt.rcParams["figure.figsize"] = (3, 3)
                        sc.pl.spatial(
                            Batch_list[j],
                            color=["mclust"],
                            title=f"Mclust-{key_add[j]}",
                            spot_size=self.spot_size,
                            show=False,
                            save="STAligner_" + str(k) + "_" + str(i) + ".png",
                        )
                        shutil.move(
                            test_file_path
                            + "/figures/show"
                            + "STAligner_"
                            + str(k)
                            + "_"
                            + str(i)
                            + ".png",
                            path + "STAligner_" + str(k) + "_" + str(i) + "_Mclust.png",
                        )
                        i = i + 1

                    k = k + 1
                    i = 1
                    for j in range(len(self.multi_files)):
                        plt.rcParams["figure.figsize"] = (3, 3)
                        sc.pl.spatial(
                            Batch_list[j],
                            color=["louvain"],
                            title=f"louvain-{key_add[j]}",
                            spot_size=self.spot_size,
                            show=False,
                            save="STAligner_" + str(k) + "_" + str(i) + ".png",
                        )
                        shutil.move(
                            test_file_path
                            + "/figures/show"
                            + "STAligner_"
                            + str(k)
                            + "_"
                            + str(i)
                            + ".png",
                            path
                            + "STAligner_"
                            + str(k)
                            + "_"
                            + str(i)
                            + "_louvain.png",
                        )

                        i = i + 1

                    k = k + 1
                    i = 1
                    plt.rcParams["figure.figsize"] = (3, 3)
                    sc.pl.umap(
                        adata_new[~adata_new.obs["GroundTruth"].isin(["unknown"])],
                        color=["batch_name"],
                        title="STAligner-Mclust",
                        show=False,
                        s=6,
                        save="STAligner_" + str(i) + ".png",
                    )
                    shutil.move(
                        test_file_path + "/figures/umapSTAligner_" + str(i) + ".png",
                        path + "STAligner_" + str(k) + "_" + str(i) + "_Batchsumap.png",
                    )

                    k = k + 1
                    plt.rcParams["figure.figsize"] = (3, 3)
                    i = 1
                    sc.pl.umap(
                        adata_new[~adata_new.obs["GroundTruth"].isin(["unknown"])],
                        color=["GroundTruth"],
                        title="Ground Truth",
                        show=False,
                        s=6,
                        save="STAligner_" + str(i) + ".png",
                    )
                    shutil.move(
                        test_file_path + "/figures/umapSTAligner_" + str(i) + ".png",
                        path
                        + "STAligner_"
                        + str(k)
                        + "_"
                        + str(i)
                        + "_GroundTruthumap.png",
                    )

                    k = k + 1
                    plt.rcParams["figure.figsize"] = (3, 3)
                    i = 1
                    sc.pl.umap(
                        adata_new,
                        color=["louvain"],
                        title="STAligner-louvain",
                        show=False,
                        s=6,
                        save="STAligner_" + str(i) + ".png",
                    )
                    shutil.move(
                        test_file_path + "/figures/umapSTAligner_" + str(i) + ".png",
                        path + "STAligner_" + str(k) + "_" + str(i) + "_louvain.png",
                    )
                else:
                    i = 1
                    k = 1
                    for j in range(len(self.multi_files)):
                        plt.rcParams["figure.figsize"] = (3, 3)
                        sc.pl.spatial(
                            Batch_list[j],
                            color=["mclust"],
                            title="STAligner-mclust",
                            spot_size=self.spot_size,
                            show=False,
                            save="STAligner_" + str(k) + "_" + str(i) + ".png",
                        )
                        shutil.move(
                            test_file_path
                            + "/figures/show"
                            + "STAligner_"
                            + str(k)
                            + "_"
                            + str(i)
                            + ".png",
                            path + "STAligner_" + str(k) + "_" + str(i) + "_mclust.png",
                        )

                        i = i + 1

                    k = k + 1
                    i = 1
                    for j in range(len(self.multi_files)):
                        plt.rcParams["figure.figsize"] = (3, 3)
                        sc.pl.spatial(
                            Batch_list[j],
                            color=["louvain"],
                            title=f"louvain-{key_add[j]}",
                            spot_size=self.spot_size,
                            show=False,
                            save="STAligner_" + str(k) + "_" + str(i) + ".png",
                        )
                        shutil.move(
                            test_file_path
                            + "/figures/show"
                            + "STAligner_"
                            + str(k)
                            + "_"
                            + str(i)
                            + ".png",
                            path
                            + "STAligner_"
                            + str(k)
                            + "_"
                            + str(i)
                            + "_louvain.png",
                        )

                        i = i + 1

                    k = k + 1
                    i = 1
                    plt.rcParams["figure.figsize"] = (3, 3)
                    sc.pl.umap(
                        adata_new[~adata_new.obs["GroundTruth"].isin(["unknown"])],
                        color=["batch_name"],
                        title="STAligner-Mclust",
                        show=False,
                        s=6,
                        save="STAligner_" + str(i) + ".png",
                    )
                    shutil.move(
                        test_file_path + "/figures/umapSTAligner_" + str(i) + ".png",
                        path + "STAligner_" + str(k) + "_" + str(i) + "_Batchsumap.png",
                    )

                    k = k + 1
                    plt.rcParams["figure.figsize"] = (3, 3)
                    i = 1
                    sc.pl.umap(
                        adata_new,
                        color=["louvain"],
                        title="STAligner-louvain",
                        show=False,
                        s=6,
                        save="STAligner_" + str(i) + ".png",
                    )
                    shutil.move(
                        test_file_path + "/figures/umapSTAligner_" + str(i) + ".png",
                        path + "STAligner_" + str(k) + "_" + str(i) + "_louvain.png",
                    )
                print(f"Result images are saved in : {test_file_path}/figures")

            def scoller():
                self.figure_ybar = ttk.Scrollbar(
                    self.figure_Frame, orient=VERTICAL, cursor="draft_small"
                )
                self.figure_ybar.pack(side=RIGHT, fill=Y)
                self.figure_ybar.config(command=self.canvas.yview)

                self.figure_xbar = ttk.Scrollbar(
                    self.figure_Frame, orient=HORIZONTAL, cursor="draft_small"
                )
                self.figure_xbar.pack(side=BOTTOM, fill=X)
                self.figure_xbar.config(command=self.canvas.xview)

                self.canvas.configure(yscrollcommand=self.figure_xbar.set)
                self.canvas.config(yscrollincrement=1)

                self.canvas.configure(xscrollcommand=self.figure_xbar.set)
                self.canvas.config(xscrollincrement=1)
                self.canvas.pack(side=LEFT, expand=YES, fill=BOTH, padx=0)  # anchor=NW
                self.canvas.create_window((0, 0), window=self.show_frame, anchor=N)
                self.show_frame.bind("<Configure>", onFrameconfigure)

            def onFrameconfigure(event):
                self.canvas.configure(
                    scrollregion=self.canvas.bbox("all"), width=1000, height=650
                )

            def choose_color():
                colorvalue = colorchooser.askcolor()
                color_list = []
                color = colorvalue[1]
                print(color)
                if "louvain_colors" in Batch_list[0].uns:
                    color_lens = [
                        Batch_list[0].uns["louvain_colors"],
                        Batch_list[0].uns["mclust_colors"],
                    ]
                    max_len_list = max(color_lens, key=len)
                    max_len = len(max_len_list)

                if color[1:3] == "ff":
                    color_list.append(color)
                    list = random.sample(color_panel.FF_color, max_len)
                    color_list = color_list + list
                elif color[1:3] == "cc":
                    color_list.append(color)
                    list = random.sample(color_panel.CC_color, max_len)
                    color_list = color_list + list
                elif color[1:3] == "99":
                    color_list.append(color)
                    list = random.sample(color_panel.NN_color, max_len)
                    color_list = color_list + list
                elif color[1:3] == "66":
                    color_list.append(color)
                    list = random.sample(color_panel.SS_color, max_len)
                    color_list = color_list + list
                elif color[1:3] == "33":
                    color_list.append(color)
                    list = random.sample(color_panel.TT_color, max_len)
                    color_list = color_list + list
                elif color[1:3] == "00":
                    color_list.append(color)
                    list = random.sample(color_panel.ZZ_color, max_len)
                    color_list = color_list + list
                else:
                    color_list.append(color)
                    list = random.sample(color_panel.random_color, max_len)
                    color_list = color_list + list

                print(color_list)
                for j in range(len(self.multi_files)):
                    Batch_list[j].uns["mclust_colors"] = color_list[
                        : len(Batch_list[j].uns["mclust_colors"])
                    ]

                if "louvain_colors" in Batch_list[0].uns:
                    for j in range(len(self.multi_files)):
                        Batch_list[j].uns["louvain_colors"] = color_list[
                            : len(Batch_list[j].uns["louvain_colors"])
                        ]

                adata_new.uns["batch_name_colors"] = color_list[: len(self.multi_files)]
                if self.label_files_exit:
                    adata_new.uns["GroundTruth_colors"] = color_list
                    for section_id in key_add:
                        adata_list[section_id].uns["GroundTruth_colors"] = color_list
                self.color_reset = color_list
                self.gene_color_type = color_panel.five_gene_color[random.randint(0, 5)]
                os.chdir(Raw_PATH)
                draw_images()
                fig_path = test_file_path + "/figures/" + self.method_flag + "_output/"
                figures = os.listdir(fig_path)
                global image_list
                image_list = []
                for i in range(len(figures)):
                    image_list.append(ttk.PhotoImage(file=fig_path + "/" + figures[i]))
                    self.result_images = ttk.Label(self.show_frame, image=image_list[i])
                    self.result_images.grid(row=i // 4, column=i % 4, sticky=W, pady=0)

            fig_path = test_file_path + "/figures/" + self.method_flag + "_output/"
            if not os.path.isdir(fig_path):
                os.mkdir(fig_path)

            draw_images()
            figures = os.listdir(fig_path)
            self.figure_Frame = ttk.Frame(self.right_panel)
            self.figure_Frame.pack(side=TOP, fill=X, expand=YES)

            self.canvas = ttk.Canvas(self.figure_Frame)
            self.show_frame = ttk.Frame(self.canvas)
            scoller()

            self.set_frame = ttk.Frame(
                self.figure_Frame, borderwidth=2, relief="sunken"
            )
            self.set_frame.pack(side="right", expand=YES, anchor=N)
            self.set_frame_one = ttk.Frame(self.set_frame)
            self.set_frame_one.pack(side=TOP, expand=YES)
            self.set_frame_two = ttk.Frame(self.set_frame)
            self.set_frame_two.pack(side=TOP, expand=YES)

            self.Reset = ttk.Button(
                self.set_frame_one,
                text="Reset all colors",
                command=choose_color,
                width=12,
            )
            self.Reset.grid(row=0, column=0, sticky=W, pady=0, padx=0)

            self.domain_color_label = ttk.Label(
                self.set_frame_one, text="Reset domain color: ", width=20
            )
            self.domain_color_label.grid(row=1, column=0, sticky=W, pady=0)

            self.Reset_domain_color = ttk.Entry(self.set_frame_one, width=10)
            self.Reset_domain_color.grid(row=1, column=1, sticky=W, pady=0)
            Tooltip(
                self.Reset_domain_color,
                "Input value must be int type and in [1:cluster_name]: 2",
            )
            self.color_update = None

            def Reset_single_domain_color():
                from tkinter import colorchooser, filedialog

                colorvalue = colorchooser.askcolor()
                color = colorvalue[1]
                print(color)
                cluster = self.Reset_domain_color.get()
                for j in range(len(self.multi_files)):
                    Batch_list[j].uns["mclust_colors"] = self.color_reset[
                        : len(Batch_list[j].uns["mclust_colors"])
                    ]
                    Batch_list[j].uns["mclust_colors"][int(cluster) - 1] = color

                adata_new.uns["batch_name_colors"] = self.color_reset[
                    : len(self.multi_files)
                ]
                self.color_reset[int(cluster) - 1] = color
                # self.color_update = self.color_reset
                if "louvain_colors" in Batch_list[0].uns:
                    for j in range(len(self.multi_files)):
                        Batch_list[j].uns["louvain_colors"] = self.color_reset[
                            : len(Batch_list[j].uns["louvain_colors"])
                        ]

                draw_images()
                # import matplotlib.pyplot as plt
                # plt.rcParams["figure.figsize"] = (3, 3)
                # plt.rcParams['font.sans-serif'] = "Arial"
                # i = 1
                # k = 1
                # for section_id in self.multi_files:
                #     i = i + 1
                #
                # k = k + 1
                # i = 1
                # spot_size = 200
                # for j in range(len(self.multi_files)):
                #     plt.rcParams["figure.figsize"] = (3, 3)
                #     sc.pl.spatial(Batch_list[j], color=['mclust'], title='STAligner', spot_size=spot_size, show=False,
                #                   save='STAligner_' + str(k) + '_' + str(i) + '.png')
                #     shutil.move(test_file_path + '/figures/show' + 'STAligner_' + str(k) + '_' + str(i) + '.png',
                #                 path + 'STAligner_' + str(k) + '_' + str(i) + '.png')
                #
                #     i = i + 1
                #
                # k = k + 2
                # plt.rcParams["figure.figsize"] = (3, 3)
                # i = 1
                # plt.rcParams["figure.figsize"] = (3, 3)
                # sc.pl.umap(adata_new, color=['batch_name'], title='Batchs', show=False, s=6, save='STAligner_' + str(
                #     i) + '.png')
                # shutil.move(test_file_path + '/figures/umapSTAligner_' + str(i) + '.png',
                #             path + 'STAligner_' + str(k) + '_' + str(i) + '.png')
                #
                # k = k + 1
                # i = 1
                # spot_size = 200
                # for j in range(len(self.multi_files)):
                #     plt.rcParams["figure.figsize"] = (3, 3)
                #     sc.pl.spatial(Batch_list[j], color=['louvain'], title='STAligner-louvain', spot_size=spot_size,
                #                   show=False,
                #                   save='STAligner_' + str(k) + '_' + str(i) + '.png')
                #     shutil.move(test_file_path + '/figures/show' + 'STAligner_' + str(k) + '_' + str(i) + '.png',
                #                 path + 'STAligner_' + str(k) + '_' + str(i) + '.png')
                #     i = i + 1
                figures = os.listdir(fig_path)
                global image_list
                image_list = []
                for i in range(len(figures)):
                    img = Image.open(fig_path + "/" + figures[i])
                    print(img.size)
                    image_list.append(ttk.PhotoImage(file=fig_path + "/" + figures[i]))
                    self.result_images = ttk.Label(self.show_frame, image=image_list[i])
                    self.result_images.grid(
                        row=i // 4, column=i % 4, sticky=W, pady=0, padx=0
                    )
                    s = i

            self.Reset_domain_btn = ttk.Button(
                self.set_frame_one,
                width=10,
                text="Confirm",
                command=Reset_single_domain_color,
            )
            self.Reset_domain_btn.grid(row=1, column=2, sticky=W, pady=10)
            self.update_image_label = ttk.Label(
                self.set_frame_one, text="Reset image dpi: ", width=20
            )
            self.update_image_label.grid(row=2, column=0, sticky=W, pady=10)
            self.update_image_scale = ttk.Entry(self.set_frame_one, width=10)
            self.update_image_scale.grid(row=2, column=1, sticky=W, pady=10)
            Tooltip(
                self.update_image_scale, "Input value must be int type and >= 300: 300"
            )

            def ipdate_hd():
                dpi = self.update_image_scale.get()
                print(dpi)
                import matplotlib.pyplot as plt

                plt.rcParams["figure.figsize"] = (3, 3)
                sc.set_figure_params(dpi=dpi)
                draw_images()
                figures = os.listdir(fig_path)
                print(len(figures))

            self.Reset_domain_btn = ttk.Button(
                self.set_frame_one, width=10, text="Save", command=ipdate_hd
            )
            self.Reset_domain_btn.grid(row=2, column=2, sticky=W, pady=0)

            self.gene_visualization_label = ttk.Label(
                self.set_frame_one, text="Input gene name: ", width=20
            )
            self.gene_visualization_label.grid(row=3, column=0, sticky=W, pady=0)

            self.gene_visualization_entry = ttk.Entry(self.set_frame_one, width=10)
            self.gene_visualization_entry.grid(row=3, column=1, sticky=W, pady=0)
            Tooltip(self.gene_visualization_entry, "Gene name: NEFH/ATP2B4")

            def gene_visualization():
                try:
                    gene = self.gene_visualization_entry.get()
                    print(adata_list[self.multi_files[-1]].var.index)
                    if gene in adata_list[self.multi_files[-1]].var_names:

                        sc.pl.spatial(
                            adata_list[self.multi_files[-1]],
                            img_key="hires",
                            color=gene,
                            title="$" + gene + "$",
                            show=False,
                            save=gene + ".png",
                            sopt_size=150,
                        )
                    else:
                        print(f"{gene} is not in adata!")
                    global img0
                    photo = Image.open(test_file_path + "/figures/show" + gene + ".png")
                    img0 = ImageTk.PhotoImage(photo)
                    img1 = ttk.Label(self.set_frame_two, image=img0)
                    img1.grid(row=0, column=0, sticky=W, pady=0)
                    if os.path.exists(test_file_path + "/figures/show" + gene + ".png"):
                        os.remove(test_file_path + "/figures/show" + gene + ".png")
                        print("Figures exits")
                    else:
                        print("Figures no exits！")
                    pass
                except:
                    Messagebox.show_error(
                        "Python Error", "Make sure gene name in dataset"
                    )

            self.gene_visualization_btn = ttk.Button(
                self.set_frame_one, width=10, command=gene_visualization, text="Show"
            )
            self.gene_visualization_btn.grid(row=3, column=2, sticky=W, pady=0)

            global image_list
            image_list = []
            s = 0
            for i in range(len(figures)):
                image_list.append(ttk.PhotoImage(file=fig_path + "/" + figures[i]))
                self.result_images = ttk.Label(self.show_frame, image=image_list[i])
                self.result_images.grid(row=i // 4, column=i % 4, sticky=W, pady=0)
                s = i

            self.Entry = ttk.Entry(self.set_frame_one, width=10)
            self.Entry.grid(row=0, column=1, sticky=W, pady=0)
            Tooltip(self.Entry, "Local port number: 8049")

            def VIEW_3D():
                import webbrowser

                self.web_Server_Thread("/STAligner_3D/webcache", int(self.Entry.get()))
                http = "http://127.0.0.1:" + self.Entry.get() + "/"
                # 'http://127.0.0.1:8050/'
                webbrowser.open(http)

            self.Reset = ttk.Button(
                self.set_frame_one, text="3D VIEW", command=VIEW_3D, width=10
            )
            self.Reset.grid(row=0, column=2, sticky=W, pady=0)
        else:
            Messagebox.show_warning(
                title="Attention", message="Waiting for models training!!!"
            )

    def STAligner_Multiple_Sections_analysis(self):
        self.Dataload()
        self.model_train_flag = False
        self.cluster_flag = False
        for i in range(self.result_queue.qsize() - 1):
            adata_remove = self.result_queue.get()
            del adata_remove
        adata_new = self.result_queue.get()
        if (
            "GroundTruth" in adata_new.obs.columns
            or "GroundTruth_colors" in adata_new.uns.columns
        ):
            self.label_files_exit = True
            adata_list = {}
            key_add = []
            for section_id in self.multi_files:
                file_name = section_id.rsplit("\\", 1)[-1]
                k = file_name.rsplit(".", 1)[-2]
                key_add.append(k)
                temp_adata = sc.read(section_id)
                temp_adata.var_names_make_unique()
                temp_adata.obs_names = [x + "_" + k for x in temp_adata.obs_names]
                adata_list[k] = temp_adata.copy()
            # self.result_queue.put(adata_list)
            # adata_list.write_h5ad(os.path.join(running_path, "result", "STAligner_GroundTruth_output.h5ad"))
            # del temp_adata, adata_list
        if "STAligner" not in adata_new.obsm:
            staligner = STAligner(
                model_dir=running_path + "/result/",
                in_features=3000,
                hidden_dims=[512, 30],
                n_models=5,
                device=torch.device("cuda:0"),
            )
            adata_new = staligner.train(
                adata_new,
                iter_comb=[(0, 1)],
                n_epochs=int(self.cluster_value),
                lr=float(self.alpha_value),
                margin=float(self.adjust_value),
            )
            sc.pp.neighbors(adata_new, use_rep="STAligner", random_state=666)
            sc.tl.umap(adata_new, random_state=666)

        data_save_path = test_file_path + "/STAligner_3D"
        if not os.path.exists(data_save_path):
            os.makedirs(data_save_path)
            del adata_new.uns["edgeList"]
            adata_new.write_h5ad(
                os.path.join(data_save_path, "STAligner_Multi_DLPFC.h5ad")
            )
            json_file = {
                "Coordinate": "spatial",
                "Annotatinos": ["mclust"],
                "Meshes": {},
                "mesh_coord": "fixed.json",
                "Genes": ["all"],
            }
            fixed_file = {"xmin": 0, "ymin": 0, "margin": 0, "zmin": 0, "binsize": 1}
            json.dump(
                json_file,
                open(os.path.join(data_save_path, "Multi-DLPFC.json"), "w"),
                indent=4,
            )
            json.dump(fixed_file, open(os.path.join(data_save_path, "fixed.json"), "w"))
            self.web_Thread(
                data_save_path,
                "STAligner_Multi_DLPFC.h5ad",
                "Multi-DLPFC.json",
                "Multi-DLPFC",
            )

        # self.result_queue.put(adata_new)
        data_path = os.path.join(running_path, "result", "STAligner_output.h5ad")
        print(adata_new)
        del adata_new.uns["edgeList"]
        print(f"Training file is saved in {data_path} ")
        adata_new.write_h5ad(data_path)

        self.pb.stop()
        self.pb["value"] = 100
        self.setvar("prog-message", "STAligner run over!")
        self.setvar("End-time", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        EndTime = datetime.now().replace(microsecond=0)
        self.setvar("total-time-cost", EndTime - self.StartTime)
        self.model_train_flag = True
        self.cluster_flag = True

    def STAligner_Thread(self):
        T = threading.Thread(target=self.STAligner_Multiple_Sections_analysis)
        # T.setDaemon(True)
        T.start()

    def STAMarker_show(self):
        if self.cluster_flag:
            self.result_flag = True
            # print(f"self.result_queue长度：{self.result_queue.qsize()}!!!")
            self.gene_color_type = "viridis"
            ann_data = self.result_queue.get()
            data_path = (
                test_file_path
                + "/STAMarker_ouput/"
                + self.method_flag
                + "_"
                + self.data_type
                + "_output"
            )
            df = pd.read_csv(
                os.path.join(running_path, "result", "STAMarker_find_SVGs.txt"),
                sep="\t",
                index_col=False,
            )
            genes_list = df.values.tolist()
            # print(f"now self.result_queue长度：{self.result_queue.qsize()}!!!")

            def draw_images():
                i = 1
                temp_path = os.getcwd()
                os.chdir(test_file_path)
                from matplotlib import pyplot as plt

                plt.rcParams["font.sans-serif"] = "Arial"
                plt.rcParams["figure.figsize"] = (3, 3)
                path = (
                    test_file_path
                    + "/figures/"
                    + self.method_flag
                    + "_"
                    + self.data_type
                )
                if self.label_files_exit:
                    sc.pl.spatial(
                        ann_data,
                        img_key="hires",
                        color="GroundTruth",
                        title="Ground Truth",
                        s=6,
                        show=False,
                        save="STAMarker_" + str(i) + ".png",
                    )
                    shutil.move(
                        test_file_path + "/figures/showSTAMarker_" + str(i) + ".png",
                        path + "/STAMarker_" + str(i) + ".png",
                    )
                    i = i + 1

                plt.rcParams["figure.figsize"] = (3, 3)
                sc.pl.spatial(
                    ann_data,
                    img_key="hires",
                    color="Consensus_clustering",
                    title="STAMarker",
                    s=6,
                    show=False,
                    spot_size=self.spot_size,
                    save="STAMarker_" + str(i) + ".png",
                )
                shutil.move(
                    test_file_path + "/figures/showSTAMarker_" + str(i) + ".png",
                    path + "/STAMarker_" + str(i) + ".png",
                )

                genes = genes_list[: self.show_gene_num]
                if self.show_gene_name in genes_list:
                    genes = genes.append(self.show_gene_name)
                plt.rcParams["figure.figsize"] = (3, 3)
                j = 0
                i = i + 1
                for k in genes:
                    sc.pl.spatial(
                        ann_data,
                        img_key="hires",
                        color=k,
                        title=k,
                        s=6,
                        show=False,
                        spot_size=self.spot_size,
                        color_map=self.gene_color_type,
                        save="STAMarker_" + str(i) + "_" + str(j) + ".png",
                    )
                    shutil.move(
                        test_file_path
                        + "/figures/showSTAMarker_"
                        + str(i)
                        + "_"
                        + str(j)
                        + ".png",
                        path + "/STAMarker_" + str(i) + "_" + str(j) + ".png",
                    )
                    j = j + 1
                print(f"Result images are saved in : {test_file_path}/figures")
                os.chdir(temp_path)

            def scoller():
                self.figure_ybar = ttk.Scrollbar(
                    self.figure_Frame, orient=VERTICAL, cursor="draft_small"
                )
                self.figure_ybar.pack(side=RIGHT, fill=Y)
                self.figure_ybar.config(command=self.canvas.yview)

                self.figure_xbar = ttk.Scrollbar(
                    self.figure_Frame, orient=HORIZONTAL, cursor="draft_small"
                )
                self.figure_xbar.pack(side=BOTTOM, fill=X)
                self.figure_xbar.config(command=self.canvas.xview)

                self.canvas.configure(yscrollcommand=self.figure_xbar.set)
                self.canvas.config(yscrollincrement=1)

                self.canvas.configure(xscrollcommand=self.figure_xbar.set)
                self.canvas.config(xscrollincrement=1)
                self.canvas.pack(side=LEFT, expand=YES, fill=BOTH, padx=0)  # anchor=NW
                self.canvas.create_window((0, 0), window=self.show_frame, anchor=N)
                self.show_frame.bind("<Configure>", onFrameconfigure)

            def onFrameconfigure(event):
                self.canvas.configure(
                    scrollregion=self.canvas.bbox("all"), width=1000, height=650
                )

            def choose_color():
                colorvalue = colorchooser.askcolor()
                color_list = []
                color = colorvalue[1]
                # if len(ann_data.uns['Consensus_clustering_colors']) > self.cluster_value:
                #     self.cluster_value = len(ann_data.uns['Consensus_clustering_colors'])
                if color[1:3] == "ff":
                    color_list.append(color)
                    list = random.sample(color_panel.FF_color, self.cluster_value)
                    color_list = color_list + list
                elif color[1:3] == "cc":
                    color_list.append(color)
                    list = random.sample(color_panel.CC_color, self.cluster_value)
                    color_list = color_list + list
                elif color[1:3] == "99":
                    color_list.append(color)
                    list = random.sample(color_panel.NN_color, self.cluster_value)
                    color_list = color_list + list
                elif color[1:3] == "66":
                    color_list.append(color)
                    list = random.sample(color_panel.SS_color, self.cluster_value)
                    color_list = color_list + list
                elif color[1:3] == "33":
                    color_list.append(color)
                    list = random.sample(color_panel.TT_color, self.cluster_value)
                    color_list = color_list + list
                elif color[1:3] == "00":
                    color_list.append(color)
                    list = random.sample(color_panel.ZZ_color, self.cluster_value)
                    color_list = color_list + list
                else:
                    color_list.append(color)
                    list = random.sample(color_panel.random_color, self.cluster_value)
                    color_list = color_list + list

                print(color_list)
                if self.label_files_exit:
                    ann_data.uns["GroundTruth_colors"] = color_list[
                        : len(ann_data.uns["GroundTruth_colors"])
                    ]
                # ann_data.uns['Consensus_clustering_colors'] = color_list[
                #                                               :len(ann_data.uns['Consensus_clustering_colors'])]
                self.gene_color_type = color_panel.five_gene_color[random.randint(0, 5)]
                self.color_reset = color_list
                draw_images()

                fig_path = (
                    test_file_path
                    + "/figures/"
                    + self.method_flag
                    + "_"
                    + self.data_type
                )  # './figures/showSTAMarker_10XVisium'
                figures = os.listdir(fig_path)
                global image_list
                image_list = []
                for i in range(len(figures)):
                    image_list.append(ttk.PhotoImage(file=fig_path + "/" + figures[i]))
                    self.result_images = ttk.Label(self.show_frame, image=image_list[i])
                    self.result_images.grid(row=i // 3, column=i % 3, sticky=W, pady=0)

            if self.data_type is None:
                self.data_type = "10x"
            fig_path = (
                test_file_path + "/figures/" + self.method_flag + "_" + self.data_type
            )
            if not os.path.isdir(fig_path):
                os.mkdir(fig_path)

            images_path = glob.glob(data_path + "/*.png")
            num = 4
            ll = 0
            for i in images_path:
                plt.Figure(figsize=(3, 3))
                image = plt.imread(i)
                plt.imshow(image)
                plt.axis("off")
                plt.savefig(i)
                plt.close()
                shutil.move(
                    i,
                    os.path.join(
                        fig_path, "STAMarker_GSEA_" + str(num) + "_" + str(ll) + ".png"
                    ),
                )
                ll = ll + 1
                pass
            draw_images()
            self.figure_Frame = ttk.Frame(self.right_panel)
            self.figure_Frame.pack(side=TOP, fill=X, expand=YES)

            self.canvas = ttk.Canvas(self.figure_Frame)
            self.show_frame = ttk.Frame(self.canvas)
            scoller()

            self.set_frame = ttk.Frame(
                self.figure_Frame, borderwidth=2, relief="sunken"
            )
            self.set_frame.pack(side="right", expand=YES, anchor=N)
            self.set_frame_one = ttk.Frame(self.set_frame)
            self.set_frame_one.pack(side=TOP, expand=YES)
            self.set_frame_two = ttk.Frame(self.set_frame)
            self.set_frame_two.pack(side=TOP, expand=YES)

            self.Reset = ttk.Button(
                self.set_frame_one,
                text="Reset all colors",
                command=choose_color,
                width=12,
            )
            self.Reset.grid(row=0, column=0, sticky=W, pady=0, padx=0)

            self.domain_color_label = ttk.Label(
                self.set_frame_one, text="Reset domain color: ", width=20
            )
            self.domain_color_label.grid(row=1, column=0, sticky=W, pady=0)

            self.Reset_domain_color = ttk.Entry(self.set_frame_one, width=10)
            self.Reset_domain_color.grid(row=1, column=1, sticky=W, pady=0)
            Tooltip(
                self.Reset_domain_color,
                "Input value must be int type and in [1:cluster_name]: 2",
            )
            self.color_update = None

            def Reset_single_domain_color():
                from tkinter import colorchooser, filedialog

                colorvalue = colorchooser.askcolor()
                color = colorvalue[1]
                print(color)
                cluster = self.Reset_domain_color.get()
                print(cluster)
                self.color_reset[int(cluster) - 1] = color
                # ann_data.uns['Consensus_clustering_colors'] = self.color_reset[:len(ann_data.uns['Consensus_clustering_colors'])]
                # # self.color_update = ann_data.uns['mclust_colors']
                import matplotlib.pyplot as plt

                plt.rcParams["figure.figsize"] = (3, 3)
                draw_images()

                figures = os.listdir(fig_path)
                global image_list
                image_list = []
                for i in range(len(figures)):
                    img = Image.open(fig_path + "/" + figures[i])
                    print(img.size)
                    image_list.append(ttk.PhotoImage(file=fig_path + "/" + figures[i]))
                    self.result_images = ttk.Label(self.show_frame, image=image_list[i])
                    self.result_images.grid(
                        row=i // 3, column=i % 3, sticky=W, pady=0, padx=0
                    )
                    s = i

            self.Reset_domain_btn = ttk.Button(
                self.set_frame_one,
                width=10,
                text="Confirm",
                command=Reset_single_domain_color,
            )
            self.Reset_domain_btn.grid(row=1, column=2, sticky=W, pady=10)
            self.update_image_label = ttk.Label(
                self.set_frame_one, text="Reset image dpi: ", width=20
            )
            self.update_image_label.grid(row=2, column=0, sticky=W, pady=10)
            self.update_image_scale = ttk.Entry(self.set_frame_one, width=10)
            self.update_image_scale.grid(row=2, column=1, sticky=W, pady=10)
            Tooltip(
                self.update_image_scale, "Input value must be int type and >= 300: 300"
            )

            def ipdate_hd():
                dpi = self.update_image_scale.get()
                print(dpi)
                print(self.color_update)
                # ann_data.uns['Consensus_clustering_colors'] = self.color_update
                import matplotlib.pyplot as plt

                plt.rcParams["figure.figsize"] = (3, 3)
                sc.set_figure_params(dpi=dpi)
                draw_images()

            self.Reset_domain_btn = ttk.Button(
                self.set_frame_one, width=10, text="Save", command=ipdate_hd
            )
            self.Reset_domain_btn.grid(row=2, column=2, sticky=W, pady=0)

            self.gene_visualization_label = ttk.Label(
                self.set_frame_one, text="Input gene name: ", width=20
            )
            self.gene_visualization_label.grid(row=3, column=0, sticky=W, pady=0)

            self.gene_visualization_entry = ttk.Entry(self.set_frame_one, width=10)
            self.gene_visualization_entry.grid(row=3, column=1, sticky=W, pady=0)
            Tooltip(self.gene_visualization_entry, "Gene name: NEFH/ATP2B4")

            def gene_visualization():
                gene = self.gene_visualization_entry.get()
                if gene in ann_data.var_names:
                    sc.pl.spatial(
                        ann_data,
                        img_key="hires",
                        color=gene,
                        title="$" + gene + "$",
                        show=False,
                        save=gene + ".png",
                    )
                else:
                    print(f"{gene} is not in current data!")
                global img0
                photo = Image.open(test_file_path + "/figures/show" + gene + ".png")
                img0 = ImageTk.PhotoImage(photo)
                img1 = ttk.Label(self.set_frame_two, image=img0)
                img1.grid(row=0, column=0, sticky=W, pady=0)
                if os.path.exists(test_file_path + "/figures/show" + gene + ".png"):
                    os.remove(test_file_path + "/figures/show" + gene + ".png")
                    print("yes")
                else:
                    print("error！")

            self.gene_visualization_btn = ttk.Button(
                self.set_frame_one, width=10, command=gene_visualization, text="Show"
            )
            self.gene_visualization_btn.grid(row=3, column=2, sticky=W, pady=0)

            figures = os.listdir(fig_path)
            global image_list
            image_list = []
            for i in range(len(figures)):
                image_list.append(ttk.PhotoImage(file=fig_path + "/" + figures[i]))
                self.result_images = ttk.Label(self.show_frame, image=image_list[i])
                self.result_images.grid(row=i // 3, column=i % 3, sticky=W, pady=0)
                s = i

            def VIEW_3D():
                import webbrowser

                self.web_Server_Thread(os.path.join("/DLPFC/webcache"), 8050)
                webbrowser.open("http://127.0.0.1:8050/")

            self.Reset = ttk.Button(
                self.set_frame_one, text="3D VIEW", command=VIEW_3D, width=10
            )
            self.Reset.grid(row=0, column=1, sticky=W, pady=0)
            self.result_queue.put(ann_data)
        else:
            Messagebox.show_warning(
                title="Attention", message="Waiting for models training!!!"
            )

    def STAMarker_Data_Analysis(self):
        model_dir = test_file_path + "/STAMarker_ouput/"
        self.model_train_flag = False
        self.cluster_flag = False
        if os.path.exists(model_dir):
            shutil.rmtree(model_dir)
        for i in range(self.result_queue.qsize() - 1):
            adata_remove = self.result_queue.get()
            del adata_remove
        if not self.result_queue.empty():
            ann_data = self.result_queue.get()
            if "GroundTruth" in ann_data.obs.columns:
                self.label_files_exit = True
            if "highly_variable" in ann_data.var.columns:
                ann_data = ann_data[:, ann_data.var["highly_variable"]]
            else:
                sc.pp.highly_variable_genes(
                    ann_data, flavor="seurat_v3", n_top_genes=3000
                )
                ann_data = ann_data[:, ann_data.var["highly_variable"]]
            Cal_Spatial_Net(ann_data, int(self.rad_cutoff_value))
            Stats_Spatial_Net(ann_data)
            stamarker = STAMarker(
                model_dir=model_dir,
                in_features=ann_data.shape[1],
                hidden_dims=[512, 30],
                n_models=int(self.alpha_value),
                device=torch.device("cuda:0"),
            )
            stamarker.train(
                ann_data,
                lr=1e-4,
                n_epochs=int(self.train_epoch_num),
                gradient_clip=5.0,
                use_net="Spatial_Net",
                resume=False,
                plot_consensus=True,
                clf_n_epochs=int(self.cla_epoch_num),
                n_clusters=int(self.cluster_value),
            )
            stamarker.predict(ann_data, use_net="Spatial_Net")
            output = stamarker.select_spatially_variable_genes(
                ann_data, use_smap="smap", alpha=1.5
            )
            ann_data.obs["Consensus_clustering"] = ann_data.uns["STAMarker"][
                "consensus_labels"
            ].astype(str)
            genes_list = output["gene_list"]
            df = pd.DataFrame(genes_list)
            df.to_csv(
                os.path.join(running_path, "result", "STAMarker_find_SVGs.txt"),
                sep="\t",
                index=False,
                header=False,
            )
            if self.data_type is None:
                self.data_type = "10x"
            data_path = (
                test_file_path
                + "/STAMarker_ouput/"
                + self.method_flag
                + "_"
                + self.data_type
                + "_output"
            )
            if not os.path.isdir(data_path):
                os.mkdir(data_path)
            num = 1
            enr = gp.enrichr(
                gene_list=genes_list,  # or "./tests/data/gene_list.txt",
                gene_sets=["GO_Biological_Process_2018", "KEGG_2019"],
                organism="human",
                outdir=None,
            )

            ax = dotplot(
                enr.results,
                column="Adjusted P-value",
                x="Gene_set",
                size=10,
                top_term=5,
                figsize=(3, 3),
                title="Enrich",
                xticklabels_rot=45,
                show_ring=True,
                ofname=os.path.join(data_path, "STAMarker_SVGs_" + str(num) + ".png"),
                marker="o",
            )
            num = num + 1
            ax = barplot(
                enr.results,
                column="Adjusted P-value",
                group="Gene_set",
                size=10,
                top_term=5,
                figsize=(3, 3),
                ofname=os.path.join(data_path, "STAMarker_SVGs_" + str(num) + ".png"),
                color=["darkred", "darkblue"],
            )
            num = num + 1
            ax = dotplot(
                enr.res2d,
                title="KEGG_2021",
                cmap="viridis_r",
                size=10,
                figsize=(3, 3),
                ofname=os.path.join(data_path, "STAMarker_SVGs_" + str(num) + ".png"),
            )
            num = num + 1
            ax = barplot(
                enr.res2d,
                title="KEGG_2021",
                figsize=(3, 3),
                color="darkred",
                ofname=os.path.join(data_path, "STAMarker_SVGs_" + str(num) + ".png"),
            )

            # self.result_queue.put(ann_data)
            ann_data.write_h5ad(
                os.path.join(running_path, "result", "STAMarker_output.h5ad")
            )
            self.pb.stop()
            self.pb["value"] = 100
            self.setvar("prog-message", "STAMarker run over!")
            self.setvar("End-time", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            EndTime = datetime.now().replace(microsecond=0)
            self.setvar("total-time-cost", EndTime - self.StartTime)
            self.model_train_flag = True
            self.cluster_flag = True

    def STAMarker_Thread_two(self):
        T = threading.Thread(target=self.STAMarker_Data_Analysis)
        # T.setDaemon(True)
        T.start()

    def STAGE_show(self):
        if self.cluster_flag:
            self.result_flag = True
            print(f"self.result_queue.qsize() is {self.result_queue.qsize()}")
            adata = self.result_queue.get()
            adata_sample = self.result_queue.get()
            adata_stage = self.result_queue.get()
            gene_names1 = set(adata.var_names)
            gene_names2 = set(adata_sample.var_names)
            gene_names3 = set(adata_stage.var_names)

            common_gene_names_list = []
            common_gene_names = gene_names1 & gene_names2 & gene_names3
            common_gene_names_list = [gene for gene in common_gene_names]

            def draw_images():
                i = 1
                j = 1
                from matplotlib import pyplot as plt

                os.chdir(Raw_PATH)
                plt.rcParams["font.sans-serif"] = "Arial"
                plt.rcParams["figure.figsize"] = (3, 3)
                path = (
                    test_file_path
                    + "/figures/"
                    + self.method_flag
                    + "_"
                    + self.data_type
                    + "/"
                )
                if self.label_files_exit:
                    sc.pl.spatial(
                        adata,
                        img_key="hires",
                        color="layer",
                        title="Ground Truth",
                        show=False,
                        save="STAGE_" + str(i) + ".png",
                        spot_size=self.spot_size,
                    )
                    shutil.move(
                        test_file_path + "/figures/showSTAGE_" + str(i) + ".png",
                        path + "STAGE_" + str(j) + "_" + str(i) + ".png",
                    )

                    adata.obsm["coord"][:, 1] = adata.obsm["coord"][:, 1] * (-1)
                    i = i + 1
                    used_adata = adata[adata.obs["layer"].isna() == False,]
                    plt.rcParams["figure.figsize"] = (3, 3)
                    sc.pl.umap(
                        used_adata,
                        color="layer",
                        title="Original",
                        show=False,
                        s=6,
                        save="STAGE_" + str(i) + ".png",
                    )
                    shutil.move(
                        test_file_path + "/figures/umapSTAGE_" + str(i) + ".png",
                        path + "STAGE_" + str(j) + "_" + str(i) + ".png",
                    )

                    i = i + 1
                    plt.rcParams["figure.figsize"] = (3, 3)
                    sc.tl.paga(used_adata, groups="layer")
                    sc.pl.paga(
                        used_adata,
                        color="layer",
                        title="Original-PAGA",
                        show=False,
                        save="STAGE_" + str(i) + ".png",
                    )
                    shutil.move(
                        test_file_path + "/figures/pagaSTAGE_" + str(i) + ".png",
                        path + "STAGE_" + str(j) + "_" + str(i) + ".png",
                    )

                    i = i + 1
                    used_adata_sample = adata_sample[
                        adata_sample.obs["layer"].isna() == False,
                    ]
                    plt.rcParams["figure.figsize"] = (3, 3)
                    used_adata_sample.uns["layer_colors"] = used_adata.uns[
                        "layer_colors"
                    ]
                    sc.pl.umap(
                        used_adata_sample,
                        color="layer",
                        title="Down-Sampling",
                        show=False,
                        save="STAGE_" + str(i) + ".png",
                    )
                    shutil.move(
                        test_file_path + "/figures/umapSTAGE_" + str(i) + ".png",
                        path + "STAGE_" + str(j) + "_" + str(i) + ".png",
                    )

                    i = i + 1
                    plt.rcParams["figure.figsize"] = (3, 3)
                    sc.tl.paga(used_adata_sample, groups="layer")
                    sc.pl.paga(
                        used_adata_sample,
                        color="layer",
                        title="Down-Sampling-PAGA",
                        show=False,
                        save="STAGE_" + str(i) + ".png",
                    )
                    shutil.move(
                        test_file_path + "/figures/pagaSTAGE_" + str(i) + ".png",
                        path + "STAGE_" + str(j) + "_" + str(i) + ".png",
                    )

                    i = i + 1
                    used_adata_stage = adata_stage[
                        adata_stage.obs["layer"].isna() == False,
                    ]
                    plt.rcParams["figure.figsize"] = (3, 3)
                    used_adata_stage.uns["layer_colors"] = used_adata.uns[
                        "layer_colors"
                    ]
                    sc.pl.umap(
                        used_adata_stage,
                        color="layer",
                        title="Recoverd",
                        show=False,
                        save="STAGE_" + str(i) + ".png",
                    )
                    shutil.move(
                        test_file_path + "/figures/umapSTAGE_" + str(i) + ".png",
                        path + "STAGE_" + str(j) + "_" + str(i) + ".png",
                    )

                    i = i + 1
                    plt.rcParams["figure.figsize"] = (3, 3)
                    sc.tl.paga(used_adata_stage, groups="layer")
                    sc.pl.paga(
                        used_adata_stage,
                        color="layer",
                        title="Recoverd-PAGA",
                        show=False,
                        save="STAGE_" + str(i) + ".png",
                    )
                    shutil.move(
                        test_file_path + "/figures/pagaSTAGE_" + str(i) + ".png",
                        path + "STAGE_" + str(j) + "_" + str(i) + ".png",
                    )
                j = j + 1
                i = 1

                if self.show_gene_name not in common_gene_names:
                    self.show_gene_name = common_gene_names_list[0]
                plt.rcParams["figure.figsize"] = (3, 3)
                sc.pl.embedding(
                    adata,
                    basis="coord",
                    color=self.show_gene_name,
                    title=self.show_gene_name,
                    s=6,
                    show=False,
                    color_map=self.gene_color_type,
                    save="STAGE_" + str(i) + ".png",
                )
                shutil.move(
                    test_file_path + "/figures/coordSTAGE_" + str(i) + ".png",
                    path + "STAGE_" + str(j) + "_" + str(i) + ".png",
                )

                i = i + 1
                adata_sample.obsm["coord"][:, 1] = adata_sample.obsm["coord"][:, 1] * (
                    -1
                )
                plt.rcParams["figure.figsize"] = (3, 3)
                sc.pl.embedding(
                    adata_sample,
                    basis="coord",
                    color=self.show_gene_name,
                    title=self.show_gene_name,
                    s=6,
                    show=False,
                    color_map=self.gene_color_type,
                    save="STAGE_" + str(i) + ".png",
                )
                shutil.move(
                    test_file_path + "/figures/coordSTAGE_" + str(i) + ".png",
                    path + "STAGE_" + str(j) + "_" + str(i) + ".png",
                )

                i = i + 1
                plt.rcParams["figure.figsize"] = (3, 3)
                sc.pl.embedding(
                    adata_stage,
                    basis="coord",
                    color=self.show_gene_name,
                    title=self.show_gene_name,
                    s=6,
                    show=False,
                    color_map=self.gene_color_type,
                    save="STAGE_" + str(i) + ".png",
                )
                shutil.move(
                    test_file_path + "/figures/coordSTAGE_" + str(i) + ".png",
                    path + "STAGE_" + str(j) + "_" + str(i) + ".png",
                )
                print(f"Result images are saved in : {test_file_path}/figures")

            def scoller():
                self.figure_ybar = ttk.Scrollbar(
                    self.figure_Frame, orient=VERTICAL, cursor="draft_small"
                )
                self.figure_ybar.pack(side=RIGHT, fill=Y)
                self.figure_ybar.config(command=self.canvas.yview)

                self.figure_xbar = ttk.Scrollbar(
                    self.figure_Frame, orient=HORIZONTAL, cursor="draft_small"
                )
                self.figure_xbar.pack(side=BOTTOM, fill=X)
                self.figure_xbar.config(command=self.canvas.xview)

                self.canvas.configure(yscrollcommand=self.figure_xbar.set)
                self.canvas.config(yscrollincrement=1)

                self.canvas.configure(xscrollcommand=self.figure_xbar.set)
                self.canvas.config(xscrollincrement=1)
                self.canvas.pack(side=LEFT, expand=YES, fill=BOTH, padx=0)  # anchor=NW
                self.canvas.create_window((0, 0), window=self.show_frame, anchor=N)
                self.show_frame.bind("<Configure>", onFrameconfigure)

            def onFrameconfigure(event):
                self.canvas.configure(
                    scrollregion=self.canvas.bbox("all"), width=1000, height=650
                )

            def choose_color():
                adata.obsm["coord"][:, 1] = adata.obsm["coord"][:, 1] * (-1)
                adata_sample.obsm["coord"][:, 1] = adata_sample.obsm["coord"][:, 1] * (
                    -1
                )
                self.gene_color_type = color_panel.five_gene_color[random.randint(0, 5)]
                draw_images()

                fig_path = (
                    test_file_path
                    + "/figures/"
                    + self.method_flag
                    + "_"
                    + self.data_type
                )
                figures = os.listdir(fig_path)
                global image_list
                image_list = []
                for i in range(len(figures)):
                    image_list.append(ttk.PhotoImage(file=fig_path + "/" + figures[i]))
                    self.result_images = ttk.Label(self.show_frame, image=image_list[i])
                    self.result_images.grid(row=i // 4, column=i % 4, sticky=W, pady=0)

            if self.data_type is None:
                self.data_type = "10x"
            fig_path = (
                test_file_path + "/figures/" + self.method_flag + "_" + self.data_type
            )
            if not os.path.isdir(fig_path):
                os.mkdir(fig_path)

            draw_images()
            self.figure_Frame = ttk.Frame(self.right_panel)
            self.figure_Frame.pack(side=TOP, fill=X, expand=YES)

            self.canvas = ttk.Canvas(self.figure_Frame)
            self.show_frame = ttk.Frame(self.canvas)
            scoller()

            figures = os.listdir(fig_path)
            global image_list
            image_list = []
            for i in range(len(figures)):
                image_list.append(ttk.PhotoImage(file=fig_path + "/" + figures[i]))
                self.result_images = ttk.Label(self.show_frame, image=image_list[i])
                self.result_images.grid(row=i // 4, column=i % 4, sticky=W, pady=0)
                s = i

            self.set_frame = ttk.Frame(
                self.figure_Frame, borderwidth=2, relief="sunken"
            )
            self.set_frame.pack(side="right", expand=YES, anchor=N)
            self.set_frame_one = ttk.Frame(self.set_frame)
            self.set_frame_one.pack(side=TOP, expand=YES)
            self.set_frame_two = ttk.Frame(self.set_frame)
            self.set_frame_two.pack(side=TOP, expand=YES)

            self.Reset = ttk.Button(
                self.set_frame_one,
                text="Reset all colors",
                command=choose_color,
                width=12,
            )
            self.Reset.grid(row=0, column=0, sticky=W, pady=0, padx=0)

            def Reset():
                python = sys.executable
                os.execl(python, python, *sys.argv)

            self.Reset = ttk.Button(
                self.set_frame_one, text="Reset STABox", command=Reset, width=12
            )
            self.Reset.grid(row=0, column=1, sticky=W, pady=0, padx=0)

            self.domain_color_label = ttk.Label(
                self.set_frame_one, text="Reset domain color: ", width=20
            )
            self.domain_color_label.grid(row=1, column=0, sticky=W, pady=0)

            self.Reset_domain_color = ttk.Entry(self.set_frame_one, width=10)
            self.Reset_domain_color.grid(row=1, column=1, sticky=W, pady=0)
            Tooltip(
                self.Reset_domain_color,
                "Input value must be int type and in [1:cluster_name]: 2",
            )
            self.color_update = None

            def Reset_single_domain_color():
                from tkinter import colorchooser, filedialog

                colorvalue = colorchooser.askcolor()
                color = colorvalue[1]
                print(color)
                cluster = self.Reset_domain_color.get()
                import matplotlib.pyplot as plt

                plt.rcParams["figure.figsize"] = (3, 3)
                draw_images()
                figures = os.listdir(fig_path)
                global image_list
                image_list = []
                for i in range(len(figures)):
                    img = Image.open(fig_path + "/" + figures[i])
                    image_list.append(ttk.PhotoImage(file=fig_path + "/" + figures[i]))
                    self.result_images = ttk.Label(self.show_frame, image=image_list[i])
                    self.result_images.grid(
                        row=i // 3, column=i % 3, sticky=W, pady=0, padx=0
                    )
                    s = i

            self.Reset_domain_btn = ttk.Button(
                self.set_frame_one,
                width=10,
                text="Confirm",
                command=Reset_single_domain_color,
            )
            self.Reset_domain_btn.grid(row=1, column=2, sticky=W, pady=10)
            self.update_image_label = ttk.Label(
                self.set_frame_one, text="Reset image dpi: ", width=20
            )
            self.update_image_label.grid(row=2, column=0, sticky=W, pady=10)
            self.update_image_scale = ttk.Entry(self.set_frame_one, width=10)
            self.update_image_scale.grid(row=2, column=1, sticky=W, pady=10)
            Tooltip(
                self.update_image_scale, "Input value must be int type and >= 300: 300"
            )

            def ipdate_hd():
                dpi = self.update_image_scale.get()
                import matplotlib.pyplot as plt

                plt.rcParams["figure.figsize"] = (3, 3)
                sc.set_figure_params(dpi=dpi)
                draw_images()
                figures = os.listdir(fig_path)

            self.Reset_domain_btn = ttk.Button(
                self.set_frame_one, width=10, text="Save", command=ipdate_hd
            )
            self.Reset_domain_btn.grid(row=2, column=2, sticky=W, pady=0)

            self.gene_visualization_label = ttk.Label(
                self.set_frame_one, text="Input gene name: ", width=20
            )
            self.gene_visualization_label.grid(row=3, column=0, sticky=W, pady=0)

            self.gene_visualization_entry = ttk.Entry(self.set_frame_one, width=10)
            self.gene_visualization_entry.grid(row=3, column=1, sticky=W, pady=0)
            Tooltip(self.gene_visualization_entry, "Gene name: NEFH/ATP2B4")

            def gene_visualization():
                gene = self.gene_visualization_entry.get()
                if gene not in adata.var_names:
                    raise
                sc.pl.spatial(
                    adata,
                    img_key="hires",
                    color=gene,
                    title="$" + gene + "$",
                    show=False,
                    save=gene + ".png",
                )
                global img0
                photo = Image.open(test_file_path + "/figures/show" + gene + ".png")
                img0 = ImageTk.PhotoImage(photo)
                img1 = ttk.Label(self.set_frame_two, image=img0)
                img1.grid(row=0, column=0, sticky=W, pady=0)
                if os.path.exists(test_file_path + "/figures/show" + gene + ".png"):
                    os.remove(test_file_path + "/figures/show" + gene + ".png")
                    print("yes")
                else:
                    print("error！")

            self.gene_visualization_btn = ttk.Button(
                self.set_frame_one, width=10, command=gene_visualization, text="Show"
            )
            self.gene_visualization_btn.grid(row=3, column=2, sticky=W, pady=0)

            self.result_queue.put(adata)
            self.result_queue.put(adata_sample)
            self.result_queue.put(adata_stage)
        else:
            Messagebox.show_warning(
                title="Attention", message="Waiting for models training!!!"
            )

    def STAGE_10XData_Analysis(self):
        self.model_train_flag = False
        self.cluster_flag = False
        for i in range(self.result_queue.qsize() - 1):
            adata_remove = self.result_queue.get()
            del adata_remove
        adata = self.result_queue.get()
        if "GroundTruth" in adata.obs.columns:
            self.label_files_exit = True
            adata.obs["layer"] = adata.obs["GroundTruth"]
        adata.obsm["coord"] = adata.obsm["spatial"]
        functions = ["generation", "recovery", "3d_model"]
        filepath = (
            test_file_path + "/" + self.method_flag + "_" + self.data_type + "_output"
        )
        if os.path.isfile(filepath):
            os.mkdir(filepath)
        if self.functions_choose not in functions:
            self.functions_choose = functions[0]
        if self.data_type is None:
            self.data_type = "10x"
        adata_sample, adata_stage = STAGE(
            adata,
            save_path=filepath,
            data_type=self.data_type,
            experiment=self.functions_choose,
            down_ratio=float(self.alpha_value),
            coord_sf=77,
            train_epoch=int(self.genes),
            batch_size=512,
            learning_rate=1e-3,
            w_recon=0.1,
            w_w=0.1,
            w_l1=0.1,
        )
        sc.pp.pca(adata, n_comps=30)
        sc.pp.neighbors(adata, use_rep="X_pca")
        sc.tl.umap(adata)

        sc.pp.pca(adata_sample, n_comps=30)
        sc.pp.neighbors(adata_sample, use_rep="X_pca")
        sc.tl.umap(adata_sample)

        sc.pp.pca(adata_stage, n_comps=30)
        sc.pp.neighbors(adata_stage, use_rep="X_pca")
        sc.tl.umap(adata_stage)

        plt.rcParams["font.sans-serif"] = "Arial"
        self.gene_color_type = "viridis"
        # self.result_queue.put(adata)
        # self.result_queue.put(adata_sample)
        # self.result_queue.put(adata_stage)
        adata.write_h5ad(os.path.join(running_path, "result", "STAGE_raw_output.h5ad"))
        adata_sample.write_h5ad(
            os.path.join(running_path, "result", "STAGE_adata_sample_output.h5ad")
        )
        adata_stage.write_h5ad(
            os.path.join(running_path, "result", "STAGE_adata_stage_output.h5ad")
        )

        self.pb.stop()
        self.pb["value"] = 100
        self.setvar("prog-message", "STAGE run over!")
        self.setvar("End-time", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        EndTime = datetime.now().replace(microsecond=0)
        self.setvar("total-time-cost", EndTime - self.StartTime)
        self.model_train_flag = True
        self.cluster_flag = True

    def STAGE_Thread(self):
        T = threading.Thread(target=self.STAGE_10XData_Analysis)
        # T.setDaemon(True)
        T.start()

    def SpaGCN_show(self):
        if self.cluster_flag:
            self.result_flag = True
            print(
                f"self.Mcluster_num type and value are {self.Mcluster_num}, {type(self.Mcluster_num)}"
            )
            if self.data_type is None:
                self.data_type = "10x"

            def draw_images(sdata):
                i = 1
                temp_path = os.getcwd()
                os.chdir(test_file_path)
                from matplotlib import pyplot as plt

                plt.rcParams["font.sans-serif"] = "Arial"
                plt.rcParams["figure.figsize"] = (3, 3)
                path = (
                    test_file_path
                    + "/figures/"
                    + self.method_flag
                    + "_"
                    + self.data_type
                    + "/"
                )
                if self.label_files_exit:
                    sc.pl.spatial(
                        sdata,
                        color="GroundTruth",
                        title=["Ground Truth"],
                        show=False,
                        spot_size=self.spot_size,
                        save="SpaGCN_" + str(i) + ".png",
                    )
                    shutil.move(
                        test_file_path + "/figures/showSpaGCN_" + str(i) + ".png",
                        path + "SpaGCN_" + str(i) + ".png",
                    )
                    i = i + 1
                    domains = "pred"
                    sc.pl.spatial(
                        sdata,
                        color=domains,
                        title=["SpaGCN-pred"],
                        show=False,
                        spot_size=self.spot_size,
                        save="SpaGCN_" + str(i) + ".png",
                    )
                    shutil.move(
                        test_file_path + "/figures/showSpaGCN_" + str(i) + ".png",
                        path + "SpaGCN_" + str(i) + ".png",
                    )

                    i = i + 1
                    domains = "refined_pred"
                    sc.pl.spatial(
                        sdata,
                        color=domains,
                        title=["SpaGCN-refined_pred"],
                        show=False,
                        spot_size=self.spot_size,
                        save="SpaGCN_" + str(i) + ".png",
                    )
                    shutil.move(
                        test_file_path + "/figures/showSpaGCN_" + str(i) + ".png",
                        path + "SpaGCN_" + str(i) + ".png",
                    )

                    # plt.rcParams["figure.figsize"] = (3, 3)
                    # sc.pl.umap(adata, color="mclust", title=['SpaGCN-mclust-umap'], show=False, s=6,
                    #            save='SpaGCN_' + str(i) + '.png')
                    # shutil.move(test_file_path + '/figures/umapSpaGCN_' + str(i) + '.png',
                    #             path + '/SpaGCN_mclust_umap.png')
                    #
                    # i = i + 1
                    # sc.pl.spatial(adata, color="mclust", title=['SpaGCN-mclust'], show=False,
                    #               spot_size=self.spot_size, save='SpaGCN_' + str(i) + '.png')
                    # shutil.move(test_file_path + '/figures/showSpaGCN_' + str(i) + '.png',
                    #             path + '/SpaGCN_mclust.png')

                    i = i + 1
                    sc.pl.umap(
                        adata,
                        color="louvain",
                        title=["SpaGCN-louvain-umap"],
                        show=False,
                        s=6,
                        save="SpaGCN_" + str(i) + ".png",
                    )
                    shutil.move(
                        test_file_path + "/figures/umapSpaGCN_" + str(i) + ".png",
                        path + "/SpaGCN_louvain_umap.png",
                    )

                    i = i + 1
                    sc.pl.spatial(
                        adata,
                        color="louvain",
                        title=["SpaGCN-louvain"],
                        show=False,
                        spot_size=self.spot_size,
                        save="SpaGCN_" + str(i) + ".png",
                    )
                    shutil.move(
                        test_file_path + "/figures/showSpaGCN_" + str(i) + ".png",
                        path + "/SpaGCN_louvain.png",
                    )
                else:
                    domains = "pred"
                    sc.pl.spatial(
                        sdata,
                        color=domains,
                        title=["SpaGCN-pred"],
                        show=False,
                        spot_size=self.spot_size,
                        save="SpaGCN_" + str(i) + ".png",
                    )
                    shutil.move(
                        test_file_path + "/figures/showSpaGCN_" + str(i) + ".png",
                        path + "SpaGCN_" + str(i) + ".png",
                    )

                    i = i + 1
                    domains = "refined_pred"
                    sc.pl.spatial(
                        sdata,
                        color=domains,
                        title=["SpaGCN-refined_pred"],
                        show=False,
                        spot_size=self.spot_size,
                        save="SpaGCN_" + str(i) + ".png",
                    )
                    shutil.move(
                        test_file_path + "/figures/showSpaGCN_" + str(i) + ".png",
                        path + "SpaGCN_" + str(i) + ".png",
                    )

                    # i = i + 1
                    # plt.rcParams["figure.figsize"] = (3, 3)
                    # sc.pl.umap(adata, color="mclust", title=['SpaGCN-mclust-umap'], show=False, s=6,
                    #            save='SpaGCN_' + str(i) + '.png')
                    # shutil.move(test_file_path + '/figures/umapSpaGCN_' + str(i) + '.png',
                    #             path + '/SpaGCN_mclust_umap.png')
                    #
                    # i = i + 1
                    # sc.pl.spatial(adata, color="mclust", title=['SpaGCN-mclust'], show=False,
                    #               spot_size=self.spot_size, save='SpaGCN_' + str(i) + '.png')
                    # shutil.move(test_file_path + '/figures/showSpaGCN_' + str(i) + '.png',
                    #             path + '/SpaGCN_mclust.png')

                    i = i + 1
                    sc.pl.umap(
                        adata,
                        color="louvain",
                        title=["SpaGCN-louvain-umap"],
                        show=False,
                        s=6,
                        save="SpaGCN_" + str(i) + ".png",
                    )
                    shutil.move(
                        test_file_path + "/figures/umapSpaGCN_" + str(i) + ".png",
                        path + "/SpaGCN_louvain_umap.png",
                    )

                    i = i + 1
                    sc.pl.spatial(
                        adata,
                        color="louvain",
                        title=["SpaGCN-louvain"],
                        show=False,
                        spot_size=self.spot_size,
                        save="SpaGCN_" + str(i) + ".png",
                    )
                    shutil.move(
                        test_file_path + "/figures/showSpaGCN_" + str(i) + ".png",
                        path + "/SpaGCN_louvain.png",
                    )
                    os.chdir(temp_path)
                print(f"Result images are saved in : {test_file_path}/figures")

            def scoller():
                self.figure_ybar = ttk.Scrollbar(
                    self.figure_Frame, orient=VERTICAL, cursor="draft_small"
                )
                self.figure_ybar.pack(side=RIGHT, fill=Y)
                self.figure_ybar.config(command=self.canvas.yview)

                self.figure_xbar = ttk.Scrollbar(
                    self.figure_Frame, orient=HORIZONTAL, cursor="draft_small"
                )
                self.figure_xbar.pack(side=BOTTOM, fill=X)
                self.figure_xbar.config(command=self.canvas.xview)

                self.canvas.configure(yscrollcommand=self.figure_xbar.set)
                self.canvas.config(yscrollincrement=1)

                self.canvas.configure(xscrollcommand=self.figure_xbar.set)
                self.canvas.config(xscrollincrement=1)
                self.canvas.pack(side=LEFT, expand=YES, fill=BOTH, padx=0)  # anchor=NW
                self.canvas.create_window((0, 0), window=self.show_frame, anchor=N)
                self.show_frame.bind("<Configure>", onFrameconfigure)

            def onFrameconfigure(event):
                self.canvas.configure(
                    scrollregion=self.canvas.bbox("all"), width=1000, height=650
                )

            def choose_color():
                colorvalue = colorchooser.askcolor()
                color_list = []
                color = colorvalue[1]
                color_lens = [
                    adata.uns["louvain_colors"],
                    adata.uns["pred_colors"],
                    adata.uns["refined_pred_colors"],
                ]
                max_len = max(color_lens, key=len)
                if len(max_len) > self.cluster_value:
                    self.cluster_value = len(max_len)
                print(color)
                if color[1:3] == "ff":
                    color_list.append(color)
                    list = random.sample(color_panel.FF_color, self.cluster_value)
                    color_list = color_list + list
                elif color[1:3] == "cc":
                    color_list.append(color)
                    list = random.sample(color_panel.CC_color, self.cluster_value)
                    color_list = color_list + list
                elif color[1:3] == "99":
                    color_list.append(color)
                    list = random.sample(color_panel.NN_color, self.cluster_value)
                    color_list = color_list + list
                elif color[1:3] == "66":
                    color_list.append(color)
                    list = random.sample(color_panel.SS_color, self.cluster_value)
                    color_list = color_list + list
                elif color[1:3] == "33":
                    color_list.append(color)
                    list = random.sample(color_panel.TT_color, self.cluster_value)
                    color_list = color_list + list
                elif color[1:3] == "00":
                    color_list.append(color)
                    list = random.sample(color_panel.ZZ_color, self.cluster_value)
                    color_list = color_list + list
                else:
                    color_list.append(color)
                    list = random.sample(color_panel.random_color, self.cluster_value)
                    color_list = color_list + list

                print(color_list)
                # sdata = adata
                if self.label_files_exit:
                    adata.uns["GroundTruth_colors"] = color_list
                adata.uns["pred_colors"] = color_list[: len(adata.uns["pred_colors"])]
                adata.uns["refined_pred_colors"] = color_list[
                    : len(adata.uns["refined_pred_colors"])
                ]
                adata.uns["louvain_colors"] = color_list[
                    : len(adata.uns["louvain_colors"])
                ]
                # adata.uns['mclust_colors'] = color_list[:len(adata.uns['mclust_colors'])]
                self.color_reset = color_list
                os.chdir(Raw_PATH)
                draw_images(adata)

                fig_path = (
                    test_file_path
                    + "/figures/"
                    + self.method_flag
                    + "_"
                    + self.data_type
                    + "/"
                )
                figures = os.listdir(fig_path)
                global image_list
                image_list = []
                for i in range(len(figures)):
                    image_list.append(ttk.PhotoImage(file=fig_path + "/" + figures[i]))
                    self.result_images = ttk.Label(self.show_frame, image=image_list[i])
                    self.result_images.grid(row=i // 3, column=i % 3, sticky=W, pady=0)

            fig_path = (
                test_file_path
                + "/figures/"
                + self.method_flag
                + "_"
                + self.data_type
                + "/"
            )
            if not os.path.isdir(fig_path):
                os.mkdir(fig_path)

            adata = self.result_queue.get()
            # adata = mclust_R(adata, used_obsm='SpaGCN', num_cluster=self.Mcluster_num)
            # sc.tl.louvain(adata, resolution=self.louvain_res)

            draw_images(adata)
            self.figure_Frame = ttk.Frame(self.right_panel)
            self.figure_Frame.pack(side=TOP, fill=X, expand=YES)

            self.canvas = ttk.Canvas(self.figure_Frame)
            self.show_frame = ttk.Frame(self.canvas)
            scoller()
            self.set_frame = ttk.Frame(
                self.figure_Frame, borderwidth=2, relief="sunken"
            )
            self.set_frame.pack(side="right", expand=YES, anchor=N)
            self.set_frame_one = ttk.Frame(self.set_frame)
            self.set_frame_one.pack(side=TOP, expand=YES)
            self.set_frame_two = ttk.Frame(self.set_frame)
            self.set_frame_two.pack(side=TOP, expand=YES)

            self.Reset = ttk.Button(
                self.set_frame_one,
                text="Reset all colors",
                command=choose_color,
                width=12,
            )
            self.Reset.grid(row=0, column=0, sticky=W, pady=0, padx=0)

            self.domain_color_label = ttk.Label(
                self.set_frame_one, text="Reset domain color: ", width=20
            )
            self.domain_color_label.grid(row=1, column=0, sticky=W, pady=0)

            self.Reset_domain_color = ttk.Entry(self.set_frame_one, width=10)
            self.Reset_domain_color.grid(row=1, column=1, sticky=W, pady=0)
            Tooltip(
                self.Reset_domain_color,
                "Input value must be int type and in [1:cluster_name]: 2",
            )
            self.color_update = None

            def Reset_single_domain_color():
                from tkinter import colorchooser, filedialog

                colorvalue = colorchooser.askcolor()
                color = colorvalue[1]
                print(color)
                cluster = self.Reset_domain_color.get()
                self.color_reset[int(cluster) - 1] = color
                adata.uns["louvain_colors"] = self.color_reset[
                    : len(adata.uns["louvain_colors"])
                ]
                # adata.uns['mclust_colors'] = self.color_reset[:len(adata.uns['mclust_colors'])]
                adata.uns["pred_colors"] = self.color_reset[
                    : len(adata.uns["pred_colors"])
                ]
                adata.uns["refined_pred_colors"] = self.color_reset[
                    : len(adata.uns["refined_pred_colors"])
                ]
                self.color_update = self.color_reset
                draw_images(adata)

                figures = os.listdir(fig_path)
                global image_list
                image_list = []
                for i in range(len(figures)):
                    img = Image.open(fig_path + "/" + figures[i])
                    print(img.size)
                    image_list.append(ttk.PhotoImage(file=fig_path + "/" + figures[i]))
                    self.result_images = ttk.Label(self.show_frame, image=image_list[i])
                    self.result_images.grid(
                        row=i // 3, column=i % 3, sticky=W, pady=0, padx=0
                    )
                    s = i

            self.Reset_domain_btn = ttk.Button(
                self.set_frame_one,
                width=10,
                text="Confirm",
                command=Reset_single_domain_color,
            )
            self.Reset_domain_btn.grid(row=1, column=2, sticky=W, pady=10)
            self.update_image_label = ttk.Label(
                self.set_frame_one, text="Reset image dpi: ", width=20
            )
            self.update_image_label.grid(row=2, column=0, sticky=W, pady=10)
            self.update_image_scale = ttk.Entry(self.set_frame_one, width=10)
            self.update_image_scale.grid(row=2, column=1, sticky=W, pady=10)
            Tooltip(
                self.update_image_scale, "Input value must be int type and >= 300: 300"
            )

            def ipdate_hd():
                dpi = self.update_image_scale.get()
                import matplotlib.pyplot as plt

                plt.rcParams["figure.figsize"] = (3, 3)
                sc.set_figure_params(dpi=dpi)
                draw_images(adata)

            self.Reset_domain_btn = ttk.Button(
                self.set_frame_one, width=10, text="Save", command=ipdate_hd
            )
            self.Reset_domain_btn.grid(row=2, column=2, sticky=W, pady=0)

            self.gene_visualization_label = ttk.Label(
                self.set_frame_one, text="Input gene name: ", width=20
            )
            self.gene_visualization_label.grid(row=3, column=0, sticky=W, pady=0)

            self.gene_visualization_entry = ttk.Entry(self.set_frame_one, width=10)
            self.gene_visualization_entry.grid(row=3, column=1, sticky=W, pady=0)
            Tooltip(self.gene_visualization_entry, "Gene name: NEFH/ATP2B4")

            def gene_visualization():
                try:
                    gene = self.gene_visualization_entry.get()
                    if gene not in adata.var_names:
                        raise "gene not in adata!"
                    else:
                        sc.pl.spatial(
                            adata,
                            img_key="hires",
                            color=gene,
                            title="$" + gene + "$",
                            show=False,
                            save=gene + ".png",
                            sopt_size=self.spot_size,
                        )
                    global img0
                    photo = Image.open(test_file_path + "/figures/show" + gene + ".png")
                    img0 = ImageTk.PhotoImage(photo)
                    img1 = ttk.Label(self.set_frame_two, image=img0)
                    img1.grid(row=0, column=0, sticky=W, pady=0)
                    if os.path.exists(test_file_path + "/figures/show" + gene + ".png"):
                        os.remove(test_file_path + "/figures/show" + gene + ".png")
                        print("Figures exits")
                    else:
                        print("Figures no exits！")
                    pass
                except:
                    Messagebox.show_error(
                        "Python Error", "Make sure gene name in dataset"
                    )

            self.gene_visualization_btn = ttk.Button(
                self.set_frame_one, width=10, command=gene_visualization, text="Show"
            )
            self.gene_visualization_btn.grid(row=3, column=2, sticky=W, pady=0)

            global image_list
            figures = os.listdir(fig_path)
            image_list = []
            s = 0
            for i in range(len(figures)):
                image_list.append(ttk.PhotoImage(file=fig_path + "/" + figures[i]))
                self.result_images = ttk.Label(self.show_frame, image=image_list[i])
                self.result_images.grid(row=i // 3, column=i % 3, sticky=W, pady=0)
                s = i

            def VIEW_3D():
                import webbrowser

                self.web_Server_Thread("/DLPFC/webcache", 8029)
                webbrowser.open("http://127.0.0.1:8029/")

            self.Reset = ttk.Button(
                self.set_frame_one, text="3D VIEW", command=VIEW_3D, width=10
            )
            self.Reset.grid(row=0, column=1, sticky=W, pady=0, padx=(0, 100))
            self.result_queue.put(adata)
        else:
            Messagebox.show_warning(
                title="Attention", message="Waiting for models training!!!"
            )

    def SpaGCN_10XData_Analysis(self):
        try:
            self.model_train_flag = False
            self.cluster_flag = False
            for i in range(self.result_queue.qsize() - 1):
                adata_remove = self.result_queue.get()
                del adata_remove
            file_path = self.file_path.rsplit("/", 1)[-2]
            tif_image = glob.glob(file_path + "/*.tif")
            histology = False
            adata = self.result_queue.get()
            if "GroundTruth" in adata.obs.columns:
                self.label_files_exit = True
            x_array = adata.obsm["spatial"][:, 0].tolist()
            y_array = adata.obsm["spatial"][:, 1].tolist()
            x_pixel = adata.obsm["spatial"][:, 0].tolist()
            y_pixel = adata.obsm["spatial"][:, 1].tolist()
            spg.prefilter_genes(adata, min_cells=3)
            spg.prefilter_specialgenes(adata)
            if self.data_type == "10x":
                sc.pp.normalize_per_cell(adata)
            sc.pp.log1p(adata)
            if len(tif_image) > 0:
                histology = True
                img = cv2.imread(tif_image[0])
                adj = spg.calculate_adj_matrix(
                    x=x_pixel,
                    y=y_pixel,
                    x_pixel=x_pixel,
                    y_pixel=y_pixel,
                    image=img,
                    beta=int(self.rad_cutoff_value),
                    alpha=1,
                    histology=histology,
                )
            else:
                adj = spg.calculate_adj_matrix(
                    x=x_pixel, y=y_pixel, histology=histology
                )
            l = spg.search_l(
                float(self.genes), adj, start=0.01, end=1000, tol=0.01, max_run=100
            )
            # If the number of clusters known, we can use the spg.search_res() fnction to search for suitable resolution(optional)
            # For this toy data, we set the number of clusters=7 since this tissue has 7 layers
            # n_clusters = 7
            r_seed = t_seed = n_seed = 100
            res = spg.search_res(
                adata,
                adj,
                l,
                int(self.cluster_value),
                start=0.7,
                step=0.1,
                tol=5e-3,
                lr=0.05,
                max_epochs=20,
                r_seed=r_seed,
                t_seed=t_seed,
                n_seed=n_seed,
            )
            clf = spg.SpaGCN()
            clf.set_l(l)
            random.seed(r_seed)
            torch.manual_seed(t_seed)
            np.random.seed(n_seed)
            clf.train(
                adata,
                adj,
                init_spa=True,
                init="louvain",
                res=res,
                tol=5e-3,
                lr=0.05,
                max_epochs=int(self.alpha_value),
            )
            y_pred, prob = clf.predict()
            adata.obsm["SpaGCN"] = clf.embed
            adata.obs["pred"] = y_pred
            adata.obs["pred"] = adata.obs["pred"].astype("category")
            # shape="hexagon" for Visium data, "square" for ST data.
            adj_2d = spg.calculate_adj_matrix(x=x_array, y=y_array, histology=False)
            refined_pred = spg.refine(
                sample_id=adata.obs.index.tolist(),
                pred=adata.obs["pred"].tolist(),
                dis=adj_2d,
                shape="hexagon",
            )
            adata.obs["refined_pred"] = refined_pred
            adata.obs["refined_pred"] = adata.obs["refined_pred"].astype("category")
            adata.obs["x_pixel"] = adata.obsm["spatial"][:, 1]
            adata.obs["y_pixel"] = adata.obsm["spatial"][:, 0]
            sc.pp.neighbors(adata, use_rep="SpaGCN")
            sc.tl.umap(adata)

            # self.result_queue.put(adata)
            adata.write_h5ad(os.path.join(running_path, "result", "SpaGCN_output.h5ad"))

            self.pb.stop()
            self.pb["value"] = 100
            self.setvar("prog-message", "SpaGCN run over!")
            self.setvar("End-time", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            EndTime = datetime.now().replace(microsecond=0)
            self.setvar("total-time-cost", EndTime - self.StartTime)
            self.model_train_flag = True
            self.cluster_flag = True
        except:
            self.model_train_flag = False
            raise "Check whether the files exists!"

    def SpaGCN_Thread(self):
        T = threading.Thread(target=self.SpaGCN_10XData_Analysis)
        # T.setDaemon(True)
        T.start()
        pass

    def SEDR_show(self):
        if self.cluster_flag:
            self.result_flag = True
            if self.data_type is None:
                self.data_type = "10x"

            def draw_images(sdata):
                temp_path = os.getcwd()
                os.chdir(test_file_path)
                from matplotlib import pyplot as plt

                plt.rcParams["font.sans-serif"] = "Arial"
                plt.rcParams["figure.figsize"] = (3, 3)

                domains = ["SEDR_leiden", "SEDR_mclust", "SEDR_kmeans"]
                path = (
                    test_file_path
                    + "/figures/"
                    + self.method_flag
                    + "_"
                    + self.data_type
                    + "/"
                )
                for i in domains:
                    sc.pl.spatial(
                        sdata,
                        color=i,
                        title=i,
                        show=False,
                        save="SEDR_" + i + ".png",
                        spot_size=self.spot_size,
                    )
                    shutil.move(
                        test_file_path + "/figures/showSEDR_" + i + ".png",
                        path + i + ".png",
                    )

                    sc.pl.umap(
                        adata,
                        color=i,
                        title=f"{i}_umap",
                        show=False,
                        s=6,
                        save="SEDR_" + str(i) + ".png",
                    )
                    shutil.move(
                        test_file_path + "/figures/umapSEDR_" + i + ".png",
                        path + i + "_umap.png",
                    )

                print(f"Result images are saved in : {test_file_path}/figures")
                os.chdir(temp_path)

            def scoller():
                self.figure_ybar = ttk.Scrollbar(
                    self.figure_Frame, orient=VERTICAL, cursor="draft_small"
                )
                self.figure_ybar.pack(side=RIGHT, fill=Y)
                self.figure_ybar.config(command=self.canvas.yview)

                self.figure_xbar = ttk.Scrollbar(
                    self.figure_Frame, orient=HORIZONTAL, cursor="draft_small"
                )
                self.figure_xbar.pack(side=BOTTOM, fill=X)
                self.figure_xbar.config(command=self.canvas.xview)

                self.canvas.configure(yscrollcommand=self.figure_xbar.set)
                self.canvas.config(yscrollincrement=1)

                self.canvas.configure(xscrollcommand=self.figure_xbar.set)
                self.canvas.config(xscrollincrement=1)
                self.canvas.pack(side=LEFT, expand=YES, fill=BOTH, padx=0)  # anchor=NW
                self.canvas.create_window((0, 0), window=self.show_frame, anchor=N)
                self.show_frame.bind("<Configure>", onFrameconfigure)

            def onFrameconfigure(event):
                self.canvas.configure(
                    scrollregion=self.canvas.bbox("all"), width=1000, height=650
                )

            def choose_color():
                colorvalue = colorchooser.askcolor()
                color_list = []
                color = colorvalue[1]
                print(color)
                self.cluster_value = max(
                    len(adata.uns["SEDR_leiden_colors"]),
                    len(adata.uns["SEDR_mclust_colors"]),
                    len(adata.uns["SEDR_kmeans_colors"]),
                )
                if color[1:3] == "ff":
                    color_list.append(color)
                    list = random.sample(color_panel.FF_color, self.cluster_value)
                    color_list = color_list + list
                elif color[1:3] == "cc":
                    color_list.append(color)
                    list = random.sample(color_panel.CC_color, self.cluster_value)
                    color_list = color_list + list
                elif color[1:3] == "99":
                    color_list.append(color)
                    list = random.sample(color_panel.NN_color, self.cluster_value)
                    color_list = color_list + list
                elif color[1:3] == "66":
                    color_list.append(color)
                    list = random.sample(color_panel.SS_color, self.cluster_value)
                    color_list = color_list + list
                elif color[1:3] == "33":
                    color_list.append(color)
                    list = random.sample(color_panel.TT_color, self.cluster_value)
                    color_list = color_list + list
                elif color[1:3] == "00":
                    color_list.append(color)
                    list = random.sample(color_panel.ZZ_color, self.cluster_value)
                    color_list = color_list + list
                else:
                    color_list.append(color)
                    list = random.sample(color_panel.random_color, self.cluster_value)
                    color_list = color_list + list

                print(color_list)
                adata.uns["SEDR_leiden_colors"] = color_list[
                    : len(adata.uns["SEDR_leiden_colors"])
                ]
                adata.uns["SEDR_mclust_colors"] = color_list[
                    : len(adata.uns["SEDR_mclust_colors"])
                ]
                adata.uns["SEDR_kmeans_colors"] = color_list[
                    : len(adata.uns["SEDR_kmeans_colors"])
                ]
                self.color_reset = color_list
                os.chdir(Raw_PATH)
                draw_images(adata)

                fig_path = (
                    test_file_path
                    + "/figures/"
                    + self.method_flag
                    + "_"
                    + self.data_type
                    + "/"
                )
                figures = os.listdir(fig_path)
                global image_list
                image_list = []
                for i in range(len(figures)):
                    image_list.append(ttk.PhotoImage(file=fig_path + "/" + figures[i]))
                    self.result_images = ttk.Label(self.show_frame, image=image_list[i])
                    self.result_images.grid(row=i // 3, column=i % 3, sticky=W, pady=0)

            fig_path = (
                test_file_path
                + "/figures/"
                + self.method_flag
                + "_"
                + self.data_type
                + "/"
            )
            if not os.path.isdir(fig_path):
                os.mkdir(fig_path)

            adata = self.result_queue.get()
            # eval_cluster = int(self.Mcluster_num)
            # eval_resolution = res_search_fixed_clus(adata, eval_cluster)
            # sc.tl.leiden(adata, key_added="SEDR_leiden", resolution=eval_resolution)
            #
            # import rpy2.robjects as robjects
            # robjects.r.library("mclust")
            #
            # import rpy2.robjects.numpy2ri
            # rpy2.robjects.numpy2ri.activate()
            #
            # rmclust = robjects.r['Mclust']
            # res2 = rmclust(adata.X, eval_cluster, 'EEE')
            # mclust_res = np.array(res2[-2])
            #
            # adata.obs['SEDR_mclust'] = mclust_res
            # # adata.obs['SEDR_mclust'] = mclust_R(adata_sedr.X, eval_cluster)
            # adata.obs['SEDR_mclust'] = adata.obs['SEDR_mclust'].astype('int')
            # adata.obs['SEDR_mclust'] = adata.obs['SEDR_mclust'].astype('category')
            #
            # kmeans = KMeans(n_clusters=self.kmeans_num)
            # kmeans.fit(adata.X)
            # adata.obs['SEDR_kmeans'] = kmeans.labels_
            # adata.obs['SEDR_kmeans'] = adata.obs['SEDR_kmeans'].astype('int')
            # adata.obs['SEDR_kmeans'] = adata.obs['SEDR_kmeans'].astype('category')

            draw_images(adata)
            self.figure_Frame = ttk.Frame(self.right_panel)
            self.figure_Frame.pack(side=TOP, fill=X, expand=YES)

            self.canvas = ttk.Canvas(self.figure_Frame)
            self.show_frame = ttk.Frame(self.canvas)

            scoller()
            self.set_frame = ttk.Frame(
                self.figure_Frame, borderwidth=2, relief="sunken"
            )
            self.set_frame.pack(side="right", expand=YES, anchor=N)
            self.set_frame_one = ttk.Frame(self.set_frame)
            self.set_frame_one.pack(side=TOP, expand=YES)
            self.set_frame_two = ttk.Frame(self.set_frame)
            self.set_frame_two.pack(side=TOP, expand=YES)

            self.Reset = ttk.Button(
                self.set_frame_one,
                text="Reset all colors",
                command=choose_color,
                width=12,
            )
            self.Reset.grid(row=0, column=0, sticky=W, pady=0, padx=0)

            self.domain_color_label = ttk.Label(
                self.set_frame_one, text="Reset domain color: ", width=20
            )
            self.domain_color_label.grid(row=1, column=0, sticky=W, pady=0)

            self.Reset_domain_color = ttk.Entry(self.set_frame_one, width=10)
            self.Reset_domain_color.grid(row=1, column=1, sticky=W, pady=0)
            Tooltip(
                self.Reset_domain_color,
                "Input value must be int type and in [1:cluster_name]: 2",
            )
            self.color_update = None

            def Reset_single_domain_color():
                from tkinter import colorchooser, filedialog

                colorvalue = colorchooser.askcolor()
                color = colorvalue[1]
                print(color)
                cluster = self.Reset_domain_color.get()
                print(cluster)
                self.color_reset[int(cluster) - 1] = color
                adata.uns["SEDR_leiden_colors"] = self.color_reset[
                    : len(adata.uns["SEDR_leiden_colors"])
                ]
                adata.uns["SEDR_mclust_colors"] = self.color_reset[
                    : len(adata.uns["SEDR_mclust_colors"])
                ]
                adata.uns["SEDR_kmeans_colors"] = self.color_reset[
                    : len(adata.uns["SEDR_kmeans_colors"])
                ]
                draw_images(adata)

                figures = os.listdir(fig_path)
                global image_list
                image_list = []
                for i in range(len(figures)):
                    img = Image.open(fig_path + "/" + figures[i])
                    print(img.size)
                    image_list.append(ttk.PhotoImage(file=fig_path + "/" + figures[i]))
                    self.result_images = ttk.Label(self.show_frame, image=image_list[i])
                    self.result_images.grid(
                        row=i // 4, column=i % 4, sticky=W, pady=0, padx=0
                    )
                    s = i

            self.Reset_domain_btn = ttk.Button(
                self.set_frame_one,
                width=10,
                text="Confirm",
                command=Reset_single_domain_color,
            )
            self.Reset_domain_btn.grid(row=1, column=2, sticky=W, pady=10)
            self.update_image_label = ttk.Label(
                self.set_frame_one, text="Reset image dpi: ", width=20
            )
            self.update_image_label.grid(row=2, column=0, sticky=W, pady=10)
            self.update_image_scale = ttk.Entry(self.set_frame_one, width=10)
            self.update_image_scale.grid(row=2, column=1, sticky=W, pady=10)
            Tooltip(
                self.update_image_scale, "Input value must be int type and >= 300: 300"
            )

            def ipdate_hd():
                dpi = self.update_image_scale.get()
                print(dpi)
                import matplotlib.pyplot as plt

                plt.rcParams["figure.figsize"] = (3, 3)
                sc.set_figure_params(dpi=dpi)
                draw_images(adata)
                figures = os.listdir(fig_path)
                print(len(figures))

            self.Reset_domain_btn = ttk.Button(
                self.set_frame_one, width=10, text="Save", command=ipdate_hd
            )
            self.Reset_domain_btn.grid(row=2, column=2, sticky=W, pady=0)

            self.gene_visualization_label = ttk.Label(
                self.set_frame_one, text="Input gene name: ", width=20
            )
            self.gene_visualization_label.grid(row=3, column=0, sticky=W, pady=0)

            self.gene_visualization_entry = ttk.Entry(self.set_frame_one, width=10)
            self.gene_visualization_entry.grid(row=3, column=1, sticky=W, pady=0)
            Tooltip(self.gene_visualization_entry, "Gene name: NEFH/ATP2B4")

            def gene_visualization():
                try:
                    gene = self.gene_visualization_entry.get()
                    sc.pl.spatial(
                        adata,
                        img_key="hires",
                        color=gene,
                        title="$" + gene + "$",
                        show=False,
                        save=gene + ".png",
                        sopt_size=50,
                    )
                    global img0
                    photo = Image.open(test_file_path + "/figures/show" + gene + ".png")
                    img0 = ImageTk.PhotoImage(photo)
                    img1 = ttk.Label(self.set_frame_two, image=img0)
                    img1.grid(row=0, column=0, sticky=W, pady=0)
                    if os.path.exists(test_file_path + "/figures/show" + gene + ".png"):
                        os.remove(test_file_path + "/figures/show" + gene + ".png")
                        print("Figures exits")
                    else:
                        print("Figures no exits！")
                    pass
                except:
                    Messagebox.show_error(
                        "Python Error", "Make sure gene name in dataset"
                    )

            self.gene_visualization_btn = ttk.Button(
                self.set_frame_one, width=10, command=gene_visualization, text="Show"
            )
            self.gene_visualization_btn.grid(row=3, column=2, sticky=W, pady=0)

            global image_list
            figures = os.listdir(fig_path)
            image_list = []
            s = 0
            for i in range(len(figures)):
                image_list.append(ttk.PhotoImage(file=fig_path + "/" + figures[i]))
                self.result_images = ttk.Label(self.show_frame, image=image_list[i])
                self.result_images.grid(row=i // 4, column=i % 4, sticky=W, pady=0)
                s = i

            def VIEW_3D():
                import webbrowser

                self.web_Server_Thread("/DLPFC/webcache", 8029)
                webbrowser.open("http://127.0.0.1:8029/")

            self.Reset = ttk.Button(
                self.set_frame_one, text="3D VIEW", command=VIEW_3D, width=10
            )
            self.Reset.grid(row=0, column=1, sticky=W, pady=0, padx=(0, 100))
            self.result_queue.put(adata)
        else:
            Messagebox.show_warning(
                title="Attention", message="Waiting for models training!!!"
            )

    def SEDR_Data_Analysis(self):
        try:
            self.model_train_flag = False
            self.cluster_flag = False
            for i in range(self.result_queue.qsize() - 1):
                adata_remove = self.result_queue.get()
                del adata_remove
            adata = self.result_queue.get()
            adata.obs["total_exp"] = adata.X.sum(axis=1)
            adata.var_names_make_unique()
            if self.data_type is None:
                self.data_type = "10x"
            if self.data_type == "10x":
                adata_X = adata_preprocess(
                    adata, min_cells=5, pca_n_comps=params.cell_feat_dim
                )
            else:
                from sklearn.decomposition import PCA

                params.k = 5
                adata_X = PCA(n_components=200, random_state=42).fit_transform(adata.X)
                adata.obsm["X_pca"] = adata_X
            graph_dict = graph_construction(
                adata.obsm["spatial"], adata.shape[0], params
            )
            params.cell_num = adata.shape[0]
            print("==== Graph Construction Finished")

            # ################## Model training
            sedr_net = SEDR_Train(adata_X, graph_dict, params)
            sedr_net.epochs = int(self.train_epoch_num)
            if params.using_dec:
                sedr_net.train_with_dec()
            else:
                sedr_net.train_without_dec()
            sedr_feat, _, _, _ = sedr_net.process()

            adata_sedr = anndata.AnnData(sedr_feat)
            adata_sedr.obsm["spatial"] = adata.obsm["spatial"]

            adata_sedr.obs_names = adata.obs_names
            sc.pp.neighbors(adata_sedr, n_neighbors=params.eval_graph_n)
            sc.tl.umap(adata_sedr)

            # self.result_queue.put(adata_sedr)
            adata_sedr.write_h5ad(
                os.path.join(running_path, "result", "SEDR_output.h5ad")
            )
            self.pb.stop()
            self.pb["value"] = 100
            self.setvar("prog-message", "SEDR run over!")
            self.setvar("End-time", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            EndTime = datetime.now().replace(microsecond=0)
            self.setvar("total-time-cost", EndTime - self.StartTime)
            self.model_train_flag = True
            self.cluster_flag = True
        except:
            self.model_train_flag = False
            raise "Check whether the files exists!"

    def SEDR_Thread(self):
        T = threading.Thread(target=self.SEDR_Data_Analysis)
        # T.setDaemon(True)
        T.start()

    def SCANPY_show(self):
        if self.cluster_flag:
            self.result_flag = True
            if self.data_type is None:
                self.data_type = "10x"
            plt.rcParams["font.sans-serif"] = "Arial"
            self.gene_color_type = "viridis"

            def draw_images():
                i = 1
                os.chdir(Raw_PATH)
                from matplotlib import pyplot as plt

                plt.rcParams["figure.figsize"] = (3, 3)
                path = (
                    test_file_path
                    + "/figures/"
                    + self.method_flag
                    + "_"
                    + self.data_type
                )
                if self.label_files_exit:
                    sc.pl.spatial(
                        adata,
                        img_key="hires",
                        color="GroundTruth",
                        title="Ground Truth",
                        show=False,
                        save="SCANPY_" + str(i) + ".png",
                    )
                    shutil.move(
                        test_file_path + "/figures/showSCANPY_" + str(i) + ".png",
                        path + "/SCANPY_" + str(i) + ".png",
                    )

                    i = i + 1
                    plt.rcParams["figure.figsize"] = (3, 3)
                    sc.pl.umap(
                        adata,
                        color="SCANPY_leiden",
                        title="SCANPY-leiden-umap",
                        show=False,
                        s=6,
                        save="SCANPY_" + str(i) + ".png",
                    )
                    shutil.move(
                        test_file_path + "/figures/umapSCANPY_" + str(i) + ".png",
                        path + "/SCANPY_" + str(i) + ".png",
                    )

                    i = i + 1
                    sc.pl.spatial(
                        adata,
                        color="SCANPY_leiden",
                        title="SCANPY-leiden",
                        show=False,
                        save="SCANPY_" + str(i) + ".png",
                    )
                    shutil.move(
                        test_file_path + "/figures/showSCANPY_" + str(i) + ".png",
                        path + "/SCANPY_" + str(i) + ".png",
                    )

                    i = i + 1
                    plt.rcParams["figure.figsize"] = (3, 3)
                    sc.pl.umap(
                        adata,
                        color="SCANPY_kmeans",
                        title="SCANPY-kmeans-umap",
                        show=False,
                        s=6,
                        save="SCANPY_" + str(i) + ".png",
                    )
                    shutil.move(
                        test_file_path + "/figures/umapSCANPY_" + str(i) + ".png",
                        path + "/SCANPY_" + str(i) + ".png",
                    )

                    i = i + 1
                    sc.pl.spatial(
                        adata,
                        color="SCANPY_kmeans",
                        title="SCANPY",
                        show=False,
                        save="SCANPY_" + str(i) + ".png",
                    )
                    shutil.move(
                        test_file_path + "/figures/showSCANPY_" + str(i) + ".png",
                        path + "/SCANPY_" + str(i) + ".png",
                    )

                    adata.obs.GroundTruth = adata.obs.GroundTruth.astype(str)
                    i = i + 1
                    sc.tl.paga(adata, groups="GroundTruth")
                    plt.rcParams["figure.figsize"] = (4, 3)
                    sc.pl.paga(
                        adata,
                        color="GroundTruth",
                        title="PAGA",
                        show=False,
                        save="SCANPY_" + str(i) + ".png",
                    )
                    shutil.move(
                        test_file_path + "/figures/pagaSCANPY_" + str(i) + ".png",
                        path + "/SCANPY_" + str(i) + ".png",
                    )

                    i = i + 1
                    if int(self.cluster_value) > len(adata.uns["SCANPY_leiden_colors"]):
                        Messagebox.show_error(
                            "Python Error", "Make sure gene name in dataset"
                        )
                    else:
                        sc.pl.rank_genes_groups_heatmap(
                            adata,
                            groups=str(self.cluster_value),
                            groupby="SCANPY_leiden",
                            show=False,
                            save="SCANPY_" + str(i) + ".png",
                        )
                        shutil.move(
                            test_file_path
                            + "/figures/heatmapSCANPY_"
                            + str(i)
                            + ".png",
                            path + "/SCANPY_" + str(i) + ".png",
                        )
                else:
                    sc.pp.calculate_qc_metrics(adata, inplace=True)
                    sc.pl.spatial(
                        adata,
                        img_key="hires",
                        color="log1p_total_counts",
                        title="log1p_total_counts",
                        show=False,
                        spot_size=20,
                        color_map=self.gene_color_type,
                        save="SCANPY_" + str(i) + ".png",
                    )
                    shutil.move(
                        test_file_path + "/figures/showSCANPY_" + str(i) + ".png",
                        path + "/SCANPY_" + str(i) + ".png",
                    )

                    i = i + 1
                    plt.rcParams["figure.figsize"] = (3, 3)
                    sc.pl.umap(
                        adata,
                        color="SCANPY_leiden",
                        title="SCANPY-leiden-umap",
                        show=False,
                        s=6,
                        save="SCANPY_" + str(i) + ".png",
                    )
                    shutil.move(
                        test_file_path + "/figures/umapSCANPY_" + str(i) + ".png",
                        path + "/SCANPY_" + str(i) + ".png",
                    )

                    i = i + 1
                    sc.pl.spatial(
                        adata,
                        color="SCANPY_leiden",
                        title="SCANPY-leiden",
                        show=False,
                        save="SCANPY_" + str(i) + ".png",
                    )
                    shutil.move(
                        test_file_path + "/figures/showSCANPY_" + str(i) + ".png",
                        path + "/SCANPY_" + str(i) + ".png",
                    )

                    i = i + 1
                    plt.rcParams["figure.figsize"] = (3, 3)
                    sc.pl.umap(
                        adata,
                        color="SCANPY_kmeans",
                        title="SCANPY-kmeans-umap",
                        show=False,
                        s=6,
                        save="SCANPY_" + str(i) + ".png",
                    )
                    shutil.move(
                        test_file_path + "/figures/umapSCANPY_" + str(i) + ".png",
                        path + "/SCANPY_" + str(i) + ".png",
                    )

                    i = i + 1
                    sc.pl.spatial(
                        adata,
                        color="SCANPY_kmeans",
                        title="SCANPY-kmeans",
                        show=False,
                        save="SCANPY_" + str(i) + ".png",
                    )
                    shutil.move(
                        test_file_path + "/figures/showSCANPY_" + str(i) + ".png",
                        path + "/SCANPY_" + str(i) + ".png",
                    )

                    i = i + 1
                    if int(self.cluster_value) > len(adata.uns["SCANPY_leiden_colors"]):
                        Messagebox.show_error(
                            "Python Error", "Make sure gene name in dataset"
                        )
                    else:
                        sc.pl.rank_genes_groups_heatmap(
                            adata,
                            groups=str(self.cluster_value),
                            groupby="SCANPY_leiden",
                            show=False,
                            save="SCANPY_" + str(i) + ".png",
                        )
                        shutil.move(
                            test_file_path
                            + "/figures/heatmapSCANPY_"
                            + str(i)
                            + ".png",
                            path + "/SCANPY_" + str(i) + ".png",
                        )
                print(f"Result images are saved in : {test_file_path}/figures")

            def scoller():
                self.figure_ybar = ttk.Scrollbar(
                    self.figure_Frame, orient=VERTICAL, cursor="draft_small"
                )
                self.figure_ybar.pack(side=RIGHT, fill=Y)
                self.figure_ybar.config(command=self.canvas.yview)

                self.figure_xbar = ttk.Scrollbar(
                    self.figure_Frame, orient=HORIZONTAL, cursor="draft_small"
                )
                self.figure_xbar.pack(side=BOTTOM, fill=X)
                self.figure_xbar.config(command=self.canvas.xview)

                self.canvas.configure(yscrollcommand=self.figure_xbar.set)
                self.canvas.config(yscrollincrement=1)

                self.canvas.configure(xscrollcommand=self.figure_xbar.set)
                self.canvas.config(xscrollincrement=1)
                self.canvas.pack(side=LEFT, expand=YES, fill=BOTH, padx=0)  # anchor=NW
                self.canvas.create_window((0, 0), window=self.show_frame, anchor=N)
                self.show_frame.bind("<Configure>", onFrameconfigure)

            def onFrameconfigure(event):
                self.canvas.configure(
                    scrollregion=self.canvas.bbox("all"), width=1000, height=650
                )

            def choose_color():
                colorvalue = colorchooser.askcolor()
                color_list = []
                color = colorvalue[1]
                print(color)
                color_lens = max(
                    len(adata.uns["SCANPY_leiden_colors"]),
                    len(adata.uns["SCANPY_kmeans_colors"]),
                )
                if color[1:3] == "ff":
                    color_list.append(color)
                    list = random.sample(color_panel.FF_color, color_lens)
                    color_list = color_list + list
                elif color[1:3] == "cc":
                    color_list.append(color)
                    list = random.sample(color_panel.CC_color, color_lens)
                    color_list = color_list + list
                elif color[1:3] == "99":
                    color_list.append(color)
                    list = random.sample(color_panel.NN_color, color_lens)
                    color_list = color_list + list
                elif color[1:3] == "66":
                    color_list.append(color)
                    list = random.sample(color_panel.SS_color, color_lens)
                    color_list = color_list + list
                elif color[1:3] == "33":
                    color_list.append(color)
                    list = random.sample(color_panel.TT_color, color_lens)
                    color_list = color_list + list
                elif color[1:3] == "00":
                    color_list.append(color)
                    list = random.sample(color_panel.ZZ_color, color_lens)
                    color_list = color_list + list
                else:
                    color_list.append(color)
                    list = random.sample(color_panel.random_color, color_lens)
                    color_list = color_list + list

                print(color_list)
                if self.label_files_exit:
                    adata.uns["GroundTruth_colors"] = color_list[
                        : len(adata.uns["GroundTruth_colors"])
                    ]
                adata.uns["SCANPY_leiden_colors"] = color_list[
                    : len(adata.uns["SCANPY_leiden_colors"])
                ]
                adata.uns["SCANPY_kmeans_colors"] = color_list[
                    : len(adata.uns["SCANPY_kmeans_colors"])
                ]
                self.color_reset = color_list
                self.gene_color_type = color_panel.five_gene_color[random.randint(0, 5)]
                os.chdir(Raw_PATH)
                draw_images()

                fig_path = (
                    test_file_path
                    + "/figures/"
                    + self.method_flag
                    + "_"
                    + self.data_type
                )
                figures = os.listdir(fig_path)
                global image_list
                image_list = []
                for i in range(len(figures)):
                    image_list.append(ttk.PhotoImage(file=fig_path + "/" + figures[i]))
                    self.result_images = ttk.Label(self.show_frame, image=image_list[i])
                    self.result_images.grid(row=i // 3, column=i % 3, sticky=W, pady=0)

            fig_path = (
                test_file_path + "/figures/" + self.method_flag + "_" + self.data_type
            )
            if not os.path.isdir(fig_path):
                os.mkdir(fig_path)

            adata = self.result_queue.get()
            # if self.label_files_exit:
            #     adata = adata[adata.obs['GroundTruth'] == adata.obs['GroundTruth'],]
            # else:
            #     adata.obsm["spatial"] = adata.obsm["spatial"] * (-1)
            #
            # eval_cluster = int(self.Mcluster_num)
            # eval_resolution = res_search_fixed_clus(adata, eval_cluster)
            # sc.tl.leiden(adata, key_added="SCANPY_leiden", resolution=eval_resolution)
            #
            # kmeans = KMeans(n_clusters=self.kmeans_num)
            # kmeans.fit(adata.X)
            # adata.obs['SCANPY_kmeans'] = kmeans.labels_
            # adata.obs['SCANPY_kmeans'] = adata.obs['SCANPY_kmeans'].astype('int')
            # adata.obs['SCANPY_kmeans'] = adata.obs['SCANPY_kmeans'].astype('category')
            # sc.tl.rank_genes_groups(adata, "leiden", inplace=True)

            draw_images()
            self.figure_Frame = ttk.Frame(self.right_panel)
            self.figure_Frame.pack(side=TOP, fill=X, expand=YES)

            self.canvas = ttk.Canvas(self.figure_Frame)
            self.show_frame = ttk.Frame(self.canvas)
            scoller()

            self.set_frame = ttk.Frame(
                self.figure_Frame, borderwidth=2, relief="sunken"
            )
            self.set_frame.pack(side="right", expand=YES, anchor=N)
            self.set_frame_one = ttk.Frame(self.set_frame)
            self.set_frame_one.pack(side=TOP, expand=YES)
            self.set_frame_two = ttk.Frame(self.set_frame)
            self.set_frame_two.pack(side=TOP, expand=YES)

            self.Reset = ttk.Button(
                self.set_frame_one,
                text="Reset all colors",
                command=choose_color,
                width=12,
            )
            self.Reset.grid(row=0, column=0, sticky=W, pady=0, padx=0)

            self.domain_color_label = ttk.Label(
                self.set_frame_one, text="Reset domain color: ", width=20
            )
            self.domain_color_label.grid(row=1, column=0, sticky=W, pady=0)

            self.Reset_domain_color = ttk.Entry(self.set_frame_one, width=10)
            self.Reset_domain_color.grid(row=1, column=1, sticky=W, pady=0)
            Tooltip(
                self.Reset_domain_color,
                "Input value must be int type and in [1:cluster_name]: 2",
            )
            self.color_update = None

            def Reset_single_domain_color():
                from tkinter import colorchooser

                colorvalue = colorchooser.askcolor()
                color = colorvalue[1]
                print(color)
                cluster = self.Reset_domain_color.get()
                self.color_reset[int(cluster) - 1] = color
                adata.uns["SCANPY_leiden_colors"] = self.color_reset[
                    : len(adata.uns["SCANPY_leiden_colors"])
                ]
                adata.uns["SCANPY_kmeans_colors"] = self.color_reset[
                    : len(adata.uns["SCANPY_kmeans_colors"])
                ]
                draw_images()
                figures = os.listdir(fig_path)
                global image_list
                image_list = []
                for i in range(len(figures)):
                    img = Image.open(fig_path + "/" + figures[i])
                    print(img.size)
                    image_list.append(ttk.PhotoImage(file=fig_path + "/" + figures[i]))
                    self.result_images = ttk.Label(self.show_frame, image=image_list[i])
                    self.result_images.grid(
                        row=i // 3, column=i % 3, sticky=W, pady=0, padx=0
                    )
                    s = i

            self.Reset_domain_btn = ttk.Button(
                self.set_frame_one,
                width=10,
                text="Confirm",
                command=Reset_single_domain_color,
            )
            self.Reset_domain_btn.grid(row=1, column=2, sticky=W, pady=10)
            self.update_image_label = ttk.Label(
                self.set_frame_one, text="Reset image dpi: ", width=20
            )
            self.update_image_label.grid(row=2, column=0, sticky=W, pady=10)
            self.update_image_scale = ttk.Entry(self.set_frame_one, width=10)
            self.update_image_scale.grid(row=2, column=1, sticky=W, pady=10)
            Tooltip(
                self.update_image_scale, "Input value must be int type and >= 300: 300"
            )

            def ipdate_hd():
                dpi = self.update_image_scale.get()
                print(dpi)
                import matplotlib.pyplot as plt

                plt.rcParams["figure.figsize"] = (3, 3)
                sc.set_figure_params(dpi=dpi)
                draw_images()

            self.Reset_domain_btn = ttk.Button(
                self.set_frame_one, width=10, text="Save", command=ipdate_hd
            )
            self.Reset_domain_btn.grid(row=2, column=2, sticky=W, pady=0)

            self.gene_visualization_label = ttk.Label(
                self.set_frame_one, text="Input gene name: ", width=20
            )
            self.gene_visualization_label.grid(row=3, column=0, sticky=W, pady=0)

            self.gene_visualization_entry = ttk.Entry(self.set_frame_one, width=10)
            self.gene_visualization_entry.grid(row=3, column=1, sticky=W, pady=0)
            Tooltip(self.gene_visualization_entry, "Gene name: NEFH/ATP2B4")

            def gene_visualization():
                gene = self.gene_visualization_entry.get()
                if gene not in adata.var_names:
                    print(f"{gene} is mot in adata!!Please input right gene name!")
                sc.pl.spatial(
                    adata,
                    img_key="hires",
                    color=gene,
                    title="$" + gene + "$",
                    sopt_size=self.spot_size,
                    show=False,
                    save=gene + ".png",
                )
                global img0
                photo = Image.open(test_file_path + "/figures/show" + gene + ".png")
                img0 = ImageTk.PhotoImage(photo)
                img1 = ttk.Label(self.set_frame_two, image=img0)
                img1.grid(row=0, column=0, sticky=W, pady=0)
                if os.path.exists(test_file_path + "/figures/show" + gene + ".png"):
                    os.remove(test_file_path + "/figures/show" + gene + ".png")
                    print("yes")
                else:
                    print("error！")

            self.gene_visualization_btn = ttk.Button(
                self.set_frame_one, width=10, command=gene_visualization, text="Show"
            )
            self.gene_visualization_btn.grid(row=3, column=2, sticky=W, pady=0)
            figures = os.listdir(fig_path)
            global image_list
            image_list = []
            for i in range(len(figures)):
                image_list.append(ttk.PhotoImage(file=fig_path + "/" + figures[i]))
                self.result_images = ttk.Label(self.show_frame, image=image_list[i])
                self.result_images.grid(row=i // 3, column=i % 3, sticky=W, pady=0)
                s = i

            self.Entry = ttk.Entry(self.set_frame_one, width=10)
            self.Entry.grid(row=0, column=1, sticky=W, pady=0)
            Tooltip(self.Entry, "Local port number: 8050")

            def VIEW_3D():
                import webbrowser

                current_file_path = os.path.abspath(__file__)
                test_file_path = os.path.dirname(current_file_path)
                path = os.path.dirname(test_file_path)
                data_save_path = running_path + "/module_3D_data/DLPFC"
                self.web_Server_Thread(
                    os.path.join(data_save_path, "webcache"), int(self.Entry.get())
                )
                http = "http://127.0.0.1:" + self.Entry.get() + "/"
                # 'http://127.0.0.1:8050/'
                webbrowser.open(http)
                os.chdir(Raw_PATH)

            self.Reset = ttk.Button(
                self.set_frame_one, text="3D VIEW", command=VIEW_3D, width=10
            )
            self.Reset.grid(row=0, column=2, sticky=W, pady=0)
            self.result_queue.put(adata)
        else:
            Messagebox.show_warning(
                title="Attention", message="Waiting for models training!!!"
            )

    def SCANPY_Data_Analysis(self):
        self.model_train_flag = False
        self.cluster_flag = False
        for i in range(self.result_queue.qsize() - 1):
            adata_remove = self.result_queue.get()
            del adata_remove
        adata = self.result_queue.get()
        if "highly_variable" in adata.var.columns:
            adata = adata[:, adata.var["highly_variable"]]
        else:
            sc.pp.highly_variable_genes(
                adata, flavor="seurat", n_top_genes=3000, inplace=True
            )
            adata = adata[:, adata.var["highly_variable"]]
        if "GroundTruth" in adata.obs.columns:
            self.label_files_exit = True

        sc.pp.pca(
            adata,
            n_comps=int(self.rad_cutoff_value),
            use_highly_variable=True,
            svd_solver="arpack",
        )
        sc.pp.neighbors(adata, use_rep="X_pca")
        sc.tl.umap(adata)

        # self.result_queue.put(adata)
        adata.write_h5ad(os.path.join(running_path, "result", "SCANPY_output.h5ad"))
        self.pb.stop()
        self.pb["value"] = 100
        self.setvar("prog-message", "SCANPY run over!")
        self.setvar("End-time", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        EndTime = datetime.now().replace(microsecond=0)
        self.setvar("total-time-cost", EndTime - self.StartTime)
        self.model_train_flag = True
        self.cluster_flag = True

    def SCANPY_Thread(self):
        T = threading.Thread(target=self.SCANPY_Data_Analysis)
        T.setDaemon(True)
        T.start()

    def Data_process(self):
        self.StartTime = datetime.now().replace(microsecond=0)
        self.setvar("prog-time-started", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        self.setvar("current-file-msg", self.method_flag)
        if self.method_flag == "STAGATE":
            self._value = (
                "Log: " + self.method_flag + " is running ,"
                "the visualization results are displayed below......"
            )
            self.setvar("scroll-message", self._value)
            self.pb.start()
            self.setvar("prog-message", "STAGATE is running!!")
            self.label_files_exit = False
            self.STAGATE_Thread()
        elif self.method_flag == "STAligner":
            self._value = (
                "Log: "
                + self.method_flag
                + "is running, the visualization results are displayed "
                "below...... "
            )
            self.setvar("scroll-message", self._value)
            self.pb.start()
            self.setvar("prog-message", "STAligner is running!!")
            self.label_files_exit = False
            self.STAligner_Thread()
        elif self.method_flag == "STAMarker":
            self._value = (
                "Log: "
                + self.method_flag
                + "is running, the visualization results are displayed "
                "below...... "
            )
            self.setvar("scroll-message", self._value)
            self.pb.start()
            self.setvar("prog-message", "STAMarker is running!!")
            self.STAMarker_Thread_two()
        elif self.method_flag == "STAGE":
            self._value = (
                "Log: " + self.method_flag + " is running ,"
                "the visualization results are displayed below......"
            )
            self.setvar("scroll-message", self._value)
            if self.data_type not in ["10x", "ST", "Slide-seq"]:
                self.data_type = "10x"
            self.pb.start()
            self.setvar("prog-message", "STAGE is running!!")
            self.STAGE_Thread()
        elif self.method_flag == "SpaGCN":
            self.label_files_exit = False
            self._value = (
                "Log: " + self.method_flag + " is running ,"
                "the visualization results are displayed below......"
            )
            self.setvar("scroll-message", self._value)
            self.pb.start()
            self.setvar("prog-message", "SpaGCN is running!!")
            self.SpaGCN_Thread()

        elif self.method_flag == "SEDR":
            self.label_files_exit = False
            self._value = (
                "Log: " + self.method_flag + " is running ,"
                "the visualization results are displayed below......"
            )
            self.setvar("scroll-message", self._value)
            self.pb.start()
            self.setvar("prog-message", "SEDR is running!!")
            self.SEDR_Thread()
        elif self.method_flag == "SCANPY":
            self.label_files_exit = False
            self._value = (
                "Log: " + self.method_flag + " is running ,"
                "the visualization results are displayed below......"
            )
            self.setvar("scroll-message", self._value)
            self.pb.start()
            self.setvar("prog-message", "SCANPY is running!!")
            self.SCANPY_Thread()
        else:
            Messagebox.show_error(
                "Other methods should be used by STABox packages!", "Warning!"
            )

    def settings(self):
        settingbox = ttk.Toplevel(title="settings")
        settingbox.geometry("300x100+100+100")
        settingbox.lift()
        ttk.Label(settingbox, text="R_HOME").grid(row=0, sticky="w")
        ttk.Label(settingbox, text="R_USER").grid(row=1, sticky="w")
        R_HOME_box = ttk.Entry(
            settingbox, textvariable="R_HOME", validate="focusout", width=30
        )
        R_HOME_box.grid(row=0, column=1)
        R_USER_box = ttk.Entry(
            settingbox, textvariable="R_USER", validate="focusout", width=30
        )
        R_USER_box.grid(row=1, column=1)

        def load_setting():
            path = "../Renv_setting.yaml"
            if os.path.exists(path):
                with open(path, "r") as f:
                    yamlData = yaml.safe_load(f)
                os.environ["R_HOME"] = yamlData["R_HOME"]
                os.environ["R_USER"] = yamlData["R_USER"]
                self.setvar("R_HOME", yamlData["R_HOME"])
                self.setvar("R_USER", yamlData["R_USER"])
                settingbox.destroy()
            else:
                Messagebox.show_error("Make sure yaml exist!", "Error!")
                settingbox.destroy()

        load_setting_button = ttk.Button(
            settingbox, text="Load yaml", width=9, command=load_setting
        )
        load_setting_button.grid(row=2, column=0)

        def get_data():
            os.environ["R_HOME"] = R_HOME_box.get()
            os.environ["R_USER"] = R_USER_box.get()
            setting = {"R_HOME": R_HOME_box.get(), "R_USER": R_USER_box.get()}
            with open("../Renv_setting.yaml", "w") as f:
                yaml.dump(setting, f)
            settingbox.destroy()

        check_button = ttk.Button(settingbox, text="Confirm", width=8, command=get_data)
        check_button.grid(row=2, column=1)


class CollapsingFrame(ttk.Frame):
    """A collapsible frame widget that opens and closes with a click."""

    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.columnconfigure(0, weight=1)
        self.cumulative_rows = 0

        # widget images
        self.images = [
            ttk.PhotoImage(file=PATH / "icons8_double_up_24px.png"),
            ttk.PhotoImage(file=PATH / "icons8_double_right_24px.png"),
        ]

    def add(self, child, title="", bootstyle=PRIMARY, **kwargs):
        """Add a child to the collapsible frame

        Parameters:

            child (Frame):
                The child frame to add to the widget.

            title (str):
                The title appearing on the collapsible section header.

            bootstyle (str):
                The style to apply to the collapsible section header.

            **kwargs (Dict):
                Other optional keyword arguments.
        """
        if child.winfo_class() != "TFrame":
            return

        style_color = Bootstyle.ttkstyle_widget_color(bootstyle)
        frm = ttk.Frame(self, bootstyle=style_color)
        frm.grid(row=self.cumulative_rows, column=0, sticky=EW)

        # header title
        header = ttk.Label(master=frm, text=title, bootstyle=(style_color, INVERSE))
        if kwargs.get("textvariable"):
            header.configure(textvariable=kwargs.get("textvariable"))
        header.pack(side=LEFT, fill=BOTH, padx=10)

        # header toggle button
        def _func(c=child):
            return self._toggle_open_close(c)

        btn = ttk.Button(
            master=frm,
            image=self.images[0],
            bootstyle=style_color,
            # 注意这边
            command=_func,
        )
        btn.pack(side=RIGHT)

        # assign toggle button to child so that it can be toggled
        child.btn = btn
        child.grid(row=self.cumulative_rows + 1, column=0, sticky=NSEW)

        # increment the row assignment
        self.cumulative_rows += 2

    def _toggle_open_close(self, child):
        """Open or close the section and change the toggle button
        image accordingly.

        Parameters:

            child (Frame):
                The child element to add or remove from grid manager.
        """
        if child.winfo_viewable():
            child.grid_remove()
            child.btn.configure(image=self.images[1])
        else:
            child.grid()
            child.btn.configure(image=self.images[0])


class PrintRedirector:
    def __init__(self, text_widget, max_lines):
        self.text_widget = text_widget
        self.max_lines = max_lines
        self.current_lines = 0

    def write(self, message):
        lines = message.strip().split("\n")
        for line in lines:
            if self.current_lines >= self.max_lines:
                break
            self.text_widget.insert(tk.END, line + "\n")
            self.current_lines += 1
        self.text_widget.see(tk.END)

    def flush(self):
        pass
