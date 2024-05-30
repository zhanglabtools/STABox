import random
import re
import subprocess
import warnings
import sys
import matplotlib
import tkinter as tk
import cv2
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from scipy.sparse import issparse, csr_matrix
import anndata
import yaml
sys.path.append("..")
plt.rc('font', family='Arial')

from upsetplot import plot, from_contents
import gseapy as gp
from gseapy import barplot, dotplot

matplotlib.use('Agg')

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
from ..pl.utils import Cal_Spatial_Net, Stats_Spatial_Net, Cal_Spatial_Net_3D, mclust_R, Cal_Spatial_Net_new, parse_args, select_svgs
from ..model import STAligner
from ..model import STAMarker
from ..extension.STAGE import STAGE
from ..module_3D.DLPFC_Data import webcache_main, webServer
from ..extension import SpaGCN as spg
from ..extension.SEDR.graph_func import graph_construction
from ..extension.SEDR.utils_func import adata_preprocess
from ..extension.SEDR.SEDR_train import SEDR_Train
from ..extension.SEDR.SEDR_parameter import params, res_search_fixed_clus
import json

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
    'properties-dark': 'icons8_settings_24px.png',
    'properties-light': 'icons8_settings_24px_2.png',
    'add-to-backup-dark': 'icons8_add_folder_24px.png',
    'add-to-backup-light': 'icons8_add_book_24px.png',
    'stop-backup-dark': 'icons8_cancel_24px.png',
    'stop-backup-light': 'icons8_cancel_24px_1.png',
    'play': 'icons8_play_24px_1.png',
    'refresh': 'icons8_refresh_24px_1.png',
    'stop-dark': 'icons8_stop_24px.png',
    'stop-light': 'icons8_stop_24px_1.png',
    'opened-folder': 'icons8_opened_folder_24px.png',
    'logo': 'backup.png'
}
methods = ['STAGATE', 'STAligner', 'STAMarker', 'STAGE', 'SpaGCN', 'SEDR', 'SCANPY']
data_type = ['10x', 'Slide-seqV2', 'Stereo-seq', 'MERFISH', 'Slide-seq', 'ST', 'STARmap', 'HDST', 'H5AD-files',
             'Multi-files']

PATH = Path(__file__).parent / 'assets'
Raw_PATH = Path(__file__).parent

current_file_path = os.path.abspath(__file__)
test_file_path = os.path.dirname(current_file_path)
running_path = os.path.dirname(test_file_path)
print(f'running_path={running_path}')
print(f'current_file_path={current_file_path}')
print(f'test_file_path={test_file_path}')
print(os.path.join(running_path, 'module_3D_data\\DLPFC'))

import tkinter
import tkinter.ttk as ttks
from tkinter import messagebox


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
        self.R_HOME = None
        self.R_USER = None
        self.color_reset = None
        self.gene_color_type = None

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

        self.data_type = None
        self.upload_file_path = None
        self.label_files_exit = False

        self.result_queue = queue.Queue()
        self.method = None

        self.photoimages = []
        # imgpath = 'VIEW/assets'
        imgpath = test_file_path + '/assets'
        for key, val in image_files.items():
            _path = imgpath + '/' + val
            self.photoimages.append(ttk.PhotoImage(name=key, file=_path))

        buttonbar = ttk.Frame(self, style='primary.TFrame')
        buttonbar.pack(fill=X, pady=1, side=TOP)

        sty = ttk.Style()
        sty.configure('my.TButton', font="Arial")

        self.backup = ttk.Button(
            master=buttonbar, text='Load',
            image='add-to-backup-light',
            compound=LEFT,
            style='my.TButton',
            command=self.Data_Preprocess_thread  # self.Data_Preprocess_thread
        )
        self.backup.pack(side=LEFT, ipadx=5, ipady=5, padx=(1, 0), pady=1)

        btn = ttk.Button(
            master=buttonbar,
            text='Preprocess',
            image='refresh',
            compound=LEFT,
            style='my.TButton',
            command=self.Preprocess
        )
        btn.pack(side=LEFT, ipadx=5, ipady=5, padx=0, pady=1)

        btn = ttk.Button(
            master=buttonbar,
            text='Run',
            image='play',
            compound=LEFT,
            style='my.TButton',
            command=self.Data_process
        )
        btn.pack(side=LEFT, ipadx=5, ipady=5, padx=0, pady=1)

        btn = ttk.Button(
            master=buttonbar,
            text='Restart',
            image='stop-light',
            compound=LEFT,
            style='my.TButton',
            command=self.Restart
        )
        btn.pack(side=LEFT, ipadx=5, ipady=5, padx=0, pady=1)

        btn = ttk.Button(
            master=buttonbar,
            text='Settings',
            image='properties-light',
            compound=LEFT,
            style='my.TButton',
            command=self.settings
        )
        btn.pack(side=LEFT, ipadx=5, ipady=5, padx=0, pady=1)

        self.left_panel = ttk.Frame(self, style='bg.TFrame')
        self.left_panel.pack(side=LEFT, fill=Y)

        self.bus_cf = CollapsingFrame(self.left_panel)
        self.bus_cf.pack(fill=X, pady=1)

        self.file_path_frm = ttk.Frame(self.bus_cf, padding=10)
        self.file_path_frm.columnconfigure(1, weight=2)
        self.bus_cf.add(
            child=self.file_path_frm,
            font=('Arial', 10),
            title='Load h5ad files',
            bootstyle=SECONDARY
        )

        self.path_load_flag = False
        style = Style()
        style.configure("TCheckbutton", font=("Arial", 16))
        style.configure('TButton', font=('Arial', 16))
        self.radio_frame = ttk.Frame(self.file_path_frm)
        self.radio_frame.pack(side=TOP)

        def on_radio_select(value):
            if self.path_load_flag:
                self.Single_file_moduel.destroy()
                self.path_load_flag = False
            if value == "single-h5ad":

                self.Single_file_moduel = ttk.Frame(self.file_path_frm)
                self.Single_file_moduel.pack(after=self.radio_frame)
                self.file_entry = ttk.Entry(self.Single_file_moduel, textvariable='folder-path', font="Arial")
                self.file_entry.pack(side=LEFT, fill=X, expand=YES)
                self.file_entry.insert(END, 'select your datas here!')

                def set_data_type_choose(event):
                    print('当前选择{}'.format(self.data_updata.get()))
                    self.data_type = self.data_updata.get()

                select_data_type = ttk.StringVar()
                self.data_updata = ttk.Combobox(
                    master=self.Single_file_moduel,
                    textvariable=select_data_type,
                    font=('Arial', 10),
                    values=data_type,
                    height=8,
                    width=6,
                    state='normal',
                    cursor='plus',
                )
                self.data_updata.pack(side=LEFT)
                self.data_updata.current(0)

                self.data_updata.bind('<<ComboboxSelected>>', set_data_type_choose)

                def file_btn_getdirectory():
                    self.path_load_flag = False
                    self.get_directory()

                self.file_btn = ttk.Button(
                    master=self.Single_file_moduel,
                    image='opened-folder',
                    bootstyle=(LINK, SECONDARY),
                    command=file_btn_getdirectory
                )
                self.file_btn.pack(side=RIGHT)
                self.path_load_flag = True
            else:

                self.Single_file_moduel = ttk.Frame(self.file_path_frm)
                self.Single_file_moduel.pack(after=self.radio_frame)

                file_entry = ttk.Entry(self.Single_file_moduel, textvariable='folder-path')
                file_entry.pack(side=LEFT, fill=X, expand=YES)

                def Select_files():
                    self.multi_files = filedialog.askdirectory()
                    file_entry.delete(0, tk.END)
                    file_entry.insert(0, self.multi_files)
                    self.path_load_flag = False
                    self.Select_files()

                file_choose_btn = ttk.Button(master=self.Single_file_moduel, text="Choose", width=6,
                                             command=Select_files)
                file_choose_btn.pack(side=RIGHT)
                self.path_load_flag = True

        radio_var = tk.StringVar()
        radio1 = ttk.Radiobutton(self.radio_frame, text="Analysis single h5ad file", style="TCheckbutton",
                                 variable=radio_var, value="single-h5ad",
                                 command=lambda: on_radio_select("single-h5ad"))
        radio1.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        radio2 = ttk.Radiobutton(self.radio_frame, text="Analysis Multi-h5ad files", style="TCheckbutton",
                                 variable=radio_var, value="multi h5ad", command=lambda: on_radio_select("Multi-h5ad"))
        radio2.grid(row=1, column=0, padx=5, pady=5, sticky="w")

        self.bus_frm = ttk.Frame(self.bus_cf, padding=10)
        self.bus_frm.columnconfigure(1, weight=2)
        self.bus_cf.add(
            child=self.bus_frm,
            font=('Arial', 10),
            title='Select Methods',
            bootstyle=SECONDARY
        )
        self.frames = ttk.Frame(self.bus_frm)
        self.frames.pack(side=TOP, anchor=W)

        def set_value_before_choose():
            print('Methods：', select_text_data.get())
            new_select_data = []
            for i in methods:
                if select_text_data.get() in i:
                    new_select_data.append(i)

            self.select_box_obj["value"] = new_select_data

        select_text_data = ttk.StringVar()
        self.select_box_obj = ttk.Combobox(
            master=self.frames,
            textvariable=select_text_data,
            font=('Arial', 10),
            values=methods,
            height=8,
            width=20,
            state='normal',
            cursor='plus',
            postcommand=set_value_before_choose
        )
        self.select_box_obj.pack(side=LEFT, padx=20, anchor=W)
        self.select_box_obj.current(0)

        def cutoff_num_check():
            try:
                self.rad_cutoff_value = float(self.rad_cutoff.get())
                print(self.rad_cutoff_value)
            except:
                Messagebox.show_info('please input numbers!')

        def alpha_num_check():
            try:
                self.alpha_value = float(self.alpha.get())
                print(self.alpha_value)
            except:
                Messagebox.show_info('please input numbers!')

        def cluster_num_check():
            try:
                self.cluster_value = int(self.cluster.get())
                print(self.cluster_value)
            except:
                Messagebox.show_info('please input numbers!')

        def gene_name_check():
            try:
                self.genes = str(self.gene_name.get())
                print(self.genes)
            except:
                Messagebox.show_info('please input right genes name!')

        def functions_info_check():
            try:
                self.functions_choose = str(self.function_box.get())
                print(self.functions_choose)
            except:
                Messagebox.show_info('please input right genes name!')

        def detail_info_check():
            # try:
            self.detail_info = str(self.detail_info_box.get())
            print(self.detail_info)
            # except:
            #     Messagebox.show_info('please input right genes name!')

        def adjust_num_check():
            try:
                self.adjust_value = float(self.adjust.get())
                print(self.adjust_value)
            except:
                Messagebox.show_info('please input numbers!')

        def referance_adjust():
            try:
                self.p_frame = tk.Frame(master=self.bus_frm)
                self.p_frame.pack(fill=ttk.BOTH, expand=True, side=BOTTOM, anchor=W, padx=20, pady=10)
                self.label = ttk.Label(self.p_frame, text='Parameter choose: ', width=25)
                self.label.pack(side=TOP, anchor=W, pady=10)
                self.para_frame = tk.Frame(master=self.p_frame, highlightbackground="black", highlightthickness=1)
                self.para_frame.pack(side=BOTTOM, anchor=W, pady=10)

                self.btn_states = False
                if self.method_flag == 'STAGATE':
                    self.rad_cutoff_value = int(150)
                    self.alpha_value = float(0.65)
                    self.cluster_value = int(7)
                    self.genes = 'TAS1R3'
                    self.detail_info = "Deciphering spatial domains"
                    self.rad_cutoff_info = ttk.Label(master=self.para_frame, text="rad_cutoff", font="Arial")
                    self.rad_cutoff_info.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")

                    self.rad_cutoff = ttk.Entry(
                        master=self.para_frame,
                        textvariable='rad_cutoff',
                        validate="focusout",
                        validatecommand=cutoff_num_check,
                        width=20
                    )
                    self.rad_cutoff.grid(row=1, column=1, padx=20, pady=10)
                    self.setvar('rad_cutoff', int(150))
                    self.alpha_info = ttk.Label(master=self.para_frame, text="alpha", font="Arial")
                    self.alpha_info.grid(row=2, column=0, padx=20, pady=10, sticky="nsew")
                    self.alpha = ttk.Entry(
                        master=self.para_frame,
                        textvariable='alpha',
                        validate="focusout",
                        validatecommand=alpha_num_check,
                        width=20
                    )
                    self.alpha.grid(row=2, column=1, padx=20, pady=10)
                    self.setvar('alpha', float(0.65))
                    self.cluster_info = ttk.Label(master=self.para_frame, text="n_cluster", font="Arial")
                    self.cluster_info.grid(row=3, column=0, padx=20, pady=10, sticky="nsew")
                    self.cluster = ttk.Entry(
                        master=self.para_frame,
                        textvariable='cluster',
                        validate="focusout",
                        validatecommand=cluster_num_check,
                        width=20
                    )
                    self.cluster.grid(row=3, column=1, padx=20, pady=10)
                    self.setvar('cluster', int(7))
                    self.genes_info = ttk.Label(master=self.para_frame, text="Denoising", font="Arial")
                    self.genes_info.grid(row=4, column=0, padx=20, pady=10, sticky="nsew")
                    self.gene_name = ttk.Entry(
                        master=self.para_frame,
                        textvariable='gene_name',
                        validate="focusout",
                        validatecommand=gene_name_check,
                        width=20
                    )
                    self.gene_name.grid(row=4, column=1, padx=20, pady=10)
                    self.setvar('gene_name', "NEFH")
                    self.detail_infos = ttk.Label(master=self.para_frame, text="Details", font="Arial")
                    self.detail_infos.grid(row=5, column=0, padx=20, pady=10, sticky="nsew")
                    self.detail_info_box = ttk.Entry(
                        master=self.para_frame,
                        textvariable='detail_info',
                        validate="focusout",
                        validatecommand=detail_info_check,
                        width=20
                    )
                    self.detail_info_box.grid(row=5, column=1, padx=20, pady=10)
                    self.setvar('detail_info', "Deciphering spatial domains")
                    Tooltip(self.rad_cutoff, "spot Indicates the neighbor distance: 10X-150")
                    Tooltip(self.alpha, "Algorithm hyperparameter: 0.65/[0-1]")
                    Tooltip(self.cluster, "Number of cluster labels: 7")
                    Tooltip(self.gene_name, "Visual gene name: NEFH/ATP2B4/TAS1R3")
                    Tooltip(self.detail_info_box, "Editable information: STAGATE")

                    method_btn.configure(state=DISABLED)

                    self.information = 'Deep learning'
                    self.functions = 'Deciphering spatial domains'
                    self.Run_time = '3-5 mins'
                    self.Datatime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    style_head = ttk.Style()
                    style_head.configure("Treeview.Heading", font="Arial")
                    self.tv.insert('', END,
                                   values=(self.method_flag,
                                           self.information, self.functions, self.Run_time,
                                           self.Datatime))

                elif self.method_flag == 'STAligner':
                    self.rad_cutoff_value = int(150)
                    self.adjust_value = int(1)
                    self.cluster_value = int(7)
                    self.alpha_value = float(0.65)
                    self.detail_info = "Alignment and integration of spatially transcriptomes"
                    self.rad_cutoff_info = ttk.Label(master=self.para_frame, text="rad_cutoff", font="Arial")
                    self.rad_cutoff_info.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")
                    self.rad_cutoff = ttk.Entry(
                        master=self.para_frame,
                        textvariable='rad_cutoff',
                        validate="focusout",
                        validatecommand=cutoff_num_check,
                        width=20
                    )
                    self.rad_cutoff.grid(row=1, column=1, padx=20, pady=10)
                    self.setvar('rad_cutoff', int(150))

                    self.cluster_info = ttk.Label(master=self.para_frame, text="n_cluster", font="Arial")
                    self.cluster_info.grid(row=2, column=0, padx=20, pady=10, sticky="nsew")
                    self.cluster = ttk.Entry(
                        master=self.para_frame,
                        textvariable='cluster',
                        validate="focusout",
                        validatecommand=cluster_num_check,
                        width=20
                    )
                    self.cluster.grid(row=2, column=1, padx=20, pady=10)
                    self.setvar('cluster', int(7))

                    self.adjust_info = ttk.Label(master=self.para_frame, text="margin", font="Arial")
                    self.adjust_info.grid(row=3, column=0, padx=20, pady=10, sticky="nsew")
                    self.adjust = ttk.Entry(
                        master=self.para_frame,
                        textvariable='margin',
                        validate="focusout",
                        validatecommand=adjust_num_check,
                        width=20
                    )
                    self.adjust.grid(row=3, column=1, padx=20, pady=10)
                    self.setvar('margin', int(1))

                    self.alpha_info = ttk.Label(master=self.para_frame, text="alpha", font="Arial")
                    self.alpha_info.grid(row=4, column=0, padx=20, pady=10, sticky="nsew")
                    self.alpha = ttk.Entry(
                        master=self.para_frame,
                        textvariable='alpha',
                        validate="focusout",
                        validatecommand=alpha_num_check,
                        width=20
                    )
                    self.alpha.grid(row=4, column=1, padx=20, pady=10)
                    self.setvar('alpha', float(0.65))

                    self.detail_infos = ttk.Label(master=self.para_frame, text="Details", font="Arial")
                    self.detail_infos.grid(row=5, column=0, padx=20, pady=10, sticky="nsew")
                    self.detail_info_box = ttk.Entry(
                        master=self.para_frame,
                        textvariable='detail_info',
                        validate="focusout",
                        validatecommand=detail_info_check,
                        width=20
                    )
                    self.detail_info_box.grid(row=5, column=1, padx=20, pady=10)
                    self.setvar('detail_info', "Alignment and integration of spatially transcriptomes")

                    Tooltip(self.rad_cutoff, "spot Indicates the neighbor distance: 10X-150")
                    Tooltip(self.cluster, "Number of cluster labels: 7")
                    Tooltip(self.adjust, "Slice alignment hyperparameter: 1")
                    Tooltip(self.alpha, "Louvain resolution: 0.65")
                    Tooltip(self.detail_info_box, "Editable information: STAligner")

                    method_btn.configure(state=DISABLED)

                    self.information = 'Deep learning'
                    self.functions = 'Alignment and integration of spatially transcriptomes'
                    self.Run_time = '5-10 mins'
                    self.Authods = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    self.tv.insert('', END,
                                   values=(self.method_flag,
                                           self.information, self.functions, self.Run_time,
                                           self.Authods))

                elif self.method_flag == 'STAMarker':
                    self.rad_cutoff_value = int(150)
                    self.alpha_value = int(0)
                    self.cluster_value = int(7)
                    self.genes = 'NEFH'
                    self.detail_info = "Deciphering spatial domains SVGs"
                    self.rad_cutoff_info = ttk.Label(master=self.para_frame, text="rad_cutoff", font="Arial")
                    self.rad_cutoff_info.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")

                    self.rad_cutoff = ttk.Entry(
                        master=self.para_frame,
                        textvariable='rad_cutoff',
                        validate="focusout",
                        validatecommand=cutoff_num_check,
                        width=20
                    )
                    self.rad_cutoff.grid(row=1, column=1, padx=20, pady=10)
                    self.setvar('rad_cutoff', int(150))

                    self.alpha_info = ttk.Label(master=self.para_frame, text="alpha", font="Arial")
                    self.alpha_info.grid(row=2, column=0, padx=20, pady=10, sticky="nsew")
                    self.alpha = ttk.Entry(
                        master=self.para_frame,
                        textvariable='alpha',
                        validate="focusout",
                        validatecommand=alpha_num_check,
                        width=20
                    )
                    self.alpha.grid(row=2, column=1, padx=20, pady=10)
                    self.setvar('alpha', int(0))

                    self.cluster_info = ttk.Label(master=self.para_frame, text="cluster_num", font="Arial")
                    self.cluster_info.grid(row=3, column=0, padx=20, pady=10, sticky="nsew")

                    self.cluster = ttk.Entry(
                        master=self.para_frame,
                        textvariable='cluster',
                        validate="focusout",
                        validatecommand=cluster_num_check,
                        width=20
                    )
                    self.cluster.grid(row=3, column=1, padx=20, pady=10)
                    self.setvar('cluster', int(7))

                    self.genes_info = ttk.Label(master=self.para_frame, text="SVGs", font="Arial")
                    self.genes_info.grid(row=4, column=0, padx=20, pady=10, sticky="nsew")

                    self.gene_name = ttk.Entry(
                        master=self.para_frame,
                        textvariable='gene_name',
                        validate="focusout",
                        validatecommand=gene_name_check,
                        width=20
                    )
                    self.gene_name.grid(row=4, column=1, padx=20, pady=10)
                    self.setvar('gene_name', "NEFH")

                    self.detail_infos = ttk.Label(master=self.para_frame, text="Details", font="Arial")
                    self.detail_infos.grid(row=5, column=0, padx=20, pady=10, sticky="nsew")

                    self.detail_info_box = ttk.Entry(
                        master=self.para_frame,
                        textvariable='detail_info',
                        validate="focusout",
                        validatecommand=detail_info_check,
                        width=20
                    )
                    self.detail_info_box.grid(row=5, column=1, padx=20, pady=10)
                    self.setvar('detail_info', "Deciphering spatial domains SVGs")

                    Tooltip(self.rad_cutoff, "spot Indicates the neighbor distance: 10X-150")
                    Tooltip(self.alpha, "Algorithm hyperparameter: 0,1,2,3")
                    Tooltip(self.cluster, "Number of cluster labels: 7")
                    Tooltip(self.gene_name, "Gene name: ATP2B4")
                    Tooltip(self.detail_info_box, "Editable information: STAMarker")

                    method_btn.configure(state=DISABLED)

                    self.information = 'Deep learning'
                    self.functions = 'Deciphering spatial domains SVGs'
                    self.Run_time = '10-30 mins'
                    self.Datatime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    style_head = ttk.Style()
                    style_head.configure("Treeview.Heading", font="Arial")
                    self.tv.insert('', END,
                                   values=(self.method_flag,
                                           self.information, self.functions, self.Run_time,
                                           self.Datatime))

                elif self.method_flag == 'STAGE':
                    self.alpha_value = float(0.5)
                    self.genes = 'PCP4'
                    self.functions_choose = "recovery"
                    self.detail_info = "Gene information enhancement and recovery"
                    self.alpha_info = ttk.Label(master=self.para_frame, text="alpha", font="Arial")
                    self.alpha_info.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")

                    self.alpha = ttk.Entry(
                        master=self.para_frame,
                        textvariable='alpha',
                        validate="focusout",
                        validatecommand=alpha_num_check,
                        width=20
                    )
                    self.alpha.grid(row=1, column=1, padx=20, pady=10)
                    self.setvar('alpha', float(0.5))

                    self.genes_info = ttk.Label(master=self.para_frame, text="Gene", font="Arial")
                    self.genes_info.grid(row=2, column=0, padx=20, pady=10, sticky="nsew")

                    self.gene_name = ttk.Entry(
                        master=self.para_frame,
                        textvariable='gene_name',
                        validate="focusout",
                        validatecommand=gene_name_check,
                        width=20
                    )
                    self.gene_name.grid(row=2, column=1, padx=20, pady=10)
                    self.setvar('gene_name', "PCP4")

                    self.function_infos = ttk.Label(master=self.para_frame, text="functions", font="Arial")
                    self.function_infos.grid(row=3, column=0, padx=20, pady=10, sticky="nsew")

                    self.function_box = ttk.Entry(
                        master=self.para_frame,
                        textvariable='function_infos',
                        validate="focusout",
                        validatecommand=functions_info_check,
                        width=20
                    )
                    self.function_box.grid(row=3, column=1, padx=20, pady=10)
                    self.setvar('function_infos', "recovery")

                    self.detail_infos = ttk.Label(master=self.para_frame, text="Details", font="Arial")
                    self.detail_infos.grid(row=4, column=0, padx=20, pady=10, sticky="nsew")

                    self.detail_info_box = ttk.Entry(
                        master=self.para_frame,
                        textvariable='detail_info',
                        validate="focusout",
                        validatecommand=detail_info_check,
                        width=20
                    )
                    self.detail_info_box.grid(row=4, column=1, padx=20, pady=10)
                    self.setvar('detail_info', "Gene information enhancement and recovery")

                    Tooltip(self.alpha, "Algorithm hyperparameter: 0.5/[0-1]")
                    Tooltip(self.gene_name, "Visual gene name: PCP4/FABP4")
                    Tooltip(self.function_box, "Functions choose: generation, recovery, 3d_model")
                    Tooltip(self.detail_info_box, "Editable information: STAGE")

                    method_btn.configure(state=DISABLED)

                    self.information = 'Supervised learning-gene information enhancement'
                    self.functions = 'Gene expression prediction and tissue segmentation'
                    self.Run_time = '10-40 mins'
                    self.Datatime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    style_head = ttk.Style()
                    style_head.configure("Treeview.Heading", font="Arial")
                    self.tv.insert('', END,
                                   values=(self.method_flag,
                                           self.information, self.functions, self.Run_time,
                                           self.Datatime))
                elif self.method_flag == 'SpaGCN':
                    self.rad_cutoff_value = int(49)
                    self.alpha_value = int(1)
                    self.cluster_value = int(7)
                    self.genes = float(0.5)
                    self.detail_info = "Deciphering spatial domains"
                    self.rad_cutoff_info = ttk.Label(master=self.para_frame, text="weight", font="Arial")
                    self.rad_cutoff_info.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")

                    self.rad_cutoff = ttk.Entry(
                        master=self.para_frame,
                        textvariable='rad_cutoff',
                        validate="focusout",
                        validatecommand=cutoff_num_check,
                        width=20
                    )
                    self.rad_cutoff.grid(row=1, column=1, padx=20, pady=10)
                    self.setvar('rad_cutoff', int(49))

                    self.alpha_info = ttk.Label(master=self.para_frame, text="alpha", font="Arial")
                    self.alpha_info.grid(row=2, column=0, padx=20, pady=10, sticky="nsew")
                    self.alpha = ttk.Entry(
                        master=self.para_frame,
                        textvariable='alpha',
                        validate="focusout",
                        validatecommand=alpha_num_check,
                        width=20
                    )
                    self.alpha.grid(row=2, column=1, padx=20, pady=10)
                    self.setvar('alpha', int(1))

                    self.cluster_info = ttk.Label(master=self.para_frame, text="n_cluster", font="Arial")
                    self.cluster_info.grid(row=3, column=0, padx=20, pady=10, sticky="nsew")

                    self.cluster = ttk.Entry(
                        master=self.para_frame,
                        textvariable='cluster',
                        validate="focusout",
                        validatecommand=cluster_num_check,
                        width=20
                    )
                    self.cluster.grid(row=3, column=1, padx=20, pady=10)
                    self.setvar('cluster', int(7))

                    self.genes_info = ttk.Label(master=self.para_frame, text="percentage", font="Arial")
                    self.genes_info.grid(row=4, column=0, padx=20, pady=10, sticky="nsew")
                    self.gene_name = ttk.Entry(
                        master=self.para_frame,
                        textvariable='gene_name',
                        validate="focusout",
                        validatecommand=gene_name_check,
                        width=20
                    )
                    self.gene_name.grid(row=4, column=1, padx=20, pady=10, sticky="nsew")
                    self.setvar('gene_name', float(0.5))

                    self.detail_infos = ttk.Label(master=self.para_frame, text="Details", font="Arial")
                    self.detail_infos.grid(row=5, column=0, padx=20, pady=10, sticky="nsew")
                    self.detail_info_box = ttk.Entry(
                        master=self.para_frame,
                        textvariable='detail_info',
                        validate="focusout",
                        validatecommand=detail_info_check,
                        width=20
                    )
                    self.detail_info_box.grid(row=5, column=1, padx=20, pady=10, sticky="nsew")
                    self.setvar('detail_info', "Deciphering spatial domains")

                    Tooltip(self.rad_cutoff, "Image feature extraction parameters: 49(10X)")
                    Tooltip(self.alpha, "Algorithm hyperparameter: 1")
                    Tooltip(self.cluster, "Number of cluster labels: 7")
                    Tooltip(self.gene_name, "image use rate: 0.5")
                    Tooltip(self.detail_info_box, "Editable information: SpaGCN")

                    method_btn.configure(state=DISABLED)

                    self.information = 'SpaGCN Deep learning'
                    self.functions = 'Deciphering spatial domains'
                    self.Run_time = '3-5 mins'
                    self.Datatime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    style_head = ttk.Style()
                    style_head.configure("Treeview.Heading", font="Arial")
                    self.tv.insert('', END,
                                   values=(self.method_flag,
                                           self.information, self.functions, self.Run_time,
                                           self.Datatime))
                elif self.method_flag == 'SEDR':
                    self.rad_cutoff_value = int(10)
                    self.alpha_value = float(0.1)
                    self.cluster_value = int(7)
                    self.genes = float(0.2)
                    self.detail_info = "Deciphering spatial domains"

                    self.rad_cutoff_info = ttk.Label(master=self.para_frame, text="k_value", font="Arial")
                    self.rad_cutoff_info.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")

                    self.rad_cutoff = ttk.Entry(
                        master=self.para_frame,
                        textvariable='rad_cutoff',
                        validate="focusout",
                        validatecommand=cutoff_num_check,
                        width=20
                    )
                    self.rad_cutoff.grid(row=1, column=1, padx=20, pady=10)
                    self.setvar('rad_cutoff', int(10))

                    self.alpha_info = ttk.Label(master=self.para_frame, text="alpha", font="Arial")
                    self.alpha_info.grid(row=2, column=0, padx=20, pady=10, sticky="nsew")
                    self.alpha = ttk.Entry(
                        master=self.para_frame,
                        textvariable='alpha',
                        validate="focusout",
                        validatecommand=alpha_num_check,
                        width=20
                    )
                    self.alpha.grid(row=2, column=1, padx=20, pady=10)
                    self.setvar('alpha', float(0.1))

                    self.cluster_info = ttk.Label(master=self.para_frame, text="n_cluster", font="Arial")
                    self.cluster_info.grid(row=3, column=0, padx=20, pady=10)
                    self.cluster = ttk.Entry(
                        master=self.para_frame,
                        textvariable='cluster',
                        validate="focusout",
                        validatecommand=cluster_num_check,
                        width=20
                    )
                    self.cluster.grid(row=3, column=1, padx=20, pady=10)
                    self.setvar('cluster', int(7))

                    self.genes_info = ttk.Label(master=self.para_frame, text="p_drop", font="Arial")
                    self.genes_info.grid(row=4, column=0, padx=20, pady=10)
                    self.gene_name = ttk.Entry(
                        master=self.para_frame,
                        textvariable='gene_name',
                        validate="focusout",
                        validatecommand=gene_name_check,
                        width=20
                    )
                    self.gene_name.grid(row=4, column=1, padx=20, pady=10)
                    self.setvar('gene_name', float(0.2))

                    self.detail_infos = ttk.Label(master=self.para_frame, text="Details", font="Arial")
                    self.detail_infos.grid(row=5, column=0, padx=20, pady=10)

                    self.detail_info_box = ttk.Entry(
                        master=self.para_frame,
                        textvariable='detail_info',
                        validate="focusout",
                        validatecommand=detail_info_check,
                        width=20
                    )
                    self.detail_info_box.grid(row=5, column=1, padx=20, pady=10)
                    self.setvar('detail_info', "Deciphering spatial domains")

                    Tooltip(self.rad_cutoff, "parameter k in spatial graph: 10X-10")
                    Tooltip(self.alpha, "Weight of GCN loss: 0.1")
                    Tooltip(self.cluster, "Number of cluster labels: 7")
                    Tooltip(self.gene_name, "Dropout rate.: 0.2")
                    Tooltip(self.detail_info_box, "Editable information: SEDR")

                    method_btn.configure(state=DISABLED)

                    self.information = 'Deep learning'
                    self.functions = 'Deciphering spatial domains'
                    self.Run_time = '3-5 mins'
                    self.Datatime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    style_head = ttk.Style()
                    style_head.configure("Treeview.Heading", font="Arial")
                    self.tv.insert('', END,
                                   values=(self.method_flag,
                                           self.information, self.functions, self.Run_time,
                                           self.Datatime))
                elif self.method_flag == 'SCANPY':
                    self.rad_cutoff_value = int(50)
                    self.alpha_value = float(0.65)
                    self.cluster_value = str(3)
                    self.genes = 'TAS1R3'
                    self.detail_info = "Deciphering spatial domains"
                    self.rad_cutoff_info = ttk.Label(master=self.para_frame, text="PCA-dim", font="Arial")
                    self.rad_cutoff_info.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")

                    self.rad_cutoff = ttk.Entry(
                        master=self.para_frame,
                        textvariable='rad_cutoff',
                        validate="focusout",
                        validatecommand=cutoff_num_check,
                        width=20
                    )
                    self.rad_cutoff.grid(row=1, column=1, padx=20, pady=10)
                    self.setvar('rad_cutoff', int(50))
                    self.alpha_info = ttk.Label(master=self.para_frame, text="Resolution", font="Arial")
                    self.alpha_info.grid(row=2, column=0, padx=20, pady=10, sticky="nsew")
                    self.alpha = ttk.Entry(
                        master=self.para_frame,
                        textvariable='alpha',
                        validate="focusout",
                        validatecommand=alpha_num_check,
                        width=20
                    )
                    self.alpha.grid(row=2, column=1, padx=20, pady=10)
                    self.setvar('alpha', int(0.65))
                    self.cluster_info = ttk.Label(master=self.para_frame, text="Nth-cluster", font="Arial")
                    self.cluster_info.grid(row=3, column=0, padx=20, pady=10, sticky="nsew")
                    self.cluster = ttk.Entry(
                        master=self.para_frame,
                        textvariable='cluster',
                        validate="focusout",
                        validatecommand=cluster_num_check,
                        width=20
                    )
                    self.cluster.grid(row=3, column=1, padx=20, pady=10)
                    self.setvar('cluster', str(3))
                    self.genes_info = ttk.Label(master=self.para_frame, text="Gene", font="Arial")
                    self.genes_info.grid(row=4, column=0, padx=20, pady=10, sticky="nsew")
                    self.gene_name = ttk.Entry(
                        master=self.para_frame,
                        textvariable='gene_name',
                        validate="focusout",
                        validatecommand=gene_name_check,
                        width=20
                    )
                    self.gene_name.grid(row=4, column=1, padx=20, pady=10)
                    self.setvar('gene_name', "NEFH")
                    self.detail_infos = ttk.Label(master=self.para_frame, text="Details", font="Arial")
                    self.detail_infos.grid(row=5, column=0, padx=20, pady=10, sticky="nsew")
                    self.detail_info_box = ttk.Entry(
                        master=self.para_frame,
                        textvariable='detail_info',
                        validate="focusout",
                        validatecommand=detail_info_check,
                        width=20
                    )
                    self.detail_info_box.grid(row=5, column=1, padx=20, pady=10)
                    self.setvar('detail_info', "Analysis of single-cell and spatial transcriptome processes")
                    Tooltip(self.rad_cutoff, "dimensions of pca reduction: 10X-50")
                    Tooltip(self.alpha, "resolution of leiden clustering algorithm: 0.65/[0-1]")
                    Tooltip(self.cluster, "variable gene difference detection cluster: 3")
                    Tooltip(self.gene_name, "visual gene name: NEFH/ATP2B4/TAS1R3")
                    Tooltip(self.detail_info_box, "Editable information: SCANPY")

                    method_btn.configure(state=DISABLED)

                    self.information = 'Statistical data analysis'
                    self.functions = 'Analysis of single-cell and spatial transcriptome processes'
                    self.Run_time = '3-5 mins'
                    self.Datatime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    style_head = ttk.Style()
                    style_head.configure("Treeview.Heading", font="Arial")
                    self.tv.insert('', END,
                                   values=(self.method_flag,
                                           self.information, self.functions, self.Run_time,
                                           self.Datatime))
                else:
                    Messagebox.ok(message='please input right Methods!')
                    self.btn_states = True
            except:
                Messagebox.ok(message='error! try again!')
                self.btn_states = True

        self.btn_states = True
        method_btn = ttk.Button(
            master=self.frames,
            text='Confirm',
            width=6,
            style='my.TButton',
            command=referance_adjust
        )
        method_btn.pack(side=LEFT)
        self.method_flag = methods[0]

        def submit_result(event):
            if self.btn_states == False:
                self.para_frame.destroy()
                self.btn_states = True
                self.p_frame.destroy()
            method_btn.configure(state=ACTIVE)
            self.method_flag = self.select_box_obj.get()
            print('当前选择{}'.format(self.select_box_obj.get()))

        self.select_box_obj.bind('<<ComboboxSelected>>', submit_result)

        status_cf = CollapsingFrame(self.left_panel)
        status_cf.pack(fill=BOTH, pady=1)

        status_frm = ttk.Frame(status_cf, padding=10)
        status_frm.columnconfigure(1, weight=1)
        status_cf.add(
            child=status_frm,
            title='Running Status',
            bootstyle=SECONDARY
        )

        lbl = ttk.Label(
            master=status_frm,
            textvariable='prog-message'
        )
        lbl.configure(font="Arial")
        lbl.grid(row=0, column=0, columnspan=2, sticky=W)
        self.setvar('prog-message', 'choosing methods...')

        self.pb = ttk.Progressbar(
            master=status_frm,
            mode='determinate',
            length=100,
            orient=HORIZONTAL
        )
        self.pb.grid(row=1, column=0, columnspan=2, sticky=EW, pady=(10, 5))
        self.pb["maximum"] = 100
        self.pb["value"] = 0

        lbl = ttk.Label(status_frm, text='Start time:')
        lbl.configure(font="Arial")
        lbl.grid(row=2, column=0, sticky=W, pady=2)

        lbl = ttk.Label(status_frm, textvariable='prog-time-started')
        lbl.grid(row=2, column=1, columnspan=2, sticky=EW, padx=2, pady=5)

        self.setvar('prog-time-started', None)
        lbl = ttk.Label(status_frm, text='End time:')
        lbl.configure(font="Arial")
        lbl.grid(row=3, column=0, sticky=W, pady=2)
        lbl = ttk.Label(status_frm, textvariable='End-time')
        lbl.grid(row=3, column=1, columnspan=2, sticky=EW, pady=5, padx=2)
        self.setvar('End-time', None)

        lbl = ttk.Label(status_frm, text='Cost time:')
        lbl.configure(font="Arial")
        lbl.grid(row=4, column=0, sticky=W, pady=2)
        lbl = ttk.Label(status_frm, textvariable='total-time-cost')
        lbl.grid(row=4, column=1, columnspan=2, sticky=EW, pady=5, padx=2)
        self.setvar('total-time-cost', None)

        sep = ttk.Separator(status_frm, bootstyle=SECONDARY)
        sep.grid(row=5, column=0, columnspan=2, pady=10, sticky=EW)

        lbl = ttk.Label(status_frm, text='Now used method:')
        lbl.configure(font="Arial")
        lbl.grid(row=6, column=0, sticky=W, pady=5)
        lbl = ttk.Label(status_frm, textvariable='current-file-msg')
        lbl.configure(font="Arial")
        lbl.grid(row=6, column=1, columnspan=2, pady=5, sticky=EW, padx=5)
        self.setvar('current-file-msg', None)

        lbl = ttk.Label(self.left_panel, image='logo', style='bg.TLabel')
        lbl.pack(side='bottom')

        self.right_panel = ttk.Frame(self, padding=(2, 1))
        self.right_panel.pack(side=RIGHT, fill=BOTH, expand=YES)

        self.info_Frame = ttk.Frame(self.right_panel)
        self.info_Frame.pack(side=TOP, fill=X)

        self.ybar = ttk.Scrollbar(self.info_Frame)
        self.ybar.pack(side=RIGHT, fill=Y)

        self.tv = ttk.Treeview(self.info_Frame, show='headings', height=5, yscrollcommand=self.ybar.set)
        self.tv.configure(columns=(
            'Method', 'information', 'functions',
            'Estimated time cost', 'datetime'
        ))
        style = ttk.Style()

        def fixed_map(option):
            return [elm for elm in style.map('Treeview', query_opt=option) if elm[:2] != ('!disabled', '!selected')]

        style.map('Treeview',
                  foreground=fixed_map('foreground'),
                  background=fixed_map('background'),
                  )
        style.configure('Treeview.Heading', font='Arial')
        style.configure('Treeview', font='Arial')
        self.tv.column('Method', width=150, stretch=True)

        for col in ['information', 'Estimated time cost', 'datetime']:
            self.tv.column(col, stretch=False)

        for col in self.tv['columns']:
            self.tv.heading(col, text=col.title(), anchor=W)

        self.ybar.config(command=self.tv.yview)
        self.tv.pack(fill=X, pady=1)

        self.scroll_cf = CollapsingFrame(self.right_panel)
        self.scroll_cf.pack(fill=BOTH, expand=YES)

        self.output_container = ttk.Frame(self.scroll_cf, padding=1)
        self.scroll_cf.add(self.output_container, textvariable='scroll-message')

        self._value = 'Log: ' + 'STABox' + ' is ready, the visualization results are displayed below......'
        self.setvar('scroll-message', self._value)

        # R mcluster envs setting
        if self.R_HOME is not None and self.R_USER is not None:
            print(self.R_HOME)
            print(self.R_USER)
        else:
            try:
                os_path = os.environ['PATH'].split(';')
                for path in os_path:
                    if re.findall(r'.*R-.*', path):
                        new_path = path.split('\\bin')[0]

                self.R_HOME = new_path

                for path in os_path:
                    if re.findall(r'.*Anaconda\\envs.*?', path):
                        user_path = path.split('\\envs')[0]

                current_path = os.path.dirname(os.path.abspath(__file__))
                current_path = current_path.split('\\VIEW')[0]
                current_path = current_path.rsplit('\\', 1)[-1]
                self.R_USER = user_path + '\\envs\\' + current_path + '\\Lib\\site-packages\\rpy2'
                os.environ['R_HOME'] = self.R_HOME
                os.environ['R_USER'] = self.R_USER
                print(self.R_HOME)
                print(self.R_USER)

            except:
                Messagebox.show_error(title='Hi', message='please check R environ first!')

    def Select_files(self):
        files_show = ttk.Toplevel(self.bus_frm)
        files_show.title("Files select")
        files_show.geometry('300x250')
        frame0 = tk.Frame(files_show)
        frame0.pack()
        path_label = tk.Label(frame0, text=f'Files in {self.multi_files}:  ')
        path_label.grid(row=0, column=0)

        frame1 = ttk.Frame(files_show)
        frame1.pack(side="bottom")

        file_list = os.listdir(self.multi_files)
        print(file_list)

        def getselect(item):
            print(item, 'selected')

        def unselectall():
            for index, item in enumerate(list1):
                v[index].set('')

        def selectall():
            for index, item in enumerate(list1):
                v[index].set(item)

        def showselect():
            selected = [i.get() for i in v if i.get()]
            print(selected)
            self.multi_files = [os.path.join(self.multi_files, i) for i in selected]
            print(self.multi_files)
            files_show.destroy()
            tk.messagebox.showwarning(title='Attention', message=f"you have selected {len(selected)} file!")

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
            ttk.Checkbutton(frame2, text=item, variable=v[-1], onvalue=item, offvalue='',
                            command=lambda item=item: getselect(item)).grid(row=index, column=0,
                                                                            sticky='w')

        seltone = ttk.Radiobutton(frame1, text='All', variable=opt, value=1, command=selectall)
        seltone.grid(row=0, column=0)
        selttwo = ttk.Radiobutton(frame1, text='Cancel', variable=opt, value=0, command=unselectall)
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
            print('Dataload self.result_queue.qsize()', self.result_queue.qsize())
            print('self.file_path', self.file_path)
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
        style.configure('TButton', font=('Arial', 16))
        self.Data_load_toplevel = ttk.Toplevel(self.backup)
        self.Data_load_toplevel.attributes('-topmost', 'true')
        self.Data_load_toplevel.title("Data Loading")
        self.Data_load_toplevel.geometry('550x320')
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
                file_label_one = ttk.Label(self.load_moduel, text="Count_txt_file path:",
                                           font=('Arial', 16))
                file_label_one.grid(row=0, column=0, padx=5, pady=5, sticky="w")

                file_entry_one = ttk.Entry(self.load_moduel)
                file_entry_one.grid(row=0, column=1, padx=5, pady=5)

                file_label_two = ttk.Label(self.load_moduel, text="Location_csv_file path:",
                                           font=('Arial', 16))
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

                confirm_button_one = ttk.Button(self.load_moduel, text="Choose", style="TButton",
                                                command=browse_folder_one)
                confirm_button_one.grid(row=0, column=2, padx=5, pady=5)
                confirm_button_two = ttk.Button(self.load_moduel, text="Choose", style="TButton",
                                                command=browse_folder_two)
                confirm_button_two.grid(row=1, column=2, padx=5, pady=5)

                def load_datas():
                    count = pd.read_csv(file_entry_one.get(), sep='\t', index_col=0)
                    location = pd.read_csv(file_entry_two.get(), index_col=0)
                    adata = sc.AnnData(count.T)
                    adata.var_names_make_unique()
                    coor_df = location.loc[adata.obs_names, ['xcoord', 'ycoord']]
                    adata.obsm["spatial"] = coor_df.to_numpy()
                    sc.pp.normalize_total(adata, target_sum=1e4)
                    sc.pp.log1p(adata)
                    sc.pp.calculate_qc_metrics(adata, inplace=True)
                    path = file_entry_one.get()
                    adata.write_h5ad(path.rsplit('/', 1)[-2] + '/new_adata.h5ad')
                    self.path_load_flag = False
                    self.Data_load_toplevel.destroy()
                    Messagebox.show_warning(title="Attenetion", message='Files has been converted to h5ad!!!')

                confirm_button = ttk.Button(self.load_moduel, text="Confirm", style="TButton", command=load_datas)
                confirm_button.grid(row=2, column=1, padx=5, pady=5)

            elif value == "h5_to_h5ad":
                self.path_load_flag = True
                self.load_moduel = tk.Frame(self.Data_load_toplevel)
                self.load_moduel.pack(after=self.radio_moduel)
                file_label_one = ttk.Label(self.load_moduel, text="H5file folder Path:",
                                           font=('Arial', 16))
                file_label_one.grid(row=0, column=0, padx=5, pady=5)

                file_entry = ttk.Entry(self.load_moduel)
                file_entry.grid(row=0, column=1, padx=5, pady=5)

                def browse_folder():
                    folder_path = filedialog.askdirectory()
                    file_entry.delete(0, tk.END)
                    file_entry.insert(0, folder_path)

                confirm_button_one = ttk.Button(self.load_moduel, text="Choose", style="TButton",
                                                command=browse_folder)
                confirm_button_one.grid(row=0, column=2, padx=5, pady=5)

                def load_datas():
                    h5file = glob.glob(file_entry.get() + '/*.h5')
                    adata = sc.read_visium(path=file_entry.get(), count_file=h5file[0].rsplit("\\", 1)[-1])
                    adata.var_names_make_unique()
                    sc.pp.normalize_total(adata, target_sum=1e4)
                    sc.pp.log1p(adata)
                    sc.pp.calculate_qc_metrics(adata, inplace=True)
                    adata.write_h5ad(file_entry.get() + '/new_adata.h5ad')
                    self.path_load_flag = False
                    self.Data_load_toplevel.destroy()
                    Messagebox.show_warning(title="Attenetion", message='Files has been converted to h5ad!!!')

                confirm_button = ttk.Button(self.load_moduel, text="Confirm", style="TButton", command=load_datas)
                confirm_button.grid(row=2, column=1, padx=5, pady=5)

            elif value == "txt_to_h5ad" or value == "tsv_to_h5ad":
                self.path_load_flag = True
                self.load_moduel = tk.Frame(self.Data_load_toplevel)
                self.load_moduel.pack(after=self.radio_moduel)
                file_label_one = ttk.Label(self.load_moduel, text="Count_tsv_file Path:",
                                           font=('Arial', 16))
                file_label_one.grid(row=0, column=0, padx=5, pady=5, sticky="w")

                file_entry_one = ttk.Entry(self.load_moduel)
                file_entry_one.grid(row=0, column=1, padx=5, pady=5)

                file_label_two = ttk.Label(self.load_moduel, text="Location_tsv_file Path:",
                                           font=('Arial', 16))
                file_label_two.grid(row=1, column=0, padx=5, pady=5, sticky="w")

                file_entry_two = ttk.Entry(self.load_moduel)
                file_entry_two.grid(row=1, column=1, padx=5, pady=5)

                def browse_folder_one():
                    folder_path = filedialog.askdirectory()
                    file_entry_one.delete(0, tk.END)
                    file_entry_one.insert(0, folder_path)

                def browse_folder_two():
                    folder_path = filedialog.askdirectory()
                    file_entry_one.delete(0, tk.END)
                    file_entry_one.insert(0, folder_path)

                confirm_button_one = ttk.Button(self.load_moduel, text="Choose", style="TButton",
                                                command=browse_folder_one)
                confirm_button_one.grid(row=0, column=2, padx=5, pady=5)
                confirm_button_two = ttk.Button(self.load_moduel, text="Choose", style="TButton",
                                                command=browse_folder_two)
                confirm_button_two.grid(row=1, column=2, padx=5, pady=5)

                def load_datas():
                    count = pd.read_csv(file_entry_one.get(), sep='\t')
                    location = pd.read_csv(file_entry_two.get(), sep='\t')
                    adata = sc.AnnData(count)
                    adata.var_names_make_unique()
                    adata.obsm["spatial"] = location.to_numpy()
                    sc.pp.normalize_total(adata, target_sum=1e4)
                    sc.pp.log1p(adata)
                    path = file_entry_one.get()
                    sc.pp.calculate_qc_metrics(adata, inplace=True)
                    adata.write_h5ad(path.rsplit('/', 1)[-2] + '/new_adata.h5ad')
                    self.path_load_flag = False
                    self.Data_load_toplevel.destroy()
                    Messagebox.show_warning(title="Attenetion", message='Files has been converted to h5ad!!!')

                confirm_button = ttk.Button(self.load_moduel, text="Confirm", style="TButton", command=load_datas)
                confirm_button.grid(row=2, column=1, padx=5, pady=5)
            else:
                pass

        radio_var = tk.StringVar()
        radio1 = ttk.Radiobutton(self.radio_moduel, text="Raw_count_txt_and_location_csv_to_h5ad", style="TCheckbutton",
                                 variable=radio_var,
                                 value="txt_csv_to_h5ad", command=lambda: on_radio_select("txt_csv_to_h5ad"))
        radio1.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        radio2 = ttk.Radiobutton(self.radio_moduel, text="Raw_h5file_to_h5ad", style="TCheckbutton", variable=radio_var,
                                 value="h5_to_h5ad",
                                 command=lambda: on_radio_select("h5_to_h5ad"))
        radio2.grid(row=1, column=0, padx=5, pady=5, sticky="w")
        radio3 = ttk.Radiobutton(self.radio_moduel, text="Raw_count_txt_and_location_txt_to_h5ad", style="TCheckbutton",
                                 variable=radio_var,
                                 value="txt_to_h5ad", command=lambda: on_radio_select("txt_to_h5ad"))
        radio3.grid(row=2, column=0, padx=5, pady=5, sticky="w")
        radio4 = ttk.Radiobutton(self.radio_moduel, text="Raw_count_tsv_and_location_tsv_to_h5ad", style="TCheckbutton",
                                 variable=radio_var,
                                 value="tsv_to_h5ad", command=lambda: on_radio_select("tsv_to_h5ad"))
        radio4.grid(row=3, column=0, padx=5, pady=5, sticky="w")

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
        self.Data_load_toplevel.attributes('-topmost', 'true')
        self.Data_load_toplevel.title("Data Preprocess")
        self.Data_load_toplevel.geometry('550x320')
        self.radio_moduel = tk.Frame(self.Data_load_toplevel)
        self.radio_moduel.pack(side=TOP)
        style = Style()
        style.configure("TCheckbutton", font=("Arial", 16))
        style.configure('TButton', font=('Arial', 16))

        def on_radio_select(value):
            if self.path_load_flag:
                self.load_moduel.destroy()
                self.path_load_flag = False
            if value == "add_location":
                self.path_load_flag = True
                self.load_moduel = tk.Frame(self.Data_load_toplevel)
                self.load_moduel.pack(after=self.radio_moduel)
                file_label_one = ttk.Label(self.load_moduel, text="H5ad file path:", font=('Arial', 16))  # 创建Label控件
                file_label_one.grid(row=0, column=0, padx=5, pady=5, sticky="w")

                file_entry_one = ttk.Entry(self.load_moduel)  # 创建Entry控件
                file_entry_one.grid(row=0, column=1, padx=5, pady=5)

                file_label_two = ttk.Label(self.load_moduel, text="Location csv file Path:",
                                           font=('Arial', 16))  # 创建Label控件
                file_label_two.grid(row=1, column=0, padx=5, pady=5, sticky="w")

                file_entry_two = ttk.Entry(self.load_moduel)  # 创建Entry控件
                file_entry_two.grid(row=1, column=1, padx=5, pady=5)

                file_label_three = ttk.Label(self.load_moduel, text="Filter highly variable genes:",
                                             font=('Arial', 16))  # 创建Label控件
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

                confirm_button_one = ttk.Button(self.load_moduel, text="Choose", style="TButton",
                                                command=browse_folder_one)  # 创建确认按钮
                confirm_button_one.grid(row=0, column=2, padx=5, pady=5)
                confirm_button_two = ttk.Button(self.load_moduel, text="Choose", style="TButton",
                                                command=browse_folder_two)  # 创建确认按钮
                confirm_button_two.grid(row=1, column=2, padx=5, pady=5)

                def load_datas():
                    adata = sc.read(file_entry_one.get(), sep='\t', index_col=0)
                    location = pd.read_csv(file_entry_two.get(), sep=',', index_col=0)
                    adata.var_names_make_unique()
                    coor_df = location.loc[adata.obs_names, ['x', 'y']]
                    adata.obsm["spatial"] = coor_df.to_numpy()
                    sc.pp.normalize_total(adata, target_sum=1e4)
                    sc.pp.log1p(adata)
                    sc.pp.calculate_qc_metrics(adata, inplace=True)
                    if file_entry_three.get():
                        sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=int(file_entry_three.get()))
                    if 'highly_variable' in adata.var.columns:
                        adata = adata[:, adata.var['highly_variable']]
                    path = file_entry_one.get()
                    adata.write_h5ad(path.rsplit('/', 1)[-2] + '/new_adata.h5ad')
                    self.path_load_flag = False
                    self.Data_load_toplevel.destroy()
                    Messagebox.show_warning(title="Attenetion", message='Information has been added into h5ad files!!!')

                confirm_button = ttk.Button(self.load_moduel, text="Confirm", style="TButton", command=load_datas)
                confirm_button.grid(row=3, column=1, padx=5, pady=5)

            elif value == "add_GroundTurth":
                self.path_load_flag = True
                self.load_moduel = tk.Frame(self.Data_load_toplevel)
                self.load_moduel.pack(after=self.radio_moduel)
                file_label_one = ttk.Label(self.load_moduel, text="H5ad file path:", font=('Arial', 16))  # 创建Label控件
                file_label_one.grid(row=0, column=0, padx=5, pady=5, sticky="w")

                file_entry_one = ttk.Entry(self.load_moduel)
                file_entry_one.grid(row=0, column=1, padx=5, pady=5)

                file_label_two = ttk.Label(self.load_moduel, text="GroundTruth txt file Path:",
                                           font=('Arial', 16))  # 创建Label控件
                file_label_two.grid(row=1, column=0, padx=5, pady=5, sticky="w")

                file_entry_two = ttk.Entry(self.load_moduel)
                file_entry_two.grid(row=1, column=1, padx=5, pady=5)

                file_label_three = ttk.Label(self.load_moduel, text="Filter highly variable genes:",
                                             font=('Arial', 16))  # 创建Label控件
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

                confirm_button_one = ttk.Button(self.load_moduel, text="Choose", style="TButton",
                                                command=browse_folder_one)  # 创建确认按钮
                confirm_button_one.grid(row=0, column=2, padx=5, pady=5)

                confirm_button_two = ttk.Button(self.load_moduel, text="Choose", style="TButton",
                                                command=browse_folder_two)  # 创建确认按钮
                confirm_button_two.grid(row=1, column=2, padx=5, pady=5)

                def load_datas():
                    adata = sc.read(file_entry_one.get())
                    adata.var_names_make_unique()
                    Ann_df = pd.read_csv(file_entry_two.get(), sep='\t', header=None, index_col=0)
                    Ann_df.columns = ['Ground Truth']
                    adata.obs['GroundTruth'] = Ann_df.loc[adata.obs_names, 'Ground Truth']
                    sc.pp.normalize_total(adata, target_sum=1e4)
                    sc.pp.log1p(adata)
                    sc.pp.calculate_qc_metrics(adata, inplace=True)
                    if file_entry_three.get():
                        sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=int(file_entry_three.get()))
                    if 'highly_variable' in adata.var.columns:
                        adata = adata[:, adata.var['highly_variable']]
                    path = file_entry_one.get()
                    adata.write_h5ad(path.rsplit('/', 1)[-2] + '/new_adata.h5ad')
                    del adata
                    self.path_load_flag = False
                    self.Data_load_toplevel.destroy()
                    Messagebox.show_warning(title="Attenetion", message='Information has been added into h5ad files!!!')

                confirm_button = ttk.Button(self.load_moduel, text="Confirm", style="TButton", command=load_datas)
                confirm_button.grid(row=3, column=1, padx=5, pady=5)
            else:
                self.path_load_flag = True
                self.load_moduel = tk.Frame(self.Data_load_toplevel)
                self.load_moduel.pack(after=self.radio_moduel)
                file_label_one = ttk.Label(self.load_moduel, text="H5ad file path:", font=('Arial', 16))  # 创建Label控件
                file_label_one.grid(row=0, column=0, padx=5, pady=5, sticky="w")

                file_entry_one = ttk.Entry(self.load_moduel)
                file_entry_one.grid(row=0, column=1, padx=5, pady=5)

                file_label_two = ttk.Label(self.load_moduel, text="Filter highly variable genes:",
                                           font=('Arial', 16))
                file_label_two.grid(row=1, column=0, padx=5, pady=5, sticky="w")

                file_entry_two = ttk.Entry(self.load_moduel)
                file_entry_two.grid(row=1, column=1, padx=5, pady=5)

                def browse_folder_one():
                    folder_path = filedialog.askopenfilename()
                    file_entry_one.delete(0, tk.END)
                    file_entry_one.insert(0, folder_path)

                confirm_button_one = ttk.Button(self.load_moduel, text="Choose", style="TButton",
                                                command=browse_folder_one)
                confirm_button_one.grid(row=0, column=2, padx=5, pady=5)

                def load_datas():
                    adata = sc.read(file_entry_one.get())
                    adata.var_names_make_unique()
                    sc.pp.normalize_total(adata, target_sum=1e4)
                    sc.pp.log1p(adata)
                    sc.pp.calculate_qc_metrics(adata, inplace=True)
                    if file_entry_two.get():
                        sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=int(file_entry_two.get()))
                    if 'highly_variable' in adata.var.columns:
                        adata = adata[:, adata.var['highly_variable']]
                    path = file_entry_one.get()
                    adata.write_h5ad(path.rsplit('/', 1)[-2] + '/new_adata.h5ad')
                    self.path_load_flag = False
                    self.Data_load_toplevel.destroy()
                    Messagebox.show_warning(title="Attenetion", message='Information has been added into h5ad files!!!')

                confirm_button = ttk.Button(self.load_moduel, text="Confirm", style="TButton", command=load_datas)
                confirm_button.grid(row=2, column=1, padx=5, pady=5)

        radio_var = tk.StringVar()
        radio1 = ttk.Radiobutton(self.radio_moduel, text="H5ad_file_add_location_csv_information", style="TCheckbutton",
                                 variable=radio_var,
                                 value="add_location", command=lambda: on_radio_select("add_location"))
        radio1.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        radio2 = ttk.Radiobutton(self.radio_moduel, text="H5ad_file_add_GroundTurth_txt_information",
                                 style="TCheckbutton", variable=radio_var,
                                 value="add_GroundTurth", command=lambda: on_radio_select("add_GroundTurth"))
        radio2.grid(row=1, column=0, padx=5, pady=5, sticky="w")
        # radio3 = ttk.Radiobutton(self.radio_moduel, text="Multiple h5ad conversions to STAligner input format",
        #                          style="TCheckbutton", variable=radio_var,
        #                          value="multiple_h5ad_files", command=lambda: on_radio_select("multiple_h5ad_files"))
        # radio3.grid(row=2, column=0, padx=5, pady=5, sticky="w")
        radio3 = ttk.Radiobutton(self.radio_moduel, text="Only_filter_top_n_genes_and_normalization",
                                 style="TCheckbutton", variable=radio_var,
                                 value="filter_top_n", command=lambda: on_radio_select("filter_top_n"))
        radio3.grid(row=2, column=0, padx=5, pady=5, sticky="w")

        def on_closing():
            self.path_load_flag = False
            self.Data_load_toplevel.destroy()

        self.Data_load_toplevel.protocol("WM_DELETE_WINDOW", on_closing)

    def ShutdowmSoftware(self):
        sys.exit()

    def web_Server_Thread(self, data_save_path, ports=8050):
        T = threading.Thread(target=webServer, args=(data_save_path, ports,))
        T.setDaemon(True)
        T.start()

    def web_Thread(self, filepath, inh5data_, conf_file_, data_name):
        T = threading.Thread(target=webcache_main, args=(filepath, inh5data_, conf_file_, data_name,))
        T.setDaemon(True)
        T.start()

    def show(self):
        plt.rcParams['font.sans-serif'] = "Arial"
        self.gene_color_type = 'viridis'

        def draw_images():
            i = 1
            os.chdir(Raw_PATH)
            from matplotlib import pyplot as plt
            plt.rcParams["figure.figsize"] = (3, 3)
            path = test_file_path + '/figures/' + self.method_flag + '_' + self.data_type
            if self.label_files_exit:
                sc.pl.spatial(adata, img_key="hires", color="GroundTruth", title='Ground Truth', show=False,
                              save='STAGATE_' + str(i) + '.png')
                shutil.move(test_file_path + '/figures/showSTAGATE_' + str(i) + '.png', path + '/STAGATE_' + str(i) + '.png')

                i = i + 1
                obs_df = adata.obs.dropna()
                ARI = adjusted_rand_score(obs_df['mclust'], obs_df['GroundTruth'])

                plt.rcParams["figure.figsize"] = (3, 3)
                sc.pl.umap(adata, color="mclust", title=['STAGATE (ARI=%.2f)' % ARI], show=False, s=6,
                           save='STAGATE_' + str(i) + '.png')
                shutil.move(test_file_path + '/figures/umapSTAGATE_' + str(i) + '.png', path + '/STAGATE_' + str(i) + '.png')

                i = i + 1
                sc.pl.spatial(adata, color="mclust", title=['STAGATE (ARI=%.2f)' % ARI], show=False,
                              save='STAGATE_' + str(i) + '.png')
                shutil.move(test_file_path + '/figures/showSTAGATE_' + str(i) + '.png',
                            path + '/STAGATE_' + str(i) + '.png')

                if self.genes in adata.var_names:
                    i = i + 1
                    sc.pl.spatial(adata, img_key="hires", color=self.genes, show=False,
                                  title="RAW-$" + self.genes + "$",
                                  vmax='p99', color_map=self.gene_color_type, save='STAGATE_' + str(i) + '.png')
                    shutil.move(test_file_path + '/figures/showSTAGATE_' + str(i) + '.png',
                                path + '/STAGATE_' + str(i) + '.png')

                adata.obs.GroundTruth = adata.obs.GroundTruth.astype(str)
                i = i + 1
                sc.tl.paga(adata, groups='GroundTruth')
                plt.rcParams["figure.figsize"] = (4, 3)
                sc.pl.paga(adata, color="GroundTruth", title='PAGA', show=False, save='STAGATE_' + str(i) + '.png')
                shutil.move(test_file_path + '/figures/pagaSTAGATE_' + str(i) + '.png', path + '/STAGATE_' + str(i) + '.png')
            else:
                sc.pp.calculate_qc_metrics(adata, inplace=True)
                sc.pl.spatial(adata, img_key="hires", color="log1p_total_counts", title='log1p_total_counts',
                              show=False,
                              spot_size=20, color_map=self.gene_color_type, save='STAGATE_' + str(i) + '.png')
                shutil.move(test_file_path + '/figures/showSTAGATE_' + str(i) + '.png',
                            path + '/STAGATE_' + str(i) + '.png')

                i = i + 1
                plt.rcParams["figure.figsize"] = (3, 3)
                sc.pl.umap(adata, color="mclust", title=['STAGATE-Mclust'], show=False, s=6,
                           save='STAGATE_' + str(i) + '.png')
                shutil.move(test_file_path + '/figures/umapSTAGATE_' + str(i) + '.png',
                            path + '/STAGATE_' + str(i) + '.png')

                i = i + 1
                sc.pl.spatial(adata, color="mclust", title=['STAGATE-Mclust'], show=False, spot_size=20,
                              save='STAGATE_' + str(i) + '.png')
                shutil.move(test_file_path + '/figures/showSTAGATE_' + str(i) + '.png',
                            path + '/STAGATE_' + str(i) + '.png')
                if self.genes in adata.var_names:
                    plot_gene = self.genes
                    i = i + 1
                    sc.pl.spatial(adata, img_key="hires", color=plot_gene, show=False, title="RAW-$" + plot_gene + "$",
                                  spot_size=20, vmax='p99', color_map=self.gene_color_type,
                                  save='STAGATE_' + str(i) + '.png')
                    shutil.move(test_file_path + '/figures/showSTAGATE_' + str(i) + '.png',
                                path + '/STAGATE_' + str(i) + '.png')
                    i = i + 1
                    sc.pl.spatial(adata, img_key="hires", color=plot_gene, show=False,
                                  title="STAGATE-$" + plot_gene + "$",
                                  layer='STAGATE_ReX', vmax='p99', color_map=self.gene_color_type,
                                  spot_size=20, save='STAGATE_' + str(i) + '.png')
                    shutil.move(test_file_path + '/figures/showSTAGATE_' + str(i) + '.png',
                                path + '/STAGATE_' + str(i) + '.png')
                i = i + 1
                plt.rcParams["figure.figsize"] = (3, 3)
                sc.pl.umap(adata, color="louvain", title=['STAGATE-louvain'], show=False, s=6,
                           save='STAGATE_' + str(i) + '.png')
                shutil.move(test_file_path + '/figures/umapSTAGATE_' + str(i) + '.png',
                            path + '/STAGATE_' + str(i) + '.png')

                i = i + 1
                sc.pl.spatial(adata, color="louvain", title=['STAGATE-louvain'], show=False, spot_size=20,
                              save='STAGATE_' + str(i) + '.png')
                shutil.move(test_file_path + '/figures/showSTAGATE_' + str(i) + '.png',
                            path + '/STAGATE_' + str(i) + '.png')
            adata.write_h5ad(self.file_path.rsplit('/', 1)[-2] + '/' + self.method_flag + '_result.h5ad')

        def scoller():
            self.figure_ybar = ttk.Scrollbar(self.figure_Frame, orient=VERTICAL, cursor='draft_small')
            self.figure_ybar.pack(side=RIGHT, fill=Y)
            self.figure_ybar.config(command=self.canvas.yview)

            self.figure_xbar = ttk.Scrollbar(self.figure_Frame, orient=HORIZONTAL, cursor='draft_small')
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
            self.canvas.configure(scrollregion=self.canvas.bbox("all"), width=1000, height=650)

        def choose_color():
            colorvalue = colorchooser.askcolor()
            color_list = []
            color = colorvalue[1]
            print(color)
            if 'louvain_colors' in adata.uns:
                color_lens = [adata.uns['louvain_colors'], adata.uns['mclust_colors']]
                max_len = max(color_lens, key=len)
            else:
                max_len = adata.uns['mclust_colors']
            if len(max_len) > self.cluster_value:
                self.cluster_value = len(max_len)
            if color[1:3] == 'ff':
                color_list.append(color)
                list = random.sample(color_panel.FF_color, self.cluster_value)
                color_list = color_list + list
            elif color[1:3] == 'cc':
                color_list.append(color)
                list = random.sample(color_panel.CC_color, self.cluster_value)
                color_list = color_list + list
            elif color[1:3] == '99':
                color_list.append(color)
                list = random.sample(color_panel.NN_color, self.cluster_value)
                color_list = color_list + list
            elif color[1:3] == '66':
                color_list.append(color)
                list = random.sample(color_panel.SS_color, self.cluster_value)
                color_list = color_list + list
            elif color[1:3] == '33':
                color_list.append(color)
                list = random.sample(color_panel.TT_color, self.cluster_value)
                color_list = color_list + list
            elif color[1:3] == '00':
                color_list.append(color)
                list = random.sample(color_panel.ZZ_color, self.cluster_value)
                color_list = color_list + list
            else:
                color_list.append(color)
                list = random.sample(color_panel.random_color, self.cluster_value)
                color_list = color_list + list

            print(color_list)
            if self.label_files_exit:
                adata.uns['GroundTruth_colors'] = color_list[:len(adata.uns['GroundTruth_colors'])]
            adata.uns['mclust_colors'] = color_list[:len(adata.uns['mclust_colors'])]
            if 'louvain_colors' in adata.uns:
                adata.uns['louvain_colors'] = color_list[:len(adata.uns['louvain_colors'])]
            self.color_reset = color_list
            self.gene_color_type = color_panel.five_gene_color[random.randint(0, 5)]
            os.chdir(Raw_PATH)
            draw_images()

            fig_path = test_file_path + '/figures/' + self.method_flag + '_' + self.data_type
            figures = os.listdir(fig_path)
            global image_list
            image_list = []
            for i in range(len(figures)):
                image_list.append(ttk.PhotoImage(file=fig_path + '/' + figures[i]))
                self.result_images = ttk.Label(self.show_frame, image=image_list[i])
                self.result_images.grid(row=i // 3, column=i % 3, sticky=W, pady=0)

        fig_path = test_file_path + '/figures/' + self.method_flag + '_' + self.data_type
        if not os.path.isdir(fig_path):
            os.mkdir(fig_path)

        adata = self.result_queue.get()
        if self.label_files_exit:
            adata = adata[adata.obs['GroundTruth'] == adata.obs['GroundTruth'],]
        else:
            adata.obsm["spatial"] = adata.obsm["spatial"] * (-1)
        draw_images()
        self.figure_Frame = ttk.Frame(self.right_panel)
        self.figure_Frame.pack(side=TOP, fill=X, expand=YES)

        self.canvas = ttk.Canvas(self.figure_Frame)
        self.show_frame = ttk.Frame(self.canvas)
        scoller()

        self.set_frame = ttk.Frame(self.figure_Frame, borderwidth=2, relief="sunken")
        self.set_frame.pack(side='right', expand=YES, anchor=N)
        self.set_frame_one = ttk.Frame(self.set_frame)
        self.set_frame_one.pack(side=TOP, expand=YES)
        self.set_frame_two = ttk.Frame(self.set_frame)
        self.set_frame_two.pack(side=TOP, expand=YES)

        self.Reset = ttk.Button(self.set_frame_one, text='Reset all colors', command=choose_color, width=12)
        self.Reset.grid(row=0, column=0, sticky=W, pady=0, padx=0)

        self.domain_color_label = ttk.Label(self.set_frame_one, text='Reset domain color: ', width=20)
        self.domain_color_label.grid(row=1, column=0, sticky=W, pady=0)

        self.Reset_domain_color = ttk.Entry(self.set_frame_one, width=10)
        self.Reset_domain_color.grid(row=1, column=1, sticky=W, pady=0)
        Tooltip(self.Reset_domain_color, "Input value must be int type and in [1:cluster_name]: 2")
        self.color_update = None

        def Reset_single_domain_color():
            from tkinter import colorchooser, filedialog
            colorvalue = colorchooser.askcolor()
            color = colorvalue[1]
            print(color)
            cluster = self.Reset_domain_color.get()
            print(cluster)
            adata.uns['mclust_colors'] = self.color_reset[:len(adata.uns['mclust_colors'])]
            adata.uns['mclust_colors'][int(cluster) - 1] = color
            if 'louvain_colors' in adata.uns:
                adata.uns['louvain_colors'] = self.color_reset[:len(adata.uns['louvain_colors'])]
                adata.uns['louvain_colors'][int(cluster) - 1] = color
            self.color_reset[int(cluster) - 1] = color
            self.color_update = self.color_reset
            print(f"adata.uns['mclust_colors'] = {adata.uns['mclust_colors']}")
            import matplotlib.pyplot as plt
            plt.rcParams["figure.figsize"] = (3, 3)
            i = 1
            path = test_file_path + '/figures/' + self.method_flag + '_' + self.data_type

            if self.label_files_exit:
                i = i + 1
                obs_df = adata.obs.dropna()
                ARI = adjusted_rand_score(obs_df['mclust'], obs_df['GroundTruth'])

                plt.rcParams["figure.figsize"] = (3, 3)
                sc.pl.umap(adata, color="mclust", title=['STAGATE (ARI=%.2f)' % ARI], show=False, s=6,
                           save='STAGATE_' + str(i) + '.png')
                shutil.move(test_file_path + '/figures/umapSTAGATE_' + str(i) + '.png', path + '/STAGATE_' + str(i) + '.png')
                i = i + 1
                sc.pl.spatial(adata, color="mclust", title=['STAGATE (ARI=%.2f)' % ARI], show=False,
                              spot_size=150, save='STAGATE_' + str(i) + '.png')
                shutil.move(test_file_path + '/figures/showSTAGATE_' + str(i) + '.png', path + '/STAGATE_' + str(i) + '.png')
            else:
                i = i + 1
                plt.rcParams["figure.figsize"] = (3, 3)
                sc.pl.umap(adata, color="mclust", title=['STAGATE'], show=False, s=6,
                           save='STAGATE_' + str(i) + '.png')
                shutil.move(test_file_path + '/figures/umapSTAGATE_' + str(i) + '.png',
                            path + '/STAGATE_' + str(i) + '.png')
                i = i + 1
                sc.pl.spatial(adata, color="mclust", title=['STAGATE'], show=False,
                              spot_size=150, save='STAGATE_' + str(i) + '.png')
                shutil.move(test_file_path + '/figures/showSTAGATE_' + str(i) + '.png', path + '/STAGATE_' + str(i) + '.png')
                i = i + 1
                plt.rcParams["figure.figsize"] = (3, 3)
                sc.pl.umap(adata, color="louvain", title=['STAGATE-louvain'], show=False, s=6,
                           save='STAGATE_' + str(i) + '.png')
                shutil.move(test_file_path + '/figures/umapSTAGATE_' + str(i) + '.png',
                            path + '/STAGATE_' + str(i) + '.png')

                i = i + 1
                sc.pl.spatial(adata, color="louvain", title=['STAGATE-louvain'], show=False, spot_size=150,
                              save='STAGATE_' + str(i) + '.png')
                shutil.move(test_file_path + '/figures/showSTAGATE_' + str(i) + '.png',
                            path + '/STAGATE_' + str(i) + '.png')
            figures = os.listdir(fig_path)
            global image_list
            image_list = []
            for i in range(len(figures)):
                img = Image.open(fig_path + '/' + figures[i])
                print(img.size)
                image_list.append(ttk.PhotoImage(file=fig_path + '/' + figures[i]))
                self.result_images = ttk.Label(self.show_frame, image=image_list[i])
                self.result_images.grid(row=i // 3, column=i % 3, sticky=W, pady=0, padx=0)
                s = i

        self.Reset_domain_btn = ttk.Button(self.set_frame_one, width=10, text='Confirm',
                                           command=Reset_single_domain_color)
        self.Reset_domain_btn.grid(row=1, column=2, sticky=W, pady=10)
        self.update_image_label = ttk.Label(self.set_frame_one, text='Reset image dpi: ', width=20)
        self.update_image_label.grid(row=2, column=0, sticky=W, pady=10)
        self.update_image_scale = ttk.Entry(self.set_frame_one, width=10)
        self.update_image_scale.grid(row=2, column=1, sticky=W, pady=10)
        Tooltip(self.update_image_scale, "Input value must be int type and >= 300: 300")

        def ipdate_hd():
            dpi = self.update_image_scale.get()
            print(dpi)
            print(self.color_update)

            import matplotlib.pyplot as plt
            plt.rcParams["figure.figsize"] = (3, 3)
            sc.set_figure_params(dpi=dpi)
            draw_images()
            figures = os.listdir(fig_path)
            print(len(figures))

        self.Reset_domain_btn = ttk.Button(self.set_frame_one, width=10, text='Save', command=ipdate_hd)
        self.Reset_domain_btn.grid(row=2, column=2, sticky=W, pady=0)

        self.gene_visualization_label = ttk.Label(self.set_frame_one, text='Input gene name: ', width=20)
        self.gene_visualization_label.grid(row=3, column=0, sticky=W, pady=0)

        self.gene_visualization_entry = ttk.Entry(self.set_frame_one, width=10)
        self.gene_visualization_entry.grid(row=3, column=1, sticky=W, pady=0)
        Tooltip(self.gene_visualization_entry, "Gene name: NEFH/ATP2B4")

        def gene_visualization():
            gene = self.gene_visualization_entry.get()
            if gene not in adata.var_names:
                print(f'{gene} is mot in adata!!Please input right gene name!')
            sc.pl.spatial(adata, img_key="hires", color=gene, title="$" + gene + "$", spot_size=150, show=False,
                          save=gene + '.png')
            global img0
            photo = Image.open(test_file_path + '/figures/show' + gene + '.png')
            img0 = ImageTk.PhotoImage(photo)
            img1 = ttk.Label(self.set_frame_two, image=img0)
            img1.grid(row=0, column=0, sticky=W, pady=0)
            if os.path.exists(test_file_path + '/figures/show' + gene + '.png'):
                os.remove(test_file_path + '/figures/show' + gene + '.png')
                print("yes")
            else:
                print("error！")

        self.gene_visualization_btn = ttk.Button(self.set_frame_one, width=10, command=gene_visualization, text='Show')
        self.gene_visualization_btn.grid(row=3, column=2, sticky=W, pady=0)
        figures = os.listdir(fig_path)
        global image_list
        image_list = []
        for i in range(len(figures)):
            image_list.append(ttk.PhotoImage(file=fig_path + '/' + figures[i]))
            self.result_images = ttk.Label(self.show_frame, image=image_list[i])
            self.result_images.grid(row=i // 3, column=i % 3, sticky=W, pady=0)
            s = i

        self.Entry = ttk.Entry(self.set_frame_one, width=10)
        self.Entry.grid(row=0, column=1, sticky=W, pady=0)
        Tooltip(self.Entry, "Local port number: 8050")

        def VIEW_3D():
            import webbrowser
            self.web_Server_Thread("/DLPFC/webcache", int(self.Entry.get()))
            http = 'http://127.0.0.1:' + self.Entry.get() + '/'
            # 'http://127.0.0.1:8050/'
            webbrowser.open(http)

        self.Reset = ttk.Button(self.set_frame_one, text='3D VIEW', command=VIEW_3D, width=10)
        self.Reset.grid(row=0, column=2, sticky=W, pady=0)

        self.pb.stop()
        self.setvar('prog-message', 'STAGATE run over!')
        self.setvar('End-time', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        EndTime = datetime.now().replace(microsecond=0)
        self.setvar('total-time-cost', EndTime - self.StartTime)

    def get_directory(self):
        """Open dialogue to get directory and update variable"""
        self.update_idletasks()
        self.file_path = filedialog.askopenfilename()
        print(self.file_path)
        if self.file_path:
            self.setvar('folder-path', self.file_path)

        if self.path_load_flag:
            self.path_load_flag = False
            self.paned_window.destroy()

        if self.file_path:
            self.path_load_flag = True
            self.Dataload_thread()

    def Slideseq_data_process(self):
        h5ad_file = glob.glob(self.file_path + '/*.h5ad')
        if os.path.exists(self.file_path) and len(h5ad_file) != 0:
            self.setvar('data_name', 'SlideSeq-V2 Data')
            adata = sc.read(h5ad_file[0])
            adata.var_names_make_unique()
            fsize = os.path.getsize(h5ad_file[0])
            self.setvar('data_size', str(round(fsize / 1024 / 1024, 2)) + 'MB')
            self.setvar('data_shape', str(adata.shape[0]) + '×' + str(adata.shape[1]))
            sc.pp.calculate_qc_metrics(adata, inplace=True)
            sc.pp.filter_genes(adata, min_cells=50)
            sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
            self.result_queue.put(adata)
            pass
        else:
            Messagebox.show_error(title="File error", message="Make sure only one h5ad file in path!")

    def Stereoseq_data_process(self):
        h5ad_file = glob.glob(self.file_path + '/*.h5ad')
        if os.path.exists(self.file_path) and len(h5ad_file) != 0:
            self.setvar('data_name', 'Stereo-seq Data')
            adata = sc.read(h5ad_file[0])
            adata.var_names_make_unique()
            fsize = os.path.getsize(h5ad_file[0])
            self.setvar('data_size', str(round(fsize / 1024 / 1024, 2)) + 'MB')
            self.setvar('data_shape', str(adata.shape[0]) + '×' + str(adata.shape[1]))
            sc.pp.calculate_qc_metrics(adata, inplace=True)
            sc.pp.filter_genes(adata, min_cells=50)
            sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
            self.result_queue.put(adata)
            pass
        else:
            Messagebox.show_error(title="File error", message="Make sure only one h5ad file in path!")
        pass

    def Visium_data_process(self):
        if self.multi_files is not None and self.method_flag == "STAligner":
            Batch_list = []
            adj_list = []
            key_add = []
            for i in self.multi_files:
                file_name = i.rsplit('\\', 1)[-1]
                key_add.append(file_name.rsplit('.', 1)[-2])
                adata_ = sc.read_h5ad(i)
                adata_.X = csr_matrix(adata_.X)
                adata_.var_names_make_unique(join="++")
                sc.pp.filter_genes(adata_, min_cells=50)
                adata_.obs_names = [x + '_' + file_name.rsplit('.', 1)[-2] for x in adata_.obs_names]
                Cal_Spatial_Net_new(adata_, rad_cutoff=self.rad_cutoff_value)
                sc.pp.highly_variable_genes(adata_, flavor="seurat_v3", n_top_genes=5000)
                if 'highly_variable' in adata_.var.columns:
                    adata_ = adata_[:, adata_.var['highly_variable']]
                sc.pp.normalize_total(adata_, target_sum=1e4)
                sc.pp.log1p(adata_)
                adj_list.append(adata_.uns['adj'])
                Batch_list.append(adata_)
            adata = ad.concat(Batch_list, label="slice_name", keys=key_add)
            adata.obs["batch_name"] = adata.obs["slice_name"].astype('category')
            adj_concat = np.asarray(adj_list[0].todense())
            for batch_id in range(1, len(self.multi_files)):
                adj_concat = scipy.linalg.block_diag(adj_concat, np.asarray(adj_list[batch_id].todense()))
            adata.uns['edgeList'] = np.nonzero(adj_concat)
        elif self.method_flag == "STAGATE":
            adata = sc.read(self.file_path)
            adata.var_names_make_unique()
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)

        self.paned_window = tk.PanedWindow(self.right_panel, orient=tk.HORIZONTAL, sashrelief=tk.SUNKEN, height=100)
        self.paned_window.pack(fill=X, before=self.info_Frame)
        text_widget = ttk.Text(self.paned_window)
        self.paned_window.add(text_widget)
        max_lines = 8
        redirector = PrintRedirector(text_widget, max_lines)
        sys.stdout = redirector
        sys.stdout = redirector
        print(adata)
        sys.stdout = sys.__stdout__
        self.result_queue.put(adata)
        self.data_load_flag = True
        del adata

    def Run_STAGATE(self):
        if not self.result_queue.empty():
            adata = self.result_queue.get()
            if 'GroundTruth' in adata.obs.columns:
                self.label_files_exit = True
            if 'STAGATE' not in adata.obsm:

                data_save_path = running_path + '/result/'
                Cal_Spatial_Net(adata, self.rad_cutoff_value)
                Stats_Spatial_Net(adata)
                stagate = STAGATE(model_dir=data_save_path,
                                  in_features=3000, hidden_dims=[512, 30])
                adata = stagate.train(adata)
                sc.pp.neighbors(adata, use_rep='STAGATE')
                sc.tl.umap(adata)
                adata = mclust_R(adata, used_obsm='STAGATE', num_cluster=self.cluster_value)
                sc.tl.louvain(adata, resolution=self.alpha_value)

            data_save_path = test_file_path + '/DLPFC'
            if not os.path.exists(data_save_path):
                os.makedirs(data_save_path)
                adata.write_h5ad(os.path.join(data_save_path, 'DLPFC.h5ad'))
                json_file = {
                    "Coordinate": "spatial",
                    "Annotatinos": ["mclust"],
                    "Meshes": {
                    },
                    "mesh_coord": "fixed.json",
                    "Genes": [
                        "all"
                    ]
                }
                fixed_file = {"xmin": 0, "ymin": 0, "margin": 0, "zmin": 0, "binsize": 1}
                json.dump(json_file, open(os.path.join(data_save_path, 'DLPFC.json'), 'w'), indent=4)
                json.dump(fixed_file, open(os.path.join(data_save_path, 'fixed.json'), 'w'))
                self.web_Thread(data_save_path, 'DLPFC.h5ad', 'DLPFC.json', 'Single-DLPFC')

            self.result_queue.put(adata)
            self.show()

    def STAGATE_Thread(self):
        T = threading.Thread(target=self.Run_STAGATE)
        T.setDaemon(True)
        T.start()

    def STAGATE_Slideseq_data_analysis(self):
        self.Slideseq_data_process()
        adata = self.result_queue.get()
        new_adata = adata.copy()
        Cal_Spatial_Net(adata, rad_cutoff=self.rad_cutoff_value)
        Stats_Spatial_Net(adata)
        train_STAGATE(adata)
        sc.pp.neighbors(adata, use_rep='STAGATE')
        sc.tl.umap(adata)
        sc.tl.louvain(adata, resolution=self.alpha_value)
        # adata.obsm["spatial"] = adata.obsm["spatial"] * (-1)

        sc.pp.pca(new_adata, n_comps=30)
        sc.pp.neighbors(new_adata, use_rep='X_pca')
        sc.tl.louvain(new_adata, resolution=self.alpha_value, key_added='scanpy')
        sc.tl.umap(new_adata)
        self.gene_color_type = 'viridis'

        def draw_images():
            from matplotlib import pyplot as plt
            plt.rcParams['font.sans-serif'] = "Arial"
            i = 1
            plt.rcParams["figure.figsize"] = (3, 3)
            sc.pl.embedding(adata, basis="spatial", color="log1p_total_counts", title='Raw Data', s=6, show=False,
                            save='SlideseqV2/STAGATE_' + str(i) + '.png', color_map=self.gene_color_type)
            i = i + 1
            plt.rcParams["figure.figsize"] = (3, 3)
            sc.pl.embedding(adata, basis="spatial", color="louvain", title='STAGATE', s=6, show=False,
                            save='SlideseqV2/STAGATE_' + str(i) + '.png')
            i = i + 1
            plt.rcParams["figure.figsize"] = (3, 3)
            sc.pl.umap(adata, color='louvain', title='STAGATE', show=False, s=6, save='STAGATE_' + str(i) + '.png')
            shutil.move('./figures/umapSTAGATE_' + str(i) + '.png',
                        './figures/spatialSlideseqV2/STAGATE_' + str(i) + '.png')
            i = i + 1
            plt.rcParams["figure.figsize"] = (3, 3)
            sc.pl.embedding(new_adata, basis="spatial", color="scanpy", title='SCANPY', s=6, show=False,
                            save='SlideseqV2/STAGATE_' + str(i) + '.png')
            i = i + 1
            plt.rcParams["figure.figsize"] = (3, 3)
            sc.pl.umap(new_adata, color='scanpy', title='SCANPY', show=False, s=6, save='STAGATE_' + str(i) + '.png')
            shutil.move('./figures/umapSTAGATE_' + str(i) + '.png',
                        './figures/spatialSlideseqV2/STAGATE_' + str(i) + '.png')

        def scoller():
            self.figure_ybar = ttk.Scrollbar(self.figure_Frame, orient=VERTICAL, cursor='draft_small')
            self.figure_ybar.pack(side=RIGHT, fill=Y)
            self.figure_ybar.config(command=self.canvas.yview)

            self.figure_xbar = ttk.Scrollbar(self.figure_Frame, orient=HORIZONTAL, cursor='draft_small')
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
            self.canvas.configure(scrollregion=self.canvas.bbox("all"), width=1000, height=650)

        def choose_color():
            colorvalue = colorchooser.askcolor()
            color_list = []
            color = colorvalue[1]
            color_lens = [adata.uns['louvain_colors'], new_adata.uns['scanpy_colors']]
            max_len = max(color_lens, key=len)
            if len(max_len) > self.cluster_value:
                self.cluster_value = len(max_len)
            if color[1:3] == 'ff':
                color_list.append(color)
                list = random.sample(color_panel.FF_color, self.cluster_value)
                color_list = color_list + list
            elif color[1:3] == 'cc':
                color_list.append(color)
                list = random.sample(color_panel.CC_color, self.cluster_value)
                color_list = color_list + list
            elif color[1:3] == '99':
                color_list.append(color)
                list = random.sample(color_panel.NN_color, self.cluster_value)
                color_list = color_list + list
            elif color[1:3] == '66':
                color_list.append(color)
                list = random.sample(color_panel.SS_color, self.cluster_value)
                color_list = color_list + list
            elif color[1:3] == '33':
                color_list.append(color)
                list = random.sample(color_panel.TT_color, self.cluster_value)
                color_list = color_list + list
            elif color[1:3] == '00':
                color_list.append(color)
                list = random.sample(color_panel.ZZ_color, self.cluster_value)
                color_list = color_list + list
            else:
                color_list.append(color)
                list = random.sample(color_panel.random_color, self.cluster_value)
                color_list = color_list + list

            print(color_list)
            adata.uns['louvain_colors'] = color_list[:len(adata.uns['louvain_colors'])]
            new_adata.uns['scanpy_colors'] = color_list[:len(new_adata.uns['scanpy_colors'])]
            self.color_reset = color_list
            self.gene_color_type = color_panel.five_gene_color[random.randint(0, 5)]
            draw_images()

            fig_path = './figures/spatialSlideseqV2'
            figures = os.listdir(fig_path)
            global image_list
            image_list = []
            for i in range(len(figures)):
                image_list.append(ttk.PhotoImage(file=fig_path + '/' + figures[i]))
                self.result_images = ttk.Label(self.show_frame, image=image_list[i])
                self.result_images.grid(row=i // 3, column=i % 3, sticky=W, pady=0)

        fig_path = './figures/spatialSlideseqV2'
        if not os.path.isdir(fig_path):
            os.mkdir(fig_path)

        draw_images()
        figures = os.listdir(fig_path)
        self.figure_Frame = ttk.Frame(self.right_panel)
        self.figure_Frame.pack(side=TOP, fill=X, expand=YES)

        self.canvas = ttk.Canvas(self.figure_Frame)
        self.show_frame = ttk.Frame(self.canvas)
        scoller()
        self.set_frame = ttk.Frame(self.figure_Frame, borderwidth=2, relief="sunken")
        self.set_frame.pack(side='right', expand=YES, anchor=N)
        self.set_frame_one = ttk.Frame(self.set_frame)
        self.set_frame_one.pack(side=TOP, expand=YES)
        self.set_frame_two = ttk.Frame(self.set_frame)
        self.set_frame_two.pack(side=TOP, expand=YES)

        self.Reset = ttk.Button(self.set_frame_one, text='Reset all colors', command=choose_color, width=12)
        self.Reset.grid(row=0, column=0, sticky=W, pady=0, padx=0)

        self.domain_color_label = ttk.Label(self.set_frame_one, text='Reset domain color: ', width=20)
        self.domain_color_label.grid(row=1, column=0, sticky=W, pady=0)

        self.Reset_domain_color = ttk.Entry(self.set_frame_one, width=10)
        self.Reset_domain_color.grid(row=1, column=1, sticky=W, pady=0)
        Tooltip(self.Reset_domain_color, "Input value must be int type and in [1:cluster_name]: 2")
        self.color_update = None

        def Reset_single_domain_color():
            from tkinter import colorchooser, filedialog
            colorvalue = colorchooser.askcolor()
            color = colorvalue[1]
            print(color)
            cluster = self.Reset_domain_color.get()
            print(cluster)
            adata.uns['louvain_colors'] = self.color_reset[:len(adata.uns['louvain_colors'])]
            adata.uns['louvain_colors'][int(cluster) - 1] = color
            new_adata.uns['scanpy_colors'] = self.color_reset[:len(new_adata.uns['scanpy_colors'])]
            new_adata.uns['scanpy_colors'][int(cluster) - 1] = color

            self.color_update = adata.uns['louvain_colors']
            import matplotlib.pyplot as plt
            plt.rcParams["figure.figsize"] = (3, 3)
            plt.rcParams['font.sans-serif'] = "Arial"
            i = 1
            # plt.rcParams["figure.figsize"] = (3, 3)
            # sc.pl.embedding(adata, basis="spatial", color="log1p_total_counts", title='Raw Data', s=6, show=False,
            #                 save='SlideseqV2/STAGATE_' + str(i) + '.png')
            i = i + 1
            plt.rcParams["figure.figsize"] = (3, 3)
            sc.pl.embedding(adata, basis="spatial", color="louvain", title='STAGATE', s=6, show=False,
                            save='SlideseqV2/STAGATE_' + str(i) + '.png')
            i = i + 1
            plt.rcParams["figure.figsize"] = (3, 3)
            sc.pl.umap(adata, color='louvain', title='STAGATE', show=False, s=6, save='STAGATE_' + str(i) + '.png')
            shutil.move('./figures/umapSTAGATE_' + str(i) + '.png',
                        './figures/spatialSlideseqV2/STAGATE_' + str(i) + '.png')
            i = i + 1
            plt.rcParams["figure.figsize"] = (3, 3)
            sc.pl.embedding(new_adata, basis="spatial", color="scanpy", title='SCANPY', s=6, show=False,
                            save='SlideseqV2/STAGATE_' + str(i) + '.png')
            i = i + 1
            plt.rcParams["figure.figsize"] = (3, 3)
            sc.pl.umap(new_adata, color='scanpy', title='SCANPY', show=False, s=6, save='STAGATE_' + str(i) + '.png')
            shutil.move('./figures/umapSTAGATE_' + str(i) + '.png',
                        './figures/spatialSlideseqV2/STAGATE_' + str(i) + '.png')

            figures = os.listdir(fig_path)
            global image_list
            image_list = []
            for i in range(len(figures)):
                img = Image.open(fig_path + '/' + figures[i])
                print(img.size)
                image_list.append(ttk.PhotoImage(file=fig_path + '/' + figures[i]))
                self.result_images = ttk.Label(self.show_frame, image=image_list[i])
                self.result_images.grid(row=i // 3, column=i % 3, sticky=W, pady=0, padx=0)
                s = i

            pass

        self.Reset_domain_btn = ttk.Button(self.set_frame_one, width=10, text='Confirm',
                                           command=Reset_single_domain_color)
        self.Reset_domain_btn.grid(row=1, column=2, sticky=W, pady=10)
        self.update_image_label = ttk.Label(self.set_frame_one, text='Reset image dpi: ', width=20)
        self.update_image_label.grid(row=2, column=0, sticky=W, pady=10)
        self.update_image_scale = ttk.Entry(self.set_frame_one, width=10)
        self.update_image_scale.grid(row=2, column=1, sticky=W, pady=10)
        Tooltip(self.update_image_scale, "Input value must be int type and >= 300: 300")

        def ipdate_hd():
            dpi = self.update_image_scale.get()
            print(dpi)
            print(self.color_update)
            adata.uns['louvain_colors'] = self.color_update
            new_adata.uns['scanpy_colors'] = self.color_update[:len(new_adata.uns['scanpy_colors'])]

            import matplotlib.pyplot as plt
            plt.rcParams["figure.figsize"] = (3, 3)
            sc.set_figure_params(dpi=dpi)
            draw_images()
            figures = os.listdir(fig_path)
            print(len(figures))

        self.Reset_domain_btn = ttk.Button(self.set_frame_one, width=10, text='Save', command=ipdate_hd)
        self.Reset_domain_btn.grid(row=2, column=2, sticky=W, pady=0)

        self.gene_visualization_label = ttk.Label(self.set_frame_one, text='Input gene name: ', width=20)
        self.gene_visualization_label.grid(row=3, column=0, sticky=W, pady=0)

        self.gene_visualization_entry = ttk.Entry(self.set_frame_one, width=10)
        self.gene_visualization_entry.grid(row=3, column=1, sticky=W, pady=0)
        Tooltip(self.gene_visualization_entry, f"Gene name: {adata.var.index[:1]}")

        def gene_visualization():
            try:
                gene = self.gene_visualization_entry.get()
                if gene in adata.var.index:
                    sc.pl.spatial(adata, img_key="hires", color=gene, title="$" + gene + "$", show=False,
                                  save=gene + '.png', sopt_size=50)
                    global img0
                    photo = Image.open('figures/show' + gene + '.png')
                    img0 = ImageTk.PhotoImage(photo)
                    img1 = ttk.Label(self.set_frame_two, image=img0)
                    img1.grid(row=0, column=0, sticky=W, pady=0)
                    if os.path.exists('figures/show' + gene + '.png'):
                        os.remove('figures/show' + gene + '.png')
                    else:
                        Messagebox.show_error('Error', message="image is no exist!")
            except:
                Messagebox.show_error("Python Error", "Make sure gene name in dataset")

        self.gene_visualization_btn = ttk.Button(self.set_frame_one, width=10, command=gene_visualization, text='Show')
        self.gene_visualization_btn.grid(row=3, column=2, sticky=W, pady=0)

        global image_list
        image_list = []
        s = 0
        for i in range(len(figures)):
            image_list.append(ttk.PhotoImage(file=fig_path + '/' + figures[i]))
            self.result_images = ttk.Label(self.show_frame, image=image_list[i])
            self.result_images.grid(row=i // 3, column=i % 3, sticky=W, pady=0)
            s = i

        def Reset():
            python = sys.executable
            os.execl(python, python, *sys.argv)
            pass

        self.Reset = ttk.Button(self.set_frame_one, text='Reset STABox', command=Reset, width=12)
        self.Reset.grid(row=0, column=1, sticky=W, pady=0)

        self.pb.stop()
        self.setvar('prog-message', 'STAGATE run over!')
        self.setvar('End-time', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        EndTime = datetime.now().replace(microsecond=0)
        self.setvar('total-time-cost', EndTime - self.StartTime)

    def STAGATE_Thread_two(self):
        T = threading.Thread(target=self.STAGATE_Slideseq_data_analysis)
        T.setDaemon(True)
        T.start()

    def STAGATE_Stereoseq_data_analysis(self):
        self.Stereoseq_data_process()
        adata = self.result_queue.get()
        new_adata = adata.copy()
        Cal_Spatial_Net(adata, rad_cutoff=self.rad_cutoff_value)
        Stats_Spatial_Net(adata)
        train_STAGATE(adata)
        sc.pp.neighbors(adata, use_rep='STAGATE')
        sc.tl.umap(adata)
        sc.tl.louvain(adata, resolution=self.alpha_value)
        adata.obsm["spatial"] = adata.obsm["spatial"] * (-1)

        sc.pp.pca(new_adata, n_comps=30)
        sc.pp.neighbors(new_adata, use_rep='X_pca')
        sc.tl.louvain(new_adata, resolution=self.alpha_value, key_added='scanpy')
        sc.tl.umap(new_adata)
        self.gene_color_type = 'viridis'

        def draw_images():
            from matplotlib import pyplot as plt
            plt.rcParams['font.sans-serif'] = "Arial"
            i = 1
            plt.rcParams["figure.figsize"] = (3, 3)
            sc.pl.embedding(adata, basis="spatial", color="log1p_total_counts", title='Raw Data', s=6, show=False,
                            save='Stereseq/STAGATE_' + str(i) + '.png', color_map=self.gene_color_type)
            i = i + 1
            plt.rcParams["figure.figsize"] = (3, 3)
            sc.pl.embedding(adata, basis="spatial", color="louvain", title='STAGATE', s=6, show=False,
                            save='Stereseq/STAGATE_' + str(i) + '.png')
            i = i + 1
            plt.rcParams["figure.figsize"] = (3, 3)
            sc.pl.umap(adata, color='louvain', title='STAGATE', show=False, s=6, save='STAGATE_' + str(i) + '.png')
            shutil.move('./figures/umapSTAGATE_' + str(i) + '.png',
                        './figures/spatialStereseq/STAGATE_' + str(i) + '.png')
            i = i + 1
            plt.rcParams["figure.figsize"] = (3, 3)
            sc.pl.embedding(new_adata, basis="spatial", color="scanpy", title='SCANPY', s=6, show=False,
                            save='SlideseqV2/STAGATE_' + str(i) + '.png')
            i = i + 1
            plt.rcParams["figure.figsize"] = (3, 3)
            sc.pl.umap(new_adata, color='scanpy', title='SCANPY', show=False, s=6, save='STAGATE_' + str(i) + '.png')
            shutil.move('./figures/umapSTAGATE_' + str(i) + '.png',
                        './figures/spatialStereseq/STAGATE_' + str(i) + '.png')

        def scoller():
            self.figure_ybar = ttk.Scrollbar(self.figure_Frame, orient=VERTICAL, cursor='draft_small')
            self.figure_ybar.pack(side=RIGHT, fill=Y)
            self.figure_ybar.config(command=self.canvas.yview)

            self.figure_xbar = ttk.Scrollbar(self.figure_Frame, orient=HORIZONTAL, cursor='draft_small')
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
            self.canvas.configure(scrollregion=self.canvas.bbox("all"), width=1000, height=650)

        def choose_color():
            colorvalue = colorchooser.askcolor()
            color_list = []
            color = colorvalue[1]
            color_lens = [adata.uns['louvain_colors'], new_adata.uns['scanpy_colors']]
            max_len = max(color_lens, key=len)
            if len(max_len) > self.cluster_value:
                self.cluster_value = len(max_len)
            if color[1:3] == 'ff':
                color_list.append(color)
                list = random.sample(color_panel.FF_color, self.cluster_value)
                color_list = color_list + list
            elif color[1:3] == 'cc':
                color_list.append(color)
                list = random.sample(color_panel.CC_color, self.cluster_value)
                color_list = color_list + list
            elif color[1:3] == '99':
                color_list.append(color)
                list = random.sample(color_panel.NN_color, self.cluster_value)
                color_list = color_list + list
            elif color[1:3] == '66':
                color_list.append(color)
                list = random.sample(color_panel.SS_color, self.cluster_value)
                color_list = color_list + list
            elif color[1:3] == '33':
                color_list.append(color)
                list = random.sample(color_panel.TT_color, self.cluster_value)
                color_list = color_list + list
            elif color[1:3] == '00':
                color_list.append(color)
                list = random.sample(color_panel.ZZ_color, self.cluster_value)
                color_list = color_list + list
            else:
                color_list.append(color)
                list = random.sample(color_panel.random_color, self.cluster_value)
                color_list = color_list + list

            print(color_list)
            adata.uns['louvain_colors'] = color_list[:len(adata.uns['louvain_colors'])]
            new_adata.uns['scanpy_colors'] = color_list[:len(new_adata.uns['scanpy_colors'])]
            self.color_reset = color_list
            self.gene_color_type = color_panel.five_gene_color[random.randint(0, 5)]
            draw_images()

            fig_path = './figures/spatialStereseq'
            figures = os.listdir(fig_path)
            global image_list
            image_list = []
            for i in range(len(figures)):
                image_list.append(ttk.PhotoImage(file=fig_path + '/' + figures[i]))
                self.result_images = ttk.Label(self.show_frame, image=image_list[i])
                self.result_images.grid(row=i // 3, column=i % 3, sticky=W, pady=0)

        fig_path = './figures/spatialStereseq'
        if not os.path.isdir(fig_path):
            os.mkdir(fig_path)

        draw_images()
        figures = os.listdir(fig_path)
        self.figure_Frame = ttk.Frame(self.right_panel)
        self.figure_Frame.pack(side=TOP, fill=X, expand=YES)

        self.canvas = ttk.Canvas(self.figure_Frame)
        self.show_frame = ttk.Frame(self.canvas)
        scoller()
        self.set_frame = ttk.Frame(self.figure_Frame, borderwidth=2, relief="sunken")
        self.set_frame.pack(side='right', expand=YES, anchor=N)
        self.set_frame_one = ttk.Frame(self.set_frame)
        self.set_frame_one.pack(side=TOP, expand=YES)
        self.set_frame_two = ttk.Frame(self.set_frame)
        self.set_frame_two.pack(side=TOP, expand=YES)

        self.Reset = ttk.Button(self.set_frame_one, text='Reset all colors', command=choose_color, width=12)
        self.Reset.grid(row=0, column=0, sticky=W, pady=0, padx=0)

        self.domain_color_label = ttk.Label(self.set_frame_one, text='Reset domain color: ', width=20)
        self.domain_color_label.grid(row=1, column=0, sticky=W, pady=0)

        self.Reset_domain_color = ttk.Entry(self.set_frame_one, width=10)
        self.Reset_domain_color.grid(row=1, column=1, sticky=W, pady=0)
        Tooltip(self.Reset_domain_color, "Input value must be int type and in [1:cluster_name]: 2")
        self.color_update = None

        def Reset_single_domain_color():
            from tkinter import colorchooser
            colorvalue = colorchooser.askcolor()
            color = colorvalue[1]
            cluster = self.Reset_domain_color.get()
            adata.uns['louvain_colors'] = self.color_reset[:len(adata.uns['louvain_colors'])]
            adata.uns['louvain_colors'][int(cluster) - 1] = color
            new_adata.uns['scanpy_colors'] = self.color_reset[:len(new_adata.uns['scanpy_colors'])]
            new_adata.uns['scanpy_colors'][int(cluster) - 1] = color

            self.color_update = adata.uns['louvain_colors']
            import matplotlib.pyplot as plt
            plt.rcParams["figure.figsize"] = (3, 3)
            plt.rcParams['font.sans-serif'] = "Arial"
            i = 1
            # plt.rcParams["figure.figsize"] = (3, 3)
            # sc.pl.embedding(adata, basis="spatial", color="log1p_total_counts", title='Raw Data', s=6, show=False,
            #                 save='SlideseqV2/STAGATE_' + str(i) + '.png')
            i = i + 1
            plt.rcParams["figure.figsize"] = (3, 3)
            sc.pl.embedding(adata, basis="spatial", color="louvain", title='STAGATE', s=6, show=False,
                            save='SlideseqV2/STAGATE_' + str(i) + '.png')
            i = i + 1
            plt.rcParams["figure.figsize"] = (3, 3)
            sc.pl.umap(adata, color='louvain', title='STAGATE', show=False, s=6, save='STAGATE_' + str(i) + '.png')
            shutil.move('./figures/umapSTAGATE_' + str(i) + '.png',
                        './figures/spatialSlideseqV2/STAGATE_' + str(i) + '.png')
            i = i + 1
            plt.rcParams["figure.figsize"] = (3, 3)
            sc.pl.embedding(new_adata, basis="spatial", color="scanpy", title='SCANPY', s=6, show=False,
                            save='SlideseqV2/STAGATE_' + str(i) + '.png')
            i = i + 1
            plt.rcParams["figure.figsize"] = (3, 3)
            sc.pl.umap(new_adata, color='scanpy', title='SCANPY', show=False, s=6, save='STAGATE_' + str(i) + '.png')
            shutil.move('./figures/umapSTAGATE_' + str(i) + '.png',
                        './figures/spatialSlideseqV2/STAGATE_' + str(i) + '.png')

            figures = os.listdir(fig_path)
            global image_list
            image_list = []
            for i in range(len(figures)):
                img = Image.open(fig_path + '/' + figures[i])
                print(img.size)
                image_list.append(ttk.PhotoImage(file=fig_path + '/' + figures[i]))
                self.result_images = ttk.Label(self.show_frame, image=image_list[i])
                self.result_images.grid(row=i // 3, column=i % 3, sticky=W, pady=0, padx=0)
                s = i

        self.Reset_domain_btn = ttk.Button(self.set_frame_one, width=10, text='Confirm',
                                           command=Reset_single_domain_color)
        self.Reset_domain_btn.grid(row=1, column=2, sticky=W, pady=10)
        self.update_image_label = ttk.Label(self.set_frame_one, text='Reset image dpi: ', width=20)
        self.update_image_label.grid(row=2, column=0, sticky=W, pady=10)
        self.update_image_scale = ttk.Entry(self.set_frame_one, width=10)
        self.update_image_scale.grid(row=2, column=1, sticky=W, pady=10)
        Tooltip(self.update_image_scale, "Input value must be int type and >= 300: 300")

        def ipdate_hd():
            dpi = self.update_image_scale.get()
            adata.uns['louvain_colors'] = self.color_update
            new_adata.uns['scanpy_colors'] = self.color_update[:len(new_adata.uns['scanpy_colors'])]

            import matplotlib.pyplot as plt
            plt.rcParams["figure.figsize"] = (3, 3)
            sc.set_figure_params(dpi=dpi)
            draw_images()

        self.Reset_domain_btn = ttk.Button(self.set_frame_one, width=10, text='Save', command=ipdate_hd)
        self.Reset_domain_btn.grid(row=2, column=2, sticky=W, pady=0)

        self.gene_visualization_label = ttk.Label(self.set_frame_one, text='Input gene name: ', width=20)
        self.gene_visualization_label.grid(row=3, column=0, sticky=W, pady=0)

        self.gene_visualization_entry = ttk.Entry(self.set_frame_one, width=10)
        self.gene_visualization_entry.grid(row=3, column=1, sticky=W, pady=0)
        Tooltip(self.gene_visualization_entry, f"Gene name: {adata.var.index[:2]}")

        def gene_visualization():
            try:
                gene = self.gene_visualization_entry.get()
                print(adata.var.index)
                if gene not in adata.var.index:
                    gene = adata.var.index[-1]
                sc.pl.spatial(adata, img_key="hires", color=gene, title="$" + gene + "$", show=False,
                              save=gene + '.png', sopt_size=50)
                global img0
                photo = Image.open('figures/show' + gene + '.png')
                img0 = ImageTk.PhotoImage(photo)
                img1 = ttk.Label(self.set_frame_two, image=img0)
                img1.grid(row=0, column=0, sticky=W, pady=0)
                if os.path.exists('figures/show' + gene + '.png'):
                    os.remove('figures/show' + gene + '.png')
                else:
                    Messagebox.show_error('Error', message="image is no exist!")
                pass
            except:
                Messagebox.show_error("Gene Error", "Make sure gene name in dataset")

        self.gene_visualization_btn = ttk.Button(self.set_frame_one, width=10, command=gene_visualization, text='Show')
        self.gene_visualization_btn.grid(row=3, column=2, sticky=W, pady=0)

        # end
        global image_list
        image_list = []
        s = 0
        for i in range(len(figures)):
            image_list.append(ttk.PhotoImage(file=fig_path + '/' + figures[i]))
            self.result_images = ttk.Label(self.show_frame, image=image_list[i])
            self.result_images.grid(row=i // 3, column=i % 3, sticky=W, pady=0)
            s = i

        def Reset():
            python = sys.executable
            os.execl(python, python, *sys.argv)
            pass

        self.Reset = ttk.Button(self.set_frame_one, text='Reset STABox', command=Reset, width=12)
        self.Reset.grid(row=0, column=1, sticky=W, pady=0)

        self.pb.stop()
        self.setvar('prog-message', 'STAGATE run over!')
        self.setvar('End-time', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        EndTime = datetime.now().replace(microsecond=0)
        self.setvar('total-time-cost', EndTime - self.StartTime)

    def STAGATE_Thread_three(self):
        T = threading.Thread(target=self.STAGATE_Stereoseq_data_analysis)
        # 设置线程为守护线程，防止退出主线程时，子线程仍在运行
        T.setDaemon(True)
        T.start()

    def STAGATE_STARmap_Process(self):
        h5ad_file = glob.glob(self.file_path + '/*.h5ad')
        if os.path.exists(self.file_path) and len(h5ad_file) != 0:
            self.setvar('data_name', 'STARmap Data')
            adata = sc.read(h5ad_file[0])
            adata.var_names_make_unique()
            fsize = os.path.getsize(h5ad_file[0])
            self.setvar('data_size', str(round(fsize / 1024 / 1024, 2)) + 'MB')
            self.setvar('data_shape', str(adata.shape[0]) + '×' + str(adata.shape[1]))
            sc.pp.calculate_qc_metrics(adata, inplace=True)
            sc.pp.filter_genes(adata, min_cells=50)
            sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
            self.result_queue.put(adata)
        else:
            Messagebox.show_error(title="File error", message="Make sure only one h5ad file in path!")

    def STAGATE_STARmap_analysis(self):
        self.STAGATE_STARmap_Process()
        adata = self.result_queue.get()
        new_adata = adata.copy()
        Cal_Spatial_Net(adata, rad_cutoff=self.rad_cutoff_value)
        Stats_Spatial_Net(adata)
        train_STAGATE(adata)
        sc.pp.neighbors(adata, use_rep='STAGATE')
        sc.tl.umap(adata)
        sc.tl.louvain(adata, resolution=self.alpha_value)
        sc.pp.neighbors(adata, use_rep='STAGATE')
        sc.tl.umap(adata)
        adata = mclust_R(adata, used_obsm='STAGATE', num_cluster=self.cluster_value)

        sc.pp.pca(new_adata, n_comps=30)
        sc.pp.neighbors(new_adata, use_rep='X_pca')
        sc.tl.louvain(new_adata, resolution=self.alpha_value, key_added='scanpy')
        sc.tl.umap(new_adata)
        self.gene_color_type = 'viridis'

        def draw_images():
            from matplotlib import pyplot as plt
            plt.rcParams['font.sans-serif'] = "Arial"
            i = 1
            plt.rcParams["figure.figsize"] = (3, 3)
            sc.pl.embedding(adata, basis="spatial", color="log1p_total_counts", title='Raw Data', s=6, show=False,
                            save='STARmap/STAGATE_' + str(i) + '.png', color_map=self.gene_color_type)
            i = i + 1
            plt.rcParams["figure.figsize"] = (3, 3)
            sc.pl.embedding(adata, basis="spatial", color="louvain", title='STAGATE', s=6, show=False,
                            save='STARmap/STAGATE_' + str(i) + '.png')
            i = i + 1
            plt.rcParams["figure.figsize"] = (3, 3)
            sc.pl.embedding(adata, basis="spatial", color="mclust", title='STAGATE', s=6, show=False,
                            save='STARmap/STAGATE_' + str(i) + '.png')
            i = i + 1
            plt.rcParams["figure.figsize"] = (3, 3)
            sc.pl.umap(adata, color='louvain', title='STAGATE-louvain', show=False, s=6,
                       save='STAGATE_' + str(i) + '.png')
            shutil.move('./figures/umapSTAGATE_' + str(i) + '.png',
                        './figures/spatialSTARmap/STAGATE_' + str(i) + '.png')
            i = i + 1
            plt.rcParams["figure.figsize"] = (3, 3)
            sc.pl.umap(adata, color='mclust', title='STAGATE-mclust', show=False, s=6,
                       save='STAGATE_' + str(i) + '.png')
            shutil.move('./figures/umapSTAGATE_' + str(i) + '.png',
                        './figures/spatialSTARmap/STAGATE_' + str(i) + '.png')
            i = i + 1
            plt.rcParams["figure.figsize"] = (3, 3)
            sc.pl.embedding(new_adata, basis="spatial", color="scanpy", title='SCANPY', s=6, show=False,
                            save='STARmap/STAGATE_' + str(i) + '.png')
            i = i + 1
            plt.rcParams["figure.figsize"] = (3, 3)
            sc.pl.umap(new_adata, color='scanpy', title='SCANPY', show=False, s=6, save='STAGATE_' + str(i) + '.png')
            shutil.move('./figures/umapSTAGATE_' + str(i) + '.png',
                        './figures/spatialSTARmap/STAGATE_' + str(i) + '.png')

        def scoller():
            self.figure_ybar = ttk.Scrollbar(self.figure_Frame, orient=VERTICAL, cursor='draft_small')
            self.figure_ybar.pack(side=RIGHT, fill=Y)
            self.figure_ybar.config(command=self.canvas.yview)

            self.figure_xbar = ttk.Scrollbar(self.figure_Frame, orient=HORIZONTAL, cursor='draft_small')
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
            self.canvas.configure(scrollregion=self.canvas.bbox("all"), width=1000, height=650)

        def choose_color():
            colorvalue = colorchooser.askcolor()
            color_list = []
            color = colorvalue[1]
            if color[1:3] == 'ff':
                color_list.append(color)
                list = random.sample(color_panel.FF_color, self.cluster_value)
                color_list = color_list + list
            elif color[1:3] == 'cc':
                color_list.append(color)
                list = random.sample(color_panel.CC_color, self.cluster_value)
                color_list = color_list + list
            elif color[1:3] == '99':
                color_list.append(color)
                list = random.sample(color_panel.NN_color, self.cluster_value)
                color_list = color_list + list
            elif color[1:3] == '66':
                color_list.append(color)
                list = random.sample(color_panel.SS_color, self.cluster_value)
                color_list = color_list + list
            elif color[1:3] == '33':
                color_list.append(color)
                list = random.sample(color_panel.TT_color, self.cluster_value)
                color_list = color_list + list
            elif color[1:3] == '00':
                color_list.append(color)
                list = random.sample(color_panel.ZZ_color, self.cluster_value)
                color_list = color_list + list
            else:
                color_list.append(color)
                list = random.sample(color_panel.random_color, self.cluster_value)
                color_list = color_list + list

            print(color_list)
            adata.uns['louvain_colors'] = color_list[:len(adata.uns['louvain_colors'])]
            new_adata.uns['scanpy_colors'] = color_list[:len(new_adata.uns['scanpy_colors'])]
            adata.uns['mclust_colors'] = color_list[:len(adata.uns['mclust_colors'])]
            self.color_reset = color_list
            self.gene_color_type = color_panel.five_gene_color[random.randint(0, 5)]
            draw_images()

            fig_path = './figures/spatialSTARmap'
            figures = os.listdir(fig_path)
            global image_list
            image_list = []
            for i in range(len(figures)):
                image_list.append(ttk.PhotoImage(file=fig_path + '/' + figures[i]))
                self.result_images = ttk.Label(self.show_frame, image=image_list[i])
                self.result_images.grid(row=i // 3, column=i % 3, sticky=W, pady=0)

        fig_path = './figures/spatialSTARmap'
        if not os.path.isdir(fig_path):
            os.mkdir(fig_path)

        draw_images()
        figures = os.listdir(fig_path)
        self.figure_Frame = ttk.Frame(self.right_panel)
        self.figure_Frame.pack(side=TOP, fill=X, expand=YES)

        self.canvas = ttk.Canvas(self.figure_Frame)
        self.show_frame = ttk.Frame(self.canvas)
        scoller()
        self.set_frame = ttk.Frame(self.figure_Frame, borderwidth=2, relief="sunken")
        self.set_frame.pack(side='right', expand=YES, anchor=N)
        self.set_frame_one = ttk.Frame(self.set_frame)
        self.set_frame_one.pack(side=TOP, expand=YES)
        self.set_frame_two = ttk.Frame(self.set_frame)
        self.set_frame_two.pack(side=TOP, expand=YES)

        self.Reset = ttk.Button(self.set_frame_one, text='Reset all colors', command=choose_color, width=12)
        self.Reset.grid(row=0, column=0, sticky=W, pady=0, padx=0)

        self.domain_color_label = ttk.Label(self.set_frame_one, text='Reset domain color: ', width=20)
        self.domain_color_label.grid(row=1, column=0, sticky=W, pady=0)

        self.Reset_domain_color = ttk.Entry(self.set_frame_one, width=10)
        self.Reset_domain_color.grid(row=1, column=1, sticky=W, pady=0)
        Tooltip(self.Reset_domain_color, "Input value must be int type and in [1:cluster_name]: 2")
        self.color_update = None

        def Reset_single_domain_color():
            from tkinter import colorchooser
            colorvalue = colorchooser.askcolor()
            color = colorvalue[1]
            cluster = self.Reset_domain_color.get()
            adata.uns['louvain_colors'] = self.color_reset
            adata.uns['louvain_colors'][int(cluster) - 1] = color
            adata.uns['mclust_colors'] = self.color_reset
            adata.uns['mclust_colors'][int(cluster) - 1] = color

            new_adata.uns['scanpy_colors'] = self.color_reset[:len(new_adata.uns['scanpy_colors'])]
            new_adata.uns['scanpy_colors'][int(cluster) - 1] = color

            self.color_update = adata.uns['louvain_colors']
            import matplotlib.pyplot as plt
            plt.rcParams["figure.figsize"] = (3, 3)
            plt.rcParams['font.sans-serif'] = "Arial"
            i = 1
            # plt.rcParams["figure.figsize"] = (3, 3)
            # sc.pl.embedding(adata, basis="spatial", color="log1p_total_counts", title='Raw Data', s=6, show=False,
            #                 save='SlideseqV2/STAGATE_' + str(i) + '.png')
            i = i + 1
            plt.rcParams["figure.figsize"] = (3, 3)
            sc.pl.embedding(adata, basis="spatial", color="louvain", title='STAGATE-louvain', s=6, show=False,
                            save='STARmap/STAGATE_' + str(i) + '.png')
            i = i + 1
            plt.rcParams["figure.figsize"] = (3, 3)
            sc.pl.embedding(adata, basis="spatial", color="mclust", title='STAGATE-mclust', s=6, show=False,
                            save='STARmap/STAGATE_' + str(i) + '.png')
            i = i + 1
            plt.rcParams["figure.figsize"] = (3, 3)
            sc.pl.umap(adata, color='louvain', title='STAGATE', show=False, s=6, save='STAGATE_' + str(i) + '.png')
            shutil.move('./figures/umapSTAGATE_' + str(i) + '.png',
                        './figures/spatialSTARmap/STAGATE_' + str(i) + '.png')
            i = i + 1
            plt.rcParams["figure.figsize"] = (3, 3)
            sc.pl.umap(adata, color='mclust', title='STAGATE', show=False, s=6, save='STAGATE_' + str(i) + '.png')
            shutil.move('./figures/umapSTAGATE_' + str(i) + '.png',
                        './figures/spatialSTARmap/STAGATE_' + str(i) + '.png')
            i = i + 1
            plt.rcParams["figure.figsize"] = (3, 3)
            sc.pl.embedding(new_adata, basis="spatial", color="scanpy", title='SCANPY', s=6, show=False,
                            save='STARmap/STAGATE_' + str(i) + '.png')
            i = i + 1
            plt.rcParams["figure.figsize"] = (3, 3)
            sc.pl.umap(new_adata, color='scanpy', title='SCANPY', show=False, s=6, save='STAGATE_' + str(i) + '.png')
            shutil.move('./figures/umapSTAGATE_' + str(i) + '.png',
                        './figures/spatialSTARmap/STAGATE_' + str(i) + '.png')

            figures = os.listdir(fig_path)
            global image_list
            image_list = []
            for i in range(len(figures)):
                img = Image.open(fig_path + '/' + figures[i])
                print(img.size)
                image_list.append(ttk.PhotoImage(file=fig_path + '/' + figures[i]))
                self.result_images = ttk.Label(self.show_frame, image=image_list[i])
                self.result_images.grid(row=i // 3, column=i % 3, sticky=W, pady=0, padx=0)
                s = i

        self.Reset_domain_btn = ttk.Button(self.set_frame_one, width=10, text='Confirm',
                                           command=Reset_single_domain_color)
        self.Reset_domain_btn.grid(row=1, column=2, sticky=W, pady=10)
        self.update_image_label = ttk.Label(self.set_frame_one, text='Reset image dpi: ', width=20)
        self.update_image_label.grid(row=2, column=0, sticky=W, pady=10)
        self.update_image_scale = ttk.Entry(self.set_frame_one, width=10)
        self.update_image_scale.grid(row=2, column=1, sticky=W, pady=10)
        Tooltip(self.update_image_scale, "Input value must be int type and >= 300: 300")

        def ipdate_hd():
            dpi = self.update_image_scale.get()
            print(dpi)
            print(self.color_update)
            adata.uns['louvain_colors'] = self.color_update
            new_adata.uns['scanpy_colors'] = self.color_update[:len(new_adata.uns['scanpy_colors'])]

            import matplotlib.pyplot as plt
            plt.rcParams["figure.figsize"] = (3, 3)
            sc.set_figure_params(dpi=dpi)
            draw_images()
            figures = os.listdir(fig_path)
            print(len(figures))

        self.Reset_domain_btn = ttk.Button(self.set_frame_one, width=10, text='Save', command=ipdate_hd)
        self.Reset_domain_btn.grid(row=2, column=2, sticky=W, pady=0)

        self.gene_visualization_label = ttk.Label(self.set_frame_one, text='Input gene name: ', width=20)
        self.gene_visualization_label.grid(row=3, column=0, sticky=W, pady=0)

        self.gene_visualization_entry = ttk.Entry(self.set_frame_one, width=10)
        self.gene_visualization_entry.grid(row=3, column=1, sticky=W, pady=0)
        Tooltip(self.gene_visualization_entry, "Gene name: NEFH/ATP2B4")

        def gene_visualization():
            try:
                gene = self.gene_visualization_entry.get()
                if gene not in adata.var.index:
                    gene = adata.var.index[-1]
                sc.pl.spatial(adata, img_key="hires", color=gene, title="$" + gene + "$", show=False,
                              save=gene + '.png', sopt_size=50)
                global img0
                photo = Image.open('figures/show' + gene + '.png')
                img0 = ImageTk.PhotoImage(photo)
                img1 = ttk.Label(self.set_frame_two, image=img0)
                img1.grid(row=0, column=0, sticky=W, pady=0)
                if os.path.exists('figures/show' + gene + '.png'):
                    os.remove('figures/show' + gene + '.png')
                else:
                    Messagebox.show_error('Error', message="image is no exist!")
                pass
            except:
                Messagebox.show_error("Python Error", "Make sure gene name in dataset")

        self.gene_visualization_btn = ttk.Button(self.set_frame_one, width=10, command=gene_visualization, text='Show')
        self.gene_visualization_btn.grid(row=3, column=2, sticky=W, pady=0)

        # end
        global image_list
        image_list = []
        s = 0
        for i in range(len(figures)):
            image_list.append(ttk.PhotoImage(file=fig_path + '/' + figures[i]))
            self.result_images = ttk.Label(self.show_frame, image=image_list[i])
            self.result_images.grid(row=i // 3, column=i % 3, sticky=W, pady=0)
            s = i

        def Reset():
            python = sys.executable
            os.execl(python, python, *sys.argv)
            pass

        self.Reset = ttk.Button(self.set_frame_one, text='Reset STABox', command=Reset, width=12)
        self.Reset.grid(row=0, column=1, sticky=W, pady=0)

        self.pb.stop()
        self.setvar('prog-message', 'STAGATE run over!')
        self.setvar('End-time', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        EndTime = datetime.now().replace(microsecond=0)
        self.setvar('total-time-cost', EndTime - self.StartTime)
        pass

    def STAGATE_Thread_four(self):
        T = threading.Thread(target=self.STAGATE_STARmap_analysis)
        # 设置线程为守护线程，防止退出主线程时，子线程仍在运行
        T.setDaemon(True)
        T.start()

    def STAGATE_3Ddomain_process(self):
        try:
            files = os.listdir(self.Visium_file_path)
            fsize_one = os.path.getsize(os.path.join(self.Visium_file_path, files[0]))
            fsize_two = os.path.getsize(os.path.join(self.Visium_file_path, files[1]))

            if fsize_one > fsize_two:
                exprefile = files[0]
                coorfile = files[1]
            else:
                exprefile = files[1]
                coorfile = files[0]

            self.data_name = exprefile.rsplit('.', 1)[-2]
            self.setvar('data_name', self.data_name)
            data = pd.read_csv(os.path.join(self.Visium_file_path, exprefile), sep='\t', index_col=0)
            Aligned_coor = pd.read_csv(os.path.join(self.Visium_file_path, coorfile), sep='\t', index_col=0)
            adata = sc.AnnData(data)
            fsize = os.path.getsize(os.path.join(self.Visium_file_path, exprefile))
            self.data_size = round(fsize / 1024 / 1024, 2)
            self.setvar('data_size', str(self.data_size) + 'MB')
            self.data_shape = adata.shape
            self.setvar('data_shape', str(self.data_shape[0]) + '×' + str(self.data_shape[1]))
            sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
            adata.obs['X'] = Aligned_coor.loc[adata.obs_names, 'X']
            adata.obs['Y'] = Aligned_coor.loc[adata.obs_names, 'Y']
            adata.obs['Z'] = Aligned_coor.loc[adata.obs_names, 'Z']
            adata.obs['Section_id'] = Aligned_coor.loc[adata.obs_names, 'Section']
            adata.obsm['spatial'] = adata.obs.loc[:, ['X', 'Y']].values
            self.result_queue.put(adata)
        except:
            Messagebox.show_info('Check whether the 3D Data files exists!')

    def STAGATE_3Ddomain_analysis(self):
        self.STAGATE_3Ddomain_process()
        adata = self.result_queue.get()
        section_order = ['Puck_180531_13', 'Puck_180531_16', 'Puck_180531_17',
                         'Puck_180531_18', 'Puck_180531_19', 'Puck_180531_22',
                         'Puck_180531_23']
        # self.rad_cutoff_value = 50
        Cal_Spatial_Net_3D(adata, rad_cutoff_2D=self.rad_cutoff_value, rad_cutoff_Zaxis=50, key_section='Section_id',
                           section_order=section_order, verbose=True)
        adata = train_STAGATE(adata)
        sc.pp.neighbors(adata, use_rep='STAGATE')
        sc.tl.umap(adata)
        adata = mclust_R(adata, used_obsm='STAGATE', num_cluster=self.cluster_value)

        data_save_path = '../H5AD_Save/STAGATE_3DSpatialData'
        if not os.path.exists(data_save_path):
            os.makedirs(data_save_path)
            adata.write_h5ad(os.path.join(data_save_path, '3DSpatialData.h5ad'))
            json_file = {
                "Coordinate": "spatial3D",
                "Annotatinos": ["mclust"],
                "Meshes": {
                },
                "mesh_coord": "fixed.json",
                "Genes": [
                    "all"
                ]
            }
            fixed_file = {"xmin": 0, "ymin": 0, "margin": 0, "zmin": 0, "binsize": 1}
            json.dump(json_file, open(os.path.join(data_save_path, '3DSpatialData.json'), 'w'), indent=4)
            json.dump(fixed_file, open(os.path.join(data_save_path, 'fixed.json'), 'w'))
            self.web_Thread(data_save_path, '3DSpatialData.h5ad', '3DSpatialData.json', '3D_HippoData')

        # 2D
        adata_2D = adata.copy()
        adata_2D.uns['Spatial_Net'] = adata.uns['Spatial_Net_2D'].copy()
        adata_2D = train_STAGATE(adata_2D)
        sc.pp.neighbors(adata_2D, use_rep='STAGATE')
        sc.tl.umap(adata_2D)
        adata = mclust_R(adata_2D, used_obsm='STAGATE', num_cluster=self.cluster_value)
        section_colors = ['#02899A', '#0E994D', '#86C049', '#FBB21F', '#F48022', '#DA5326', '#BA3326']

        def draw_images(sdata, adata_2D, fig_path, section_colors):
            from matplotlib import pyplot as plt
            plt.rcParams['font.sans-serif'] = "Arial"
            i = 1
            fig = plt.figure(figsize=(4, 4))
            ax1 = plt.axes(projection='3d')
            for it, label in enumerate(np.unique(adata.obs['Section_id'])):
                temp_Coor = adata.obs.loc[adata.obs['Section_id'] == label, :]
                temp_xd = temp_Coor['X']
                temp_yd = temp_Coor['Y']
                temp_zd = temp_Coor['Z']
                ax1.scatter3D(temp_xd, temp_yd, temp_zd, c=section_colors[it], s=0.2, marker="o", label=label)

            ax1.set_xlabel('')
            ax1.set_ylabel('')
            ax1.set_zlabel('')

            ax1.set_xticklabels([])
            ax1.set_yticklabels([])
            ax1.set_zticklabels([])

            plt.legend(bbox_to_anchor=(1, 0.8), markerscale=10, frameon=False)
            plt.title('Ground Truth')

            ax1.elev = 45
            ax1.azim = -20
            plt.savefig(fig_path + '/STAGATE_' + str(i) + '.png')
            plt.close()

            i = i + 1
            plt.rcParams["figure.figsize"] = (3, 3)
            sc.pl.umap(sdata, color='mclust', title='mclust(STAGATE-3D)', show=False, s=6,
                       save='STAGATE_' + str(i) + '.png')
            shutil.move('./figures/umapSTAGATE_' + str(i) + '.png', fig_path + '/STAGATE_' + str(i) + '.png')

            i = i + 1
            plt.rcParams["figure.figsize"] = (3, 3)
            sc.pl.umap(sdata, color='Section_id', title='Section_id(STAGATE-3D)', show=False, s=6,
                       save='STAGATE_' + str(i) + '.png')
            shutil.move('./figures/umapSTAGATE_' + str(i) + '.png', fig_path + '/STAGATE_' + str(i) + '.png')

            i = i + 1
            fig = plt.figure(figsize=(4, 4))
            ax1 = plt.axes(projection='3d')
            for it, label in enumerate(np.unique(adata.obs['mclust'])):
                temp_Coor = adata.obs.loc[adata.obs['mclust'] == label, :]
                temp_xd = temp_Coor['X']
                temp_yd = temp_Coor['Y']
                temp_zd = temp_Coor['Z']
                ax1.scatter3D(temp_xd, temp_yd, temp_zd, c=adata.uns['mclust_colors'][it], s=0.2, marker="o",
                              label=label)

            ax1.set_xlabel('')
            ax1.set_ylabel('')
            ax1.set_zlabel('')

            ax1.set_xticklabels([])
            ax1.set_yticklabels([])
            ax1.set_zticklabels([])

            plt.legend(bbox_to_anchor=(1.2, 0.8), markerscale=10, frameon=False)
            plt.title('STAGATE-3D')

            ax1.elev = 45
            ax1.azim = -20
            plt.savefig(fig_path + '/STAGATE_' + str(i) + '.png')

            i = i + 1
            plt.rcParams["figure.figsize"] = (3, 3)
            sc.pl.umap(adata_2D, color='mclust', title='mclust(STAGATE-2D)', show=False, s=6,
                       save='STAGATE_' + str(i) + '.png')
            shutil.move('./figures/umapSTAGATE_' + str(i) + '.png', fig_path + '/STAGATE_' + str(i) + '.png')

            i = i + 1
            plt.rcParams["figure.figsize"] = (3, 3)
            sc.pl.umap(adata_2D, color='Section_id', title='Section_id(STAGATE-2D)', show=False, s=6,
                       save='STAGATE_' + str(i) + '.png')
            shutil.move('./figures/umapSTAGATE_' + str(i) + '.png', fig_path + '/STAGATE_' + str(i) + '.png')

            i = i + 1
            fig = plt.figure(figsize=(4, 4))
            ax1 = plt.axes(projection='3d')
            for it, label in enumerate(np.unique(adata_2D.obs['mclust'])):
                temp_Coor = adata_2D.obs.loc[adata_2D.obs['mclust'] == label, :]
                temp_xd = temp_Coor['X']
                temp_yd = temp_Coor['Y']
                temp_zd = temp_Coor['Z']
                ax1.scatter3D(temp_xd, temp_yd, temp_zd, c=adata_2D.uns['mclust_colors'][it], s=0.2, marker="o",
                              label=label)

            ax1.set_xlabel('')
            ax1.set_ylabel('')
            ax1.set_zlabel('')

            ax1.set_xticklabels([])
            ax1.set_yticklabels([])
            ax1.set_zticklabels([])

            plt.legend(bbox_to_anchor=(1.2, 0.8), markerscale=10, frameon=False)
            plt.title('STAGATE-2D')

            ax1.elev = 45
            ax1.azim = -20
            plt.savefig(fig_path + '/STAGATE_' + str(i) + '.png')
            plt.close()

        def scoller():
            self.figure_ybar = ttk.Scrollbar(self.figure_Frame, orient=VERTICAL, cursor='draft_small')
            self.figure_ybar.pack(side=RIGHT, fill=Y)
            self.figure_ybar.config(command=self.canvas.yview)
            self.canvas.configure(yscrollcommand=self.figure_ybar.set)
            self.canvas.config(yscrollincrement=1)  # 设置滚动条的步长
            self.canvas.pack(side=LEFT, expand=YES, fill=BOTH, anchor=NW)
            self.canvas.create_window((0, 0), window=self.show_frame, anchor=N)
            self.show_frame.bind("<Configure>", onFrameconfigure)

        def onFrameconfigure(event):
            # width=1450 -> width=1500
            self.canvas.configure(scrollregion=self.canvas.bbox("all"), width=1400, height=650)

        def choose_color():
            colorvalue = colorchooser.askcolor()
            color_list = []
            color = colorvalue[1]
            self.cluster_value = len(section_order)

            if color[1:3] == 'ff':
                color_list.append(color)
                list = random.sample(color_panel.FF_color, self.cluster_value)
                color_list = color_list + list
            elif color[1:3] == 'cc':
                color_list.append(color)
                list = random.sample(color_panel.CC_color, self.cluster_value)
                color_list = color_list + list
            elif color[1:3] == '99':
                color_list.append(color)
                list = random.sample(color_panel.NN_color, self.cluster_value)
                color_list = color_list + list
            elif color[1:3] == '66':
                color_list.append(color)
                list = random.sample(color_panel.SS_color, self.cluster_value)
                color_list = color_list + list
            elif color[1:3] == '33':
                color_list.append(color)
                list = random.sample(color_panel.TT_color, self.cluster_value)
                color_list = color_list + list
            elif color[1:3] == '00':
                color_list.append(color)
                list = random.sample(color_panel.ZZ_color, self.cluster_value)
                color_list = color_list + list
            else:
                color_list.append(color)
                list = random.sample(color_panel.random_color, self.cluster_value)
                color_list = color_list + list

            print(color_list)
            adata.uns['mclust_colors'] = color_list[:len(adata.uns['mclust_colors'])]
            adata.uns['Section_id_colors'] = color_list[:len(adata.uns['Section_id_colors'])]
            adata_2D.uns['mclust_colors'] = color_list[:len(adata_2D.uns['mclust_colors'])]
            adata_2D.uns['Section_id_colors'] = color_list[:len(adata_2D.uns['Section_id_colors'])]
            section_colors = color_list
            fig_path = './figures/3DSpatialDomain'
            os.chdir(Raw_PATH)
            draw_images(adata, adata_2D, fig_path, section_colors)
            figures = os.listdir(fig_path)
            global image_list
            image_list = []
            for i in range(len(figures)):
                image_list.append(ttk.PhotoImage(file=fig_path + '/' + figures[i]))
                self.result_images = ttk.Label(self.show_frame, image=image_list[i])
                self.result_images.grid(row=i // 3, column=i % 3, sticky=W, pady=0)

        fig_path = './figures/3DSpatialDomain'
        if not os.path.isdir(fig_path):
            os.mkdir(fig_path)

        draw_images(adata, adata_2D, fig_path, section_colors)
        figures = os.listdir(fig_path)
        self.figure_Frame = ttk.Frame(self.right_panel)
        self.figure_Frame.pack(side=TOP, fill=X, expand=YES)

        self.canvas = ttk.Canvas(self.figure_Frame)
        self.show_frame = ttk.Frame(self.canvas)
        scoller()

        global image_list
        image_list = []
        s = 0
        for i in range(len(figures)):
            image_list.append(ttk.PhotoImage(file=fig_path + '/' + figures[i]))
            self.result_images = ttk.Label(self.show_frame, image=image_list[i])
            self.result_images.grid(row=i // 3, column=i % 3, sticky=W, pady=0)
            s = i

        self.color_choose = ttk.Button(self.show_frame, text='Reset colors', command=choose_color, width=10)
        self.color_choose.grid(row=s // 3 + 1, column=0, sticky=W, pady=0)

        def VIEW_3D():
            import webbrowser
            data_save_path = '../H5AD_Save/DLPFC'
            self.web_Server_Thread(os.path.join(data_save_path, 'webcache'), 8050)
            webbrowser.open('http://127.0.0.1:8050/')

        self.Reset = ttk.Button(self.show_frame, text='3D VIEW', command=VIEW_3D, width=12)
        self.Reset.grid(row=s // 3 + 1, column=1, sticky=W, pady=0, padx=(0, 100))

        def Reset():
            python = sys.executable
            os.execl(python, python, *sys.argv)

        self.Reset = ttk.Button(self.show_frame, text='Reset STABox', command=Reset, width=12)
        self.Reset.grid(row=s // 3 + 1, column=2, sticky=W, pady=0, padx=(0, 200))

        self.pb.stop()
        self.setvar('prog-message', 'STAGATE run over!')
        self.setvar('End-time', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        EndTime = datetime.now().replace(microsecond=0)
        self.setvar('total-time-cost', EndTime - self.StartTime)
        pass

    def STAligner_Multiple_Sections_analysis(self):
        self.Dataload()
        adata_new = self.result_queue.get()
        if 'GroundTruth' in adata_new.obs.columns:
            self.label_files_exit = True
            if self.cluster_value != len(
                    adata_new.obs['GroundTruth'][adata_new.obs['GroundTruth'] != 'unknown'].unique()):
                self.cluster_value = len(
                    adata_new.obs['GroundTruth'][adata_new.obs['GroundTruth'] != 'unknown'].unique())
            adata_list = {}
            key_add = []
            for section_id in self.multi_files:
                file_name = section_id.rsplit('\\', 1)[-1]
                k = file_name.rsplit('.', 1)[-2]
                key_add.append(k)
                temp_adata = sc.read(section_id)
                temp_adata.var_names_make_unique()
                temp_adata.obs_names = [x + '_' + k for x in temp_adata.obs_names]
                adata_list[k] = temp_adata.copy()
            del temp_adata
        if 'STAligner' not in adata_new.obsm:
            staligner = STAligner(model_dir=running_path + "/result/",
                                  in_features=3000, hidden_dims=[512, 30],
                                  n_models=5, device=torch.device("cuda:0"))
            adata_new = staligner.train(adata_new, iter_comb=[(0, 1)], margin=self.adjust_value)
            mclust_R(adata_new, num_cluster=self.cluster_value, used_obsm='STAligner')
            sc.pp.neighbors(adata_new, use_rep='STAligner', random_state=666)
            sc.tl.umap(adata_new, random_state=666)
            sc.tl.louvain(adata_new, random_state=666, key_added="louvain",
                          resolution=float(self.alpha_value))

        data_save_path = test_file_path + '/STAligner_Multi-DLPFC'
        if not os.path.exists(data_save_path):
            os.makedirs(data_save_path)
            del adata_new.uns['edgeList']
            adata_new.write_h5ad(os.path.join(data_save_path, 'STAligner_Multi_DLPFC.h5ad'))
            json_file = {
                "Coordinate": "spatial",
                "Annotatinos": ["mclust"],
                "Meshes": {
                },
                "mesh_coord": "fixed.json",
                "Genes": [
                    "all"
                ]
            }
            fixed_file = {"xmin": 0, "ymin": 0, "margin": 0, "zmin": 0, "binsize": 1}
            json.dump(json_file, open(os.path.join(data_save_path, 'Multi-DLPFC.json'), 'w'), indent=4)
            json.dump(fixed_file, open(os.path.join(data_save_path, 'fixed.json'), 'w'))
            self.web_Thread(data_save_path, 'STAligner_Multi_DLPFC.h5ad', 'Multi-DLPFC.json', 'Multi-DLPFC')

        Batch_list = []
        for section_id in key_add:
            Batch_list.append(adata_new[adata_new.obs['batch_name'] == section_id])

        def draw_images():
            os.chdir(Raw_PATH)
            from matplotlib import pyplot as plt
            plt.rcParams['font.sans-serif'] = "Arial"
            path = test_file_path + '/figures/' + self.method_flag + '_output/'
            if self.label_files_exit:
                i = 1
                k = 1
                for section_id in key_add:
                    plt.rcParams["figure.figsize"] = (3, 3)
                    sc.pl.spatial(adata_list[section_id], img_key="hires", color=["GroundTruth"],
                                  title=section_id + "-Ground Truth",
                                  show=False, save='STAligner_' + str(k) + '_' + str(i) + '.png')
                    shutil.move(test_file_path + '/figures/show' + 'STAligner_' + str(k) + '_' + str(i) + '.png',
                                path + 'STAligner_' + str(k) + '_' + str(i) + '.png')

                    i = i + 1

                k = k + 1
                i = 1
                spot_size = 200
                for j in range(len(self.multi_files)):
                    plt.rcParams["figure.figsize"] = (3, 3)
                    sc.pl.spatial(Batch_list[j], color=['mclust'], title='STAligner',
                                  spot_size=spot_size, show=False,
                                  save='STAligner_' + str(k) + '_' + str(i) + '.png')
                    shutil.move(test_file_path + '/figures/show' + 'STAligner_' + str(k) + '_' + str(i) + '.png',
                                path + 'STAligner_' + str(k) + '_' + str(i) + '.png')

                    i = i + 1

                k = k + 1
                i = 1
                plt.rcParams["figure.figsize"] = (3, 3)
                sc.pl.umap(adata_new[~adata_new.obs['GroundTruth'].isin(['unknown'])], color=['batch_name'],
                           title='Batchs',
                           show=False, s=6, save='STAligner_' + str(i) + '.png')
                shutil.move(test_file_path + '/figures/umapSTAligner_' + str(i) + '.png',
                            path + 'STAligner_' + str(k) + '_' + str(i) + '.png')

                k = k + 1
                plt.rcParams["figure.figsize"] = (3, 3)
                i = 1
                sc.pl.umap(adata_new[~adata_new.obs['GroundTruth'].isin(['unknown'])], color=['GroundTruth'],
                           title='Ground Truth',
                           show=False, s=6, save='STAligner_' + str(i) + '.png')
                shutil.move(test_file_path + '/figures/umapSTAligner_' + str(i) + '.png',
                            path + 'STAligner_' + str(k) + '_' + str(i) + '.png')
            else:
                i = 1
                k = 1

                spot_size = 200
                for j in range(len(self.multi_files)):
                    plt.rcParams["figure.figsize"] = (3, 3)
                    sc.pl.spatial(Batch_list[j], color=['mclust'], title='STAligner-mclust',
                                  spot_size=spot_size, show=False,
                                  save='STAligner_' + str(k) + '_' + str(i) + '.png')
                    shutil.move(test_file_path + '/figures/show' + 'STAligner_' + str(k) + '_' + str(i) + '.png',
                                path + 'STAligner_' + str(k) + '_' + str(i) + '.png')

                    i = i + 1

                k = k + 1
                i = 1
                plt.rcParams["figure.figsize"] = (3, 3)
                sc.pl.umap(adata_new, color=['batch_name'], title='Batchs', show=False, s=6, save='STAligner_' + str(
                    i) + '.png')
                shutil.move(test_file_path + '/figures/umapSTAligner_' + str(i) + '.png',
                            path + 'STAligner_' + str(k) + '_' + str(i) + '.png')

                k = k + 1
                i = 1
                spot_size = 200
                for j in range(len(self.multi_files)):
                    plt.rcParams["figure.figsize"] = (3, 3)
                    sc.pl.spatial(Batch_list[j], color=['louvain'], title='STAligner-louvain', spot_size=spot_size,
                                  show=False, save='STAligner_' + str(k) + '_' + str(i) + '.png')
                    shutil.move(test_file_path + '/figures/show' + 'STAligner_' + str(k) + '_' + str(i) + '.png',
                                path + 'STAligner_' + str(k) + '_' + str(i) + '.png')
                    i = i + 1

        def scoller():
            self.figure_ybar = ttk.Scrollbar(self.figure_Frame, orient=VERTICAL, cursor='draft_small')
            self.figure_ybar.pack(side=RIGHT, fill=Y)
            self.figure_ybar.config(command=self.canvas.yview)

            self.figure_xbar = ttk.Scrollbar(self.figure_Frame, orient=HORIZONTAL, cursor='draft_small')
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
            self.canvas.configure(scrollregion=self.canvas.bbox("all"), width=1000, height=650)

        def choose_color():
            colorvalue = colorchooser.askcolor()
            color_list = []
            color = colorvalue[1]
            print(color)
            if 'louvain_colors' in Batch_list[0].uns:
                color_lens = [Batch_list[0].uns['louvain_colors'], Batch_list[0].uns['mclust_colors']]
                max_len = max(color_lens, key=len)
                if len(max_len) > self.cluster_value:
                    self.cluster_value = len(max_len)

            if color[1:3] == 'ff':
                color_list.append(color)
                list = random.sample(color_panel.FF_color, self.cluster_value)
                color_list = color_list + list
            elif color[1:3] == 'cc':
                color_list.append(color)
                list = random.sample(color_panel.CC_color, self.cluster_value)
                color_list = color_list + list
            elif color[1:3] == '99':
                color_list.append(color)
                list = random.sample(color_panel.NN_color, self.cluster_value)
                color_list = color_list + list
            elif color[1:3] == '66':
                color_list.append(color)
                list = random.sample(color_panel.SS_color, self.cluster_value)
                color_list = color_list + list
            elif color[1:3] == '33':
                color_list.append(color)
                list = random.sample(color_panel.TT_color, self.cluster_value)
                color_list = color_list + list
            elif color[1:3] == '00':
                color_list.append(color)
                list = random.sample(color_panel.ZZ_color, self.cluster_value)
                color_list = color_list + list
            else:
                color_list.append(color)
                list = random.sample(color_panel.random_color, self.cluster_value)
                color_list = color_list + list

            print(color_list)
            for j in range(len(self.multi_files)):
                Batch_list[j].uns['mclust_colors'] = color_list[:len(Batch_list[j].uns['mclust_colors'])]

            if 'louvain_colors' in Batch_list[0].uns:
                for j in range(len(self.multi_files)):
                    Batch_list[j].uns['louvain_colors'] = color_list[:len(Batch_list[j].uns['louvain_colors'])]

            adata_new.uns['batch_name_colors'] = color_list[:len(self.multi_files)]
            if self.label_files_exit:
                adata_new.uns['GroundTruth_colors'] = color_list
                for section_id in key_add:
                    adata_list[section_id].uns['GroundTruth_colors'] = color_list
            self.color_reset = color_list
            self.gene_color_type = color_panel.five_gene_color[random.randint(0, 5)]
            os.chdir(Raw_PATH)
            draw_images()
            fig_path = test_file_path + '/figures/' + self.method_flag + '_output/'
            figures = os.listdir(fig_path)
            global image_list
            image_list = []
            for i in range(len(figures)):
                image_list.append(ttk.PhotoImage(file=fig_path + '/' + figures[i]))
                self.result_images = ttk.Label(self.show_frame, image=image_list[i])
                self.result_images.grid(row=i // 4, column=i % 4, sticky=W, pady=0)

        fig_path = test_file_path + '/figures/' + self.method_flag + '_output/'
        if not os.path.isdir(fig_path):
            os.mkdir(fig_path)

        draw_images()
        figures = os.listdir(fig_path)
        self.figure_Frame = ttk.Frame(self.right_panel)
        self.figure_Frame.pack(side=TOP, fill=X, expand=YES)

        self.canvas = ttk.Canvas(self.figure_Frame)
        self.show_frame = ttk.Frame(self.canvas)
        scoller()

        self.set_frame = ttk.Frame(self.figure_Frame, borderwidth=2, relief="sunken")
        self.set_frame.pack(side='right', expand=YES, anchor=N)
        self.set_frame_one = ttk.Frame(self.set_frame)
        self.set_frame_one.pack(side=TOP, expand=YES)
        self.set_frame_two = ttk.Frame(self.set_frame)
        self.set_frame_two.pack(side=TOP, expand=YES)

        self.Reset = ttk.Button(self.set_frame_one, text='Reset all colors', command=choose_color, width=12)
        self.Reset.grid(row=0, column=0, sticky=W, pady=0, padx=0)

        self.domain_color_label = ttk.Label(self.set_frame_one, text='Reset domain color: ', width=20)
        self.domain_color_label.grid(row=1, column=0, sticky=W, pady=0)

        self.Reset_domain_color = ttk.Entry(self.set_frame_one, width=10)
        self.Reset_domain_color.grid(row=1, column=1, sticky=W, pady=0)
        Tooltip(self.Reset_domain_color, "Input value must be int type and in [1:cluster_name]: 2")
        self.color_update = None

        def Reset_single_domain_color():
            from tkinter import colorchooser, filedialog
            colorvalue = colorchooser.askcolor()
            color = colorvalue[1]
            print(color)
            cluster = self.Reset_domain_color.get()
            print(cluster)
            for j in range(len(self.multi_files)):
                Batch_list[j].uns['mclust_colors'] = self.color_reset[:len(Batch_list[j].uns['mclust_colors'])]
                Batch_list[j].uns['mclust_colors'][int(cluster) - 1] = color

            adata_new.uns['batch_name_colors'] = self.color_reset[:len(self.multi_files)]
            self.color_reset[int(cluster) - 1] = color
            # self.color_update = self.color_reset
            if 'louvain_colors' in Batch_list[0].uns:
                for j in range(len(self.multi_files)):
                    Batch_list[j].uns['louvain_colors'] = self.color_reset[:len(Batch_list[j].uns['louvain_colors'])]

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
                img = Image.open(fig_path + '/' + figures[i])
                print(img.size)
                image_list.append(ttk.PhotoImage(file=fig_path + '/' + figures[i]))
                self.result_images = ttk.Label(self.show_frame, image=image_list[i])
                self.result_images.grid(row=i // 4, column=i % 4, sticky=W, pady=0, padx=0)
                s = i

        self.Reset_domain_btn = ttk.Button(self.set_frame_one, width=10, text='Confirm',
                                           command=Reset_single_domain_color)
        self.Reset_domain_btn.grid(row=1, column=2, sticky=W, pady=10)
        self.update_image_label = ttk.Label(self.set_frame_one, text='Reset image dpi: ', width=20)
        self.update_image_label.grid(row=2, column=0, sticky=W, pady=10)
        self.update_image_scale = ttk.Entry(self.set_frame_one, width=10)
        self.update_image_scale.grid(row=2, column=1, sticky=W, pady=10)
        Tooltip(self.update_image_scale, "Input value must be int type and >= 300: 300")

        def ipdate_hd():
            dpi = self.update_image_scale.get()
            print(dpi)
            import matplotlib.pyplot as plt
            plt.rcParams["figure.figsize"] = (3, 3)
            sc.set_figure_params(dpi=dpi)
            draw_images()
            figures = os.listdir(fig_path)
            print(len(figures))

        self.Reset_domain_btn = ttk.Button(self.set_frame_one, width=10, text='Save', command=ipdate_hd)
        self.Reset_domain_btn.grid(row=2, column=2, sticky=W, pady=0)

        self.gene_visualization_label = ttk.Label(self.set_frame_one, text='Input gene name: ', width=20)
        self.gene_visualization_label.grid(row=3, column=0, sticky=W, pady=0)

        self.gene_visualization_entry = ttk.Entry(self.set_frame_one, width=10)
        self.gene_visualization_entry.grid(row=3, column=1, sticky=W, pady=0)
        Tooltip(self.gene_visualization_entry, "Gene name: NEFH/ATP2B4")

        def gene_visualization():
            try:
                gene = self.gene_visualization_entry.get()
                print(adata_list[self.multi_files[-1]].var.index)
                if gene in adata_list[self.multi_files[-1]].var_names:

                    sc.pl.spatial(adata_list[self.multi_files[-1]], img_key="hires", color=gene, title="$" + gene + "$",
                                  show=False, save=gene + '.png', sopt_size=150)
                else:
                    print(f'{gene} is not in adata!')
                global img0
                photo = Image.open(test_file_path + '/figures/show' + gene + '.png')
                img0 = ImageTk.PhotoImage(photo)
                img1 = ttk.Label(self.set_frame_two, image=img0)
                img1.grid(row=0, column=0, sticky=W, pady=0)
                if os.path.exists(test_file_path + '/figures/show' + gene + '.png'):
                    os.remove(test_file_path + '/figures/show' + gene + '.png')
                    print("Figures exits")
                else:
                    print("Figures no exits！")
                pass
            except:
                Messagebox.show_error("Python Error", "Make sure gene name in dataset")

        self.gene_visualization_btn = ttk.Button(self.set_frame_one, width=10, command=gene_visualization, text='Show')
        self.gene_visualization_btn.grid(row=3, column=2, sticky=W, pady=0)

        # end
        global image_list
        image_list = []
        s = 0
        for i in range(len(figures)):
            image_list.append(ttk.PhotoImage(file=fig_path + '/' + figures[i]))
            self.result_images = ttk.Label(self.show_frame, image=image_list[i])
            self.result_images.grid(row=i // 4, column=i % 4, sticky=W, pady=0)
            s = i

        self.Entry = ttk.Entry(self.set_frame_one, width=10)
        self.Entry.grid(row=0, column=1, sticky=W, pady=0)
        Tooltip(self.Entry, "Local port number: 8049")

        def VIEW_3D():
            import webbrowser
            self.web_Server_Thread("/STAligner_Multi-DLPFC/webcache", int(self.Entry.get()))
            http = 'http://127.0.0.1:' + self.Entry.get() + '/'
            # 'http://127.0.0.1:8050/'
            webbrowser.open(http)

        self.Reset = ttk.Button(self.set_frame_one, text='3D VIEW', command=VIEW_3D, width=10)
        self.Reset.grid(row=0, column=2, sticky=W, pady=0)

        self.pb.stop()
        self.setvar('prog-message', 'STAligner run over!')
        self.setvar('End-time', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        EndTime = datetime.now().replace(microsecond=0)
        self.setvar('total-time-cost', EndTime - self.StartTime)

    def STAligner_Thread(self):
        T = threading.Thread(target=self.STAligner_Multiple_Sections_analysis)
        T.setDaemon(True)
        T.start()

    def STAligner_3DSlices_process(self):
        try:
            Batch_list = []
            adj_list = []
            section_ids = os.listdir(self.Visium_file_path)
            self.data_name = 'Mouse_Hippos'
            self.setvar('data_name', self.data_name)
            for section_id in section_ids:
                location = glob.glob(os.path.join(self.Visium_file_path, section_id)
                                     + '/BeadMapping_*' + '/BeadLocationsForR.csv')
                express = glob.glob(os.path.join(self.Visium_file_path, section_id)
                                    + '/BeadMapping_*' + '/MappedDGEForR.csv')
                counts = pd.read_csv(express[0], sep=',', index_col=0)
                coor_df = pd.read_csv(location[0], index_col=0, sep=',')
                fsize = os.path.getsize(express[0])
                adata = sc.AnnData(counts.T)
                adata.var_names_make_unique()
                coor_df = coor_df.loc[adata.obs_names, ['xcoord', 'ycoord']]
                adata.obs[['X', 'Y']] = coor_df.loc[adata.obs_names, ['xcoord', 'ycoord']]
                adata.obsm["spatial"] = coor_df.to_numpy()
                sc.pp.calculate_qc_metrics(adata, inplace=True)
                sc.pp.filter_genes(adata, min_cells=50)

                adata.obs_names = [x + '_' + section_id for x in adata.obs_names]
                Cal_Spatial_Net_new(adata, rad_cutoff=self.rad_cutoff_value)

                sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=10000)
                sc.pp.normalize_total(adata, target_sum=1e4)
                sc.pp.log1p(adata)

                adata = adata[:, adata.var['highly_variable']]
                adj_list.append(adata.uns['adj'])
                Batch_list.append(adata)

            adata_concat = ad.concat(Batch_list, label="slice_name", keys=section_ids)
            adata_concat.obs["batch_name"] = adata_concat.obs["slice_name"].astype('category')

            self.data_size = round(fsize * len(section_ids) / 1024 / 1024, 2)
            self.setvar('data_size', str(self.data_size) + 'MB')
            self.data_shape = adata_concat.shape
            self.setvar('data_shape', str(self.data_shape[0]) + '×' + str(self.data_shape[1]))
            self.result_queue.put(Batch_list)
            self.result_queue.put(adata_concat)
        except:
            Messagebox.show_info('Check whether the Embryo files exists!')

    def STAligner_3DSlices_analysis(self):
        self.STAligner_3DSlices_process()
        Batch_list = self.result_queue.get()
        adata_concat = self.result_queue.get()
        section_ids = os.listdir(self.Visium_file_path)
        used_device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        iter_comb = []
        for i in range(len(section_ids) - 1):
            iter_comb.append((i, i + 1))

        adata_concat = train_STAligner_subgraph(adata_concat, verbose=True, knn_neigh=100, n_epochs=600,
                                                iter_comb=iter_comb, Batch_list=Batch_list, device=used_device)
        sc.pp.neighbors(adata_concat, use_rep='STAligner', random_state=666)
        sc.tl.louvain(adata_concat, random_state=666, key_added="louvain", resolution=self.alpha_value)
        sc.tl.umap(adata_concat, random_state=666)

        for it in range(len(section_ids)):
            Batch_list[it].obs['louvain'] = adata_concat[adata_concat.obs['batch_name']
                                                         == section_ids[it]].obs['louvain'].values

        landmark_domain = ['3']
        from ..pl.utils import ICP_align
        for comb in iter_comb:
            print(comb)
            i, j = comb[0], comb[1]
            adata_target = Batch_list[i]
            adata_ref = Batch_list[j]
            slice_target = section_ids[i]
            slice_ref = section_ids[j]

            aligned_coor = ICP_align(adata_concat, adata_target, adata_ref, slice_target, slice_ref, landmark_domain)
            adata_target.obsm["spatial"] = aligned_coor

        adata_concat.obs['Z'] = list(Batch_list[0].shape[0] * [0]) + list(Batch_list[1].shape[0] * [10]) \
                                + list(Batch_list[2].shape[0] * [20]) + list(Batch_list[3].shape[0] * [30]) \
                                + list(Batch_list[4].shape[0] * [40]) + list(Batch_list[5].shape[0] * [50]) + \
                                list(Batch_list[6].shape[0] * [60])

        # adata_concat.write_h5ad("D:\\Users\\lqlu\\download\\3D_SRT_Data\\3Ddata\\3Ddata.h5ad")

        All_coor = adata_concat.obs[['X', 'Y', 'Z']].copy()
        All_coor.loc[adata_concat.obs['batch_name'] == section_ids[0], :2] = Batch_list[0].obsm["spatial"]
        All_coor.loc[adata_concat.obs['batch_name'] == section_ids[1], :2] = Batch_list[1].obsm["spatial"]
        All_coor.loc[adata_concat.obs['batch_name'] == section_ids[2], :2] = Batch_list[2].obsm["spatial"]
        All_coor.loc[adata_concat.obs['batch_name'] == section_ids[3], :2] = Batch_list[3].obsm["spatial"]
        All_coor.loc[adata_concat.obs['batch_name'] == section_ids[4], :2] = Batch_list[4].obsm["spatial"]
        All_coor.loc[adata_concat.obs['batch_name'] == section_ids[5], :2] = Batch_list[5].obsm["spatial"]
        All_coor.loc[adata_concat.obs['batch_name'] == section_ids[6], :2] = Batch_list[6].obsm["spatial"]

        def draw_images(adata_concat, Batch_list):
            from matplotlib import pyplot as plt
            plt.rcParams['font.sans-serif'] = "Arial"
            plt.rcParams['font.size'] = 10
            i = 1
            j = 0
            spot_size = 30  # 0.8
            for ss in range(len(section_ids)):
                plt.rcParams["figure.figsize"] = (3, 3)
                sc.pl.spatial(Batch_list[ss], img_key=None, color=["louvain"], title=[section_ids[ss]], show=False,
                              spot_size=spot_size, size=1.5, legend_fontsize=15,
                              save='STAligner_3DAlignment/STAligner_' + str(j) + '_' + str(i) + '.png')
                i = i + 1

            landmark_domain = '3'
            j = j + 1
            i = 1
            for ss in range(len(section_ids)):
                plt.rcParams["figure.figsize"] = (3, 3)
                sc.pl.spatial(Batch_list[ss], img_key=None, color=["louvain"],
                              title=[section_ids[ss]], show=False, spot_size=spot_size, groups=[landmark_domain],
                              size=1.5,
                              legend_fontsize=15,
                              save='STAligner_3DAlignment/STAligner_' + str(j) + '_' + str(i) + '.png')
                i = i + 1

            j = j + 1
            i = 1
            plt.rcParams["figure.figsize"] = (3, 3)
            sc.pl.umap(adata_concat, color=['batch_name'], title='Batchs', show=False, s=6, ncols=2, wspace=0.8,
                       save='STAligner_umap' + str(i) + '.png')
            shutil.move('./figures/umapSTAligner_umap' + str(i) + '.png',
                        './figures/showSTAligner_3DAlignment/' + 'STAligner_' + str(j) + '_' + str(i) + '.png')

            i = i + 1
            plt.rcParams["figure.figsize"] = (3, 3)
            sc.pl.umap(adata_concat, color=['louvain'], title='louvain', show=False, s=6, ncols=3, wspace=0.5,
                       save='STAligner_umap' + str(i) + '.png')
            shutil.move('./figures/umapSTAligner_umap' + str(i) + '.png',
                        './figures/showSTAligner_3DAlignment/' + 'STAligner_' + str(j) + '_' + str(i) + '.png')

            j = j + 1
            i = 1
            fig = plt.figure(figsize=(5, 4))
            ax1 = plt.axes(projection='3d')
            landmark_domain = '3'

            for it, label in enumerate(np.unique(adata_concat.obs['louvain'])):
                temp_Coor = All_coor.loc[adata_concat.obs['louvain'] == label, :]
                temp_xd = temp_Coor['X']
                temp_yd = temp_Coor['Y']
                temp_zd = temp_Coor['Z']
                if label == landmark_domain:
                    ax1.scatter3D(temp_xd, temp_yd, temp_zd, c=adata_concat.uns['louvain_colors'][it],
                                  s=0.02, marker="o", label=label, alpha=1)
                else:
                    ax1.scatter3D(temp_xd, temp_yd, temp_zd, c=adata_concat.uns['louvain_colors'][it],
                                  s=0.02, marker="o", label=label, alpha=0.05)

            plt.legend(bbox_to_anchor=(1.2, 0.8), markerscale=10, frameon=False)
            plt.title('3D stacked slices')
            ax1.elev = 20
            ax1.azim = -40

            ax1.set_xlabel('')
            ax1.set_ylabel('')
            ax1.set_zlabel('')
            ax1.set_xticklabels([])
            ax1.set_yticklabels([])
            ax1.set_zticklabels([])
            plt.savefig('./figures/showSTAligner_3DAlignment/STAligner_' + str(j) + '_' + str(i) + '.png')
            plt.close()

        def scoller():
            self.figure_ybar = ttk.Scrollbar(self.figure_Frame, orient=VERTICAL, cursor='draft_small')
            self.figure_ybar.pack(side=RIGHT, fill=Y)
            self.figure_ybar.config(command=self.canvas.yview)
            self.canvas.configure(yscrollcommand=self.figure_ybar.set)
            self.canvas.config(yscrollincrement=1)  # 设置滚动条的步长
            self.canvas.pack(side=LEFT, expand=YES, fill=BOTH, anchor=NW)
            self.canvas.create_window((0, 0), window=self.show_frame, anchor=N)
            self.show_frame.bind("<Configure>", onFrameconfigure)

        def onFrameconfigure(event):
            # width=1450 -> width=1500
            self.canvas.configure(scrollregion=self.canvas.bbox("all"), width=1750, height=650)

        def choose_color():
            colorvalue = colorchooser.askcolor()
            color_list = []
            color = colorvalue[1]
            print(color)

            if self.cluster_value != len(adata_concat.uns['louvain_colors']):
                self.cluster_value = len(adata_concat.uns['louvain_colors'])
            if len(section_ids) > self.cluster_value:
                self.cluster_value = len(section_ids)
            if color[1:3] == 'ff':
                color_list.append(color)
                list = random.sample(color_panel.FF_color, self.cluster_value)
                color_list = color_list + list
            elif color[1:3] == 'cc':
                color_list.append(color)
                list = random.sample(color_panel.CC_color, self.cluster_value)
                color_list = color_list + list
            elif color[1:3] == '99':
                color_list.append(color)
                list = random.sample(color_panel.NN_color, self.cluster_value)
                color_list = color_list + list
            elif color[1:3] == '66':
                color_list.append(color)
                list = random.sample(color_panel.SS_color, self.cluster_value)
                color_list = color_list + list
            elif color[1:3] == '33':
                color_list.append(color)
                list = random.sample(color_panel.TT_color, self.cluster_value)
                color_list = color_list + list
            elif color[1:3] == '00':
                color_list.append(color)
                list = random.sample(color_panel.ZZ_color, self.cluster_value)
                color_list = color_list + list
            else:
                color_list.append(color)
                list = random.sample(color_panel.random_color, self.cluster_value)
                color_list = color_list + list

            print(color_list)
            adata_concat.uns['batch_name_colors'] = color_list[:len(adata_concat.uns['batch_name_colors'])]
            for i in range(len(section_ids)):
                Batch_list[i].uns['louvain_colors'] = color_list[:len(Batch_list[i].uns['louvain_colors'])]
            adata_concat.uns['louvain_colors'] = color_list[:len(adata_concat.uns['louvain_colors'])]
            draw_images(adata_concat, Batch_list)

            fig_path = './figures/showSTAligner_3DAlignment'
            figures = os.listdir(fig_path)
            global image_list
            image_list = []
            for i in range(len(figures)):
                image_list.append(ttk.PhotoImage(file=fig_path + '/' + figures[i]))
                self.result_images = ttk.Label(self.show_frame, image=image_list[i])
                self.result_images.grid(row=i // 3, column=i % 3, sticky=W, pady=0)

        fig_path = './figures/showSTAligner_3DAlignment'
        if not os.path.isdir(fig_path):
            os.mkdir(fig_path)

        draw_images(adata_concat, Batch_list)
        figures = os.listdir(fig_path)
        self.figure_Frame = ttk.Frame(self.right_panel)
        self.figure_Frame.pack(side=TOP, fill=X, expand=YES)

        self.canvas = ttk.Canvas(self.figure_Frame)
        self.show_frame = ttk.Frame(self.canvas)
        scoller()

        global image_list
        image_list = []
        s = 0
        for i in range(len(figures)):
            image_list.append(ttk.PhotoImage(file=fig_path + '/' + figures[i]))
            self.result_images = ttk.Label(self.show_frame, image=image_list[i])
            self.result_images.grid(row=i // 3, column=i % 3, sticky=W, pady=0)
            s = i

        self.color_choose = ttk.Button(self.show_frame, text='reset colors', command=choose_color, width=10)
        self.color_choose.grid(row=s // 3 + 1, column=1, sticky=W, pady=0)

        def Reset():
            python = sys.executable
            os.execl(python, python, *sys.argv)

        self.Reset = ttk.Button(self.show_frame, text='Reset STABox', command=Reset, width=12)
        self.Reset.grid(row=s // 3 + 1, column=2, sticky=W, pady=0)

        self.pb.stop()
        self.setvar('prog-message', 'STAligner run over!')
        self.setvar('End-time', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        EndTime = datetime.now().replace(microsecond=0)
        self.setvar('total-time-cost', EndTime - self.StartTime)
        pass

    def STAligner_Thread_two(self):
        T = threading.Thread(target=self.STAligner_3DSlices_analysis)
        T.setDaemon(True)
        T.start()

    def STAligner_EmbryoData_process(self):
        try:
            files = os.listdir(self.Visium_file_path)
            files.sort(key=lambda i: int(re.search(r'(\d+)', i).group()))
            print(files)
            section_ids = []
            for i in files:
                section_ids.append(i.rsplit('.', 2)[-3])
            Batch_list = []
            adj_list = []
            self.data_name = 'Mouse Embryo'
            self.setvar('data_name', self.data_name)

            for section_id in section_ids:
                print(section_id)
                adata = sc.read_h5ad(os.path.join(self.Visium_file_path, section_id + ".MOSTA.h5ad"))
                adata.X = adata.layers['count']

                # make spot name unique
                adata.obs_names = [x + '_' + section_id for x in adata.obs_names]

                # self.rad_cutoff_value = 1.3
                Cal_Spatial_Net_new(adata, rad_cutoff=self.rad_cutoff_value)

                # Normalization
                sc.pp.normalize_total(adata, target_sum=1e4)
                sc.pp.log1p(adata)
                sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=5000)
                adata = adata[:, adata.var['highly_variable']]

                adj_list.append(adata.uns['adj'])
                Batch_list.append(adata)

            adata_concat = ad.concat(Batch_list, label="slice_name", keys=section_ids)
            adata_concat.obs["batch_name"] = adata_concat.obs["slice_name"].astype('category')

            fsize = os.path.getsize(os.path.join(self.Visium_file_path, files[0]))
            self.data_size = round(fsize * 4 / 1024 / 1024, 2)
            self.setvar('data_size', str(self.data_size) + 'MB')
            self.data_shape = adata_concat.shape
            self.setvar('data_shape', str(self.data_shape[0]) + '×' + str(self.data_shape[1]))

            adj_concat = np.asarray(adj_list[0].todense())
            for batch_id in range(1, len(section_ids)):
                adj_concat = scipy.linalg.block_diag(adj_concat, np.asarray(adj_list[batch_id].todense()))
            adata_concat.uns['edgeList'] = np.nonzero(adj_concat)

            self.result_queue.put(Batch_list)
            self.result_queue.put(adata_concat)

        except:
            Messagebox.show_info('Check whether the Embryo files exists!')
        pass

    def STAligner_EmbryoData_analysis(self):
        self.STAligner_EmbryoData_process()
        files = os.listdir(self.Visium_file_path)
        files.sort(key=lambda i: int(re.search(r'(\d+)', i).group()))
        section_ids = []
        for i in files:
            section_ids.append(i.rsplit('.', 2)[-3])
        iter_comb = [(0, 3), (1, 3), (2, 3)]
        used_device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        Batch_list = self.result_queue.get()
        adata_concat = self.result_queue.get()
        # Fixed slice pairs to align. For example, (0, 1) means slice 0 will be algined with slice 1 as reference.
        # self.adjust_value = 2.5
        adata_concat = train_STAligner(adata_concat, verbose=True, knn_neigh=100, n_epochs=1000,
                                       iter_comb=iter_comb, margin=self.adjust_value, device=used_device)

        colors_default = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                          '#8c564b', '#e377c2', '#bcbd22', '#17becf', '#aec7e8',
                          '#ffbb78', '#98df8a', '#ff9896', '#bec1d4', '#bb7784',
                          '#0000ff', '#FFFF00', '#800080', '#e07b91',
                          '#959595', '#7d87b9', '#d6bcc0',
                          '#8e063b', '#4a6fe3', '#8595e1', '#b5bbe3', '#e6afb9',
                          '#d33f6a', '#11c638', '#8dd593', '#c6dec7', '#ead3c6', '#f0b98d',
                          '#ef9708', '#0fcfc0', '#9cded6', '#d5eae7', '#f3e1eb', '#f6c4e1',
                          '#111010', '#f79cd4']

        sc.pp.neighbors(adata_concat, use_rep='STAligner', random_state=666)
        sc.tl.umap(adata_concat, random_state=666)
        sc.tl.louvain(adata_concat, random_state=666, key_added="louvain", resolution=self.alpha_value)
        adata_concat.uns['louvain_colors'] = [colors_default[0:][i] for i in
                                              np.sort(adata_concat.obs['louvain'].unique().astype('int'))]
        for ss in range(len(section_ids)):
            Batch_list[ss].obs['louvain'] = adata_concat[adata_concat.obs['batch_name'] == section_ids[ss]].obs[
                'louvain'].values
            Batch_list[ss].uns['louvain_colors'] = \
                [colors_default[0:][i] for i in np.sort(
                    adata_concat[adata_concat.obs['batch_name'] == section_ids[ss]].obs['louvain'].unique().astype(
                        'int'))]

            cluster_len = len(
                adata_concat[adata_concat.obs['batch_name'] == section_ids[ss]].obs['annotation'].unique())
            Batch_list[ss].uns['annotation_colors'] = [colors_default[0:][i] for i in range(cluster_len)]

        def draw_images(adata_concat):
            from matplotlib import pyplot as plt
            plt.rcParams['font.sans-serif'] = "Arial"
            plt.rcParams['font.size'] = 10
            i = 1
            j = 0
            spot_size = 0.8  # 0.8
            for ss in range(len(section_ids)):
                plt.rcParams["figure.figsize"] = (3, 3)
                ls = sc.pl.spatial(Batch_list[ss], img_key=None, color=["louvain"], title=[section_ids[ss]], show=False,
                                   spot_size=spot_size, size=1.5)
                # 图的类标签数字太大
                # ls[0].legend(loc="right", bbox_to_anchor=[0, 1], ncol=2, fancybox=True)
                # ls[0].legend(loc="center right", bbox_to_anchor=(1.05, 0.4), ncol=2, fancybox=True)
                ls[0].legend(loc=3, bbox_to_anchor=(1.05, 0.3), ncol=2, fancybox=True, prop={'size': 6})
                plt.tight_layout()
                plt.savefig('./figures/showSTAlignerEmbryoData/STAligner_' + str(j) + '_' + str(i) + '.png',
                            bbox_inches='tight')
                plt.close()
                i = i + 1

            for ii in range(len(adata_concat.obs['louvain'].unique())):
                j = j + 1
                i = 1
                for ss in range(len(section_ids)):
                    plt.rcParams["figure.figsize"] = (3, 3)
                    sc.pl.spatial(Batch_list[ss], img_key=None, color=["louvain"],
                                  title=[''], show=False, spot_size=spot_size, groups=[str(ii)], size=1.5,
                                  save='STAlignerEmbryoData/STAligner_' + str(j) + '_' + str(i) + '.png')
                    i = i + 1

            i = 1
            plt.rcParams["figure.figsize"] = (3, 3)
            sc.pl.umap(adata_concat, color=['batch_name'], title='Batchs', show=False, s=6, ncols=3, wspace=0.5,
                       legend_loc='on data', save='STAligner_umap' + str(i) + '.png')
            shutil.move('./figures/umapSTAligner_umap' + str(i) + '.png',
                        './figures/showSTAlignerEmbryoData/' + 'STAligner_umap' + str(i) + '.png')

            # j = j + 1
            i = i + 1
            plt.rcParams["figure.figsize"] = (3, 3)
            sc.pl.umap(adata_concat, color=['louvain'], title='louvain', show=False, s=6, ncols=3, wspace=0.5,
                       legend_loc='on data', save='STAligner_umap' + str(i) + '.png')
            shutil.move('./figures/umapSTAligner_umap' + str(i) + '.png',
                        './figures/showSTAlignerEmbryoData/' + 'STAligner_umap' + str(i) + '.png')

        def scoller():
            self.figure_ybar = ttk.Scrollbar(self.figure_Frame, orient=VERTICAL, cursor='draft_small')
            self.figure_ybar.pack(side=RIGHT, fill=Y)
            self.figure_ybar.config(command=self.canvas.yview)
            self.canvas.configure(yscrollcommand=self.figure_ybar.set)
            self.canvas.config(yscrollincrement=1)  # 设置滚动条的步长
            self.canvas.pack(side=LEFT, expand=YES, fill=BOTH, anchor=NW)
            self.canvas.create_window((0, 0), window=self.show_frame, anchor=N)
            self.show_frame.bind("<Configure>", onFrameconfigure)

        def onFrameconfigure(event):
            # width=1450 -> width=1500
            self.canvas.configure(scrollregion=self.canvas.bbox("all"), width=2000, height=650)

        def choose_color():
            colorvalue = colorchooser.askcolor()
            color_list = []
            color = colorvalue[1]
            print(color)

            if self.cluster_value != len(adata_concat.uns['louvain_colors']):
                self.cluster_value = len(adata_concat.uns['louvain_colors'])
            if color[1:3] == 'ff':
                color_list.append(color)
                list = random.sample(color_panel.FF_color, self.cluster_value)
                color_list = color_list + list
            elif color[1:3] == 'cc':
                color_list.append(color)
                list = random.sample(color_panel.CC_color, self.cluster_value)
                color_list = color_list + list
            elif color[1:3] == '99':
                color_list.append(color)
                list = random.sample(color_panel.NN_color, self.cluster_value)
                color_list = color_list + list
            elif color[1:3] == '66':
                color_list.append(color)
                list = random.sample(color_panel.SS_color, self.cluster_value)
                color_list = color_list + list
            elif color[1:3] == '33':
                color_list.append(color)
                list = random.sample(color_panel.TT_color, self.cluster_value)
                color_list = color_list + list
            elif color[1:3] == '00':
                color_list.append(color)
                list = random.sample(color_panel.ZZ_color, self.cluster_value)
                color_list = color_list + list
            else:
                color_list.append(color)
                list = random.sample(color_panel.random_color, self.cluster_value)
                color_list = color_list + list

            print(color_list)
            adata_concat.uns['batch_name_colors'] = color_list[:len(adata_concat.uns['batch_name_colors'])]
            for i in range(len(section_ids)):
                Batch_list[i].uns['louvain_colors'] = color_list[:len(Batch_list[i].uns['louvain_colors'])]
            adata_concat.uns['louvain_colors'] = color_list[:len(adata_concat.uns['louvain_colors'])]
            draw_images(adata_concat)

            fig_path = './figures/showSTAlignerEmbryoData'
            figures = os.listdir(fig_path)
            global image_list
            image_list = []
            for i in range(len(figures)):
                image_list.append(ttk.PhotoImage(file=fig_path + '/' + figures[i]))
                self.result_images = ttk.Label(self.show_frame, image=image_list[i])
                self.result_images.grid(row=i // 4, column=i % 4, sticky=W, pady=0)

        fig_path = './figures/showSTAlignerEmbryoData'
        if not os.path.isdir(fig_path):
            os.mkdir(fig_path)

        draw_images(adata_concat)
        figures = os.listdir(fig_path)
        self.figure_Frame = ttk.Frame(self.right_panel)
        self.figure_Frame.pack(side=TOP, fill=X, expand=YES)

        self.canvas = ttk.Canvas(self.figure_Frame)
        self.show_frame = ttk.Frame(self.canvas)
        scoller()

        global image_list
        image_list = []
        s = 0
        for i in range(len(figures)):
            image_list.append(ttk.PhotoImage(file=fig_path + '/' + figures[i]))
            self.result_images = ttk.Label(self.show_frame, image=image_list[i])
            self.result_images.grid(row=i // 4, column=i % 4, sticky=W, pady=0)
            s = i

        self.color_choose = ttk.Button(self.show_frame, text='Reset colors', command=choose_color, width=10)
        self.color_choose.grid(row=s // 4 + 1, column=1, sticky=W, pady=0)

        def Reset():
            python = sys.executable
            os.execl(python, python, *sys.argv)

        self.Reset = ttk.Button(self.show_frame, text='Reset STABox', command=Reset, width=12)
        self.Reset.grid(row=s // 4 + 1, column=2, sticky=W, pady=0)

        self.pb.stop()
        self.setvar('prog-message', 'STAligner run over!')
        self.setvar('End-time', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        EndTime = datetime.now().replace(microsecond=0)
        self.setvar('total-time-cost', EndTime - self.StartTime)
        pass

    def STAligner_Thread_three(self):
        T = threading.Thread(target=self.STAligner_EmbryoData_analysis)
        T.setDaemon(True)
        T.start()

    def STAMarker_SlideseqV2_Process(self):
        h5ad_file = glob.glob(self.file_path + '/*.h5ad')
        if os.path.exists(self.file_path) and len(h5ad_file) != 0:
            self.setvar('data_name', 'SlideSeq-V2 Data')
            adata = sc.read(h5ad_file[0])
            adata.var_names_make_unique()
            fsize = os.path.getsize(h5ad_file[0])
            self.setvar('data_size', str(round(fsize / 1024 / 1024, 2)) + 'MB')
            self.setvar('data_shape', str(adata.shape[0]) + '×' + str(adata.shape[1]))
            self.result_queue.put(adata)
            pass
        else:
            Messagebox.show_error(title="File error", message="Make sure only one h5ad file in path!")
        pass

    def STAMarker_SlideseqV2_Analysis(self):
        # self.STAMarker_SlideseqV2_Process()
        ann_data = self.result_queue.get()
        # self.rad_cutoff_value = 40
        _path = os.path.dirname(os.path.abspath(__file__))
        _path = _path.rsplit('\\VIEW')[-2]
        _path = _path + '\\_params\\'
        config = dict()
        config.update(parse_args(_path + "model.yaml"))
        config.update(parse_args(_path + "trainer.yaml"))
        if not torch.cuda.is_available():
            config["stagate_trainer"]["gpus"] = None
            config["classifier_trainer"]["gpus"] = None

        data_module = make_spatial_data(ann_data)
        data_module.prepare_data(n_top_genes=3000, rad_cutoff=self.rad_cutoff_value, min_counts=20)

        data_path = './STAMarker_' + self.data_type + '_output'
        if not os.path.isdir(data_path):
            os.mkdir(data_path)

        stamarker = STAMarker(20, data_path + "/", config)
        stamarker.train_auto_encoders(data_module)
        stamarker.clustering(data_module, "louvain", self.alpha_value)
        stamarker.consensus_clustering2(self.cluster_value)
        stamarker.train_classifiers(data_module, self.cluster_value)
        smap = stamarker.compute_smaps(data_module)
        consensus_labels = np.load(stamarker.save_dir + "consensus.npy")
        ann_data.obs["Consensus_clustering"] = consensus_labels.astype(str)
        smaps_mean = np.mean(np.array(smap), axis=0)

        saliency_scores = stamarker.get_sailiency_scores(data_module)
        saliency_scores.to_csv(os.path.join(data_path, "STAMarker_SVGs.csv"))
        alpha = 1.5
        log_scores = np.log(saliency_scores)
        genes_df = log_scores.apply(lambda x: x > x.mean() + alpha * x.std(), axis=0)
        genes_list = genes_df.index[genes_df.sum(axis=1) > 0].tolist()
        # 富集分析
        names = gp.get_library_name()
        num = 1
        enr = gp.enrichr(gene_list=genes_list,  # or "./tests/data/gene_list.txt",
                         gene_sets=['Allen_Brain_Atlas_10x_scRNA_2021', 'KEGG_2019_Mouse'],
                         organism='mouse',
                         outdir=None,
                         )
        # categorical scatterplot
        ax = barplot(enr.results,
                     column="Adjusted P-value",
                     group='Gene_set',
                     size=6,
                     top_term=5,
                     figsize=(3, 3),
                     ofname=os.path.join(data_path, "STAMarker_SVGs_" + str(num) + ".png"),
                     color=['darkred', 'darkblue']  # set colors for group
                     )

        num = num + 1
        ax = dotplot(enr.res2d, title='KEGG_2019_Mouse', cmap='viridis_r', size=6, figsize=(3, 3),
                     ofname=os.path.join(data_path, "STAMarker_SVGs_" + str(num) + ".png"))

        num = num + 1
        ax = barplot(enr.res2d, title='KEGG_2019_Mouse', figsize=(3, 3), size=6, color='darkred',
                     ofname=os.path.join(data_path, "STAMarker_SVGs_" + str(num) + ".png"))

        num = num + 1
        ax = dotplot(enr.results,
                     column="Adjusted P-value",
                     x='Gene_set',  # set x axis, so you could do a multi-sample/library comparsion
                     size=6,
                     top_term=5,
                     figsize=(3, 3),
                     title="KEGG",
                     xticklabels_rot=45,
                     show_ring=True,
                     ofname=os.path.join(data_path, "STAMarker_SVGs_" + str(num) + ".png"),
                     marker='o',
                     )

        self.gene_color_type = 'viridis'

        def draw_images():
            i = 1
            from matplotlib import pyplot as plt
            plt.rcParams['font.sans-serif'] = "Arial"
            plt.rcParams["figure.figsize"] = (3, 3)
            path = 'figures/' + self.method_flag + '_' + self.data_type

            plt.rcParams["figure.figsize"] = (3, 3)
            sc.pl.spatial(ann_data, img_key="hires", color="Consensus_clustering", title="STAMarker", s=6, show=False,
                          save='STAMarker_' + str(i) + '.png')
            shutil.move('figures/showSTAMarker_' + str(i) + '.png', path + '/STAMarker_' + str(i) + '.png')
            i = i + 1
            domain_svg_list = []
            smaps = pd.DataFrame(smaps_mean[1])
            for domain_ind in range(5):
                domain_svg_list.append(select_svgs(np.log(1 + smaps), domain_ind, consensus_labels, alpha=1.25))
            upset_domains_df = from_contents({f"Spatial domain {ind}": l for ind, l in enumerate(domain_svg_list)})
            fig, ax = plt.subplots(1, 1, figsize=(3, 3))
            plt.rcParams["figure.figsize"] = (3, 3)
            df = pd.DataFrame(upset_domains_df.index.to_frame().apply(np.sum, axis=1))
            df.columns = ["counts"]
            df.index = upset_domains_df["id"]
            df_counts = df.groupby("counts")["counts"].agg("count")
            ax.bar(df_counts.index, df_counts.values)
            ax.set_xticks(df_counts.index)
            ax.set_xlabel("Number of spatial domains")
            ax.set_ylabel("Number of genes")
            plt.tight_layout()
            plt.savefig(path + '/STAMarker_' + str(i) + '.png')
            plt.close()

            genes = genes_list[:4]
            plt.rcParams["figure.figsize"] = (3, 3)
            j = 0
            i = i + 1
            for k in genes:
                sc.pl.spatial(ann_data, img_key="hires", color=k, title=k, s=6, show=False,
                              color_map=self.gene_color_type,
                              save='STAMarker_' + str(i) + "_" + str(j) + '.png')
                shutil.move('figures/showSTAMarker_' + str(i) + "_" + str(j) + '.png', path + '/STAMarker_' +
                            str(i) + "_" + str(j) + '.png')
                j = j + 1

        def scoller():
            self.figure_ybar = ttk.Scrollbar(self.figure_Frame, orient=VERTICAL, cursor='draft_small')
            self.figure_ybar.pack(side=RIGHT, fill=Y)
            self.figure_ybar.config(command=self.canvas.yview)

            self.figure_xbar = ttk.Scrollbar(self.figure_Frame, orient=HORIZONTAL, cursor='draft_small')
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
            self.canvas.configure(scrollregion=self.canvas.bbox("all"), width=1000, height=650)

        def choose_color():
            colorvalue = colorchooser.askcolor()
            color_list = []
            color = colorvalue[1]
            if len(ann_data.uns['Consensus_clustering_colors']) > self.cluster_value:
                self.cluster_value = len(ann_data.uns['Consensus_clustering_colors'])
            if color[1:3] == 'ff':
                color_list.append(color)
                list = random.sample(color_panel.FF_color, self.cluster_value)
                color_list = color_list + list
            elif color[1:3] == 'cc':
                color_list.append(color)
                list = random.sample(color_panel.CC_color, self.cluster_value)
                color_list = color_list + list
            elif color[1:3] == '99':
                color_list.append(color)
                list = random.sample(color_panel.NN_color, self.cluster_value)
                color_list = color_list + list
            elif color[1:3] == '66':
                color_list.append(color)
                list = random.sample(color_panel.SS_color, self.cluster_value)
                color_list = color_list + list
            elif color[1:3] == '33':
                color_list.append(color)
                list = random.sample(color_panel.TT_color, self.cluster_value)
                color_list = color_list + list
            elif color[1:3] == '00':
                color_list.append(color)
                list = random.sample(color_panel.ZZ_color, self.cluster_value)
                color_list = color_list + list
            else:
                color_list.append(color)
                list = random.sample(color_panel.random_color, self.cluster_value)
                color_list = color_list + list

            print(color_list)
            # if self.label_files_exit:
            #     ann_data.uns['GroundTruth_colors'] = color_list[:len(ann_data.uns['GroundTruth_colors'])]
            ann_data.uns['Consensus_clustering_colors'] = color_list[:len(ann_data.uns['Consensus_clustering_colors'])]
            self.gene_color_type = color_panel.five_gene_color[random.randint(0, 5)]
            draw_images()

            fig_path = 'figures/' + self.method_flag + '_' + self.data_type  # './figures/showSTAMarker_10XVisium'
            figures = os.listdir(fig_path)
            global image_list
            image_list = []
            for i in range(len(figures)):
                image_list.append(ttk.PhotoImage(file=fig_path + '/' + figures[i]))
                self.result_images = ttk.Label(self.show_frame, image=image_list[i])
                self.result_images.grid(row=i // 3, column=i % 3, sticky=W, pady=0)

        fig_path = 'figures/' + self.method_flag + '_' + self.data_type
        if not os.path.isdir(fig_path):
            os.mkdir(fig_path)

        images_path = glob.glob(data_path + '/*.png')
        num = 4
        ll = 0
        for i in images_path:
            plt.Figure(figsize=(3, 3))
            image = plt.imread(i)
            plt.imshow(image)
            plt.axis('off')
            plt.savefig(i)
            plt.close()
            shutil.move(i, os.path.join(fig_path, "STAMarker_GSEA_" + str(num) + '_' + str(ll) + '.png'))
            ll = ll + 1
            pass
        draw_images()
        self.figure_Frame = ttk.Frame(self.right_panel)
        self.figure_Frame.pack(side=TOP, fill=X, expand=YES)

        self.canvas = ttk.Canvas(self.figure_Frame)
        self.show_frame = ttk.Frame(self.canvas)
        scoller()

        self.set_frame = ttk.Frame(self.figure_Frame, borderwidth=2, relief="sunken")
        self.set_frame.pack(side='right', expand=YES, anchor=N)
        self.set_frame_one = ttk.Frame(self.set_frame)
        self.set_frame_one.pack(side=TOP, expand=YES)
        self.set_frame_two = ttk.Frame(self.set_frame)
        self.set_frame_two.pack(side=TOP, expand=YES)

        self.Reset = ttk.Button(self.set_frame_one, text='Reset all colors', command=choose_color, width=12)
        self.Reset.grid(row=0, column=0, sticky=W, pady=0, padx=0)

        self.domain_color_label = ttk.Label(self.set_frame_one, text='Reset domain color: ', width=20)
        self.domain_color_label.grid(row=1, column=0, sticky=W, pady=0)

        self.Reset_domain_color = ttk.Entry(self.set_frame_one, width=10)
        self.Reset_domain_color.grid(row=1, column=1, sticky=W, pady=0)
        Tooltip(self.Reset_domain_color, "Input value must be int type and in [1:cluster_name]: 2")
        self.color_update = None

        def Reset_single_domain_color():
            from tkinter import colorchooser, filedialog
            colorvalue = colorchooser.askcolor()
            color = colorvalue[1]
            print(color)
            cluster = self.Reset_domain_color.get()
            print(cluster)
            ann_data.uns['Consensus_clustering_colors'] = self.color_reset
            ann_data.uns['Consensus_clustering_colors'][int(cluster) - 1] = color

            self.color_update = ann_data.uns['mclust_colors']
            import matplotlib.pyplot as plt
            plt.rcParams["figure.figsize"] = (3, 3)
            draw_images()

            figures = os.listdir(fig_path)
            global image_list
            image_list = []
            for i in range(len(figures)):
                img = Image.open(fig_path + '/' + figures[i])
                print(img.size)
                image_list.append(ttk.PhotoImage(file=fig_path + '/' + figures[i]))
                self.result_images = ttk.Label(self.show_frame, image=image_list[i])
                self.result_images.grid(row=i // 3, column=i % 3, sticky=W, pady=0, padx=0)
                s = i

        self.Reset_domain_btn = ttk.Button(self.set_frame_one, width=10, text='Confirm',
                                           command=Reset_single_domain_color)
        self.Reset_domain_btn.grid(row=1, column=2, sticky=W, pady=10)
        self.update_image_label = ttk.Label(self.set_frame_one, text='Reset image dpi: ', width=20)
        self.update_image_label.grid(row=2, column=0, sticky=W, pady=10)
        self.update_image_scale = ttk.Entry(self.set_frame_one, width=10)
        self.update_image_scale.grid(row=2, column=1, sticky=W, pady=10)
        Tooltip(self.update_image_scale, "Input value must be int type and >= 300: 300")

        def ipdate_hd():
            dpi = self.update_image_scale.get()
            print(dpi)
            print(self.color_update)
            ann_data.uns['Consensus_clustering_colors'] = self.color_update

            import matplotlib.pyplot as plt
            plt.rcParams["figure.figsize"] = (3, 3)
            sc.set_figure_params(dpi=dpi)
            draw_images()
            figures = os.listdir(fig_path)
            print(len(figures))

        self.Reset_domain_btn = ttk.Button(self.set_frame_one, width=10, text='Save', command=ipdate_hd)
        self.Reset_domain_btn.grid(row=2, column=2, sticky=W, pady=0)

        self.gene_visualization_label = ttk.Label(self.set_frame_one, text='Input gene name: ', width=20)
        self.gene_visualization_label.grid(row=3, column=0, sticky=W, pady=0)

        self.gene_visualization_entry = ttk.Entry(self.set_frame_one, width=10)
        self.gene_visualization_entry.grid(row=3, column=1, sticky=W, pady=0)
        Tooltip(self.gene_visualization_entry, "Gene name: NEFH/ATP2B4")

        def gene_visualization():
            gene = self.gene_visualization_entry.get()
            sc.pl.spatial(ann_data, img_key="hires", color=gene, title="$" + gene + "$", show=False, save=gene + '.png')
            global img0
            photo = Image.open('figures/show' + gene + '.png')
            img0 = ImageTk.PhotoImage(photo)
            img1 = ttk.Label(self.set_frame_two, image=img0)
            img1.grid(row=0, column=0, sticky=W, pady=0)
            if os.path.exists('figures/show' + gene + '.png'):
                os.remove('figures/show' + gene + '.png')
                print("yes")
            else:
                print("error！")
            pass

        self.gene_visualization_btn = ttk.Button(self.set_frame_one, width=10, command=gene_visualization, text='Show')
        self.gene_visualization_btn.grid(row=3, column=2, sticky=W, pady=0)

        figures = os.listdir(fig_path)
        global image_list
        image_list = []
        for i in range(len(figures)):
            image_list.append(ttk.PhotoImage(file=fig_path + '/' + figures[i]))
            self.result_images = ttk.Label(self.show_frame, image=image_list[i])
            self.result_images.grid(row=i // 3, column=i % 3, sticky=W, pady=0)
            s = i

        def VIEW_3D():
            import webbrowser
            data_save_path = '../H5AD_Save/DLPFC'
            self.web_Server_Thread(os.path.join(data_save_path, 'webcache'), 8050)
            webbrowser.open('http://127.0.0.1:8050/')

        self.Reset = ttk.Button(self.set_frame_one, text='3D VIEW', command=VIEW_3D, width=10)
        self.Reset.grid(row=0, column=1, sticky=W, pady=0)

        self.pb.stop()
        self.setvar('prog-message', 'STAMarker run over!')
        self.setvar('End-time', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        EndTime = datetime.now().replace(microsecond=0)
        self.setvar('total-time-cost', EndTime - self.StartTime)

    def STAMarker_Thread(self):
        T = threading.Thread(target=self.STAMarker_SlideseqV2_Analysis)
        T.setDaemon(True)
        T.start()

    def STAMarker_10XData_Process(self):
        data_file = glob.glob(self.file_path + '/*.h5')
        truth_file = glob.glob(self.file_path + '/*.txt')
        truth_file_csv = glob.glob(self.file_path + '/*.csv')
        data_name = data_file[0].rsplit("\\", 1)[-1]
        fsize = os.path.getsize(os.path.join(self.file_path, data_name))
        adata = sc.read_visium(path=self.file_path, count_file=data_name)
        adata.var_names_make_unique()
        self.data_name = data_name.rsplit('.', 1)[-2]
        self.setvar('data_name', self.data_name)
        self.data_size = round(fsize / 1024 / 1024, 2)
        self.setvar('data_size', str(self.data_size) + 'MB')
        self.data_shape = adata.shape
        self.setvar('data_shape', str(self.data_shape[0]) + '×' + str(self.data_shape[1]))

        if len(truth_file) > 0:
            self.label_files_exit = True
            Ann_df = pd.read_csv(os.path.join(self.file_path, truth_file[0].rsplit('\\', 1)[-1]), sep='\t',
                                 header=None, index_col=0)
            Ann_df.columns = ['Ground Truth']
            adata.obs['GroundTruth'] = Ann_df.loc[adata.obs_names, 'Ground Truth']

            adata.obs.GroundTruth = adata.obs.GroundTruth.astype(str)
            adata.obs['GroundTruth'] = adata.obs['GroundTruth']
        if len(truth_file_csv) > 0:
            self.label_files_exit = True
            Ann_df = pd.read_csv(os.path.join(self.file_path, truth_file_csv[0].rsplit('\\', 1)[-1]), sep=',',
                                 index_col=0)
            Ann_df.columns = ['Ground Truth']
            adata.obs['GroundTruth'] = Ann_df.loc[adata.obs_names, 'Ground Truth']
        self.result_queue.put(adata)

    def STAMarker_Data_Analysis(self):
        if not self.result_queue.empty():
            ann_data = self.result_queue.get()
            if 'highly_variable' in ann_data.var.columns:
                ann_data = ann_data[:, ann_data.var['highly_variable']]
            else:
                sc.pp.highly_variable_genes(ann_data, flavor="seurat_v3", n_top_genes=3000)
                ann_data = ann_data[:, ann_data.var['highly_variable']]
            # if 'STAMarker' not in ann_data.uns:
            Cal_Spatial_Net(ann_data, self.rad_cutoff_value)
            Stats_Spatial_Net(ann_data)
            stamarker = STAMarker(model_dir=running_path + "/result/",
                                  in_features=ann_data.shape[1], hidden_dims=[512, 30],
                                  n_models=3, device=torch.device("cuda:0"))
            stamarker.train(ann_data, lr=1e-4, n_epochs=500, gradient_clip=5.0, use_net="Spatial_Net",
                            resume=False, plot_consensus=True, n_clusters=self.cluster_value)
            stamarker.predict(ann_data, use_net="Spatial_Net")
            output = stamarker.select_spatially_variable_genes(ann_data, use_smap="smap", alpha=1.5)
            ann_data.obs['Consensus_clustering'] = ann_data.uns['STAMarker']["consensus_labels"]
            genes_list = output['gene_list']
            smaps_mean = np.mean(np.array(ann_data.obsm['smap']), axis=0)
            data_path = running_path + "/result/" + self.method_flag + '_' + self.data_type + '_output'
            if not os.path.isdir(data_path):
                os.mkdir(data_path)
            num = 1
            enr = gp.enrichr(gene_list=genes_list,  # or "./tests/data/gene_list.txt",
                             gene_sets=['GO_Biological_Process_2018', 'KEGG_2019'],
                             organism='human',
                             outdir=None,
                             )

            ax = dotplot(enr.results,
                         column="Adjusted P-value",
                         x='Gene_set',
                         size=10,
                         top_term=5,
                         figsize=(3, 3),
                         title="Enrich",
                         xticklabels_rot=45,
                         show_ring=True,
                         ofname=os.path.join(data_path, "STAMarker_SVGs_" + str(num) + ".png"),
                         marker='o',
                         )
            num = num + 1
            ax = barplot(enr.results,
                         column="Adjusted P-value",
                         group='Gene_set',
                         size=10,
                         top_term=5,
                         figsize=(3, 3),
                         ofname=os.path.join(data_path, "STAMarker_SVGs_" + str(num) + ".png"),
                         color=['darkred', 'darkblue']
                         )
            num = num + 1
            ax = dotplot(enr.res2d, title='KEGG_2021', cmap='viridis_r', size=10, figsize=(3, 3),
                         ofname=os.path.join(data_path, "STAMarker_SVGs_" + str(num) + ".png"))
            num = num + 1
            ax = barplot(enr.res2d, title='KEGG_2021', figsize=(3, 3), color='darkred',
                         ofname=os.path.join(data_path, "STAMarker_SVGs_" + str(num) + ".png"))
            self.gene_color_type = 'viridis'

            def draw_images():
                i = 1
                temp_path = os.getcwd()
                os.chdir(test_file_path)
                from matplotlib import pyplot as plt
                plt.rcParams['font.sans-serif'] = "Arial"
                plt.rcParams["figure.figsize"] = (3, 3)
                path = test_file_path + '/figures/' + self.method_flag + '_' + self.data_type
                if self.label_files_exit:
                    sc.pl.spatial(ann_data, img_key="hires", color="GroundTruth", title="Ground Truth", s=6, show=False,
                                  save='STAMarker_' + str(i) + '.png')
                    shutil.move(test_file_path + '/figures/showSTAMarker_' + str(i) + '.png', path + '/STAMarker_' + str(i) + '.png')
                    i = i + 1

                plt.rcParams["figure.figsize"] = (3, 3)
                sc.pl.spatial(ann_data, img_key="hires", color="Consensus_clustering", title="STAMarker", s=6,
                              show=False, save='STAMarker_' + str(i) + '.png')
                shutil.move(test_file_path + '/figures/showSTAMarker_' + str(i) + '.png', path + '/STAMarker_' + str(i) + '.png')
                # i = i + 1
                # domain_svg_list = []
                # smaps = pd.DataFrame(smaps_mean)
                # for domain_ind in range(5):
                #     domain_svg_list.append(select_svgs(np.log(1 + smaps), domain_ind, ann_data.uns['STAMarker']["consensus_labels"], alpha=1.25))
                # upset_domains_df = from_contents({f"Spatial domain {ind}": l for ind, l in enumerate(domain_svg_list)})
                # fig, ax = plt.subplots(1, 1, figsize=(3, 3))
                # plt.rcParams["figure.figsize"] = (3, 3)
                # df = pd.DataFrame(upset_domains_df.index.to_frame().apply(np.sum, axis=1))
                # df.columns = ["counts"]
                # df.index = upset_domains_df["id"]
                # df_counts = df.groupby("counts")["counts"].agg("count")
                # ax.bar(df_counts.index, df_counts.values)
                # ax.set_xticks(df_counts.index)
                # ax.set_xlabel("Number of spatial domains")
                # ax.set_ylabel("Number of genes")
                # plt.tight_layout()
                # plt.savefig(path + 'STAMarker_' + str(i) + '.png')
                # plt.close()

                genes = genes_list[:4]
                plt.rcParams["figure.figsize"] = (3, 3)
                j = 0
                i = i + 1
                for k in genes:
                    sc.pl.spatial(ann_data, img_key="hires", color=k, title=k, s=6, show=False,
                                  color_map=self.gene_color_type, save='STAMarker_' + str(i) + "_" + str(j) + '.png')
                    shutil.move(test_file_path + '/figures/showSTAMarker_' + str(i) + "_" + str(j) + '.png', path + '/STAMarker_' +
                                str(i) + "_" + str(j) + '.png')
                    j = j + 1
                os.chdir(temp_path)

            def scoller():
                self.figure_ybar = ttk.Scrollbar(self.figure_Frame, orient=VERTICAL, cursor='draft_small')
                self.figure_ybar.pack(side=RIGHT, fill=Y)
                self.figure_ybar.config(command=self.canvas.yview)

                self.figure_xbar = ttk.Scrollbar(self.figure_Frame, orient=HORIZONTAL, cursor='draft_small')
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
                self.canvas.configure(scrollregion=self.canvas.bbox("all"), width=1000, height=650)

            def choose_color():
                colorvalue = colorchooser.askcolor()
                color_list = []
                color = colorvalue[1]
                # if len(ann_data.uns['Consensus_clustering_colors']) > self.cluster_value:
                #     self.cluster_value = len(ann_data.uns['Consensus_clustering_colors'])
                if color[1:3] == 'ff':
                    color_list.append(color)
                    list = random.sample(color_panel.FF_color, self.cluster_value)
                    color_list = color_list + list
                elif color[1:3] == 'cc':
                    color_list.append(color)
                    list = random.sample(color_panel.CC_color, self.cluster_value)
                    color_list = color_list + list
                elif color[1:3] == '99':
                    color_list.append(color)
                    list = random.sample(color_panel.NN_color, self.cluster_value)
                    color_list = color_list + list
                elif color[1:3] == '66':
                    color_list.append(color)
                    list = random.sample(color_panel.SS_color, self.cluster_value)
                    color_list = color_list + list
                elif color[1:3] == '33':
                    color_list.append(color)
                    list = random.sample(color_panel.TT_color, self.cluster_value)
                    color_list = color_list + list
                elif color[1:3] == '00':
                    color_list.append(color)
                    list = random.sample(color_panel.ZZ_color, self.cluster_value)
                    color_list = color_list + list
                else:
                    color_list.append(color)
                    list = random.sample(color_panel.random_color, self.cluster_value)
                    color_list = color_list + list

                print(color_list)
                if self.label_files_exit:
                    ann_data.uns['GroundTruth_colors'] = color_list[:len(ann_data.uns['GroundTruth_colors'])]
                # ann_data.uns['Consensus_clustering_colors'] = color_list[
                #                                               :len(ann_data.uns['Consensus_clustering_colors'])]
                self.gene_color_type = color_panel.five_gene_color[random.randint(0, 5)]
                self.color_reset = color_list
                draw_images()

                fig_path = test_file_path + '/figures/' + self.method_flag + '_' + self.data_type  # './figures/showSTAMarker_10XVisium'
                figures = os.listdir(fig_path)
                global image_list
                image_list = []
                for i in range(len(figures)):
                    image_list.append(ttk.PhotoImage(file=fig_path + '/' + figures[i]))
                    self.result_images = ttk.Label(self.show_frame, image=image_list[i])
                    self.result_images.grid(row=i // 3, column=i % 3, sticky=W, pady=0)

            fig_path = test_file_path + '/figures/' + self.method_flag + '_' + self.data_type
            if not os.path.isdir(fig_path):
                os.mkdir(fig_path)

            images_path = glob.glob(data_path + '/*.png')
            num = 4
            ll = 0
            for i in images_path:
                plt.Figure(figsize=(3, 3))
                image = plt.imread(i)
                plt.imshow(image)
                plt.axis('off')
                plt.savefig(i)
                plt.close()
                shutil.move(i, os.path.join(fig_path, "STAMarker_GSEA_" + str(num) + '_' + str(ll) + '.png'))
                ll = ll + 1
                pass
            draw_images()
            self.figure_Frame = ttk.Frame(self.right_panel)
            self.figure_Frame.pack(side=TOP, fill=X, expand=YES)

            self.canvas = ttk.Canvas(self.figure_Frame)
            self.show_frame = ttk.Frame(self.canvas)
            scoller()

            self.set_frame = ttk.Frame(self.figure_Frame, borderwidth=2, relief="sunken")
            self.set_frame.pack(side='right', expand=YES, anchor=N)
            self.set_frame_one = ttk.Frame(self.set_frame)
            self.set_frame_one.pack(side=TOP, expand=YES)
            self.set_frame_two = ttk.Frame(self.set_frame)
            self.set_frame_two.pack(side=TOP, expand=YES)

            self.Reset = ttk.Button(self.set_frame_one, text='Reset all colors', command=choose_color, width=12)
            self.Reset.grid(row=0, column=0, sticky=W, pady=0, padx=0)

            self.domain_color_label = ttk.Label(self.set_frame_one, text='Reset domain color: ', width=20)
            self.domain_color_label.grid(row=1, column=0, sticky=W, pady=0)

            self.Reset_domain_color = ttk.Entry(self.set_frame_one, width=10)
            self.Reset_domain_color.grid(row=1, column=1, sticky=W, pady=0)
            Tooltip(self.Reset_domain_color, "Input value must be int type and in [1:cluster_name]: 2")
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
                    img = Image.open(fig_path + '/' + figures[i])
                    print(img.size)
                    image_list.append(ttk.PhotoImage(file=fig_path + '/' + figures[i]))
                    self.result_images = ttk.Label(self.show_frame, image=image_list[i])
                    self.result_images.grid(row=i // 3, column=i % 3, sticky=W, pady=0, padx=0)
                    s = i

            self.Reset_domain_btn = ttk.Button(self.set_frame_one, width=10, text='Confirm',
                                               command=Reset_single_domain_color)
            self.Reset_domain_btn.grid(row=1, column=2, sticky=W, pady=10)
            self.update_image_label = ttk.Label(self.set_frame_one, text='Reset image dpi: ', width=20)
            self.update_image_label.grid(row=2, column=0, sticky=W, pady=10)
            self.update_image_scale = ttk.Entry(self.set_frame_one, width=10)
            self.update_image_scale.grid(row=2, column=1, sticky=W, pady=10)
            Tooltip(self.update_image_scale, "Input value must be int type and >= 300: 300")

            def ipdate_hd():
                dpi = self.update_image_scale.get()
                print(dpi)
                print(self.color_update)
                # ann_data.uns['Consensus_clustering_colors'] = self.color_update
                import matplotlib.pyplot as plt
                plt.rcParams["figure.figsize"] = (3, 3)
                sc.set_figure_params(dpi=dpi)
                draw_images()
                figures = os.listdir(fig_path)
                print(len(figures))

            self.Reset_domain_btn = ttk.Button(self.set_frame_one, width=10, text='Save', command=ipdate_hd)
            self.Reset_domain_btn.grid(row=2, column=2, sticky=W, pady=0)

            self.gene_visualization_label = ttk.Label(self.set_frame_one, text='Input gene name: ', width=20)
            self.gene_visualization_label.grid(row=3, column=0, sticky=W, pady=0)

            self.gene_visualization_entry = ttk.Entry(self.set_frame_one, width=10)
            self.gene_visualization_entry.grid(row=3, column=1, sticky=W, pady=0)
            Tooltip(self.gene_visualization_entry, "Gene name: NEFH/ATP2B4")

            def gene_visualization():
                gene = self.gene_visualization_entry.get()
                if gene in ann_data.var_names:
                    sc.pl.spatial(ann_data, img_key="hires", color=gene, title="$" + gene + "$", show=False,
                                  save=gene + '.png')
                else:
                    print(f'{gene} is not in current data!')
                global img0
                photo = Image.open(test_file_path + '/figures/show' + gene + '.png')
                img0 = ImageTk.PhotoImage(photo)
                img1 = ttk.Label(self.set_frame_two, image=img0)
                img1.grid(row=0, column=0, sticky=W, pady=0)
                if os.path.exists(test_file_path + '/figures/show' + gene + '.png'):
                    os.remove(test_file_path + '/figures/show' + gene + '.png')
                    print("yes")
                else:
                    print("error！")

            self.gene_visualization_btn = ttk.Button(self.set_frame_one, width=10, command=gene_visualization, text='Show')
            self.gene_visualization_btn.grid(row=3, column=2, sticky=W, pady=0)

            figures = os.listdir(fig_path)
            global image_list
            image_list = []
            for i in range(len(figures)):
                image_list.append(ttk.PhotoImage(file=fig_path + '/' + figures[i]))
                self.result_images = ttk.Label(self.show_frame, image=image_list[i])
                self.result_images.grid(row=i // 3, column=i % 3, sticky=W, pady=0)
                s = i

            def VIEW_3D():
                import webbrowser
                self.web_Server_Thread(os.path.join('/DLPFC/webcache'), 8050)
                webbrowser.open('http://127.0.0.1:8050/')

            self.Reset = ttk.Button(self.set_frame_one, text='3D VIEW', command=VIEW_3D, width=10)
            self.Reset.grid(row=0, column=1, sticky=W, pady=0)

            self.pb.stop()
            self.setvar('prog-message', 'STAMarker run over!')
            self.setvar('End-time', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            EndTime = datetime.now().replace(microsecond=0)
            self.setvar('total-time-cost', EndTime - self.StartTime)

    def STAMarker_Thread_two(self):
        T = threading.Thread(target=self.STAMarker_Data_Analysis)
        T.setDaemon(True)
        T.start()

    def STAMarker_Thread_one(self):
        T = threading.Thread(target=self.STAMarker_SlideSeqV2_Analysis)
        T.setDaemon(True)
        T.start()

    def STAMarker_StereoSeq_Process(self):
        h5ad_file = glob.glob(self.file_path + '/*.h5ad')
        if os.path.exists(self.file_path) and len(h5ad_file) != 0:
            self.setvar('data_name', 'Stereo-Seq Data')
            adata = sc.read(h5ad_file[0])
            adata.var_names_make_unique()
            fsize = os.path.getsize(h5ad_file[0])
            self.setvar('data_size', str(round(fsize / 1024 / 1024, 2)) + 'MB')
            self.setvar('data_shape', str(adata.shape[0]) + '×' + str(adata.shape[1]))
            self.result_queue.put(adata)
            pass
        else:
            Messagebox.show_error(title="File error", message="Make sure only one h5ad file in path!")
        pass

    def STAMarker_StereoSeq_Analysis(self):
        self.STAMarker_StereoSeq_Process()
        ann_data = self.result_queue.get()
        _path = os.path.dirname(os.path.abspath(__file__))
        _path = _path.rsplit('\\VIEW')[-2]
        _path = _path + '\\_params\\'
        config = dict()
        config.update(parse_args(_path + "model.yaml"))
        config.update(parse_args(_path + "trainer.yaml"))
        if not torch.cuda.is_available():
            config["stagate_trainer"]["gpus"] = None
            config["classifier_trainer"]["gpus"] = None

        data_module = make_spatial_data(ann_data)
        data_module.prepare_data(n_top_genes=3000, rad_cutoff=self.rad_cutoff_value, min_counts=20)

        data_path = './STAMarker_StereoSeq'
        if not os.path.isdir(data_path):
            os.mkdir(data_path)

        stamarker = STAMarker(20, data_path + "/", config)
        stamarker.train_auto_encoders(data_module)
        stamarker.clustering(data_module, "louvain", self.alpha_value)
        stamarker.consensus_clustering2(self.cluster_value)
        stamarker.train_classifiers(data_module, self.cluster_value)
        smap = stamarker.compute_smaps(data_module)
        smaps_mean = np.mean(np.array(smap), axis=0)
        consensus_labels = np.load(stamarker.save_dir + "consensus.npy")
        ann_data.obs["Consensus_clustering"] = consensus_labels.astype(str)

        saliency_scores = stamarker.get_sailiency_scores(data_module)
        # save saliency scores to file
        saliency_scores.to_csv(os.path.join(data_path, "STAMarker_SVGs.csv"))
        alpha = 1.5
        log_scores = np.log(saliency_scores)
        genes_df = log_scores.apply(lambda x: x > x.mean() + alpha * x.std(), axis=0)
        genes_list = genes_df.index[genes_df.sum(axis=1) > 0].tolist()
        # 富集分析
        # names = gp.get_library_name()
        num = 1
        enr = gp.enrichr(gene_list=genes_list,  # or "./tests/data/gene_list.txt",
                         gene_sets=['Allen_Brain_Atlas_10x_scRNA_2021', 'KEGG_2019_Mouse'],
                         organism='human',  # don't forget to set organism to the one you desired! e.g. Yeast
                         outdir=None,  # don't write to disk
                         )
        # categorical scatterplot
        ax = dotplot(enr.results,
                     column="Adjusted P-value",
                     x='Gene_set',
                     size=10,
                     top_term=5,
                     figsize=(3, 3),
                     title="KEGG",
                     xticklabels_rot=45,
                     show_ring=True,
                     ofname=os.path.join(data_path, "STAMarker_SVGs_" + str(num) + ".png"),
                     marker='o',
                     )
        num = num + 1
        ax = barplot(enr.results,
                     column="Adjusted P-value",
                     group='Gene_set',
                     size=10,
                     top_term=5,
                     figsize=(3, 3),
                     ofname=os.path.join(data_path, "STAMarker_SVGs_" + str(num) + ".png"),
                     color=['darkred', 'darkblue']
                     )
        num = num + 1
        ax = dotplot(enr.res2d, title='KEGG_2019_Mouse', cmap='viridis_r', size=10, figsize=(3, 3),
                     ofname=os.path.join(data_path, "STAMarker_SVGs_" + str(num) + ".png"))
        num = num + 1
        ax = barplot(enr.res2d, title='KEGG_2019_Mouse', figsize=(3, 3), color='darkred',
                     ofname=os.path.join(data_path, "STAMarker_SVGs_" + str(num) + ".png"))

        # def draw_images():
        #     i = 1
        #     from matplotlib import pyplot as plt
        #     plt.rcParams['font.sans-serif'] = "Arial"
        #     plt.rcParams["figure.figsize"] = (3, 3)
        #     sc.pl.spatial(ann_data, img_key="hires", color="log1p_total_counts", title="Main Tissue Area", s=6,
        #                   show=False, save='STAMarker_StereoSeq/STAMarker_' + str(i) + '.png')
        #     i = i + 1
        #     plt.rcParams["figure.figsize"] = (3, 3)
        #     sc.pl.spatial(ann_data, img_key="hires", color="Consensus_clustering", title="STAMarker", s=6, show=False,
        #                   save='STAMarker_StereoSeq/STAMarker_' + str(i) + '.png')
        #     i = i + 1
        #     domain_svg_list = []
        #     for domain_ind in range(5):
        #         domain_svg_list.append(select_svgs(np.log(1 + smap), domain_ind, consensus_labels, alpha=1.25))
        #     upset_domains_df = from_contents({f"Spatial domain {ind}": l for ind, l in enumerate(domain_svg_list)})
        #     fig, ax = plt.subplots(1, 1, figsize=(3, 3))
        #     plt.rcParams["figure.figsize"] = (3, 3)
        #     df = pd.DataFrame(upset_domains_df.index.to_frame().apply(np.sum, axis=1))
        #     df.columns = ["counts"]
        #     df.index = upset_domains_df["id"]
        #     df_counts = df.groupby("counts")["counts"].agg("count")
        #     ax.bar(df_counts.index, df_counts.values)
        #     ax.set_xticks(df_counts.index)
        #     ax.set_xlabel("Number of spatial domains")
        #     ax.set_ylabel("Number of genes")
        #     plt.tight_layout()
        #     plt.savefig('./figures/showSTAMarker_StereoSeq/STAMarker_' + str(i) + '.png')
        #     plt.close()
        #
        #     genes = genes_list[:4]
        #     plt.rcParams["figure.figsize"] = (3, 3)
        #     j = 0
        #     i = i + 1
        #     for k in genes:
        #         sc.pl.spatial(ann_data, img_key="hires", color=k, title=k, s=6, show=False,
        #                       save='STAMarker_StereoSeq/STAMarker_' + str(i) + "_" + str(j) + '.png')
        #         j = j + 1
        #
        # def scoller():
        #     self.figure_ybar = ttk.Scrollbar(self.figure_Frame, orient=VERTICAL, cursor='draft_small')
        #     self.figure_ybar.pack(side=RIGHT, fill=Y)
        #     self.figure_ybar.config(command=self.canvas.yview)
        #     self.canvas.configure(yscrollcommand=self.figure_ybar.set)
        #     self.canvas.config(yscrollincrement=1)  # 设置滚动条的步长
        #     self.canvas.pack(side=LEFT, expand=YES, fill=BOTH, anchor=NW)
        #     self.canvas.create_window((0, 0), window=self.show_frame, anchor=N)
        #     self.show_frame.bind("<Configure>", onFrameconfigure)
        #
        # def onFrameconfigure(event):
        #     # width=1450 -> width=1500
        #     self.canvas.configure(scrollregion=self.canvas.bbox("all"), width=2200, height=650)
        #
        # def choose_color():
        #     colorvalue = colorchooser.askcolor()
        #     color_list = []
        #     color = colorvalue[1]
        #     ls = self.cluster_value
        #
        #     if color[1:3] == 'ff':
        #         color_list.append(color)
        #         list = random.sample(color_panel.FF_color, ls)
        #         color_list = color_list + list
        #     elif color[1:3] == 'cc':
        #         color_list.append(color)
        #         list = random.sample(color_panel.CC_color, ls)
        #         color_list = color_list + list
        #     elif color[1:3] == '99':
        #         color_list.append(color)
        #         list = random.sample(color_panel.NN_color, ls)
        #         color_list = color_list + list
        #     elif color[1:3] == '66':
        #         color_list.append(color)
        #         list = random.sample(color_panel.SS_color, ls)
        #         color_list = color_list + list
        #     elif color[1:3] == '33':
        #         color_list.append(color)
        #         list = random.sample(color_panel.TT_color, ls)
        #         color_list = color_list + list
        #     elif color[1:3] == '00':
        #         color_list.append(color)
        #         list = random.sample(color_panel.ZZ_color, ls)
        #         color_list = color_list + list
        #     else:
        #         color_list.append(color)
        #         list = random.sample(color_panel.random_color, ls)
        #         color_list = color_list + list
        #
        #     print(color_list)
        #
        #     # ann_data.uns['GroundTruth_colors'] = color_list[:len(ann_data.uns['GroundTruth_colors'])]
        #     ann_data.uns['Consensus_clustering_colors'] = color_list[:len(ann_data.uns['Consensus_clustering_colors'])]
        #     draw_images()
        #
        #     fig_path = './figures/showSTAMarker_StereoSeq'
        #     figures = os.listdir(fig_path)
        #     global image_list
        #     image_list = []
        #     for i in range(len(figures)):
        #         image_list.append(ttk.PhotoImage(file=fig_path + '/' + figures[i]))
        #         self.result_images = ttk.Label(self.show_frame, image=image_list[i])
        #         self.result_images.grid(row=i // 3, column=i % 3, sticky=W, pady=0)
        #
        # fig_path = './figures/showSTAMarker_StereoSeq'
        # if not os.path.isdir(fig_path):
        #     os.mkdir(fig_path)
        #
        # images_path = glob.glob(data_path + '/*.png')
        # num = 4
        # ll = 0
        # for i in images_path:
        #     plt.Figure(figsize=(3, 3))
        #     image = plt.imread(i)
        #     plt.imshow(image)
        #     plt.axis('off')
        #     plt.savefig(i)
        #     plt.close()
        #     shutil.move(i, os.path.join(fig_path, "STAMarker_GSEA_" + str(num) + '_' + str(ll) + '.png'))
        #     ll = ll + 1
        #     pass
        # draw_images()
        # self.figure_Frame = ttk.Frame(self.right_panel)
        # self.figure_Frame.pack(side=TOP, fill=X, expand=YES)
        #
        # self.canvas = ttk.Canvas(self.figure_Frame)
        # self.show_frame = ttk.Frame(self.canvas)
        # scoller()
        #
        # figures = os.listdir(fig_path)
        # global image_list
        # image_list = []
        # for i in range(len(figures)):
        #     image_list.append(ttk.PhotoImage(file=fig_path + '/' + figures[i]))
        #     self.result_images = ttk.Label(self.show_frame, image=image_list[i])
        #     self.result_images.grid(row=i // 3, column=i % 3, sticky=W, pady=0)
        #     s = i
        #
        # self.color_choose = ttk.Button(self.show_frame, text='Reset colors', command=choose_color, width=10)
        # self.color_choose.grid(row=s // 3 + 1, column=1, sticky=W, pady=0)
        #
        # def Reset():
        #     python = sys.executable
        #     os.execl(python, python, *sys.argv)
        #
        # self.Reset = ttk.Button(self.show_frame, text='Reset STABox', command=Reset, width=12)
        # self.Reset.grid(row=s // 3 + 1, column=2, sticky=W, pady=0)
        #
        # self.pb.stop()
        # self.setvar('prog-message', 'STAMarker run over!')
        # self.setvar('End-time', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        # EndTime = datetime.now().replace(microsecond=0)
        # self.setvar('total-time-cost', EndTime - self.StartTime)
        # pass
        self.gene_color_type = 'viridis'

        def draw_images():
            i = 1
            from matplotlib import pyplot as plt
            plt.rcParams['font.sans-serif'] = "Arial"
            plt.rcParams["figure.figsize"] = (3, 3)
            path = 'figures/' + self.method_flag + '_' + self.data_type
            # if self.label_files_exit:
            #     sc.pl.spatial(ann_data, img_key="hires", color="GroundTruth", title="Ground Truth", s=6, show=False,
            #                   save='STAMarker_' + str(i) + '.png')
            #     shutil.move('figures/showSTAMarker_' + str(i) + '.png', path + '/STAMarker_' + str(i) + '.png')
            #     i = i + 1

            plt.rcParams["figure.figsize"] = (3, 3)
            sc.pl.spatial(ann_data, img_key="hires", color="Consensus_clustering", title="STAMarker", s=6, show=False,
                          save='STAMarker_' + str(i) + '.png')
            shutil.move('figures/showSTAMarker_' + str(i) + '.png', path + '/STAMarker_' + str(i) + '.png')
            i = i + 1
            domain_svg_list = []
            smaps = pd.DataFrame(smaps_mean[1])
            for domain_ind in range(5):
                domain_svg_list.append(select_svgs(np.log(1 + smaps), domain_ind, consensus_labels, alpha=1.25))
            upset_domains_df = from_contents({f"Spatial domain {ind}": l for ind, l in enumerate(domain_svg_list)})
            fig, ax = plt.subplots(1, 1, figsize=(3, 3))
            plt.rcParams["figure.figsize"] = (3, 3)
            df = pd.DataFrame(upset_domains_df.index.to_frame().apply(np.sum, axis=1))
            df.columns = ["counts"]
            df.index = upset_domains_df["id"]
            df_counts = df.groupby("counts")["counts"].agg("count")
            ax.bar(df_counts.index, df_counts.values)
            ax.set_xticks(df_counts.index)
            ax.set_xlabel("Number of spatial domains")
            ax.set_ylabel("Number of genes")
            plt.tight_layout()
            plt.savefig(path + '/STAMarker_' + str(i) + '.png')
            plt.close()

            genes = genes_list[:4]
            plt.rcParams["figure.figsize"] = (3, 3)
            j = 0
            i = i + 1
            for k in genes:
                sc.pl.spatial(ann_data, img_key="hires", color=k, title=k, s=6, show=False,
                              color_map=self.gene_color_type,
                              save='STAMarker_' + str(i) + "_" + str(j) + '.png')
                shutil.move('figures/showSTAMarker_' + str(i) + "_" + str(j) + '.png', path + '/STAMarker_' +
                            str(i) + "_" + str(j) + '.png')
                j = j + 1

        def scoller():
            self.figure_ybar = ttk.Scrollbar(self.figure_Frame, orient=VERTICAL, cursor='draft_small')
            self.figure_ybar.pack(side=RIGHT, fill=Y)
            self.figure_ybar.config(command=self.canvas.yview)

            self.figure_xbar = ttk.Scrollbar(self.figure_Frame, orient=HORIZONTAL, cursor='draft_small')
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
            self.canvas.configure(scrollregion=self.canvas.bbox("all"), width=1000, height=650)

        def choose_color():
            colorvalue = colorchooser.askcolor()
            color_list = []
            color = colorvalue[1]
            if len(ann_data.uns['Consensus_clustering_colors']) > self.cluster_value:
                self.cluster_value = len(ann_data.uns['Consensus_clustering_colors'])
            if color[1:3] == 'ff':
                color_list.append(color)
                list = random.sample(color_panel.FF_color, self.cluster_value)
                color_list = color_list + list
            elif color[1:3] == 'cc':
                color_list.append(color)
                list = random.sample(color_panel.CC_color, self.cluster_value)
                color_list = color_list + list
            elif color[1:3] == '99':
                color_list.append(color)
                list = random.sample(color_panel.NN_color, self.cluster_value)
                color_list = color_list + list
            elif color[1:3] == '66':
                color_list.append(color)
                list = random.sample(color_panel.SS_color, self.cluster_value)
                color_list = color_list + list
            elif color[1:3] == '33':
                color_list.append(color)
                list = random.sample(color_panel.TT_color, self.cluster_value)
                color_list = color_list + list
            elif color[1:3] == '00':
                color_list.append(color)
                list = random.sample(color_panel.ZZ_color, self.cluster_value)
                color_list = color_list + list
            else:
                color_list.append(color)
                list = random.sample(color_panel.random_color, self.cluster_value)
                color_list = color_list + list

            print(color_list)
            # if self.label_files_exit:
            #     ann_data.uns['GroundTruth_colors'] = color_list[:len(ann_data.uns['GroundTruth_colors'])]
            ann_data.uns['Consensus_clustering_colors'] = color_list[:len(ann_data.uns['Consensus_clustering_colors'])]
            self.gene_color_type = color_panel.five_gene_color[random.randint(0, 5)]
            draw_images()

            fig_path = 'figures/' + self.method_flag + '_' + self.data_type  # './figures/showSTAMarker_10XVisium'
            figures = os.listdir(fig_path)
            global image_list
            image_list = []
            for i in range(len(figures)):
                image_list.append(ttk.PhotoImage(file=fig_path + '/' + figures[i]))
                self.result_images = ttk.Label(self.show_frame, image=image_list[i])
                self.result_images.grid(row=i // 3, column=i % 3, sticky=W, pady=0)

        fig_path = 'figures/' + self.method_flag + '_' + self.data_type
        if not os.path.isdir(fig_path):
            os.mkdir(fig_path)

        images_path = glob.glob(data_path + '/*.png')
        num = 4
        ll = 0
        for i in images_path:
            plt.Figure(figsize=(3, 3))
            image = plt.imread(i)
            plt.imshow(image)
            plt.axis('off')
            plt.savefig(i)
            plt.close()
            shutil.move(i, os.path.join(fig_path, "STAMarker_GSEA_" + str(num) + '_' + str(ll) + '.png'))
            ll = ll + 1
            pass
        draw_images()
        self.figure_Frame = ttk.Frame(self.right_panel)
        self.figure_Frame.pack(side=TOP, fill=X, expand=YES)

        self.canvas = ttk.Canvas(self.figure_Frame)
        self.show_frame = ttk.Frame(self.canvas)
        scoller()

        self.set_frame = ttk.Frame(self.figure_Frame, borderwidth=2, relief="sunken")
        self.set_frame.pack(side='right', expand=YES, anchor=N)
        self.set_frame_one = ttk.Frame(self.set_frame)
        self.set_frame_one.pack(side=TOP, expand=YES)
        self.set_frame_two = ttk.Frame(self.set_frame)
        self.set_frame_two.pack(side=TOP, expand=YES)

        self.Reset = ttk.Button(self.set_frame_one, text='Reset all colors', command=choose_color, width=12)
        self.Reset.grid(row=0, column=0, sticky=W, pady=0, padx=0)

        self.domain_color_label = ttk.Label(self.set_frame_one, text='Reset domain color: ', width=20)
        self.domain_color_label.grid(row=1, column=0, sticky=W, pady=0)

        self.Reset_domain_color = ttk.Entry(self.set_frame_one, width=10)
        self.Reset_domain_color.grid(row=1, column=1, sticky=W, pady=0)
        Tooltip(self.Reset_domain_color, "Input value must be int type and in [1:cluster_name]: 2")
        self.color_update = None

        def Reset_single_domain_color():
            from tkinter import colorchooser, filedialog
            colorvalue = colorchooser.askcolor()
            color = colorvalue[1]
            print(color)
            cluster = self.Reset_domain_color.get()
            print(cluster)
            ann_data.uns['Consensus_clustering_colors'] = self.color_reset
            ann_data.uns['Consensus_clustering_colors'][int(cluster) - 1] = color

            self.color_update = ann_data.uns['mclust_colors']
            import matplotlib.pyplot as plt
            plt.rcParams["figure.figsize"] = (3, 3)
            draw_images()

            figures = os.listdir(fig_path)
            global image_list
            image_list = []
            for i in range(len(figures)):
                img = Image.open(fig_path + '/' + figures[i])
                print(img.size)
                image_list.append(ttk.PhotoImage(file=fig_path + '/' + figures[i]))
                self.result_images = ttk.Label(self.show_frame, image=image_list[i])
                self.result_images.grid(row=i // 3, column=i % 3, sticky=W, pady=0, padx=0)
                s = i

        self.Reset_domain_btn = ttk.Button(self.set_frame_one, width=10, text='Confirm',
                                           command=Reset_single_domain_color)
        self.Reset_domain_btn.grid(row=1, column=2, sticky=W, pady=10)
        self.update_image_label = ttk.Label(self.set_frame_one, text='Reset image dpi: ', width=20)
        self.update_image_label.grid(row=2, column=0, sticky=W, pady=10)
        self.update_image_scale = ttk.Entry(self.set_frame_one, width=10)
        self.update_image_scale.grid(row=2, column=1, sticky=W, pady=10)
        Tooltip(self.update_image_scale, "Input value must be int type and >= 300: 300")

        def ipdate_hd():
            dpi = self.update_image_scale.get()
            print(dpi)
            print(self.color_update)
            ann_data.uns['Consensus_clustering_colors'] = self.color_update

            import matplotlib.pyplot as plt
            plt.rcParams["figure.figsize"] = (3, 3)
            sc.set_figure_params(dpi=dpi)
            draw_images()
            figures = os.listdir(fig_path)
            print(len(figures))

        self.Reset_domain_btn = ttk.Button(self.set_frame_one, width=10, text='Save', command=ipdate_hd)
        self.Reset_domain_btn.grid(row=2, column=2, sticky=W, pady=0)

        self.gene_visualization_label = ttk.Label(self.set_frame_one, text='Input gene name: ', width=20)
        self.gene_visualization_label.grid(row=3, column=0, sticky=W, pady=0)

        self.gene_visualization_entry = ttk.Entry(self.set_frame_one, width=10)
        self.gene_visualization_entry.grid(row=3, column=1, sticky=W, pady=0)
        Tooltip(self.gene_visualization_entry, "Gene name: NEFH/ATP2B4")

        def gene_visualization():
            gene = self.gene_visualization_entry.get()
            sc.pl.spatial(ann_data, img_key="hires", color=gene, title="$" + gene + "$", show=False, save=gene + '.png')
            global img0
            photo = Image.open('figures/show' + gene + '.png')
            img0 = ImageTk.PhotoImage(photo)
            img1 = ttk.Label(self.set_frame_two, image=img0)
            img1.grid(row=0, column=0, sticky=W, pady=0)
            if os.path.exists('figures/show' + gene + '.png'):
                os.remove('figures/show' + gene + '.png')
                print("yes")
            else:
                print("error！")
            pass

        self.gene_visualization_btn = ttk.Button(self.set_frame_one, width=10, command=gene_visualization, text='Show')
        self.gene_visualization_btn.grid(row=3, column=2, sticky=W, pady=0)

        # end
        figures = os.listdir(fig_path)
        global image_list
        image_list = []
        for i in range(len(figures)):
            image_list.append(ttk.PhotoImage(file=fig_path + '/' + figures[i]))
            self.result_images = ttk.Label(self.show_frame, image=image_list[i])
            self.result_images.grid(row=i // 3, column=i % 3, sticky=W, pady=0)
            s = i

        def VIEW_3D():
            import webbrowser
            data_save_path = '../H5AD_Save/DLPFC'
            self.web_Server_Thread(os.path.join(data_save_path, 'webcache'), 8050)
            webbrowser.open('http://127.0.0.1:8050/')

        self.Reset = ttk.Button(self.set_frame_one, text='3D VIEW', command=VIEW_3D, width=10)
        self.Reset.grid(row=0, column=1, sticky=W, pady=0)

        self.pb.stop()
        self.setvar('prog-message', 'STAMarker run over!')
        self.setvar('End-time', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        EndTime = datetime.now().replace(microsecond=0)
        self.setvar('total-time-cost', EndTime - self.StartTime)

    def STAMarker_Thread_three(self):
        T = threading.Thread(target=self.STAMarker_StereoSeq_Analysis)
        T.setDaemon(True)
        T.start()

    def STAGE_10XData_Analysis(self):
        adata = self.result_queue.get()
        adata.obsm["coord"] = adata.obsm["spatial"]
        functions = ['generation', 'recovery', '3d_model']
        filepath = test_file_path + '/result/' + self.method_flag + '_' + self.data_type + '_output'
        if os.path.isfile(filepath):
            os.mkdir(filepath)
        if self.functions_choose not in functions:
            self.functions_choose = functions[0]

        adata_sample, adata_stage = STAGE(
            adata,
            save_path=filepath,
            data_type=self.data_type,
            experiment=self.functions_choose,
            down_ratio=self.alpha_value,
            coord_sf=77,
            train_epoch=2000,
            batch_size=512,
            learning_rate=1e-3,
            w_recon=0.1,
            w_w=0.1,
            w_l1=0.1
        )
        sc.pp.pca(adata, n_comps=30)
        sc.pp.neighbors(adata, use_rep='X_pca')
        sc.tl.umap(adata)

        sc.pp.pca(adata_sample, n_comps=30)
        sc.pp.neighbors(adata_sample, use_rep='X_pca')
        sc.tl.umap(adata_sample)

        sc.pp.pca(adata_stage, n_comps=30)
        sc.pp.neighbors(adata_stage, use_rep='X_pca')
        sc.tl.umap(adata_stage)

        plt.rcParams['font.sans-serif'] = "Arial"
        self.gene_color_type = 'viridis'

        # adata.obsm["coord"] = adata.obsm["coord"] * (-1)
        # adata_stage.obsm["coord"] = adata_stage.obsm["coord"] * (-1)
        # adata_stage.obsm["coord"] = adata_stage.obsm["coord"] * (-1)

        def draw_images():
            i = 1
            j = 1
            from matplotlib import pyplot as plt
            plt.rcParams['font.sans-serif'] = "Arial"
            plt.rcParams["figure.figsize"] = (3, 3)
            path = test_file_path + '/figures/' + self.method_flag + '_' + self.data_type + '/'
            if self.label_files_exit:
                sc.pl.spatial(adata, img_key="hires", color="layer", title='Ground Truth', show=False,
                              save='STAGE_' + str(j) + '_' + str(i) + '.png')
                shutil.move(test_file_path + '/figures/showSTAGE_' + str(i) + '.png', path + 'STAGE_' + str(j) + '_' + str(i) + '.png')

                adata.obsm['coord'][:, 1] = adata.obsm['coord'][:, 1] * (-1)
                i = i + 1
                used_adata = adata[adata.obs['layer'].isna() == False,]
                plt.rcParams["figure.figsize"] = (3, 3)
                sc.pl.umap(used_adata, color='layer', title='Original', show=False, s=6,
                           save='STAGE_' + str(i) + '.png')
                shutil.move(test_file_path + '/figures/umapSTAGE_' + str(i) + '.png', path + 'STAGE_' + str(j) + '_' + str(i) + '.png')

                i = i + 1
                plt.rcParams["figure.figsize"] = (3, 3)
                sc.tl.paga(used_adata, groups='layer')
                sc.pl.paga(used_adata, color="layer", title='Original-PAGA', show=False,
                           save='STAGE_' + str(i) + '.png')
                shutil.move(test_file_path + '/figures/pagaSTAGE_' + str(i) + '.png', path + 'STAGE_' + str(j) + '_' + str(i) + '.png')

                i = i + 1
                used_adata_sample = adata_sample[adata_sample.obs['layer'].isna() == False,]
                plt.rcParams["figure.figsize"] = (3, 3)
                used_adata_sample.uns['layer_colors'] = used_adata.uns['layer_colors']
                sc.pl.umap(used_adata_sample, color='layer', title='Down-Sampling', show=False,
                           save='STAGE_' + str(i) + '.png')
                shutil.move(test_file_path + '//figures/umapSTAGE_' + str(i) + '.png', path + 'STAGE_' + str(j) + '_' + str(i) + '.png')

                i = i + 1
                plt.rcParams["figure.figsize"] = (3, 3)
                sc.tl.paga(used_adata_sample, groups='layer')
                sc.pl.paga(used_adata_sample, color="layer", title='Down-Sampling-PAGA', show=False,
                           save='STAGE_' + str(i) + '.png')
                shutil.move(test_file_path + '/figures/pagaSTAGE_' + str(i) + '.png', path + 'STAGE_' + str(j) + '_' + str(i) + '.png')

                i = i + 1
                used_adata_stage = adata_stage[adata_stage.obs['layer'].isna() == False,]
                plt.rcParams["figure.figsize"] = (3, 3)
                used_adata_stage.uns['layer_colors'] = used_adata.uns['layer_colors']
                sc.pl.umap(used_adata_stage, color='layer', title='Recoverd', show=False,
                           save='STAGE_' + str(i) + '.png')
                shutil.move(test_file_path + '/figures/umapSTAGE_' + str(i) + '.png', path + 'STAGE_' + str(j) + '_' + str(i) + '.png')

                i = i + 1
                plt.rcParams["figure.figsize"] = (3, 3)
                sc.tl.paga(used_adata_stage, groups='layer')
                sc.pl.paga(used_adata_stage, color="layer", title='Recoverd-PAGA', show=False,
                           save='STAGE_' + str(i) + '.png')
                shutil.move(test_file_path + '/figures/pagaSTAGE_' + str(i) + '.png', path + 'STAGE_' + str(j) + '_' + str(i) + '.png')
            j = j + 1
            i = 1
            plt.rcParams["figure.figsize"] = (3, 3)
            sc.pl.embedding(adata, basis="coord", color=self.genes, title=self.genes, s=6, show=False,
                            color_map=self.gene_color_type, save='STAGE_' + str(i) + '.png')
            shutil.move(test_file_path + '/figures/coordSTAGE_' + str(i) + '.png', path + 'STAGE_' + str(j) + '_' + str(i) + '.png')

            i = i + 1
            adata_sample.obsm['coord'][:, 1] = adata_sample.obsm['coord'][:, 1] * (-1)
            plt.rcParams["figure.figsize"] = (3, 3)
            sc.pl.embedding(adata_sample, basis="coord", color=self.genes, title=self.genes, s=6, show=False,
                            color_map=self.gene_color_type, save='STAGE_' + str(i) + '.png')
            shutil.move(test_file_path + '/figures/coordSTAGE_' + str(i) + '.png', path + 'STAGE_' + str(j) + '_' + str(i) + '.png')

            i = i + 1
            plt.rcParams["figure.figsize"] = (3, 3)
            sc.pl.embedding(adata_stage, basis="coord", color=self.genes, title=self.genes, s=6, show=False,
                            color_map=self.gene_color_type, save='STAGE_' + str(i) + '.png')
            shutil.move(test_file_path + '/figures/coordSTAGE_' + str(i) + '.png', path + 'STAGE_' + str(j) + '_' + str(i) + '.png')

        def scoller():
            self.figure_ybar = ttk.Scrollbar(self.figure_Frame, orient=VERTICAL, cursor='draft_small')
            self.figure_ybar.pack(side=RIGHT, fill=Y)
            self.figure_ybar.config(command=self.canvas.yview)

            self.figure_xbar = ttk.Scrollbar(self.figure_Frame, orient=HORIZONTAL, cursor='draft_small')
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
            self.canvas.configure(scrollregion=self.canvas.bbox("all"), width=1000, height=650)

        def choose_color():
            adata.obsm['coord'][:, 1] = adata.obsm['coord'][:, 1] * (-1)
            adata_sample.obsm['coord'][:, 1] = adata_sample.obsm['coord'][:, 1] * (-1)
            self.gene_color_type = color_panel.five_gene_color[random.randint(0, 5)]
            draw_images()

            fig_path = test_file_path + '/figures/' + self.method_flag + '_' + self.data_type
            figures = os.listdir(fig_path)
            global image_list
            image_list = []
            for i in range(len(figures)):
                image_list.append(ttk.PhotoImage(file=fig_path + '/' + figures[i]))
                self.result_images = ttk.Label(self.show_frame, image=image_list[i])
                self.result_images.grid(row=i // 4, column=i % 4, sticky=W, pady=0)

        fig_path = test_file_path + '/figures/' + self.method_flag + '_' + self.data_type
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
            image_list.append(ttk.PhotoImage(file=fig_path + '/' + figures[i]))
            self.result_images = ttk.Label(self.show_frame, image=image_list[i])
            self.result_images.grid(row=i // 4, column=i % 4, sticky=W, pady=0)
            s = i

        self.set_frame = ttk.Frame(self.figure_Frame, borderwidth=2, relief="sunken")
        self.set_frame.pack(side='right', expand=YES, anchor=N)
        self.set_frame_one = ttk.Frame(self.set_frame)
        self.set_frame_one.pack(side=TOP, expand=YES)
        self.set_frame_two = ttk.Frame(self.set_frame)
        self.set_frame_two.pack(side=TOP, expand=YES)

        self.Reset = ttk.Button(self.set_frame_one, text='Reset all colors', command=choose_color, width=12)
        self.Reset.grid(row=0, column=0, sticky=W, pady=0, padx=0)

        def Reset():
            python = sys.executable
            os.execl(python, python, *sys.argv)

        self.Reset = ttk.Button(self.set_frame_one, text='Reset STABox', command=Reset, width=12)
        self.Reset.grid(row=0, column=1, sticky=W, pady=0, padx=0)

        self.domain_color_label = ttk.Label(self.set_frame_one, text='Reset domain color: ', width=20)
        self.domain_color_label.grid(row=1, column=0, sticky=W, pady=0)

        self.Reset_domain_color = ttk.Entry(self.set_frame_one, width=10)
        self.Reset_domain_color.grid(row=1, column=1, sticky=W, pady=0)
        Tooltip(self.Reset_domain_color, "Input value must be int type and in [1:cluster_name]: 2")
        self.color_update = None

        def Reset_single_domain_color():
            from tkinter import colorchooser, filedialog
            colorvalue = colorchooser.askcolor()
            color = colorvalue[1]
            print(color)
            cluster = self.Reset_domain_color.get()
            print(cluster)
            import matplotlib.pyplot as plt
            plt.rcParams["figure.figsize"] = (3, 3)
            draw_images()
            figures = os.listdir(fig_path)
            global image_list
            image_list = []
            for i in range(len(figures)):
                img = Image.open(fig_path + '/' + figures[i])
                image_list.append(ttk.PhotoImage(file=fig_path + '/' + figures[i]))
                self.result_images = ttk.Label(self.show_frame, image=image_list[i])
                self.result_images.grid(row=i // 3, column=i % 3, sticky=W, pady=0, padx=0)
                s = i

            pass

        self.Reset_domain_btn = ttk.Button(self.set_frame_one, width=10, text='Confirm',
                                           command=Reset_single_domain_color)
        self.Reset_domain_btn.grid(row=1, column=2, sticky=W, pady=10)
        self.update_image_label = ttk.Label(self.set_frame_one, text='Reset image dpi: ', width=20)
        self.update_image_label.grid(row=2, column=0, sticky=W, pady=10)
        self.update_image_scale = ttk.Entry(self.set_frame_one, width=10)
        self.update_image_scale.grid(row=2, column=1, sticky=W, pady=10)
        Tooltip(self.update_image_scale, "Input value must be int type and >= 300: 300")

        def ipdate_hd():
            dpi = self.update_image_scale.get()
            import matplotlib.pyplot as plt
            plt.rcParams["figure.figsize"] = (3, 3)
            sc.set_figure_params(dpi=dpi)
            draw_images()
            figures = os.listdir(fig_path)
            print(len(figures))

        self.Reset_domain_btn = ttk.Button(self.set_frame_one, width=10, text='Save', command=ipdate_hd)
        self.Reset_domain_btn.grid(row=2, column=2, sticky=W, pady=0)

        self.gene_visualization_label = ttk.Label(self.set_frame_one, text='Input gene name: ', width=20)
        self.gene_visualization_label.grid(row=3, column=0, sticky=W, pady=0)

        self.gene_visualization_entry = ttk.Entry(self.set_frame_one, width=10)
        self.gene_visualization_entry.grid(row=3, column=1, sticky=W, pady=0)
        Tooltip(self.gene_visualization_entry, "Gene name: NEFH/ATP2B4")

        def gene_visualization():
            gene = self.gene_visualization_entry.get()
            if gene not in adata.var_names:
                raise
            sc.pl.spatial(adata, img_key="hires", color=gene, title="$" + gene + "$", show=False, save=gene + '.png')
            global img0
            photo = Image.open(test_file_path + '/figures/show' + gene + '.png')
            img0 = ImageTk.PhotoImage(photo)
            img1 = ttk.Label(self.set_frame_two, image=img0)
            img1.grid(row=0, column=0, sticky=W, pady=0)
            if os.path.exists(test_file_path + '/figures/show' + gene + '.png'):
                os.remove(test_file_path + '/figures/show' + gene + '.png')
                print("yes")
            else:
                print("error！")
            pass

        self.gene_visualization_btn = ttk.Button(self.set_frame_one, width=10, command=gene_visualization, text='Show')
        self.gene_visualization_btn.grid(row=3, column=2, sticky=W, pady=0)

        self.pb.stop()
        self.setvar('prog-message', 'STAGE run over!')
        self.setvar('End-time', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        EndTime = datetime.now().replace(microsecond=0)
        self.setvar('total-time-cost', EndTime - self.StartTime)

    def STAGE_Thread(self):
        T = threading.Thread(target=self.STAGE_10XData_Analysis)
        T.setDaemon(True)
        T.start()
        pass

    def STAGE_10XMouse_BrainData_Process(self):
        try:
            data_file = glob.glob(self.Visium_file_path + '/*.h5')
            data_name = data_file[0].rsplit("\\", 1)[-1]
            self.data_name = data_name.rsplit('.', 1)[-2]
            self.setvar('data_name', self.data_name)
            fsize = os.path.getsize(os.path.join(self.Visium_file_path, data_name))
            adata = sc.read_visium(path=self.Visium_file_path, count_file=data_name)
            adata.var_names_make_unique()

            self.data_size = round(fsize / 1024 / 1024, 2)
            self.setvar('data_size', str(self.data_size) + 'MB')
            self.data_shape = adata.shape
            self.setvar('data_shape', str(self.data_shape[0]) + '×' + str(self.data_shape[1]))

            adata.obsm["coord"] = adata.obs.loc[:, ['array_col', 'array_row']].to_numpy()
            sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
            self.result_queue.put(adata)
        except:
            Messagebox.show_info('Check whether the 10X Visium files exists!')
        pass
        pass

    def STAGE_10XMouse_BrainData_Analysis(self):
        self.STAGE_10XMouse_BrainData_Process()
        adata = self.result_queue.get()
        filepath = './GATE_OUT_Mouse_Brain_Files'
        if os.path.isfile(filepath):
            os.mkdir(filepath)
        adata_stage = STAGE(
            adata,
            save_path=filepath,
            data_type='10x',
            experiment='generation',
            coord_sf=77,
            train_epoch=2000,
            batch_size=512,
            learning_rate=1e-3,
            w_recon=0.1,
            w_w=0.1,
            w_l1=0.1
        )

        adata.obsm['coord'][:, 1] = adata.obsm['coord'][:, 1] * (-1)
        adata_stage.obsm['coord'][:, 1] = adata_stage.obsm['coord'][:, 1] * (-1)

        show_gene = ["Hpca", "Camk2n1", "Mast3", "Gse1", "Sipa1l3", "Sorl1", self.genes]

        def draw_images():
            j = 1
            from matplotlib import pyplot as plt
            plt.rcParams['font.sans-serif'] = "Arial"

            for i in range(len(show_gene)):
                plt.rcParams["figure.figsize"] = (3, 3)
                sc.pl.embedding(adata, basis="coord", color=show_gene[i], title=show_gene[i], s=6, show=False,
                                save='STAGE_' + str(i) + '.png')
                shutil.move('./figures/coordSTAGE_' + str(i) + '.png',
                            './figures/showSTAGE_10X_Mouse_Brain_Visium/STAGE_' + str(j) + '_' + str(i) + '.png')
            j = j + 1
            for i in range(len(show_gene)):
                plt.rcParams["figure.figsize"] = (3, 3)
                sc.pl.embedding(adata_stage, basis="coord", color=show_gene[i], title=show_gene[i], s=6, show=False,
                                save='STAGE_' + str(i) + '.png')
                shutil.move('./figures/coordSTAGE_' + str(i) + '.png',
                            './figures/showSTAGE_10X_Mouse_Brain_Visium/STAGE_' + str(j) + '_' + str(i) + '.png')

        def scoller():
            self.figure_ybar = ttk.Scrollbar(self.figure_Frame, orient=VERTICAL, cursor='draft_small')
            self.figure_ybar.pack(side=RIGHT, fill=Y)
            self.figure_ybar.config(command=self.canvas.yview)
            self.canvas.configure(yscrollcommand=self.figure_ybar.set)
            self.canvas.config(yscrollincrement=1)  # 设置滚动条的步长
            self.canvas.pack(side=LEFT, expand=YES, fill=BOTH, anchor=NW)
            self.canvas.create_window((0, 0), window=self.show_frame, anchor=N)
            self.show_frame.bind("<Configure>", onFrameconfigure)

        def onFrameconfigure(event):
            # width=1450 -> width=1500
            self.canvas.configure(scrollregion=self.canvas.bbox("all"), width=1500, height=650)

        def choose_color():
            colorvalue = colorchooser.askcolor()
            color_list = []
            color = colorvalue[1]
            print(color)
            ls = len(show_gene)

            if color[1:3] == 'ff':
                color_list.append(color)
                list = random.sample(color_panel.FF_color, ls)
                color_list = color_list + list
            elif color[1:3] == 'cc':
                color_list.append(color)
                list = random.sample(color_panel.CC_color, ls)
                color_list = color_list + list
            elif color[1:3] == '99':
                color_list.append(color)
                list = random.sample(color_panel.NN_color, ls)
                color_list = color_list + list
            elif color[1:3] == '66':
                color_list.append(color)
                list = random.sample(color_panel.SS_color, ls)
                color_list = color_list + list
            elif color[1:3] == '33':
                color_list.append(color)
                list = random.sample(color_panel.TT_color, ls)
                color_list = color_list + list
            elif color[1:3] == '00':
                color_list.append(color)
                list = random.sample(color_panel.ZZ_color, ls)
                color_list = color_list + list
            else:
                color_list.append(color)
                list = random.sample(color_panel.random_color, ls)
                color_list = color_list + list

            print(color_list)
            draw_images()

            fig_path = './figures/showSTAGE_10X_Mouse_Brain_Visium'
            figures = os.listdir(fig_path)
            global image_list
            image_list = []
            for i in range(len(figures)):
                image_list.append(ttk.PhotoImage(file=fig_path + '/' + figures[i]))
                self.result_images = ttk.Label(self.show_frame, image=image_list[i])
                self.result_images.grid(row=i // 3, column=i % 3, sticky=W, pady=0)

        fig_path = './figures/showSTAGE_10X_Mouse_Brain_Visium'
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
            image_list.append(ttk.PhotoImage(file=fig_path + '/' + figures[i]))
            self.result_images = ttk.Label(self.show_frame, image=image_list[i])
            self.result_images.grid(row=i // 3, column=i % 3, sticky=W, pady=0)
            s = i

        self.color_choose = ttk.Button(self.show_frame, text='Reset colors', command=choose_color, width=12)
        self.color_choose.grid(row=s // 3 + 1, column=1, sticky=W, pady=0)

        def Reset():
            python = sys.executable
            os.execl(python, python, *sys.argv)

        self.Reset = ttk.Button(self.show_frame, text='Reset STABox', command=Reset, width=12)
        self.Reset.grid(row=s // 3 + 1, column=2, sticky=W, pady=0)

        self.pb.stop()
        self.setvar('prog-message', 'STAGE run over!')
        self.setvar('End-time', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        EndTime = datetime.now().replace(microsecond=0)
        self.setvar('total-time-cost', EndTime - self.StartTime)
        pass

    def STAGE_Thread_one(self):
        T = threading.Thread(target=self.STAGE_10XMouse_BrainData_Analysis)
        T.setDaemon(True)
        T.start()

    def STAGE_10XBreast_Cancer_Process(self):
        try:
            data_file = glob.glob(self.Visium_file_path + '/*.h5')
            data_name = data_file[0].rsplit("\\", 1)[-1]
            self.data_name = '10XBreast-Cancer'
            self.setvar('data_name', self.data_name)
            fsize = os.path.getsize(os.path.join(self.Visium_file_path, data_name))
            adata = sc.read_visium(path=self.Visium_file_path, count_file=data_name)
            adata.var_names_make_unique()

            self.data_size = round(fsize / 1024 / 1024, 2)
            self.setvar('data_size', str(self.data_size) + 'MB')
            self.data_shape = adata.shape
            self.setvar('data_shape', str(self.data_shape[0]) + '×' + str(self.data_shape[1]))

            adata.obsm["coord"] = adata.obs.loc[:, ['array_col', 'array_row']].to_numpy()

            sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)

            adata_sc_path = self.Visium_file_path + '\\Wu_etal_2021_BRCA_scRNASeq'
            adata_sc = sc.read_mtx(os.path.join(adata_sc_path, "count_matrix_sparse.mtx"))
            adata_sc = sc.AnnData(adata_sc.X.T)
            gene = pd.read_csv(os.path.join(adata_sc_path, "count_matrix_genes.tsv"), index_col=0, header=None)
            barcode = pd.read_csv(os.path.join(adata_sc_path, "count_matrix_barcodes.tsv"), index_col=0, header=None)
            meta = pd.read_csv(os.path.join(adata_sc_path, "metadata.csv"), index_col=0)
            adata_sc.obs.index = barcode.index.values
            adata_sc.var.index = gene.index.values
            adata_sc.obs.loc[meta.index, meta.columns] = meta

            sc.pp.highly_variable_genes(adata_sc, flavor="seurat_v3", n_top_genes=3000)
            sc.pp.normalize_total(adata_sc, target_sum=1e4)
            sc.pp.log1p(adata_sc)

            sc.tl.rank_genes_groups(adata_sc, 'celltype_major', method='wilcoxon')
            DE_df = sc.get.rank_genes_groups_df(adata_sc, group=None, log2fc_min=4)
            DE_df = DE_df[DE_df.pvals_adj < 0.05]

            self.result_queue.put(adata)
            self.result_queue.put(DE_df)
        except:
            Messagebox.show_info('Check whether the 10X Visium files exists!')
        pass

    def STAGE_10XBreast_Cancer_Analysis(self):
        self.STAGE_10XBreast_Cancer_Process()
        adata = self.result_queue.get()
        DE_df = self.result_queue.get()

        filepath = './GATE_OUT_Breast_Cancer'
        if os.path.isfile(filepath):
            os.mkdir(filepath)

        adata_stage = STAGE(
            adata,
            save_path=filepath,
            data_type='10x',
            experiment='generation',
            coord_sf=77,
            train_epoch=2000,
            batch_size=512,
            learning_rate=1e-3,
            w_recon=0.1,
            w_w=0.1,
            w_l1=0.1
        )

        DE_df = DE_df[DE_df.names.isin(adata_stage.var.index)]
        expr_raw = pd.DataFrame(sp.coo_matrix(adata.X).todense())
        expr_raw.columns = adata.var.index
        expr_raw.index = adata.obs.index

        expr_stage = pd.DataFrame(sp.coo_matrix(adata_stage.X).todense())
        expr_stage.columns = adata_stage.var.index
        expr_stage.index = adata_stage.obs.index

        for celltype in DE_df.group.unique():
            adata.obs[celltype] = expr_raw.loc[:, DE_df.names[DE_df.group == celltype]].mean(axis=1)
            adata_stage.obs[celltype] = expr_stage.loc[:, DE_df.names[DE_df.group == celltype]].mean(axis=1)

        adata.obsm['coord'][:, 1] = adata.obsm['coord'][:, 1] * (-1)
        adata_stage.obsm['coord'][:, 1] = adata_stage.obsm['coord'][:, 1] * (-1)

        # tissue segmentation
        model_raw = KMeans(n_clusters=3)
        model_stage = KMeans(n_clusters=3)
        model_raw.fit(adata.obs[DE_df.group.unique()])
        model_stage.fit(adata_stage.obs[DE_df.group.unique()])

        adata.obs["K-means"] = model_raw.predict(adata.obs[DE_df.group.unique()])
        adata_stage.obs["K-means"] = model_stage.predict(adata_stage.obs[DE_df.group.unique()])
        adata.obs["segmentation"] = adata.obs["K-means"].replace([0, 1, 2], ["R1", "R2", "R3"])
        adata_stage.obs["segmentation"] = adata_stage.obs["K-means"].replace([0, 1, 2], ["S1", "S2", "S3"])
        adata.uns['segmentation_colors'] = ['#0000FF', '#FFFF00', '#FF0000']
        adata_stage.uns['segmentation_colors'] = ['#0000FF', '#FFFF00', '#FF0000']

        # self.genes = "LYZ"
        show_celltype = ["T-cells", "Cancer Epithelial", "CAFs", "Myeloid"]
        show_gene = ["CD3D", "KRT18", "COL1A1", self.genes]

        def draw_images():
            j = 1
            from matplotlib import pyplot as plt
            plt.rcParams['font.sans-serif'] = "Arial"

            for i in range(len(show_celltype)):
                plt.rcParams["figure.figsize"] = (3, 3)
                sc.pl.embedding(adata, basis="coord", color=show_celltype[i], title=show_celltype[i], s=6, show=False,
                                save='STAGE_' + str(i) + '.png')
                shutil.move('./figures/coordSTAGE_' + str(i) + '.png',
                            './figures/showSTAGE_10X_Breast_Cancer_Visium/STAGE_' + str(j) + '_' + str(i) + '.png')
            j = j + 1
            for i in range(len(show_celltype)):
                plt.rcParams["figure.figsize"] = (3, 3)
                sc.pl.embedding(adata_stage, basis="coord", color=show_celltype[i], title=show_celltype[i], s=6,
                                show=False,
                                save='STAGE_' + str(i) + '.png')
                shutil.move('./figures/coordSTAGE_' + str(i) + '.png',
                            './figures/showSTAGE_10X_Breast_Cancer_Visium/STAGE_' + str(j) + '_' + str(i) + '.png')
            j = j + 1
            for i in range(len(show_gene)):
                plt.rcParams["figure.figsize"] = (3, 3)
                sc.pl.embedding(adata, basis="coord", color=show_gene[i], title=show_gene[i], s=6, show=False,
                                save='STAGE_' + str(i) + '.png')
                shutil.move('./figures/coordSTAGE_' + str(i) + '.png',
                            './figures/showSTAGE_10X_Breast_Cancer_Visium/STAGE_' + str(j) + '_' + str(i) + '.png')
            j = j + 1
            for i in range(len(show_gene)):
                plt.rcParams["figure.figsize"] = (3, 3)
                sc.pl.embedding(adata_stage, basis="coord", color=show_gene[i], title=show_gene[i], s=6, show=False,
                                save='STAGE_' + str(i) + '.png')
                shutil.move('./figures/coordSTAGE_' + str(i) + '.png',
                            './figures/showSTAGE_10X_Breast_Cancer_Visium/STAGE_' + str(j) + '_' + str(i) + '.png')

            j = j + 1
            plt.rcParams["figure.figsize"] = (3, 3)
            sc.pl.embedding(adata, basis="coord", color='segmentation', title='Tissue Segmentation', s=6, show=False,
                            save='STAGE_' + str(i) + '.png')
            shutil.move('./figures/coordSTAGE_' + str(i) + '.png',
                        './figures/showSTAGE_10X_Breast_Cancer_Visium/STAGE_' + str(j) + '_' + str(i) + '.png')

            j = j + 1
            plt.rcParams["figure.figsize"] = (3, 3)
            sc.pl.embedding(adata, basis="coord", color='segmentation', title='Tissue Segmentation', s=6, show=False,
                            save='STAGE_' + str(i) + '.png')
            shutil.move('./figures/coordSTAGE_' + str(i) + '.png',
                        './figures/showSTAGE_10X_Breast_Cancer_Visium/STAGE_' + str(j) + '_' + str(i) + '.png')

        def scoller():
            self.figure_ybar = ttk.Scrollbar(self.figure_Frame, orient=VERTICAL, cursor='draft_small')
            self.figure_ybar.pack(side=RIGHT, fill=Y)
            self.figure_ybar.config(command=self.canvas.yview)
            self.canvas.configure(yscrollcommand=self.figure_ybar.set)
            self.canvas.config(yscrollincrement=1)  # 设置滚动条的步长
            self.canvas.pack(side=LEFT, expand=YES, fill=BOTH, anchor=NW)
            self.canvas.create_window((0, 0), window=self.show_frame, anchor=N)
            self.show_frame.bind("<Configure>", onFrameconfigure)

        def onFrameconfigure(event):
            # width=1450 -> width=1500
            self.canvas.configure(scrollregion=self.canvas.bbox("all"), width=1500, height=650)

        def choose_color():
            colorvalue = colorchooser.askcolor()
            color_list = []
            color = colorvalue[1]
            print(color)
            ls = len(show_gene)

            if color[1:3] == 'ff':
                color_list.append(color)
                list = random.sample(color_panel.FF_color, ls)
                color_list = color_list + list
            elif color[1:3] == 'cc':
                color_list.append(color)
                list = random.sample(color_panel.CC_color, ls)
                color_list = color_list + list
            elif color[1:3] == '99':
                color_list.append(color)
                list = random.sample(color_panel.NN_color, ls)
                color_list = color_list + list
            elif color[1:3] == '66':
                color_list.append(color)
                list = random.sample(color_panel.SS_color, ls)
                color_list = color_list + list
            elif color[1:3] == '33':
                color_list.append(color)
                list = random.sample(color_panel.TT_color, ls)
                color_list = color_list + list
            elif color[1:3] == '00':
                color_list.append(color)
                list = random.sample(color_panel.ZZ_color, ls)
                color_list = color_list + list
            else:
                color_list.append(color)
                list = random.sample(color_panel.random_color, ls)
                color_list = color_list + list

            print(color_list)
            adata.uns['segmentation_colors'] = color_list[:len(adata.uns['segmentation_colors'])]
            adata_stage.uns['segmentation_colors'] = color_list[:len(adata_stage.uns['segmentation_colors'])]
            draw_images()

            fig_path = './figures/showSTAGE_10X_Breast_Cancer_Visium'
            figures = os.listdir(fig_path)
            global image_list
            image_list = []
            for i in range(len(figures)):
                image_list.append(ttk.PhotoImage(file=fig_path + '/' + figures[i]))
                self.result_images = ttk.Label(self.show_frame, image=image_list[i])
                self.result_images.grid(row=i // 3, column=i % 3, sticky=W, pady=0)

        fig_path = './figures/showSTAGE_10X_Breast_Cancer_Visium'
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
            image_list.append(ttk.PhotoImage(file=fig_path + '/' + figures[i]))
            self.result_images = ttk.Label(self.show_frame, image=image_list[i])
            self.result_images.grid(row=i // 3, column=i % 3, sticky=W, pady=0)
            s = i

        self.color_choose = ttk.Button(self.show_frame, text='reset colors', command=choose_color, width=10)
        self.color_choose.grid(row=s // 3 + 1, column=1, sticky=W, pady=0)

        def Reset():
            python = sys.executable
            os.execl(python, python, *sys.argv)

        self.Reset = ttk.Button(self.show_frame, text='Reset STABox', command=Reset, width=12)
        self.Reset.grid(row=s // 3 + 1, column=2, sticky=W, pady=0)

        self.pb.stop()
        self.setvar('prog-message', 'STAGE run over!')
        self.setvar('End-time', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        EndTime = datetime.now().replace(microsecond=0)
        self.setvar('total-time-cost', EndTime - self.StartTime)

    def STAGE_Thread_two(self):
        T = threading.Thread(target=self.STAGE_10XBreast_Cancer_Analysis)
        T.setDaemon(True)
        T.start()

    def STAGE_3D_Generation_Model_Process(self):
        try:
            data_file = glob.glob(self.Visium_file_path + '/*.h5ad')
            data_name = data_file[0].rsplit("\\", 1)[-1]
            fsize = os.path.getsize(os.path.join(self.Visium_file_path, data_name))
            self.data_name = data_name.rsplit('.', 1)[-2]
            self.setvar('data_name', self.data_name)
            adata = sc.read_h5ad(data_file[0])
            adata.var_names_make_unique()

            self.data_size = round(fsize / 1024 / 1024, 2)
            self.setvar('data_size', str(self.data_size) + 'MB')
            self.data_shape = adata.shape
            self.setvar('data_shape', str(self.data_shape[0]) + '×' + str(self.data_shape[1]))

            sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
            self.result_queue.put(adata)
        except:
            Messagebox.show_info('Check whether the files exists!')
        pass

    def STAGE_3D_Generation_Model_Analysis(self):
        self.STAGE_3D_Generation_Model_Process()
        adata = self.result_queue.get()

        filepath = './GATE_OUT_T5_3D_model'
        if os.path.isfile(filepath):
            os.mkdir(filepath)

        adata_stage, adata_simu = STAGE(
            adata,
            save_path=filepath,
            data_type='Slide-seq',
            experiment='3d_model',
            coord_sf=6000,
            sec_name='section',
            select_section=[1, 3, 5, 6, 8],
            gap=0.05,
            train_epoch=1000,
            batch_size=8192,
            learning_rate=1e-3,
            w_recon=0.1,
            w_w=0.1,
            w_l1=0.1
        )
        # self.genes = Pcp4
        show_gene = [self.genes, "Cdhr1"]
        colors = 'Blues'

        def draw_images(colors):
            j = 1
            from matplotlib import pyplot as plt
            plt.rcParams['font.sans-serif'] = "Arial"

            j = j + 1
            for i in range(len(show_gene)):
                plt.rcParams["figure.figsize"] = (3, 3)
                sc.pl.embedding(adata[adata.obs['section'] == 4], basis="coord", color=show_gene[i], title=show_gene[i],
                                s=6, show=False, save='STAGE_' + str(i) + '.png')
                shutil.move('./figures/coordSTAGE_' + str(i) + '.png',
                            './figures/showSTAGE_3D_model/STAGE_' + str(j) + '_' + str(i) + '.png')
            j = j + 1
            for i in range(len(show_gene)):
                plt.rcParams["figure.figsize"] = (3, 3)
                sc.pl.embedding(adata_stage[adata_stage.obs['section'] == 4], basis="coord", color=show_gene[i],
                                title=show_gene[i], s=6, show=False, save='STAGE_' + str(i) + '.png')
                shutil.move('./figures/coordSTAGE_' + str(i) + '.png',
                            './figures/showSTAGE_3D_model/STAGE_' + str(j) + '_' + str(i) + '.png')

            j = j + 1
            i = 1
            if self.genes != "Pcp4":
                self.genes = "Pcp4"
            plot_gene = self.genes
            plot_expr = np.array(sp.coo_matrix(adata_stage[:, plot_gene].X).todense())

            fig = plt.figure(figsize=(4, 4))
            ax1 = plt.axes(projection='3d')

            ax1.scatter3D(adata_stage.obs["xcoord"], adata_stage.obs["ycoord"], adata_stage.obs["zcoord"],
                          c=plot_expr, cmap=plt.get_cmap(colors), s=1 * plot_expr / max(plot_expr),
                          vmin=0.3 * max(plot_expr), marker="o")

            ax1.set_xlabel('')
            ax1.set_ylabel('')
            ax1.set_zlabel('')

            ax1.set_xticklabels([])
            ax1.set_yticklabels([])
            ax1.set_zticklabels([])

            ax1.set_zlim(min(adata.obs["zcoord"]), max(adata.obs["zcoord"]))
            plt.title(plot_gene)

            ax1.elev = 15
            ax1.azim = 80
            plt.savefig('./figures/showSTAGE_3D_model/STAGE_' + str(j) + '_' + str(i) + '.png')
            plt.close()

            j = j + 1
            i = 1
            plot_expr = np.array(sp.coo_matrix(adata_simu[:, plot_gene].X).todense())

            fig = plt.figure(figsize=(4, 4))
            ax1 = plt.axes(projection='3d')

            ax1.scatter3D(adata_simu.obs["xcoord"], adata_simu.obs["ycoord"], adata_simu.obs["zcoord"],
                          c=plot_expr, cmap=plt.get_cmap(colors), s=1 * plot_expr / max(plot_expr),
                          vmin=0.3 * max(plot_expr), marker="o")

            ax1.set_xlabel('')
            ax1.set_ylabel('')
            ax1.set_zlabel('')

            ax1.set_xticklabels([])
            ax1.set_yticklabels([])
            ax1.set_zticklabels([])

            ax1.set_zlim(min(adata.obs["zcoord"]), max(adata.obs["zcoord"]))
            plt.title(plot_gene)

            ax1.elev = 15
            ax1.azim = 80
            plt.savefig('./figures/showSTAGE_3D_model/STAGE_' + str(j) + '_' + str(i) + '.png')
            plt.close()

        def scoller():
            self.figure_ybar = ttk.Scrollbar(self.figure_Frame, orient=VERTICAL, cursor='draft_small')
            self.figure_ybar.pack(side=RIGHT, fill=Y)
            self.figure_ybar.config(command=self.canvas.yview)
            self.canvas.configure(yscrollcommand=self.figure_ybar.set)
            self.canvas.config(yscrollincrement=1)  # 设置滚动条的步长
            self.canvas.pack(side=LEFT, expand=YES, fill=BOTH, anchor=NW)
            self.canvas.create_window((0, 0), window=self.show_frame, anchor=N)
            self.show_frame.bind("<Configure>", onFrameconfigure)

        def onFrameconfigure(event):
            # width=1450 -> width=1500
            self.canvas.configure(scrollregion=self.canvas.bbox("all"), width=1500, height=650)

        def choose_color():
            colorvalue = colorchooser.askcolor()
            color_list = []
            color = colorvalue[1]
            print(color)
            ls = len(show_gene)

            if color[1:3] == 'ff':
                color_list.append(color)
                list = random.sample(color_panel.FF_color, ls)
                color_list = color_list + list
                colors = 'Greens'
            elif color[1:3] == 'cc':
                color_list.append(color)
                list = random.sample(color_panel.CC_color, ls)
                color_list = color_list + list
                colors = 'BrBG'
            elif color[1:3] == '99':
                color_list.append(color)
                list = random.sample(color_panel.NN_color, ls)
                color_list = color_list + list
                colors = 'Oranges'
            elif color[1:3] == '66':
                color_list.append(color)
                list = random.sample(color_panel.SS_color, ls)
                color_list = color_list + list
            elif color[1:3] == '33':
                color_list.append(color)
                list = random.sample(color_panel.TT_color, ls)
                color_list = color_list + list
                colors = 'OrRd'
            elif color[1:3] == '00':
                color_list.append(color)
                list = random.sample(color_panel.ZZ_color, ls)
                color_list = color_list + list
                colors = 'BuGn'
            else:
                color_list.append(color)
                list = random.sample(color_panel.random_color, ls)
                color_list = color_list + list
                colors = 'GnBu'

            draw_images(colors)

            fig_path = './figures/showSTAGE_3D_model'
            figures = os.listdir(fig_path)
            global image_list
            image_list = []
            for i in range(len(figures)):
                image_list.append(ttk.PhotoImage(file=fig_path + '/' + figures[i]))
                self.result_images = ttk.Label(self.show_frame, image=image_list[i])
                self.result_images.grid(row=i // 3, column=i % 3, sticky=W, pady=0)

        fig_path = './figures/showSTAGE_3D_model'
        if not os.path.isdir(fig_path):
            os.mkdir(fig_path)

        draw_images(colors)
        self.figure_Frame = ttk.Frame(self.right_panel)
        self.figure_Frame.pack(side=TOP, fill=X, expand=YES)

        self.canvas = ttk.Canvas(self.figure_Frame)
        self.show_frame = ttk.Frame(self.canvas)

        scoller()

        figures = os.listdir(fig_path)
        global image_list
        image_list = []
        for i in range(len(figures)):
            image_list.append(ttk.PhotoImage(file=fig_path + '/' + figures[i]))
            self.result_images = ttk.Label(self.show_frame, image=image_list[i])
            self.result_images.grid(row=i // 3, column=i % 3, sticky=W, pady=0)
            s = i

        self.color_choose = ttk.Button(self.show_frame, text='reset colors', command=choose_color, width=10)
        self.color_choose.grid(row=s // 3 + 1, column=1, sticky=W, pady=0)

        def Reset():
            python = sys.executable
            os.execl(python, python, *sys.argv)

        self.Reset = ttk.Button(self.show_frame, text='Reset STABox', command=Reset, width=12)
        self.Reset.grid(row=s // 3 + 1, column=2, sticky=W, pady=0)

        self.pb.stop()
        self.setvar('prog-message', 'STAGE run over!')
        self.setvar('End-time', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        EndTime = datetime.now().replace(microsecond=0)
        self.setvar('total-time-cost', EndTime - self.StartTime)

    def STAGE_Thread_Three(self):
        T = threading.Thread(target=self.STAGE_3D_Generation_Model_Analysis)
        T.setDaemon(True)
        T.start()

    def SpaGCN_10XData_Process(self):
        try:
            file_path = self.file_path.rsplit('/', 1)[-2]
            tif_image = glob.glob(file_path + '/*.tif')
            histology = False
            adata = self.result_queue.get()

            x_array = adata.obsm['spatial'][:, 0].tolist()
            y_array = adata.obsm['spatial'][:, 1].tolist()
            x_pixel = adata.obsm['spatial'][:, 0].tolist()
            y_pixel = adata.obsm['spatial'][:, 1].tolist()
            spg.prefilter_genes(adata, min_cells=3)
            spg.prefilter_specialgenes(adata)
            if self.data_type == '10x':
                sc.pp.normalize_per_cell(adata)
            sc.pp.log1p(adata)
            if len(tif_image) > 0:
                histology = True
                img = cv2.imread(tif_image[0])
                adj = spg.calculate_adj_matrix(x=x_pixel, y=y_pixel, x_pixel=x_pixel, y_pixel=y_pixel, image=img,
                                               beta=self.rad_cutoff_value, alpha=self.alpha_value, histology=histology)
            else:
                adj = spg.calculate_adj_matrix(x=x_pixel, y=y_pixel, histology=histology)
            # p = float(self.genes)
            l = spg.search_l(float(self.genes), adj, start=0.01, end=1000, tol=0.01, max_run=100)
            # If the number of clusters known, we can use the spg.search_res() fnction to search for suitable resolution(optional)
            # For this toy data, we set the number of clusters=7 since this tissue has 7 layers
            # n_clusters = 7
            r_seed = t_seed = n_seed = 100
            res = spg.search_res(adata, adj, l, self.cluster_value, start=0.7, step=0.1, tol=5e-3, lr=0.05,
                                 max_epochs=20, r_seed=r_seed, t_seed=t_seed, n_seed=n_seed)

            clf = spg.SpaGCN()
            clf.set_l(l)
            random.seed(r_seed)
            torch.manual_seed(t_seed)
            np.random.seed(n_seed)
            clf.train(adata, adj, init_spa=True, init="louvain", res=res, tol=5e-3, lr=0.05, max_epochs=200)
            y_pred, prob = clf.predict()
            adata.obs["pred"] = y_pred
            adata.obs["pred"] = adata.obs["pred"].astype('category')
            # shape="hexagon" for Visium data, "square" for ST data.
            adj_2d = spg.calculate_adj_matrix(x=x_array, y=y_array, histology=False)
            refined_pred = spg.refine(sample_id=adata.obs.index.tolist(), pred=adata.obs["pred"].tolist(), dis=adj_2d,
                                      shape="hexagon")
            adata.obs["refined_pred"] = refined_pred
            adata.obs["refined_pred"] = adata.obs["refined_pred"].astype('category')
            adata.obs['x_pixel'] = adata.obsm['spatial'][:, 1]
            adata.obs['y_pixel'] = adata.obsm['spatial'][:, 0]
            self.result_queue.put(adata)
        except:
            Messagebox.show_info('Attention', 'Check whether the 10X Visium files exists!')
        pass

    def SpaGCN_10XData_Analysis(self):
        self.SpaGCN_10XData_Process()

        def draw_images(sdata):
            i = 1
            temp_path = os.getcwd()
            os.chdir(test_file_path)
            from matplotlib import pyplot as plt
            plt.rcParams['font.sans-serif'] = "Arial"
            plt.rcParams["figure.figsize"] = (3, 3)
            path = test_file_path + '/figures/' + self.method_flag + '_' + self.data_type + '/'
            if self.label_files_exit:
                sc.pl.spatial(sdata, color='GroundTruth', title=['Ground Truth'], show=False, spot_size=300,
                              save='SpaGCN_' + str(i) + '.png')
                shutil.move(test_file_path + '/figures/showSpaGCN_' + str(i) + '.png', path + 'SpaGCN_' + str(i) + '.png')
                i = i + 1
                domains = "pred"
                sc.pl.spatial(sdata, color=domains, title=['SpaGCN-pred'], show=False, spot_size=300,
                              save='SpaGCN_' + str(i) + '.png')
                shutil.move(test_file_path + '/figures/showSpaGCN_' + str(i) + '.png', path + 'SpaGCN_' + str(i) + '.png')

                i = i + 1
                domains = "refined_pred"
                sc.pl.spatial(sdata, color=domains, title=['SpaGCN-refined_pred'], show=False, spot_size=300,
                              save='SpaGCN_' + str(i) + '.png')
                shutil.move(test_file_path + '/figures/showSpaGCN_' + str(i) + '.png', path + 'SpaGCN_' + str(i) + '.png')
            else:
                domains = "pred"
                sc.pl.spatial(sdata, color=domains, title=['SpaGCN-pred'], show=False, spot_size=300,
                              save='SpaGCN_' + str(i) + '.png')
                shutil.move(test_file_path + '/figures/showSpaGCN_' + str(i) + '.png', path + 'SpaGCN_' + str(i) + '.png')

                i = i + 1
                domains = "refined_pred"
                sc.pl.spatial(sdata, color=domains, title=['SpaGCN-refined_pred'], show=False, spot_size=300,
                              save='SpaGCN_' + str(i) + '.png')
                shutil.move(test_file_path + '/figures/showSpaGCN_' + str(i) + '.png', path + 'SpaGCN_' + str(i) + '.png')
                os.chdir(temp_path)

        def scoller():
            self.figure_ybar = ttk.Scrollbar(self.figure_Frame, orient=VERTICAL, cursor='draft_small')
            self.figure_ybar.pack(side=RIGHT, fill=Y)
            self.figure_ybar.config(command=self.canvas.yview)

            self.figure_xbar = ttk.Scrollbar(self.figure_Frame, orient=HORIZONTAL, cursor='draft_small')
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
            self.canvas.configure(scrollregion=self.canvas.bbox("all"), width=1000, height=650)

        def choose_color():
            colorvalue = colorchooser.askcolor()
            color_list = []
            color = colorvalue[1]
            print(color)
            if color[1:3] == 'ff':
                color_list.append(color)
                list = random.sample(color_panel.FF_color, self.cluster_value)
                color_list = color_list + list
            elif color[1:3] == 'cc':
                color_list.append(color)
                list = random.sample(color_panel.CC_color, self.cluster_value)
                color_list = color_list + list
            elif color[1:3] == '99':
                color_list.append(color)
                list = random.sample(color_panel.NN_color, self.cluster_value)
                color_list = color_list + list
            elif color[1:3] == '66':
                color_list.append(color)
                list = random.sample(color_panel.SS_color, self.cluster_value)
                color_list = color_list + list
            elif color[1:3] == '33':
                color_list.append(color)
                list = random.sample(color_panel.TT_color, self.cluster_value)
                color_list = color_list + list
            elif color[1:3] == '00':
                color_list.append(color)
                list = random.sample(color_panel.ZZ_color, self.cluster_value)
                color_list = color_list + list
            else:
                color_list.append(color)
                list = random.sample(color_panel.random_color, self.cluster_value)
                color_list = color_list + list

            print(color_list)
            sdata = adata
            if self.label_files_exit:
                sdata.uns['GroundTruth_colors'] = color_list
            sdata.uns['pred_colors'] = color_list
            sdata.uns['refined_pred_colors'] = color_list
            self.color_reset = color_list
            os.chdir(Raw_PATH)
            draw_images(sdata)

            fig_path = test_file_path + '/figures/' + self.method_flag + '_' + self.data_type + '/'
            figures = os.listdir(fig_path)
            global image_list
            image_list = []
            for i in range(len(figures)):
                image_list.append(ttk.PhotoImage(file=fig_path + '/' + figures[i]))
                self.result_images = ttk.Label(self.show_frame, image=image_list[i])
                self.result_images.grid(row=i // 3, column=i % 3, sticky=W, pady=0)

        fig_path = test_file_path + '/figures/' + self.method_flag + '_' + self.data_type + '/'
        if not os.path.isdir(fig_path):
            os.mkdir(fig_path)

        adata = self.result_queue.get()
        draw_images(adata)
        self.figure_Frame = ttk.Frame(self.right_panel)
        self.figure_Frame.pack(side=TOP, fill=X, expand=YES)

        self.canvas = ttk.Canvas(self.figure_Frame)
        self.show_frame = ttk.Frame(self.canvas)

        scoller()

        self.set_frame = ttk.Frame(self.figure_Frame, borderwidth=2, relief="sunken")
        self.set_frame.pack(side='right', expand=YES, anchor=N)
        self.set_frame_one = ttk.Frame(self.set_frame)
        self.set_frame_one.pack(side=TOP, expand=YES)
        self.set_frame_two = ttk.Frame(self.set_frame)
        self.set_frame_two.pack(side=TOP, expand=YES)

        self.Reset = ttk.Button(self.set_frame_one, text='Reset all colors', command=choose_color, width=12)
        self.Reset.grid(row=0, column=0, sticky=W, pady=0, padx=0)

        self.domain_color_label = ttk.Label(self.set_frame_one, text='Reset domain color: ', width=20)
        self.domain_color_label.grid(row=1, column=0, sticky=W, pady=0)

        self.Reset_domain_color = ttk.Entry(self.set_frame_one, width=10)
        self.Reset_domain_color.grid(row=1, column=1, sticky=W, pady=0)
        Tooltip(self.Reset_domain_color, "Input value must be int type and in [1:cluster_name]: 2")
        self.color_update = None

        def Reset_single_domain_color():
            from tkinter import colorchooser, filedialog
            colorvalue = colorchooser.askcolor()
            color = colorvalue[1]
            print(color)
            cluster = self.Reset_domain_color.get()
            print(cluster)
            adata.uns['pred_colors'] = self.color_reset[:len(adata.uns['pred_colors'])]
            adata.uns['refined_pred_colors'] = self.color_reset[:len(adata.uns['refined_pred_colors'])]
            self.color_reset[int(cluster) - 1] = color
            self.color_update = self.color_reset
            draw_images(adata)

            figures = os.listdir(fig_path)
            global image_list
            image_list = []
            for i in range(len(figures)):
                img = Image.open(fig_path + '/' + figures[i])
                print(img.size)
                image_list.append(ttk.PhotoImage(file=fig_path + '/' + figures[i]))
                self.result_images = ttk.Label(self.show_frame, image=image_list[i])
                self.result_images.grid(row=i // 4, column=i % 4, sticky=W, pady=0, padx=0)
                s = i

        self.Reset_domain_btn = ttk.Button(self.set_frame_one, width=10, text='Confirm',
                                           command=Reset_single_domain_color)
        self.Reset_domain_btn.grid(row=1, column=2, sticky=W, pady=10)
        self.update_image_label = ttk.Label(self.set_frame_one, text='Reset image dpi: ', width=20)
        self.update_image_label.grid(row=2, column=0, sticky=W, pady=10)
        self.update_image_scale = ttk.Entry(self.set_frame_one, width=10)
        self.update_image_scale.grid(row=2, column=1, sticky=W, pady=10)
        Tooltip(self.update_image_scale, "Input value must be int type and >= 300: 300")

        def ipdate_hd():
            dpi = self.update_image_scale.get()
            print(dpi)
            print(self.color_update)
            adata.uns['pred_colors'] = self.color_update
            adata.uns['refined_pred_colors'] = self.color_update

            import matplotlib.pyplot as plt
            plt.rcParams["figure.figsize"] = (3, 3)
            sc.set_figure_params(dpi=dpi)
            draw_images(adata)
            figures = os.listdir(fig_path)
            print(len(figures))

        self.Reset_domain_btn = ttk.Button(self.set_frame_one, width=10, text='Save', command=ipdate_hd)
        self.Reset_domain_btn.grid(row=2, column=2, sticky=W, pady=0)

        self.gene_visualization_label = ttk.Label(self.set_frame_one, text='Input gene name: ', width=20)
        self.gene_visualization_label.grid(row=3, column=0, sticky=W, pady=0)

        self.gene_visualization_entry = ttk.Entry(self.set_frame_one, width=10)
        self.gene_visualization_entry.grid(row=3, column=1, sticky=W, pady=0)
        Tooltip(self.gene_visualization_entry, "Gene name: NEFH/ATP2B4")

        def gene_visualization():
            try:
                gene = self.gene_visualization_entry.get()
                sc.pl.spatial(adata, img_key="hires", color=gene, title="$" + gene + "$",
                              show=False, save=gene + '.png', sopt_size=50)
                global img0
                photo = Image.open(test_file_path + '/figures/show' + gene + '.png')
                img0 = ImageTk.PhotoImage(photo)
                img1 = ttk.Label(self.set_frame_two, image=img0)
                img1.grid(row=0, column=0, sticky=W, pady=0)
                if os.path.exists(test_file_path + '/figures/show' + gene + '.png'):
                    os.remove(test_file_path + '/figures/show' + gene + '.png')
                    print("Figures exits")
                else:
                    print("Figures no exits！")
                pass
            except:
                Messagebox.show_error("Python Error", "Make sure gene name in dataset")

        self.gene_visualization_btn = ttk.Button(self.set_frame_one, width=10, command=gene_visualization, text='Show')
        self.gene_visualization_btn.grid(row=3, column=2, sticky=W, pady=0)

        global image_list
        figures = os.listdir(fig_path)
        image_list = []
        s = 0
        for i in range(len(figures)):
            image_list.append(ttk.PhotoImage(file=fig_path + '/' + figures[i]))
            self.result_images = ttk.Label(self.show_frame, image=image_list[i])
            self.result_images.grid(row=i // 4, column=i % 4, sticky=W, pady=0)
            s = i

        def VIEW_3D():
            import webbrowser
            self.web_Server_Thread('/DLPFC/webcache', 8029)
            webbrowser.open('http://127.0.0.1:8029/')

        self.Reset = ttk.Button(self.set_frame_one, text='3D VIEW', command=VIEW_3D, width=10)
        self.Reset.grid(row=0, column=1, sticky=W, pady=0, padx=(0, 100))

        self.pb.stop()
        self.setvar('prog-message', 'SpaGCN run over!')
        self.setvar('End-time', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        EndTime = datetime.now().replace(microsecond=0)
        self.setvar('total-time-cost', EndTime - self.StartTime)

    def SpaGCN_Thread(self):
        T = threading.Thread(target=self.SpaGCN_10XData_Analysis)
        T.setDaemon(True)
        T.start()
        pass

    def SpaGCN_SlideseqV2Data_Process(self):
        if os.path.exists(self.file_path):
            adata = self.result_queue.get()
            adata.var_names_make_unique()
            x_array = adata.obsm['spatial'][:, 0].tolist()
            y_array = adata.obsm['spatial'][:, 1].tolist()
            x_pixel = adata.obsm['spatial'][:, 0].tolist()
            y_pixel = adata.obsm['spatial'][:, 1].tolist()
            adj = spg.calculate_adj_matrix(x=x_pixel, y=y_pixel, histology=False)
            spg.prefilter_genes(adata, min_cells=3)
            spg.prefilter_specialgenes(adata)
            sc.pp.normalize_per_cell(adata)
            sc.pp.log1p(adata)
            p = float(self.genes)
            l = spg.search_l(p, adj, start=0.01, end=1000, tol=0.01, max_run=100)
            r_seed = t_seed = n_seed = 100
            res = spg.search_res(adata, adj, l, self.cluster_value, start=0.7, step=0.1, tol=5e-3, lr=0.05,
                                 max_epochs=20, r_seed=r_seed, t_seed=t_seed, n_seed=n_seed)
            clf = spg.SpaGCN()
            clf.set_l(l)
            random.seed(r_seed)
            torch.manual_seed(t_seed)
            np.random.seed(n_seed)
            clf.train(adata, adj, init_spa=True, init="louvain", res=res, tol=5e-3, lr=0.05, max_epochs=200)
            y_pred, prob = clf.predict()
            adata.obs["pred"] = y_pred
            adata.obs["pred"] = adata.obs["pred"].astype('category')
            # shape="hexagon" for Visium data, "square" for ST data.
            adj_2d = spg.calculate_adj_matrix(x=x_array, y=y_array, histology=False)
            refined_pred = spg.refine(sample_id=adata.obs.index.tolist(), pred=adata.obs["pred"].tolist(),
                                      dis=adj_2d, shape="hexagon")
            adata.obs["refined_pred"] = refined_pred
            adata.obs["refined_pred"] = adata.obs["refined_pred"].astype('category')
            adata.obs['x_pixel'] = adata.obsm['spatial'][:, 1]
            adata.obs['y_pixel'] = adata.obsm['spatial'][:, 0]

            self.result_queue.put(adata)
        else:
            Messagebox.show_error(title="File error", message="Make sure only one h5ad file in path!")

    def SpaGCN_SlideseqV2Data_Analysis(self):
        self.SpaGCN_SlideseqV2Data_Process()

        def draw_images(sdata):
            i = 1
            from matplotlib import pyplot as plt
            plt.rcParams['font.sans-serif'] = "Arial"
            plt.rcParams["figure.figsize"] = (3, 3)

            domains = "pred"
            sc.pl.spatial(sdata, color=domains, title=['SpaGCN-pred'], show=False, spot_size=20,
                          save='SpaGCN_' + str(i) + '.png')
            shutil.move('figures/showSpaGCN_' + str(i) + '.png', 'VIEW/figures/SpaGCN-h5ad/SpaGCN_' + str(i) + '.png')

            i = i + 1
            domains = "refined_pred"
            sc.pl.spatial(sdata, color=domains, title=['SpaGCN-refined_pred'], show=False, spot_size=20,
                          save='SpaGCN_' + str(i) + '.png')
            shutil.move('figures/showSpaGCN_' + str(i) + '.png', 'VIEW/figures/SpaGCN-h5ad/SpaGCN_' + str(i) + '.png')

        def scoller():
            self.figure_ybar = ttk.Scrollbar(self.figure_Frame, orient=VERTICAL, cursor='draft_small')
            self.figure_ybar.pack(side=RIGHT, fill=Y)
            self.figure_ybar.config(command=self.canvas.yview)
            self.canvas.configure(yscrollcommand=self.figure_ybar.set)
            self.canvas.config(yscrollincrement=1)  # 设置滚动条的步长
            self.canvas.pack(side=LEFT, expand=YES, fill=BOTH, anchor=NW)
            self.canvas.create_window((0, 0), window=self.show_frame, anchor=N)
            self.show_frame.bind("<Configure>", onFrameconfigure)

        def onFrameconfigure(event):
            # width=1450 -> width=1500
            self.canvas.configure(scrollregion=self.canvas.bbox("all"), width=1500, height=650)

        def choose_color():
            colorvalue = colorchooser.askcolor()
            color_list = []
            color = colorvalue[1]
            print(color)
            if color[1:3] == 'ff':
                color_list.append(color)
                list = random.sample(color_panel.FF_color, self.cluster_value)
                color_list = color_list + list
            elif color[1:3] == 'cc':
                color_list.append(color)
                list = random.sample(color_panel.CC_color, self.cluster_value)
                color_list = color_list + list
            elif color[1:3] == '99':
                color_list.append(color)
                list = random.sample(color_panel.NN_color, self.cluster_value)
                color_list = color_list + list
            elif color[1:3] == '66':
                color_list.append(color)
                list = random.sample(color_panel.SS_color, self.cluster_value)
                color_list = color_list + list
            elif color[1:3] == '33':
                color_list.append(color)
                list = random.sample(color_panel.TT_color, self.cluster_value)
                color_list = color_list + list
            elif color[1:3] == '00':
                color_list.append(color)
                list = random.sample(color_panel.ZZ_color, self.cluster_value)
                color_list = color_list + list
            else:
                color_list.append(color)
                list = random.sample(color_panel.random_color, self.cluster_value)
                color_list = color_list + list

            print(color_list)
            sdata = adata
            if self.label_files_exit:
                sdata.uns['GroundTruth_colors'] = color_list
            sdata.uns['pred_colors'] = color_list
            sdata.uns['refined_pred'] = color_list
            os.chdir(Raw_PATH)
            draw_images(sdata)

            fig_path = 'VIEW/figures/SpaGCN-h5add'
            figures = os.listdir(fig_path)
            global image_list
            image_list = []
            for i in range(len(figures)):
                image_list.append(ttk.PhotoImage(file=fig_path + '/' + figures[i]))
                self.result_images = ttk.Label(self.show_frame, image=image_list[i])
                self.result_images.grid(row=i // 3, column=i % 3, sticky=W, pady=0)

        fig_path = 'VIEW/figures/SpaGCN-h5ad'
        if not os.path.isdir(fig_path):
            os.mkdir(fig_path)

        adata = self.result_queue.get()
        draw_images(adata)
        self.figure_Frame = ttk.Frame(self.right_panel)
        self.figure_Frame.pack(side=TOP, fill=X, expand=YES)

        self.canvas = ttk.Canvas(self.figure_Frame)
        self.show_frame = ttk.Frame(self.canvas)
        scoller()

        figures = os.listdir(fig_path)
        global image_list
        image_list = []
        for i in range(len(figures)):
            image_list.append(ttk.PhotoImage(file=fig_path + '/' + figures[i]))
            self.result_images = ttk.Label(self.show_frame, image=image_list[i])
            self.result_images.grid(row=i // 3, column=i % 3, sticky=W, pady=0)
            s = i

        self.color_choose = ttk.Button(self.show_frame, text='Reset colors', command=choose_color, width=10)
        self.color_choose.grid(row=s // 3 + 1, column=0, sticky=W, pady=0, padx=(100, 0))

        def Reset():
            python = sys.executable
            os.execl(python, python, *sys.argv)

        self.Reset = ttk.Button(self.show_frame, text='Reset STABox', command=Reset, width=12)
        self.Reset.grid(row=s // 3 + 1, column=2, sticky=W, pady=0, padx=(0, 100))

        self.pb.stop()
        self.setvar('prog-message', 'SpaGCN run over!')
        self.setvar('End-time', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        EndTime = datetime.now().replace(microsecond=0)
        self.setvar('total-time-cost', EndTime - self.StartTime)

    def SpaGCN_Thread_two(self):
        T = threading.Thread(target=self.SpaGCN_SlideseqV2Data_Analysis)
        T.setDaemon(True)
        T.start()
        pass

    def SEDR_Data_Process(self):
        try:
            adata = self.result_queue.get()
            adata.obs['total_exp'] = adata.X.sum(axis=1)
            adata.var_names_make_unique()
            if self.data_type == '10x':
                adata_X = adata_preprocess(adata, min_cells=5, pca_n_comps=params.cell_feat_dim)
            else:
                from sklearn.decomposition import PCA
                params.k = 5
                adata_X = PCA(n_components=200, random_state=42).fit_transform(adata.X)
                adata.obsm['X_pca'] = adata_X
            graph_dict = graph_construction(adata.obsm['spatial'], adata.shape[0], params)
            params.cell_num = adata.shape[0]
            print('==== Graph Construction Finished')

            # ################## Model training
            sedr_net = SEDR_Train(adata_X, graph_dict, params)
            if params.using_dec:
                sedr_net.train_with_dec()
            else:
                sedr_net.train_without_dec()
            sedr_feat, _, _, _ = sedr_net.process()

            adata_sedr = anndata.AnnData(sedr_feat)
            adata_sedr.obsm['spatial'] = adata.obsm['spatial']

            adata_sedr.obs_names = adata.obs_names

            sc.pp.neighbors(adata_sedr, n_neighbors=params.eval_graph_n)
            sc.tl.umap(adata_sedr)

            eval_cluster = self.cluster_value
            eval_resolution = res_search_fixed_clus(adata_sedr, eval_cluster)
            sc.tl.leiden(adata_sedr, key_added="SEDR_leiden", resolution=eval_resolution)
            adata.obs['SEDR_leiden'] = adata_sedr.obs['SEDR_leiden']

            import rpy2.robjects as robjects
            robjects.r.library("mclust")

            import rpy2.robjects.numpy2ri
            rpy2.robjects.numpy2ri.activate()

            rmclust = robjects.r['Mclust']
            res2 = rmclust(adata_sedr.X, eval_cluster, 'EEE')
            mclust_res = np.array(res2[-2])

            adata.obs['SEDR_mclust'] = mclust_res
            # adata.obs['SEDR_mclust'] = mclust_R(adata_sedr.X, eval_cluster)
            adata.obs['SEDR_mclust'] = adata.obs['SEDR_mclust'].astype('int')
            adata.obs['SEDR_mclust'] = adata.obs['SEDR_mclust'].astype('category')

            kmeans = KMeans(n_clusters=eval_cluster)
            kmeans.fit(adata_sedr.X)
            adata.obs['SEDR_kmeans'] = kmeans.labels_
            adata.obs['SEDR_kmeans'] = adata.obs['SEDR_kmeans'].astype('int')
            adata.obs['SEDR_kmeans'] = adata.obs['SEDR_kmeans'].astype('category')
            self.result_queue.put(adata)
        except:
            Messagebox.show_info('Check whether the 10X Visium files exists!')

    def SEDR_Data_Analysis(self):
        self.SEDR_Data_Process()

        def draw_images(sdata):
            temp_path = os.getcwd()
            os.chdir(test_file_path)
            from matplotlib import pyplot as plt
            plt.rcParams['font.sans-serif'] = "Arial"
            plt.rcParams["figure.figsize"] = (3, 3)

            domains = ["SEDR_leiden", "SEDR_mclust", "SEDR_kmeans"]
            path = test_file_path + '/figures/' + self.method_flag + '_' + self.data_type + '/'
            for i in domains:
                sc.pl.spatial(sdata, color=i, title=i, show=False, save='SEDR_' + i + '.png')
                shutil.move(test_file_path + '/figures/showSEDR_' + i + '.png', path + 'SEDR_' + i + '.png')

            os.chdir(temp_path)

        def scoller():
            self.figure_ybar = ttk.Scrollbar(self.figure_Frame, orient=VERTICAL, cursor='draft_small')
            self.figure_ybar.pack(side=RIGHT, fill=Y)
            self.figure_ybar.config(command=self.canvas.yview)

            self.figure_xbar = ttk.Scrollbar(self.figure_Frame, orient=HORIZONTAL, cursor='draft_small')
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
            self.canvas.configure(scrollregion=self.canvas.bbox("all"), width=1000, height=650)

        def choose_color():
            colorvalue = colorchooser.askcolor()
            color_list = []
            color = colorvalue[1]
            print(color)
            self.cluster_value = max(len(adata.uns['SEDR_leiden_colors']), len(adata.uns['SEDR_mclust_colors']),
                                     len(adata.uns['SEDR_kmeans_colors']))
            if color[1:3] == 'ff':
                color_list.append(color)
                list = random.sample(color_panel.FF_color, self.cluster_value)
                color_list = color_list + list
            elif color[1:3] == 'cc':
                color_list.append(color)
                list = random.sample(color_panel.CC_color, self.cluster_value)
                color_list = color_list + list
            elif color[1:3] == '99':
                color_list.append(color)
                list = random.sample(color_panel.NN_color, self.cluster_value)
                color_list = color_list + list
            elif color[1:3] == '66':
                color_list.append(color)
                list = random.sample(color_panel.SS_color, self.cluster_value)
                color_list = color_list + list
            elif color[1:3] == '33':
                color_list.append(color)
                list = random.sample(color_panel.TT_color, self.cluster_value)
                color_list = color_list + list
            elif color[1:3] == '00':
                color_list.append(color)
                list = random.sample(color_panel.ZZ_color, self.cluster_value)
                color_list = color_list + list
            else:
                color_list.append(color)
                list = random.sample(color_panel.random_color, self.cluster_value)
                color_list = color_list + list

            print(color_list)
            adata.uns['SEDR_leiden_colors'] = color_list[:len(adata.uns['SEDR_leiden_colors'])]
            adata.uns['SEDR_mclust_colors'] = color_list[:len(adata.uns['SEDR_mclust_colors'])]
            adata.uns['SEDR_kmeans_colors'] = color_list[:len(adata.uns['SEDR_kmeans_colors'])]
            self.color_reset = color_list
            os.chdir(Raw_PATH)
            draw_images(adata)

            fig_path = test_file_path + '/figures/' + self.method_flag + '_' + self.data_type + '/'
            figures = os.listdir(fig_path)
            global image_list
            image_list = []
            for i in range(len(figures)):
                image_list.append(ttk.PhotoImage(file=fig_path + '/' + figures[i]))
                self.result_images = ttk.Label(self.show_frame, image=image_list[i])
                self.result_images.grid(row=i // 3, column=i % 3, sticky=W, pady=0)

        fig_path = test_file_path + '/figures/' + self.method_flag + '_' + self.data_type + '/'
        if not os.path.isdir(fig_path):
            os.mkdir(fig_path)

        adata = self.result_queue.get()
        draw_images(adata)
        self.figure_Frame = ttk.Frame(self.right_panel)
        self.figure_Frame.pack(side=TOP, fill=X, expand=YES)

        self.canvas = ttk.Canvas(self.figure_Frame)
        self.show_frame = ttk.Frame(self.canvas)

        scoller()
        self.set_frame = ttk.Frame(self.figure_Frame, borderwidth=2, relief="sunken")
        self.set_frame.pack(side='right', expand=YES, anchor=N)
        self.set_frame_one = ttk.Frame(self.set_frame)
        self.set_frame_one.pack(side=TOP, expand=YES)
        self.set_frame_two = ttk.Frame(self.set_frame)
        self.set_frame_two.pack(side=TOP, expand=YES)

        self.Reset = ttk.Button(self.set_frame_one, text='Reset all colors', command=choose_color, width=12)
        self.Reset.grid(row=0, column=0, sticky=W, pady=0, padx=0)

        self.domain_color_label = ttk.Label(self.set_frame_one, text='Reset domain color: ', width=20)
        self.domain_color_label.grid(row=1, column=0, sticky=W, pady=0)

        self.Reset_domain_color = ttk.Entry(self.set_frame_one, width=10)
        self.Reset_domain_color.grid(row=1, column=1, sticky=W, pady=0)
        Tooltip(self.Reset_domain_color, "Input value must be int type and in [1:cluster_name]: 2")
        self.color_update = None

        def Reset_single_domain_color():
            from tkinter import colorchooser, filedialog
            colorvalue = colorchooser.askcolor()
            color = colorvalue[1]
            print(color)
            cluster = self.Reset_domain_color.get()
            print(cluster)
            self.color_reset[int(cluster) - 1] = color
            adata.uns['SEDR_leiden_colors'] = self.color_reset[:len(adata.uns['SEDR_leiden_colors'])]
            adata.uns['SEDR_mclust_colors'] = self.color_reset[:len(adata.uns['SEDR_mclust_colors'])]
            adata.uns['SEDR_kmeans_colors'] = self.color_reset[:len(adata.uns['SEDR_kmeans_colors'])]
            draw_images(adata)

            figures = os.listdir(fig_path)
            global image_list
            image_list = []
            for i in range(len(figures)):
                img = Image.open(fig_path + '/' + figures[i])
                print(img.size)
                image_list.append(ttk.PhotoImage(file=fig_path + '/' + figures[i]))
                self.result_images = ttk.Label(self.show_frame, image=image_list[i])
                self.result_images.grid(row=i // 4, column=i % 4, sticky=W, pady=0, padx=0)
                s = i

        self.Reset_domain_btn = ttk.Button(self.set_frame_one, width=10, text='Confirm',
                                           command=Reset_single_domain_color)
        self.Reset_domain_btn.grid(row=1, column=2, sticky=W, pady=10)
        self.update_image_label = ttk.Label(self.set_frame_one, text='Reset image dpi: ', width=20)
        self.update_image_label.grid(row=2, column=0, sticky=W, pady=10)
        self.update_image_scale = ttk.Entry(self.set_frame_one, width=10)
        self.update_image_scale.grid(row=2, column=1, sticky=W, pady=10)
        Tooltip(self.update_image_scale, "Input value must be int type and >= 300: 300")

        def ipdate_hd():
            dpi = self.update_image_scale.get()
            print(dpi)
            import matplotlib.pyplot as plt
            plt.rcParams["figure.figsize"] = (3, 3)
            sc.set_figure_params(dpi=dpi)
            draw_images(adata)
            figures = os.listdir(fig_path)
            print(len(figures))

        self.Reset_domain_btn = ttk.Button(self.set_frame_one, width=10, text='Save', command=ipdate_hd)
        self.Reset_domain_btn.grid(row=2, column=2, sticky=W, pady=0)

        self.gene_visualization_label = ttk.Label(self.set_frame_one, text='Input gene name: ', width=20)
        self.gene_visualization_label.grid(row=3, column=0, sticky=W, pady=0)

        self.gene_visualization_entry = ttk.Entry(self.set_frame_one, width=10)
        self.gene_visualization_entry.grid(row=3, column=1, sticky=W, pady=0)
        Tooltip(self.gene_visualization_entry, "Gene name: NEFH/ATP2B4")

        def gene_visualization():
            try:
                gene = self.gene_visualization_entry.get()
                sc.pl.spatial(adata, img_key="hires", color=gene, title="$" + gene + "$",
                              show=False, save=gene + '.png', sopt_size=50)
                global img0
                photo = Image.open(test_file_path + '/figures/show' + gene + '.png')
                img0 = ImageTk.PhotoImage(photo)
                img1 = ttk.Label(self.set_frame_two, image=img0)
                img1.grid(row=0, column=0, sticky=W, pady=0)
                if os.path.exists(test_file_path + '/figures/show' + gene + '.png'):
                    os.remove(test_file_path + '/figures/show' + gene + '.png')
                    print("Figures exits")
                else:
                    print("Figures no exits！")
                pass
            except:
                Messagebox.show_error("Python Error", "Make sure gene name in dataset")

        self.gene_visualization_btn = ttk.Button(self.set_frame_one, width=10, command=gene_visualization, text='Show')
        self.gene_visualization_btn.grid(row=3, column=2, sticky=W, pady=0)

        # end
        global image_list
        figures = os.listdir(fig_path)
        image_list = []
        s = 0
        for i in range(len(figures)):
            image_list.append(ttk.PhotoImage(file=fig_path + '/' + figures[i]))
            self.result_images = ttk.Label(self.show_frame, image=image_list[i])
            self.result_images.grid(row=i // 4, column=i % 4, sticky=W, pady=0)
            s = i

        def VIEW_3D():
            import webbrowser
            self.web_Server_Thread('/DLPFC/webcache', 8029)
            webbrowser.open('http://127.0.0.1:8029/')

        self.Reset = ttk.Button(self.set_frame_one, text='3D VIEW', command=VIEW_3D, width=10)
        self.Reset.grid(row=0, column=1, sticky=W, pady=0, padx=(0, 100))

        self.pb.stop()
        self.setvar('prog-message', 'SEDR run over!')
        self.setvar('End-time', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        EndTime = datetime.now().replace(microsecond=0)
        self.setvar('total-time-cost', EndTime - self.StartTime)

    def SEDR_Thread(self):
        T = threading.Thread(target=self.SEDR_Data_Analysis)
        T.setDaemon(True)
        T.start()

    def SCANPY_Data_Analysis(self):
        adata = self.result_queue.get()
        if 'highly_variable' in adata.var.columns:
            adata = adata[:, adata.var['highly_variable']]
        else:
            sc.pp.highly_variable_genes(adata, flavor='seurat', n_top_genes=2000, inplace=True)
            adata = adata[:, adata.var['highly_variable']]
        if 'GroundTruth' in adata.obs.columns:
            self.label_files_exit = True

        sc.pp.pca(adata, n_comps=int(self.rad_cutoff_value), use_highly_variable=True, svd_solver='arpack')
        sc.pp.neighbors(adata)
        sc.tl.umap(adata)
        sc.tl.leiden(adata, resolution=float(self.alpha_value), key_added="leiden")
        sc.tl.rank_genes_groups(adata, "leiden", inplace=True)

        plt.rcParams['font.sans-serif'] = "Arial"
        self.gene_color_type = 'viridis'

        def draw_images():
            i = 1
            os.chdir(Raw_PATH)
            from matplotlib import pyplot as plt
            plt.rcParams["figure.figsize"] = (3, 3)
            path = test_file_path + '/figures/' + self.method_flag + '_' + self.data_type
            if self.label_files_exit:
                sc.pl.spatial(adata, img_key="hires", color="GroundTruth", title='Ground Truth', show=False,
                              save='SCANPY_' + str(i) + '.png')
                shutil.move(test_file_path + '/figures/showSCANPY_' + str(i) + '.png', path + '/SCANPY_' + str(i) + '.png')

                i = i + 1

                plt.rcParams["figure.figsize"] = (3, 3)
                sc.pl.umap(adata, color="leiden", title='SCANPY', show=False, s=6, save='SCANPY_' + str(i) + '.png')
                shutil.move(test_file_path + '/figures/umapSCANPY_' + str(i) + '.png', path + '/SCANPY_' + str(i) + '.png')

                i = i + 1
                sc.pl.spatial(adata, color="leiden", title='SCANPY', show=False, save='SCANPY_' + str(i) + '.png')
                shutil.move(test_file_path + '/figures/showSCANPY_' + str(i) + '.png', path + '/SCANPY_' + str(i) + '.png')

                i = i + 1
                sc.pl.rank_genes_groups_heatmap(adata, groups=str(self.cluster_value), groupby="leiden", show=False,
                                                save='SCANPY_' + str(i) + '.png')
                # shutil.move('figures/heatmapSCANPY__' + str(i) + '.png', path + '/SCANPY_' + str(i) + '.png')

                if self.genes in adata.var_names:
                    i = i + 1
                    sc.pl.spatial(adata, img_key="hires", color=self.genes, show=False,
                                  title="RAW-$" + self.genes + "$",
                                  vmax='p99', color_map=self.gene_color_type, save='SCANPY_' + str(i) + '.png')
                    shutil.move(test_file_path + '/figures/showSCANPY__' + str(i) + '.png', path + '/SCANPY_' + str(i) + '.png')

                adata.obs.GroundTruth = adata.obs.GroundTruth.astype(str)
                i = i + 1
                sc.tl.paga(adata, groups='GroundTruth')
                plt.rcParams["figure.figsize"] = (4, 3)
                sc.pl.paga(adata, color="GroundTruth", title='PAGA', show=False, save='SCANPY_' + str(i) + '.png')
                shutil.move(test_file_path + '/figures/pagaSCANPY_' + str(i) + '.png', path + '/SCANPY_' + str(i) + '.png')
            else:
                sc.pp.calculate_qc_metrics(adata, inplace=True)
                sc.pl.spatial(adata, img_key="hires", color="log1p_total_counts", title='log1p_total_counts',
                              show=False,
                              spot_size=20, color_map=self.gene_color_type, save='SCANPY_' + str(i) + '.png')
                shutil.move(test_file_path + '/figures/showSCANPY_' + str(i) + '.png', path + '/SCANPY_' + str(i) + '.png')

                i = i + 1
                plt.rcParams["figure.figsize"] = (3, 3)
                sc.pl.umap(adata, color="leiden", title='SCANPY', show=False, s=6, save='SCANPY_' + str(i) + '.png')
                shutil.move(test_file_path + '/figures/umapSCANPY_' + str(i) + '.png', path + '/SCANPY_' + str(i) + '.png')

                i = i + 1
                sc.pl.spatial(adata, color="leiden", title='SCANPY', show=False, save='SCANPY_' + str(i) + '.png')
                shutil.move(test_file_path + '/figures/showSCANPY_' + str(i) + '.png', path + '/SCANPY_' + str(i) + '.png')

                i = i + 1
                sc.pl.rank_genes_groups_heatmap(adata, groups=str(self.cluster_value), groupby="leiden", show=False,
                                                save='SCANPY_' + str(i) + '.png')
                # shutil.move(test_file_path + '/figures/heatmapSCANPY__' + str(i) + '.png', path + '/SCANPY_' + str(i) + '.png')

                if self.genes in adata.var_names:
                    i = i + 1
                    sc.pl.spatial(adata, img_key="hires", color=self.genes, show=False,
                                  title="RAW-$" + self.genes + "$",
                                  vmax='p99', color_map=self.gene_color_type, save='SCANPY_' + str(i) + '.png')
                    shutil.move(test_file_path + '/figures/showSCANPY__' + str(i) + '.png', path + '/SCANPY_' + str(i) + '.png')

        def scoller():
            self.figure_ybar = ttk.Scrollbar(self.figure_Frame, orient=VERTICAL, cursor='draft_small')
            self.figure_ybar.pack(side=RIGHT, fill=Y)
            self.figure_ybar.config(command=self.canvas.yview)

            self.figure_xbar = ttk.Scrollbar(self.figure_Frame, orient=HORIZONTAL, cursor='draft_small')
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
            self.canvas.configure(scrollregion=self.canvas.bbox("all"), width=1000, height=650)

        def choose_color():
            colorvalue = colorchooser.askcolor()
            color_list = []
            color = colorvalue[1]
            print(color)
            color_lens = len(adata.uns['leiden_colors'])
            if color[1:3] == 'ff':
                color_list.append(color)
                list = random.sample(color_panel.FF_color, color_lens)
                color_list = color_list + list
            elif color[1:3] == 'cc':
                color_list.append(color)
                list = random.sample(color_panel.CC_color, color_lens)
                color_list = color_list + list
            elif color[1:3] == '99':
                color_list.append(color)
                list = random.sample(color_panel.NN_color, color_lens)
                color_list = color_list + list
            elif color[1:3] == '66':
                color_list.append(color)
                list = random.sample(color_panel.SS_color, color_lens)
                color_list = color_list + list
            elif color[1:3] == '33':
                color_list.append(color)
                list = random.sample(color_panel.TT_color, color_lens)
                color_list = color_list + list
            elif color[1:3] == '00':
                color_list.append(color)
                list = random.sample(color_panel.ZZ_color, color_lens)
                color_list = color_list + list
            else:
                color_list.append(color)
                list = random.sample(color_panel.random_color, color_lens)
                color_list = color_list + list

            print(color_list)
            if self.label_files_exit:
                adata.uns['GroundTruth_colors'] = color_list[:len(adata.uns['GroundTruth_colors'])]
            adata.uns['leiden_colors'] = color_list[:len(adata.uns['leiden_colors'])]
            self.color_reset = color_list
            self.gene_color_type = color_panel.five_gene_color[random.randint(0, 5)]
            os.chdir(Raw_PATH)
            draw_images()

            fig_path = test_file_path + '/figures/' + self.method_flag + '_' + self.data_type
            figures = os.listdir(fig_path)
            global image_list
            image_list = []
            for i in range(len(figures)):
                image_list.append(ttk.PhotoImage(file=fig_path + '/' + figures[i]))
                self.result_images = ttk.Label(self.show_frame, image=image_list[i])
                self.result_images.grid(row=i // 3, column=i % 3, sticky=W, pady=0)

        fig_path = test_file_path + '/figures/' + self.method_flag + '_' + self.data_type
        if not os.path.isdir(fig_path):
            os.mkdir(fig_path)

        adata = self.result_queue.get()
        if self.label_files_exit:
            adata = adata[adata.obs['GroundTruth'] == adata.obs['GroundTruth'],]
        else:
            adata.obsm["spatial"] = adata.obsm["spatial"] * (-1)
        draw_images()
        self.figure_Frame = ttk.Frame(self.right_panel)
        self.figure_Frame.pack(side=TOP, fill=X, expand=YES)

        self.canvas = ttk.Canvas(self.figure_Frame)
        self.show_frame = ttk.Frame(self.canvas)
        scoller()

        self.set_frame = ttk.Frame(self.figure_Frame, borderwidth=2, relief="sunken")
        self.set_frame.pack(side='right', expand=YES, anchor=N)
        self.set_frame_one = ttk.Frame(self.set_frame)
        self.set_frame_one.pack(side=TOP, expand=YES)
        self.set_frame_two = ttk.Frame(self.set_frame)
        self.set_frame_two.pack(side=TOP, expand=YES)

        self.Reset = ttk.Button(self.set_frame_one, text='Reset all colors', command=choose_color, width=12)
        self.Reset.grid(row=0, column=0, sticky=W, pady=0, padx=0)

        self.domain_color_label = ttk.Label(self.set_frame_one, text='Reset domain color: ', width=20)
        self.domain_color_label.grid(row=1, column=0, sticky=W, pady=0)

        self.Reset_domain_color = ttk.Entry(self.set_frame_one, width=10)
        self.Reset_domain_color.grid(row=1, column=1, sticky=W, pady=0)
        Tooltip(self.Reset_domain_color, "Input value must be int type and in [1:cluster_name]: 2")
        self.color_update = None

        def Reset_single_domain_color():
            from tkinter import colorchooser
            colorvalue = colorchooser.askcolor()
            color = colorvalue[1]
            print(color)
            cluster = self.Reset_domain_color.get()
            print(cluster)
            adata.uns['leiden_colors'] = self.color_reset[:len(adata.uns['leiden_colors'])]
            adata.uns['leiden_colors'][int(cluster) - 1] = color
            self.color_reset[int(cluster) - 1] = color
            self.color_update = self.color_reset
            draw_images()
            figures = os.listdir(fig_path)
            global image_list
            image_list = []
            for i in range(len(figures)):
                img = Image.open(fig_path + '/' + figures[i])
                print(img.size)
                image_list.append(ttk.PhotoImage(file=fig_path + '/' + figures[i]))
                self.result_images = ttk.Label(self.show_frame, image=image_list[i])
                self.result_images.grid(row=i // 3, column=i % 3, sticky=W, pady=0, padx=0)
                s = i

            pass

        self.Reset_domain_btn = ttk.Button(self.set_frame_one, width=10, text='Confirm',
                                           command=Reset_single_domain_color)
        self.Reset_domain_btn.grid(row=1, column=2, sticky=W, pady=10)
        self.update_image_label = ttk.Label(self.set_frame_one, text='Reset image dpi: ', width=20)
        self.update_image_label.grid(row=2, column=0, sticky=W, pady=10)
        self.update_image_scale = ttk.Entry(self.set_frame_one, width=10)
        self.update_image_scale.grid(row=2, column=1, sticky=W, pady=10)
        Tooltip(self.update_image_scale, "Input value must be int type and >= 300: 300")

        def ipdate_hd():
            dpi = self.update_image_scale.get()
            print(dpi)
            import matplotlib.pyplot as plt
            plt.rcParams["figure.figsize"] = (3, 3)
            sc.set_figure_params(dpi=dpi)
            draw_images()
            figures = os.listdir(fig_path)
            print(len(figures))

        self.Reset_domain_btn = ttk.Button(self.set_frame_one, width=10, text='Save', command=ipdate_hd)
        self.Reset_domain_btn.grid(row=2, column=2, sticky=W, pady=0)

        self.gene_visualization_label = ttk.Label(self.set_frame_one, text='Input gene name: ', width=20)
        self.gene_visualization_label.grid(row=3, column=0, sticky=W, pady=0)

        self.gene_visualization_entry = ttk.Entry(self.set_frame_one, width=10)
        self.gene_visualization_entry.grid(row=3, column=1, sticky=W, pady=0)
        Tooltip(self.gene_visualization_entry, "Gene name: NEFH/ATP2B4")

        def gene_visualization():
            gene = self.gene_visualization_entry.get()
            if gene not in adata.var_names:
                print(f'{gene} is mot in adata!!Please input right gene name!')
            sc.pl.spatial(adata, img_key="hires", color=gene, title="$" + gene + "$", spot_size=20, show=False,
                          save=gene + '.png')
            global img0
            photo = Image.open(test_file_path + '/figures/show' + gene + '.png')
            img0 = ImageTk.PhotoImage(photo)
            img1 = ttk.Label(self.set_frame_two, image=img0)
            img1.grid(row=0, column=0, sticky=W, pady=0)
            if os.path.exists(test_file_path + '/figures/show' + gene + '.png'):
                os.remove(test_file_path + '/figures/show' + gene + '.png')
                print("yes")
            else:
                print("error！")

        self.gene_visualization_btn = ttk.Button(self.set_frame_one, width=10, command=gene_visualization, text='Show')
        self.gene_visualization_btn.grid(row=3, column=2, sticky=W, pady=0)
        figures = os.listdir(fig_path)
        global image_list
        image_list = []
        for i in range(len(figures)):
            image_list.append(ttk.PhotoImage(file=fig_path + '/' + figures[i]))
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
            data_save_path = running_path + '/module_3D_data/DLPFC'
            self.web_Server_Thread(os.path.join(data_save_path, 'webcache'), int(self.Entry.get()))
            http = 'http://127.0.0.1:' + self.Entry.get() + '/'
            # 'http://127.0.0.1:8050/'
            webbrowser.open(http)
            os.chdir(Raw_PATH)

        self.Reset = ttk.Button(self.set_frame_one, text='3D VIEW', command=VIEW_3D, width=10)
        self.Reset.grid(row=0, column=2, sticky=W, pady=0)

        self.pb.stop()
        self.setvar('prog-message', 'SCANPY run over!')
        self.setvar('End-time', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        EndTime = datetime.now().replace(microsecond=0)
        self.setvar('total-time-cost', EndTime - self.StartTime)

    def SCANPY_Thread(self):
        T = threading.Thread(target=self.SCANPY_Data_Analysis)
        T.setDaemon(True)
        T.start()

    def Data_process(self):
        self.StartTime = datetime.now().replace(microsecond=0)
        self.setvar('prog-time-started', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        print(self.data_type)
        print(self.method_flag)
        print(self.file_path)
        try:
            os.environ['R_HOME'] = self.R_HOME
            os.environ['R_USER'] = self.R_USER
            self.setvar('current-file-msg', self.method_flag)
            if self.method_flag == 'STAGATE':
                self._value = 'Log: ' + self.method_flag + ' is running ,' \
                                                           'the visualization results are displayed below......'
                self.setvar('scroll-message', self._value)
                self.pb.start()
                self.setvar('prog-message', 'STAGATE is running!!')
                self.label_files_exit = False
                self.STAGATE_Thread()

                # elif self.data_type == 'Slide-seqV2':
                #     self.pb.start()
                #     self.setvar('prog-message', 'STAGATE is running!!')
                #     self.STAGATE_Thread_two()
                #
                # elif self.data_type == 'Stereo-seq':
                #     self.pb.start()
                #     self.setvar('prog-message', 'STAGATE is running!!')
                #     self.STAGATE_Thread_three()
                #
                # elif self.data_type == 'STARmap':
                #     self.pb.start()
                #     self.setvar('prog-message', 'STAGATE is running!!')
                #     self.STAGATE_Thread_four()
                #
                # elif self.data_type == '3D-Data':
                #     self.pb.start()
                #     self.setvar('prog-message', 'STAGATE is running!!')
                #     self.STAGATE_Thread_five()
                #
                # elif self.data_type == 'MERFISH':
                #     self.pb.start()
                #     self.setvar('prog-message', 'STAGATE is running!!')
                #     self.STAGATE_Thread_six()
                #
                # elif self.data_type == 'Slide-seq':
                #     self.pb.start()
                #     self.setvar('prog-message', 'STAGATE is running!!')
                #     self.STAGATE_Thread_sevent()
                #
                # elif self.data_type == 'ST':
                #     self.pb.start()
                #     self.setvar('prog-message', 'STAGATE is running!!')
                #     self.STAGATE_Thread_eight()
                #
                # elif self.data_type == 'HDST':
                #     self.pb.start()
                #     self.setvar('prog-message', 'STAGATE is running!!')
                #     self.STAGATE_Thread_nine()
                #
                # elif self.data_type == 'H5AD-files':
                #     self.pb.start()
                #     self.setvar('prog-message', 'STAGATE is running!!')
                #     self.STAGATE_Thread_ten()

                # else:
                #     Messagebox.show_error(title="Error", message='check out right Datasets!!!')
                #     pass
            elif self.method_flag == 'STAligner':
                self._value = 'Log: ' + self.method_flag + 'is running, the visualization results are displayed ' \
                                                           'below...... '
                self.setvar('scroll-message', self._value)
                # if self.data_type == '10x':
                self.pb.start()
                self.setvar('prog-message', 'STAligner is running!!')
                self.label_files_exit = False
                self.STAligner_Thread()
                # elif self.data_type == 'Multi-files':
                #     self.pb.start()
                #     self.setvar('prog-message', 'STAligner is running!!')
                #     self.STAligner_Thread_one()
                # elif self.data_type == '3D-Slices':
                #     self.pb.start()
                #     self.setvar('prog-message', 'STAligner is running!!')
                #     self.STAligner_Thread_two()
                # elif self.data_type == 'Embryo':
                #     self.pb.start()
                #     self.setvar('prog-message', 'STAligner is running!!')
                #     self.STAligner_Thread_three()
                # else:
                #     pass
            elif self.method_flag == 'STAMarker':
                self._value = 'Log: ' + self.method_flag + ' is running, the visualization results are displayed below......'
                self.setvar('scroll-message', self._value)
                # if self.data_type == '10x':
                self.pb.start()
                self.setvar('prog-message', 'STAMarker is running!!')
                self.STAMarker_Thread_two()
                # else:
                #     self.pb.start()
                #     self.setvar('prog-message', 'STAMarker is running!!')
                #     self.STAMarker_Thread()
                # elif self.data_type == 'SlideseqV2-MouseHippo':
                #     self.pb.start()
                #     self.setvar('prog-message', 'STAMarker is running!!')
                #     self.STAMarker_Thread_one()
                # elif self.data_type == 'Stereo-seq':
                #     self.pb.start()
                #     self.setvar('prog-message', 'STAMarker is running!!')
                #     self.STAMarker_Thread_three()
                # else:
                #     pass
                # pass
            elif self.method_flag == 'STAGE':
                self._value = 'Log: ' + self.method_flag + ' is running ,' \
                                                           'the visualization results are displayed below......'
                self.setvar('scroll-message', self._value)
                if self.data_type not in ['10x', 'ST', 'Slide-seq']:
                    self.data_type = '10x'
                self.pb.start()
                self.setvar('prog-message', 'STAGE is running!!')
                self.STAGE_Thread()
                # elif self.data_type == '10X_Visium_V1_Adult_Mouse_Brain':
                #     self.pb.start()
                #     self.setvar('prog-message', 'STAGE is running!!')
                #     self.STAGE_Thread_one()
                # elif self.data_type == '10X_Breast_Cancer':
                #     self.pb.start()
                #     self.setvar('prog-message', 'STAGE is running!!')
                #     self.STAGE_Thread_two()
                # elif self.data_type == '3D_Generation_Model':
                #     self.pb.start()
                #     self.setvar('prog-message', 'STAGE is running!!')
                #     self.STAGE_Thread_Three()
                # else:
                #     pass
                # pass
            elif self.method_flag == 'SpaGCN':
                self._value = 'Log: ' + self.method_flag + ' is running ,' \
                                                           'the visualization results are displayed below......'
                self.setvar('scroll-message', self._value)
                # if self.data_type == 'Slide-seqV2':
                #     self.pb.start()
                #     self.setvar('prog-message', 'SpaGCN is running!!')
                #     self.SpaGCN_Thread_two()
                self.pb.start()
                self.setvar('prog-message', 'SpaGCN is running!!')
                self.SpaGCN_Thread()

                # elif self.data_type == 'Slide-seqV2':
                #     self.pb.start()
                #     self.setvar('prog-message', 'SpaGCN is running!!')
                #     self.SpaGCN_Thread_two()
                #
                # elif self.data_type == 'Stereo-seq':
                #     self.pb.start()
                #     self.setvar('prog-message', 'SpaGCN is running!!')
                #     self.STAGATE_Thread_three()
                #
                # elif self.data_type == 'Multiple-Sections':
                #     self.pb.start()
                #     self.setvar('prog-message', 'SpaGCN is running!!')
                #     self.STAGATE_Thread_four()
                #
                # elif self.data_type == '3D-Data':
                #     self.pb.start()
                #     self.setvar('prog-message', 'SpaGCN is running!!')
                #     self.STAGATE_Thread_five()
                # else:
                #     Messagebox.show_info('check out right Datasets!!!')
                # pass
            elif self.method_flag == 'SEDR':
                self._value = 'Log: ' + self.method_flag + ' is running ,' \
                                                           'the visualization results are displayed below......'
                self.setvar('scroll-message', self._value)
                self.pb.start()
                self.setvar('prog-message', 'SEDR is running!!')
                self.SEDR_Thread()
            else:
                self._value = 'Log: ' + self.method_flag + ' is running ,' \
                                                           'the visualization results are displayed below......'
                self.setvar('scroll-message', self._value)
                self.pb.start()
                self.setvar('prog-message', 'SCANPY is running!!')
                self.SCANPY_Thread()

        except:
            Messagebox.show_error('Python Error', 'check out right files !!')

    def settings(self):
        settingbox = ttk.Toplevel(title='settings')
        settingbox.geometry('300x100+100+100')
        settingbox.lift()
        ttk.Label(settingbox, text='R_HOME').grid(row=0, sticky="w")
        ttk.Label(settingbox, text='R_USER').grid(row=1, sticky="w")
        R_HOME = ttk.StringVar()
        R_USER = ttk.StringVar()
        R_HOME_box = ttk.Entry(settingbox, textvariable=R_HOME, validate="focusout", width=30)
        R_HOME_box.grid(row=0, column=1)
        R_USER_box = ttk.Entry(settingbox, textvariable=R_USER, validate="focusout", width=30)
        R_USER_box.grid(row=1, column=1)

        def load_setting():
            path = filedialog.askopenfilename()
            if len(path) != 0:
                with open(path, 'r') as f:
                    yamlData = yaml.safe_load(f)
                self.R_HOME = yamlData['R_HOME']
                self.R_USER = yamlData['R_USER']
                settingbox.destroy()
            else:
                settingbox.destroy()
                Messagebox.show_error('Make sure yaml exist!', 'Error!')

        load_setting_button = ttk.Button(settingbox, text="Load yaml", width=9, command=load_setting)
        load_setting_button.grid(row=2, column=0)

        def get_data():
            self.R_HOME = R_HOME_box.get()
            self.R_USER = R_USER_box.get()
            setting = {"R_HOME": self.R_HOME,
                       "R_USER": self.R_USER}
            with open('../Renv_setting.yaml', "w") as f:
                yaml.dump(setting, f)
            settingbox.destroy()
            print('self.R_HOME: ', self.R_HOME)
            print('self.R_USER: ', self.R_USER)

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
            ttk.PhotoImage(file=PATH / 'icons8_double_up_24px.png'),
            ttk.PhotoImage(file=PATH / 'icons8_double_right_24px.png')
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
        if child.winfo_class() != 'TFrame':
            return

        style_color = Bootstyle.ttkstyle_widget_color(bootstyle)
        frm = ttk.Frame(self, bootstyle=style_color)
        frm.grid(row=self.cumulative_rows, column=0, sticky=EW)

        # header title
        header = ttk.Label(
            master=frm,
            text=title,
            bootstyle=(style_color, INVERSE)
        )
        if kwargs.get('textvariable'):
            header.configure(textvariable=kwargs.get('textvariable'))
        header.pack(side=LEFT, fill=BOTH, padx=10)

        # header toggle button
        def _func(c=child):
            return self._toggle_open_close(c)

        btn = ttk.Button(
            master=frm,
            image=self.images[0],
            bootstyle=style_color,
            # 注意这边
            command=_func
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
        lines = message.strip().split('\n')
        for line in lines:
            if self.current_lines >= self.max_lines:
                break
            self.text_widget.insert(tk.END, line + '\n')
            self.current_lines += 1
        self.text_widget.see(tk.END)

    def flush(self):
        pass


if __name__ == '__main__':
    # os.environ['R_HOME'] = 'D:\\Users\\lqlu\\work\\software\\R-4.2.1'
    # os.environ['R_USER'] = 'D:\\Users\\lqlu\\work\\software\\Anaconda\\envs\\STAKITS\\Lib\\site-packages\\rpy2'
    app = ttk.Window("STABox")
    app.option_add('*Font', 'Arial')
    app.iconphoto(False, ttk.PhotoImage(file=os.path.join(PATH, 'app_icon.png')))
    STABox(app)
    app.mainloop()
