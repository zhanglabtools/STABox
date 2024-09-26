从零开始搭建STABox环境

conda create -n stomics python=3.10
conda activate stomics
conda deactivate
conda activate stomics
pip install 'scanpy[leiden]'
pip install requests
nvidia-smi =>查看显卡驱动版本
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.183.01             Driver Version: 535.183.01   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
pip3 install torch torchvision torchaudio =>此时 torch 版本是2.4.1,后面安装dgl库会删除torch 2.4.1,转而安装 torch 2.4.0 影响不大
pip list
pip install progress
No module named 'yaml' =》pip install pyyaml 
pip install torch_geometric
pip install hnswlib
pip install intervaltree
pip install annoy
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
No module named 'ot'=》pip install POT 
pip install  dgl -f https://data.dgl.ai/wheels/torch-2.4/cu121/repo.html 
pip list  =》可以看到 torch 版本已经变成 2.4.0
conda install r-base  =>安装rpy2前先用codna安装 R语言环境
R 终端进入R语言环境 输入 install.packages("mclust") 安装mclust包,随便选一个源，我选的是20，安装成功后 q() 退出R语言环境
pip install rpy2

至此 https://stagate.readthedocs.io/en/latest/T1_DLPFC.html 可以运行成功

pip install louvain

至此 https://stagate.readthedocs.io/en/latest/T4_Stereo.html 可以运行成功
至此 https://stabox-tutorial.readthedocs.io/en/latest/Tutorial_STABox_STAligner.html

图形界面需要额外安装以下包：
pip install ttkbootstrap
pip install opencv-python
pip install upsetplot
pip install gseapy

本地安装端口转发工具XLaunch 把服务器端图形界面export display到本地电脑，通常会有些卡顿
cd STABox/src
python  -m stabox.view.app 

至此 stabox 界面可以启动成功

