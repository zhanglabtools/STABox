```
# 从零开始搭建STABox环境

1. 创建并激活环境：
   ```bash
   conda create -n stomics python=3.10
   conda activate stomics
   ```

2. 安装必要的包：
   ```bash
   pip install 'scanpy[leiden]'
   pip install requests
   ```

3. 查看显卡驱动版本：
   ```bash
   nvidia-smi
   ```

   输出示例：
   ```
   +---------------------------------------------------------------------------------------+
   | NVIDIA-SMI 535.183.01             Driver Version: 535.183.01   CUDA Version: 12.2     |
   |-----------------------------------------+----------------------+----------------------+
   ```

4. 安装 PyTorch：
   ```bash
   pip3 install torch torchvision torchaudio
   ```
   > 注意：此时 torch 版本是 2.4.1，后面安装 dgl 库会删除 torch 2.4.1，转而安装 torch 2.4.0，影响不大。

5. 查看已安装包：
   ```bash
   pip list
   ```

6. 安装其他依赖：
   ```bash
   pip install progress
   ```

7. 处理缺少模块：
   ```bash
   pip install pyyaml  # No module named 'yaml'
   pip install torch_geometric
   pip install hnswlib
   pip install intervaltree
   pip install annoy
   pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
   pip install POT  # No module named 'ot'
   pip install dgl -f https://data.dgl.ai/wheels/torch-2.4/cu121/repo.html
   ```

8. 再次查看已安装包，确认 torch 版本：
   ```bash
   pip list
   ```
   > 可以看到 torch 版本已经变成 2.4.0。

9. 安装 R 环境：
   ```bash
   conda install r-base  # 安装 rpy2 前先用 conda 安装 R 语言环境
   ```

10. 进入 R 语言环境，安装 mclust 包：
    ```R
    R
    install.packages("mclust")  # 随便选一个源，我选的是 20
    q()  # 退出 R 语言环境
    ```

11. 安装 rpy2：
    ```bash
    pip install rpy2
    ```

12. 至此，可以运行以下链接成功：
    - [T1_DLPFC](https://stagate.readthedocs.io/en/latest/T1_DLPFC.html)

13. 安装 louvain：
    ```bash
    pip install louvain
    ```

14. 至此，可以运行以下链接成功：
    - [T4_Stereo](https://stagate.readthedocs.io/en/latest/T4_Stereo.html)

15. 运行以下链接：
    - [Tutorial_STABox_STAligner](https://stabox-tutorial.readthedocs.io/en/latest/Tutorial_STABox_STAligner.html)

16. 图形界面需要额外安装以下包：
    ```bash
    pip install ttkbootstrap
    pip install opencv-python
    pip install upsetplot
    pip install gseapy
    ```

17. 本地安装端口转发工具 XLaunch，把服务器端图形界面 export display 到本地电脑，通常会有些卡顿：
    ```bash
    cd STABox/src
    python -m stabox.view.app
    ```

> 至此，STABOX 界面可以启动成功。
具体 pip 环境见 requirement.txt 文件。
```