# STABox

STABox is a one-stop platform for spatial transcriptomics data that provide a unified data processing pipeline, versatile data analysis modules, and interactive visualization. It integrates a suite of advanced analysis tools based on graph neural networks. STABox supports interactive 2D/3D visualization of spatial transcriptomics data, simplifying the generation and refinement of publication-ready high-quality images. STABox is extensible, allowing for seamless integration with various analysis methods to facilitate comprehensive downstream analysis of spatial transcriptomics data. 

![image-20240529151225098](/STABox_overview.png)

Folder structure: 

```
stabox
├─src
│  └─stabox
│      ├─dataset
│      ├─extension
│      ├─model
│      ├─module_3D
│      ├─pl
│      ├─pp
│      └─view
└─tests
```
- `config`: save configuration yaml files
- `extension`: save the third-party code, e.g. `SEDR`, `SpaGCN`
- `dataset`: save the code for loading data. All loading functions should return an `AnnData` object with spatial information in `.obsm['spatial']`.
- `model`: save the model code, including `STAgate`, `STAligner` and `STAMarker`. 
All methods should be inherited from `BaseModelMixin` in [`_mixin.py`](./src/stabox/model/_mixin.py).
- `module_3D`: save the converted 3D data for subsequent interactive visualization of 3D data. 
- `pl`: save the result image output after model training.
- `pp`: save the preprocessing code, all preporcessing functions should take `AnnData` as input and return `AnnData` as output.
- `view`: save the visualization code for gui.

## Installation

The STABox package is developed based on the Python libraries [Scanpy](https://scanpy.readthedocs.io/en/stable/), [PyTorch](https://pytorch.org/), [DGL](https://github.com/dmlc/dgl/), and [PyG](https://github.com/pyg-team/pytorch_geometric) (*PyTorch Geometric*) framework, and can be run on GPU (recommend) or CPU.

First clone the repository. 

```
git clone https://github.com/zhanglabtools/STABox.git
cd STABox-main
```

It's recommended to create a separate conda environment for running STABox:

```
#create an environment called env_STABox
conda create -n env_STABox python=3.8

#activate your environment
conda activate env_STABox
```



The use of the mclust algorithm requires **R** environment, the **rpy2** package (Python) and the **mclust** package (R). See https://pypi.org/project/rpy2/ and https://cran.r-project.org/web/packages/mclust/index.html for detail.

Install **R** environment in python by conda:

```
conda install -c conda-forge r-base
```

Other required packages are listed in **STABox_env.yaml**.



##### Run STABox toolkit

```
cd STABox-main\src
python -m stabox.view.app
```

If run successfully, you will launch the following GUI:

![image-20240529204657589](/STABox_GUI.png)



## Tutorials

Step-by-step jupyter tutorials are included in https://stabox-tutorial.readthedocs.io/en/latest/ to show how to use the python library of STABox.

We also provide a video demo to show the key steps in running the GUI of STABox [here](https://drive.google.com/drive/folders/1Hd5HqJsekoZ_0BBkuDIjolSAhsRAdy6y?usp=drive_link).
The test dataset download is available by clicking [here](https://drive.google.com/drive/folders/1qaULEZ7gpc32A7L9-d-Vgo3_Pxx5ri04?usp=drive_link).
3D visualization datasets can be obtained [here](https://drive.google.com/drive/folders/13L2hB8gIZwI9vq_xyM6SG4CaLi_lNjN6?usp=drive_link)(note that the downloaded visualization datasets need to be saved in the 'view' folder)




## Contact

We are continuously updating and improving the software. If you have any questions or suggestions, please feel free to contact us longquanlu99@163.com. 



## Citation

...

## FAQs

Q: How to install **PyG** from whl files?

A: Please download the whl files from https://pytorch-geometric.com/whl/index.html. Note that the version of python, torch, PyG, and cuda should match. 

