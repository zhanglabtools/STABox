## Docker镜像使用说明

STABOX所需的依赖环境已提供为Docker镜像，方便用户快速部署。

## 下载 Docker 镜像

请从以下链接下载STABOX的Docker镜像压缩文件（XZ格式）：

[下载链接](https://drive.google.com/drive/folders/1E1u3BdQH5oguPvfY2y2kxDWnYbStZSoJ?usp=sharing)

下载完成后，使用以下命令解压文件：

```bash
xz -d -T0 zhanglab_stabox.tar.xz
```

其中，`-T0`选项表示自动选择多个CPU进行解压，以加快解压速度。

## 加载 Docker 镜像

确保你已经安装并配置了Docker环境。然后使用以下命令加载镜像：

```bash
docker load -i zhanglab_stabox.tar
```

可以通过以下命令查看已加载的镜像：

```bash
docker images
```

应该可以看到 `zhanglab_stabox` 镜像。

## 运行 Docker 容器

运行容器时，可以通过以下命令启动容器：

```bash
docker run --gpus all -it -v /mnt/disk1/LZJ/project/STABox:/home -d zhanglab_stabox
```

- `--gpus all`：表示允许容器访问宿主机的所有GPU。
- `-v /mnt/disk1/LZJ/project/STABox:/home`：此参数将本地的STABox代码目录映射到Docker容器的`/home`目录，便于在容器中运行代码。

## GPU 支持问题

如果你遇到以下错误：

```bash
docker: Error response from daemon: could not select device driver "" with capabilities: [[gpu]].
```

说明你的Docker环境目前不支持GPU。要启用GPU支持，请按照以下步骤安装 `nvidia-docker2`：

1. 安装 `nvidia-container-toolkit`：

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
```

2. 更新 apt 包列表并安装 `nvidia-container-toolkit`：

```bash
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
```

3. 配置 Docker 使用 NVIDIA GPU：

```bash
sudo nvidia-ctk runtime configure --runtime=docker
```

4. 重启 Docker 服务：

```bash
sudo systemctl restart docker
```

## 运行 STABOX

完成以上设置后，你就可以在Docker容器中运行STABOX进行训练、测试、预测等操作。
