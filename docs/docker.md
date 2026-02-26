### Docker 设置

本仓库中的所有示例都提供了正常运行和使用 Docker 运行的说明。虽然并非必需，但我们建议使用 Docker，因为它可以简化软件安装，提供更稳定的环境，并且对于依赖 ROS 的示例，还可以避免安装 ROS 并导致系统资源占用过高。

*   基本的 Docker 安装说明在[这里](https://docs.docker.com/engine/install/) 。
*   Docker 必须以[无根模式](https://docs.docker.com/engine/security/rootless/)安装。
*   要使用您的 GPU，您还必须安装 [NVIDIA 容器工具包](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) 。
*   通过 `snap` 安装的 Docker 版本与 NVIDIA 容器工具包不兼容，导致其无法访问 `libnvidia-ml.so` （ [问题](https://github.com/NVIDIA/nvidia-container-toolkit/issues/154) ）。可以使用 `sudo snap remove docker` 卸载 snap 版本。
*   Docker Desktop 与 NVIDIA 运行时不兼容（ [问题](https://github.com/NVIDIA/nvidia-container-toolkit/issues/229) ）。可以使用 `sudo apt remove docker-desktop` 卸载 Docker Desktop。

如果从头开始，并且您的主机是 Ubuntu 22.04，您可以使用便捷脚本 `scripts/docker/install_docker_ubuntu22.sh` 和 `scripts/docker/install_nvidia_container_toolkit.sh` 完成上述所有操作。

构建 Docker 镜像并使用以下命令启动容器：

```bash
docker compose -f scripts/docker/compose.yml up --build
```

要构建并运行特定示例的 Docker 镜像，请使用以下命令：

```bash
docker compose -f examples/<example_name>/compose.yml up --build
```

其中 `<example_name>` 是您要运行的示例的名称。

首次运行任何示例时，Docker 都会构建镜像。趁此机会去喝杯咖啡吧。由于镜像已被缓存，后续运行速度会更快。