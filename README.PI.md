# openpi

openpi 收集并发布了 Physical Intelligence 团队开源的机器人模型与相关包，仓库地址与更多信息见 Physical Intelligence 官网。

目前仓库包含三类模型：
- [π₀ 模型](https://www.physicalintelligence.company/blog/pi0)：基于流（flow-based）的视觉-语言-动作模型（VLA）。
- [π₀-FAST 模型](https://www.physicalintelligence.company/research/fast)：基于 FAST 动作 tokenizer 的自回归 VLA。
- [π₀.₅ 模型](https://www.physicalintelligence.company/blog/pi05)：相较 π₀ 在开放世界泛化上改进的版本，使用了 [knowledge insulation](https://www.physicalintelligence.company/research/knowledge_insulation) 训练。注意：在本仓库中，我们目前仅对 $\pi_{0.5}$ 的流匹配（flow matching）头提供训练与推理支持。

我们为所有模型提供了预训练的基础模型（base model）检查点，这些检查点在 10k+ 小时的机器人数据上预训练，并提供了开箱即用的示例以及用于在你自己的数据上微调的示例。

这是一次实验性工作：$\pi_0$ 是为我们自有机器人开发的，这些机器人与诸如 [ALOHA](https://tonyzhaozh.github.io/aloha/) 或 [DROID](https://droid-dataset.github.io/) 等常用平台不同。我们乐观地认为研究者与工程师可以将 $\pi_0$ 适配到他们的平台上并开展新的实验，但并不保证每次尝试都能成功。总之：$\pi_0$ 可能适合也可能不适合你的环境，你可以自行尝试并评估效果。

## 更新记录

- [2025 年 9 月] 在 openpi 中发布了对 PyTorch 的支持。
- [2025 年 9 月] 发布了 pi05，即在开放世界泛化上改进的 pi0 版本。
- [2025 年 9 月] 为 DROID 训练添加了 [改进的 idle filter](examples/droid/README_train.md#data-filtering)。
- [2025 年 6 月] 添加了使用 `openpi` 在完整 [DROID 数据集](https://droid-dataset.github.io/) 上训练 VLA 的 [说明](examples/droid/README_train.md)。这是对用于训练 pi0-FAST-DROID 的训练流水线的开源近似实现。


## 系统与硬件要求

要运行本仓库中的模型，你需要配备 NVIDIA GPU，且至少满足以下规格。下列估算基于单 GPU 配置；你也可以通过在训练配置中设置 `fsdp_devices` 使用多 GPU 的模型并行以降低单卡显存占用。注意：当前训练脚本尚不支持多节点训练。

| 模式                | 需要内存        | 示例 GPU              |
| ------------------- | --------------- | --------------------- |
| 推理（Inference）     | > 8 GB          | RTX 4090              |
| 微调（LoRA）         | > 22.5 GB       | RTX 4090              |
| 全量微调（Full）     | > 70 GB         | A100 (80GB) / H100    |

本仓库已在 Ubuntu 22.04 上测试，当前不支持其他操作系统。

## 安装

克隆仓库时，请确保更新子模块：

```bash
git clone --recurse-submodules git@github.com:Physical-Intelligence/openpi.git

# 若已克隆仓库，请运行：
git submodule update --init --recursive
```

我们使用 [uv](https://docs.astral.sh/uv/) 管理 Python 依赖。参考 [uv 安装说明](https://docs.astral.sh/uv/getting-started/installation/) 进行安装。安装完成后，运行如下命令创建环境：

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

注意：需要设置 `GIT_LFS_SKIP_SMUDGE=1` 以便正确拉取 LeRobot 作为依赖。

**Docker**：作为 uv 安装的替代方案，我们提供了使用 Docker 安装 openpi 的说明。如果本地环境配置出现问题，建议使用 Docker 来简化安装流程。更多详情见 [Docker 设置](docs/docker.md)。




## 模型检查点

### 基础模型（Base Models）
我们提供了多个基础 VLA 模型检查点。这些检查点在 10k+ 小时的机器人数据上预训练，可用于后续微调。

| 模型         | 用途         | 描述                                                                                                        | 检查点 路径                                     |
| ------------ | ----------- | ----------------------------------------------------------------------------------------------------------- | ---------------------------------------------- |
| $\pi_0$      | 微调         | 基础 [π₀ 模型](https://www.physicalintelligence.company/blog/pi0) 用于微调                                     | `gs://openpi-assets/checkpoints/pi0_base`      |
| $\pi_0$-FAST | 微调         | 基础自回归 [π₀-FAST 模型](https://www.physicalintelligence.company/research/fast) 用于微调                      | `gs://openpi-assets/checkpoints/pi0_fast_base` |
| $\pi_{0.5}$  | 微调         | 基础 [π₀.₅ 模型](https://www.physicalintelligence.company/blog/pi05) 用于微调                                   | `gs://openpi-assets/checkpoints/pi05_base`     |

### 微调后的模型（Fine-Tuned Models）
我们还提供了面向多个机器人平台和任务的“专家级”检查点。这些模型是在上述基础模型上微调得到，旨在直接在目标机器人上运行。它们可能适用于你的机器人，也可能不适用。由于这些检查点是基于较小的数据集（例如来自更常见的 ALOHA 与 DROID Franka 设置）微调得到的，可能无法完美泛化到你的平台，但我们发现其中一些（尤其是 DROID 检查点）在实践中具有较好的泛化能力。

| 模型                          | 用途               | 描述                                                                                                                                                                                        | 检查点 路径                                         |
| ---------------------------- | ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------- |
| $\pi_0$-FAST-DROID           | 推理（Inference）    | 在 [DROID 数据集](https://droid-dataset.github.io/) 上微调的 $\pi_0$-FAST：可在 DROID 机器人平台上对新场景进行多种台面操作任务的 0-shot 推理。                                                                      | `gs://openpi-assets/checkpoints/pi0_fast_droid`   |
| $\pi_0$-DROID                | 微调               | 在 [DROID 数据集](https://droid-dataset.github.io/) 上微调的 $\pi_0$：推理速度比 $\pi_0$-FAST-DROID 更快，但在跟随语言指令方面可能略逊一筹。                                                                        | `gs://openpi-assets/checkpoints/pi0_droid`        |
| $\pi_0$-ALOHA-towel          | 推理               | 在内部 ALOHA 数据上微调的 $\pi_0$：可在 ALOHA 平台上对各种毛巾进行 0-shot 折叠。                                                                                                                | `gs://openpi-assets/checkpoints/pi0_aloha_towel`  |
| $\pi_0$-ALOHA-tupperware     | 推理               | 在内部 ALOHA 数据上微调的 $\pi_0$：可执行从保鲜盒中取出食物等操作。                                                                                                                           | `gs://openpi-assets/checkpoints/pi0_aloha_tupperware` |
| $\pi_0$-ALOHA-pen-uncap      | 推理               | 在公开 ALOHA 数据上微调的 $\pi_0$：可完成开笔盖任务。                                                                                                                                        | `gs://openpi-assets/checkpoints/pi0_aloha_pen_uncap` |
| $\pi_{0.5}$-LIBERO          | 推理               | 针对 [LIBERO 基准](https://libero-project.github.io/datasets) 微调的 $\pi_{0.5}$：在基准上取得了领先表现（详见 [LIBERO README](examples/libero/README.md)）。                                                        | `gs://openpi-assets/checkpoints/pi05_libero`      |
| $\pi_{0.5}$-DROID           | 推理 / 微调         | 在 [DROID 数据集](https://droid-dataset.github.io/) 上基于 [knowledge insulation](https://www.physicalintelligence.company/research/knowledge_insulation) 微调的 $\pi_{0.5}$：推理速度快且具备良好的语言跟随能力。 | `gs://openpi-assets/checkpoints/pi05_droid`       |

默认情况下，检查点会在需要时自动从 `gs://openpi-assets` 下载，并缓存在 `~/.cache/openpi`。你可以通过设置环境变量 `OPENPI_DATA_HOME` 来覆盖下载路径。




## 使用预训练模型运行推理

我们的预训练检查点可以通过几行代码执行推理（下面以 $\pi_0$-FAST-DROID 为例）：
```python
from openpi.training import config as _config
from openpi.policies import policy_config
from openpi.shared import download

config = _config.get_config("pi05_droid")
checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi05_droid")

# 创建已加载权重的策略对象
policy = policy_config.create_trained_policy(config, checkpoint_dir)

# 在一个示例上运行推理
example = {
    "observation/exterior_image_1_left": ...,
    "observation/wrist_image_left": ...,
    ...
    "prompt": "pick up the fork"
}
action_chunk = policy.infer(example)["actions"]
```
你也可以在 [示例笔记本](examples/inference.ipynb) 中测试。

我们提供了关于如何在 [DROID](examples/droid/README.md) 和 [ALOHA](examples/aloha_real/README.md) 机器人上运行预训练检查点推理的详细逐步示例。

**远程推理（Remote Inference）**：我们提供了用于**远程**运行模型推理的示例和代码（见 [docs/remote_inference.md](docs/remote_inference.md)）。模型可以在远程服务器上运行，并通过 websocket 将动作流传输到机器人，这便于在机器人外使用更强的 GPU 并将机器人与策略环境分离。

**无机器人测试推理**：我们提供了一个用于无机器人环境测试推理的脚本（见 [examples/simple_client/README.md](examples/simple_client/README.md)）。该脚本会生成随机观测并对模型运行推理，更多细节见链接。





## 在你自己的数据上微调基础模型

下面以在 [LIBERO 数据集](https://libero-project.github.io/datasets) 上微调 $\pi_{0.5}$ 为示例，说明如何在你自己的数据上微调基础模型。我们将介绍三步：
1. 将数据转换为 LeRobot 数据集（用于训练）
2. 定义训练配置并运行训练
3. 启动策略服务器并运行推理

### 1. 将数据转换为 LeRobot 数据集

我们在 [`examples/libero/convert_libero_data_to_lerobot.py`](examples/libero/convert_libero_data_to_lerobot.py) 中提供了一个最小示例脚本，用于将 LIBERO 数据转换为 LeRobot 数据集。你可以根据自己的数据修改脚本！原始 LIBERO 数据集可从此处下载：[here](https://huggingface.co/datasets/openvla/modified_libero_rlds)，并用下面命令运行脚本：

```bash
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/libero/data
```

**注意：** 如果你只是想在 LIBERO 上微调，可以跳过此步骤，因为我们的 LIBERO 微调配置已经指向了预先转换好的 LIBERO 数据集。本步骤仅作为可供改造的示例。

### 2. 定义训练配置并运行训练

要在自己的数据上微调基础模型，需要为数据处理与训练定义配置。我们为 LIBERO 提供了带详细注释的示例配置，可按需修改：

- [`LiberoInputs` 和 `LiberoOutputs`](src/openpi/policies/libero_policy.py)：定义了 LIBERO 环境与模型之间的数据映射，适用于训练与推理。
- [`LeRobotLiberoDataConfig`](src/openpi/training/config.py)：定义了如何从 LeRobot 数据集中处理原始 LIBERO 数据以供训练使用。
- [`TrainConfig`](src/openpi/training/config.py)：定义微调超参数、数据配置和权重加载器。

我们为 [π₀](src/openpi/training/config.py)、[π₀-FAST](src/openpi/training/config.py) 和 [π₀.₅](src/openpi/training/config.py) 在 LIBERO 数据上提供了示例微调配置。

启动训练前需要先计算训练数据的归一化统计。用你的训练配置名运行下面脚本：

```bash
uv run scripts/compute_norm_stats.py --config-name pi05_libero
```

然后可以用下面命令启动训练（`--overwrite` 用于在重新运行相同配置时覆盖已有检查点）：

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_libero --exp-name=my_experiment --overwrite
```

该命令会在控制台打印训练日志，并将检查点保存到 `checkpoints` 目录。你也可以在 Weights & Biases 仪表盘上监控训练进度。为尽可能利用 GPU 内存，请在运行训练前设置 `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9`，这会允许 JAX 使用最多 90% 的显存（默认 75%）。

**注意：** 我们提供了从预训练中*重载*归一化统计的功能，用于状态/动作归一化。如果你在一个也包含于预训练混合中的机器人上微调新任务，重载统计可能有帮助。有关如何重载归一化统计的详细信息，请参见 [norm_stats.md](docs/norm_stats.md)。

### 3. 启动策略服务器并运行推理

训练完成后，可以通过启动策略服务器并从 LIBERO 评估脚本对其进行查询来运行推理。启动模型服务器很简单（以下示例使用第 20,000 次迭代的检查点，请按需修改）：

```bash
uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi05_libero --policy.dir=checkpoints/pi05_libero/my_experiment/20000
```

该命令会启动一个监听 8000 端口的服务器，等待接收观测。然后可运行评估脚本（或机器人运行时）向服务器发送观测并获取动作。

针对 LIBERO 评估，我们提供并推荐使用 Docker 化流程来同时运行策略服务器与评估脚本。详情请参见 [LIBERO README](examples/libero/README.md)。

如果希望在自己的机器人运行时中嵌入策略服务器调用，参见 [远程推理文档](docs/remote_inference.md) 中的最小示例。



### 更多示例

我们提供了更多关于在 ALOHA 平台上微调与运行推理的示例，见以下 README：
- [ALOHA Simulator](examples/aloha_sim)
- [ALOHA Real](examples/aloha_real)
- [UR5](examples/ur5)

## PyTorch 支持

openpi 现在在保留原有 JAX 版本的同时，提供了 π₀ 和 π₀.₅ 的 PyTorch 实现！PyTorch 实现已在 LIBERO 基准上验证（包含推理与微调）。目前部分特性尚不支持（未来可能会变更）：

- π₀-FAST 模型
- 混合精度训练
- FSDP（全分片数据并行）训练
- LoRA（低秩适配）训练
- 训练期间的 EMA（指数移动平均）权重

### 环境准备
1. 确保已安装并更新所有依赖：`uv sync`

2. 检查 transformers 版本是否为 4.53.2：`uv pip show transformers`

3. 应用对 transformers 库的补丁：
    ```bash
    cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/
    ```

该操作会覆盖 transformers 库中的若干文件以完成必要的模型修改：1) 支持 AdaRMS，2) 正确控制激活精度，3) 允许在不更新 KV cache 的情况下使用缓存。

**警告**：在默认的 uv 链接模式（hardlink）下，此操作会永久更改你 uv 缓存中的 transformers 库，变更会在重新安装 transformers 后依然保留，且可能影响其他使用 transformers 的项目。若要完全撤销，请运行 `uv cache clean transformers`。

### 将 JAX 模型转换为 PyTorch

将 JAX 模型检查点转换为 PyTorch 格式：

```bash
uv run examples/convert_jax_model_to_pytorch.py \
    --checkpoint_dir /path/to/jax/checkpoint \
    --config_name <config name> \
    --output_path /path/to/converted/pytorch/checkpoint
```

### 使用 PyTorch 运行推理

PyTorch 实现使用与 JAX 相同的 API —— 你只需将检查点路径指向已转换的 PyTorch 模型：

```python
from openpi.training import config as _config
from openpi.policies import policy_config
from openpi.shared import download

config = _config.get_config("pi05_droid")
checkpoint_dir = "/path/to/converted/pytorch/checkpoint"

# 创建已加载权重的策略对象（会自动检测 PyTorch 格式）
policy = policy_config.create_trained_policy(config, checkpoint_dir)

# 运行推理（API 与 JAX 相同）
action_chunk = policy.infer(example)["actions"]
```

### 使用 PyTorch 的策略服务器

策略服务器对 PyTorch 模型的支持与 JAX 相同——只需将 `--policy.dir` 指向已转换的 PyTorch 检查点目录：

```bash
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_droid \
    --policy.dir=/path/to/converted/pytorch/checkpoint
```

### 在 PyTorch 上微调

在 PyTorch 上微调模型的步骤：

1. 将 JAX 基础模型转换为 PyTorch 格式：
   ```bash
   uv run examples/convert_jax_model_to_pytorch.py \
       --config_name <config name> \
       --checkpoint_dir /path/to/jax/base/model \
       --output_path /path/to/pytorch/base/model
   ```

2. 在你的配置中使用 `pytorch_weight_path` 指定已转换的 PyTorch 模型路径。

3. 使用以下模式之一启动训练：

```bash
# 单 GPU 训练：
uv run scripts/train_pytorch.py <config_name> --exp_name <run_name> --save_interval <interval>

# 示例：
uv run scripts/train_pytorch.py debug --exp_name pytorch_test
uv run scripts/train_pytorch.py debug --exp_name pytorch_test --resume  # 从最新检查点恢复

# 单节点多 GPU 训练：
uv run torchrun --standalone --nnodes=1 --nproc_per_node=<num_gpus> scripts/train_pytorch.py <config_name> --exp_name <run_name>

# 示例：
uv run torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_pytorch.py pi0_aloha_sim --exp_name pytorch_ddp_test
uv run torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_pytorch.py pi0_aloha_sim --exp_name pytorch_ddp_test --resume

# 多节点训练：
uv run torchrun \
    --nnodes=<num_nodes> \
    --nproc_per_node=<gpus_per_node> \
    --node_rank=<rank_of_node> \
    --master_addr=<master_ip> \
    --master_port=<port> \
    scripts/train_pytorch.py <config_name> --exp_name=<run_name> --save_interval <interval>
```

### 精度设置

JAX 与 PyTorch 实现对精度的处理如下：

**JAX：**
1. 推理：大部分权重和计算使用 bfloat16，为稳定性部分计算使用 float32
2. 训练：默认混合精度：权重与梯度使用 float32，激活与大部分计算使用 bfloat16。你可以在配置中将 `dtype` 设置为 float32 来切换为全 float32 训练。

**PyTorch：**
1. 推理：与 JAX 一致 —— 大部分权重和计算使用 bfloat16，为稳定性将部分权重转换为 float32
2. 训练：支持全 bfloat16（默认）或全 float32。可在配置中通过 `pytorch_training_precision` 修改。bfloat16 占用更少显存，但通常相比 float32 会导致更高的损失值。当前尚不支持混合精度（mixed precision）。

在开启 `torch.compile` 的情况下，JAX 与 PyTorch 在推理速度上可达到相近水平。

## 故障排查

下面收集了一些常见问题及解决方法。如果遇到问题，请先查阅本节；若仍无法解决，请按照仓库的 issue 提交流程提交问题（参见 [CONTRIBUTING.md](CONTRIBUTING.md)）。

| 问题                                      | 解决办法                                                                                                                                                                                    |
| ---------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `uv sync` 因依赖冲突失败                  | 尝试删除虚拟环境目录（`rm -rf .venv`）并重新运行 `uv sync`。若问题依旧，检查你安装的 `uv` 是否为最新版本（`uv self update`）。
| 训练过程中显存不足                        | 训练前设置 `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9`（或更高）以允许 JAX 使用更多显存。也可以使用 `--fsdp-devices <n>`（<n> 为 GPU 数量）启用 [FSDP 全分片数据并行](https://engineering.fb.com/2021/07/15/open-source/fsdp/)，这会降低显存占用但可能导致训练速度变慢（具体慢多少取决于你的环境）。若仍然不足，可考虑禁用 EMA。 |
| 策略服务器连接错误                        | 检查服务器是否已启动并监听预期端口。确认客户端与服务器之间的网络连通性与防火墙设置。                                                                                                     |
| 训练缺少 norm stats 报错                   | 在启动训练前运行 `scripts/compute_norm_stats.py`，并指定你的配置名以生成归一化统计。                                                                                                          |
| 数据集下载失败                            | 检查网络连接。对 HuggingFace 数据集，确保已登录（`huggingface-cli login`）。                                                                                                                 |
| CUDA/GPU 错误                             | 检查 NVIDIA 驱动是否正确安装。若使用 Docker，确保已安装 nvidia-container-toolkit。检查 GPU 兼容性。本项目不要求系统级别预装 CUDA 库 —— 依赖会通过 uv 安装。如果遇到 CUDA 问题，甚至可以尝试卸载系统级的 CUDA 库，因为系统库有时会导致冲突。 |
| 运行示例时报 import 错误                   | 确保已通过 `uv sync` 安装所有依赖。部分示例可能在各自 README 中列出额外依赖。                                                                                                              |
| 动作维度不匹配                            | 检查你的数据处理变换是否与机器人预期的输入/输出维度一致。查看策略类中的动作空间定义以确认匹配。                                                                                         |
| 训练损失发散                              | 检查你数据集中 `norm_stats.json` 的 `q01`、`q99` 与 `std` 值。某些很少使用的维度可能导致非常小的 `q01`/`q99`/`std`，归一化后会产生异常大的状态或动作。可手动调整归一化统计作为临时解决方案。 |
