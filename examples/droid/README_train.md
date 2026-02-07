# 在 DROID 上训练

下面介绍如何在完整的 DROID 数据集上微调 pi0.5 模型。这是对 pi05-DROID 训练流程的近似开源复现（在数据加载和使用的动作空间上有小的差异）。如果想在更小的、在 DROID 平台上采集的自定义数据集上微调模型，请参见下文。

与 openpi 其余部分使用 LeRobot 进行数据加载不同，对于完整的 DROID 训练我们需要使用 RLDS 作为数据格式（因为目前 LeRobot 在处理像 DROID 这样的大型数据集时扩展性不足——他们正在改进）。下面说明如何为 RLDS 数据加载更新你的 openpi 环境以及在哪里下载 DROID 数据集。

## 安装

RLDS 数据加载需要一些额外依赖。运行：
```bash
uv sync --group rlds
```

## 下载 DROID 数据集

安装 `gsutil`（Google Cloud CLI）后，你可以用下面的命令下载 DROID 数据集：
```
gsutil -m cp -r gs://gresearch/robotics/droid/1.0.1 <your_download_path>/droid/1.0.1
```

注意必须下载版本 1.0.1（不要用 v1.0.0）：1.0.1 包含完整的语言注释（约 75k 个 episode），而 v1.0.0 只有 30k 个 episode 的注释。如果你出于某种原因想使用其他版本，请修改 [src/openpi/training/droid_rlds_dataset.py] 中 `DroidRldsDataset` 对象的 `version="1.0.1"` 这一行。

下载 DROID RLDS 数据集需要约 1.8TB 磁盘空间。

## 运行

首先，在你的 `TrainConfig` 中将 `rlds_data_dir` 路径改为你下载 `droid` 数据集的目录（见 [src/openpi/training/config.py](src/openpi/training/config.py)）。

然后，计算归一化统计（大约需要 ~10 分钟）：
```bash
uv run --group rlds scripts/compute_norm_stats.py --config-name pi05_full_droid_finetune --max-frames 10_000_000
```

运行训练：
```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run --group rlds scripts/train.py pi05_full_droid_finetune --exp-name=my_experiment --overwrite
```

注意：原始的 pi0.5-DROID 模型使用的是关节速度（joint velocity）动作。关节速度动作与模拟评估环境不兼容（更难以模拟）。因此，我们不建议使用关节速度动作训练，而是在这里改用关节位置（joint position）动作。

## 计算需求

我们的 DROID 训练配置约需 8× H100 GPU 运行 ~2 天以收敛（100k 次迭代，bs256，约 1 个 epoch）。
如果从 PaliGemma 而不是 pi0 初始化开始，计划大约需要 8× H100 运行 ~5 天（240k 次迭代，即约 3 个 epoch）。

我们尝试过使用 LoRA 做更便宜的微调，但迄今为止没发现策略表现良好。


## 数据过滤

像任何多样化的真实机器人数据集一样，DROID 数据集并非完全“干净”，我们发现数据过滤能显著提高策略性能。具体来说，DROID 数据集中包含许多机器人不移动的空闲时间步（部分原因是用于数据采集的 VR 遥控接口），针对这些空闲转移的合适过滤可以改善策略表现。

默认情况下，我们的 openpi 训练配方实现了用于训练所有 pi-DROID 模型的相同空闲过滤器。我们通过预先计算训练期间要采样的数据集索引来实现该过滤。你可以查看 `compute_droid_nonidle_ranges.py`（位于 `examples/droid/compute_droid_nonidle_ranges.py`）了解我们如何计算这些索引。大致思路是：过滤掉那些下一段动作几乎都是空闲的时间步。在训练期间，我们的代码会自动从云端拉取预计算的索引列表并应用它们。如果你想修改空闲过滤/创建自定义采样逻辑，可以修改我们的脚本生成新的索引列表，并在 [src/openpi/training/config.py](src/openpi/training/config.py) 中通过 `filter_dict_path="<path_to_filter_dict>"` 参数提供该文件。

注意：我们的过滤索引列表仅对上文下载部分提到的 `droid/1.0.1` 数据集有效，并不适用于其他版本的 DROID 数据集；请确保你下载的是上述数据集版本！如果你有自定义的 DROID 版本，可以重新运行 `examples/droid/compute_droid_nonidle_ranges.py` 脚本生成新的采样索引列表。

## RoboArena

考虑将你的 DROID 策略提交到 RoboArena 基准（https://robo-arena.github.io/），它允许你在多样的任务与场景中、真实世界中评估你的策略！:)

如果对 RoboArena 有问题，请发邮件至 karl.pertsch@gmail.com。


# 在自定义 DROID 数据集上微调

下面说明如何在在 DROID 平台上采集的自定义（较小）数据集上微调模型。与其他数据集类似，我们首先将自定义 DROID 数据集转换为 LeRobot 格式，然后在其上微调模型（pi05-droid）。

注意：这里使用 LeRobot，因为我们假设自定义 DROID 微调数据集相对较小（<10 小时）。对于更大的数据集（如完整的 DROID 数据集），我们推荐使用 RLDS，因为效率更高（见上文示例）。


## 第 1 步：将自定义 DROID 数据集转换为 LeRobot

我们将使用 DROID 真实数据集的一个小子集作为示例。该子集仅包含 30 个示范——你会使用自己的数据集；这里给出下载我们子集的命令（1.6GB）：
```
gsutil -m cp -r gs://gresearch/robotics/droid_raw/1.0.1/IRIS/success/2023-12-04 <your_target_path>
```

我们还会下载 DROID 的语言注释，以便将示范与语言指令配对。对于你自己的数据，你可以手动添加语言指令而无需下载我们的注释。下载 DROID 语言注释（12MB）：
```
gsutil -m cp -r gs://gresearch/robotics/droid_raw/1.0.1/aggregated-annotations-030724.json <your_target_dir>
```

对于你自己的数据，请确保每个 episode 的目录包含一个名为 `recordings/MP4` 的文件夹——如果没有，你需要先使用此脚本中提到的方法从 SVO 文件提取 MP4（参见：https://github.com/droid-dataset/droid/blob/main/scripts/convert/svo_to_mp4.py）。

现在，我们使用 `convert_droid_to_lerobot.py` 脚本将该数据集转换为 LeRobot 格式（对于 30 个示范，耗时 <5 分钟）：
```
uv run examples/droid/convert_droid_data_to_lerobot.py --data_dir <your_target_path>
```

## 第 2 步：在你的自定义数据集上运行微调

现在可以在我们转换后的自定义数据集上运行微调。我们提供了一个示例配置，用于在自定义数据集上微调 `pi05_droid`。你可以轻松修改配置以适配其他基础模型，或者在 `config.py` 中使用你的自定义 DROID 数据集（搜索 `pi05_droid_finetune`）。

启动训练：
```
uv run scripts/train.py pi05_droid_finetune --exp-name=my_experiment --overwrite
```

训练完成后，可按照 [`examples/droid/README.md`](examples/droid/README.md) 中的说明部署策略并在机器人上运行。

