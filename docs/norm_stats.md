# 归一化统计

按照常见做法，我们的模型在训练和推理时会对本体感知（proprioceptive）状态输入和动作目标进行归一化。用于归一化的统计量在训练数据上计算，并与模型检查点一起存储。

## 重新加载归一化统计

当你在新数据集上微调我们的模型时，需要决定是 (A) 复用已有的归一化统计，还是 (B) 在新的训练数据上重新计算统计量。哪种做法更合适取决于你的机器人与任务与预训练数据中机器人/任务分布的相似度。下面列出了我们为各模型提供的预训练归一化统计列表。

**如果你的目标机器人与某个预训练统计相匹配，考虑重新加载相应的归一化统计。** 通过重新加载归一化统计，你的数据集中的动作将对模型来说更“熟悉”，可能带来更好的性能。你可以在训练配置中添加一个 `AssetsConfig`，指向相应的检查点目录和统计 ID，例如对 `pi0_base` 检查点中 Trossen（即 ALOHA）机器人统计的配置如下：

```python
TrainConfig(
    ...
    data=LeRobotAlohaDataConfig(
        ...
        assets=AssetsConfig(
            assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
            asset_id="trossen",
        ),
    ),
)
```

有关如何在完整训练配置中重新加载归一化统计的示例，请参见 `pi0_aloha_pen_uncap` 配置（位于 [training config 文件](https://github.com/physical-intelligence/openpi/blob/main/src/openpi/training/config.py)）。

**注意：** 要成功重新加载归一化统计，关键是你的机器人与数据集需要遵循预训练时使用的动作空间定义。我们在下文提供了动作空间定义的详细说明。

**注意 #2：** 重新加载归一化统计是否有益取决于你的机器人与任务与预训练数据分布的相似度。我们建议总是同时尝试两种方式：复用预训练统计以及在新数据集上重新计算统计（参见 [主 README](../README.md) 获取如何计算新统计的说明），然后选择对你的任务效果更好的那种方案。


## 提供的预训练归一化统计

下面列出了我们提供的预训练归一化统计。我们为 `pi0_base` 和 `pi0_fast_base` 两个模型都提供这些统计。对于 `pi0_base`，请将 `assets_dir` 设置为 `gs://openpi-assets/checkpoints/pi0_base/assets`；对于 `pi0_fast_base`，请设置为 `gs://openpi-assets/checkpoints/pi0_fast_base/assets`。

| 机器人 | 描述 | 资产 ID |
|-------|-------------|----------|
| ALOHA | 带并联夹持器的 6 自由度双臂机器人 | trossen |
| Mobile ALOHA | 安装在 Slate 底盘上的移动版 ALOHA | trossen_mobile |
| Franka Emika (DROID) | 基于 DROID 设置的 7 自由度机械臂，带并联夹持器 | droid |
| Franka Emika (non-DROID) | 配备 Robotiq 2F-85 夹持器的 Franka FR3 臂 | franka |
| UR5e | 配备 Robotiq 2F-85 夹持器的 6 自由度 UR5e 臂 | ur5e |
| UR5e bi-manual | 双臂 UR5e 配置，带 Robotiq 2F-85 夹持器 | ur5e_dual |
| ARX | 双臂 ARX-5 机械臂系统，带并联夹持器 | arx |
| ARX mobile | 安装在 Slate 底盘上的移动版双臂 ARX-5 系统 | arx_mobile |
| Fibocom mobile | 带 2 臂 ARX-5 的 Fibocom 移动机器人 | fibocom_mobile |


## Pi0 模型动作空间定义

开箱即用时，`pi0_base` 和 `pi0_fast_base` 都使用下面的动作空间定义（“左/右”是从机器人背后朝工作区方向看的定义）：
```
    "dim_0:dim_5": "left arm joint angles",
    "dim_6": "left arm gripper position",
    "dim_7:dim_12": "right arm joint angles (for bi-manual only)",
    "dim_13": "right arm gripper position (for bi-manual only)",

    # For mobile robots:
    "dim_14:dim_15": "x-y base velocity (for mobile robots only)",
```

本体感知状态使用与动作空间相同的维度定义，移动机器人除外：移动机器人的最后两个维度（基座 x-y 位置）不会包含在本体感知状态中。

对于 7 自由度机器人（如 Franka），我们在动作空间中使用前 7 个维度表示关节动作，第 8 个维度表示夹持器动作。

Pi 机器人通用信息：
- 关节角以弧度表示，位置零对应各机器人接口库报告的零位，ALOHA 的标准代码使用了稍有不同的约定（详情见 [ALOHA 示例代码](../examples/aloha_real/README.md)）。
- 夹持器位置范围为 [0.0, 1.0]，0.0 表示完全打开，1.0 表示完全闭合。
- 控制频率：UR5e 与 Franka 常用 20 Hz，ARX 与 Trossen（ALOHA）臂常用 50 Hz。

对于 DROID，我们使用 DROID 的原始动作配置：前 7 个维度为关节速度动作，第 8 个维度为夹持器动作，控制频率为 15 Hz。
