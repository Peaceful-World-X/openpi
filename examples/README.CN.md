# examples 目录说明

本目录提供 OpenPI 的推理、机器人接入、仿真评测、数据转换和模型格式转换示例。大多数运行示例采用客户端/服务端架构：GPU 机器运行 `scripts/serve_policy.py`，机器人或仿真环境运行本目录中的客户端，通过 WebSocket 请求动作。

```text
examples/
├── README.CN.md                         # 本文件：examples 目录的中文索引
├── inference.ipynb                     # 从 checkpoint 创建 policy，并用虚拟输入执行一次推理
├── policy_records.ipynb                # 查看 policy 记录中的图像、状态和动作曲线
├── convert_jax_model_to_pytorch.py     # 将 OpenPI 的 JAX/Orbax checkpoint 转换为 PyTorch 权重
│
├── aloha_real/                         # ALOHA 双臂真机运行与数据转换
│   ├── README.md                       # 真机安装、启动、checkpoint 和场景布置说明
│   ├── main.py                         # ALOHA 真机客户端入口，连接远程 policy server
│   ├── env.py                          # 将 ALOHA 真机封装为 openpi-client Environment
│   ├── real_env.py                     # 真机控制环境，负责机器人复位、观测和动作执行
│   ├── robot_utils.py                  # 电机、机械臂、夹爪、相机和状态记录工具
│   ├── constants.py                    # ALOHA 关节、夹爪和控制相关常量
│   ├── video_display.py                # 订阅观测图像并实时显示视频
│   ├── convert_aloha_data_to_lerobot.py # 将 ALOHA HDF5 数据转换为 LeRobot 格式
│   ├── Dockerfile                      # ALOHA 真机客户端容器镜像
│   ├── compose.yml                     # 同时编排客户端和 policy server
│   ├── requirements.in                 # ALOHA 环境的直接依赖定义
│   └── requirements.txt                # 锁定后的完整 Python 依赖
│
├── aloha_sim/                          # ALOHA MuJoCo 仿真运行示例
│   ├── README.md                       # 仿真环境安装及 Docker/本地启动说明
│   ├── main.py                         # ALOHA 仿真客户端入口
│   ├── env.py                          # 将 ALOHA 仿真封装为 openpi-client Environment
│   ├── saver.py                        # rollout 结束后保存观测视频
│   ├── Dockerfile                      # ALOHA 仿真客户端容器镜像
│   ├── compose.yml                     # 编排仿真客户端和 policy server
│   ├── requirements.in                 # 仿真环境的直接依赖定义
│   └── requirements.txt                # 锁定后的完整 Python 依赖
│
├── droid/                              # DROID 真机推理、训练和数据处理
│   ├── README.md                       # DROID 真机远程推理和 baseline policy 说明
│   ├── README_train.md                 # 全量 DROID 预训练、微调和数据过滤说明
│   ├── main.py                         # DROID 真机客户端，采集相机/机器人状态并执行动作
│   ├── convert_droid_data_to_lerobot.py # 将 DROID 采集数据转换为 LeRobot 格式
│   └── compute_droid_nonidle_ranges.py # 计算非静止片段，生成训练采样过滤范围
│
├── libero/                             # LIBERO 仿真 benchmark 评测与数据转换
│   ├── README.md                       # LIBERO 安装、评测命令和官方结果说明
│   ├── main.py                         # 执行 LIBERO task suite，统计成功率并保存视频
│   ├── convert_libero_data_to_lerobot.py # 将 LIBERO RLDS 数据转换为 LeRobot 格式
│   ├── Dockerfile                      # LIBERO 评测客户端容器镜像
│   ├── compose.yml                     # 编排 LIBERO 客户端和 policy server
│   ├── requirements.in                 # LIBERO 环境的直接依赖定义
│   └── requirements.txt                # 锁定后的完整 Python 依赖
│
├── simple_client/                      # 无机器人时测试远程推理服务
│   ├── README.md                       # 最小客户端的启动说明
│   ├── main.py                         # 生成虚拟观测、请求动作并统计推理延迟
│   ├── Dockerfile                      # 最小客户端容器镜像
│   ├── compose.yml                     # 编排最小客户端和 policy server
│   ├── requirements.in                 # 客户端的直接依赖定义
│   └── requirements.txt                # 锁定后的完整 Python 依赖
│
└── ur5/
    └── README.md                       # UR5 数据变换、DataConfig 和 TrainConfig 接入教程
```

## 如何选择示例

| 目标 | 建议入口 |
| --- | --- |
| 快速确认 checkpoint 能否推理 | `inference.ipynb` |
| 检查远程 policy server 是否正常 | `simple_client/main.py` |
| ALOHA 真机部署 | `aloha_real/README.md` |
| ALOHA 仿真测试 | `aloha_sim/README.md` |
| DROID 真机部署或训练 | `droid/README.md`、`droid/README_train.md` |
| LIBERO benchmark 评测 | `libero/README.md` |
| 为新机器人编写数据和策略适配 | `ur5/README.md` |
| 转换为 PyTorch 模型 | `convert_jax_model_to_pytorch.py` |

## 通用文件约定

- `main.py`：对应平台的可执行客户端或评测入口。
- `env.py`：把机器人或仿真器适配到 `openpi-client` 的运行时接口。
- `convert_*_to_lerobot.py`：把平台原始数据转换为训练使用的 LeRobot 数据集。
- `requirements.in`：人工维护的直接依赖；修改依赖时优先编辑此文件。
- `requirements.txt`：解析并锁定后的完整依赖，用于复现独立客户端环境。
- `Dockerfile`：构建对应示例的客户端环境。
- `compose.yml`：通常同时启动 policy server 和客户端/仿真器。

## 运行关系

```text
机器人或仿真环境
    │  观测（图像、状态、语言指令）
    ▼
examples/<平台>/main.py
    │  WebSocket
    ▼
scripts/serve_policy.py  ──>  OpenPI checkpoint
    │
    ▼
动作序列返回机器人或仿真环境执行
```

具体安装步骤、参数和平台限制以各子目录中的 `README.md` 为准。
