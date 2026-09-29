# UR7 客户端最小环境

这个环境只用于 `deplay.ur7_client`、UR7 RTDE 控制、RealSense 相机和远端 WebSocket 推理。策略模型、JAX、CUDA、Torch 和训练依赖继续安装在 OpenPI 服务端，不会安装到这个环境。

要求 Python 3.10、3.11 或 3.12。首次安装：

```bash
cd /16T-2/sudu/code/openpi
bash deplay/install_client_env.sh
```

脚本默认使用 `/16T-2/sudu/code/openpi/.venv`。也可以传入其他环境目录：

```bash
bash deplay/install_client_env.sh /16T-2/sudu/code/openpi/.venv-ur7
```

安装脚本会先以 editable 方式安装仓库内的 `packages/openpi-client`，再安装 [requirements-client.txt](requirements-client.txt) 中的 UR RTDE、RealSense、PyYAML、SciPy 和 NumPy。这里不安装策略模型和训练依赖。

安装完成后，在仓库根目录运行客户端：

```bash
cd /16T-2/sudu/code/openpi
./.venv/bin/python -m deplay.ur7_client --help
```

连接 GPU 策略服务器并执行机械臂：

```bash
./.venv/bin/python -m deplay.ur7_client \
  --config deplay/client.yaml \
  --server ws://GPU_SERVER_IP:9000 \
  --execute --confirm ENABLE_UR7_MOTION
```

真正执行前需要保证 GPU 服务器地址可达、策略服务已启动、UR 控制柜处于远程控制状态，并显式提供确认字符串。
