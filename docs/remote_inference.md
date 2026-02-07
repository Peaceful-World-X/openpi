
# Running openpi models remotely

我们提供了一些工具用于在远程服务器上运行 openpi 模型。此方式适用于在机器人外部使用更强的 GPU 进行推理，同时能将机器人运行环境与策略服务器分离（例如避免机器人软件带来的依赖地狱）。

## 启动远程策略服务器

要启动远程策略服务器，可直接运行：

```bash
uv run scripts/serve_policy.py --env=[DROID | ALOHA | LIBERO]
```

参数 `env` 用于指定要加载的 π_0 检查点。脚本内部会执行类似下面的命令来启动策略服务，例如对 DROID 环境使用的命令：

```bash
uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi0_fast_droid --policy.dir=gs://openpi-assets/checkpoints/pi0_fast_droid
```

该命令将启动一个策略服务器，按 `--policy.config` 和 `--policy.dir` 指定的策略提供服务。默认监听端口为 8000（可配置）。

## 在机器人端向远程策略服务器发送查询

我们提供了一个依赖最小的客户端库，便于将远程推理集成到任何机器人代码中。

首先在机器人环境中安装 `openpi-client`：

```bash
cd $OPENPI_ROOT/packages/openpi-client
pip install -e .
```

然后在机器人代码中使用客户端进行推理调用，示例：

```python
from openpi_client import image_tools
from openpi_client import websocket_client_policy

# Outside of episode loop, initialize the policy client.
# Point to the host and port of the policy server (localhost and 8000 are the defaults).
client = websocket_client_policy.WebsocketClientPolicy(host="localhost", port=8000)

for step in range(num_steps):
    # Inside the episode loop, construct the observation.
    # Resize images on the client side to minimize bandwidth / latency. Always return images in uint8 format.
    # We provide utilities for resizing images + uint8 conversion so you match the training routines.
    # The typical resize_size for pre-trained pi0 models is 224.
    # Note that the proprioceptive `state` can be passed unnormalized, normalization will be handled on the server side.
    observation = {
        "observation/image": image_tools.convert_to_uint8(
            image_tools.resize_with_pad(img, 224, 224)
        ),
        "observation/wrist_image": image_tools.convert_to_uint8(
            image_tools.resize_with_pad(wrist_img, 224, 224)
        ),
        "observation/state": state,
        "prompt": task_instruction,
    }

    # Call the policy server with the current observation.
    # This returns an action chunk of shape (action_horizon, action_dim).
    # Note that you typically only need to call the policy every N steps and execute steps
    # from the predicted action chunk open-loop in the remaining steps.
    action_chunk = client.infer(observation)["actions"]

    # Execute the actions in the environment.
    ...

```

其中 `host` 和 `port` 指定远程服务器的 IP 与端口。你也可以将它们作为命令行参数传入或写死在机器人代码中。`observation` 是一个包含观测项和 prompt 的字典，需按被服务策略的输入规范构造。关于不同环境构建该字典的具体示例，请参见 [simple client 示例](../examples/simple_client/main.py)。
