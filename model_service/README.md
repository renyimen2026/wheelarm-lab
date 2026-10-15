# Python CUDA 模型服务

[返回项目首页](../README.md) · [导航与数据参考](../docs/reference.md)

独立于 ROS 2 的本机推理进程，使用 PyTorch CUDA、Transformers、bitsandbytes NF4
4-bit 量化及 XGrammar 结构化输出。默认监听 `127.0.0.1:11435`，
不替换或关闭 `11434` 上的 Ollama，只实现项目所需的协议子集。

**本页安装、下载、启动和模型测试命令均在仓库根目录执行。**
ROS 客户端示例另有标注；不要把模型依赖安装进 ROS 环境。

## 安装与下载

创建干净的独立环境，避免克隆环境中混杂的 NumPy 二进制文件：

```bash
conda create -n wheelarm-llm python=3.10 -y
conda activate wheelarm-llm
export PYTHONNOUSERSITE=1
python -m pip install torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r model_service/requirements.txt
```

本机验证组合为 PyTorch `2.5.1+cu121`、RTX 4060 Laptop 8GB、NVIDIA 535 驱动。
这是已测试组合，不保证任意硬件可运行；CUDA 不可用时服务拒绝启动，不静默回退 CPU。

下载官方 Transformers 权重：

```bash
env -u PYTHONPATH PYTHONNOUSERSITE=1 HF_HUB_DISABLE_XET=1 \
  python -m model_service.download --models-root ~/Models/transformers
```

| 服务中的模型名 | 官方权重 | 本地子目录 |
| --- | --- | --- |
| `qwen3:4b` | [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) | `Qwen3-4B/` |
| `qwen3-vl:4b-instruct` | [Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) | `Qwen3-VL-4B-Instruct/` |

两个服务名是固定本地权重的兼容别名，不是调用 Ollama。
权重总计约 17GB，加载时才转换为 4-bit；下载大小不是显存需求。
可添加 `--model text` 或 `--model vision` 单独下载，中断后重新运行可续传。
不能直接复用 Ollama 的 GGUF/`blobs` 文件。推理只读本地 Safetensors，
不下载模型、不启用远程代码；模型许可证以各官方仓库为准。

## 启动与接入

在模型终端保持独立环境，启动服务：

```bash
conda activate wheelarm-llm
env -u PYTHONPATH PYTHONNOUSERSITE=1 \
  CUDA_VISIBLE_DEVICES=0 MKL_THREADING_LAYER=GNU \
  python -m model_service.server
```

默认先加载语言模型，完成后才开始监听；看到 `Preload complete` 和接口地址再发请求。
保持终端运行，`Ctrl+C` 关闭服务。清理 `PYTHONPATH` 只针对此进程，
不要在 ROS 终端全局删除该变量；GNU MKL 线程层用于避免与 PyTorch OpenMP 冲突。

**另开 ROS 终端，在工作空间根目录执行：**

```bash
source install/setup.bash

ros2 run wheel_arm_navigation language_goal "先去厨房再去卧室" \
  --endpoint http://127.0.0.1:11435 --preview

ros2 run wheel_arm_navigation language_observe "去厨房看看有没有椅子" \
  --endpoint http://127.0.0.1:11435 --preview

ros2 run wheel_arm_navigation scene_describe --question "画面里有哪些家具？" \
  --endpoint http://127.0.0.1:11435 --timeout 600
```

名称必须已记录。去掉导航示例的 `--preview` 后才进入人工确认和执行，
且需要已启动导航、完成定位、退出遥控。视觉示例只需相机与时钟。
现有 ROS 客户端默认仍访问 `11434`，因此这里的 `--endpoint` 不能省略。

## 加载策略

默认流程：

```text
服务启动 → 语言模型就绪 → 解析与确认 → Nav2 接受目标
                                       ├─ 导航移动
                                       └─ 后台加载视觉模型
到达 → 停稳检查 → 拍摄新帧 → 等待剩余加载 → 视觉问答
```

`language_observe` 在客户端后台线程调用加载接口，ROS 导航等待和取消处理继续运行。
一次只常驻一个模型，加载视觉模型时卸载语言模型，不同时占用两份权重。
预览、未确认、目标拒绝时不预加载；不提前拍摄移动中的画面。

预加载失败不取消导航，到达后尝试正常按需推理。到达时尚未完成则等待，
等待计入视觉超时预算。任务日志 `vision_preload` 保存状态、原因与耗时。
旧 Ollama 或未声明该能力的后端保持原流程。

| 选项 | 默认值 | 用途 |
| --- | --- | --- |
| `--models-root` | `~/Models/transformers` | 本地权重根目录 |
| `--port` | `11435` | 仅绑定 IPv4 回环地址 |
| `--preload` | `text` | `text` / `vision` 启动预加载并驻留，`none` 禁用 |
| `--keep-loaded` | 启动预加载时自动开启 | 在 `--preload none` 下也可保留最近的模型 |
| `--generation-timeout` | `120` 秒 | 生成步间检查的预算，不是加载和预处理的硬超时 |

只做视觉实验时可用 `--preload vision`。
需要完全按请求加载并卸载时使用 `--preload none`，且不加 `--keep-loaded`。
需要关闭导航期间加载时，在 ROS 命令添加 `--no-vision-preload`。

任务完成后最近驻留的是视觉模型，下次语言解析仍需切换。
退出客户端不能可靠中止已经运行的 CUDA 加载，服务端可能完成加载并继续驻留；
不会因此拍摄或发送新的机器人动作。

## 资源与性能

日志报告 CUDA 版本、显卡、加载后显存、模型加载、输入准备及生成耗时。
预加载只将等待提前，首次请求仍可能包含语法编译与算子初始化。
图片按原尺寸进入处理器，不额外缩小；不保证比 Ollama Vulkan 更快。

以下为 2026-10-14 本机单次样本，未同时运行 Gazebo，不能作为吞吐保证：

| 场景 | 首次请求 | 同模型、同请求重复执行 |
| --- | --- | --- |
| 语言解析，未启动预加载但保留模型 | 13.07 秒 | 1.52 秒 |
| 图片家具问答，未预加载视觉模型 | 15.73 秒 | 3.65 秒 |
| 启动预加载语言模型后解析 | 3.88 秒 | 1.55 秒 |

后台视觉加载另测约 9.23 秒，图片请求时 `load=0.000s`；这不代表图片处理和生成耗时为零。
模型加载后分配显存：语言约 2.48GiB、视觉约 3.27GiB，不是推理峰值。
加载和仿真共用 GPU、CPU、磁盘，可能影响流畅度；显存不足时报错并卸载，不转 CPU。
导航时序由隔离模拟 Nav2 测试验证，完整 Gazebo 并行性能仍需实际评估。

## 接口与边界

| 接口 | 行为 |
| --- | --- |
| `GET /health` | 后端信息，不证明定位就绪或模型始终已加载 |
| `POST /api/show` | 检查本地文件与能力，声明预加载扩展是否可用 |
| `POST /api/chat` | 文本或单张用户图片，非流式、关闭思考、`temperature=0` |
| `POST /api/preload` | 驻留模式下加载固定模型；HTTP 等待加载完成，ROS 客户端在后台调用 |

JSON Schema 由 XGrammar 约束生成，再用 `jsonschema` 严格验证。
截断、超时、字段不合规返回错误，不修补成可执行任务。
模型名不能指定任意路径或触发远程下载，客户端的坐标、目标、确认与失败校验保持不变。

请求上限 12MiB，单张 PNG/JPEG 最多 1920×1080，上下文最多 4096 token，
输出最多 1024 token；超限拒绝，不静默截断。
一次只执行一个模型请求，忙时返回 503，不排队。客户端退出后推理可能继续至完成或预算结束。
无用户鉴权，仅用于可信本机；不要通过反向代理或隧道暴露到公网。

## 测试

协议、输入校验和启动策略测试，不加载模型权重：

```bash
env -u PYTHONPATH PYTHONNOUSERSITE=1 MKL_THREADING_LAYER=GNU \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  python -m pytest model_service/test_service.py -q
```

已有完整本地模型时，加入真实分词器与图片处理器回归：

```bash
env -u PYTHONPATH PYTHONNOUSERSITE=1 MKL_THREADING_LAYER=GNU \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 WHEELARM_MODEL_TEST=1 \
  python -m pytest model_service/test_service.py model_service/test_grammar.py -q
```

实际 CUDA 文本测试自动启动隔离端口服务并在结束后关闭，不发机器人动作：

```bash
env -u PYTHONPATH PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 MKL_THREADING_LAYER=GNU \
  python -m model_service.smoke_test --preload text
```

添加 `--image '<实际观测目录>/rgb.png'` 可测试已有相机观测，
再加 `--overlap-vision` 验证后台加载时主线程可继续运行。
占位路径需替换为真实观测目录，不下载新图片、不启动导航；
后台加载测试本身也不代表真实机器人正在移动。
