# LeRobot / SmolVLA 工具索引

从仓库根目录运行本目录脚本。完整流程见 [VLA 指南](../docs/vla.md)，
当前采用的配置和真实结果见 [验证记录](../docs/vla_validation.md)。

环境缩写：**ROS** 为加载工作空间后的 `/usr/bin/python3 -s`；
**D** 为 `.venv-lerobot/bin/python -I`（CPU 数据环境）；
**G** 为 `.venv-smolvla/bin/python -I`（CUDA 模型环境）；
**P** 为系统 Python 的离线 JSON/NumPy 分析，不需要模型或机器人。
ROS 包内的 `observation_capture`、`action_alignment` 用 `ros2 run wheel_arm_manipulation ...` 调用。

## 当前流程入口

| 工具 | 环境 | 作用 |
| --- | --- | --- |
| `check_lerobot_data_env.py` | D | 核对数据栈、解码器及 LeRobot v3 接口，不转换数据 |
| `check_smolvla_gpu_env.py` | G | 核对核心依赖、CUDA/BF16 和小矩阵结果，不加载模型 |
| `download_smolvla_probe_assets.py` | G | 下载本地策略完整权重及 VLM 配置/处理器；首次固定下载提交号 |
| `collect_smolvla_demonstrations.py` | ROS | 按预定八条计划自主创建仿真、采集成功示范、记录保持和失败尝试；会执行动作 |
| `prepare_smolvla_collection.py` | ROS | 将扩充批次与已有集合划分合并，整理开发/独立测试转换和冻结统计 |
| `convert_lerobot_episode.py` | D | 单条记录的转换及逐像素、动作块读回验证 |
| `convert_lerobot_batch.py` | D | 原始示范级集合划分、连续片段转换、训练统计冻结和读回验证 |
| `train_smolvla_native.py` | G | 当前训练基线：LeRobot 原生批训练、FP32 可训练参数、BF16 计算、cosine warmup |
| `evaluate_smolvla_validation.py` | G | 完整验证起点的只读评估，按动作阶段统计物理误差，不更新参数 |
| `probe_smolvla_live.py` | G＋ROS 环境 | 默认只预测；`--execute` 明确请求仿真执行，支持动作块、重观测和保持验收 |
| `run_smolvla_trial.py` | ROS | 创建自己拥有的全新隔离仿真，完成一轮策略测试并清理，支持 GUI/RViz |
| `run_smolvla_acceptance_suite.py` | ROS | 匹配回归证据后，运行五个预先固定案例，遇到失败停止整组 |
| `validate_smolvla_release.py` | P | 重新验证至少五条实际试验的抓杯/保持、权重、代码哈希和配置，输出新清单 |
| `run_released_cup_policy.py` | G＋ROS 环境 | 在已有全新仿真中复用已验收配置并执行；不会启动第二套仿真 |

公开验收工具负责核心五轮，追加 GUI＋RViz 案例并合并验证的命令见指南。
`check`、素材下载、离线转换、训练和评估都不会发送机器人命令；带执行选项的实时入口和仿真测试器会发送。
新克隆只有代码，不能直接使用维护机器上的数据/检查点/发布清单绝对路径。

## 性能与辅助分析

| 工具 | 环境 | 作用 |
| --- | --- | --- |
| `smolvla_dataset_projection.py` | G 内部模块 | 每个数据集实例缓存所需列视图，避免未来动作查询解码无用图片，不修改安装包 |
| `check_smolvla_dataset_projection.py` | G | 对当前完整训练批次逐项比较输入与解码耗时；包含固定历史索引和子集，要求足够帧/至少四个片段 |
| `smolvla_precision.py` | G 内部模块 | 保留/恢复可训练参数 FP32 存储，防止严格加载后的隐式精度舍入 |
| `ros_probe_session.py` | 内部模块 | 持久 ROS 子进程、原子请求/结果文件、超时及取消等待 |
| `execute_vla_short_motion.py` | ROS 内部入口 | 同步采集、限幅/重定时、碰撞/反馈检查、控制器参考衔接和测量请求 |
| `vla_trial_summary.py` | P | 纯数值试验限制、RGB-D 抬升及保持判定，不发送动作 |
| `camera_native_timing.cpp` | C++ | 原生 Ignition transport 时间戳对照，不发布或解码图像 |
| `plot_smolvla_validation.py` | P | 将保存的预测绘成 PNG/SVG，不重新调用模型 |
| `compare_smolvla_horizons.py` | P | 在相同起点比较不同未来时间带的保存预测 |
| `compare_smolvla_independent.py` | P | 对齐独立记录的保存结果，比较持杯和阶段过渡误差 |
| `export_vla_kinematics.py` | ROS | 将安装的 URDF/TCP 运动链导出 JSON，供离线运动学诊断使用 |

绘图使用系统 Python，按需安装 `python3-matplotlib`，并用 `/usr/bin/python3 -s` 避免混入 user-site 的不同 NumPy。
原生相机计时程序的构建命令：

```bash
g++ -O2 -std=c++17 tools/camera_native_timing.cpp \
  -o /tmp/wheelarm_camera_native_timing \
  $(pkg-config --cflags --libs ignition-transport11 ignition-msgs8)
/tmp/wheelarm_camera_native_timing --help
```

## 保留的阶段性与对照实验

这些工具用于解释训练/推理问题，不能把冒烟检查或单样本记忆当作完整抓杯能力。
它们保留在仓库供复核，当前交付权重不默认使用这些实验开关。

| 工具 | 环境 | 作用 |
| --- | --- | --- |
| `check_smolvla_interface.py` | D | 单条转换数据的特征、归一化、动作块和 padding 接口检查，不下载/加载权重 |
| `probe_smolvla_forward.py` | G | 单批次 flow loss 和动作采样前向，不训练 |
| `probe_smolvla_train_step.py` | G | 一次反向及 AdamW 更新，支持实际五维 loss 对照，仅保存报告 |
| `probe_smolvla_overfit.py` | G | 一个机械臂运动或夹爪闭合样本的记忆实验，仅保存报告 |
| `probe_smolvla_two_chunks.py` | G | 两个样本交替训练对照，仅保存报告 |
| `train_smolvla_batch.py` | G | 早期自定义训练器，包含阶段/过渡采样、梯度累积和可选运动学辅助损失，保存权重 |
| `probe_smolvla_sampling_steps.py` | G | 保存检查点的采样步数与动作误差对照 |
| `probe_smolvla_noise_ensemble.py` | G | 对同一实时记录输入比较单/四采样；默认复现原权重，显式 `--compare-checkpoint` 可比较新权重 |
| `probe_gripper_reference_continuity.py` | ROS | 教师先夹起杯子，再做静止机械臂下的开口参考衔接对照；会执行动作，不使用 VLA |
| `recovery_collection_20261114.json` | 配置 | 固定六训练/两独立测试初始姿态的示范计划 |
| `rollout_recovery_collection_20261114.json` | 配置 | 历史策略接近/早期抬升姿态的恢复示范计划，失败 rollout 不作为成功标签 |

实际五维 loss、几何损失与原生 32 维 loss 的数字口径不同，不能直接按数值大小排序。
当前基线采用原生训练器；没有把几何辅助、状态增强、四采样均值或短预测窗口描述为最终成功配置。

## 运行离线测试

先在 ROS 环境中执行包测试和只依赖 ROS 消息的工具时序测试：

```bash
# 按实际工作空间修改；下面沿用主 README 的目录约定
WS="$HOME/wheelarm_ws"
REPO="$WS/src/wheelarm-lab"
source /opt/ros/humble/setup.bash
source "$WS/install/setup.bash"
cd "$REPO"
PYTHONPATH="$REPO/wheel_arm_manipulation:$PYTHONPATH" \
  /usr/bin/python3 -s -m pytest -q wheel_arm_manipulation/test
PYTHONPATH="$REPO/wheel_arm_manipulation:$PYTHONPATH" \
  /usr/bin/python3 -s tools/test_vla_motion_timing.py
```

其余工具测试在 ML 环境运行。下面明确排除上面已在 ROS 环境运行的时序模块，
避免在 `python -I` 环境内误导入 ROS 消息包：

```bash
.venv-smolvla/bin/python -I - <<'PY'
import sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path('tools').resolve()))
names = sorted(p.stem for p in Path('tools').glob('test_*.py')
               if p.name != 'test_vla_motion_timing.py')
suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(name) for name in names)
raise SystemExit(not unittest.TextTestRunner(verbosity=1).run(suite).wasSuccessful())
PY
```

这些测试检查数据/数值、归一化、来源隔离、动作 mask、训练采样、FP32 恢复、加载等价性、
实时编排失败分支、时钟错位与运动、发布配置和代码版本一致性。
使用临时数据与 mock，不自动执行真实采集、下载、完整训练或仿真运动。
需要完整模型/图形环境的闭环验收由上面的专门入口显式运行。
