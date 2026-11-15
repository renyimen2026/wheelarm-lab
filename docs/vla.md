# SmolVLA 杯子抓取操作指南

本指南对应 Ubuntu 22.04、ROS 2 Humble、Gazebo Fortress、Python 3.10 和 LeRobot 0.4.4。
当前基线是高位相机、固定桌面杯位、直立无柄陶瓷杯的慢速仿真抓取与保持。
模型已经在维护环境和用户复测中成功抓杯，具体范围、参数和结果见 [验证记录](vla_validation.md)。

仓库不包含采集图片、LeRobot 数据集、模型权重或可运行的本地发布清单。
已有本地清单可直接看“运行已有配置”；新环境应按采集、转换、训练、验证的顺序准备。
早期对照实验与当前基线的区别见 [工具索引](../tools/README.md)。

## 目录和环境约定

每个终端设置自己的工作空间路径。下面沿用主 README 的 `~/wheelarm_ws`；
本机已有的 `~/Projects/wheel_arm_ws` 等目录应替换这一行。

```bash
WS="$HOME/wheelarm_ws"
REPO="$WS/src/wheelarm-lab"
source /opt/ros/humble/setup.bash
source "$WS/install/setup.bash"
```

- `ros2 run/launch` 使用系统 ROS Python，在工作空间执行。
- `tools/` 在仓库目录执行；数据转换使用 `.venv-lerobot`，模型使用 `.venv-smolvla`。
- ML 命令使用 `python -I` 隔离 ROS/user-site Python 路径；ROS 子进程继承已加载的工作空间环境。
- 新输出目录默认拒绝覆盖。失败目录保留用于排查，重试选择新的目录，不删除旧结果冒充第一次成功。
- 以下自主采集/`--execute`/完整验收命令会实际控制仿真机械臂；先结束其他运动客户端。

## 观测、动作和成功结果

| 数据 | 语义 |
| --- | --- |
| `observation.images.mast` | 当前 RGB；原图 480×640×3，策略加载器转换为 CHW |
| `observation.state` | 实际四个关节角＋夹爪总开口，顺序固定为下表 |
| 任务文本 | 当前基线为“拿起桌上的杯子并保持夹持” |
| `action` | 未来绝对位置控制器参考，四个角度＋夹爪总开口；训练标签，不作为模型输入 |

状态/动作顺序为
`[shoulder_pan_joint, shoulder_lift_joint, elbow_joint, wrist_pitch_joint, gripper_width]`，
单位 `[rad, rad, rad, rad, m]`。夹爪总开口等于左右手指位置之和；按关节名读取，排除轮子和 effort。
内部网络保留预训练模型的 32 维容量，外部机器人接口只有五维，其余维度补齐。

图像和关节观测近似同步，默认最大差 50 ms，拒绝超过 500 ms 的旧/未来观测。
控制器状态优先读取 `reference.positions`，兼容 `desired.positions`；保存当前插值参考，不是最终轨迹终点。
实际状态在训练与推理时均可获得，控制器参考只用于制作标签。

15 Hz 是采集目标，原始记录可能存在间隔波动，必须保留真实时间戳。
离线转换采用 10 Hz 网格；`available-past` 选择网格时刻前已可用的图像，默认年龄上限 100 ms，
状态按时间戳插值。无效槽位保留拒绝原因，不通过删除槽位压缩时间；转换时拆为独立连续片段。

单步标签对应观测之后 **+0.1 s** 的参考。因此加载器的动作偏移
`[0, 0.1, …, 4.9] s` 对应物理参考偏移 `[+0.1, +0.2, …, +5.0] s`，不能再额外错移一帧。
片段末尾不足 50 步的部分有时间 padding，原生训练通过 mask 排除；完整窗口评估单独统计。

任务结果由 `cup_grasp` 或 VLA 的独立 RGB-D 检查记录。
目录存在、控制器到位或单帧闭爪都不自动等于成功。
`finish=hold` 时 `placement_verified=false` 是预期结果，因为没有执行放回。

## 安装两套独立环境

需要已安装 `uv`；PyTorch、LeRobot 与 Transformers 不安装进 ROS 系统 Python。
CPU TorchCodec wheel 当前限定 Python 3.10 / Linux x86_64，模型张量计算仍使用 CUDA。

```bash
cd "$REPO"
export PATH="$HOME/.local/bin:$PATH"
uv venv --python /usr/bin/python3 .venv-lerobot
uv pip install --python .venv-lerobot/bin/python --torch-backend cpu \
  -r tools/lerobot-data-requirements.txt
.venv-lerobot/bin/python -I tools/check_lerobot_data_env.py

uv venv --python /usr/bin/python3 .venv-smolvla
uv pip install --python .venv-smolvla/bin/python --torch-backend cu118 \
  -r tools/smolvla-gpu-requirements.txt
.venv-smolvla/bin/python -I tools/check_smolvla_gpu_env.py
```

已验证 GPU 为 RTX 4060 Laptop 8 GB；原生 batch=8 训练峰值张量显存约 3.4 GiB。
检查器只做导入和小 CUDA 运算，不下载权重，不等于已验证完整训练。
TorchCodec 需要可加载的 FFmpeg 库；Ubuntu 可安装 `ffmpeg`。
遇到 `libnppicc.so.11` 缺失时，核对安装的是要求文件中的 CPU TorchCodec，不要替换 CUDA PyTorch。
`0.5` 与 `0.5.0` 为等价版本写法，检查器按规范化版本比较。
核心模型及数据栈版本已固定，其他转依赖实际版本由环境检查报告记录，并非完整依赖锁文件。

下载模型素材：

```bash
ASSETS="$WS/observations/model_assets/smolvla_base"
.venv-smolvla/bin/python -I tools/download_smolvla_probe_assets.py --assets-dir "$ASSETS"
```

第一次下载解析远端提交号并先写 `download_plan.json`，随后下载；中断重试沿用原计划。
不重复下载第二份 VLM 主干权重；VLM 仓库只提供配置、处理器和分词文件。
重现维护环境的历史权重时，需要使用对应的素材提交号，见验证记录，不能把新下载的不同版本当作同一基线。

## 采集带标签的 MoveIt 示范

终端 A，在工作空间启动全新场景：

```bash
cd "$WS"
ros2 launch wheel_arm_manipulation cup_test.launch.py \
  camera_mount:=mast cup_model:=ceramic_cup
```

终端 B，加载上述 ROS 环境后连续采集：

```bash
cd "$WS"
ros2 run wheel_arm_manipulation observation_capture \
  --record --record-references --hz 15 --rgb-reliability reliable \
  --instruction "拿起桌上的杯子并保持夹持" \
  --output-dir "$WS/observations/vla/batch01"
```

终端 C，从终端 B 的 `[记录目录]` 复制本轮完整目录，明确赋值后再执行：

```bash
read -r -p "粘贴本轮完整记录目录: " EPISODE_DIR
export EPISODE_DIR
if [ -n "$EPISODE_DIR" ] && [ -f "$EPISODE_DIR/frames.jsonl" ]; then
  ros2 run wheel_arm_manipulation cup_grasp \
    --execute --finish hold --hold-seconds 10 --episode-dir "$EPISODE_DIR"
else
  echo "目录不正确，请核对采集终端打印的完整路径。"
fi
```

确认后执行 `open → pregrasp → approach → close → lift → hold`。
完成保持验证后，在终端 B 用 Ctrl+C 正常结束记录。
变量不会从一个终端自动传到另一个终端；空变量会把 `cat "$EPISODE_DIR/task_metadata.json"`
变成读取根目录下的文件。

```text
<episode>/
├── images/000000.png …
├── frames.jsonl
├── controller_references.jsonl
├── summary.json
├── task_metadata.json
└── task_events.jsonl
```

`summary.json` 保存实际频率、间隔、拒绝/保存计数、配对年龄、QoS 及转换/写入耗时。
`task_metadata.json` 保存任务 ID、完成状态、成功结果及抬升/放回验证；事件文件保存阶段边界和保持记录。
未关联 `--episode-dir` 的记录可能只有观测而没有任务元数据，不自动补成成功示范。

在终端 C 离线检查：

```bash
ros2 run wheel_arm_manipulation action_alignment "$EPISODE_DIR" \
  --task-segment grasp-hold --chunk-size 4 --chunk-step 0.1 \
  --time-grid --image-policy available-past
```

生成 `grasp_hold_available_time_grid.jsonl` 和报告，原始采集文件不覆盖。
不加 `--time-grid` 时，工具生成绝对参考、参考减实际状态、未来动作块三种诊断候选；
候选文件本身不是最终 LeRobot 数据集。当前训练路线选择未来绝对位置参考。

需要自主扩充示范时，可从仓库目录运行下面的独立采集器；它会创建和控制自己拥有的仿真副本：

```bash
/usr/bin/python3 -s tools/collect_smolvla_demonstrations.py \
  --root "$WS/observations/vla/recovery_batch" \
  --plan tools/recovery_collection_20261114.json
```

该计划预先固定六条训练和两条独立测试初始状态；采集器只对自己的进程清理，失败尝试保留。
`--resume` 复用已有计划与结果，不改变已经固定的集合划分。
另一个 `rollout_recovery_collection_20261114.json` 是从历史策略接近状态整理出的恢复示范计划，
其中 `source_rollout` 是历史来源标记，不是新克隆仓库自带的文件。

## LeRobot 转换和集合划分

先按**原始完整示范**划分 train/validation；所有派生连续片段继承父记录集合，不能随机拆帧后再划分。
批量转换输入 JSON 结构如下，任务 ID 必须与各自元数据一致，路径须替换为实际绝对路径：

```json
{
  "split_unit": "original_recording_all_derived_segments_inherit_split",
  "episodes": [
    {"split": "train", "task_id": "训练记录的任务ID", "episode_dir": "/absolute/path/to/train_episode"},
    {"split": "validation", "task_id": "验证记录的任务ID", "episode_dir": "/absolute/path/to/validation_episode"}
  ]
}
```

```bash
cd "$REPO"
DATASET="$WS/observations/lerobot/grasp_hold_batch"
.venv-lerobot/bin/python -I tools/convert_lerobot_batch.py \
  "$WS/observations/vla/collection_split.json" --output "$DATASET"
```

每个连续片段转换为一个 LeRobot episode，训练/验证分别存放。
转换器重新核对源记录和网格，逐行、逐像素读回，并核对 4/50 步动作和尾部 padding。
`training_stats.json` 只来自训练集合；`source_mapping.jsonl` 和转换清单保留来源及哈希。
`--normalization-root <既有数据集>` 用于扩充训练数据时继承兼容的冻结统计，不用验证集重算归一化。

`prepare_smolvla_collection.py <新采集根目录> --old-split <已有划分JSON> --convert-root <新目录>`
用于合并已成功采集的扩充批次，生成 `development` 和 `independent_test` 两份数据；
它需要已有开发集合划分，不是从空目录开始的入口。独立测试记录不进入开发验证集合。

## 当前训练基线

设置 `ASSETS` 和 `DATASET` 为上面准备好的本地目录；训练不连接 ROS 或机器人。

```bash
BASE_RUN="$WS/observations/training/native_2000"
.venv-smolvla/bin/python -I tools/train_smolvla_native.py "$DATASET" \
  --assets-dir "$ASSETS" --steps 2000 --batch-size 8 --save-every 500 \
  --learning-rate 1e-4 --project-delta-columns --output-dir "$BASE_RUN"

FINAL_RUN="$WS/observations/training/native_finetune_2000"
.venv-smolvla/bin/python -I tools/train_smolvla_native.py "$DATASET" \
  --assets-dir "$ASSETS" --initial-checkpoint "$BASE_RUN/checkpoint_final" \
  --steps 2000 --batch-size 8 --save-every 500 --learning-rate 3e-5 \
  --project-delta-columns --output-dir "$FINAL_RUN"
CHECKPOINT="$FINAL_RUN/checkpoint_final"
```

使用 LeRobot 原生批加载、官方 32 维 flow loss 和时间 padding mask、AdamW、cosine warmup。
当前路线不启用自定义几何损失或状态增强。VLM 主干冻结，可训练参数 FP32 存储，计算使用 BF16；
重新加载时恢复保存的 FP32 权重精度。第二段是从权重继续微调并新建优化器，不是恢复旧优化器状态。
这套命令描述训练流程，不承诺任意新采集批次在相同步数必然得到同一效果。

`--project-delta-columns` 只优化未来动作列读取，当前 RGB 仍正常解码并输入模型；
不修改安装目录中的 LeRobot。它解决的是加载性能，不是额外的动作监督。

完整离线评估：

```bash
.venv-smolvla/bin/python -I tools/evaluate_smolvla_validation.py "$DATASET" \
  --checkpoint "$CHECKPOINT" --assets-dir "$ASSETS" \
  --output-dir "$WS/observations/evaluations/full_validation"
```

若另有独立测试转换目录，添加 `--evaluation-dataset <independent_test目录>`。
评估读取所有完整 50 步起点，分实际动作所属阶段统计物理误差；重叠动作比较不是独立样本。
固定少量起点的训练期评估、低 loss 或单样本记忆都不能代替闭环抓杯测试。

## 从只读预测到完整验收

在上面的全新杯子场景中先只读预测；不指定 `--execute` 不发送动作：

```bash
.venv-smolvla/bin/python -I tools/probe_smolvla_live.py "$DATASET" \
  --checkpoint "$CHECKPOINT" --assets-dir "$ASSETS" --samples 3 \
  --output-dir "$WS/observations/diagnostics/live_predictions"
```

完整单轮回归会自行启动全新、隔离的仿真并执行策略，不在策略试验中运行教师抓取：

```bash
REGRESSION="$WS/observations/autonomous/cup_regression"
/usr/bin/python3 -s tools/run_smolvla_trial.py \
  --dataset "$DATASET" --checkpoint "$CHECKPOINT" --assets-dir "$ASSETS" \
  --seed 20261123 --ensemble-size 1 --gripper-start-source reference \
  --output-dir "$REGRESSION"
```

当前验收配置为 50 步预测，每轮执行前 10 步、名义间隔 0.1 秒，再重新观测。
轨迹按现有速度/加速度约束放慢，实际耗时不等于名义一秒。
`reference` 衔接要求控制器参考新鲜、静止且与实测差在原每指 1 mm 容限内；
四关节起点和模型输入仍使用实际状态，预测目标不替换为预设抓取程序。

核心验收预先固定五个新案例、两个初始姿态和多个采样种子；任何失败都会停止整组，不挑成功案例替代：

```bash
SUITE="$WS/observations/autonomous/cup_acceptance"
/usr/bin/python3 -s tools/run_smolvla_acceptance_suite.py \
  --dataset "$DATASET" --checkpoint "$CHECKPOINT" --assets-dir "$ASSETS" \
  --regression-trial "$REGRESSION" --ensemble-size 1 --gripper-start-source reference \
  --first-seed 20261131 --output-dir "$SUITE"
```

先确认 `suite_result.json` 全部通过。公开的五轮工具只生成核心验收清单；
复现本次完整使用环境还应追加 GUI＋RViz 复测。使用核心清单的默认种子，保持其余配置相同：

```bash
GUI_TRIAL="$WS/observations/autonomous/cup_gui_check"
/usr/bin/python3 -s tools/run_smolvla_trial.py \
  --dataset "$DATASET" --checkpoint "$CHECKPOINT" --assets-dir "$ASSETS" \
  --seed 20261131 --ensemble-size 1 --gripper-start-source reference \
  --gui --rviz --output-dir "$GUI_TRIAL"

# 将五轮和图形界面试验一起重新核对，输出新的最终清单
/usr/bin/python3 -s tools/validate_smolvla_release.py \
  --trials "$SUITE/cup_acceptance_case_01" "$SUITE/cup_acceptance_case_02" \
    "$SUITE/cup_acceptance_case_03" "$SUITE/cup_acceptance_case_04" \
    "$SUITE/cup_acceptance_case_05" "$GUI_TRIAL" \
  --assets-dir "$ASSETS" --output "$WS/observations/autonomous/cup_release.json"
```

发布校验会重算杯子抬升和保持结果，并核对权重、运行代码和推理配置。
本次完整基线要求杯子抬升 80±10 mm，并在停止新动作后保持至少十秒墙钟和十秒 ROS 时间。
本机另保留 GUI/RViz 进程存活采样；上述公开命令的最终清单以六个实际试验目录为证据。
后续复测失败应保留失败目录并撤销该候选的可交付状态，不能仅因前五轮成功而忽略失败。

## 运行已有配置

先启动一套全新的高位相机陶瓷杯场景，再在另一个已加载工作空间的仓库终端运行：

```bash
.venv-smolvla/bin/python -I tools/run_released_cup_policy.py \
  --release "$WS/observations/autonomous/cup_release.json"
```

入口会核对清单状态、权重和运行代码哈希，复用已验收参数，自动创建新的观测输出目录。
它会发送运动命令；不会自动启动第二套仿真。源码或权重改变后旧清单不再直接适用。
维护环境已有的 `observations/delivery/.../release_manifest.json` 也采用同一入口，
但其绝对路径、数据和权重只存在于原机器，不能从 Git 克隆得到。

## 排错与验证

- `cat /task_metadata.json`：通常是当前终端没有设置 `EPISODE_DIR`。
- `.venv-.../bin/python: No such file`：检查是否在仓库目录，ML 命令不在工作空间根目录执行。
- 输出目录存在：核对是否是已完成或部分失败结果，用新目录重试。
- 相机漏帧：用 `camera_check --timing-only --seconds 30 --label baseline`，
  再加 `--reliability reliable` 比较；原生对照程序见工具索引。订阅端计时不能独自区分源漏帧与传输丢帧。
- 反馈时序问题：检查 `/clock`、实际关节/里程计及结果中的 timing 计数；并行世界、录屏、GUI 负载均需实测。
- MoveIt 错误或碰撞：保留 `ros_session/result_*.json`、实际反馈与 `simulation.log`，
  区分观测过期、执行拒绝、夹爪目标过度闭合和真实碰撞，不能把所有中断都当作模型能力失败或一概放宽容限。

全部离线测试的环境分组命令见 [tools 测试说明](../tools/README.md#运行离线测试)。
自动采集、完整闭环和 GUI 测试都不会作为离线单元测试默认执行。
