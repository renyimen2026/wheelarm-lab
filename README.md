# WheelArm Lab

面向 ROS 2 的开源轮臂机器人仿真与具身智能实验平台。

基于 **Ubuntu 22.04、ROS 2 Humble、Gazebo Fortress**，提供差速底盘、4 自由度
机械臂、室内场景，以及从遥控、建图导航到本地语言指令和视觉问答的完整实验流程。
项目持续维护，后续面向视觉语言导航（VLN）和视觉语言动作（VLA）扩展。

当前语言导航基于**已记录的命名目标点**，视觉功能基于**单帧 RGB 问答**；
尚不具备完整 VLN、自主物体搜索或语言驱动抓取能力。

[快速开始](#快速开始) · [建图与导航](#建图与导航) · [语言与视觉任务](#语言与视觉任务) ·
[参数与排错](docs/reference.md) · [模型服务](model_service/README.md) · [更新记录](CHANGELOG.md)

## 能做什么

| 模块 | 当前能力 |
| --- | --- |
| 机器人与控制 | 差速底盘、4 自由度机械臂、`ros2_control`、键盘组合键遥控 |
| 感知与状态估计 | 2D LiDAR、IMU、RGB-D 相机、轮速里程计与 IMU 的 EKF 融合 |
| 建图与导航 | `slam_toolbox` 建图与回环、AMCL 固定地图定位、Nav2 规划与避障 |
| 任务执行 | 命名目标点、顺序多点任务、任务配置、取消处理与执行日志 |
| 自然语言 | 本地 LLM 解析导航或单点观察任务，预览、人工确认与回归评测 |
| 视觉观察 | 实时单帧问答、离线重放、导航后停稳拍摄、同步 RGB-D 观测保存 |
| 模型后端 | 独立 PyTorch CUDA 服务或 Ollama；CUDA 后端支持导航途中加载视觉模型 |

默认场景为带家具和贴图的住宅，基于 AWS RoboMaker Small House World。
资产随仓库提供，运行仿真时无需在线下载；来源与许可证见
[场景资产说明](wheel_arm_gazebo/models/README.md)。

## 快速开始

### 1. 准备环境

先安装 Ubuntu 22.04 和 ROS 2 Humble，再安装仿真及导航依赖：

```bash
sudo apt update
sudo apt install \
  python3-colcon-common-extensions python3-rosdep python3-tk \
  ros-humble-ros-gz ros-humble-gz-ros2-control ros-humble-ros2-controllers \
  ros-humble-navigation2 ros-humble-nav2-bringup ros-humble-slam-toolbox \
  ros-humble-robot-localization
```

基础仿真和导航不需要语言模型、GPU 推理环境或云端 API 密钥。
Gazebo、RViz 和键盘窗口需要桌面显示环境。

### 2. 获取并构建

```bash
mkdir -p ~/wheelarm_ws/src
cd ~/wheelarm_ws/src
git clone https://github.com/renyimen2026/wheelarm-lab.git
cd ..

source /opt/ros/humble/setup.bash
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash
```

如果系统尚未初始化 rosdep，先执行一次 `sudo rosdep init` 和 `rosdep update`。

**目录约定：**后续 ROS 命令均在工作空间根目录 `wheelarm_ws/` 执行；
每个新 ROS 终端都需要加载 ROS 和 `install/setup.bash`。
模型服务命令在仓库目录 `src/wheelarm-lab/` 执行，使用独立 conda 环境，
不要将模型依赖安装到 ROS Python 环境。

### 3. 启动仿真

选择一个入口运行，不要重复启动同一套仿真：

| 入口 | 命令 | 用途 |
| --- | --- | --- |
| 基础仿真 | `ros2 launch wheel_arm_gazebo sim.launch.py` | 模型、传感器与控制器实验 |
| 在线建图导航 | `ros2 launch wheel_arm_gazebo nav.launch.py` | Gazebo、SLAM、Nav2、RViz |
| 固定地图导航 | `ros2 launch wheel_arm_gazebo nav_localization.launch.py map:=maps/wheel_arm_map.yaml` | 使用自己保存的地图和 AMCL 定位 |

第一次使用建议从在线建图开始。**仓库不包含已经建好的地图和预置目标点**，
需要先建图，再使用固定地图和命名点任务。

## 建图与导航

### 遥控探索并保存地图

```bash
ros2 launch wheel_arm_gazebo nav.launch.py
```

另开 ROS 终端启动键盘窗口，点击窗口获取焦点后操作：

```bash
ros2 run wheel_arm_teleop keyboard_teleop
```

| 按键 | 操作 |
| --- | --- |
| `W` / `S` | 前进 / 后退 |
| `A` / `D` | 左转 / 右转，可与前后移动组合 |
| `Space` | 停止底盘 |
| `Q` / `E` | 肩部水平关节正向 / 反向 |
| `R` / `F` | 肩部俯仰关节正向 / 反向 |
| `T` / `G` | 肘关节正向 / 反向 |
| `Y` / `H` | 腕部俯仰关节正向 / 反向 |
| `X` | 机械臂回零 |
| `Esc` / `Ctrl+C` / 关闭窗口 | 停止底盘并退出 |

按住移动、松键发送停止指令，失焦时停止底盘。车体仍受惯性和加减速限制影响，
不会瞬时静止。默认线速度 `0.5 m/s`、角速度 `1.0 rad/s`；初次建图可用下面的
低速命令替代上面的遥控命令，不要同时运行两个遥控窗口：

```bash
ros2 run wheel_arm_teleop keyboard_teleop --ros-args \
  -p linear_speed:=0.15 -p angular_speed:=0.45
```

探索完成后，在另一个终端保存地图：

```bash
mkdir -p maps
ros2 run nav2_map_server map_saver_cli -f maps/wheel_arm_map
```

确认生成 `maps/wheel_arm_map.yaml` 和 `maps/wheel_arm_map.pgm`，再关闭建图启动进程。

### 加载地图并自主导航

```bash
ros2 launch wheel_arm_gazebo nav_localization.launch.py \
  map:=maps/wheel_arm_map.yaml
```

在 RViz 使用 **2D Pose Estimate** 设置当前位置和朝向，确认激光与墙体基本重合，
再通过目标点工具发送导航目标。该模式不启动 SLAM，不修改保存的地图；
局部和全局代价地图仍根据激光观测更新障碍物。

**自主导航前必须退出键盘遥控和其他速度发布程序，不要只切换窗口。**
遥控窗口失焦后仍会持续发送零速度，多个程序同时控制底盘会互相干扰。

### 使用启动位姿先验

固定地图和出生位置不变时，可保存一份已经验证的地图坐标系位姿：

```bash
cp src/wheelarm-lab/wheel_arm_gazebo/config/initial_pose.example.yaml maps/initial_pose.yaml
```

编辑该文件中的 `x`、`y`、`yaw` 后启动：

```bash
ros2 launch wheel_arm_gazebo nav_localization.launch.py \
  map:=maps/wheel_arm_map.yaml initial_pose_file:=maps/initial_pose.yaml
```

示例坐标不能直接使用，Gazebo 世界坐标也不一定等于地图坐标。
先验只是 AMCL 初始化，不是自动重定位；定位不正确时，停止任务后仍可在 RViz 手动修正。
坐标获取、命令行先验及文件格式见[定位参数](docs/reference.md#定位与启动位姿)。

## 命名点与顺序任务

先启动固定地图导航、完成定位，再将机器人移动到希望保存的位置并停稳。
例如到达厨房后，记录当前位置和朝向：

```bash
ros2 run wheel_arm_navigation semantic_goal save 厨房 --ros-args -p use_sim_time:=true
```

用同样方式分别记录“卧室”“客厅”等位置，不要在同一位置连续保存多个房间名称。
默认文件为 `maps/semantic_goals.yaml`。名称必须精确匹配，不自动处理别名；
重复名称默认拒绝覆盖，可添加 `--overwrite` 更新。

```bash
# 查看所有名称和坐标
ros2 run wheel_arm_navigation semantic_goal list

# 导航到一个已记录地点
ros2 run wheel_arm_navigation semantic_goal go 厨房 --ros-args -p use_sim_time:=true

# 按顺序访问多个已记录地点
ros2 run wheel_arm_navigation mission_runner 厨房 卧室 客厅 \
  --ros-args -p use_sim_time:=true
```

目标点与地图绑定，换场景或重新建图后应重新核对。默认导航失败即停止，
`Ctrl+C` 请求取消；取消终态未确认时，必须人工检查机器人是否停止。
任务配置、循环、停留及失败策略见[任务参数](docs/reference.md#目标点与任务配置)。

## 语言与视觉任务

### 选择模型服务

机器人与模型服务分开运行，通过本机 HTTP 接口调用模型。选择一种后端即可：

| 后端 | 接口地址 | 准备方式 |
| --- | --- | --- |
| Python CUDA 服务 | `http://127.0.0.1:11435` | 独立 PyTorch CUDA 环境、Transformers 权重；见[安装与启动](model_service/README.md) |
| Ollama | `http://127.0.0.1:11434` | 安装本机 Ollama，下载下列模型并保持服务运行 |

Ollama 模型准备：

```bash
ollama pull qwen3:4b
ollama pull qwen3-vl:4b-instruct
```

使用 Ollama 时，应在服务环境设置 `OLLAMA_NO_CLOUD=1` 并使用已下载的本地模型，
不要使用远程模型别名；本机 HTTP 地址本身不保证模型在本机运行。

Python CUDA 服务准备完成后，在**仓库目录、模型 conda 环境**中启动：

```bash
env -u PYTHONPATH PYTHONNOUSERSITE=1 \
  CUDA_VISIBLE_DEVICES=0 MKL_THREADING_LAYER=GNU \
  python -m model_service.server
```

默认启动预加载语言模型。保持此终端运行，另外使用 ROS 终端操作机器人。
Transformers 权重默认保存在 `~/Models/transformers`，不能直接复用 Ollama 的 GGUF 文件。

**下面示例统一使用 Python CUDA 服务。** CLI 默认地址仍是 Ollama `11434`，
使用 CUDA 服务必须显式指定 `--endpoint http://127.0.0.1:11435`；
使用 Ollama 时去掉此参数即可。接口仅允许本机访问，不需要云端密钥。
GPU 是否可用应通过后端日志验证，不能只根据某一个环境变量判断。

### 用自然语言导航

前提是指令中的名称都已记录。先预览，不需要启动仿真或 Nav2：

```bash
ros2 run wheel_arm_navigation language_goal "先去厨房，再去卧室，最后去客厅" \
  --endpoint http://127.0.0.1:11435 --preview
```

确认解析正确后，启动固定地图导航、完成定位并退出遥控，再去掉 `--preview`：

```bash
ros2 run wheel_arm_navigation language_goal "先去厨房，再去卧室，最后去客厅" \
  --endpoint http://127.0.0.1:11435 --ros-args -p use_sim_time:=true
```

程序显示路线和坐标，核对后输入 `y` 才执行。表达不需要固定句式，但能力有边界：
最多 10 个顺序目标、统一停留 0～60 秒、整条路线循环 1～3 次。
未知地点或含糊指代应澄清，观察和抓取等非纯导航需求应拒绝。
模型只选择名称，坐标从文件读取；格式校验不能保证理解正确，人工确认不能跳过。

### 识别当前画面

先启动仿真和模型服务，无需建图或 AMCL 定位，也不必提前保存图片：

```bash
ros2 run wheel_arm_navigation scene_describe --question "画面里有哪些家具？" \
  --endpoint http://127.0.0.1:11435 --timeout 600
```

连接相机并获得有效时钟后，冻结请求时间之后的第一帧新鲜 RGB，仅推理一次。
不是持续视频识别，也不能精确回取按下 Enter 的历史帧。
图片保存到 `observations/<观测ID>/rgb.png`，结果位于该目录的 `descriptions/`。
推理失败仍保留图片，支持[离线重试](docs/reference.md#相机观测与离线重放)。

### 到指定地点后观察

前提是固定地图导航、定位、目标点和单独的相机问答均已可用。
记录目标点时要选好朝向，到达后不会自动转动寻找物体。

```bash
# 预览不会导航、拍照或调用视觉模型
ros2 run wheel_arm_navigation language_observe "去厨房看看有没有椅子" \
  --endpoint http://127.0.0.1:11435 --preview

# 核对后执行，仍需输入 y 确认
ros2 run wheel_arm_navigation language_observe "去厨房看看有没有椅子" \
  --endpoint http://127.0.0.1:11435
```

```text
解析与确认 → Nav2 接受目标 → 导航移动 → 停稳检查 → 拍摄新帧 → 视觉问答
                            └─ 后台加载视觉模型 ──────────────────┘
```

CUDA 服务一次只常驻一个模型，利用导航时间加载视觉模型；提前到达时需等待剩余加载。
预加载失败不取消导航，到达后尝试正常按需推理；可加 `--no-vision-preload` 关闭优化。
Ollama 保持原有流程，不使用此预加载扩展。

只有导航成功、连续低速里程计有效、拍摄时刻位姿接近目标且相机 TF 可用，才继续观察。
导航失败、取消、终态未知或采集检查失败时，不把当前位置图片当成目标地点画面。
参数、原始里程计模式及失败状态见[观察任务参数](docs/reference.md#导航观察参数)。

当前只支持**一个地点、一次单帧观察**，不支持多房间观察、条件分支、自主搜索或抓取。
“没看到椅子”不等于“厨房没有椅子”，任务完成也不等于识别结果已被独立验证。

## 观测数据与评测

### 保存 RGB-D 观测

`scene_snapshot` 保存同步 RGB、深度、内参和拍摄时刻 TF，不调用模型或移动机器人。
默认需要有效的地图定位；没有 `map` 时显式选择 `odom`：

```bash
ros2 run wheel_arm_navigation scene_snapshot --label "厨房观测" \
  --ros-args -p use_sim_time:=true

# 仅基础仿真、还没有地图定位时使用
ros2 run wheel_arm_navigation scene_snapshot --reference-frame odom \
  --ros-args -p use_sim_time:=true
```

同步采集和单帧问答是不同入口。模型目前只读取 RGB，不使用深度或 TF 推断物体坐标。
相机话题、文件结构和深度含义见[相机参考](docs/reference.md#相机观测与离线重放)。

### 评测语言解析

`llm_eval` 不启动导航、不读取实际地图坐标，可在没有仿真时运行：

```bash
ros2 run wheel_arm_navigation llm_eval --endpoint http://127.0.0.1:11435
```

默认使用 [24 条中文回归用例](wheel_arm_navigation/config/llm_eval.yaml)，
统计任务匹配、格式合法率、误接受率、分类结果及耗时。
这是解析器回归测试，不是独立泛化测试或 VLN 成功率评测；比较时应保持模型、
测试集、解析器及冷/热加载条件一致。更多选项见[评测参考](docs/reference.md#语言解析评测)。

### 数据保存位置

以下目录均相对于执行 ROS 命令的工作空间根目录，默认不提交到 Git：

| 目录 | 内容 |
| --- | --- |
| `maps/` | 地图 YAML/PGM、命名目标点、启动位姿先验 |
| `mission_logs/` | 执行日志、目标快照、预加载状态和失败阶段 |
| `observations/` | RGB 或 RGB-D 观测、时间元数据、视觉报告 |
| `evaluation_logs/` | LLM 解析回归评测报告 |

日志可能包含原始指令、图片和位置，公开数据前请自行检查。

## 项目结构与配置

```text
wheelarm-lab/
├── wheel_arm_description/  # xacro/URDF、传感器、Gazebo 插件
├── wheel_arm_control/      # 底盘和机械臂 ros2_control 配置
├── wheel_arm_gazebo/       # 场景、launch、EKF、SLAM/Nav2、RViz
├── wheel_arm_teleop/       # Tk 键盘遥控
├── wheel_arm_navigation/   # 命名目标、任务执行、语言解析与视觉观察
├── model_service/          # 独立 Python CUDA 服务，不参与 colcon 构建
├── docs/                   # 参数、数据格式、排错与测试参考
└── CHANGELOG.md            # 功能和配置变更历史
```

| 想调整的内容 | 位置 |
| --- | --- |
| 尺寸、惯量、轮子接触与传感器 | [wheel_arm.urdf.xacro](wheel_arm_description/urdf/wheel_arm.urdf.xacro) |
| 底盘和机械臂控制器 | [controllers.yaml](wheel_arm_control/config/controllers.yaml) |
| EKF 融合 | [ekf.yaml](wheel_arm_gazebo/config/ekf.yaml) |
| SLAM、AMCL、Nav2 | [nav2_params.yaml](wheel_arm_gazebo/config/nav2_params.yaml) |
| 无 EKF 时的对应配置 | [controllers_raw_odom.yaml](wheel_arm_control/config/controllers_raw_odom.yaml)、[nav2_params_raw_odom.yaml](wheel_arm_gazebo/config/nav2_params_raw_odom.yaml) |
| 定位先验示例 | [initial_pose.example.yaml](wheel_arm_gazebo/config/initial_pose.example.yaml) |
| 顺序任务示例 | [patrol.example.yaml](wheel_arm_navigation/config/patrol.example.yaml) |

场景、出生位置、里程计模式和手动控制命令见[参数与操作参考](docs/reference.md)。
更新后重新运行 `rosdep install`、`colcon build --symlink-install` 并加载工作空间；
模型、传感器、控制器或 launch 配置改变后，应关闭旧进程并重新启动相应组件。

## 测试与限制

ROS 包测试在工作空间根目录执行：

```bash
colcon test --packages-select wheel_arm_gazebo wheel_arm_navigation wheel_arm_teleop \
  --event-handlers console_direct+
colcon test-result --verbose
```

默认跳过需要 DDS 或窗口的可选测试。隔离 ROS、键盘窗口及模型测试见
[测试参考](docs/reference.md#运行测试)和[模型服务文档](model_service/README.md#测试)。

- 仿真定位可能漂移，轮子打滑、碰撞和时间同步都可能影响激光与地图对齐。
- 在线 SLAM 可通过回环修正地图，固定地图 AMCL 模式不会修改保存地图。
- 相机安装较低、视野有限；理想仿真深度不等于真实相机噪声模型。
- 模型会误解析、遗漏或产生幻觉，格式校验不能替代真实效果评估。
- 模型与 Gazebo 共享资源，完整仿真并行性能需在使用者硬件上验证。
- 没有跨进程底盘控制权仲裁，不适合未经验证直接迁移到实机或无人值守运行。

## 后续方向

- 物体检测与基于深度的空间定位。
- 任务级移动操作与机械臂规划。
- VLN 场景、语义标注、数据采集与评测接口。
- VLA 策略接入。

## 贡献与许可证

欢迎提交 Issue 和 Pull Request。请附上环境、启动命令、复现步骤及验证结果；
功能或配置变化同步记录到 [CHANGELOG.md](CHANGELOG.md)，使用方式变化更新相关文档。

维护者：`renyimen` · 邮箱：`renyimen2026@foxmail.com`。

项目代码采用 [Apache License 2.0](LICENSE)。第三方住宅模型与贴图遵循
[AWS 原始许可证](wheel_arm_gazebo/models/AWS_LICENSE)。
