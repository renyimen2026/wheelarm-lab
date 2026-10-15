# 参数与操作参考

[返回项目首页](../README.md) · [Python CUDA 服务](../model_service/README.md)

首次使用请先完成首页的建图导航流程。本页命令在工作空间根目录执行，
并已加载 ROS 与 `install/setup.bash`；完整选项以各入口的 `--help` 为准。

## 场景与启动入口

基础仿真 `sim.launch.py`、在线建图 `nav.launch.py`、固定地图定位
`nav_localization.launch.py` 三选一，不同时启动同一套仿真。

| 参数 | 默认值或作用 |
| --- | --- |
| `world` | 住宅场景 `indoor_house.sdf` |
| `spawn_x/y/z/yaw` | Gazebo 出生位姿 `3.5 / 1.0 / 0.05 / 0.0`，米和弧度 |
| `use_ekf` | `true`；`false` 切换原始轮速里程计配置 |
| `rviz` | 导航入口默认 `true`，自动加载 RViz 配置 |
| `localization_debug` | 默认 `true`，可关闭定位诊断日志 |

```bash
ros2 launch wheel_arm_gazebo nav.launch.py rviz:=false localization_debug:=false

ros2 launch wheel_arm_gazebo nav.launch.py \
  world:=src/wheelarm-lab/wheel_arm_gazebo/worlds/indoor_lab.sdf \
  spawn_x:=0 spawn_y:=0 spawn_z:=0
```

相对世界文件路径从工作空间根目录解析。换场景后应重新建图，住宅资产没有附带 VLN 任务或语义标注。

## 定位与启动位姿

默认 TF 关系：

```text
map → odom → base_footprint → 机器人关节与传感器
```

在线建图时 SLAM 提供 `map → odom`，固定地图模式由 AMCL 提供。
EKF 融合 `/wheel_arm/diff_drive_base_controller/odom` 与 `/imu`，
输出 `/odometry/filtered` 和 `odom → base_footprint`。
关闭 EKF 时使用底盘轮速里程计及其 TF，避免同时有两个发布者。

```bash
ros2 launch wheel_arm_gazebo nav_localization.launch.py \
  map:=maps/wheel_arm_map.yaml use_ekf:=false
```

AMCL 先验 YAML 只包含四个字段，以下仅示例，不是住宅出生点的地图坐标：

```yaml
frame_id: map
x: 1.2
y: -0.8
yaw: 0.0
```

不用文件时也可直接传入坐标：

```bash
ros2 launch wheel_arm_gazebo nav_localization.launch.py \
  map:=maps/wheel_arm_map.yaml \
  initial_pose_x:=1.2 initial_pose_y:=-0.8 initial_pose_yaw:=0.0
```

文件与命令行坐标互斥；三个坐标必须完整、有限，文件坐标系必须为 `map`。
不传先验时用 RViz 手动定位。程序不持续发布先验，不覆盖后续手动修正。
正确手动定位后可读取 `/amcl_pose`，取 `pose.pose.position.x/y`；
平面四元数朝向为 `yaw = 2 * atan2(z, w)`：

```bash
ros2 topic echo /amcl_pose --once
```

先验与地图、出生位置绑定，不自动恢复上次停止位姿，也不自动检测定位失败或全局重定位。
TF 存在或协方差较小不能单独证明定位正确，仍应检查激光与地图是否重合。

## 控制与遥控参数

| 遥控参数 | 默认值 |
| --- | --- |
| `linear_speed` / `angular_speed` | `0.5 m/s` / `1.0 rad/s` |
| `publish_rate` | `50 Hz`，按键变化时还会立即发送指令 |
| `arm_repeat_rate` | `5 Hz`，机械臂按住重复频率 |

机械臂每次按键默认改变关节 `0.05 rad`。同轴反向按键相互抵消，
空格停止后需先松开移动键再重新按下。需要 Tk 窗口，不支持纯 SSH 字符遥控。
旧参数 `command_hold_time` 已移除。

底盘线加速度限制 `±2.0 m/s²`、角加速度限制 `±3.0 rad/s²`，
指令断流 `0.2 s` 超时停止。这些限制也影响自主导航；松键不等于物理瞬停。
手动发布命令前退出其他控制程序，不与 Nav2 并行控制底盘：

```bash
ros2 topic pub /wheel_arm/diff_drive_base_controller/cmd_vel_unstamped \
  geometry_msgs/msg/Twist "{linear: {x: 0.15}, angular: {z: 0.3}}"

ros2 topic pub --once /wheel_arm/arm_controller/joint_trajectory \
  trajectory_msgs/msg/JointTrajectory \
  "{joint_names: [shoulder_pan_joint, shoulder_lift_joint, elbow_joint, wrist_pitch_joint], points: [{positions: [0.0, 0.4, -0.8, 0.3], time_from_start: {sec: 2}}]}"
```

Nav2 速度链路为 `controller_server → /cmd_vel_nav → velocity_smoother →
/wheel_arm/diff_drive_base_controller/cmd_vel_unstamped`，恢复行为可直接发送底盘指令。
不要将平滑器输入输出合并到同一话题，否则可能形成反馈循环。

## 目标点与任务配置

`semantic_goal` 支持 `save`、`set`、`list`、`go`；`--file` 可指定目标点 YAML。
仿真中的 `save`、`go` 需要 `--ros-args -p use_sim_time:=true`。
手工输入的是地图坐标，不是 Gazebo 世界坐标：

```bash
ros2 run wheel_arm_navigation semantic_goal set 客厅 --x 1.0 --y 2.0 --yaw 0.0
ros2 run wheel_arm_navigation semantic_goal --file maps/semantic_goals.yaml list
```

对应格式如下，示例坐标应替换为自己地图中的可通行位置：

```yaml
frame_id: map
goals:
  客厅:
    x: 1.0
    y: 2.0
    yaw: 0.0
```

同一目标文件不要由多个终端同时写入。导航默认等待服务 10 秒、单目标 180 秒，
可用 `semantic_goal go 客厅 --timeout 300` 调整目标等待时间。

| `mission_runner` 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `--file` | `maps/semantic_goals.yaml` | 目标点文件 |
| `--stay` | `0` | 每个成功目标后的统一停留秒数，包含最后一个 |
| `--wait` / `--timeout` | `10` / `180` | 服务及接受等待 / 单目标完成等待秒数 |
| `--repeat` | `1` | 整条路线轮数 |
| `--on-failure` | `stop` | `skip` 仅跳过已确认结束的失败目标 |
| `--name` / `--log-dir` | `navigation_task` / `mission_logs` | 任务名称 / 日志目录 |

开始前检查所有名称，执行时使用坐标快照。通信异常、Nav2 不可用或终态未确认时，
无论失败策略如何都停止。超时和停留按实际经过的时间计量，不随仿真暂停停止计时。
仓库提供[巡检配置示例](../wheel_arm_navigation/config/patrol.example.yaml)：

```bash
ros2 run wheel_arm_navigation mission_runner \
  --task src/wheelarm-lab/wheel_arm_navigation/config/patrol.example.yaml \
  --ros-args -p use_sim_time:=true
```

示例使用“客厅”“厨房入口”“卧室”，需先记录这些精确名称或自行修改路线。
`--task` 不能与位置参数目标列表混用，`--stay` 等参数可覆盖 YAML。
顺序任务状态包括 `completed`、`completed_with_failures`、`failed`、
`canceled`、`unsafe_stop`；部分成功仍返回非零退出码。

## 导航观察参数

`language_goal` 是纯导航，可访问多个地点；`language_observe` 仅支持单地点观察。
二者均要求精确名称、只使用本机模型，并在执行前人工确认。
支持自然语言表达，不是固定句式匹配；模型仍可能误解否定或附加条件。

| `language_observe` 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `--file` | `maps/semantic_goals.yaml` | 目标点文件 |
| `--model` / `--vision-model` | `qwen3:4b` / `qwen3-vl:4b-instruct` | 语言 / 视觉模型 |
| `--endpoint` | `http://127.0.0.1:11434` | CUDA 服务需显式改为 `11435` |
| `--llm-timeout` / `--vision-timeout` | `180` / `600` | 解析 / 视觉预算秒数，最大 600 |
| `--nav-wait` / `--nav-timeout` | `10` / `180` | Nav2 等待 / 目标完成等待秒数 |
| `--capture-wait` | `15` | 停稳、拍摄和定位检查总等待秒数 |
| `--settle-duration` | `0.5` | 连续低速里程计时长，按 ROS 时间 |
| `--rgb-topic` / `--max-age` | `/camera/image` / `0.5` | RGB 话题 / 图像最大年龄秒数 |
| `--odom-topic` | `/odometry/filtered` | 停稳检查里程计 |
| `--no-vision-preload` | 关闭 | 禁用导航期间可选预加载 |
| `--log-dir` / `--observations-dir` | `mission_logs` / `observations` | 日志 / 观测目录 |

关闭 EKF 时需同步切换停稳检查话题：

```bash
ros2 run wheel_arm_navigation language_observe "去厨房看看有没有椅子" \
  --endpoint http://127.0.0.1:11435 \
  --odom-topic /wheel_arm/diff_drive_base_controller/odom
```

停稳检查要求线速度模长不超过 `0.02 m/s`、角速度模长不超过 `0.04 rad/s`；
里程计 ROS 时间和接收时间年龄均不超过 `0.5 s`，子坐标系为 `base_footprint`。
拍摄时刻车体定位需距目标不超过 `0.35 m`、朝向误差不超过 `0.35 rad`，
且光学相机 TF 可用。这不是 ground truth 静止或定位的独立验证。

预检查失败不发送目标，导航失败或取消不采集，视觉失败仍保留已拍图片。
日志状态包括 `completed`、`preflight_failed`、`failed`、`unsafe_stop`、
`capture_failed`、`vision_failed`、`canceled`；预加载状态单独记录为 `vision_preload`。
退出客户端不能可靠中止已运行的服务端 CUDA 加载，但不会因此发送新的机器人动作。

## 相机观测与离线重放

RGB-D 相机相对 `base_link` 的位置为 `(0.34, 0, 0.14) m`，朝车体前方。
彩色和深度均为 `640×480`、`15 Hz`，水平视场约 60°，裁剪范围 `0.1～8 m`。

| 话题 | 类型与含义 |
| --- | --- |
| `/camera/image` | `sensor_msgs/msg/Image`，RGB |
| `/camera/depth_image` | `sensor_msgs/msg/Image`，`32FC1` 米单位深度 |
| `/camera/camera_info` | `sensor_msgs/msg/CameraInfo`，相机内参 |

光学坐标系 `camera_optical_frame` 的轴向为右、下、前；当前未桥接 RGB-D 点云。
旧内参话题 `/camera/image/camera_info` 不再使用。

`scene_snapshot` 默认参考系 `map`、等待 15 秒、同步容差 `0.02 s`、最大年龄 `0.5 s`。
只接受请求后新数据，使用 RGB 拍摄时刻 TF，不用最新 TF 替代。
同步和内参假设适用于当前对齐的 RGB-D 源，不会自动标定任意真实相机。

```text
observations/<观测ID>/
├── rgb.png
├── depth.npy           # RGB-D 采集才有，float32 米单位
├── depth_preview.png   # RGB-D 采集才有，仅作可视化
├── metadata.json       # 时间、尺寸；同步观测还含内参和 TF
└── descriptions/
    └── vision-<唯一ID>.json
```

单帧问答不伪造深度或内参。深度表示光学 z 方向距离，不是斜射线欧氏距离；
计算时使用 `depth.npy` 与内参，不用预览颜色比较不同帧的距离。
无有效深度时仍保留 RGB，并将深度有效比例记为 0。

RViz 添加 `Image` 可看 RGB；深度显示需开启范围归一化。
实时问答只依赖 RGB 和正确时钟，不依赖地图或深度。
真实相机使用系统时间戳时，追加 `--ros-args -p use_sim_time:=false`。

离线重放时，把占位内容替换为实际输出的**单个观测目录**，不是整个根目录：

```bash
ros2 run wheel_arm_navigation scene_describe 'observations/<观测ID>' \
  --question "画面里有哪些家具？哪些地方被遮挡了？" \
  --endpoint http://127.0.0.1:11435 --timeout 600
```

离线模式不初始化 ROS 节点，无需仿真。报告包含 `summary`、`objects`、
`uncertainties`，并记录图片 SHA-256、模型、问题、时间与耗时。
`image_region` 是画面左右，不是地图方位；额外导航和坐标字段会被拒绝。
每次生成新报告，不覆盖原图、深度或旧报告。

## 语言解析评测

```bash
ros2 run wheel_arm_navigation llm_eval \
  --endpoint http://127.0.0.1:11435 --repeat 3

ros2 run wheel_arm_navigation llm_eval \
  --endpoint http://127.0.0.1:11435 --category clarify --category unsupported
```

`--suite` 可指定 YAML，`--repeat` 为评测轮数（1～10），与导航循环不同；
`--output` 可指定尚不存在的报告路径。默认报告位于 `evaluation_logs/`。
自定义测试集结构见[随包用例](../wheel_arm_navigation/config/llm_eval.yaml)。

严格比较 `status`、目标顺序及重复次数、`stay`、`repeat`，不比较说明措辞。
请求和校验错误计入正确率分母，格式合法不代表语义正确。
报告保留测试集、解析器指纹和逐条结果，支持中断后追溯。
退出码：全部通过 `0`，不匹配或错误 `1`，中断 `130`。

客户端请求 `keep_alive=0`，CUDA 驻留模式会保留模型，Ollama 按其服务策略处理。
比较耗时需区分加载、语法编译和生成，并固定软硬件与权重。
不要将小规模回归成绩当成独立泛化能力或真实导航成功率。

## 常见问题

| 现象 | 优先检查 |
| --- | --- |
| 地图保存无法写入 | 先 `mkdir -p maps`，确认执行目录和权限 |
| RViz 没有机器人，提示缺少 `map` | 地图加载、AMCL 初始位姿、定位 TF |
| RViz 消息队列已满 | 对应时刻 TF、坐标系、`use_sim_time`、`/clock`，不要仅增大队列 |
| `/scan` 有消息但无扫描点 | 有限测距、雷达高度与障碍物、渲染、`lidar_link` TF |
| 旋转时扫描偏离地图 | 定位、轮子打滑、时间同步；地图刷新不能代替定位修复 |
| 到达后轮子仍转 | 其他速度发布者、平滑器输入输出是否合并、是否仍运行旧进程 |
| 观察任务一直等停稳 | 里程计话题、子坐标系、速度和时间戳，关闭 EKF 后是否改话题 |
| CUDA 服务启动失败 | Python 环境、实际 CUDA 运算、模型文件、显存和服务日志 |
| 模型连接失败 | `11435` 与 `11434` 是否选对，是否完成启动和预加载 |
| 模型忙或加载影响仿真 | 不并发启动任务，可用 `--no-vision-preload`，检查资源占用 |

仿真节点应使用同一套时钟。碰撞或打滑时，轮速里程计和 EKF 仍可能漂移，
当前没有用真实模型位姿替代定位的保证。

## 运行测试

常规包测试见首页。可选集成测试会启动模拟 action、TF 和相机，
必须选择未被其他机器人使用的 ROS 域，下面的编号仅为示例：

```bash
ROS_DOMAIN_ID=87 ROS_LOCALHOST_ONLY=1 WHEELARM_ROS_TEST=1 \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest \
  src/wheelarm-lab/wheel_arm_navigation/test \
  src/wheelarm-lab/wheel_arm_gazebo/test -q

ROS_DOMAIN_ID=88 ROS_LOCALHOST_ONLY=1 WHEELARM_ROS_TEST=1 WHEELARM_GUI_TEST=1 \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest \
  src/wheelarm-lab/wheel_arm_teleop/test -q
```

窗口测试需要桌面，会短暂打开窗口并注入按键。Gazebo 包的定位测试使用真实 AMCL
与合成地图/激光，不启动 Gazebo；不证明真实可达性、定位精度或共用 GPU 性能。
模型的协议、分词器和 CUDA 测试见[模型服务测试](../model_service/README.md#测试)。
