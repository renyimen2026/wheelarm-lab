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

## 机械臂规划参数

`wheel_arm_moveit_config` 只规划机械臂，不规划移动底盘，不依赖地图、LLM 或视觉服务。
MoveIt 与 Gazebo 使用同一份 xacro，不启动第二个 `controller_manager`、关节状态发布器
或 `robot_state_publisher`；不发布额外的 `map/odom` TF。

| 启动入口 | 行为 |
| --- | --- |
| `ros2 launch wheel_arm_moveit_config sim.launch.py` | 启动简单场景、控制器、MoveIt 与 RViz |
| `ros2 launch wheel_arm_moveit_config moveit.launch.py` | 仅启动 MoveIt 与 RViz，连接已有仿真 |

上述两个入口不要同时启动。仅连接已有仿真时，应停止导航和遥控，不需关闭仿真本身。

| 参数 | 默认值 | 范围 |
| --- | --- | --- |
| `rviz` | `true` | 两个入口，是否启动专用 RViz |
| `allow_trajectory_execution` | `true` | 两个入口；设为 `false` 时只允许规划，拒绝执行 |
| `plugin_lifetime_workaround` | `true` | 两个入口；Humble MoveIt 插件卸载时序兼容，见下文 |
| `gui` | `true` | 仿真入口，是否打开 Gazebo 窗口 |
| `camera_mount` | `low` | 两个入口；`low` 原低位相机，`mast` 高位相机，连接已有仿真时必须匹配 |
| `world` | `empty.sdf` | 仿真入口，测试场景；出生位置为 `0 / 0 / 0.05 / 0` |
| `use_sim_time` | `true` | 仅连接入口；Gazebo 测试入口固定为 `true` |
| `joint_states_topic` | `/wheel_arm/joint_states` | 仅连接入口，可改为实际的关节反馈话题 |

```bash
# 仅关闭 Gazebo 图形窗口，服务端仍模拟相机和激光，不等于完全不使用 GPU
ros2 launch wheel_arm_moveit_config sim.launch.py gui:=false

# 已有仿真，只做规划预览，不允许轨迹执行
ros2 launch wheel_arm_moveit_config moveit.launch.py allow_trajectory_execution:=false
```

基础 `wheel_arm_gazebo sim.launch.py` 也支持 `gui:=false`，默认行为不变。

### 模型与算法

- 规划组 `arm` 是 `base_link → tool0` 串联链，包含四个转动关节，轮子标记为被动关节。
- 规划坐标系为 URDF 根 `base_footprint`，假定底盘静止；没有移动底盘虚拟关节。
- OMPL 使用 `RRTConnect` 做关节空间规划，TOTG 生成时间参数。
- KDL 配置 `position_only_ik: true`；可解位置不代表能满足任意朝向，RViz 任意六维拖拽仍可能失败。
- 规划最大关节速度为 `0.6 rad/s`、最大加速度为 `0.8 rad/s²`，默认缩放均为 `0.2`。
  RViz 或请求可覆盖缩放；URDF 位置、速度、力矩上限没有修改。
- SRDF 排除相邻连杆、刚性安装重叠及双指之间的设计接触，不排除非相邻机械臂与底盘的碰撞。
- `gripper` 规划组包含两个独立平移关节，连接 `/wheel_arm/gripper_controller/follow_joint_trajectory`；
  总开口 `0～80 mm`，`gripper_tcp` 位于 `tool0` 前方 `0.12 m`，不支持力控。

| 文件 | 用途 |
| --- | --- |
| `wheel_arm_moveit_config/config/wheel_arm.srdf` | 规划链、`home`/`ready`、被动关节与碰撞排除 |
| `wheel_arm_moveit_config/config/kinematics.yaml` | KDL 位置 IK |
| `wheel_arm_moveit_config/config/joint_limits.yaml` | 规划速度、加速度和默认缩放 |
| `wheel_arm_moveit_config/config/ompl_planning.yaml` | 规划器与请求适配器 |
| `wheel_arm_moveit_config/config/moveit_controllers.yaml` | 连接已有控制器，不管理/切换控制器 |
| `wheel_arm_moveit_config/config/moveit.rviz` | 机械臂专用 MotionPlanning 面板 |

执行链路为 `move_group → /wheel_arm/arm_controller/follow_joint_trajectory → ros2_control → Gazebo`。
`Plan` 不执行；`Execute` 和 `Plan & Execute` 会运动。启动不会自动回零或执行 `ready`。
`home` 对应四关节全零，`ready` 为 `[0, -0.6, 0.9, -0.3] rad`。

### 边界与排错

通用 MoveIt 入口不会将 Gazebo 家具或 RGB-D 自动导入 Planning Scene；不能把“自碰撞通过”
理解为真实环境轨迹安全。杯子测试入口另外提供桌面碰撞模型和限定的抓取流程，
但没有通用环境重建、底盘锁定或跨进程控制权仲裁。
优先在简单场景停稳后使用命名姿态，不与遥控、导航或其他机械臂客户端并发执行。

- 当前状态不更新：检查 `/clock`、`/wheel_arm/joint_states`；若使用其他话题，通过参数覆盖。
- 控制器不可用：检查 `/wheel_arm/arm_controller/follow_joint_trajectory` Action，以及
  `/wheel_arm/controller_manager` 中 `arm_controller` 是否为 `active`。
- 起点碰撞：查看碰撞连杆和真实关节状态，不要用扩大碰撞排除范围来绕过检查。
- `No 3D sensor plugin(s) defined for octomap updates`：当前尚未配置深度碰撞地图，
  该提示不代表 OMPL 或自碰撞检查失效，但也不意味着已具备环境避障。
- RViz 的 `/recognize_objects not available`：没有接入该物体识别 Action，
  不影响命名姿态规划；它不是项目语言/视觉模型服务的接口。

### 退出兼容方案

本机 MoveIt `2.5.10` 曾在 `move_group` 和专用 RViz 退出时发生段错误。
动态加载日志确认：控制器插件先被卸载，之后 ROS 回调组析构又访问了该库原先映射的地址。
单独用临时目录中的 `rclcpp/rclcpp_action 16.0.21` 替换测试进程的 `16.0.15` 仍复现，
因此不能把升级这两个系统包当作已验证的解决方案。

项目默认启用 Linux/glibc 的局部兼容库 `wheel_arm_moveit_plugin_lifetime`：
仅在本启动文件的 `move_group`、专用 RViz 子进程中设置 `LD_PRELOAD`，
为动态加载的 `libmoveit_*` 库附加 `RTLD_NODELETE`，使其代码和静态数据保留至进程结束。
节点、控制器、动作客户端和普通 C++ 对象仍正常析构，不捕获段错误、不强制成功退出。

这不是上游析构顺序的源码修复；代价是上述库不在运行过程中热卸载。
不修改 `/opt/ros`、全局 shell 环境或系统包，也不改变 Gazebo、Nav2 和无关库的加载行为；
已有 `LD_PRELOAD` 内容会保留。不要在整个 ROS 终端全局导出此兼容库。
排查上游修复时可使用 `plugin_lifetime_workaround:=false` 对照，但本机关闭后仍能复现崩溃。

测试现在同时断言规划/执行成功和 `move_group`/RViz 正常退出，而不是只检查运动结果。
退出阶段仍可能出现插件装载器或 ROS context 相关警告，不能与段错误混为一谈。
退出整套启动后仍应确认没有残留仿真进程，尤其不要重复启动控制器。

## 末端位置命令

`wheel_arm_manipulation` 是 MoveIt 服务和 Action 的客户端，不直接发布关节位置指令，
不启动仿真或第二套控制器。先启动 `wheel_arm_moveit_config` 的任一入口。

| 命令 | 行为 |
| --- | --- |
| `arm_goal state` | 读取完整、新鲜的关节反馈并调用 FK，显示当前末端位姿 |
| `arm_goal named ready` | 从 SRDF 读取命名目标，规划并预览，不执行 |
| `arm_goal position --x 0.55 --y 0 --z 0.65` | 无碰撞位置 IK、FK 校验、关节空间规划和预览，不执行 |
| `arm_goal gripper open/closed` | 预览张开/闭合，规划组为 `gripper` |
| `arm_goal gripper width --width 0.05` | 预览总开口 50 mm，`--width` 单位为米 |
| 上述运动命令增加 `--execute` | 核对后输入 `y`，才发送轨迹执行 Action |

通过 `ros2 run wheel_arm_manipulation` 调用上述入口。只读 `state` 不要求里程计或停稳，
仍要求有效时钟、关节反馈和 MoveIt FK 服务；运动命令的预览阶段也要求停稳。
默认使用仿真时钟；非仿真测试可加 `--ros-args -p use_sim_time:=false`，不能在 Gazebo 中随意混用。

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--x/--y/--z` | 必填 | `tool0` 相对 `base_footprint` 的绝对坐标，米，仅位置命令 |
| `--execute` | 关闭 | 仅运动命令；执行还需 TTY 和逐次人工确认，不接受管道输入 |
| `--velocity` / `--acceleration` | `0.2` / `0.2` | 缩放范围 `(0, 1]`，上限来自共享 MoveIt 配置 |
| `--wait` | `10` | 每次反馈就绪、停稳或服务调用的墙钟等待秒数 |
| `--timeout` | `120` | 运动命令的执行墙钟超时，仿真运行较慢时可适当增大 |
| `--joint-states-topic` | `/wheel_arm/joint_states` | 实际关节反馈 |
| `--odom-topic` | `/odometry/filtered` | 底盘停稳反馈，`state` 不使用此话题 |

```bash
# EKF 关闭时，改用底盘控制器的里程计做停稳检查
ros2 run wheel_arm_manipulation arm_goal named ready \
  --odom-topic /wheel_arm/diff_drive_base_controller/odom --execute
```

### 检查与执行链路

1. 从共享 xacro/SRDF/规划限位读取关节、位置上限、规划速度/加速度和命名姿态，不复制一份机械臂模型。
2. 拒绝过期或不完整反馈。运动命令要求关节速度不超过 `0.02 rad/s`，底盘线速度不超过
   `0.02 m/s`、角速度不超过 `0.04 rad/s`，连续满足至少 `0.3 s`（墙钟和 ROS 时间均需推进）。
3. 检查起点和目标碰撞；XYZ 使用当前关节作为 KDL 逆解种子，再用 FK 检查位置误差不超过 `5 mm`。
4. OMPL 生成关节轨迹；检查关节集合、有限值、限位、速度/加速度、相邻点平均速度、相对时间和起终点。
5. 向 `/display_planned_path` 发布同一条轨迹。默认结束，不发送执行目标。
6. `--execute` 且输入 `y` 后，逐点复查 MoveIt 碰撞状态，再重新读取停稳反馈；
   当前关节与预览起点偏差超过 `0.03 rad` 则轨迹作废，要求重新规划。
7. 通过 `/execute_trajectory` 交给 MoveIt，最终连接原有 `arm_controller`；
   Action 成功之后还需确认停稳、关节目标偏差不超过 `0.05 rad`，XYZ 最终 FK 误差不超过 `3 cm`。

退出码：正常预览、主动不确认或验证执行成功为 `0`，失败/终态或停稳未知为 `1`，
安全确认后的中断或未发送目标前中断为 `130`。取消只针对本次目标 UUID，
不取消其他客户端的任务；取消应答不能代替终态和停稳确认。

当前仍是仿真实验入口，没有底盘锁定或跨客户端控制权仲裁。停稳依据是传感器反馈，不是 ground truth。
`arm_goal position` 不支持指定任意朝向或笛卡尔直线插补，也不自动识别物体或抓取；
杯子抓放使用下文独立入口。IK 解不保证唯一、最近或最优。
逐轨迹点复查也不等价于持续动态环境避障，执行期间不要移动场景障碍物。

## RGB-D 杯子抓取

### 场景与模型

```bash
ros2 launch wheel_arm_manipulation cup_test.launch.py \
  camera_mount:=mast cup_model:=ceramic_cup
```

该入口包含 Gazebo、控制器、MoveIt、专用 RViz、杯子真值桥接和测试桌/地面碰撞体，
不启动 Nav2 或语言模型。真值日志用于测试对照，不输入杯子感知或抓取算法。
不要与已有仿真重复启动。参数如下：

| 参数 | 默认值 | 可选值与含义 |
| --- | --- | --- |
| `camera_mount` | `low` | `mast` 高位相机适合桌面识别；`low` 保留旧模型 |
| `cup_model` | `test_cup` | `test_cup` 简化杯、`ceramic_mug` 有柄杯、`ceramic_cup` 白灰色无柄杯 |
| `gui` / `rviz` | `true` / `true` | 分别控制 Gazebo 和 RViz 窗口，关闭 GUI 仍需要传感器渲染 |

感知抓取当前只针对 `ceramic_cup`，有柄模型保留用于展示，不宣称所有模型都能由该入口可靠抓取。
固定测试桌中心为世界坐标 `(0.70, 0, 0.27435) m`，台面高度 `0.5487 m`；
杯子初始中心约为 `(0.668, 0, 0.5988) m`，落稳后会有微小变化。
这些是场景摆放值，不作为 `cup_grasp` 的杯子定位输入；世界坐标不可直接代替车体坐标。

录制演示建议保留默认小车出生位置、朝向及场景初始机械臂姿态，固定桌杯摆位。
重新启动整个 `cup_test` 可恢复初始状态；移动后重新识别和预览，不能沿用旧计划。
当前不继续扩展任意起始姿态，也不自动调整底盘到可抓取位置。

### 识别、预览与执行

```bash
ros2 run wheel_arm_manipulation cup_grasp --detect-only
ros2 run wheel_arm_manipulation cup_grasp
ros2 run wheel_arm_manipulation cup_grasp --execute

# 可选：保留每次采集的首帧 RGB-D、相机内参和拍摄时刻 TF
ros2 run wheel_arm_manipulation cup_grasp --detect-only --output-dir observations/cup_debug
```

默认仅测量和规划，执行需要交互终端输入 `y`，不支持跳过确认。主要参数：

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--detect-only` | 关闭 | 仅识别并发布标记，不发送运动目标 |
| `--execute` | 关闭 | 完整预览后人工确认执行，与 `--detect-only` 互斥 |
| `--wait` / `--timeout` | `15` / `120` | 反馈/采集/服务等待、单次运动执行墙钟超时秒数 |
| `--velocity` / `--acceleration` | `0.2` / `0.2` | 相对共享限位的缩放 |
| `--lift-height` | `0.08 m` | 允许 `0.05～0.12 m` |
| `--output-dir` | 不保存 | 每次采集首帧的 `rgb.png` 与 `geometry.npz`；不是逐帧录像 |
| `--rgb-topic` / `--depth-topic` / `--info-topic` | `/camera/image` / `/camera/depth_image` / `/camera/camera_info` | 必须为配准、同步的 RGB-D 与内参 |
| `--camera-frame` | `camera_optical_frame` | 实际光学坐标系，不是靠改 frame_id 实现图像配准 |
| `--odom-topic` | `/odometry/filtered` | 关闭 EKF 时显式改为底盘原始里程计 |

识别将有效深度反投影为点云，通过图像时刻 TF 转到 `base_footprint`；RANSAC 寻找近水平桌面，
结合中性色外观、连通区域和杯身侧面圆拟合估计杯子中心、半径和高度。
要求三个独立观测稳定，唯一候选、完整度、拟合误差和可抓尺寸均通过；不是通用语义检测。
当前感知使用 `0.1～2 m` 的近距离深度，仿真相机的远裁剪仍为 `8 m`。

以杯身中心为 `gripper_tcp` 目标，从机械臂底座转轴指向杯子的方向水平接近；
预抓取点后退 `8 cm`，夹爪保持水平，目标开口为测得直径减 `1 mm`。
通用到预抓取位由 OMPL 规划，接近/升降/退离由 URDF 数值 IK 的直线采样轨迹完成，
带限速、路径偏差和 MoveIt 碰撞检查。此开口策略仅为仿真位置控制，不代表力控或可靠夹持。

开始到预抓取位时不放开杯子与手指碰撞；接近和夹持阶段才允许左右夹指接触。
测试桌碰撞位置按本次深度测量刷新，但其他家具不会自动导入。
执行前与接近/闭爪前复查杯子；底盘相对本次识别位姿变化时停止原计划。
抬升后用深度再次检查杯口升高，通过后才放回并松爪；视觉失败或夹持状态不确定时，
保留夹持和规划附着对象，人工检查后重启场景，不自动松爪或回零。

### RViz 与深度排错

- 绿色桌面/地面是 Planning Scene 碰撞几何，不表示“安全区域”。
- 橙色带交互圆环的机械臂是目标姿态；默认红色连杆表示预览姿态碰撞，不等于实际机器人已碰撞。
  可在 `MotionPlanning → Planning Request` 关闭 `Query Goal State/Query Start State` 隐藏预览，不关闭碰撞检测。
- `/cup_grasp/markers` 为测量杯子标记，`/cup_grasp/pose` 为杯身中心，
  `/cup_grasp/image` 为带轮廓的 RGB 图；均为单次采集结果，不是持续物体跟踪。
- 专用 RViz 的 Image 直接显示 `/camera/depth_image`，显示范围为 `0.1～3 m`，不改变原始数据。
  启动对该 RViz 进程设置 `QT_USE_NATIVE_WINDOWS=1`，规避本机嵌入深度窗口崩溃。
- 高位相机中心已从立柱顶面上移 `27.5 mm`，避开本机 Fortress/Ogre2 可复现的异常渲染条件。
  改动需要重新启动仿真加载模型，不是只重启 RViz；旧低位模型、雷达与 Ogre2 均保留。

黑图时先检查原始数据，不要仅改变显示颜色：

```bash
ros2 run wheel_arm_manipulation camera_check --seconds 120
```

有效比例为零且出现 Inf/NaN 表示源深度无效，不能继续抓取；有效深度多在 `3 m` 以外时，
应检查 RViz 显示范围。接收间隔持续增大时检查仿真暂停或断流。
该工具只读，不发送运动命令；尚不能仅凭这些统计自动判断显卡、裁剪或遮挡中的具体根因。

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

`robot_chat` 是 `language_observe` 的持续终端入口，不接收位置参数任务；
启动后在 `你：` 提示符输入，每条任务均需人工确认，不保存跨轮对话记忆。
两者共用下表参数，但 `robot_chat` 的 `--endpoint` 默认是
`http://127.0.0.1:11435`，原单次入口仍默认 Ollama `11434`。
`--preview` 在交互入口同样可用，此时每条输入只解析，不发送导航或拍摄。

```bash
ros2 run wheel_arm_navigation robot_chat --file maps/semantic_goals.yaml

# 关闭 EKF 时，停稳检查也需改为原始里程计
ros2 run wheel_arm_navigation robot_chat \
  --odom-topic /wheel_arm/diff_drive_base_controller/odom
```

窗口内 `/help` 查看帮助，`/quit` 或 `/exit` 退出；空闲时 `Ctrl+C` 退出，
执行时 `Ctrl+C` 请求取消。普通失败或已确认取消后返回输入；导航终态未知时
终止会话，不继续接收任务。日志 `source` 记录 `interaction=terminal_chat` 和
`session_id`，同一窗口的多次执行可按会话关联，图片仍按每次观测分别保存。

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

# 真实 MoveIt + 模拟机械臂控制器，不启动 Gazebo
ROS_DOMAIN_ID=89 ROS_LOCALHOST_ONLY=1 WHEELARM_ROS_TEST=1 \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest \
  src/wheelarm-lab/wheel_arm_moveit_config/test -q

# 在上面的模拟控制器测试中同时短暂打开 RViz，验证 MotionPlanning 插件加载
ROS_DOMAIN_ID=89 ROS_LOCALHOST_ONLY=1 WHEELARM_ROS_TEST=1 WHEELARM_GUI_TEST=1 \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest \
  src/wheelarm-lab/wheel_arm_moveit_config/test -q -k mock

# 可选：额外启动无 GUI Gazebo，实际执行 ready 和 home，需可用的图形渲染环境
ROS_DOMAIN_ID=89 ROS_LOCALHOST_ONLY=1 WHEELARM_ROS_TEST=1 WHEELARM_GAZEBO_TEST=1 \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest \
  src/wheelarm-lab/wheel_arm_moveit_config/test -q

# 机械臂命令的模拟控制器集成测试：预览、XYZ、确认、取消、超时与反馈失败
ROS_DOMAIN_ID=90 ROS_LOCALHOST_ONLY=1 WHEELARM_ROS_TEST=1 \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest \
  src/wheelarm-lab/wheel_arm_manipulation/test -q

# 可选：在独立 Gazebo 副本中执行 XYZ 并返回 home
ROS_DOMAIN_ID=90 ROS_LOCALHOST_ONLY=1 WHEELARM_ROS_TEST=1 WHEELARM_GAZEBO_TEST=1 \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest \
  src/wheelarm-lab/wheel_arm_manipulation/test -q -k gazebo

# 可选：高位相机八个朝向的 RGB-D/雷达渲染验证，不执行抓取
ROS_DOMAIN_ID=91 ROS_LOCALHOST_ONLY=1 WHEELARM_GAZEBO_TEST=1 \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest \
  src/wheelarm-lab/wheel_arm_gazebo/test/test_rgbd_rendering.py -q

# 可选：受控场景完整 RGB-D 抓放，包含杯子换位/底盘换位；会执行仿真机械臂动作
ROS_DOMAIN_ID=92 ROS_LOCALHOST_ONLY=1 WHEELARM_GAZEBO_TEST=1 \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest \
  src/wheelarm-lab/wheel_arm_manipulation/test/test_cup_grasp_ros.py -q
```

窗口测试需要桌面，会短暂打开窗口并注入按键。Gazebo 包的定位测试使用真实 AMCL
与合成地图/激光，不启动 Gazebo；不证明真实可达性、定位精度或共用 GPU 性能。
模型的协议、分词器和 CUDA 测试见[模型服务测试](../model_service/README.md#测试)。
机械臂 Gazebo 测试自动使用独立 transport partition，结束后清理自己的进程组；
未设置 `WHEELARM_GAZEBO_TEST=1` 时跳过，不能并行占用同一个 ROS 域。
