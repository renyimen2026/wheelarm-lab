# WheelArm Lab

一个面向 ROS 2 的开源轮臂式机器人仿真平台，用于移动操作、2D SLAM、Nav2 导航与具身智能算法实验。

项目基于 ROS 2 Humble 和 Gazebo Fortress 构建。它提供可运行的差速轮式底盘、4 自由度机械臂、2D 激光雷达、IMU、RGB 相机、`ros2_control` 控制、状态估计、SLAM、导航和键盘遥控基础设施。后续将围绕视觉语言导航（VLN）和视觉语言动作（VLA）持续扩展。

历次功能和配置变化见 [更新记录](CHANGELOG.md)。

## 功能概览

- 差速轮式移动底盘与 4 自由度机械臂
- 2D LiDAR、IMU、RGB 相机等仿真传感器
- `ros2_control`：底盘差速控制器和机械臂轨迹控制器
- `robot_localization`：轮速里程计与 IMU 融合
- `slam_toolbox`：2D 激光建图与回环优化
- Nav2：全局路径规划、局部避障与目标点导航
- 带贴图家具的住宅室内场景与预配置 RViz
- 键盘遥控底盘和机械臂关节

## 环境要求

| 项目 | 版本 |
| --- | --- |
| Ubuntu | 22.04 |
| ROS 2 | Humble |
| Gazebo | Fortress |

安装基础依赖：

```bash
sudo apt update
sudo apt install \
  ros-humble-ros-gz \
  ros-humble-gz-ros2-control \
  ros-humble-ros2-controllers \
  ros-humble-navigation2 \
  ros-humble-nav2-bringup \
  ros-humble-slam-toolbox \
  ros-humble-robot-localization
```

## 快速开始

创建 ROS 2 工作空间并克隆本仓库：

```bash
mkdir -p ~/wheelarm_ws/src
cd ~/wheelarm_ws/src
git clone https://github.com/renyimen2026/wheelarm-lab.git
cd ..
```

安装依赖并编译：

```bash
source /opt/ros/humble/setup.bash
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash
```

启动基础仿真：

```bash
ros2 launch wheel_arm_gazebo sim.launch.py
```

启动 Gazebo、SLAM、Nav2 与 RViz：

```bash
ros2 launch wheel_arm_gazebo nav.launch.py
```

`nav.launch.py` 默认使用 EKF 融合后的里程计，并自动加载 RViz 配置。

## 仿真场景

默认加载 `indoor_house.sdf`：基于 AWS RoboMaker Small House World 的现成
住宅场景，包含卧室、客厅、厨房、家具和家电。世界文件已适配 Gazebo Fortress，
模型与贴图保存在本仓库，运行时无需联网下载。机器人默认出生位置为
`x=3.5, y=1.0, z=0.05, yaw=0`，坐标单位为米，角度单位为弧度。

可以通过 `spawn_x`、`spawn_y`、`spawn_z` 和 `spawn_yaw` 设置出生位姿：

```bash
ros2 launch wheel_arm_gazebo nav.launch.py spawn_x:=3.5 spawn_y:=1.0
```

原有实验室场景仍可选择：

```bash
ros2 launch wheel_arm_gazebo nav.launch.py \
  world:=src/wheelarm-lab/wheel_arm_gazebo/worlds/indoor_lab.sdf \
  spawn_x:=0 spawn_y:=0 spawn_z:=0
```

上述相对路径命令在工作空间根目录执行。切换场景后需重新建图，旧场景地图不能复用。
住宅资产来源和独立许可证见 [模型说明](wheel_arm_gazebo/models/README.md)。
该场景尚未附带 VLN 任务或语义标注。

## 项目结构

```text
wheelarm-lab/
├── wheel_arm_description/  # xacro/URDF 机器人模型、传感器与 Gazebo 插件
├── wheel_arm_control/      # ros2_control 控制器配置
├── wheel_arm_gazebo/       # 仿真世界、launch、EKF、Nav2、RViz 配置
└── wheel_arm_teleop/       # 键盘遥控节点
```

## 键盘遥控

启动仿真后，在另一个已加载工作空间环境的终端中运行：

```bash
ros2 run wheel_arm_teleop keyboard_teleop
```

| 按键 | 功能 |
| --- | --- |
| `W` / `S` | 前进 / 后退 |
| `A` / `D` | 左转 / 右转 |
| `Space` | 停止底盘 |
| `Q` / `E` | `shoulder_pan_joint` 正向 / 反向 |
| `R` / `F` | `shoulder_lift_joint` 正向 / 反向 |
| `T` / `G` | `elbow_joint` 正向 / 反向 |
| `Y` / `H` | `wrist_pitch_joint` 正向 / 反向 |
| `X` | 机械臂回零 |

默认线速度为 `0.3 m/s`，角速度为 `1.0 rad/s`；机械臂每次按键的关节增量为 `0.05 rad`。
可以在启动时覆盖速度参数，无需修改源码，例如：

```bash
ros2 run wheel_arm_teleop keyboard_teleop --ros-args \
  -p linear_speed:=0.15 -p angular_speed:=0.45
```

使用 Nav2 自主导航时，请停止键盘遥控和其他速度发布命令，避免多个节点同时控制底盘。

## 里程计与定位

默认启动的 EKF 以轮速里程计提供车体速度和非完整约束，以 IMU 提供航向角与角速度：

```text
/wheel_arm/diff_drive_base_controller/odom  -> 原始轮速里程计
/imu                                         -> IMU
/odometry/filtered                           -> EKF 融合里程计
odom -> base_footprint                       -> EKF 发布的 TF
```

需要直接使用原始轮速里程计进行对比时：

```bash
ros2 launch wheel_arm_gazebo nav.launch.py use_ekf:=false
```

## 常用操作

不启动 RViz：

```bash
ros2 launch wheel_arm_gazebo nav.launch.py rviz:=false
```

关闭定位诊断日志：

```bash
ros2 launch wheel_arm_gazebo nav.launch.py localization_debug:=false
```

手动控制底盘：

```bash
ros2 topic pub /wheel_arm/diff_drive_base_controller/cmd_vel_unstamped \
  geometry_msgs/msg/Twist \
  "{linear: {x: 0.15}, angular: {z: 0.3}}"
```

发送机械臂轨迹：

```bash
ros2 topic pub /wheel_arm/arm_controller/joint_trajectory \
  trajectory_msgs/msg/JointTrajectory \
  "{joint_names: [shoulder_pan_joint, shoulder_lift_joint, elbow_joint, wrist_pitch_joint], points: [{positions: [0.0, 0.4, -0.8, 0.3], time_from_start: {sec: 2}}]}"
```

保存当前 SLAM 地图：

```bash
mkdir -p maps
ros2 run nav2_map_server map_saver_cli -f maps/wheel_arm_map
```

## 固定地图定位与导航

先使用在线 SLAM 完成建图并保存地图，再关闭建图启动进程。
在工作空间根目录启动固定地图模式（不要与建图模式同时运行）：

```bash
source install/setup.bash
ros2 launch wheel_arm_gazebo nav_localization.launch.py map:=maps/wheel_arm_map.yaml
```

该入口启动 Gazebo、EKF、地图服务器、AMCL、Nav2 和 RViz。
地图服务器加载已保存地图，AMCL 根据 `/scan` 和里程计 TF 发布 `map -> odom`，
EKF 继续发布 `odom -> base_footprint`。该模式不启动 `slam_toolbox`，也不修改保存的地图。
局部和全局代价地图仍会根据激光观测更新障碍物。

在 RViz 中使用 `2D Pose Estimate`，在地图上点击机器人所在位置，拖动指定朝向。
收到初始位姿后，AMCL 才能建立全局定位；在此之前机器人可能不显示，
Nav2 会提示等待 `map -> odom` TF。确认激光与墙体基本重合后，再设置导航目标。

如果出现 `AMCL cannot publish a pose ... Please set the initial pose`、
`Invalid frame ID "map"` 或 RViz 消息队列已满，先确认保存地图已显示，
再设置初始位姿。若设置后提示仍持续出现，应检查初始位置、朝向和定位 TF。

地图坐标与 Gazebo 世界坐标可能不同，`spawn_x` / `spawn_y` 不能直接当作 AMCL 初始位姿。
建议先使用与建图时相同的出生位姿，用地图中的房间和家具辨认当前位置。

关闭 RViz 或使用原始轮速计也可通过参数选择：

```bash
ros2 launch wheel_arm_gazebo nav_localization.launch.py \
  map:=maps/wheel_arm_map.yaml rviz:=false use_ekf:=false
```

地图路径不存在时，启动文件会提前报错。自定义地图必须与当前场景一致，
并保留 YAML 引用的图片文件。

地图默认保存到工作空间的 `maps/` 目录，属于本地生成数据，不随源码仓库分发。
首次克隆项目后，需要先建图并保存地图，再运行固定地图模式。

## 当前状态与路线图

- [x] 轮臂机器人模型与 Gazebo Fortress 仿真
- [x] 底盘、机械臂与键盘遥控
- [x] 2D LiDAR、IMU、RGB 相机
- [x] EKF 状态估计、2D SLAM、Nav2
- [x] 地图加载与 AMCL 定位模式
- [ ] 深度相机与 3D 感知
- [ ] 任务级移动操作
- [ ] VLN 场景、数据采集和评测接口
- [ ] VLA 策略接入

## 已知限制

这是一个持续迭代的仿真研究平台。在线建图时，旋转期间的实时 LaserScan 与全局栅格地图
可能短暂不重合；扫描匹配和回环优化可能进一步修正位姿与地图。
固定地图模式由 AMCL 估计位姿，不会通过回环修改已保存地图。持续的扫描错位需要检查定位，
不能仅靠提高地图刷新频率解决。

## 贡献

欢迎提交 Issue、功能建议和 Pull Request。涉及机器人模型、控制器或导航参数的修改，请同时说明测试环境、启动命令和验证结果。

提交功能或配置改动时，请同步更新 [CHANGELOG.md](CHANGELOG.md) 的“未发布”章节；
涉及使用方式变化时，也请更新本 README。

## 许可证

本项目采用 [Apache License 2.0](LICENSE) 开源。

第三方住宅模型与贴图遵循 [AWS 原始许可证](wheel_arm_gazebo/models/AWS_LICENSE)。
