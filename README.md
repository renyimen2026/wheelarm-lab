# WheelArm Lab

一个面向 ROS 2 的开源轮臂式机器人仿真平台，用于移动操作、2D SLAM、Nav2 导航与具身智能算法实验。

项目基于 ROS 2 Humble 和 Gazebo Fortress 构建。它提供可运行的差速轮式底盘、4 自由度机械臂、2D 激光雷达、IMU、RGB 相机、`ros2_control` 控制、状态估计、SLAM、导航和键盘遥控基础设施。后续将围绕视觉语言导航（VLN）和视觉语言动作（VLA）持续扩展。

## 功能概览

- 差速轮式移动底盘与 4 自由度机械臂
- 2D LiDAR、IMU、RGB 相机等仿真传感器
- `ros2_control`：底盘差速控制器和机械臂轨迹控制器
- `robot_localization`：轮速里程计与 IMU 融合
- `slam_toolbox`：2D 激光建图与回环优化
- Nav2：全局路径规划、局部避障与目标点导航
- Gazebo 室内实验场景与预配置 RViz
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

## 当前状态与路线图

- [x] 轮臂机器人模型与 Gazebo Fortress 仿真
- [x] 底盘、机械臂与键盘遥控
- [x] 2D LiDAR、IMU、RGB 相机
- [x] EKF 状态估计、2D SLAM、Nav2
- [ ] 地图加载与定位模式
- [ ] 深度相机与 3D 感知
- [ ] 任务级移动操作
- [ ] VLN 场景、数据采集和评测接口
- [ ] VLA 策略接入

## 已知限制

这是一个持续迭代的仿真研究平台。快速旋转时，实时 LaserScan 与全局栅格地图可能出现短暂显示不同步；机器人停止后，扫描匹配和回环优化会继续校正全局地图。该现象不影响当前的 SLAM、导航与算法验证流程。

## 贡献

欢迎提交 Issue、功能建议和 Pull Request。涉及机器人模型、控制器或导航参数的修改，请同时说明测试环境、启动命令和验证结果。

## 许可证

本项目采用 [Apache License 2.0](LICENSE) 开源。
