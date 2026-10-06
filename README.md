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
- 命名目标点记录与 Nav2 导航（语义导航基础接口）
- 多点顺序任务、停留与循环执行、任务结果日志

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
  ros-humble-robot-localization \
  python3-tk
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

### 升级已有工作空间

本次增加 `wheel_arm_navigation` 和 Tk 键盘窗口，旧版终端遥控方式已替换。
在工作空间根目录安装新增依赖并编译：

```bash
source /opt/ros/humble/setup.bash
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install --packages-select \
  wheel_arm_control wheel_arm_teleop wheel_arm_navigation
source install/setup.bash
```

关闭旧遥控和导航任务，并重启仿真使新的底盘控制器限制生效。
`command_hold_time` 已移除；已有启动脚本应删除该参数，改用键盘窗口的松键事件。
已有地图可继续使用，同一地图的目标点也可复用；切换地图后应重新核对目标位置。

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
├── wheel_arm_teleop/       # 键盘遥控节点
└── wheel_arm_navigation/   # 地图命名目标点、顺序任务与执行日志
```

## 键盘遥控

启动仿真后，在本地桌面另一个已加载工作空间环境的终端中运行：

```bash
ros2 run wheel_arm_teleop keyboard_teleop
```

命令打开独立键盘窗口，点击窗口使其获得焦点后操作。
按住移动、松开对应按键立即发布该方向的零速度，失去焦点则停止所有底盘运动。
不再通过终端字符和按键超时推测松键，因此需要桌面显示环境；不支持纯 SSH 终端操作。
缺少 Tk 时安装 `sudo apt install python3-tk`。

| 按键 | 功能 |
| --- | --- |
| `W` / `S` | 前进 / 后退 |
| `A` / `D` | 左转 / 右转 |
| `W+A` / `W+D` | 前进同时左转 / 右转 |
| `S+A` / `S+D` | 后退同时左转 / 右转 |
| `Space` | 停止底盘 |
| `Q` / `E` | `shoulder_pan_joint` 正向 / 反向 |
| `R` / `F` | `shoulder_lift_joint` 正向 / 反向 |
| `T` / `G` | `elbow_joint` 正向 / 反向 |
| `Y` / `H` | `wrist_pitch_joint` 正向 / 反向 |
| `X` | 机械臂回零 |
| `Esc` / `Ctrl+C` / 关闭窗口 | 停止底盘并退出 |

组合键按各轴独立处理：松开 `W` 但仍按住 `A` 时，停止前进并继续转向。
同轴反向按键（`W+S`、`A+D`）相互抵消。空格或窗口停止按钮会解除当前移动键，
需要先松开这些键再重新按下才能继续，防止键盘自动重复导致意外恢复运动。

默认线速度为 `0.5 m/s`，角速度为 `1.0 rad/s`，速度指令以 `50 Hz` 发布，
按键状态变化时也立即发布；机械臂每次按键的关节增量为 `0.05 rad`，
持续按住机械臂按键时默认每秒重复 5 次，可通过 `arm_repeat_rate` 调整。
可以在启动时覆盖速度参数，无需修改源码，例如：

```bash
ros2 run wheel_arm_teleop keyboard_teleop --ros-args \
  -p linear_speed:=0.15 -p angular_speed:=0.45
```

使用 Nav2 自主导航时，请停止键盘遥控和其他速度发布命令，避免多个节点同时控制底盘。
需要退出键盘程序，不能只切换窗口：失去焦点时遥控仍持续发布零速度作为安全保护。

底盘控制器保留速度与加减速限制。当前线加速度限制为 `±2.0 m/s²`，
角加速度限制为 `±3.0 rad/s²`，控制指令断流 `0.2 s` 后触发超时停止。
松键会立即发送停止指令，但物理车体仍需减速；理想限制下，从 `0.5 m/s` 降到零约需
`0.25 s`，实际还受仿真接触和惯性影响，不保证瞬时静止。
上述控制器限制同样影响自主导航，不只影响遥控；修改后需重新构建 `wheel_arm_control`
并重启仿真使控制器重新加载参数。

```bash
colcon build --symlink-install --packages-select wheel_arm_teleop wheel_arm_control
source install/setup.bash
```

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

## 语义目标点导航

语义目标点是保存地图中带名称的可通行位置，例如“客厅”或“厨房入口”。
当前提供名称到地图位姿的映射，不包含语言模型、自动物体识别或完整 VLN 算法。

更新代码后重新编译：

```bash
colcon build --symlink-install --packages-select wheel_arm_navigation
source install/setup.bash
```

先启动上述固定地图导航并设置初始位姿。用 RViz 导航或键盘将机器人移动到
需要记录的位置，确认激光与地图对齐、机器人已停止后，在另一终端记录当前位置：

```bash
ros2 run wheel_arm_navigation semantic_goal save 客厅 --ros-args -p use_sim_time:=true
ros2 run wheel_arm_navigation semantic_goal save 厨房入口 --ros-args -p use_sim_time:=true
```

默认读取 `map -> base_footprint`，保存到工作空间根目录的 `maps/semantic_goals.yaml`。
请在同一工作空间根目录执行命令；多个终端不要同时写入同一目标点文件。
名称重复时拒绝覆盖，如需更新位置，添加 `--overwrite`。

列出目标点和按名称导航：

```bash
ros2 run wheel_arm_navigation semantic_goal list
ros2 run wheel_arm_navigation semantic_goal go 客厅 --ros-args -p use_sim_time:=true
```

导航时停止键盘遥控，不要同时在 RViz 发送目标。终端输出剩余距离与执行结果；
`Ctrl+C` 请求取消，默认超过 180 秒请求取消，可用 `go 客厅 --timeout 300` 修改。
若日志提示取消未确认，必须检查 Nav2 状态及机器人是否停止。
仿真中的 `save` 和 `go` 命令需要 `use_sim_time:=true`，使用与 TF 相同的仿真时钟。

也可直接输入 RViz 地图坐标（米、弧度），不能使用 Gazebo 世界坐标代替：

```bash
ros2 run wheel_arm_navigation semantic_goal set 客厅 --x 1.0 --y 2.0 --yaw 0.0
ros2 run wheel_arm_navigation semantic_goal --file maps/semantic_goals.yaml list
```

上面的手动坐标仅演示格式，不是本住宅场景的预置目标。应选择可通行空地，
为底盘和机械臂留出避障空间。目标点文件结构如下：

```yaml
frame_id: map
goals:
  客厅:
    x: 1.0
    y: 2.0
    yaw: 0.0
```

目标点与当前地图绑定，切换场景或重新建图后应重新记录，不可盲目复用旧坐标。
这些文件属于本地生成数据，不随仓库分发。

## 多点导航任务

先启动固定地图导航、完成初始定位，并记录路线中所有目标点。
更新后重新编译 `wheel_arm_navigation` 并加载 `install/setup.bash`，在工作空间根目录执行：

```bash
ros2 run wheel_arm_navigation mission_runner 客厅 厨房入口 卧室 客厅 \
  --stay 2 --ros-args -p use_sim_time:=true
```

执行器按顺序逐个调用 Nav2，到达每个目标后停留 2 秒，再发送下一个目标。
启动前检查整条路线，只要有名称未记录，就拒绝启动，不会先走部分路线。
支持目标名称重复；执行中使用启动时读取的坐标快照。
任务运行期间不要同时启动键盘遥控、其他任务或在 RViz 中发送导航目标。

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--file` | `maps/semantic_goals.yaml` | 已记录的目标点文件 |
| `--stay` | `0` | 每个成功目标后的停留秒数，包含最后一个目标 |
| `--timeout` | `180` | 每个目标接受后的最大等待秒数 |
| `--wait` | `10` | 等待 Nav2、仿真时钟及目标应答的超时秒数 |
| `--repeat` | `1` | 整条路线的执行轮数 |
| `--on-failure` | `stop` | 失败后停止；`skip` 则继续后续目标 |
| `--log-dir` | `mission_logs` | JSON 执行日志目录 |
| `--name` | `navigation_task` | 日志中的任务名称 |

例如循环两轮，允许跳过已确认结束的失败目标：

```bash
ros2 run wheel_arm_navigation mission_runner 客厅 卧室 客厅 \
  --repeat 2 --stay 1 --on-failure skip --ros-args -p use_sim_time:=true
```

`Ctrl+C` 会请求取消当前导航，或中断停留，并停止后续目标。
超时后请求取消，只有收到目标终态才允许跳过。
Nav2 不可用、通信异常或取消后目标结束未确认时，无论策略如何都停止任务。
停留、超时和日志耗时按实际经过的时间计量，不随仿真暂停而停止计时。

### 保存任务配置

可以将固定路线写成 YAML。仓库提供 `wheel_arm_navigation/config/patrol.example.yaml`：

```yaml
name: indoor_patrol
goals: [客厅, 厨房入口, 卧室, 客厅]
stay: 2.0
timeout: 180.0
wait: 10.0
repeat: 1
on_failure: stop
```

示例只定义名称，不提供住宅场景的预置坐标；必须先记录对应目标点。
运行该配置：

```bash
ros2 run wheel_arm_navigation mission_runner \
  --task src/wheelarm-lab/wheel_arm_navigation/config/patrol.example.yaml \
  --ros-args -p use_sim_time:=true
```

`--task` 不能与命令行目标列表混用；`--stay` 等命令行参数可以覆盖配置中的值。

### 执行日志

每次任务创建独立的 `mission_logs/<UTC时间>-<唯一ID>.json`，开始导航前确认日志可写。
日志记录任务配置、目标坐标快照、总体状态、成功和失败数量，以及每一步的：

- 目标名称、轮次和顺序。
- 开始/结束时间、导航耗时和实际停留时长。
- 导航状态、失败原因、Nav2 action 状态码（若收到）及是否允许继续。

任务状态包括 `completed`、`completed_with_failures`、`failed`、`canceled` 和
`unsafe_stop`，异常退出还可能记录为 `error`。有失败但跳过后走完整条路线，
仍返回非零退出码，不会将部分成功报告为全部成功。
未执行目标标记为 `not_run`；每一步前后更新日志，强制终止进程时可能保留 `running`，
这表示结果未确认，不代表任务成功。当前不支持断点恢复。
日志属于本地运行数据，默认不提交到 Git；Nav2 返回成功也不等于已验证真实定位精度。

## 测试

运行导航任务与键盘输入单元测试（工作空间根目录，默认不启动 DDS 或窗口）：

```bash
colcon test --packages-select wheel_arm_navigation wheel_arm_teleop \
  --event-handlers console_direct+
colcon test-result --verbose
```

ROS 接口集成测试使用模拟 action 服务和 TF，不验证真实路径可达性。
需要单独启用，并选择未使用的 ROS 域以免连接正在运行的机器人：

```bash
ROS_DOMAIN_ID=87 WHEELARM_ROS_TEST=1 python3 -m pytest \
  src/wheelarm-lab/wheel_arm_navigation/test -q
```

键盘 ROS 与真实窗口测试另行启用；会短暂打开窗口并注入测试按键，
仅向隔离 ROS 域发布速度消息，不控制正在运行的仿真：

```bash
ROS_DOMAIN_ID=88 WHEELARM_ROS_TEST=1 WHEELARM_GUI_TEST=1 python3 -m pytest \
  src/wheelarm-lab/wheel_arm_teleop/test -q
```

上述 ROS 域必须未被其他机器人使用。窗口测试需要本地桌面显示环境。

## 当前状态与路线图

- [x] 轮臂机器人模型与 Gazebo Fortress 仿真
- [x] 底盘、机械臂与键盘遥控
- [x] 2D LiDAR、IMU、RGB 相机
- [x] EKF 状态估计、2D SLAM、Nav2
- [x] 地图加载与 AMCL 定位模式
- [x] 命名语义目标点与 Nav2 action 接口
- [x] 多目标顺序导航与任务执行日志
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

维护者：`renyimen`，联系邮箱：`renyimen2026@foxmail.com`。

欢迎提交 Issue、功能建议和 Pull Request。涉及机器人模型、控制器或导航参数的修改，请同时说明测试环境、启动命令和验证结果。

提交功能或配置改动时，请同步更新 [CHANGELOG.md](CHANGELOG.md) 的“未发布”章节；
涉及使用方式变化时，也请更新本 README。

## 许可证

本项目采用 [Apache License 2.0](LICENSE) 开源。

第三方住宅模型与贴图遵循 [AWS 原始许可证](wheel_arm_gazebo/models/AWS_LICENSE)。
