# 住宅场景资源

来源：https://github.com/aws-robotics/aws-robomaker-small-house-world

分支：`ros2`

上游提交：`ff9631ca6d1db9c1ba656498151464b5ab74aafe`

模型、贴图和照片遵循上游 [AWS_LICENSE](AWS_LICENSE)，该许可证独立于本项目的
Apache-2.0 许可证。资源保存在本仓库，运行时可离线加载。

[indoor_house.sdf](../worlds/indoor_house.sdf) 改编自上游的 `worlds/small_house.world`。
本项目保留原住宅布局和静态家具，将世界配置适配为 Gazebo Fortress 系统插件，
并配置本地资源搜索路径和俯视相机。该场景尚未包含 VLN 语义标注或任务定义。

为通过 Fortress 校验，鞋架模型中重复的 `ixx` 标签已修正为 `izz`。
场景中的家具均为静态物体。

相框 Collada 模型的图片引用路径已改为本目录下的 `photos/`，
避免依赖上游仓库的工作空间目录结构。
