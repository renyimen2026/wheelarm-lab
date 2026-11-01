# 场景与测试模型资源

## 住宅场景

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

## 桌面抓取模型

`test_cup/`、`ceramic_mug/` 和 `ceramic_cup/` 是本项目自制的简化测试杯、有柄陶瓷杯和无柄陶瓷杯，
不是 AWS 住宅资源，采用项目根目录的 [Apache-2.0 许可证](../../LICENSE)。
两个陶瓷模型为真实空心外形；物理碰撞用杯壁和杯底的简化几何近似，不等于真实陶瓷接触模型。

生成器为 [ceramic_mug/generate_model.py](ceramic_mug/generate_model.py)，同时生成有柄/无柄资产及对应世界。
运行会重写相关生成文件；场景资产已经随仓库提供，日常启动无需重新生成。
使用入口及能力边界见[首页](../../README.md#双指夹爪与-rgb-d-抓取)。
