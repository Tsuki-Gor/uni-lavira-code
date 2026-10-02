# 03 · LaViRA 与环境到底通过什么接口交接

这次拆分的关键在于：算法交出目标，执行层提供观测与执行反馈。执行层不需要理解 LaViRA 的模型回复；算法不负责订阅 ROS、创建仿真环境或发送具体电机命令。

但现有平台能力没有完全相同。地面机器人主要通过一次阻塞导航调用完成目标，Habitat 则持续推进地图动作。读接口时既要看到共用的数据类型，也要看到这些实际能力差别。

## 1. 两组数据类型承担不同职责

算法任务数据放在 [lavira/runtime.py](../../lavira/runtime.py#L9)，环境观测与目标放在 [navigation_runtime](../../navigation_runtime/__init__.py#L9)。这里两个名称都包含 runtime，但不是两份环境实现。

| 所在位置 | 数据 | 谁主要使用它 |
| --- | --- | --- |
| `lavira/runtime.py` | `Task`、`TaskState`、`Progress`、`TaskResult` | 外部任务调度者与 LaViRA |
| `navigation_runtime/__init__.py` | `Frame`、`Panorama`、`PixelGoal`、`RelativeGoal`、`MapGoal`、`ExecutionReport` | 算法与执行能力的共同边界 |

通用 Agent 需要知道怎样提交一个 LaViRA 任务；其他导航算法则只需使用第二组数据和所需执行方法，不必接收 LaViRA 的 TODO 或决策字典。

## 2. 任务边界：一次调用描述一次任务

`Task` 当前字段如下，来源见 [定义](../../lavira/runtime.py#L9)。

| 字段 | 作用及限制 |
| --- | --- |
| `instruction` | 路线指令、目标文字或 EQA 问题；当前统一用字符串传递 |
| `kind` | `vln`、`object_nav`、`eqa` 三种之一 |
| `max_cycles` | 机器人按采图前的循环次数计数；Habitat 按新 LA 导航/回溯决策计数；均不等同于控制命令数 |
| `max_steps` | 当前用于 Habitat 的仿真动作预算；机器人 profile 设置此字段会被拒绝 |
| `answer_from_history` | Habitat EQA 是否允许在收尾阶段从已有航点全景补答 |
| `id` | 可携带 episode 标识；当前不是任务队列、去重或持久化机制 |
| `reuse_runtime_history` | Go1/G1 跨任务图片历史的显式兼容开关，默认关闭 |

任务驱动重新创建 `TaskState`，外部可以复用已经存在的连接和设备。运行时没有因为接到一个新任务就被核心自动 reset。反过来，重新连接机器人也不等于恢复上次导航任务的内部状态。

## 3. 观测边界：Frame 携带图像，也标明如何解释它

一个像素坐标只有与它所属的图像一起传递才有意义。`Frame` 为此提供图像或文件路径、深度、尺寸、朝向，以及需要地图投影时的位姿、内参和参考系。`Panorama.views` 再按视角名称组织多个 Frame。

| 字段 | 应如何读 | 当前实现差异 |
| --- | --- | --- |
| `image` / `path` | 内存图像或已保存文件的位置 | Go1/G1 全景主要携带路径，当前图像携带数组 |
| `color_order` | 通道顺序，而非可忽略的描述 | 机器人通常是 BGR，Habitat 明确为 RGB |
| `depth` / `depth_unit` | 原深度值及其单位或编码 | Go1/G1 是 `mm`；Cobot 是 `m`；Habitat 是 `normalized_0_1` |
| `width` / `height` | 图像坐标所依赖的尺寸 | 数组视图会填写；仅路径全景不额外读取文件补尺寸 |
| `heading_degrees` | 视图方向的描述 | 是局部视角信息，不能直接当世界 yaw |
| `pose` / `camera_intrinsics` | 投影时需要的采集信息 | Habitat 会提供；机器人兼容执行仍读取控制器实时数据 |
| `pose_heading_offset` | 从全景基准位姿投影此视角时叠加的角度 | Habitat 历史视角使用 |
| `reference_frame` | 这份观测属于哪个地图参考 | Habitat 用它拒绝跨地图观测 |

来源：[Frame 与 Panorama](../../navigation_runtime/__init__.py#L9)、[相机采图](../../navigation_runtime/camera_robot.py#L48)、[地图 Frame 组装](../../navigation_runtime/mapped_simulator.py#L65)。

这不是一份所有字段都保证齐全、不可变的传感器快照。`frozen=True` 禁止重新赋值数据类字段，却不会冻结 NumPy 数组、字典或内部引用。Habitat 的 CPU 位姿还保留了原共享引用行为；Go1/G1 像素导航会重新读取实时深度。两项都需要结合当前实现理解，不能仅从类型名称推断已经解决时序同步。

`Panorama.history` 是运行时提供的轨迹记录路径，主要供旧 Go1/G1 入口兼容；`record_dir` 是本次记录目录，Cobot 用来写采图和 bbox；`origin` 在 Habitat 中保存全景基准观测。它们不是一个统一的算法记忆库。

## 4. 动作边界：传数字目标，不传模型决策

模型可能返回“向左、靠近椅子”。算法先解析方向，再选择图像像素或局部位置，执行层才收到下面这些类型。

| 类型 | 表达的目标 | 单位、坐标及调用方式 |
| --- | --- | --- |
| `turn(degrees)` | 朝某方向转动 | 度，正数左转；Habitat 当前只接受 `0/+90/-90/180`，排成原 30 度离散动作 |
| `RelativeGoal(x, y)` | 机器人局部目标位置 | 米，x 向前，y 向左；相机执行器支持 |
| `PixelGoal(view, u, v, fallback)` | 某张图像中的目标像素 | 必须带 Frame；fallback 是调用者明确给出的局部目标 |
| `MapGoal(x, y, metres_per_cell, reference_frame)` | 当前地图中的位置 | x/y 为地图格，另带米/格和参考系；Habitat 地图执行器支持 |

见 [目标类型](../../navigation_runtime/__init__.py#L34)。类型会检查有限数值，MapGoal 还检查正比例尺及参考系非空；它们没有替代运动边界、碰撞检查或所有像素有效性检查。

同一 `PixelGoal` 也不是相同的底层实现。CameraRobotRuntime 用当前控制器深度投影到机器人局部位置，再交 iPlanner；GimbalRobotRuntime 交给 Cobot 的动态像素导航；MappedSimulatorRuntime 则使用 Frame 的深度、位姿、内参投影成 MapGoal。这里统一的是交接含义，不是强行采用一个投影算法。

## 5. 当前地面能力协议

[NavigationRuntime](../../navigation_runtime/__init__.py#L83) 用 `Protocol` 描述地面完整任务会调用的方法，不要求具体执行器继承共同基类。

| 方法 | 调用时机 | 应交回什么 |
| --- | --- | --- |
| `running` | 每次外层循环检查 | 当前是否继续运行 |
| `observe_panorama(step)` | LA 之前 | 全景；失败可返回 None |
| `turn(degrees)` | LA 选择方向后 | 完成原转向过程 |
| `observe_current()` | 转向之后，VA 之前 | 当前视图；失败可返回 None |
| `finish_observation()` | 需要等采图附带动作结束时 | Cobot 等复位线程；相机执行器无此附带动作 |
| `navigate(goal, step, on_trajectory_image=None)` | 目标构造之后 | 本次执行反馈，并可通过回调提供轨迹图 |
| `observe_views()` | 机器人 EQA 收尾 | 按平台原顺序提供固定相机视图 |

协议的目标联合类型不代表每个实现支持全部目标。CameraRobotRuntime 支持 PixelGoal 和 RelativeGoal；Cobot 的 GimbalRobotRuntime 只接受 PixelGoal，传其他类型会拒绝。Cobot 的 `running` 目前固定为 True，沿用以取消和预算控制循环的行为。

Habitat 的实际能力集合另有 `prepare_step`、`project`、`map_goal`、`queue_navigation`、`execute_pending`、`busy` 和 `stop` 等。它没有实现上述同形的完整阻塞 `navigate`，不能简单写成“任意 NavigationRuntime 都能承载任意 profile”。详细适配方式见 [Habitat 篇](06-habitat.md)。

## 6. 执行反馈与任务结果为什么分开

环境层使用 [ExecutionReport](../../navigation_runtime/__init__.py#L77) 报告一次动作调用。算法层用 [TaskResult](../../lavira/runtime.py#L40) 报告整次任务。两者的数据来源不同。

例如 iPlanner 给出一条路径，控制器函数执行并返回。执行器可以设置 `execution_called=True`，但 Go1/G1 缺少可靠的到达返回值，因此 `arrived` 仍为 None。Cobot 的 `arrived` 来自旧控制器布尔返回，也不是新增加的独立测量。

`detail` 可描述 `projection_unavailable`、`planner_failed` 或 `within_safety_buffer`。这些执行反馈不必然结束任务，后续是否继续仍由原算法流程决定。`last_execution` 只保存最后一次反馈，不是全程动作日志。

核心也不自动把控制器异常都转换为结果。某些异常继续向调用者传播，Go1/G1 轨迹执行路径还保留旧异常回退行为。上层系统要同时考虑正常返回和异常退出，不能仅检查 `result.reason`。

## 7. 图片回调怎样跨过边界，又不把算法塞进控制器

新 Go1/G1 调用传入一个接收文件路径的回调。控制器保存图片时调用它，不知道接收方是否为 LaViRA，也不处理 TODO、历史窗口或消息编码。

```text
TaskState.trajectory_images.append
    → CameraRobotRuntime.navigate(on_trajectory_image=...)
    → project_trajectory_to_image / execute_trajectory
    → 执行期间重规划（Go1 后台线程；G1 执行循环）
    → 保存路径后调用同一个回调
```

选择哪些图片、取几张、按什么顺序送 LA 仍属于算法。Go1/G1 全局 `iplanner_history` 保留，兼容入口按旧方式读取；新任务只积累自己的回调。任务结果会复制列表，不会在返回后被旧线程继续修改。见 [核心回调入口](../../lavira/agent.py#L183) 与 [异步归属检查](../../tests/test_delayed_trajectory_history.py#L73)。

该机制隔离了图片归属，没有等待或终止旧重规划线程，也没有解决所有线程退出与物理取消问题。

## 8. 生命周期与进度由谁管理

| 事情 | 当前负责方 |
| --- | --- |
| 连接设备、启动 ROS、初始化仿真和地图 | 平台入口或未来外部 Agent 系统 |
| 环境 reset、跨 episode 准备、关闭设备 | 外部系统；不会由 `LaViRAAgent.run` 隐式完成 |
| TODO、导航目标、回溯与任务内图片选择 | LaViRA 核心与推理模块 |
| 地图、局部路径、设备运动与传感器记录 | 运行时及其控制器/环境主机 |
| 任务排队与同一机器人互斥 | 外部系统；核心没有并发任务调度器 |
| 环境评测分数、EQA 数据集匹配 | 平台评测入口 |
| 模型连接和参数选择 | 组装入口提供，客户端使用 |

进度回调目前只发出 `stage/step/cycles`。它是同步函数调用，不是消息队列；上层回调应避免阻塞控制流程。阶段包括 started、observing、reasoning、executing、answering、finished，并不保证每轮都依次出现所有阶段。

取消使用 `threading.Event`。循环检查它，但模型请求、采图、规划和控制阻塞期间没有统一的抢占机制；旧 EQA 还可能在取消后回答问题。因此外部系统的设备停止操作和算法取消应分别管理。

下一步选读 [Go1/G1](04-go1-g1.md)、[Cobot](05-cobot.md)、[Habitat](06-habitat.md)，或直接看 [外部 Agent 接入](07-agent-integration.md)。返回：[索引](README.md)。
