# 05 · Cobot：从机械臂全景到动态导航

Cobot 同样运行 `LaViRAAgent.run`，但它获取全景和执行导航的方式与 Go1/G1 不同：机械臂带动侧面相机扫描，模型选择方向和图像中的目标点，底盘通过实时里程计和 iPlanner 持续重规划。共享核心决定本轮做什么，`GimbalRobotRuntime` 把这些需求翻译成 Cobot 已有的观测与执行操作。

本页从一次“找到椅子”的任务展开。先看 [接口契约](03-runtime-contract.md) 能帮助区分“返回图像”“尝试执行”“达到几何目标”这三件不同的事。以下内容是源码解释，不是已完成的真机验证。

## 1. 应用如何把任务交给共享核心

[`main._build_robot`](../../real-world-code/cobot_magic/main.py#L201) 建立 ROS 节点并创建 `RobotController`，设备控制器负责相机、里程计、机械臂和 iPlanner 的初始化。特别注意 [`initialize_models`](../../real-world-code/cobot_magic/robot/robot_controller.py#L153)：这里初始化的是用于运动规划的 iPlanner，不是 LaViRA 的 LA/VA 模型客户端。

命令行的 [`run_object_nav`](../../real-world-code/cobot_magic/main.py#L144) 通过任务工厂传入指令与 `max_cycles`；[`run_headless`](../../real-world-code/cobot_magic/main.py#L175) 用相同工厂启动 VLN 或 EQA。Web 的 [`start_new_task`](../../real-world-code/cobot_magic/robot/nav_controller.py#L55) 实际默认构造 **ObjectNav**，然后在线程中调用 `task.run()`。读界面注释时，应以这里的 `TaskFactory("object_nav")` 为准。

[`VLNTask.__init__`](../../real-world-code/cobot_magic/tasks/vln.py#L20) 装配模型客户端、Cobot 推理配置和 `LaViRAAgent(profile="cobot")`。ObjectNav 和 EQA 继承这一包装类，分别设置任务种类；ObjectNav 还保留默认目标文本，见 [`object_nav.py`](../../real-world-code/cobot_magic/tasks/object_nav.py#L7) 与 [`eqa.py`](../../real-world-code/cobot_magic/tasks/eqa.py#L7)。

在 [`VLNTask.run`](../../real-world-code/cobot_magic/tasks/vln.py#L37) 中，包装类创建本次日志目录与 `GimbalRobotRuntime`，然后只提交一个完整任务：

```python
runtime = GimbalRobotRuntime(
    self.robot, session_root,
    include_depth=self.task_kind == "object_nav",
)
self.result = self.agent.run(
    Task(self.instruction, self.task_kind, max_cycles=max(0, self.max_cycles)),
    runtime, cancellation=self._cancellation,
)
```

这里的 `include_depth` 控制“转向后当前观测是否附带深度”，不代表 VLN/EQA 执行时不使用深度。底层 `navigate_dynamic` 始终可以读取机器人保存的实时深度。

## 2. 每轮先扫描，再让机械臂并行复位

核心从 step 0 开始每轮导航，调用 [`GimbalRobotRuntime.observe_panorama`](../../navigation_runtime/gimbal_robot.py#L25)。适配器建立 `cycle_00` 等目录，再调用机器人 [`collect_panoramic_images`](../../real-world-code/cobot_magic/robot/robot_controller.py#L401)。

这个函数先取前相机，再依次移动双臂到 `TARGET_SEQUENCE` 中的姿态，取左右相机图像。返回值包括三个部分：RGB 列表、深度列表、尚未启动的机械臂复位线程。列表有八个方向槽位，但背后槽位 4 不填；侧相机缺失时对应槽位也可能为空。只有前视图在这里附带前深度。

| 槽位 | 方向 | 相对前方角度 | 图像来源 |
|---|---|---|---|
| 0 | front | 0° | 固定前相机 |
| 1、2、3 | left_front / left / left_back | 45°、90°、135° | 左臂各扫描姿态 |
| 4 | behind | 180° | 当前不采集 |
| 5、6、7 | right_back / right / right_front | -135°、-90°、-45° | 右臂各扫描姿态 |

适配器收到返回值后启动复位线程，把存在的图像保存并包装成 `Frame`，返回 `Panorama(views, record_dir=cycle_dir)`。这样 LA 可以在机械臂回到初始姿态的同时读取刚取得的全景。核心只看到可用视图，不需要了解机械臂关节或 ROS 调用。

机械臂线程的收尾也留在 runtime：[`finish_observation`](../../navigation_runtime/gimbal_robot.py#L51) 会等待仍在运行的复位线程。正常后续取图会调用它；LA 要求停止时核心也会先调用它再退出。这里是显式收尾点，不意味着任何异常退出都已统一完成机械臂清理。

## 3. LA 决定下一方向，停止在这里生效

[`LaViRAAgent`](../../lavira/agent.py#L95) 将可用全景、原始指令、已选择目标的描述和当前 TODO 交给 [`decide_direction`](../../lavira/reasoning/cobot.py#L105)。该函数拼接提示词，调用 LA 并解析 JSON；请求参数是 `max_new_tokens=1024, temperature=0`。提示词仍属于 `lavira/reasoning/cobot_prompts.py`，设备适配器没有参与模型消息构造。

核心首先接收 `updated_todo_list`。如果 `action == "STOP"`，等待机械臂复位，然后返回 `model_stop`；这一轮不会继续执行 VA 或底盘导航。否则取得 `turn_direction/expected_landmark`，把本次选择记入 `visited_targets`，再调用 `runtime.turn(angle)`，见 [`agent.py`](../../lavira/agent.py#L96)。

这里的 `visited_targets` 是语言决策的历史记录：它在实际导航到达前就追加，不能直接当作“已到达位置”的真值。方向角由核心的 [`COBOT_ANGLES`](../../lavira/agent.py#L22) 映射；角度转成速度、多久结束转向，则由设备 [`rotate_angle`](../../real-world-code/cobot_magic/robot/robot_controller.py#L365) 负责。

虽然推理类还保留 `generate_initial_todo`，当前 Cobot 主循环没有单独调用它。首轮 TODO 从空字符串开始，由 `decide_direction` 的结果更新；不能仅看某个方法存在，就推断实际任务必定经过它。

LA 请求或解析失败时，目前保留原来的“向右寻找 open space”默认结果，见 [`cobot.py`](../../lavira/reasoning/cobot.py#L137)。这是旧策略的一部分，不能把它描述为解析失败必然停止。

## 4. VA 如何把转向后的图像变成像素目标

[`observe_current`](../../navigation_runtime/gimbal_robot.py#L55) 先等待两秒，并等待机械臂复位线程完成，再在图像锁下复制前视图；没有 RGB 时最多重试十轮。ObjectNav 还会复制前深度。结果包含 BGR 图像、图像宽高以及米制深度标记，原图保存在本轮目录。

随后核心调用 [`query_llm_bbox`](../../lavira/reasoning/cobot.py#L181)，提供指令、LA 的进展分析与地标、当前 RGB 及宽高。VA 返回 bbox 后，推理代码在坐标最大值不超过 1000 时按归一化坐标缩放，并裁剪到图像范围。这个判断会把某些真实的小像素坐标也当成归一化值，当前保持原行为。

在 [`agent.py`](../../lavira/agent.py#L166) 中，四元素 bbox 被转成底边中心：

```python
u, v = (x1 + x2) / 2, y2
goal = PixelGoal(view, u, v)
```

如果 bbox 缺失，核心使用图像中心；它不会因此自动跳过执行。Cobot 这条分支只消费 VA 的 bbox，并没有把 VA 的 `action == "STOP"` 合并成停止判断。因此 **LA STOP**、**VA 输出带 STOP 字段**、**没有 bbox** 是不同情况，不能仅凭推理模块顶部的概述认为它们都会停止底盘。

到这里，LaViRA 的输出已经是 `PixelGoal`。相机坐标怎样投影、机器人如何运动，都交给运行环境；换用另一种高层算法时，可以直接构造同一种目标。

## 5. 为什么 Cobot 深度是米，而 Go1/G1 是毫米

Cobot 的 [`front_depth_callback`](../../real-world-code/cobot_magic/robot/robot_controller.py#L264) 在设备入口就处理深度：非 `float32` 数组先转成 `float32` 再除以 1000，已经是 `float32` 的数组按现有约定直接使用。之后 `current_front_depth`、全景附带的前深度与 `GimbalRobotRuntime` 都按米描述。

这与 Go1/G1 保留原始毫米数组、采样单点时才换算的路径不同。不要因为都是 `Frame.depth` 就再次统一除以 1000；应当遵守 `depth_unit` 与设备约定。这里也没有新增对所有 ROS 编码格式的识别，源数据类型约定仍须由接入端满足。

执行时 [`GimbalRobotRuntime.navigate`](../../navigation_runtime/gimbal_robot.py#L74) 只接受 `PixelGoal`，调用：

```python
arrived = self.robot.navigate_dynamic(goal.u, goal.v, cycle_dir=self.cycle_dir)
return ExecutionReport(execution_called=True, arrived=arrived)
```

虽然 goal 附带 `Frame`，这段适配器只将像素传给设备。设备 [`pixel_to_robot_goal`](../../real-world-code/cobot_magic/robot/robot_controller.py#L296) 默认读取实时前深度和相机内参，经 [`utils.pixel_to_robot_goal`](../../real-world-code/cobot_magic/utils.py#L148) 得到 `(向前 Z, 向左 -x_cam)`。缺少深度或有效深度时，现有逻辑可能回退到前方约一米。因此当前并不保证像素选择时的 RGB 和执行时深度严格同步。

## 6. 一次 navigate 内还有一个连续控制循环

打开 [`RobotController.navigate_dynamic`](../../real-world-code/cobot_magic/robot/robot_controller.py#L468)，这里才是底盘动作真正发生的地方。它先等待里程计；五秒后仍没有位姿时，保留使用零位姿的默认行为。随后把像素投影得到的机器人局部目标，按当前位姿转换到世界平面，存为 `global_goal`。

底层控制大致按以下顺序循环：

1. 读取当前位置，计算与 `global_goal` 的距离，检查超时和到达条件。
2. 把当前 RGB、深度与位姿放入观测历史；满足间隔后启动 iPlanner 重规划线程。
3. 取最新可用世界轨迹，变换回当前机器人坐标系。
4. 用 `TrajectoryFollower` 计算线速度和角速度，接近目标时减速，发布 ROS 速度命令。
5. 退出后发布多次零速度命令，返回是否满足到达判据。

重规划线程 [`_replan_thread_func`](../../real-world-code/cobot_magic/robot/robot_controller.py#L164) 消费最近观测，将世界目标变换为该观测的局部目标，调用 `planner.plan`，再将轨迹变回世界坐标交给控制循环。这些都是环境执行能力，未调用 LA 或 VA。

返回的 `arrived=True` 表示循环满足 `0.01 < dist < Config.ARRIVAL_THRESHOLD` 的几何判据，不代表视觉目标身份已经核实。`execution_called=True` 只表示已进入这一执行函数；即便函数末尾发布了零速度，任务结果也没有新的物理停止传感器确认。

## 7. 几何到达如何影响 TODO 和任务结束

适配器返回后，核心读取 `report.arrived`。为真时调用 `mark_first_incomplete_task_completed`，随后检查 TODO 是否完成；满足条件则以 `todo_complete` 结束，见 [`agent.py`](../../lavira/agent.py#L188)。这正是执行环境向算法反馈、算法更新任务状态的交接点。

TODO 在这里仍然是字符串。实际 helper 会找到第一行包含 `[ ]` 的文本，把该行所有 `[ ]` 替换为 `[x]`，再追加完成标记；检查完成则要求字符串非空且不再含 `[ ]`，见 [`_helpers.py`](../../lavira/reasoning/_helpers.py#L433)。如果模型把多个复选项写在一行，一次到达可能一起标记它们。该旧行为当前没有被结构重构修正，不能把此函数名理解为严格的单项状态机。

未完成时，核心进入下一轮；默认包装类最多运行十轮，或者达到调用者提供的预算。其他出口包括 LA 停止和取消。`GimbalRobotRuntime.running` 固定返回 `True`，沿用旧循环主要检查取消和预算的方式，见 [`gimbal_robot.py`](../../navigation_runtime/gimbal_robot.py#L20)；这不是实时查询机器人是否可运行。

## 8. EQA 和取消为什么仍需看完整调用链

EQA 导航结束后由核心调用共享 [`answer_question`](../../lavira/reasoning/eqa.py#L10)，运行适配器的 [`observe_views`](../../navigation_runtime/gimbal_robot.py#L80) 直接读取前、左、右固定相机，不再移动机械臂扫描。回答请求沿用 Cobot 提示词并走 LA，答案和错误写到 `TaskResult`。Cobot 的旧 EQA 包装类没有另一份问答循环。

当前取消只设置任务 `Event`，核心在循环边界检查它；机械臂扫描、模型请求和 `navigate_dynamic` 没有共同接收此事件。尤其是后者主要按到达、超时和 ROS shutdown 退出，所以“请求取消”不能解释为“立即停止当前导航”。旧行为还包括导航取消后继续执行 EQA 问答，以及 Web 切换任务时只等待旧线程两秒。

以后把 Cobot 放到通用 Agent 系统中，应由外部管理 ROS 节点、设备生命周期、运动互斥和停止确认；共享核心接收 runtime 并管理导航任务状态。已有的 runtime 与 RobotController 可供其他算法复用，旧 TaskFactory 和平台推理包装则是 LaViRA 应用的装配入口。

对照源码时，可用 [`test_cobot_loop.py`](../../tests/test_cobot_loop.py) 查看旧循环与共享循环的离线行为比较，用 [`test_frame_metadata.py`](../../tests/test_frame_metadata.py) 查看深度与尺寸约定，用 [`test_environment_boundaries.py`](../../tests/test_environment_boundaries.py) 查看真实任务工厂进入共享核心的离线链路。这些测试没有运行真实机械臂、底盘、模型服务或完整机器人栈。

继续阅读：[Habitat 的观测、地图与控制器边界](06-habitat.md)；返回：[接口契约](03-runtime-contract.md)。
