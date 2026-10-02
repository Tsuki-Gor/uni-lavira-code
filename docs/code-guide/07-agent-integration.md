# 07 · 怎样放进通用具身 Agent 系统

未来的通用 Agent 系统可以管理机器人与仿真环境，选择导航算法，再提交一次导航任务。当前仓库已经提供进程内的分界：外部准备运行时，LaViRA 完成一次任务并返回结果。尚未提供远程服务、算法注册中心或跨进程任务协议。

本篇说明现有代码能怎样接入，以及环境代码以后迁出时的实际工作范围。示例是接入方式说明，没有在真实模型、仿真或机器人上执行。

## 1. 外部系统先做哪几件事

外部系统先选中设备或仿真 episode，完成连接、配置与初始化。然后组装模型客户端、推理策略和执行器，最后调用 LaViRA。任务结束后，由外部系统判断结果、安排下一任务，并决定复用还是关闭环境。

```text
外部系统：选环境、初始化、取得独占使用权
    ↓
组装：客户端 + 推理 profile + 已准备好的 runtime
    ↓
LaViRA：完整任务调用，内部循环观察/推理/执行
    ↓
外部系统：读取结果、评估、释放或复用环境
```

当前机器人入口的组装参考 [Go1 VLNTask](../../real-world-code/unitree_go1/tasks/vln.py#L15) 与 [G1 VLNTask](../../real-world-code/unitree_g1/tasks/vln.py#L15)；Habitat 的组装参考 [rollout](../../sim-code/habitat/vlnce_baselines/ZS_Evaluator_mp.py#L57)。复用现有代码时，先分清组装和任务循环，就能避免外部系统再次复制一套 LA/VA pipeline。

## 2. 调用 LaViRA 的最小封装

下面的函数接收已经准备好的对象，不创建机器人或仿真环境；运行时方法仍可能产生真实动作。这只是应用侧封装示例，不是仓库新增的一层框架。

```python
from lavira.agent import LaViRAAgent
from lavira.runtime import Task


def run_navigation(reasoning, profile, runtime, instruction, *,
                   kind="vln", cancel=None, on_progress=None):
    agent = LaViRAAgent(reasoning, profile=profile)
    task = Task(instruction, kind=kind)
    return agent.run(task, runtime, cancellation=cancel,
                     on_progress=on_progress)
```

机器人按需要给 Task 设置 `max_cycles`，Habitat 给 `max_steps`；不要把它们当成同一种步数。Go1/G1 新调用默认独立轨迹历史，只有明确要复现旧入口的跨任务输入时才设置 `reuse_runtime_history=True`。

串行任务可以复用已经创建的客户端。并行任务需要各自持有 agent、reasoning 和独立可用的运行时；即使新建 agent，共享同一个 Habitat reasoning 仍会交叉修改任务类型、prompt、TODO 和楼梯状态。客户端若要并发复用，也需另行检查线程安全，当前实现没有为这些对象提供并发保证或设备调度锁。Task.id 只是传入标识，没有自动实现任务幂等、存储或恢复。

完整 Go1 组装例子已在 [使用说明](../lavira-usage.md) 给出。它使用平台配置对象注入客户端和推理层，不让共享包直接导入平台 `Config`。

## 3. 四种 profile 实际需要什么

| 选择 | 推理对象 | 环境能力 | 组装时需留意 |
| --- | --- | --- | --- |
| `go1` | `lavira.reasoning.go1.LaViRANavigationAPI` | CameraRobotRuntime | 三方向、二次采图、Go1 安全距离和固定相机顺序 |
| `g1` | `lavira.reasoning.g1.LaViRANavigationAPI` | CameraRobotRuntime | 四方向、首轮 TODO、G1 安全距离 |
| `cobot` | `lavira.reasoning.cobot.LaViRANavigationAPI` | GimbalRobotRuntime | 会话记录目录、复位线程与原任务循环预算 |
| `habitat` | `lavira.reasoning.habitat.VLMReasoningAgent` | MappedSimulatorRuntime 与已初始化 host | 图像高度、历史配置、地图与动作 API、数据集/任务类型 |

profile 当前是兼容策略选择，不是自动发现插件的名称。它与推理对象、运行时能力必须配套；核心不会根据一个字符串下载依赖或自动建立环境。

Habitat 的 host 初始化涉及地图、局部控制器、传感器、评测数据和可视化。`HabitatEnvironment` 已不依赖 LaViRA，但仍然是当前 Habitat 工程的环境主机；它不是只用标准库就能创建的轻量通用模拟器。外部管理过程参考 [使用说明中的初始化示例](../lavira-usage.md) 与 [Habitat 源码导读](06-habitat.md)。

## 4. 外部系统怎样接收进度、取消和结果

进度回调适合把阶段与步数交给上层展示，但目前它直接在调用线程执行。耗时日志、网络通知或持久化若要异步处理，应由外部系统承担，不能假定核心已有后台事件总线。

其他线程可以设置传入的 Event 请求取消。该信号在现有检查点处理；若上层要求立即制动，需要通过自己管理的设备能力处理，不能把 `cancel.set()` 的返回当成制动完成。

一次正常返回后，上层至少分别读取：

```python
navigation_reason = result.reason
cancel_requested = result.cancellation_requested
execution_evidence = result.last_execution
answer, answer_error = result.answer, result.answer_error
```

如果方法抛出异常，当前核心没有统一的错误 Result 包装；外部程序应按自己的任务生命周期处理异常及资源释放。对于 Habitat，环境指标来自运行时 infos 和主机评测逻辑；对于机器人，是否真正到达也需要对应设备的证据。

## 5. 如果把 LaViRA 换成另一个算法

外部系统可以保留已准备好的运行时，让另一个算法取观测并选择目标。新算法不必生成 LaViRA 的 LA/VA 字典，也不必实例化 `LaViRAAgent`。

以下是相机机器人路径的能力调用示意，前提是算法已经选定当前图像中的目标像素：

```python
from navigation_runtime import PixelGoal


def execute_selected_pixel(runtime, view, u, v, step):
    # view 应当是算法选择该像素时使用的观测。
    return runtime.navigate(PixelGoal(view, u, v), step)
```

这里没有提供默认移动 fallback；是否需要 fallback 由调用算法明确决定。CameraRobotRuntime 的现有投影仍会读取实时深度，新算法必须理解这项兼容限制。Cobot 同样接收 PixelGoal，但走自己的动态导航；不应因此推断二者能接收相同的所有目标类型。

Habitat 则需要另一种调用组合：检查 `runtime.running` → `prepare_step(target, {})` 准备本轮观测与记录 → `project` 成 MapGoal → `queue_navigation` → `execute_pending` 并重复推进。空字典表示不提供 LaViRA 专用展示信息；有未完成动作时，还需按 `busy` 状态继续执行队列。全景采集本身会推进仿真动作计数。替换算法应使用这些实际能力，不能假设地面 `navigate` 示例可原样用于 Habitat。

现有离线证据可沿 [机器人替代算法检查](../../tests/test_task_history.py#L15) 与 [Habitat 替代算法检查](../../tests/test_habitat_loop.py#L68) 阅读。它们证明能力层不需要 LaViRA 导入，尚不证明另一个完整导航算法已在真实设备上部署。

## 6. 环境代码以后迁出时，哪些东西一起走

当前分离了依赖方向，但打包仍把 `lavira` 与 `navigation_runtime` 放进同一个 wheel，见 [pyproject.toml](../../pyproject.toml#L1)。平台源码和 Habitat 工程没有全部进入这个共享 wheel。

| 代码或资源 | 迁移时归属 | 仍需处理的具体事项 |
| --- | --- | --- |
| `lavira/` | 导航算法包 | 保留推理、prompt、任务状态及模型客户端；由外部注入配置 |
| `navigation_runtime/` | 可由外部环境管理层维护 | 若独立发布，需要分开打包并让算法依赖共享契约所在包 |
| 机器人控制器、iPlanner 客户端及传感器代码 | 设备管理侧 | 保留 SDK/ROS 依赖、连接参数、校准与运动控制 |
| HabitatEnvironment、地图工具、局部规划器、配置与数据 | 仿真管理侧 | 保留目录资源、数据集、权重和原环境依赖关系 |
| 旧 `tasks/`、评测器的算法组装部分 | 应用或外部 Agent 的组合层 | 该层仍需同时知道 LaViRA 与具体运行时，按调用场景重组 |
| 日志、图片、模型凭据、任务结果存储 | 外部部署配置 | 路径和生命周期由部署方接管，不随 Python 模块搬迁自动完成 |

因此，“未来可以迁出”表示已经有可辨认的职责和依赖方向，并不意味着复制一个文件夹就完成所有部署工作。尤其是历史图片仍用路径传递，调用模型时必须还能读到原记录；图片路径列表的任务隔离不等于图片内容已复制或永久保存。

本次没有加入包发现、RPC 服务或统一设备注册中心。这些属于将来外部系统的选择，不能把当前进程内接口直接描述成已经完成分布式算法插件接入。

下一篇：[08 · 验证与限制](08-verification-and-limits.md)。返回：[索引](README.md)。
