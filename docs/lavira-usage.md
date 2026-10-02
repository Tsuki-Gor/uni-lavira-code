# 在外部程序中调用 LaViRA

Habitat、Go1、G1、Cobot 的 VLN、ObjectNav、EQA 已接入共享任务驱动，并完成离线对照与独立审核。真实模型服务、完整仿真和真机尚未验证，具体证据见 [验收说明](lavira-migration.md)。

在相应平台环境中，从仓库根目录执行 `pip install -e .`。已有平台启动命令继续使用，平台任务类负责组装共享核心。

## 完整任务调用

外部程序先准备好机器人或运行时；核心不创建、reset 或关闭环境。以下示例使用已存在的 `robot` 和原平台配置 `Config`，不会替你连接设备。

```python
from threading import Event

from lavira.agent import LaViRAAgent
from lavira.clients import LaViRAVisionClient
from lavira.reasoning.go1 import LaViRANavigationAPI
from lavira.runtime import Task
from navigation_runtime.camera_robot import CameraRobotRuntime

runtime = CameraRobotRuntime(
    robot,
    panorama_dir=Config.PANORAMA_DIR,
    directions=(("front", 0, 0), ("right", 90, -90), ("left", 270, 90)),
    safety_distance=1.5,
    require_panorama_success=True,
    recapture_delay=0.5,
    live_cameras=(("front", "camera1"), ("right", "camera2"), ("left", "camera4")),
)
reasoning = LaViRANavigationAPI(LaViRAVisionClient(Config), config=Config)
agent = LaViRAAgent(reasoning, profile="go1")
cancel = Event()
result = agent.run(
    Task("走到椅子旁边", kind="vln"),
    runtime,
    cancellation=cancel,
    on_progress=lambda event: print(event.stage, event.step),
)
print(result.reason, result.cancellation_requested, result.last_execution)
```

其他线程可调用 `cancel.set()` 请求取消；任务在原有循环检查点响应。该操作不表示控制器已制动，也不打断已有模型或轨迹阻塞调用。`physical_stop_confirmed` 没有证据时为 `None`。

`max_cycles` 默认不额外限制 Go1/G1 的循环次数，符合原入口；外部可在 `Task` 中明确设置非负预算。每次直接调用 `agent.run` 创建新的任务状态。外部仍负责保护运行时，避免多个任务同时控制同一个机器人。

Go1/G1 的新调用默认只使用本次任务生成的轨迹图片，保存在 `result.state.trajectory_images`。核心向 `navigate` 传入 `on_trajectory_image` 回调，执行器每生成一张轨迹图就回传路径；自定义执行器也需调用它，才能参与视觉记忆。Go1 的异步重规划沿用所属执行调用的回调，迟到图片不会进入另一个任务；返回的结果保存任务结束时的路径列表快照。执行记录文件仍由外部保存，不会因任务结束而删除。Go1/G1 旧任务入口显式设置 `Task(..., reuse_runtime_history=True)`，继续读取 `Panorama.history` 中跨任务累积的旧历史，保持原请求行为；新调用只有明确需要兼容时才启用该选项。

问答任务使用 `Task("椅子是什么颜色？", kind="eqa")`。导航按原规则结束后，核心调用运行时的 `observe_views()` 取得最终视图，向 LA 提问；`result.answer` 保存原解析得到的答案，`result.answer_error` 记录问答模型或解析异常。`reason` 仍描述导航结束原因，因此取消请求和答案可以同时存在，这是旧流程的行为。采图或导航本身抛出的异常仍按原路径传播。

Go1 的 `live_cameras` 保留前、右、左相机顺序；G1 不传该参数，刷新前相机。Cobot 在图像锁内取前、左、右的原有帧引用。问答不新增全景转动。核心不保存答案文件；Go1/G1 旧入口仍保存 `final_eqa_result.json`，外部 Agent 可自行处理结果。

## 让其他算法使用导航能力

`navigation_runtime` 不导入 LaViRA。其他算法可以取得 `Frame`、调用 `turn`，或者发出 `RelativeGoal(x, y)` / `PixelGoal(view, u, v, fallback)`。局部位置以米为单位，x 向前、y 向左；转角以度为单位，正值向左。像素目标的 fallback 必须由调用者明确提供。

`CameraRobotRuntime` 保留原控制器执行时读取实时深度的做法。`PixelGoal.view` 标识目标来自哪帧，但当前兼容执行尚不保证用该帧深度投影。不能把它描述成已经解决传感器同步问题。

Go1/G1 的观测保留原始毫米深度，`Frame.depth_unit="mm"`；Cobot 的采集回调已转换成米，标为 `"m"`。适配器填写已有数组视图的尺寸，仅有文件路径的全景不额外读取图像以补尺寸。

`ExecutionReport.execution_called` 只表示调用了轨迹执行函数；`arrived=None` 表示缺少到达证据。模型停止、导航成功和物理停止分别判断，不能相互推导。

Cobot 使用 `GimbalRobotRuntime` 处理全景采集、机械臂复位等待和像素导航。创建时传入机器人与本次记录目录；使用 `lavira.reasoning.cobot.LaViRANavigationAPI` 和 `profile="cobot"` 组装核心，调用时明确给出 `Task.max_cycles`（旧入口默认为 10）。`include_depth=True` 对应原 ObjectNav 的采图方式。该执行器返回旧控制器报告的 `arrived`，不增加独立的物理测量。

## Habitat 任务与地图执行

旧 Habitat 入口使用 `profile="habitat"` 和 `Task.max_steps`，预算按仿真动作计数。机器人使用 `max_cycles`；不能用 `max_steps` 代替它。Habitat 每次全景采集按原规则计入 12 步，可能越过预算后才在下一轮退出。EQA 未提交答案但已有航点全景时，会按原流程补答；`answer_from_history=False` 用于已有评测结果、不应再次触发补答的情况。

`MappedSimulatorRuntime` 接收外部准备好的观测、位姿以及提供建图、局部规划、传感器和记录能力的 host。`vlnce_baselines.habitat_environment.HabitatEnvironment` 提供现有 Habitat 能力，不创建或导入 LaViRA 客户端；旧评测器继承它并组装共享算法。核心不负责创建或 reset 该 host。

在已有 Habitat 环境中，将 `sim-code/habitat` 加入 Python 导入路径，使用现有 Habitat `config` 可以独立准备运行时。下面的 `algorithm` 是外部系统已选定、接受运行时的算法对象；这些初始化方法沿用原评测器，仍需要现有仿真数据、分割/建图依赖和所选局部控制器。

```python
import torch
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from vlnce_baselines.habitat_environment import HabitatEnvironment
from vlnce_baselines.utils.depth_utils import get_world_xz_from_pixel
from vlnce_baselines.utils.map_utils import calculate_displacement, collision_check_fmm
from navigation_runtime.mapped_simulator import MappedSimulatorRuntime

host = HabitatEnvironment(config, r2r=True)
host.initialize_controllers()
host._init_envs()
try:
    host._collect_val_traj()
    host._initialize_policy()
    if host.use_navdp:
        host.navdp_agent.reset_env(0)
    observations, pose = host._maps_initialization()  # 此处由外部 reset 环境。
    runtime = MappedSimulatorRuntime(
        host, observations, pose, tensor_api=torch, actions=HabitatSimActions,
        project_pixel=get_world_xz_from_pixel,
        displacement=calculate_displacement, collision_check=collision_check_fmm,
    )
    result = algorithm.run(runtime)
finally:
    host.envs.close()
```

跨 episode 使用时，外部调用 `host.reset()` 清理地图与控制状态，再按原流程准备下一次观测。它不重置算法；算法自己的任务状态由算法管理。`r2r` 和 `config.TASK_TYPE` 继续使用原数据集配置。

运行时通过 `observe_panorama`、`project`、`queue_navigation`、`execute_pending` 提供采集、像素投影、规划与单步执行；另一种算法可直接调用这些能力。`MapGoal` 的 x/y 是地图格坐标，`metres_per_cell` 是米/格，`reference_frame` 标明本次地图。与地图不匹配的目标会被拒绝。历史像素目标使用 `Frame` 携带的 RGB、深度、内参、基准位姿和 `pose_heading_offset`，不会换用当前位置的 RGB 或深度。

Habitat 深度标为 `normalized_0_1`，由运行时沿用原预处理转换；当前相机与地图参数仍来自现有 host。CPU 地图的位姿可能是共享引用，因此历史位姿尚不保证不可变，这项兼容例外见 [设计中的待决问题](lavira-core-design.md#待用户决定的行为问题)。

## 离线检查

在已有依赖可用的环境中，从仓库根目录执行：

```bash
python -m unittest discover -s tests -v
```

测试从 Git 的 `215e7ac` 读取原版作为对照，完整源码仓库需要保留该提交。测试替换外部模型和机器人，并使用实际图像编码；不会执行仓库中的真机控制测试。若指定临时目录，先设置 `TMPDIR`。没有 NumPy/Pillow/OpenCV 时部分组会跳过，跳过不算对应行为验收通过。
