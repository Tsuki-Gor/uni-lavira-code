# 在外部程序中调用 LaViRA

当前 Go1/G1/Cobot 的 VLN、ObjectNav、EQA 已接入共享任务驱动。Habitat 客户端与推理已独立，高层循环仍待迁移，进度见 [验收说明](lavira-migration.md)。

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

问答任务使用 `Task("椅子是什么颜色？", kind="eqa")`。导航按原规则结束后，核心调用运行时的 `observe_views()` 取得最终视图，向 LA 提问；`result.answer` 保存原解析得到的答案，`result.answer_error` 记录问答模型或解析异常。`reason` 仍描述导航结束原因，因此取消请求和答案可以同时存在，这是旧流程的行为。采图或导航本身抛出的异常仍按原路径传播。

Go1 的 `live_cameras` 保留前、右、左相机顺序；G1 不传该参数，刷新前相机。Cobot 在图像锁内取前、左、右的原有帧引用。问答不新增全景转动。核心不保存答案文件；Go1/G1 旧入口仍保存 `final_eqa_result.json`，外部 Agent 可自行处理结果。

## 让其他算法使用导航能力

`navigation_runtime` 不导入 LaViRA。其他算法可以取得 `Frame`、调用 `turn`，或者发出 `RelativeGoal(x, y)` / `PixelGoal(view, u, v, fallback)`。局部位置以米为单位，x 向前、y 向左；转角以度为单位，正值向左。像素目标的 fallback 必须由调用者明确提供。

`CameraRobotRuntime` 保留原控制器执行时读取实时深度的做法。`PixelGoal.view` 标识目标来自哪帧，但当前兼容执行尚不保证用该帧深度投影。不能把它描述成已经解决传感器同步问题。

`ExecutionReport.execution_called` 只表示调用了轨迹执行函数；`arrived=None` 表示缺少到达证据。模型停止、导航成功和物理停止分别判断，不能相互推导。

Cobot 使用 `GimbalRobotRuntime` 处理全景采集、机械臂复位等待和像素导航。创建时传入机器人与本次记录目录；使用 `lavira.reasoning.cobot.LaViRANavigationAPI` 和 `profile="cobot"` 组装核心，调用时明确给出 `Task.max_cycles`（旧入口默认为 10）。`include_depth=True` 对应原 ObjectNav 的采图方式。该执行器返回旧控制器报告的 `arrived`，不增加独立的物理测量。

## 离线检查

在已有依赖可用的环境中，从仓库根目录执行：

```bash
python -m unittest discover -s tests -v
```

测试从 Git 的 `215e7ac` 读取原版作为对照，完整源码仓库需要保留该提交。测试替换外部模型和机器人，并使用实际图像编码；不会执行仓库中的真机控制测试。若指定临时目录，先设置 `TMPDIR`。没有 NumPy/Pillow/OpenCV 时部分组会跳过，跳过不算对应行为验收通过。
