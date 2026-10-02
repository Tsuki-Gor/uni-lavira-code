# 02 · 跟随一次导航任务阅读共享核心

假设外部系统已经连接机器人，并交来“走到椅子旁边”这一指令。LaViRA 要做的是反复观察、决定往哪里走、在图像中选定目标，再让外部执行能力完成移动。设备如何连接、相机如何订阅、仿真 episode 如何创建，均在这次调用之前处理。

本篇先沿 Go1/G1 的一次决策循环阅读，再说明 Cobot 与 Habitat 怎样进入同一个完整任务入口。具体运动实现放在平台篇。

## 1. 先区分任务驱动、推理和模型请求

这里有三层算法代码，职责并不相同。

| 层次 | 收到什么 | 做什么 | 返回什么 |
| --- | --- | --- | --- |
| 任务驱动 | 指令、任务类型、运行时、取消信号 | 决定何时观察、推理、执行、停止及问答收尾 | `TaskResult` |
| 推理策略 | 图像、TODO、历史、当前任务上下文 | 组织 prompt 和多模态消息，解析模型回答 | 方向、bbox、停止意图或答案 |
| 模型客户端 | 已组织好的消息和生成参数 | 选择 LA/VA 服务、发请求、重试与统计 | 原始文本及兼容格式的统计信息 |

LA 是 Language-Action，这里主要做方向、阶段目标、TODO、停止或回溯决策；VA 是 Vision-Action，主要把目标落实到图像中的区域。LA 也会读取图像，不能理解成“LA 只看文字、VA 才看图片”。RA 对应机器人动作落地，在现有实现中由投影、局部规划器与控制器完成，不要求第三个大模型。

先打开 [LaViRAAgent.run](../../lavira/agent.py#L38)，再看机器人 [language_action](../../lavira/reasoning/go1.py#L104) 与 [vision_action](../../lavira/reasoning/go1.py#L197)，最后才进入 [客户端 generate](../../lavira/clients.py#L128)。这样能先看到控制流，再理解每次模型请求的内容。

## 2. 一次调用建立哪些状态

核心入口是：

```python
result = agent.run(task, runtime, cancellation=cancel, on_progress=callback)
```

`task` 描述要做什么；`runtime` 提供观察和执行能力。入口会检查任务类型与预算，然后建立新的 `TaskState`。该状态包括步骤、循环次数、TODO、访问目标和历史图片。Go1/G1 的步骤从 1 开始，Cobot 与 Habitat 从 0 开始，沿用旧流程的编号方式。

每次调用创建状态不代表销毁模型客户端或重置设备。客户端的连接和调用统计可以继续复用。Habitat 还会调用推理对象的 `reset()`，其当前实现清理楼梯标志、TODO 与最终目标对象；不要把它扩展理解为环境 reset 或对对象所有字段的清零。见 [任务状态](../../lavira/runtime.py#L21)、[入口初始化](../../lavira/agent.py#L47)、[Habitat 推理 reset](../../lavira/reasoning/habitat.py#L1528)。

外层循环只在运行时仍可运行且取消信号尚未设置时继续。机器人用 `max_cycles` 限制循环次数，采图前就会递增，采图失败也消耗一轮；Habitat 的 cycles 则只在发起新的 LA 导航/回溯决策时递增，不计每次 VA 或停止复核。Habitat 还使用按运行时仿真步数计数的 `max_steps`。预算在循环检查点判断，不是对每次模型或控制调用设置硬超时。

## 3. 第一次观察：先看到周围，再决定方向

机器人路径调用 `runtime.observe_panorama(step)`，取得由方向名称映射到 `Frame` 的 `Panorama`。随后核心将图像编码并加上方向说明，交给推理模块。视角顺序与编码差异都沿用原实现。

`observe_panorama()` 返回 `None` 时，地面循环会返回 `observation_failed`；返回对象但缺少部分方向并不等于整次观察失败。Cobot 即使返回 `Panorama(views={})`，也不会由这个条件直接判为失败。Go1/G1 与 Cobot 对视图的组织方式也不同：前者主要由保存的文件编码，后者使用数组及最多七个实际可用方向的标签。见 [观察分支](../../lavira/agent.py#L82) 和 [_panorama_frames](../../lavira/agent.py#L228)。

初始 TODO 也没有被强行统一：G1 在首次 LA 导航请求之前单独生成 TODO；Go1 使用占位文本，让后续 LA 更新；Cobot 从空文本开始。这会影响请求次数与模型输入，因此属于保留行为的一部分。

## 4. LA 决定方向和阶段目标

对 Go1/G1，核心把以下内容传给 `language_action`：指令、TODO、本任务允许使用的轨迹图片、当前全景和步骤。推理模块负责组织消息并解析返回字典；核心再读其中的方向与停止字段。

以 Go1 为例，LA 的历史是最近若干轨迹计划图片，再接上当前全景与 TODO。G1 先筛选文件名包含 `replan_` 的图片，再取最近若干张。即使两者拿到同一份历史路径列表，最终送入模型的历史也未必相同。见 [Go1 历史选择](../../lavira/reasoning/go1.py#L124) 与 [G1 历史选择](../../lavira/reasoning/g1.py#L131)。

核心将 `front/left/right` 映射成 0/+90/−90 度；G1 另支持 `behind`，沿用 −180 度。随后调用 `runtime.turn`。运行时只收到转角，不需要理解模型字典、TODO 或“向椅子方向走”的语言。

停止字段在此先成为一个意图。Go1/G1 若后续 VA 仍给出有效 bbox，会保留最后一次接近动作；LA 说 STOP 并不必然在这一行立刻退出。见 [方向及停止解析](../../lavira/agent.py#L111)。

## 5. 转向后重新看前方，由 VA 选目标区域

转向后核心调用 `observe_current()`，取得当前图像，再组织 VA 请求。Go1 的 VA 输入包括阶段地标和进展分析；G1 使用方向与理由构成阶段目标。VA 的模型回复先在推理模块解析、按各端旧规则转换坐标，再返回核心。

因此，运行时收到的是像素目标，不是模型原文，也不需要猜测 `[0, 1000]` 坐标是不是已经换算成像素。Go1 的换算可从 [vision_action](../../lavira/reasoning/go1.py#L197) 继续向下读；G1 和 Cobot 的解析差异见各平台篇。

核心随后从 bbox 选一个代表行走目标的像素：Go1 用底边向上四分之一高度的位置，G1 用底边中心并限制纵坐标。若没有 bbox 且未停止，Go1/G1 保留朝前 1.5 米的默认目标。这个默认移动是旧行为，重构没有把它改成安全停止。

这一段最值得对照阅读的是 [目标构造](../../lavira/agent.py#L159)：`PixelGoal` 绑定当前 `Frame` 与像素；`RelativeGoal(1.5, 0.0)` 表示机器人局部坐标中的前方目标。后者也作为 Go1/G1 像素投影失败时的显式 fallback。

## 6. 发出目标，等待执行反馈，再更新记忆

核心通过 `runtime.navigate(goal, step, ...)` 交出目标。相机执行器负责读深度、投影、请求 iPlanner、截断安全距离并调用控制器。它返回的 `ExecutionReport` 描述本次调用，不由核心推测真实到达。

新完整任务调用默认将本任务图片列表的 `append` 作为 `on_trajectory_image` 回调。控制器每保存轨迹图就回传路径，后续 LA 再按平台规则选择图片。这个绑定属于执行调用本身；Go1 上个任务的重规划线程晚完成时，仍调用上个任务的回调，不会往新任务的列表写入。

旧 Go1/G1 入口则显式启用 `reuse_runtime_history=True`，每轮从 `Panorama.history` 获取原有全局历史，保持跨任务复用。两种方式没有清空机器人的记录文件。任务返回前会复制路径列表，使已返回结果不再随迟到回调改变。见 [执行回调](../../lavira/agent.py#L183)、[结果快照](../../lavira/agent.py#L222)、[迟到回调测试](../../tests/test_delayed_trajectory_history.py#L18)。

## 7. Cobot 为什么也在同一个循环里

Cobot 的大致流程仍是全景、LA、转向、当前图像、VA、导航，但有几个不同的退出点。LA 的 STOP 会先等待采图后的机械臂复位，再结束导航；正常导航后，只有控制器报告 `arrived=True` 才更新 TODO；TODO 全部完成也可结束。

Cobot 核心只从 VA 结果读取 bbox，不按 Go1/G1 的逻辑合并 VA STOP。方向历史保存的是文字目标记录，也不是 iPlanner 图片列表。上述分支在 [Cobot LA](../../lavira/agent.py#L96)、[VA 调用](../../lavira/agent.py#L147) 和 [执行后的 TODO](../../lavira/agent.py#L188) 都能直接找到。具体线程与执行过程见 [Cobot 导读](05-cobot.md)。

这些差异由一个共享任务循环的兼容分支承接，平台 `tasks/*.py` 不再自己维护整套 LA/VA 循环。

## 8. Habitat 如何进入同一个入口

Habitat 每次外层循环调用 `MappedTask.advance(runtime, progress)`，推进一次地图任务状态。它可能继续执行已有动作，也可能因为抵达航点、超时或需要停止确认而触发新一轮模型推理，不能把每个仿真动作都等同于一次 LA/VA。

`MappedTask` 属于算法侧：它维护目标、回溯、停止确认及航点历史。`MappedSimulatorRuntime` 属于执行侧：它采集图像、投影到地图、排队动作并更新地图。共享外层负责预算、取消和最终结果，地图任务负责更细的阶段推进。见 [分派位置](../../lavira/agent.py#L71)、[advance](../../lavira/reasoning/mapped_task.py#L53) 和 [Habitat 导读](06-habitat.md)。

## 9. 三类任务如何在导航之后分开

VLN 的 `instruction` 是路线或行为指令；ObjectNav 仍通过同一任务字段交目标文本；EQA 用问题作为任务内容，并在导航结束后补上回答环节。当前没有额外的结构化 `object_category` 或复杂任务树接口。

| 任务 | 机器人侧 | Habitat 侧 |
| --- | --- | --- |
| VLN | 使用平台已有导航请求与结束规则 | 使用 VLN prompt、航点与停止确认 |
| ObjectNav | 复用核心；平台入口保留原有差异 | 选择 OBJNAV prompt，沿用地图任务流程 |
| EQA | 导航结束后读取固定相机视图，用 LA 回答问题 | 停止确认及任务收尾中通过全景回答，交外部评分 |

机器人问答实现集中在 [answer_question](../../lavira/reasoning/eqa.py#L10)。相机顺序与编码由 profile 保留，模型或解析异常进入 `answer_error`。取消后仍可能执行问答，这是原有行为；观测阶段等位置抛出的异常也不保证被统一包成 `TaskResult`。

Habitat 使用 [MappedTask.finish](../../lavira/reasoning/mapped_task.py#L302) 和 [query_llm_oracle](../../lavira/reasoning/habitat.py#L1534)。答案最终由评测器传入指标计算，而不是由通用核心维护数据集分数。

## 10. 返回结果时，分别看什么

`reason` 回答“循环为什么结束”；`answer` 回答 EQA 的问题；`last_execution` 记录最后一次执行调用。它们不应该折叠成一个未经验证的 `success=True`。

| 字段或值 | 应怎样理解 |
| --- | --- |
| `model_stop` | 按模型与平台停止规则结束；不是环境成功证明 |
| `todo_complete` | Cobot 的任务列表判定完成 |
| `budget_exhausted` | 到循环检查点时预算已用尽 |
| `cancelled` / `cancellation_requested` | 前者是结束原因，后者记录信号状态；模型停止后也可能发现信号已设置 |
| `environment_finished` | 运行时不再继续；成功指标由外部判定 |
| `observation_failed` | 地面循环未拿到全景；不表示所有采图错误都用这一原因返回 |
| `physical_stop_confirmed=None` | 没有物理停止证据；当前核心没有自动把任何一次退出设置为已制动 |

源码见 [TaskResult](../../lavira/runtime.py#L40)。对取消或错误的准确处理方式，继续读 [接口篇](03-runtime-contract.md) 和 [验证篇](08-verification-and-limits.md)。

## 继续深入模型请求

机器人客户端 [LaViRAVisionClient](../../lavira/clients.py#L29) 返回 `(文本, 统计信息)`，Habitat 客户端 [LaViRA_API](../../lavira/clients.py#L237) 返回文本。两者的 token 参数、请求选项、重试与失败行为不同，因此抽到了同一模块但没有强行合成一种请求协议。

阅读 prompt 时，应沿着“调用方传了什么 → 推理模块选了哪个模板 → 客户端真正发了什么”继续跟踪。仅看到一个 prompt 常量或方法存在，不能确认主循环会使用它。比如 Go1 推理类有 `generate_initial_todo`，当前共享流程只在 G1 的首轮分支调用这一能力。

下一篇：[03 · 接口与责任边界](03-runtime-contract.md)。返回：[阅读索引](README.md)。
