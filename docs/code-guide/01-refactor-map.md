# 01 · 这次重构究竟改变了什么

这次重构把 Habitat、Go1、G1、Cobot 的 VLN、ObjectNav、EQA 接到同一个完整任务入口。调用者准备环境和执行能力，再交给 LaViRA 维护任务记忆、调用 LA/VA、导航和收尾；环境的连接、创建、reset、关闭仍由外部负责。AirSim 和室内无人机没有迁移。

本页先解释代码为什么这样移动，再给出可逐提交审核的地图。阅读时以 `215e7ac` 为旧行为基线，以 `31873f9` 为本次重构收尾版本。后续源码若继续变化，行号链接可能需要更新。

## 1. 原来的耦合具体发生在哪里

在原版机器人目录中，`tasks/vln.py` 一边调 LA/VA，一边直接采集相机、转向、获取轨迹、执行轨迹和更新 TODO。每个平台的 `robot/navigation_api.py` 又各自构建模型消息、选择图像和解析回复。要修改公共算法，维护者必须穿过这些平台目录，判断哪些重复应同步、哪些差异必须保留。

Habitat 的耦合更集中：原 `ZS_Evaluator_mp.py` 同时创建仿真环境、准备地图和控制器、运行导航策略、计算指标；原 `agent.py` 又把模型推理与可视化对象联系起来。让其他算法复用环境时，容易连带构造 LaViRA。

因此，本次抽取的关键是明确“谁决定目标”和“谁把目标执行出来”。例如 LA 决定前往左侧，属于算法；相机选择、转角执行、深度投影和 iPlanner 轨迹执行，属于环境提供的能力。二者都可以有复杂逻辑，但依赖方向应清楚。

要直接看旧文件，可在仓库根目录执行以下只读命令；无需切换工作区或恢复旧代码：

```bash
git show 215e7ac:real-world-code/unitree_go1/tasks/vln.py
git show 215e7ac:sim-code/habitat/vlnce_baselines/ZS_Evaluator_mp.py
```

## 2. 第一条边界：模型请求与平台配置分开

三台机器人的相同客户端集中到了 [LaViRAVisionClient](../../lavira/clients.py#L29)。共享类显式接收配置；原平台客户端保留原导入路径和无参构造方式，并在兼容入口读取平台 `Config`。这样已有脚本仍可运行，通用 Agent 也能提供自己的配置。

Habitat 的 [LaViRA_API](../../lavira/clients.py#L237) 同在共享包中，但保留独立实现。它与机器人客户端的 token 参数、timeout、额外请求参数、返回结构和失败处理不完全相同。强行合并会改变实际发给模型的请求，超出本次“保留行为”的范围。

两种客户端承担模型通信，都不创建机器人或仿真环境。保留两个请求实现，不意味着保留两套任务调度。

## 3. 第二条边界：推理进入算法包，平台差异留在明确的位置

消息构建、prompt、回复解析和模型侧记忆迁入 `lavira/reasoning/`。现有差异通过四个 profile 保留：Go1 的三方向与占位 TODO、G1 的初始 TODO 请求、Cobot 的七方向与文字历史、Habitat 的地图航点及回溯。

| 原来寻找的位置 | 现在主要阅读的位置 | 迁移后的责任 |
| --- | --- | --- |
| Go1 `robot/navigation_api.py`、`prompts.py` | [go1.py](../../lavira/reasoning/go1.py#L41)、[go1_prompts.py](../../lavira/reasoning/go1_prompts.py#L11) | 保留 Go1 的 LA/VA 请求和解析 |
| G1 `robot/navigation_api.py`、`prompts.py` | [g1.py](../../lavira/reasoning/g1.py#L51)、[g1_prompts.py](../../lavira/reasoning/g1_prompts.py#L12) | 保留 G1 初始 TODO、方向及 bbox 规则 |
| Cobot `robot/navigation_api.py`、`prompts.py` | [cobot.py](../../lavira/reasoning/cobot.py#L64)、[cobot_prompts.py](../../lavira/reasoning/cobot_prompts.py#L18) | 保留方向推理、地标和 bbox 规则 |
| Habitat `agent.py`、`prompts/` | [habitat.py](../../lavira/reasoning/habitat.py#L22)、[任务 prompt 选择](../../lavira/reasoning/habitat_prompts/__init__.py#L1) | TODO、导航/回溯判断、停止确认、问答 |
| 机器人 `tasks/eqa.py` 中的问答 | [answer_question](../../lavira/reasoning/eqa.py#L10) | 导航后取图、构建问答消息并返回答案/错误 |
| 平台 `utils.py` 中的编码与解析 | [_helpers.py](../../lavira/reasoning/_helpers.py#L114) | 集中已迁移的算法辅助逻辑，保留各端原规则 |

共享推理不再通过平台 `config` 或 ROS 对象寻找依赖。原平台推理类成为薄的配置包装，例如 [Go1 兼容 API](../../real-world-code/unitree_go1/robot/navigation_api.py#L7)。Habitat 的 [旧推理入口](../../sim-code/habitat/vlnce_baselines/agent.py#L10) 则读取模型配置、传入图像高度和绘图回调。

没有把四种 profile 压成一套新 prompt，也没有把图片编码的历史差异顺便清掉。相同职责集中后，后续可以逐项决定是否统一行为；现在的保留是可解释的兼容选择。

## 4. 第三条边界：完整任务由一个高层循环驱动

[LaViRAAgent.run](../../lavira/agent.py#L38) 是新的完整任务入口。它创建本次 `TaskState`，检查预算和取消，发布进度，组织观测、推理、执行、停止及 EQA 收尾，最后返回 `TaskResult`。唯一的高层任务 `while` 位于 [第 67 行](../../lavira/agent.py#L67)。

地面平台的差异是循环中的有限分支：例如 Cobot 的 LA STOP 可以立即结束；Go1/G1 在 STOP 带 bbox 时仍做最后接近。外部入口不再各维护这些导航循环。

Habitat 的一轮任务处理放在 [MappedTask.advance](../../lavira/reasoning/mapped_task.py#L53)：维护航点、超时、回溯、停止二次确认及连续历史。它每次处理一轮并返回是否结束，由同一个 `run` 决定是否继续。地图规划、动作队列和局部控制内部仍有自己的执行逻辑，不能把“唯一高层循环”理解成仓库只能出现一个循环。

任务、状态、进度、结果位于 [lavira/runtime.py](../../lavira/runtime.py#L10)。这里的 `runtime.py` 表达算法任务数据；实际观测与执行能力放在另一个包 `navigation_runtime/`，两者名字相近，阅读时要区分。

## 5. 第四条边界：执行代码可由其他算法调用

`navigation_runtime` 定义 [Frame、Panorama、目标和执行报告](../../navigation_runtime/__init__.py#L9)。执行器用像素目标、相对位置目标或地图目标驱动导航，不以 LA 原始回复、TODO 或 LaViRA 决策类作为动作接口。Habitat 另可接收 TODO 和模型输出等展示信息用于记录，但不会据此作导航决策，其他算法可以提供空记录信息。

| 旧流程中的代码 | 新位置 | 留在环境侧的内容 |
| --- | --- | --- |
| Go1/G1 task 中的采图、转向、投影、轨迹执行 | [CameraRobotRuntime](../../navigation_runtime/camera_robot.py#L12) | 调用外部机器人对象和原 iPlanner；保持安全距离及执行规则 |
| Cobot task 中的全景、机械臂复位协调、动态导航 | [GimbalRobotRuntime](../../navigation_runtime/gimbal_robot.py#L12) | 采图等待、复位线程、调用 `navigate_dynamic` |
| Habitat rollout 中的环境步进、地图目标投影和规划 | [MappedSimulatorRuntime](../../navigation_runtime/mapped_simulator.py#L15) | FMM/NavDP、动作队列、深度处理、地图坐标和环境反馈 |
| Habitat 评测器中的创建、reset、地图和指标 | [HabitatEnvironment](../../sim-code/habitat/vlnce_baselines/habitat_environment.py#L63) | 可独立构造的仿真主机，不创建 LaViRA 推理器 |

这些执行器仍针对当前相机布局、机器人控制器和 Habitat 主机结构。完成的是算法依赖的剥离；没有把所有硬件抽象成一个无差别驱动，也没有保证任意新平台只靠修改配置即可接入。

地面 `NavigationRuntime` 协议列出地面任务调用的方法；Habitat 还有地图投影、队列执行等额外能力。当前实现保留这种差异，阅读时应沿实际调用链检查，不能只看一个协议就认为它完整描述了所有平台。

## 6. 旧入口保留了什么

原 CLI、TaskFactory、任务注册方式和平台配置仍可使用。以 [Go1 VLNTask](../../real-world-code/unitree_go1/tasks/vln.py#L15) 为例，构造函数组装客户端、推理器、核心和执行器；`run()` 调核心后，把状态同步回旧字段。旧 ObjectNav/EQA 入口继续保留任务类型和答案保存等外层责任。

Habitat 的 [ZeroShotVlnEvaluatorMP](../../sim-code/habitat/vlnce_baselines/ZS_Evaluator_mp.py#L19) 继承环境主机，补上 LaViRA 装配；[rollout](../../sim-code/habitat/vlnce_baselines/ZS_Evaluator_mp.py#L57) 构造任务和地图运行时并调用核心。数据集遍历及评测组织仍在这个兼容入口。

机器人包也调整了导入方向：导入 `RobotController` 不会顺带导入导航算法；显式请求旧算法导出时再加载。旧 `utils.py` 中算法辅助入口采用调用时委托，避免仅为了控制硬件就强制加载模型依赖。

## 7. 两项恢复行为的修正，与一项已授权的接口行为选择

抽取后审核发现两处回归，分别单独提交：NavDP 的轨迹缓存写在 host 上，检查却误查 runtime，现已恢复原有缓存复用；Go1/G1 传出的深度本来是毫米，新 `Frame` 却默认标成米，现已改正单位和图像尺寸元数据。Cobot 深度保持米制。这些修正没有重新设计导航策略。

历史隔离是用户另行确认的接口选择。新 `Task` 默认 `reuse_runtime_history=False`，本次轨迹图片进入自己的 `TaskState.trajectory_images`；旧 Go1/G1 入口明确传 `True`，继续使用累计 iPlanner 历史，保持旧请求内容。见 [Task 定义](../../lavira/runtime.py#L10) 和 [旧入口传参](../../real-world-code/unitree_go1/tasks/vln.py#L45)。

执行端通过每次导航的 `on_trajectory_image` 回调报告图片，仍保留机器人自己的累计记录。Go1 重规划线程绑定产生该图片的执行回调；核心返回前复制图片列表，迟到回调不会串入下一任务或修改已经返回的列表。这个改动没有改变线程等待、控制策略或记录文件生命周期。

解析失败后的默认移动、取消延迟、旧到达证据不足等问题仍在。完整清单与验证边界见 [08 · 验证与限制](08-verification-and-limits.md)，不要把“重构完成”理解成这些问题已被修复。

## 8. 为什么仍在同一个仓库和安装包中

第一阶段需要让原工程继续装配和运行，因此执行器与环境代码暂留本仓库。依赖方向已经允许外部系统持有它们，并将 LaViRA 换成其他算法；真正搬仓库时还需要整理发布、安装和平台部署配置。

[pyproject.toml](../../pyproject.toml#L1) 将 `lavira` 与 `navigation_runtime` 一起打包为 `uni-lavira-core`，沿用 OpenAI SDK、NumPy、Pillow、OpenCV，没有增加 RPC 或动态插件系统。wheel 不包含三个机器人完整驱动目录，也不包含 Habitat 完整环境主机及仿真依赖。

这一区分决定了接入方式：安装共享包解决算法和执行适配器的代码分发，实际设备连接、数据集、地图服务、局部规划服务仍由部署方准备。临时 wheel 的独立导入验证，不等于完整机器人或仿真部署验证。

## 9. 按审核目的阅读 22 个本地提交

下表覆盖本次重构的 22 个提交，保留原顺序。它是审核导航；具体改动以 `git show <提交>` 为准。

| 提交 | 审核目的 | 重点看什么 |
| --- | --- | --- |
| `ab901b1` | 准备：补忽略规则 | 本地索引、日志、构建等不进入版本库 |
| `1c49fc2` | 准备：保存原 HTML | 保留讨论前原始方案 |
| `dccbbe6` | 准备：修订方案和边界文档 | 先定义行为保留、范围和验收 |
| `c592b4e` | 客户端：合并机器人请求 | 显式配置、旧客户端包装、最小打包 |
| `378884e` | 客户端：抽取 Habitat 请求 | 保留其独立协议和错误返回 |
| `84e9f84` | 推理：抽取三种地面 profile | prompt、消息、解析不随搬迁改变 |
| `745d541` | 驱动：Go1/G1 进入共享循环 | 完整任务入口和相机执行边界 |
| `e29af1a` | 驱动：Cobot 进入共享循环 | 七方向、复位协调、TODO 结束 |
| `1cba202` | 推理：Habitat 脱离仿真导入 | 显式图像高度及可视化回调 |
| `733d700` | 任务：地面 EQA 在核心收尾 | 返回答案，旧入口保存结果 |
| `25178e9` | 驱动：Habitat 进入共享核心 | MappedTask 与地图执行器分工 |
| `d6ae86f` | 依赖：机器人导入不加载算法 | 兼容导出延迟加载 |
| `eb23007` | 生命周期：抽出 HabitatEnvironment | 独立环境构造和 reset |
| `92dd54d` | 清理：移除重复算法辅助实现 | 旧 utils 委托共享代码 |
| `02928e7` | 回归修正：恢复 NavDP 缓存 | 缓存所属对象检查与长轨迹测试 |
| `da8866e` | 验证：检查旧 TaskFactory 装配 | 真实任务入口组装核心 |
| `bdefe9f` | 文档：记录迁移及验收 | 使用方式和当时验证范围 |
| `c56f868` | 文档：方案转换为 Markdown | 后续直接阅读正文，HTML 留参考 |
| `c12348c` | 验证：旧入口跑完真实核心链路 | 仅替换 SDK、机器人和等待 |
| `9018976` | 回归修正：观测单位和尺寸 | 毫米/米以及各图像数组尺寸 |
| `9ed2fbf` | 已授权行为：任务历史隔离 | 新默认、旧兼容、迟到回调归属 |
| `31873f9` | 文档：更新最终边界证据 | 19 组测试、wheel 和保留限制 |

建议先读客户端和任务边界，再看对应平台的切流提交；最后单独看两项回归修正与历史隔离提交。这样能区分“代码换了位置”“恢复原行为”和“经确认改变新接口默认值”三种改动。

下一篇：[02 · 一次任务如何运行](02-task-lifecycle.md)。返回：[阅读索引](README.md)。
