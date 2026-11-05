# UMI LeRobot v3 数据支持

平台直接接入 `robot_type=UMI` 的 LeRobot v3 数据。图像可内嵌于 Parquet，
信号按 `head_pose`、`left_arm_pose`、`right_arm_pose`、对应四元数位姿和左右夹爪字段读取。
不需要先生成 `action/state`，也不会修改源数据或创建训练目标。

## 使用方式

1. 将数据集放在 Agent 的 allowed roots 下，等待目录同步；本地数据也可通过 Register 注册。
2. 数据列表显示机器人类型、LeRobot 存储版本及 Raw/Standard/Curated 阶段。
   DVT1/DVT2 不再作为列表标签。
3. 选择数据集，运行 Viewer cache。三路图像生成独立预览视频；遥测按头部、左手和右手分组。
   曲线使用 `head_pose.x` 等唯一名称；源数据与缓存 CSV 保留夹爪采集值，Viewer 显示时按 `/100` 归一化到 0–1。
   UMI 使用六个常驻按钮 Head3 / Head4 / Left3 / Left4 / Right3 / Right4，高亮表示开启，再次点击关闭。
   每个部位两组分别为 XYZ + Roll/Pitch/Yaw、
   XYZ + qw/qx/qy/qz。两组共用 `*_pose` 的 XYZ，只绘制一次，不显示 `*_quaternion_pose` 中重复的 XYZ。
   任一组开启时显示该部位 XYZ，两组都关闭时隐藏。夹爪和 Stage 使用独立的信号复选框。
   信号面板按 Head / Left / Right 三列排列，分量名称省略字段前缀，完整名称和单位保留在悬停提示中。
   每个位姿分量使用独立颜色；分组和单条曲线可独立切换，不依赖 `action/state`。
   UMI Viewer 的标题、分组和悬停提示使用英文。
4. Analysis 可直接统计 episode 数、帧数、时长与任务分布；原始信号统计取自
   `meta/stats.json`，代表完整数据集，不随页面筛选变化。没有统计值时显示未提供。
5. Split 按 episode 范围或任务筛选；Merge 仅接受机器人 profile、FPS、完整字段及已声明语义一致的源数据。
   UMI 合并必须使用 `dimension_policy=strict`，不支持截断、补维或与 DVT 数据混合。

单位和坐标系未声明时显示 `Not declared`。夹爪显示使用约定的 0–100 原始范围（小于 1.5 的值也除以 100），
不裁剪超出范围的采集值；不自动执行 `/100`、
位姿坐标变换或四元数重排。UMI 默认按时间等分生成 Stage，默认 5 段，可在操作表单调整。
已有合法阶段标注默认保留；点击 Stage 重新切分时才忽略旧阶段。
Viewer 使用 `Stage 1/N` 等时间段名称，不将时间等分结果解释为抓取、抬起等动作事件。

## 可用操作与兼容

- 通用查看、元数据分析、拆分和同结构合并适用于 UMI。
- Quality 自动提供 End-effector motion（末端位置/朝向跳变、四元数有效性、时间轴）和 S3，
  C3 可单独选择检查图像。无需 DVT 配置；检测结果是待复核候选，不自动删除数据。
  结果表的 Open in Viewer 可定位候选区间，返回 Episode review 保存人工决定。
  Server 与 Agent 都需升级，Agent 必须声明 `quality_checks` 包含 `eef`。
  阈值、单位限制和重跑规则见 [统一 Curation 文档](data_platform_unified_curation.md#quality-checks)。
- DVT 标准化、动作维度转换、平滑、动作状态数值编辑、基于关节/夹爪事件的 DVT Stage、动作状态异常检测和
  依赖特定策略输入的 Embedding 不适用于 UMI；页面与服务端均校验，Agent 会再次检查。
- DVT 操作默认统一使用 `h10w_dvt2_stage_v1`，包括原始记录为 DVT1 的数据；
  `--data-version DVT1` 或表单手动选择仍可使用旧规则。源数据 profile 不做批量改写。
- UMI 的 Stage 策略为 `time_equal_v1`，所有任务均仅按时间戳等分，不读取或生成 DVT `action/state`。
  UMI 不接受 DVT override，其他依赖 DVT 语义的操作保持不可用；未知机器人保留未知。
- `DatasetDataProfile` schema 2 允许 `legacy_data_version=null`，兼容读取 schema 1。
  数据摘要与 Viewer manifest 增加 `robot_type`、`data_profile`、`default_processing_profile`、`operation_capabilities`；
  manifest 还记录 `signal_columns_version=2`、列描述和 `fallback_stage_count`。
  更改策略或等分段数后，重建 CSV/manifest 并复用视频。
- 升级后再次准备缓存会重建不兼容的 CSV/manifest，复用兼容视频。源数据和已发布生命周期版本不做批量迁移。

## 数据写入与来源追踪

### Viewer display layouts

Viewer display semantics are resolved independently from processing profiles. A dataset containing
`state/action` vectors can represent either joint angles or end-effector poses:

- Joint vectors use their actual zero-based feature positions for the paired State / Action table
  and joint groups. Names such as `index_1` are labels, not array offsets. Additional action entries
  remain visible under Other Signals, with an empty State cell when the dimensions differ.
- Native UMI pose fields and named vectors from `UMI-GripperBody-Head` / `UMI-Gripper-Head` use the
  Head / Left / Right end-effector layout. Named vector components such as `left_qw` determine the
  mapping; vector length alone does not. State curves are solid and Action curves dashed, with
  source labels in the signal table. Quaternion-only layouts offer Head4 / Left4 / Right4 without
  synthesizing Euler angles. Packed gripper values retain their recorded scale; native scalar
  `left_gripper_pos` / `right_gripper_pos` keep their existing display normalization.
- Existing positional or qualified CSV labels remain readable. Old duplicate labels are qualified
  on read only when the full field order matches feature metadata. Otherwise the cache must be
  rebuilt. Source data, cache files, processing profiles, and published versions are not rewritten.
  If an older manifest omits robot type, a complete named Head/Left/Right XYZ + quaternion +
  gripper schema can still select end-effector display without assigning a robot identity.

This display fix requires updating Server A and reloading the Viewer. Existing Agent caches can be
reused when their manifest contains the complete feature metadata; no Agent upgrade is required.

Viewer cache 写入缓存目录，不修改源数据。Split/Merge 才会生成新数据集，默认采用带时间戳的
兄弟目录；输出存在时拒绝覆盖。dry-run 仅读取元数据和文件信息，不生成临时数据集。

对于 v3 内嵌图像，Split/Merge 直接按 `episode_index` 过滤共享 Parquet 分片，保留图像 bytes、
数值类型和维度名称，在 staging 中重建输出索引、episode/task 元数据和统计，验证后提交。
本路径不把内嵌图像转成视频。其他已有存储路径继续使用其原有实现。

源元数据、`export_manifest.json` 和其他来源 sidecar 保存在
`provenance/source_000/` 等来源目录下；输出 `meta/preprocess_split.json` 或
`meta/preprocess_merge.json` 记录 episode lineage，生命周期登记补齐源版本关联。
这些来源清单中的旧索引不代表输出位置。源中的已有符号链接按链接保存。

本地任务在输出缓存准备和登记完成后结束；远程任务还需要上传并绑定 Viewer cache。
任务失败会报告失败原因，不能将未完成的缓存当成成功结果。

## 升级顺序

1. 先在 Server A 使用现有 `sudo data-platform-update` 更新服务。
2. 检查 `/healthz` 返回 `data_profile_protocol: 3`。
3. 从同一工作区构建 Agent 包，再按照分布式部署文档升级 Agent。
4. 确认节点心跳中的 `data_profile_protocol` 为 3，并等待数据目录重新同步。
5. 选择少量 episode 验证预览，再运行拆分或合并。

旧 Agent 接收 UMI 或强制重新切分 Stage 的任务前会被中央服务以 409 拒绝，并提示升级。安装包位于 `dist/agent/`，
可用以下命令构建一个新的版本（版本名必须未使用）：

```bash
python3 scripts/build_data_platform_agent_bundle.py --output-dir dist/agent --version 0.1.0-r12-stage-defaults
```

测试覆盖共享分片、三路内嵌图像、重名维度、原始夹爪数值、缓存失效、旧 profile、
本地与远程接口、Agent 协议、拆分合并一致性、输出冲突、dry-run 和失败清理。


## Tracker 诊断、版本对比与候选回放

<a id="original-laundry-visualizations"></a>

### data_laundry 原版可视化（V0.3.43 源码候选）

`umi.cosmos_id@1.1.0` 与 `umi.replay@1.1.0` 复用 data_laundry 算法和原版查看器；
`1.0.0` 保持可执行，已有 Pipeline 不自动改版本。Jobs 的 **Open result** 和 Pipeline 已完成步骤的
**Original visualization** 打开 `/umi/results/<job_id>`。

- Cosmos 保留原版 `report.html`、`cosmos_id_viewer.html`、曲线/轨迹/缩放和区间跳转；HTTP 模式自动读取
  同任务的相机预览，下载后仍可选择附带的 `media` 目录。报告来自保存的候选，不重复推理。
- Replay 在同一次筛选执行中保存 NPZ/JSON，原版 Viser 从缓存播放机器人、目标和实际轨迹、误差及相机。
  计算中断时仅展示已执行帧；初始化失败的 episode 保留诊断，不生成虚假轨迹。
- 预览媒体保持原帧数、顺序和尺寸，图像重编码为 JPEG quality=90；数值轨迹保持计算精度。
  稀疏可播放集合保留原始 episode 编号。媒体目录标记 `VIEW_ONLY.json`，不注册成修复或训练数据集。

Server 用 `DATA_PLATFORM_UMI_VIEWER_RUNTIME` 指向管理员配置，示例为
[umi_viewer_runtime.example.json](umi_viewer_runtime.example.json)。单独创建 CPU Python 环境，安装
`deploy/data-platform/umi-viewer-requirements.lock`，提供固定 laundry 源码及 URDF/mesh assets；不自动下载依赖。
节点 Cosmos 出图使用 `report_python`，未配置时使用已安装 matplotlib 的 `cumotion_python`。
Server 依赖 Flask-Sock / websocket-client；部署生成器为 `/api/umi-viewer-sessions/<id>/socket` 配置 WebSocket 升级。

查看会话接口为 `POST /api/umi-viewer-sessions`（job_id、artifact），
`GET/DELETE /api/umi-viewer-sessions/<id>`，以及同会话的只读页面和 WebSocket。
仅允许任务所有者且仍有全部输入位置访问权的用户；HTTP 和 WebSocket 均检查权限与来源。
查看器只监听 loopback，CPU 子进程与 Web 进程隔离，单环境最多 4 个、单用户最多 2 个，空闲 30 分钟回收。
服务退出时回收子进程，重开读取原有结果，不占 GPU。

管理员可在 Agent runtime 配置 `native_result_cache_root` 导入已验证结果。缓存按输入文件指纹、参数、
算子版本、校准/模型资源和适配代码绑定，内容清单逐文件校验；网页不能指定缓存路径。
任务结果的 `result_reuse.provenance` 保留真实计算来源，外部试跑结果不得描述为历史 Pipeline 推理。
缓存复用仍检查同源诊断和平台审核前置条件。候选保持 `human_approvals=0`，本轮不提供正式数据集导出。

Server/Agent 需一起升级，兼容 CLI 为 0.3.41；无数据库或 Job 协议迁移。发布前须验收真实登录、
报告相机同步、Viser WebSocket/播放、跨用户拒绝及会话回收，部署和发布结果另行记录。

以下四个算子为 V0.3.42 源码候选，均通过现有 Pipeline / Job / Artifact 权限执行，输出报告或独立候选，不修改或发布源数据。

| 算子 | 输入和输出 | 范围 |
| --- | --- | --- |
| `umi.diagnose@1.0.0` | 一个数据位置 → JSON / HTML | 分别统计采集标记、Euler / rotvec 编码匹配、四元数、时间轴和轨迹跳变；指纹覆盖全部 metadata / Parquet |
| `umi.compare@1.0.0` | 两个同节点已注册位置 → 离线页面 / 全长视频 | 原生 v3 三路内嵌图像；episode、帧数、任务、时间戳和图像 bytes 必须一致；逐帧播放与曲线同步 |
| `umi.replay@1.0.0` | 一个位置，可选第二个对照 → 机器人跟随报告 | gravity + base / torso；手部默认 P95 7 cm / 10°；head 单独报告；第二版本复用基准的 world / hand / head 映射与躯干初始高度 |
| `umi.cosmos_id@1.0.0` | 同源诊断证据与已审核区间 → 独立 NPZ 候选 / 后检报告 | 30Hz、至少 49 连续帧、区间两侧有观测；不补帧、不跨 episode、不修夹爪；始终 `needs_review`、`human_approvals=0` |

Pipeline 首步按既有规则添加 Snapshot。对比或回放的第二个数据位置从 Comparison dataset 选择，并接受同样的数据访问检查。Cosmos 需在前序 `umi.diagnose` 后关联其 artifacts，并在网页确认目标区间后执行；该确认仅授权候选生成，不批准候选质量。插值修复仍使用原有 Curation 草稿审核。`targets` 中的 `start_frame/end_frame` 是 episode 内的半开区间 `[start,end)`，`part` 为 `left/right/head`。

采集的 `*_pose_is_valid=false` 只表示原始字段值，字段语义需采集端确认，不能当作逐帧删除或插值标签。零跳变、模型后检通过和机器人可跟随均不代表真实动作恢复或人工批准。

### 节点运行环境

首次安装请先完成 [data-laundry 源码获取、依赖安装与配置](data_laundry_setup.md)。
该流程提供固定源码安装脚本、CPU 查看器配置和 Agent 模板；不会自动下载模型权重或修改服务。

CPU 算子需要 NumPy、PyArrow、SciPy、Pillow，视频对比还需要 ffmpeg / ffprobe 和 DejaVu Sans。外部模型代码不导入 Server 主进程。Agent 仅在节点管理员配置 `DATA_PLATFORM_UMI_RUNTIME=/absolute/path/runtime.json` 后声明 replay / Cosmos 能力。配置示例见 [umi_runtime.example.json](umi_runtime.example.json)，路径均由管理员提供，网页参数不能指定解释器、模型或脚本路径。

运行环境固定 data_laundry `fd752156895c83ec75fde84796faea2513c626c7`，原生 Tracker 转换固定 OpenPI `74b73a0d65b5940e539c260f91a4ff9c7528c194`。部署时核验快照与 URDF 内容。Python 3.12 的 cuMotion 与 Cosmos 自身模型环境独立安装；执行时禁止自动安装依赖。

GPU 通过现有 FLORA 共享锁、进程检查和 73,000 MiB 显存门槛申请；节点应为该池预留设备，外部调度器也须遵守同一锁目录。取消或超时终止整个子进程组。转换数据集和请求位于 Job attempt 的私有工作目录，只有报告、候选和指纹进入 Artifact bundle；成功后清理大体积临时转换副本。

原生 v3 仅适配明确的 `h10w_tracker_rx90_rz180` 校准。转换副本标记 EVALUATION_ONLY，20 mm 夹爪二值化阈值仅保持评估格式兼容，物理语义未确认；不生成训练批准。v2.1 必须声明对应的具名 23D EE state、单位与四元数约定。其他布局明确拒绝，不依靠向量长度推断。

部署需同时升级 Server / Agent，并单独发布兼容 CLI 0.3.40。没有数据库或 Job 协议迁移；当前文档不代表已经安装、生产发布或推进 CLI latest。
