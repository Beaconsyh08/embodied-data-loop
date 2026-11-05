# 从源码准备 data-laundry / UMI 环境

本指南对应 `data-platform` 分支。平台通过独立进程调用固定版本的 data-laundry，
不把外部算法安装到 Web 主进程，不在任务执行时自动下载依赖或模型。
数据格式、审核和结果发布规则仍以 [UMI 数据支持](umi_dataset_support.md) 为准。

## 为什么使用安装脚本，而不是 submodule

采用 **外部 Git checkout + 固定 commit + 上游依赖锁 + 独立环境**。
平台 clone 不依赖其他源码仓库；只有启用 laundry 功能的机器才需要获取它。
submodule 能固定源码，但不能解决外部仓库权限、不同 Python 版本、CUDA 和模型权重的问题。
暂不增加 `.gitmodules`，也不复制外部实现到本仓库。

data-laundry 是单独授权的可选依赖，本仓库不再分发其源码。
下载本公开仓库不等于获得外部源码访问权限或再分发许可；使用者须向依赖维护者获取仓库地址和访问许可，
或由维护者提供的**已授权、包含指定 commit 的 Git 镜像**。模型同样须按各自授权准备。

| 能力 | 本指南准备的内容 | 还需要什么 |
| --- | --- | --- |
| laundry CLI、CPU 报告、原版 Viser | 固定源码、上游锁定的 CPU 依赖、可视模型和查看器配置 | 查看器播放需要已完成任务的 replay bundle |
| 平台 `umi.diagnose` / `umi.compare` | 平台已有实现 | 平台 Python 环境、ffmpeg/ffprobe、DejaVu Sans；不需要 Cosmos 权重 |
| 平台 `umi.replay` | 上述 CPU 环境、配置模板、cuMotion 安装步骤 | NVIDIA 驱动、Python 3.12 cuMotion、数据转换代码及匹配的 Tracker 校准 URDF |
| 平台 `umi.cosmos_id` | 固定适配器、配置模板和检查命令 | 匹配的 Cosmos 框架/独立模型环境、完整权重与归一化资源、GPU |

完整 GPU 流程面向 Linux x86_64/aarch64；固定 cuMotion wheel 要求 glibc ≥ 2.35。
CPU 流程以下列 Linux / Python 3.11 步骤为验证基线。

## 1. 获取平台并准备基本工具

先安装 Git、uv，以及 ffmpeg、ffprobe、DejaVu Sans。Ubuntu/Debian 可使用：

```bash
sudo apt-get update
sudo apt-get install -y git ffmpeg fonts-dejavu-core pipx
pipx install uv
pipx ensurepath
```

重新打开终端，确认 `git --version`、`uv --version`、`ffmpeg -version` 可用，然后：

```bash
git clone --branch data-platform https://github.com/Beaconsyh08/embodied-data-loop.git
cd embodied-data-loop
export PLATFORM_ROOT="$(pwd)"
export UMI_ROOT="$HOME/.local/share/embodied-data-loop/umi-fd75215"
```

`UMI_ROOT` 必须是平台 checkout 之外的**新目录**，不要使用数据集目录、现有模型目录或现有安装目录。
安装脚本拒绝覆盖已有目录；失败时保留该目录便于检查，重试可选择新的空路径。
不要移动安装好的目录，虚拟环境的命令入口包含绝对路径。

## 2. 自动获取固定源码和 CPU 环境

```bash
read -r -p "Authorized data-laundry Git URL or mirror: " LAUNDRY_REPOSITORY
python3 deploy/data-platform/setup-umi.py install --root "$UMI_ROOT" \
  --laundry-repository "${LAUNDRY_REPOSITORY:?Repository is required}"
```

脚本执行以下步骤：

1. 从显式提供的已授权仓库地址或 Git 镜像获取源码，
   detached checkout 到 `fd752156895c83ec75fde84796faea2513c626c7`。
2. 用 `uv sync --frozen --no-dev --extra viser --python 3.11` 安装上游 `uv.lock`，
   不重新解析依赖，也不安装 cuMotion 或下载 Cosmos 权重。
3. 检查实际 Git commit、工作区修改、Viser 所用的 visual meshes、报告和查看器模块导入，运行 `laundry list`。
4. 生成以下文件：

```text
$UMI_ROOT/
  laundry/                       固定版本的外部源码
    .venv/                       独立 CPU 查看器 / report 环境
  viewer-runtime.json            可供 Server 使用的 CPU 查看器配置
  agent-runtime.template.json    GPU 能力配置模板，含 /REQUIRED/ 路径及空 GPU 列表
  installation.json              成功凭据：源码 commit、上游锁 SHA-256、Python 版本、检查范围
```

上游 Git、包索引和 Python 下载源必须可访问。安装器不保存或接收 URL 内的密码/token；
使用 SSH agent 或 Git credential helper。有授权镜像时：

```bash
python3 deploy/data-platform/setup-umi.py install \
  --root "$UMI_ROOT" --laundry-repository /absolute/path/to/authorized-git-mirror
```

也可指定不含凭据的镜像 URL。镜像必须保留上述 commit；不要用另一版本替代后只修改 JSON 中的版本字段。
`--python /absolute/path/to/python3.11` 可复用预装解释器，避免 uv 下载 Python。
锁文件记录的包索引/文件地址也需要网络可达；离线安装须预先准备完整 uv 缓存，单有 Git 镜像不够。

再次验证，不执行数据修改或模型推理：

```bash
python3 deploy/data-platform/setup-umi.py check \
  --runtime "$UMI_ROOT/viewer-runtime.json" --mode viewer
"$UMI_ROOT/laundry/.venv/bin/laundry" list
```

`laundry list` 显示的是上游完整 CLI 流程，不表示这些流程均已注册到平台。
平台接入的算子仍是 `umi.diagnose`、`umi.compare`、`umi.replay`、`umi.cosmos_id`。
直接运行上游 `laundry run` 不经过平台账号、Pipeline 审核或 Job 审计；平台数据应从平台页面提交。

## 3. 平台与原生 v3 转换环境

基础平台安装/启动见 [README](../README.md)。如果尚未安装其 Python 环境，可在平台目录运行：

```bash
uv sync --python 3.11 --extra data-platform-server --extra data-platform-agent
```

这会安装平台自身依赖（包括 PyTorch），不会把 Cosmos 模型权重包含进去。
对于原生 Tracker v3，转换器来自另一个已授权的固定 OpenPI checkout：

```bash
read -r -p "Authorized converter Git URL or mirror: " CONVERTER_REPOSITORY
GIT_LFS_SKIP_SMUDGE=1 git clone --no-checkout -- \
  "${CONVERTER_REPOSITORY:?Repository is required}" "$UMI_ROOT/openpi"
GIT_LFS_SKIP_SMUDGE=1 git -C "$UMI_ROOT/openpi" checkout --detach \
  74b73a0d65b5940e539c260f91a4ff9c7528c194

env -u PYTHONHOME PYTHONPATH="$UMI_ROOT/openpi/src:$PLATFORM_ROOT" \
  "$PLATFORM_ROOT/.venv/bin/python" \
  "$UMI_ROOT/openpi/data_tool/convert_dual_hand_head_to_umi_v21.py" --help
```

这里只复用固定转换脚本和平台提供的 LeRobot 实现，不安装完整 OpenPI 训练环境，也不下载其 LFS 模型。
OpenPI 仓库访问限制与 laundry 相同，可使用保留该 commit 的授权镜像。
当前平台运行配置即使输入已经是 v2.1，也要求填齐转换器路径；检查器会验证转换源码和 `--help` 可执行。

## 4. 机器人回放：单独安装 cuMotion

在计划执行回放的计算节点执行，而非 Web 服务进程内：

```bash
env -u PYTHONPATH -u PYTHONHOME -u LAUNDRY_CUMOTION_PYTHON \
  -u LAUNDRY_CUMOTION_NO_INSTALL \
  "$UMI_ROOT/laundry/.venv/bin/laundry" setup-cumotion

env -u PYTHONPATH -u PYTHONHOME -u LAUNDRY_CUMOTION_PYTHON \
  LAUNDRY_CUMOTION_NO_INSTALL=1 \
  "$UMI_ROOT/laundry/.venv/bin/laundry" setup-cumotion --check
```

固定上游安装器创建 `laundry/.venv_cumotion`（Python 3.12），校验并提取
`isaacsim-robot-motion==6.0.1.0` wheel 中的 cuMotion 部分；它不安装整套 Isaac Sim。
cuMotion wheel 有固定 SHA-256；其 Python 依赖按固定源码中的上游安装逻辑解析，
**并非本 CPU 环境的 frozen lock**。安装成功后应保留环境和安装标记，部署时不要随任务更新它。
此步骤需要网络、磁盘及兼容的 NVIDIA 驱动；导入检查通过不等于真实机器人轨迹回放已验收。

原版可视化使用的 H10W visual meshes 已在固定 laundry 源码中。
原生 Tracker 转换需要带 TrackerF 标定坐标系的 URDF。上述固定 OpenPI checkout 包含
`data/H10W_DVT2_URDF/H10W_DVT2_TekkenD_with_Camera_2_TrackerF_Ry90.urdf`，
只有机器人和采集标定确实匹配时才使用它；其他硬件须由配置维护者提供匹配资源并适配转换。
不能把 laundry bundled 的可视化 URDF 随意代替为采集校准文件。

## 5. Cosmos：单独准备模型资源

CPU 安装不会下载或伪造以下资源。需要由模型发布方提供一个**相互匹配的发布包**：

| 配置字段 / 资源 | 准备要求 |
| --- | --- |
| `model_python` | 已安装模型依赖的独立解释器；按模型发布包的 Python/CUDA 版本及锁文件安装，不与 platform/CPU/cuMotion 环境混装 |
| `framework_root` | 匹配 checkpoint 的 cosmos-framework 源码及其版本记录，含 `cosmos_framework/inference/umi_3view_29d_identity.py` 和 `umi_3view_29d_predictor.py` |
| `checkpoint_path` | UMI 3-view / 29D ID 模型完整导出 checkpoint，包括实际权重分片、索引和配置；通用 Cosmos 权重不能直接替代 |
| 归一化统计与动作 contract | 与该 checkpoint 配套，由 framework 的 checkpoint bundle resolver 找到；必要时使用其固定 `umi_3view_29d_v3.json` contract。当前平台不提供任意 stats/contract 路径覆盖 |
| `processor_path` | 同一模型发布包所用 processor/tokenizer 资源（包括配置和所需文件） |
| `vae_path` | 匹配的 VAE 权重文件；以模型发布清单为准 |
| 资源校验 | 保存模型发布者提供的版本和 SHA-256 清单，在启用前逐项校验；不要提交权重、私有数据或访问 token 到本仓库 |

本仓库尚未提供可公开下载的上述 UMI checkpoint 与匹配 framework 发布包；因此没有通用的
`pip install cosmos` 或可替代它们的公开下载链接。拿不到该发布包时，可使用 CPU/回放能力，
不能宣称 Cosmos 推理已可用。不要靠安装同名 PyPI 包或填写空目录绕过依赖。

## 6. 填写 Agent 配置并检查

复制模板时避免覆盖已有配置，然后编辑：

```bash
cp -n "$UMI_ROOT/agent-runtime.template.json" "$UMI_ROOT/agent-runtime.json"
${EDITOR:-vi} "$UMI_ROOT/agent-runtime.json"
```

保留已生成的 `laundry_root`、`laundry_commit`、`report_python`，填写：

| 字段 | 取值 |
| --- | --- |
| `host_python` | `$PLATFORM_ROOT/.venv/bin/python` 的展开后绝对路径；host worker 需要平台完整依赖，不能使用仅含 laundry 的 CPU 查看器环境 |
| `converter_root` | `$UMI_ROOT/openpi` 的展开后绝对路径 |
| `converter_python` | `$PLATFORM_ROOT/.venv/bin/python` 的展开后绝对路径 |
| `converter_lerobot_root` | `$PLATFORM_ROOT` 的展开后绝对路径 |
| `urdf_path` | 标定匹配时使用 `$UMI_ROOT/openpi/data/H10W_DVT2_URDF/H10W_DVT2_TekkenD_with_Camera_2_TrackerF_Ry90.urdf` 的展开后绝对路径 |
| `calibration_profile` | 当前适配只支持 `h10w_tracker_rx90_rz180`，其他采集标定不能照抄 |
| `lock_dir` | 与同节点 FLORA/其他遵守该协议的执行器一致的共享 GPU 锁目录；所有执行服务账号需有读写权限 |
| `gpus` | 显式选择预留设备编号，例如 `[0]`，先用 `nvidia-smi` 核对；不能保留空列表或照搬别人的编号 |
| `cumotion_python` | 上一步创建的 `laundry/.venv_cumotion/bin/python`，回放必需 |
| Cosmos 字段 | 按上一节填写；只启用 replay 时可保留 Cosmos 字段占位，Cosmos 检查会明确失败 |

JSON 不展开 `$HOME`、`$UMI_ROOT` 或 `~`，必须写实际绝对路径。
当前共享 GPU 准入要求设备无外部计算/图形进程、空闲显存达到 73,000 MiB、共享锁可用。
不要对现有任务占用的卡进行试跑。路径能由当前用户读取，不代表服务账号可读取；
同时检查父目录遍历、源码/资源读取、锁目录和输出目录写入权限。

```bash
python3 deploy/data-platform/setup-umi.py check \
  --runtime "$UMI_ROOT/agent-runtime.json" --mode replay
python3 deploy/data-platform/setup-umi.py check \
  --runtime "$UMI_ROOT/agent-runtime.json" --mode cosmos
```

只执行准备启用的模式。检查器校验实际源码 commit/干净状态、必需路径、转换器导入、
cuMotion 或 Cosmos 模块导入以及模型的统计/contract 解析。
它不加载模型权重、不运行 GPU 推理，也不证明资源语义正确、GPU 容量充足或业务验收通过。

## 7. 接入平台服务与最小验收

在**执行节点 Agent 服务**实际使用的环境文件里设置展开后的绝对路径：

```text
DATA_PLATFORM_UMI_RUNTIME=/absolute/path/agent-runtime.json
```

在需要打开原版查看器的 **Server 服务**环境里设置：

```text
DATA_PLATFORM_UMI_VIEWER_RUNTIME=/absolute/path/viewer-runtime.json
```

Server 和 Agent 分处不同机器时，各自在本机运行 CPU 安装步骤，配置本机路径；不要复制指向另一台机器的 JSON。
Server 只需要 CPU 查看环境，不需要 Cosmos 权重。服务账号必须能执行其配置的解释器和读取完整源码/资源。
配置环境文件不等于已生效，按 [运维手册](data_platform_operations.md#original-umi-visualization-runtime) 的环境和滚动更新规则加载配置，
保留正在执行任务的旧执行器；安装脚本本身不修改服务、不重启进程、不提交 Job。

在已注册且有权限的小型测试数据副本上，依次验收：

1. 节点能力中检查目标算子可用及不可用原因；先运行 `umi.diagnose`，验证报告和输入文件指纹。
2. 回放先选一个符合格式的 episode，检查 Job 成功、结果可打开、Viser 播放与相机同步。
3. Cosmos 先准备同源诊断，网页确认目标区间后提交一个试点；检查候选、后检报告和源文件未变。
4. 确认错误账号不能查看任务结果，查看会话关闭后回收；删除验收创建的临时账号。

候选始终保留 `needs_review` / `human_approvals=0`，可回放或后检通过不等于训练批准。
本指南和安装器不改变 Server/Agent 协议或 CLI 版本，不代表新平台发行或模型在线验收已经完成。
