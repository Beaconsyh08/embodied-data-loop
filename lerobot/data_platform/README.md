# Data Platform 源码结构

按当前业务职责分包。根目录保留本地 CLI、Flask 组合根和 WSGI 入口；
业务代码使用下列领域路径，历史导入集中由 `_compat.py` 兼容。

```text
data_platform/
├── __main__.py / cli.py / viewer.py / wsgi.py  # 公开启动入口与组合根
├── access/                 # 密码、角色、MCP 凭据、源保护与删除审批
├── control/                # 控制面存储、管理迁移、列表查询、管理 CLI
├── datasets/               # 元数据、引用、扫描、结果、缓存与相机布局
│   ├── lifecycle/          # 版本/Manifest/物化服务与 SQLite repository
│   ├── tasks/              # 任务目录、AI 建议与阶段文本
│   └── cleanup/            # 源失联确认、Agent 检查与自有缓存清理
├── runtime/                # Agent、本地执行器、进程监督、资源、Job/Attempt
├── operators/              # 算子库：契约、注册、Schema、执行适配
│   └── implementations/    # 实际计算实现；按需加载
│       ├── preprocess/     # 转换、标准化、拆分合并、修复
│       ├── caption/        # Caption/FLORA/融合、信号、媒体、独立 GPU worker
│       ├── mcap/           # 导入协议、运行环境、独立 v3 worker
│       ├── umi/            # 诊断、对比、Replay/Cosmos 与原版报告资源
│       ├── labeling/       # 检测与标签
│       ├── tagging/        # 几何、规则、VLM 标签
│       ├── embedding/      # 特征提取、模型后端与降维
│       ├── compare/        # 数据对比
│       └── construction/   # 数据构建
├── processing/             # 固定版本 Pipeline、计划、提交与协调器
├── annotations/            # Caption 目录、任务层级、读取与发布编排
├── memory/                 # 因果文本记忆、模型试验与修订审核
├── evaluation/             # 人工参考修订、事实审核与指标
├── ai_tools/               # 助手、运行器、Qwen、经验、策略与反馈
├── artifacts/              # OSS、归档、迁移、审计及离线 MEM 导出
├── ops/                    # 环境、安装、发行、批准、备份与验收
├── web/                    # 环境页面与 UMI 查看会话
├── routes/                 # HTTP 边界与授权
├── services/               # 控制台共享状态及服务适配
├── templates/ / static/    # Jinja、原生 JavaScript/CSS
└── _compat.py / _paths.py  # 旧导入映射与稳定资源根
```

## 算子放在哪里

1. `contracts.py` 定义 `OperatorSpec`、`ExecutionContext` 和 `OperatorResult`；
   `builtins.py` / `umi.py` 登记版本、参数、资源和效果，`registry.py` 提供唯一注册入口。
2. 计算代码放到 `implementations/<领域>/`。读数据、媒体、信号与格式的共享助手也位于
   `implementations/`，处理入口尽量薄，复用现有 v2.1/v3.0 和发布契约。
3. `execution.py` 与 `processing/` 负责适配和编排；权限、租约、调度和 Job 发布继续属于
   `access/`、`control/`、`runtime/`，不进入算法实现。
4. `memory/`、`evaluation/` 保留自身的因果、审核和指标领域契约；
   算子通过明确入口调用它们，不复制另一套审核或存储逻辑。

算子目录发现只读取定义，不导入模型、GPU worker 或整个实现库。
新增算子沿用 `OperatorSpec` 注册和现有 Job/Attempt，不另建独立执行队列。

## 兼容与验证

旧导入如 `lerobot.data_platform.execution` 与新路径 `runtime.supervisor` 指向同一模块对象；
历史 `precompute.*` 映射到 `operators.implementations.*`，旧 `python -m` 命令仍可使用。
兼容映射按需加载，避免重复注册数据库模型、算子、锁和缓存。
仓库内部和新增代码使用新路径；不要在旧位置添加实现或复制文件。
部署 Shell 包装器保留稳定的旧模块命令，兼容缺少源码虚拟环境时回退到旧的已安装发行包；
`run-command.sh` 同时识别新旧源码布局。

物理 `.py` 路径已经迁移。独立 MCAP/FLORA/UMI worker 的路径与资源随调用方一起更新；
外部脚本应使用新文件位置或公开模块入口，不根据旧文件位置寻找模板。
共享模板/静态文件通过 `_paths.PACKAGE_ROOT` 定位。

`tests/datasets/test_package_layout.py` 覆盖新旧导入顺序、共享状态、旧命令、
算子惰性发现、Schema 和资源路径；打包还需检查 wheel 内的 worker 与模板。
完整开发与验收说明见 [开发指南](../../docs/data_platform_development.md)。
