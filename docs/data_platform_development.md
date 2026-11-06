# Data Platform 开发与源码导航

本文按当前源码职责定位修改入口；页面使用见 [用户指南](data_platform_page_user_guide.md)，
部署与线上状态见 [运维手册](data_platform_operations.md)。

## 目录与依赖边界

完整目录树和新增算子的放置方式见 [源码目录说明](../lerobot/data_platform/README.md)。
根目录保留 `cli.py`、`viewer.py`、`wsgi.py` 等入口；其余代码按实际职责组织。

| 领域 | 职责与修改入口 |
| --- | --- |
| [access/](../lerobot/data_platform/access/) | 密码、角色、源保护、删除审批与 MCP 凭据 |
| [control/](../lerobot/data_platform/control/) | 控制面存储、管理迁移、列表与摘要查询 |
| [datasets/](../lerobot/data_platform/datasets/) | 元数据、引用、扫描、结果、缓存；子包 `lifecycle/`、`tasks/`、`cleanup/` |
| [runtime/](../lerobot/data_platform/runtime/) | Agent、本地执行器、Job/Attempt、资源配额与进程监督 |
| [operators/](../lerobot/data_platform/operators/) | 算子定义、Schema、注册与执行适配 |
| [operators/implementations/](../lerobot/data_platform/operators/implementations/) | 转换、质量、标注、MCAP、UMI、Embedding、对比与构建的计算实现 |
| [processing/](../lerobot/data_platform/processing/) | 固定版本计划、提交、顺序 Pipeline 和协调器 |
| [annotations/](../lerobot/data_platform/annotations/)、[memory/](../lerobot/data_platform/memory/)、[evaluation/](../lerobot/data_platform/evaluation/) | 标注目录与任务编排、因果文本记忆、人工参考和指标 |
| [ai_tools/](../lerobot/data_platform/ai_tools/) | 助手工具、Qwen、运行器、偏好、经验与策略 |
| [artifacts/](../lerobot/data_platform/artifacts/) | OSS、报告归档、迁移、审计与离线 MEM 导出 |
| [ops/](../lerobot/data_platform/ops/) | 环境、安装、发行、批准、备份、回退与验证 |
| [web/](../lerobot/data_platform/web/)、[routes/](../lerobot/data_platform/routes/)、[services/](../lerobot/data_platform/services/) | 查看会话、HTTP 边界、控制台共享服务；由 `viewer.py` 组合 |
| [templates/](../lerobot/data_platform/templates/)、[static/](../lerobot/data_platform/static/) | Jinja 与原生 JavaScript/CSS；没有独立前端构建步骤 |
| [deploy/data-platform/](../deploy/data-platform/) | Server/Agent 安装脚本、服务和网络配置模板 |
| [独立 FLORA 服务](../services/)、[轻量 CLI](../clients/data-platform-cli/) | 独立服务部署与独立版本客户端 |
| [scripts/](../scripts/)、[tests/datasets/](../tests/datasets/)、[tests/services/](../tests/services/) | 离线工具、平台领域/API/浏览器与独立服务测试 |

## 本地开发与验证

从仓库根目录准备依赖：

```bash
uv sync --extra dev --extra test
```

Server、Agent、UMI 或模型测试另需对应的可选依赖和外部运行环境；
具体配置见 [分布式部署](distributed_data_platform_deployment.md) 与 [UMI 安装](data_laundry_setup.md)。

先运行所改领域的测试，再按影响范围扩展。例如修改发布工具时：

```bash
.venv/bin/python -m pytest tests/datasets/test_releases.py -q
uv run ruff check lerobot/data_platform/ops/releases.py lerobot/data_platform/ops/validation.py
uv run ruff format --check lerobot/data_platform/ops/releases.py lerobot/data_platform/ops/validation.py
git diff --check
```

| 改动 | 优先验证 |
| --- | --- |
| 处理与格式 | `test_preprocess_ops.py`、`test_preprocess_routes.py`；同时检查 v2.1 与 v3.0 共享 shard |
| 版本与物化 | `test_lifecycle.py`；源数据不变、lineage、输出冲突及原子发布 |
| 算子/流程/Jobs | `test_operators.py`、`test_processing.py`、`test_job_management.py` 与受影响执行器测试 |
| Viewer/标注交互 | 对应领域测试加真实浏览器选择、播放、切换、权限和 JavaScript 错误检查 |
| 独立 FLORA 服务 | `tests/services/` 中受影响模块；真实 GPU 推理另记环境验收 |
| 纯文档 | 相对链接/锚点、命令与当前实现契约、Shell 语法和 `git diff --check` |

完整发行回归清单由 `ops.validation.VALIDATION_TESTS` 维护，
并行参数与 worker 缓存隔离也在同一模块。`ops/releases.py` 保留验证导出，
构建时仍使用该完整清单；浏览器验收由 `ops/browser_validation.py` 独立约束。
局部测试通过不代表完整发行或真实模型验收通过。

## 新增代码与兼容

1. 领域逻辑进入对应包；算子实际计算放入 `operators/implementations/<领域>/`，
   元数据定义和注册保留在 `operators/`，Pipeline 编排保留在 `processing/`。
2. 新 HTTP 路由只处理输入、权限与响应，复用 `RouteContext` 的服务、registry、锁和缓存。
   `viewer.py` 继续作为组合根；复杂算法不继续累积在入口文件。
3. `_compat.py` 保留旧模块名与 `python -m` 入口，返回同一个规范模块对象。
   源码内部使用新路径；模板、静态文件通过 `_paths.PACKAGE_ROOT` 读取。
4. 目录整理保留算子 ID/版本、请求/结果结构、任务快照、数据库和权限契约。
   移动 worker 时同步检查独立解释器、旁边的协议文件和 wheel 资源，不能只验证源码导入。

此次是按职责迁移完整模块；`viewer.py`、`datasets/lifecycle/service.py` 和
`runtime/agent.py` 内部仍有较大的函数，后续围绕具体行为逐步拆分并保持回归覆盖。

## 文档与发布

README 保留入口与简述，具体操作维护在对应手册；历史设计单独标明日期和适用范围。
新增能力同步更新 [文档索引](README.md)、相关契约和 [变更记录](data_platform_changelog.md)，
架构职责变化再更新 [状态说明](distributed_data_platform_current_status.md)。

发布前检查实际源码、其他 worktree 与已安装版本，避免旧分支覆盖新功能。
按 [发布流程](data_platform_operations.md) 完成干净提交、发行验证、开发验收和生产批准。
每次平台升级同步发布并验收独立版本的兼容 CLI；纯文档或内部整理不表示已部署。
