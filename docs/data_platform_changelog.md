# Public source changes

This distribution starts from a reviewed source snapshot. Deployment histories, private
acceptance records and internal commit identities are not included.

- Dataset discovery, Viewer, quality review, annotation and versioned curation workflows.
- Node Agents, bounded AI assistance and explicit confirmation of processing operations.
- Optional UMI integration with pinned source checks and separately supplied model resources.
- Public installation examples require the operator to provide their own repository URLs,
  platform endpoints, model locations, network policy and release artifacts.

No hosted deployment or model inference acceptance is implied by this source snapshot.

## 2026-11-06 · 领域目录与算子实现归整（未新增发行）

- 顶层 82 个业务模块按职责归入 access、control、datasets、runtime、annotations、artifacts、ops 和 web；保留 CLI、Viewer、WSGI 入口。
- 原 `precompute` 及 Caption/FLORA、MCAP、Curation 计算统一进入 `operators/implementations/`；算子定义、执行适配和 Pipeline 编排保持分层。
- 内部调用、独立服务和打包清单使用新路径；旧导入和 `python -m` 入口集中兼容，同一模块对象保留模型、锁和缓存身份。部署包装器保留可回退到旧发行包的稳定命令。
- 同步资源定位、独立 worker、源码导航及开发契约，加入导入顺序、命令、惰性发现、Schema 与资源回归。
- Server/Agent 算子 ID、版本、请求/结果和权限协议不变；CLI 保持 0.3.42。本次为源码重构，没有部署或推进 CLI latest；后续平台升级仍需独立 CLI 发布验收。

## 2026-11-05 · 文档导航与发布验证整理（未新增发行）

- 精简仓库 README，移除重复的旧单环境运维步骤；CLI README 按使用流程和能力兼容组织。
- 文档索引按使用、开发、运维与历史设计分组，补齐 UMI、混合 FLORA、人工评测与记忆入口。
- 校正架构状态和 UMI 升级说明，新增源码导航与测试选择说明；历史计划标明适用基线。
- 发布测试清单、并行选项和缓存隔离统一归入 `release_validation.py`，保留原调用入口和完整测试范围。
- Server/Agent 业务逻辑与协议、CLI 版本均不变；本次没有构建、部署或发布 CLI。
