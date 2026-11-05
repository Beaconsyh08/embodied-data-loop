# 持续学习第一版：离线验收记录

日期：2026-10-26。交付内容见[功能与治理契约](data_platform_continuous_learning.md)。

## 源码与环境

验收使用独立工作树的 `.venv/bin/python`，已验证 pytest、发行模块和 Agent bundle 脚本可导入。
所有执行对象使用临时合成数据；MySQL 是测试进程拥有的临时 socket-only 实例，不连接业务控制库。

## 已完成的验证

下列批次有重叠，不能相加作为独立用例总数。

| 范围 | 结果 | 说明 |
|---|---|---|
| 学习模块最终 SQLite 回归 | 28 passed | 包含真实 Chrome 学习管理／反馈流程、来源／权限、账本、游标、手动任务及离线闭环 |
| 学习与策略硬化组合 | 59 passed | 包含转换后拆分、转换后质检、模板绑定与开发集退步门禁；最终时间边界修正另由学习模块全量与 MySQL 补测覆盖 |
| 隔离 MySQL 与 LangGraph | 26 passed | 当时完整学习模块与3项真实持久恢复场景，包含浏览器；不使用真实模型 |
| MySQL 收据／覆盖率／闭环补测 | 5 passed | 补测缺失最终确认降级、去重分母、回填和两个闭环分支 |
| MySQL 最终时间／来源补测 | 4 passed | 冻结回填边界、延迟协调器完成、个人经历隔离、已审核案例来源与隐藏集拒绝 |
| CLI、安装器与更新 | 45 passed | 状态／反馈、TLS、登录、CSRF、确认、结果链接及安装兼容 |
| 控制面／任务／Pipeline／发布基础回归 | 178 passed, 1 skipped | 单独的外部 MySQL 锁测试未配置其专用 URL；学习 MySQL 验收使用上面的隔离实例 |
| 最终发布模块及新增审计／浏览器入口 | 95 passed | 学习浏览器案例已加入必须实际执行的发布浏览器门禁 |
| 现有 Pipeline／助手／工作区真实浏览器回归 | 13 passed | 编辑、执行、审核、普通角色限制、异步恢复及操作卡；使用临时合成数据 |

Ruff check、format check 和 `git diff --check` 通过。没有运行整个机器人／训练／仿真测试全集。

历史回填补测实际发现并修复两个精度问题：旧 Run／Attempt 使用 MySQL FLOAT；Job 的 DATETIME(0) 也可能向下一秒舍入。
历史扫描采用 Job 完成时间和冻结的整秒上界；实时 Run 扫描跟踪协调器终态时间并保留边界余量。
测试明确区分“暂停历史回填”和“继续实时采集”，并验证重启、去重及来源撤销。

## 离线闭环证明的范围

通过分支：有效执行证据 → 合成案例草案 → 人工审核 → 受限候选 → 开发门禁／验收门禁 → 测试库中的人工启用 → 来源撤销使绑定失效。
退步分支：候选在开发门禁被拒绝，不消耗隐藏阶段，原启用版本保留。

闭环评测的模型响应与实验执行边界使用替身；持久事务、来源核验、策略门禁和启用校验使用真实实现。
Pipeline／Agent 浏览器回归另外使用临时合成数据验证原执行与审核流程。
这些测试证明闭环与边界行为，不证明候选在真实模型上提高了正确率，不构成新的独立隐藏集或真实节点六场景验收。
初始公开案例只作开发回归资料，未生成或宣称完成新隐藏集。

## 重验命令

从包含本提交的工作树运行：

```bash
DATA_PLATFORM_BROWSER_EXECUTABLE=/usr/bin/google-chrome \
  .venv/bin/python -m pytest tests/datasets/test_ai_learning.py \
  tests/datasets/test_strategy_hardening.py -q

DATA_PLATFORM_BROWSER_EXECUTABLE=/usr/bin/google-chrome \
  .venv/bin/python experiments/assistant_langgraph/run_acceptance.py --learning
```

上述测试使用模型替身。它们不会设置真实学习预算或启动收费策略评测。
