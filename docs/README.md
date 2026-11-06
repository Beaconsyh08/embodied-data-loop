<!---
Copyright 2020 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Data Platform 文档导航

平台 Markdown 手册可直接阅读，无需构建文档站点。按工作选择入口；
[开发指南](data_platform_development.md) 提供源码地图和验证方法。

## 使用平台

| 需要做什么 | 文档 |
| --- | --- |
| 查页面入口、参数、角色与操作结果 | [页面操作手册](data_platform_page_user_guide.md) |
| 登录、恢复密码、理解数据范围和任务配额 | [账号](data_platform_accounts.md)、[权限](data_platform_permissions.md)、[任务管理](data_platform_multi_user_management.md) |
| 终端访问和外部助手接入 | [CLI](../clients/data-platform-cli/README.md)、[MCP](data_platform_mcp.md) |
| 导入、合并与任务定义 | [MCAP](data_platform_mcap_import.md)、[Merge 对齐](data_platform_merge_alignment.md)、[任务目录](data_platform_task_catalog.md) |
| 质量复核与构建 | [统一 Curation](data_platform_unified_curation.md)、[生命周期](data_lifecycle_architecture.md) |
| 时序标注与人工评测 | [Temporal Caption / FLORA](temporal_caption_trial.md)、[FLORA 人工评测试点](flora_evaluation_pilot.md) |
| 文本记忆与离线导出 | [Text memory](memory_annotation.md)、[MEM export](mem_export.md) |
| UMI 诊断、回放与 Cosmos 候选 | [UMI 数据支持](umi_dataset_support.md)、[data-laundry 安装](data_laundry_setup.md) |

## 开发与架构

| 需要做什么 | 文档 |
| --- | --- |
| 找源码、添加算子、选择测试 | [领域目录与算子放置规则](../lerobot/data_platform/README.md)、[开发指南](data_platform_development.md) |
| 了解当前能力、架构与限制 | [平台状态说明](distributed_data_platform_current_status.md)、[架构实现核对](data_platform_architecture_review.md) |
| 算子、Pipeline 与助手接口 | [统一算子和流程](data_platform_operators.md)、[LangGraph 运行器](data_platform_assistant_langgraph.md) |
| 后台经验、策略与反馈 | [Hindsight](data_platform_hindsight.md)、[策略实验](data_platform_strategies.md)、[使用学习](data_platform_continuous_learning.md)、[学习验收](data_platform_continuous_learning_acceptance.md) |
| 独立 FLORA 服务与 GPU 路由 | [HTTP API](flora_http_api.md) |
| 查变化与兼容性 | [变更记录](data_platform_changelog.md) |

## 部署与运维

| 需要做什么 | 文档 |
| --- | --- |
| 更新、验收、恢复和排障 | [运维手册](data_platform_operations.md) |
| 首次建立 dev/prod、批准与回退 | [双环境部署](data_platform_environments.md) |
| 接入数据节点与 Agent | [分布式部署](distributed_data_platform_deployment.md) |
| 找安装包、数据库、缓存与数据目录 | [存储位置](data_platform_storage_locations.md) |

文档日期和版本限定源码核对范围；实时安装、Agent 能力和业务验收需按运维手册确认。

## 历史设计

[任务管理实施计划](data_platform_job_management_plan.md) 保留 2026-09-08 的设计基线，
[交互优化设计](data_platform_ux_design.md) 保留当时的页面设计与验收范围。
其中的“尚未实现”或部署快照属于当时状态；当前契约使用上方手册。

<a id="data-platform-processing-architecture"></a>

算子架构与外部接入已归入上方“开发与架构”和“使用平台”，保持原导航锚点可用。
下方说明只适用于上游 LeRobot 的 `docs/source/` 文档站点。

# Generating the documentation

To generate the documentation, you first have to build it. Several packages are necessary to build the doc,
you can install them with the following command, at the root of the code repository:

```bash
pip install -e ".[docs]"
```

You will also need `nodejs`. Please refer to their [installation page](https://nodejs.org/en/download)

---
**NOTE**

You only need to generate the documentation to inspect it locally (if you're planning changes and want to
check how they look before committing for instance). You don't have to `git commit` the built documentation.

---

## Building the documentation

Once you have setup the `doc-builder` and additional packages, you can generate the documentation by
typing the following command:

```bash
doc-builder build lerobot docs/source/ --build_dir ~/tmp/test-build
```

You can adapt the `--build_dir` to set any temporary folder that you prefer. This command will create it and generate
the MDX files that will be rendered as the documentation on the main website. You can inspect them in your favorite
Markdown editor.

## Previewing the documentation

To preview the docs, first install the `watchdog` module with:

```bash
pip install watchdog
```

Then run the following command:

```bash
doc-builder preview lerobot docs/source/
```

The docs will be viewable at [http://localhost:3000](http://localhost:3000). You can also preview the docs once you have opened a PR. You will see a bot add a comment to a link where the documentation with your changes lives.

---
**NOTE**

The `preview` command only works with existing doc files. When you add a completely new file, you need to update `_toctree.yml` & restart `preview` command (`ctrl-c` to stop it & call `doc-builder preview ...` again).

---

## Adding a new element to the navigation bar

Accepted files are Markdown (.md).

Create a file with its extension and put it in the source directory. You can then link it to the toc-tree by putting
the filename without the extension in the [`_toctree.yml`](https://github.com/huggingface/lerobot/blob/main/docs/source/_toctree.yml) file.

## Renaming section headers and moving sections

It helps to keep the old links working when renaming the section header and/or moving sections from one document to another. This is because the old links are likely to be used in Issues, Forums, and Social media and it'd make for a much more superior user experience if users reading those months later could still easily navigate to the originally intended information.

Therefore, we simply keep a little map of moved sections at the end of the document where the original section was. The key is to preserve the original anchor.

So if you renamed a section from: "Section A" to "Section B", then you can add at the end of the file:

```
Sections that were moved:

[ <a href="#section-b">Section A</a><a id="section-a"></a> ]
```
and of course, if you moved it to another file, then:

```
Sections that were moved:

[ <a href="../new-file#section-b">Section A</a><a id="section-a"></a> ]
```

Use the relative style to link to the new file so that the versioned docs continue to work.

For an example of a rich moved sections set please see the very end of [the transformers Trainer doc](https://github.com/huggingface/transformers/blob/main/docs/source/en/main_classes/trainer.md).

### Adding a new tutorial

Adding a new tutorial or section is done in two steps:

- Add a new file under `./source`. This file can either be ReStructuredText (.rst) or Markdown (.md).
- Link that file in `./source/_toctree.yml` on the correct toc-tree.

Make sure to put your new file under the proper section. If you have a doubt, feel free to ask in a Github Issue or PR.

### Writing source documentation

Values that should be put in `code` should either be surrounded by backticks: \`like so\`. Note that argument names
and objects like True, None or any strings should usually be put in `code`.

#### Writing a multi-line code block

Multi-line code blocks can be useful for displaying examples. They are done between two lines of three backticks as usual in Markdown:


````
```
# first line of code
# second line
# etc
```
````

#### Adding an image

Due to the rapidly growing repository, it is important to make sure that no files that would significantly weigh down the repository are added. This includes images, videos, and other non-text files. We prefer to leverage a hf.co hosted `dataset` like
the ones hosted on [`hf-internal-testing`](https://huggingface.co/hf-internal-testing) in which to place these files and reference
them by URL. We recommend putting them in the following dataset: [huggingface/documentation-images](https://huggingface.co/datasets/huggingface/documentation-images).
If an external contribution, feel free to add the images to your PR and ask a Hugging Face member to migrate your images
to this dataset.
