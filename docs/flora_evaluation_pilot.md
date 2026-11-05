# FLORA 人工标注与评测试点

从数据集的 Temporal Caption Review 点击「人工标注与评测」，进入同一数据集的试点工作台。
入口：`/remote/<location_id>/temporal-caption/evaluation`。仅有该数据集写权限的账号可以修改；Viewer 只读。

## 人工审查

1. 从左侧队列选择 episode，核对原始 Instruction，播放同步的独立原始视角。
2. 在「修改参考标注」中调整动作起止秒数、拆分/删除动作段、修改描述。填写视频支持的原子事实，按动作、对象身份、双手协作、细节分类；不确定内容不作为评分事实。
3. 独立修改动作质量区间（失败 1、次优 2、正常 3），确保覆盖整个视频。异常区间必须说明依据。填写整个任务 1–5 完成度与原因。
4. 随时保存草稿。核对四个部分后确认参考标注；该版本才参与评测。系统保存原始 FLORA 草稿、来源摘要、每次修订与操作者。多人修改冲突返回 409，不覆盖别人的修改。
5. 在「核对描述质量」中，对 tIoU≥0.5 的候选配对勾选正确覆盖的参考事实，并记录有视频证据的错误。全部候选配对核对完，才产生该条 Joint 分数。
6. 查看指标或导出。之后修改参考标注会撤回确认，清空对应描述核对与固定区间预测，避免旧评分误用于新参考。

这批参考由 FLORA 原稿辅助人工修改，**不是独立盲标的标准答案**。试点按任务/来源/视频特征挑选，不是随机抽样，不能据此外推整个数据集表现。实际用户确认前，平台不会替用户审批任何真实参考。

## 指标协议 `flora-human-pilot-v1`

- Segmentation F1@0.5 / @0.75：原始时间半开区间，以整数微秒计算 tIoU；不吸附参考外边界；最大基数一对一匹配。`2TP/(Npred+Nref)`，按 episode 平均。保留原始重叠动作和空闲间隙。
- Caption factual quality：参考事实按动作/身份/协作/细节赋 0.50/0.25/0.20/0.05 组权重，同组多项平分；只有已被视频支持的组进入参考质量 R。模型正确覆盖质量 M；独立错误家族罚分 F 为该组权重 ×1.25，同一家族重复只罚一次。`q=2M/(R+M+F)`。当前以登录人工判断覆盖/错误，**不是论文指定 LLM judge 的复现**。
- Joint F1：tIoU≥0.5 的边上按 q 最大权一对一匹配，`2Σq/(Npred+Nref)`。未核对完候选边时不报告完整总分。
- Oracle Caption：先冻结人工参考边界，在这些固定窗口重新运行 Captioner，再导入与参考哈希绑定的结果，逐项人工核对，先按参考段平均再按 episode 平均。**现有整条 FLORA 预测不得代替 oracle-window 预测**。本版支持导入与计算，工作台不自动启动该 GPU 推理；尚未导入时明确显示待评。
- Outcome MAE：已确认人工 1–5 分与模型整数输出的平均绝对误差，原始尺度（不乘 100），越低越好。
- Dense：按原始时间每秒一次采样（0,1,2,… 且小于时长），重叠质量区间取更严重标签，跨已确认 episode 汇总。固定三类 Macro F1；Balanced accuracy 平均有参考样本的类别召回率；Bad-action F1 将 1/2 合并；Failure recall 无失败参考时为 N/A。缺失模型预测计入召回分母并报告覆盖率，不静默排除。正常误报失败率作为诊断。

本试点使用 DenseCaption 风格 0.5 阈值，与 WGO 的 0.75/边界吸附/二元文本裁判协议不同。上述自定义数据、人工裁判、抽样和输入处理方式决定其分数不可直接与论文榜单比较。Ranking、Rubric 与 VOC 不在本次三项试点范围。

权威指标来源：[FLORA 论文](https://arxiv.org/html/2610.31779v1)、[官方仓库](https://github.com/NVIDIA/Flora)。

## 接口和 CLI

`GET /api/control/locations/<location_id>/evaluation` 查看队列；`POST .../pilot` 用指定 episode 列表建立最多 10 条冻结队列。
原始独立相机视频需先准备到 Lifecycle 目录的 `evaluation-media/<artifact_key(location_id)>/<episode>/<camera>.mp4`，或已存在于该推理产物的 camera-videos 目录。

`GET/POST .../episodes/<episode>` 查看或保存修订；提交包含 `revision` 与 `action`，动作包括 save、approve、assess、oracle-import、oracle-assess。GET 的 `?revision=N` 读取历史版本。所有写入经过权限、CSRF、数据范围和乐观锁检查。
`GET .../metrics` 计算当前已确认版本；`GET .../export` 导出参考、来源与指标。

```sh
data-platform evaluation --server https://platform.example.invalid --location LOCATION_UUID --web
data-platform evaluation --server https://platform.example.invalid --location LOCATION_UUID --export-file pilot.json
data-platform evaluation --server https://platform.example.invalid --location LOCATION_UUID --episode 18 --oracle-file oracle-18.json
```

Oracle 文件格式为 `{reference_sha256, model, provenance, segments: [{start_s, end_s, caption}]}`。摘要必须匹配导出文件中的当前 reference；顺序及时间必须精确匹配。provenance 记录模型版本、推理 run、输入和生成时间。导入并不批准语义质量。

## 验证

指标手算例覆盖重复预测的一对一匹配、不同阈值、最大权匹配、事实覆盖/重复错误、10/30 FPS 等价、缺失预测及无失败参考；修订验证冲突、失效、不可变原稿和 oracle 绑定。真实 Chrome 覆盖编辑/保存/刷新/确认/语义评分/指标与移动端。真实试点验收只读，审批留给人工。
