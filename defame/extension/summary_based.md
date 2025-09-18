# summary_based 各流程区别说明

本模块下包含多种 summary-based fact-checking 流程，主要区别如下：

## 1. DynamicSummary
- 继承自 `Procedure`，动态多轮迭代，每轮根据当前文档状态规划下一步 actions。
- 每轮调用 `planner.plan_next_actions`，执行 actions，开发 reasoning，直到获得结论或达到最大迭代次数。
- 适合需要多步推理和信息补充的场景。

## 2. StaticSummary
- 只执行一次 action 规划和 evidence 检索，不做多轮迭代。
- 适合信息充分、一步即可判断的场景。

## 3. AllActionsSummary
- 基于 DynamicSummary，每轮都强制添加 claim 文本的 Search（文本和图片模式）。
- actions 更全面，确保每轮都检索 claim 相关的文本和图片证据。
- 适合需要最大化信息检索的场景。

## 4. NoDevelop
- 基于 DynamicSummary，但每轮不调用 `_develop`，即不做 develop 步骤。
- 适合只需要 evidence 检索和判断，不需要 develop 过程的场景。

## 5. NoQA
- 继承自 `Procedure`，不进行问题生成（QA），直接生成搜索 query 并检索证据。
- 只做 evidence 检索和 reasoning，总结后直接判断。
- 适合无需提问、只需检索和总结的场景。

---

各流程适用场景不同，可根据任务需求选择合适的 summary-based fact-checking 流程。
