在原DEFAME代码基础上进行修改，通过结合MDAgents的Adaptive结构，来提点并减少token使用和早停等等

## 复杂度分类
参考MDAgents中对多模态的医疗决策问题进行复杂度分类，创建`classify.py`文件并在common.label中添加新的枚举类`DifficultyLabel`以及对于的描述`DEFAULT_DIFFICULTY_DEFINITIONS`，用以表示问题的复杂度。

`classify.py`中实现了一个分类器Classifier，并结合了原DEFAME中的ezmm多模态序列类，将多模态事实检查claim作为输入，输出复杂度标签。这里默认多模态模型为GPT-4o。

## 修改**CR+**的`benchmark.py`
针对数据集**claimreview2024+**，原数据加载已经实现claim的定义，在此基础上，将分类器集成在benchmark中，初始化会判断是否需要复杂度分类，如果原json中有了difficulty字段，则直接加载，否则通过分类器进行预测并将预测结果存入json，方便下次使用。

## Adaptive Frame 1

### Easy

针对easy类型，在label中的描述为`"Verifiable through single reliable source or simple method. Content (text/image/video) is clear and straightforward, requiring no specialized knowledge."`。被判断为easy的claim通过简单的方法就能判断，这里我的思路是：
planner->summarize->judge 删除develop 改成最大迭代次数2
本质上和原来的defame流程是一样的，因为简单，planner也会直接选择最简单的工具，也不会太多迭代，也不太需要develop。

### Medium

将develop+judge阶段改成MAD即多智能体辩论

### Hard

planner指定工具执行并获取相关信息之后，leader将分配任务目标给各个group然后各个group根据获取到的信息执行各自的task，最后leader汇总结果。替换的是summarize+develop+judge


根据代码中的定义，三个难度等级分别关注以下问题：

## DifficultyLabel.EASY (简单)
关注的问题：
- **信息源验证**：可通过单一可靠信息源验证
- **验证方法**：使用简单的验证方法即可
- **内容清晰度**：文本/图像/视频内容清晰直观
- **知识要求**：不需要专业知识背景
- **理解难度**：内容表述简单明了，容易理解

## DifficultyLabel.MEDIUM (中等)
关注的问题：
- **多源交叉验证**：需要对比和交叉引用多个信息源
- **深度分析**：内容可能需要详细分析
- **时空上下文**：需要验证时间和空间背景信息
- **多模态关联**：需要验证多模态内容（文本、图像、视频等）之间的关联性
- **复杂性增加**：比简单级别需要更多的验证步骤

## DifficultyLabel.HARD (困难)
关注的问题：
- **信息源缺失**：缺乏可靠的信息源
- **信息冲突**：存在相互矛盾的信息
- **专业知识需求**：需要专门的领域知识才能分析
- **复杂内容关系**：需要分析文本、视觉和上下文元素之间的复杂关系
- **多维度分析**：涉及跨文本、视觉和上下文的综合分析能力

这个分级体系主要考虑了**验证复杂度**、**所需资源**、**专业知识要求**和**内容分析难度**四个维度。