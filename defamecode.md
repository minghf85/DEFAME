# DEFAME框架技术文档

## 1. 项目概述

DEFAME (Dynamic Evidence-based FAct-checking with Multimodal Experts) 是一个强大的多模态声明验证系统，将事实检查任务分解为动态的6阶段管道，利用多模态大语言模型(MLLM)来完成规划、推理和证据总结等子任务。

## 2. 项目结构树

```
DEFAME/
├── defame/                           # 核心框架代码
│   ├── __init__.py
│   ├── fact_checker.py              # 主入口：核心事实检验类
│   ├── common/                      # 通用组件
│   │   ├── action.py               # 动作定义
│   │   ├── claim.py                # 声明数据结构
│   │   ├── content.py              # 内容处理
│   │   ├── evidence.py             # 证据结构
│   │   ├── label.py                # 标签定义
│   │   ├── report.py               # 报告生成
│   │   ├── modeling.py             # 模型接口
│   │   └── ...
│   ├── modules/                     # 核心模块
│   │   ├── actor.py                # 执行器：执行动作获取证据
│   │   ├── claim_extractor.py      # 声明提取器
│   │   ├── doc_summarizer.py       # 文档摘要器
│   │   ├── judge.py                # 判断器：确定声明真实性
│   │   └── planner.py              # 规划器：选择下一步动作
│   ├── procedure/                   # 流程变体
│   │   ├── procedure.py            # 基础流程抽象类
│   │   └── variants/               # 不同实现变体
│   │       ├── qa_based/           # 基于问答的方法
│   │       └── summary_based/      # 基于摘要的方法
│   ├── evidence_retrieval/          # 证据检索系统
│   │   ├── tools/                  # 各种工具
│   │   │   ├── searcher.py         # 搜索工具
│   │   │   ├── text_extractor.py   # 文本提取
│   │   │   ├── object_detector.py  # 目标检测
│   │   │   ├── face_recognizer.py  # 人脸识别
│   │   │   ├── geolocator.py       # 地理定位
│   │   │   └── manipulation_detector.py # 篡改检测
│   │   ├── integrations/           # 外部集成
│   │   └── scraping/              # 网页抓取
│   ├── prompts/                    # 提示模板
│   ├── eval/                       # 评估框架
│   └── utils/                      # 工具函数
├── config/                         # 配置文件
├── scripts/                        # 执行脚本
└── third_party/                    # 第三方库
```

## 3. 核心入口：FactChecker类

### 3.1 主要职责
- 端到端的事实验证流程控制
- 协调各个模块的工作
- 管理声明提取和验证过程

### 3.2 核心方法

```python
class FactChecker:
    def __init__(self, llm, tools, procedure_variant, ...):
        """初始化事实检验器及所有子模块"""
        
    def extract_claims(self, content) -> list[Claim]:
        """从内容中提取可检验的声明"""
        
    def check_content(self, content) -> tuple[Label, list[Report], list[dict]]:
        """端到端检验内容，返回聚合的真实性判断"""
        
    def verify_claim(self, claim) -> tuple[Report, dict]:
        """核心方法：验证单个声明"""
```

### 3.3 初始化流程
1. 创建大语言模型实例
2. 初始化声明提取器(ClaimExtractor)
3. 初始化工具集(Tools)
4. 创建核心模块：规划器(Planner)、执行器(Actor)、判断器(Judge)
5. 初始化文档摘要器(DocSummarizer)
6. 设置具体的验证流程(Procedure)

## 4. 核心模块详解

### 4.1 声明提取器 (ClaimExtractor)
**位置**: `defame/modules/claim_extractor.py`
**功能**: 从输入内容中提取可检验的声明
**特性**:
- 支持声明解释(interpret)
- 支持声明分解(decompose)
- 支持去上下文化(decontextualize)
- 支持检验价值过滤(filter_check_worthy)

### 4.2 规划器 (Planner)
**位置**: `defame/modules/planner.py`
**功能**: 基于当前知识状态选择下一步执行的动作
**核心方法**:
```python
def plan_next_actions(self, doc: Report, all_actions=False) -> (list[Action], str):
    """根据当前文档状态规划下一步动作"""
```

### 4.3 执行器 (Actor)
**位置**: `defame/modules/actor.py`
**功能**: 执行给定的动作并返回证据
**核心方法**:
```python
def perform(self, actions: list[Action], doc: Report = None) -> list[Evidence]:
    """执行一系列动作，返回证据列表"""
```

### 4.4 判断器 (Judge)
**位置**: `defame/modules/judge.py`
**功能**: 基于收集的证据确定声明的真实性
**核心方法**:
```python
def judge(self, doc: Report, is_final: bool = True) -> Label:
    """对给定文档进行真实性判断"""
```

## 5. 证据检索系统

### 5.1 工具体系
DEFAME提供多种专业化工具用于证据检索：

- **Searcher**: 网络搜索工具
- **TextExtractor**: 文本提取工具  
- **ObjectDetector**: 目标检测工具
- **FaceRecognizer**: 人脸识别工具
- **Geolocator**: 地理定位工具
- **ManipulationDetector**: 图像篡改检测工具
- **CredibilityChecker**: 可信度检查工具

### 5.2 工具基类
```python
class Tool(ABC):
    """所有工具的基类"""
    actions: list[type(Action)]  # 该工具支持的动作类型
    
    def perform(self, action: Action, summarize: bool = True, doc: Report = None) -> Evidence:
        """执行动作并返回证据"""
```

## 6. 流程控制系统 (Procedure)

### 6.1 流程抽象
**位置**: `defame/procedure/procedure.py`
所有流程的基类，定义了事实检验的算法策略：

```python
class Procedure(ABC):
    def apply_to(self, doc: Report) -> (Label, dict[str, Any]):
        """对事实检验文档执行特定的验证流程"""
```

### 6.2 主要流程变体
- **DynamicSummary** (defame): 默认的动态摘要流程
- **InFact**: 基于InFact系统的流程
- **QA-based**: 基于问答的各种流程变体
- **Summary-based**: 基于摘要的各种流程变体

## 7. 工作流程详解

### 7.1 整体流程
```
输入内容 → 声明提取 → 逐个声明验证 → 结果聚合 → 输出判断
```

### 7.2 单个声明验证流程
```
1. 初始化报告文档
2. 重置执行器状态
3. 设置时间限制
4. 应用选定的验证流程：
   - 规划下一步动作
   - 执行动作获取证据
   - 判断当前证据是否足够
   - 如不足够，继续规划-执行循环
5. 最终判断
6. 生成摘要和理由
```

### 7.3 证据收集循环
在验证过程中，系统进行动态的证据收集：
1. **规划阶段**: Planner根据当前状态选择合适的动作
2. **执行阶段**: Actor执行动作，调用相应工具获取证据
3. **判断阶段**: Judge评估当前证据是否足以做出判断
4. **循环控制**: 根据max_iterations参数控制最大迭代次数

## 8. 扩展和定制

### 8.1 添加新工具
1. 继承Tool基类
2. 实现perform方法
3. 定义支持的Action类型
4. 在工具初始化中注册

### 8.2 自定义流程
1. 继承Procedure基类
2. 实现apply_to方法
3. 在PROCEDURE_REGISTRY中注册

### 8.3 配置参数
通过FactChecker的初始化参数可以灵活配置：
- LLM模型选择和参数
- 工具集配置
- 流程变体选择
- 各种开关选项(interpret, decompose等)
- 迭代和结果限制

## 9. 核心数据结构

### 9.1 Claim (声明)
表示待验证的声明，包含文本内容、日期、图像等信息

### 9.2 Evidence (证据)
表示从各种工具获取的证据信息

### 9.3 Report (报告)
事实检验的核心文档，记录整个验证过程和结果

### 9.4 Label (标签)
验证结果标签：SUPPORTED, REFUTED, NEI, CONFLICTING等

### 9.5 Action (动作)
表示可执行的动作，如搜索、文本提取、图像分析等

## 10. 使用示例

### 10.1 基本使用
```python
from defame.fact_checker import FactChecker

# 初始化事实检验器
fact_checker = FactChecker(
    llm="gpt_4o_mini",
    procedure_variant="defame",
    max_iterations=5
)

# 检验内容
content = ["这是一个需要验证的声明"]
label, reports, metas = fact_checker.check_content(content)

print(f"验证结果: {label.value}")
```

### 10.2 自定义配置
```python
# 使用自定义工具配置
tools_config = {
    "searcher": {"engine": "google"},
    "text_extractor": {"max_length": 1000}
}

fact_checker = FactChecker(
    llm="gpt_4o",
    tools_config=tools_config,
    procedure_variant="infact",
    interpret=True,
    decompose=True
)
```

## 11. 总结

DEFAME框架通过模块化设计实现了高度可扩展的事实检验系统。核心的FactChecker类协调各个专业化模块，通过动态的规划-执行-判断循环来收集证据并做出最终判断。框架支持多模态证据处理，可以灵活配置和扩展，适用于各种事实检验场景。