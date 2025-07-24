# DEFAME框架数据流转与结构定义

## 1. 数据流转概览

DEFAME框架的数据流转遵循以下主要路径：

```
原始输入 → Content → Claims → Report → Evidence → Final Verdict
   ↓          ↓        ↓        ↓         ↓           ↓
 字符串/多媒体  解释分解   去上下文化  增量构建   工具执行    聚合判断
```

## 2. 核心数据结构定义

### 2.1 Content（内容）
**位置**: `defame/common/content.py`
**作用**: 表示待处理的原始输入内容

```python
class Content(MultimodalSequence):
    # 基本属性
    author: str               # 作者
    date: datetime           # 发布日期
    origin: str              # 来源URL
    meta_info: str           # 元信息
    id: str | int            # 内容标识符
    
    # 处理过程中添加的属性
    interpretation: str      # 内容解释（由ClaimExtractor添加）
    topic: str              # 主题标题（由ClaimExtractor添加）
    claims: list[Claim]     # 提取的声明列表
    verdict: Label          # 聚合的最终判决
```

**数据变化**:
- **输入时**: 只包含基本文本/多媒体内容和元数据
- **经过ClaimExtractor**: 添加interpretation、topic、claims
- **最终**: 添加aggregated verdict

### 2.2 Claim（声明）
**位置**: `defame/common/claim.py`
**作用**: 表示从内容中提取的单个可检验声明

```python
class Claim(MultimodalSequence):
    id: str                      # 声明唯一标识
    context: Content             # 原始上下文内容
    scope: tuple[int, int]       # 在原文中的位置范围
    verdict: Label               # 该声明的判决结果
    justification: MultimodalSequence  # 判决理由
    
    # 继承自context的属性
    @property author, date, origin, meta_info
```

**数据变化**:
- **提取时**: 包含基本声明文本和上下文引用
- **去上下文化后**: 文本变为自包含描述
- **验证完成后**: 添加verdict和justification

### 2.3 Report（报告）
**位置**: `defame/common/report.py`
**作用**: 事实检验的核心文档，记录整个验证过程

```python
class Report:
    claim: Claim                    # 被验证的声明
    verdict: Label                  # 最终判决
    justification: str              # 判决理由摘要
    
    # 过程记录
    reasoning_blocks: list[ReasoningBlock]  # 推理步骤
    actions_blocks: list[ActionsBlock]      # 执行的动作
    evidence_blocks: list[EvidenceBlock]    # 收集的证据
```

**数据变化**:
- **初始化**: 只包含待验证的Claim
- **验证过程中**: 逐步添加推理、动作、证据块
- **完成后**: 添加最终verdict和justification摘要

### 2.4 Action（动作）
**位置**: `defame/common/action.py`
**作用**: 表示可执行的操作指令

```python
class Action(ABC):
    name: str                    # 动作名称
    requires_image: bool         # 是否需要图像
    additional_info: str         # 附加信息
    _init_parameters: dict       # 初始化参数
```

**主要子类**:
- **Search**: 网络搜索动作
- **Extract**: 文本提取动作
- **DetectObjects**: 目标检测动作
- **RecognizeFaces**: 人脸识别动作
- **Geolocate**: 地理定位动作
- **DetectManipulation**: 篡改检测动作

### 2.5 Evidence（证据）
**位置**: `defame/common/evidence.py`
**作用**: 执行动作后获得的证据信息

```python
@dataclass
class Evidence:
    raw: Results                    # 工具的原始输出
    action: Action                  # 产生该证据的动作
    takeaways: MultimodalSequence   # 对事实检验有用的信息摘要
    
    def is_useful(self) -> bool:    # 判断证据是否有用
```

**数据变化**:
- **工具执行后**: 包含raw results
- **LLM处理后**: 添加takeaways摘要（如果有用的话）

### 2.6 Results（结果）
**位置**: `defame/common/results.py`
**作用**: 工具执行的原始输出基类

**主要实现**:
- **SearchResults**: 搜索结果
  ```python
  @dataclass
  class SearchResults(Results):
      sources: list[Source]       # 搜索到的来源列表
      query: Query               # 触发搜索的查询
  ```

### 2.7 Label（标签）
**位置**: `defame/common/label.py`
**作用**: 表示验证结果的分类标签

```python
class Label(Enum):
    SUPPORTED = "supported"                    # 支持
    NEI = "not enough information"             # 信息不足
    REFUTED = "refuted"                       # 反驳
    CONFLICTING = "conflicting evidence"       # 证据冲突
    CHERRY_PICKING = "cherry-picking"          # 选择性引用
    REFUSED_TO_ANSWER = "error: refused to answer"  # 拒绝回答
    OUT_OF_CONTEXT = "out of context"         # 断章取义
    MISCAPTIONED = "miscaptioned"             # 错误标题
```

## 3. 数据在各模块中的变化

### 3.1 FactChecker主流程
```python
# 输入: Content | list[str | Item]
input_data → Content对象

# extract_claims()
Content → list[Claim]

# verify_claim() (对每个Claim)
Claim → Report(with Evidence) → (Label, metadata)

# check_content()
list[Label] → aggregated Label
```

### 3.2 ClaimExtractor模块

**输入**: `Content`对象
**输出**: `list[Claim]`对象

**处理步骤**:
1. **interpret()**: `Content` → `Content(+interpretation, +topic)`
2. **decompose()**: `Content` → `list[Claim]`
3. **decontextualize()**: `Claim` → `Claim(修改文本)`
4. **filter_check_worthy()**: `list[Claim]` → `list[Claim](过滤后)`

### 3.3 Planner模块

**输入**: `Report`对象（当前状态）
**输出**: `list[Action]`对象

**数据流**:
```
Report(current state) → Analysis → list[Action](next steps)
```

### 3.4 Actor模块

**输入**: `list[Action]`对象
**输出**: `list[Evidence]`对象

**处理过程**:
```
Action → Tool.perform() → Results → Evidence(raw + takeaways)
```

### 3.5 Judge模块

**输入**: `Report`对象（包含所有证据）
**输出**: `Label`对象

**判断过程**:
```
Report(with evidences) → LLM analysis → Label + reasoning
```

## 4. 工具系统的数据处理

### 4.1 Searcher工具
```
Search(query) → SearchPlatform.search() → SearchResults → Evidence
```

### 4.2 图像分析工具
```
ImageAction + Image → SpecificTool → SpecificResults → Evidence
```

### 4.3 文本提取工具
```
Extract(url) → WebScraper → TextContent → Evidence
```

## 5. 数据聚合机制

### 5.1 单声明验证
```
Claim → Report → iterative(Plan→Act→Judge) → Final Label
```

### 5.2 多声明聚合
```python
def aggregate_predictions(veracities: Sequence[Label]) -> Label:
    # 优先级聚合策略:
    # 1. 如果所有标签相同，返回该标签
    # 2. 如果有REFUSED_TO_ANSWER，返回REFUSED_TO_ANSWER
    # 3. 如果有REFUTED，返回REFUTED  
    # 4. 如果有CONFLICTING，返回CONFLICTING
    # 5. 如果有CHERRY_PICKING，返回CHERRY_PICKING
    # 6. 否则返回NEI
```

## 6. 数据持久化

### 6.1 Report保存
```python
Report.save_to(target_dir)  # 保存为Markdown格式的报告
```

### 6.2 统计信息
```python
meta["Statistics"] = {
    "Duration": float,        # 处理时长
    "Model": dict,           # LLM使用统计
    "Tools": dict            # 工具使用统计
}
```

## 7. 关键数据转换点

### 7.1 输入规范化
```
str/list[str]/Item → Content对象
```

### 7.2 声明提取
```
Content → Claim(原始) → Claim(去上下文化) → Claim(过滤后)
```

### 7.3 证据积累
```
空Report → +Action+Evidence → +Action+Evidence → ... → Final Report
```

### 7.4 结果输出
```
Report → Label + Justification + Statistics
```

## 8. 数据验证与错误处理

### 8.1 输入验证
- Content对象的完整性检查
- Claim的有效性验证
- Action参数的正确性校验

### 8.2 错误恢复
- LLM调用失败的重试机制
- 工具执行异常的fallback策略
- 结果解析错误的容错处理

### 8.3 数据一致性
- Report中Evidence与Action的对应关系
- Claim与Context的引用完整性
- 时间戳和ID的唯一性保证

## 9. 性能优化相关的数据设计

### 9.1 缓存机制
- SearchResults的查询缓存
- LLM响应的结果缓存
- 工具输出的临时存储

### 9.2 并行处理支持
- Claim列表的并行验证设计
- Action批量执行的数据结构
- Evidence的线程安全访问

### 9.3 内存管理
- 大型Results的懒加载
- 临时数据的及时清理
- 多媒体内容的引用管理

这个数据流转文档展示了DEFAME框架中数据从原始输入到最终输出的完整变化过程，以及各个模块如何协同处理和转换数据。