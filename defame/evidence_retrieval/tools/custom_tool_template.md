# DEFAME 自定义工具创建指南

本文档提供了在 DEFAME 事实检查系统中创建自定义工具的完整指南和模板。

## 概述

DEFAME 使用模块化的工具系统来扩展事实检查能力。每个工具都可以执行一个或多个动作（Actions），并返回证据（Evidence）用于事实验证。

## 创建步骤

### 0. 实现集成（可选）

如果您的工具需要调用外部资源，请在 `defame/evidence_retrieval/integrations` 中实现相应的集成模块。

### 1. 实现工具和动作

在 `defame/evidence_retrieval/tools` 中创建新文件，实现继承自 `Tool` 基类的工具。

### 2. 注册工具

在 `defame/evidence_retrieval/tools/__init__.py` 中注册您的工具。

### 3. 添加工具配置

在 DEFAME 超参数或配置文件中指定工具。

### 4. 注册到基准测试（可选）

如果要在基准测试中使用工具，需要在相应的基准测试文件中注册。

## 详细实现

### 模板文件

以下是完整的工具实现模板：

#### 1. 动作类（Action）

```python
from defame.common import Action

class YourCustomAction(Action):
    """
    简短描述您的动作的功能。这个描述会直接用于 LLM 规划模块，
    告知 LLM 这个动作的用途和使用方法。
    """
    name = "your_action_name"
    requires_image = False  # 如果动作需要图像输入，设置为 True
    
    def __init__(self, param1: str, param2: int = None, image: str = None):
        """
        动作的初始化方法。这里的文档字符串也会被用于 LLM。
        
        @param param1: 必需参数的描述
        @param param2: 可选参数的描述
        @param image: 如果需要图像，添加此参数
        """
        self._save_parameters(locals())  # 必须调用此方法保存参数
        
        # 验证参数
        if param1 is None:
            raise ValueError("param1 是必需的")
            
        self.param1 = param1
        self.param2 = param2
        
        if image is not None:
            from ezmm import Image
            self.image = Image(reference=image)
    
    def __str__(self):
        return f'{self.name}({self.param1}, {self.param2})'
    
    def __eq__(self, other):
        return (isinstance(other, YourCustomAction) and 
                self.param1 == other.param1 and 
                self.param2 == other.param2)
    
    def __hash__(self):
        return hash((self.name, self.param1, self.param2))
```

#### 2. 结果类（Results）

```python
from dataclasses import dataclass, field
from typing import Optional
from defame.common.results import Results

@dataclass
class YourCustomResults(Results):
    """存储工具执行结果的数据类"""
    source: str
    result_data: any
    confidence: Optional[float] = None
    text: str = field(init=False)
    
    def __post_init__(self):
        self.text = str(self)
    
    def __str__(self):
        return f'来源: {self.source}\n结果: {self.result_data}'
    
    def is_useful(self) -> Optional[bool]:
        """判断结果是否有用"""
        return self.result_data is not None and self.confidence > 0.5
```

#### 3. 工具类（Tool）

```python
from typing import Optional
import torch
from defame.evidence_retrieval.tools.tool import Tool
from defame.common.results import Results
from defame.common import logger

class YourCustomTool(Tool):
    """
    您的自定义工具类的描述。
    解释这个工具的主要功能和用途。
    """
    name = "your_custom_tool"
    actions = [YourCustomAction]  # 此工具支持的动作列表
    
    def __init__(self, custom_config: dict = None, **kwargs):
        """
        初始化自定义工具
        
        @param custom_config: 工具特定的配置
        """
        super().__init__(**kwargs)
        self.custom_config = custom_config or {}
        
        # 初始化工具所需的资源
        self._initialize_resources()
    
    def _initialize_resources(self):
        """初始化工具所需的资源（模型、API客户端等）"""
        try:
            # 例如：初始化模型或API客户端
            logger.info(f"初始化 {self.name} 工具")
            # self.model = load_model()
            # self.api_client = APIClient(api_key=self.custom_config.get('api_key'))
        except Exception as e:
            logger.error(f"初始化 {self.name} 工具失败: {e}")
            raise
    
    def _perform(self, action: YourCustomAction) -> Results:
        """
        执行动作的核心方法
        
        @param action: 要执行的动作
        @return: 执行结果
        """
        try:
            logger.info(f"执行动作: {action}")
            
            # 根据动作类型执行相应的逻辑
            if isinstance(action, YourCustomAction):
                return self._handle_custom_action(action)
            else:
                raise ValueError(f"不支持的动作类型: {type(action)}")
                
        except Exception as e:
            logger.error(f"执行动作 {action} 失败: {e}")
            raise
    
    def _handle_custom_action(self, action: YourCustomAction) -> YourCustomResults:
        """
        处理自定义动作的具体实现
        
        @param action: 自定义动作
        @return: 处理结果
        """
        # 实现您的核心逻辑
        source = f"CustomTool-{action.param1}"
        
        # 示例：处理图像（如果需要）
        if hasattr(action, 'image') and action.image:
            # 处理图像逻辑
            pass
        
        # 示例：调用外部API或模型
        result_data = self._process_data(action.param1, action.param2)
        
        # 计算置信度
        confidence = self._calculate_confidence(result_data)
        
        return YourCustomResults(
            source=source,
            result_data=result_data,
            confidence=confidence
        )
    
    def _process_data(self, param1: str, param2: int) -> any:
        """处理数据的具体实现"""
        # 实现您的数据处理逻辑
        return f"处理结果: {param1} - {param2}"
    
    def _calculate_confidence(self, result_data: any) -> float:
        """计算结果的置信度"""
        # 实现置信度计算逻辑
        return 0.8
    
    def _summarize(self, results: Results, **kwargs) -> str:
        """
        总结结果（可选重写）
        
        @param results: 要总结的结果
        @return: 总结文本
        """
        if hasattr(results, 'confidence') and results.confidence:
            return f"工具 {self.name} 的结果置信度为 {results.confidence:.2f}"
        return f"工具 {self.name} 完成处理"
```

### 实际示例

让我为您创建一个具体的示例 - 一个简单的文本分析工具：

```python
# defame/evidence_retrieval/tools/text_analyzer.py

import re
from dataclasses import dataclass, field
from typing import Optional, Dict, List
from defame.common import Action, logger
from defame.common.results import Results
from defame.evidence_retrieval.tools.tool import Tool

class AnalyzeText(Action):
    """分析文本内容，提取关键信息如情感、关键词、实体等。
    适用于分析声明文本或网页内容。"""
    name = "analyze_text"
    
    def __init__(self, text: str, analysis_type: str = "sentiment"):
        """
        @param text: 要分析的文本内容
        @param analysis_type: 分析类型，可选：'sentiment'（情感分析）、
            'keywords'（关键词提取）、'entities'（实体识别）
        """
        self._save_parameters(locals())
        
        if not text or not text.strip():
            raise ValueError("文本不能为空")
        
        valid_types = ['sentiment', 'keywords', 'entities']
        if analysis_type not in valid_types:
            raise ValueError(f"分析类型必须是以下之一: {valid_types}")
        
        self.text = text.strip()
        self.analysis_type = analysis_type
    
    def __str__(self):
        return f'{self.name}(text="{self.text[:50]}...", type={self.analysis_type})'
    
    def __eq__(self, other):
        return (isinstance(other, AnalyzeText) and 
                self.text == other.text and 
                self.analysis_type == other.analysis_type)
    
    def __hash__(self):
        return hash((self.name, self.text, self.analysis_type))

@dataclass
class TextAnalysisResults(Results):
    """文本分析结果"""
    source: str
    analysis_type: str
    sentiment_score: Optional[float] = None
    keywords: Optional[List[str]] = None
    entities: Optional[Dict[str, List[str]]] = None
    confidence: Optional[float] = None
    text: str = field(init=False)
    
    def __post_init__(self):
        self.text = str(self)
    
    def __str__(self):
        result_parts = [f"文本分析结果 ({self.analysis_type}):"]
        
        if self.sentiment_score is not None:
            sentiment = "积极" if self.sentiment_score > 0 else "消极" if self.sentiment_score < 0 else "中性"
            result_parts.append(f"情感: {sentiment} (分数: {self.sentiment_score:.2f})")
        
        if self.keywords:
            result_parts.append(f"关键词: {', '.join(self.keywords[:5])}")
        
        if self.entities:
            for entity_type, entity_list in self.entities.items():
                if entity_list:
                    result_parts.append(f"{entity_type}: {', '.join(entity_list[:3])}")
        
        if self.confidence:
            result_parts.append(f"置信度: {self.confidence:.2f}")
        
        return '\n'.join(result_parts)
    
    def is_useful(self) -> Optional[bool]:
        return (self.sentiment_score is not None or 
                self.keywords or 
                self.entities) and (self.confidence or 0) > 0.3

class TextAnalyzer(Tool):
    """
    文本分析工具，可以进行情感分析、关键词提取和实体识别。
    适用于分析声明内容、新闻文章等文本信息。
    """
    name = "text_analyzer"
    actions = [AnalyzeText]
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 简单的情感词典（实际使用中可以使用更复杂的模型）
        self.positive_words = {'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic'}
        self.negative_words = {'bad', 'terrible', 'awful', 'horrible', 'disgusting', 'hate'}
        
        logger.info("文本分析工具初始化完成")
    
    def _perform(self, action: AnalyzeText) -> Results:
        logger.info(f"执行文本分析: {action.analysis_type}")
        
        if action.analysis_type == "sentiment":
            return self._analyze_sentiment(action)
        elif action.analysis_type == "keywords":
            return self._extract_keywords(action)
        elif action.analysis_type == "entities":
            return self._extract_entities(action)
        else:
            raise ValueError(f"不支持的分析类型: {action.analysis_type}")
    
    def _analyze_sentiment(self, action: AnalyzeText) -> TextAnalysisResults:
        """简单的情感分析"""
        words = action.text.lower().split()
        positive_count = sum(1 for word in words if word in self.positive_words)
        negative_count = sum(1 for word in words if word in self.negative_words)
        
        # 简单的情感分数计算
        total_sentiment_words = positive_count + negative_count
        if total_sentiment_words == 0:
            sentiment_score = 0.0
            confidence = 0.3
        else:
            sentiment_score = (positive_count - negative_count) / len(words)
            confidence = min(total_sentiment_words / len(words) * 2, 1.0)
        
        return TextAnalysisResults(
            source="TextAnalyzer-Sentiment",
            analysis_type="sentiment",
            sentiment_score=sentiment_score,
            confidence=confidence
        )
    
    def _extract_keywords(self, action: AnalyzeText) -> TextAnalysisResults:
        """简单的关键词提取"""
        # 移除标点符号和常见停用词
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'}
        
        words = re.findall(r'\b\w+\b', action.text.lower())
        words = [word for word in words if word not in stop_words and len(word) > 2]
        
        # 简单的词频统计
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # 提取最频繁的词作为关键词
        keywords = sorted(word_freq.keys(), key=lambda x: word_freq[x], reverse=True)[:10]
        
        confidence = min(len(keywords) / 10, 1.0)
        
        return TextAnalysisResults(
            source="TextAnalyzer-Keywords",
            analysis_type="keywords",
            keywords=keywords,
            confidence=confidence
        )
    
    def _extract_entities(self, action: AnalyzeText) -> TextAnalysisResults:
        """简单的实体识别"""
        entities = {
            "dates": re.findall(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', action.text),
            "emails": re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', action.text),
            "urls": re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', action.text),
            "numbers": re.findall(r'\b\d+\b', action.text)
        }
        
        # 过滤空的实体类型
        entities = {k: v for k, v in entities.items() if v}
        
        confidence = min(len(entities) / 4, 1.0) if entities else 0.2
        
        return TextAnalysisResults(
            source="TextAnalyzer-Entities",
            analysis_type="entities",
            entities=entities,
            confidence=confidence
        )
```

### 注册工具

在 `defame/evidence_retrieval/tools/__init__.py` 中添加：

```python
# 导入您的工具
from .text_analyzer import TextAnalyzer, AnalyzeText

# 添加到注册表
TOOL_REGISTRY = [
    # ... 现有工具 ...
    TextAnalyzer,
]

ACTION_REGISTRY = {
    # ... 现有动作 ...
    AnalyzeText,
}
```

### 配置工具

在配置文件中添加工具配置：

```yaml
# config/your_config.yaml
tools:
  text_analyzer:
    enabled: true
    # 工具特定配置
```

## 注意事项

1. **文档字符串**：动作的类级别和 `__init__` 方法的文档字符串会直接用于 LLM，请确保描述清晰准确。

2. **参数验证**：在动作的 `__init__` 方法中验证参数的有效性。

3. **错误处理**：实现适当的错误处理和日志记录。

4. **性能考虑**：对于耗时操作，考虑添加缓存或优化机制。

5. **测试**：为您的工具创建适当的测试用例。

6. **依赖管理**：如果工具需要额外的依赖，请更新 `requirements.txt`。

## 测试您的工具

创建测试文件来验证工具功能：

```python
# tests/test_text_analyzer.py
import pytest
from defame.evidence_retrieval.tools.text_analyzer import TextAnalyzer, AnalyzeText

def test_text_analyzer():
    tool = TextAnalyzer()
    action = AnalyzeText(text="This is a great and wonderful day!", analysis_type="sentiment")
    result = tool.perform(action)
    
    assert result.result.sentiment_score > 0
    assert result.result.confidence > 0
```

这个模板提供了创建自定义工具的完整框架。您可以根据具体需求修改和扩展这个模板。
