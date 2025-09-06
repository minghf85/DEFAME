# Classifier 使用说明

## 概述

`Classifier` 是一个使用 GPT-4o 模型对多模态声明内容进行事实检查难度分类的模块。它可以将声明分类为三个难度等级：简单(EASY)、中等(MEDIUM)、困难(HARD)。

## 主要功能

### 1. 难度分类
- **EASY**: 可以使用单一可靠来源或直接的事实检查方法高信度验证的声明
- **MEDIUM**: 需要交叉引用多个可靠来源或涉及一定复杂性，但仍可通过适度努力解决的声明  
- **HARD**: 由于缺乏可靠来源、信息冲突或需要专业知识而难以验证的声明

### 2. 支持的内容类型
- 纯文本声明
- 多模态声明（包含图像、视频、音频的声明）

## 使用方法

### 基本用法

```python
from defame.extension.classify import Classifier
from defame.common.claim import Claim

# 创建分类器实例
classifier = Classifier()

# 创建声明
claim = Claim("北京是中国的首都")

# 进行分类
difficulty = classifier.classify_difficulty(claim)
print(f"分类结果: {difficulty.value}")  # 输出: easy
```

### 批量分类

```python
# 准备多个声明
claims = [
    Claim("太阳从东方升起"),
    Claim("某种新药的临床试验结果显示疗效显著"),
    Claim("2023年全球平均气温创历史新高")
]

# 批量分类
results = classifier.classify_batch(claims)

# 查看结果
for claim, result in zip(claims, results):
    if result:
        print(f"'{claim}' -> {result.value}")
    else:
        print(f"'{claim}' -> 分类失败")
```

### 获取统计信息

```python
# 获取分类统计
stats = classifier.get_classification_stats(claims)
print(f"总计: {stats['total']}")
print(f"简单: {stats['easy']}")
print(f"中等: {stats['medium']}")  
print(f"困难: {stats['hard']}")
print(f"失败: {stats['failed']}")
```

### 自定义模型

```python
# 使用不同的模型
classifier = Classifier(model_name="openai:gpt-4o-mini")

# 或使用其他支持的模型
classifier = Classifier(model_name="deepseek:deepseek-chat")
```

## 配置要求

### API 密钥配置

确保在 `config/api_keys.yaml` 中配置了相应的 API 密钥：

```yaml
openai_api_key: "your_openai_api_key_here"
deepseek_api_key: "your_deepseek_api_key_here"  # 如果使用 DeepSeek
```

### 模型支持

查看 `config/available_models.csv` 了解所有支持的模型。

## 错误处理

分类器包含完善的错误处理机制：

1. **API 错误**: 自动重试最多3次
2. **解析错误**: 返回 `None` 并记录错误信息
3. **网络错误**: 捕获并提示相应错误信息

## 注意事项

1. **成本考虑**: 每次分类调用都会产生 API 费用，建议合理使用
2. **准确性**: 分类结果基于模型判断，可能不完全准确
3. **多模态支持**: 当前版本对多模态内容的处理可能有限
4. **语言支持**: 主要针对中英文内容优化

## 扩展功能

### 自定义分类提示

可以通过继承 `Classifier` 类来自定义分类逻辑：

```python
class CustomClassifier(Classifier):
    def _create_classification_system_prompt(self) -> str:
        # 自定义系统提示
        return "Your custom system prompt here..."
    
    def _parse_classification_response(self, response: str):
        # 自定义响应解析逻辑
        # ... your custom logic
        return super()._parse_classification_response(response)
```

## 示例输出

```
声明: "北京是中国的首都"
分类结果: EASY

声明: "2023年全球CO2排放量比2022年增长了2.1%"  
分类结果: MEDIUM

声明: "某研究表明新发现的量子材料可以实现室温超导"
分类结果: HARD
```
