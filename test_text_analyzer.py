# test_text_analyzer.py
"""
基于spaCy的文本分析工具测试脚本
用于验证自定义工具的功能
"""

import sys
import os

# 添加项目根目录到路径
sys.path.append('/root/autodl-tmp/DEFAME')

def test_spacy_availability():
    """测试spaCy可用性"""
    print("🔍 测试spaCy环境...")
    
    try:
        import spacy
        print(f"✅ spaCy版本: {spacy.__version__}")
        
        # 检查可用模型
        available_models = spacy.util.get_installed_models()
        if available_models:
            print(f"✅ 可用模型: {available_models}")
        else:
            print("⚠️  没有预训练模型，将使用基础英文处理器")
        
        # 尝试加载模型
        try:
            nlp = spacy.load("en_core_web_sm")
            print("✅ 成功加载 en_core_web_sm 模型")
        except OSError:
            try:
                from spacy.lang.en import English
                nlp = English()
                print("✅ 使用spaCy基础英文处理器")
            except Exception as e:
                print(f"❌ spaCy初始化失败: {e}")
                return False
        
        return True
        
    except ImportError as e:
        print(f"❌ spaCy导入失败: {e}")
        return False


def test_sentiment_analysis():
    """测试情感分析功能"""
    print("\n🔍 测试情感分析功能...")
    
    from defame.evidence_retrieval.tools.text_analyzer import TextAnalyzer, AnalyzeText
    
    tool = TextAnalyzer()
    
    test_cases = [
        ("This is a great and wonderful day! I love this amazing product.", "积极"),
        ("This is terrible and awful. I hate this disappointing product.", "消极"),
        ("The weather report indicates cloudy conditions today.", "中性"),
        ("I'm absolutely thrilled with the fantastic results!", "积极"),
        ("This is completely false and misleading information.", "消极")
    ]
    
    for text, expected in test_cases:
        action = AnalyzeText(text=text, analysis_type="sentiment")
        result = tool.perform(action, summarize=True)
        
        print(f"\n📝 测试文本: {text[:50]}...")
        print(f"📊 分析结果:\n{result.raw}")
        print(f"💭 总结: {result.takeaways}")
        print(f"✅ 预期: {expected}, 实际: {getattr(result.raw, 'sentiment_label', '未知')}")


def test_keyword_extraction():
    """测试关键词提取功能"""
    print("\n🔍 测试关键词提取功能...")
    
    from defame.evidence_retrieval.tools.text_analyzer import TextAnalyzer, AnalyzeText
    
    tool = TextAnalyzer()
    
    test_text = """
    Artificial intelligence and machine learning are transforming technology industries worldwide. 
    Deep learning algorithms enable advanced computer vision and natural language processing capabilities. 
    These revolutionary technologies are dramatically improving healthcare diagnostics, financial analysis, 
    and educational systems across the globe.
    """
    
    action = AnalyzeText(text=test_text, analysis_type="keywords")
    result = tool.perform(action, summarize=True)
    
    print(f"📝 测试文本: {test_text.strip()[:100]}...")
    print(f"📊 关键词提取结果:\n{result.raw}")
    print(f"💭 总结: {result.takeaways}")


def test_entity_recognition():
    """测试实体识别功能"""
    print("\n🔍 测试实体识别功能...")
    
    from defame.evidence_retrieval.tools.text_analyzer import TextAnalyzer, AnalyzeText
    
    tool = TextAnalyzer()
    
    test_text = """
    Contact Dr. John Smith at john.smith@university.edu or call 555-123-4567. 
    Visit our website at https://www.example.com for more information. 
    The conference is scheduled for December 25, 2024 in New York City. 
    The project budget is $50,000 and involves 100 participants.
    Apple Inc. and Microsoft Corporation are major technology companies.
    """
    
    action = AnalyzeText(text=test_text, analysis_type="entities")
    result = tool.perform(action, summarize=True)
    
    print(f"📝 测试文本: {test_text.strip()[:100]}...")
    print(f"📊 实体识别结果:\n{result.raw}")
    print(f"💭 总结: {result.takeaways}")


def test_comprehensive_analysis():
    """测试综合分析功能"""
    print("\n🔍 测试综合分析功能...")
    
    from defame.evidence_retrieval.tools.text_analyzer import TextAnalyzer, AnalyzeText
    
    tool = TextAnalyzer()
    
    test_text = """
    BREAKING NEWS: Revolutionary breakthrough in artificial intelligence research! 
    Scientists at Stanford University have developed amazing machine learning algorithms 
    that could transform healthcare. Dr. Sarah Johnson (sarah.johnson@stanford.edu) 
    announced the fantastic results on January 15, 2024. The $2.5 million project 
    took 3 years to complete. Contact the research team at 650-555-0123. 
    This incredible advancement represents a major milestone in AI development!
    """
    
    action = AnalyzeText(text=test_text, analysis_type="all")
    result = tool.perform(action, summarize=True)
    
    print(f"📝 测试文本: {test_text.strip()[:100]}...")
    print(f"📊 综合分析结果:\n{result.raw}")
    print(f"💭 总结: {result.takeaways}")


def test_error_handling():
    """测试错误处理"""
    print("\n🔍 测试错误处理...")
    
    from defame.evidence_retrieval.tools.text_analyzer import AnalyzeText
    
    try:
        # 测试空文本
        empty_action = AnalyzeText(text="", analysis_type="sentiment")
        print("❌ 空文本测试失败 - 应该抛出异常")
    except ValueError as e:
        print(f"✅ 空文本错误处理正确: {e}")
    
    try:
        # 测试无效分析类型
        invalid_action = AnalyzeText(text="Test text", analysis_type="invalid")
        print("❌ 无效类型测试失败 - 应该抛出异常")
    except ValueError as e:
        print(f"✅ 无效类型错误处理正确: {e}")


def test_spacy_features():
    """测试spaCy特定功能"""
    print("\n🔍 测试spaCy特定功能...")
    
    from defame.evidence_retrieval.tools.text_analyzer import TextAnalyzer
    
    tool = TextAnalyzer()
    
    if tool.nlp:
        print("✅ spaCy模型已加载")
        
        # 测试基本NLP功能
        test_text = "Apple Inc. is planning to release new products in 2024."
        doc = tool.nlp(test_text)
        
        print(f"📝 测试文本: {test_text}")
        print("🏷️  词性标注:")
        for token in doc:
            if not token.is_space:
                print(f"  {token.text}: {token.pos_} ({token.lemma_})")
        
        print("🏢 命名实体:")
        for ent in doc.ents:
            print(f"  {ent.text}: {ent.label_}")
            
    else:
        print("⚠️  spaCy模型未加载，使用基础功能")


def main():
    """主测试函数"""
    print("=" * 70)
    print("🚀 开始测试基于spaCy的文本分析工具")
    print("=" * 70)
    
    try:
        # 首先测试spaCy环境
        spacy_ok = test_spacy_availability()
        
        # 运行功能测试
        test_sentiment_analysis()
        test_keyword_extraction()
        test_entity_recognition()
        test_comprehensive_analysis()
        test_error_handling()
        
        if spacy_ok:
            test_spacy_features()
        
        print("\n" + "=" * 70)
        print("✅ 所有测试完成!")
        print("=" * 70)
        
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
