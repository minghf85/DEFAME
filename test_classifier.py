#!/usr/bin/env python3
"""测试 Classifier 类的功能"""

from defame.extension import Classifier
from defame.common.claim import Claim
from defame.common.content import Content
from defame.common.label import DifficultyLabel
from ezmm import MultimodalSequence
from ezmm import Image
def test_classifier():
    """测试分类器的基本功能"""
    img = Image("in/example/2025-01-18_298.png")
    # 创建测试声明
    test_claims = [
        # 简单声明
        Claim(f"{img.reference}, shows a 'Die in' climate protest in Austria, where protesters got inside of body bags to signify the catastrophic impact that current climate policy could have on the world.", id="easy_claim"),
        
        # # 中等难度声明
        # Claim("2023年全球平均气温比20世纪平均气温高出1.2摄氏度", id="medium_claim"),
        
        # # 困难声明
        # Claim("某种新药的临床试验结果显示其对罕见疾病的治疗效果比现有药物提高30%", id="hard_claim")
    ]
    
    print("创建分类器...")
    try:
        # 创建分类器实例
        classifier = Classifier()
        print("分类器创建成功")
        
        # 测试单个声明分类
        print("\n测试单个声明分类:")
        for claim in test_claims:
            print(f"\n声明: {claim}")
            difficulty = classifier.classify_difficulty(claim)
            if difficulty:
                print(f"分类结果: {difficulty.value}")
            else:
                print("分类失败")
        
        # 测试批量分类
        print("\n测试批量分类:")
        batch_results = classifier.classify_batch(test_claims)
        for i, (claim, result) in enumerate(zip(test_claims, batch_results)):
            result_str = result.value if result else "失败"
            print(f"声明 {i+1}: {result_str}")
        
        # 获取统计信息
        print("\n分类统计信息:")
        stats = classifier.get_classification_stats(test_claims)
        for key, value in stats.items():
            print(f"{key}: {value}")
            
    except Exception as e:
        print(f"测试过程中发生错误: {e}")
        print("请确保:")
        print("1. 已正确配置 OpenAI API 密钥")
        print("2. 网络连接正常")
        print("3. 所有依赖包已安装")

def test_multimodal_claim():
    """测试多模态声明分类"""
    print("\n测试多模态声明分类:")
    
    try:
        # 创建包含文本的声明（模拟多模态内容）
        content = Content("这张图片显示了一个不寻常的天文现象")
        multimodal_claim = Claim(context=content, id="multimodal_claim")
        
        classifier = Classifier()
        difficulty = classifier.classify_difficulty(multimodal_claim)
        
        if difficulty:
            print(f"多模态声明分类结果: {difficulty.value}")
        else:
            print("多模态声明分类失败")
            
    except Exception as e:
        print(f"多模态测试失败: {e}")

if __name__ == "__main__":
    print("开始测试 Classifier 类")
    print("=" * 50)
    
    test_classifier()
    # test_multimodal_claim()
    
    print("\n测试完成!")
