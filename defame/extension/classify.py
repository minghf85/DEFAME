"""使用gpt-4o只通过多模态声明内容进行事实检查难度分类的模块"""

from defame.common.label import *
from defame.common.claim import Claim
from defame.common.modeling import make_model
from defame.prompts.classify_difficulty_prompt import ClassifyDifficultyPrompt
from typing import Optional


class Classifier:
    """仅通过声明内容进行难度分类的模块"""
    
    def __init__(self, model_name: str = "OPENAI:gpt-4o-2024-08-06"):
        """
        初始化分类器
        
        Args:
            model_name: 模型名称，默认使用gpt-4o
        """
        self.model = make_model(model_name)

    def classify_difficulty(self, claim: Claim) -> Optional[DifficultyLabel]:
        """
        对声明进行难度分类
        
        Args:
            claim: 需要分类的声明对象
            
        Returns:
            DifficultyLabel: 分类结果 (EASY, MEDIUM, HARD)，如果分类失败返回None
        """
        try:
            # 创建分类提示
            prompt = ClassifyDifficultyPrompt(claim)
            
            # 使用模型进行分类
            response = self.model.generate(
                prompt=prompt,
                temperature=0.1,  # 使用低温度以获得更一致的结果
                max_attempts=3
            )
            
            if response is None:
                return None
            
            # 使用prompt的extract方法解析响应
            if isinstance(response, dict) and 'response' in response:
                # 如果response是dict，提取实际的响应文本
                return prompt.extract(response['response'])
            else:
                # 如果response是字符串，直接解析
                return prompt.extract(str(response))
            
        except Exception as e:
            print(f"分类过程中发生错误: {e}")
            return None
    
    def classify_batch(self, claims: list[Claim]) -> list[Optional[DifficultyLabel]]:
        """
        批量分类多个声明
        
        Args:
            claims: 声明列表
            
        Returns:
            list[Optional[DifficultyLabel]]: 分类结果列表
        """
        results = []
        for claim in claims:
            result = self.classify_difficulty(claim)
            results.append(result)
        return results
    
    def get_classification_stats(self, claims: list[Claim]) -> dict:
        """
        获取分类统计信息
        
        Args:
            claims: 声明列表
            
        Returns:
            dict: 包含各难度等级数量的统计信息
        """
        classifications = self.classify_batch(claims)
        
        stats = {
            "total": len(claims),
            "easy": 0,
            "medium": 0,
            "hard": 0,
            "failed": 0
        }
        
        for classification in classifications:
            if classification is None:
                stats["failed"] += 1
            else:
                stats[classification.value] += 1
        
        return stats