"""难度分类提示类，避免循环导入"""

from typing import Optional
from defame.common import Prompt, Claim
from defame.common.label import DifficultyLabel, DEFAULT_DIFFICULTY_DEFINITIONS


class ClassifyDifficultyPrompt(Prompt):
    template_file_path = "defame/prompts/classify_difficulty.md"

    def __init__(self, claim: Claim):
        # 从label.py中获取难度定义并格式化
        difficulty_definitions = "\n".join([
            f"- **{label.value.upper()}**: {definition}"
            for label, definition in DEFAULT_DIFFICULTY_DEFINITIONS.items()
        ])
        
        placeholder_targets = {
            "[CLAIM]": str(claim),
            "[DIFFICULTY_DEFINITIONS]": difficulty_definitions,
        }
        super().__init__(placeholder_targets=placeholder_targets)

    def extract(self, response: str) -> Optional[DifficultyLabel]:
        """
        从模型响应中提取难度分类结果
        
        Args:
            response: 模型的原始响应
            
        Returns:
            DifficultyLabel: 解析后的难度标签，如果解析失败返回None
        """
        # 清理响应文本
        cleaned_response = response.strip().upper()
        
        # 尝试匹配难度标签
        for difficulty_label in DifficultyLabel:
            if difficulty_label.value.upper() in cleaned_response:
                return difficulty_label
        
        # 如果没有找到匹配的标签，返回None
        return None
