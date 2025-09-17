"""Specialized Agent that can use a specific tool or several tools."""
from defame.common import Report, logger, Model, Prompt, Label
from defame.evidence_retrieval.tools import Tool
from defame.common.claim import Claim
class SAgent:
    def __init__(self, name: str, description: str, tools: list[Tool], claim: Claim, label: Label):
        self.name = name
        self.description = description
        self.tools = tools
        self.goal = "Use the tools to find evidence to verify the claim." # 由一个Leader为各个SAgent分配的任务
        self.claim = claim
        self.labels = label
