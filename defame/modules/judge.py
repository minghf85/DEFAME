import dataclasses

from defame.common import Report, logger, Model, Prompt, Label
from defame.common.label import DEFAULT_LABEL_DEFINITIONS
from defame.prompts.prompts import JudgePrompt, JudgeNaively, JudgeMinimal
import json


@dataclasses.dataclass()
class FinalAnswer:
    response: str
    answer: str


class Judge:
    """Determines the truthfulness of a claim given a collection of evidence."""

    def __init__(self,
                 llm: Model,
                 classes: list[Label],
                 class_definitions: dict[Label, str] = None,
                 extra_rules: str = None):
        self.llm = llm
        self.classes = set(classes)

        if Label.NEI not in class_definitions:
            class_definitions[Label.NEI] = DEFAULT_LABEL_DEFINITIONS[Label.NEI]
        self.class_definitions = class_definitions

        self.extra_rules = extra_rules
        self.max_retries = 5
        self.latest_reasoning = None

    def judge(self, doc: Report, is_final: bool = True) -> Label:
        classes = self.classes.copy()

        # If this is a non-final judgement (i.e. there are follow-up retrievals/actions allowed)
        # enable to predict NEI (otherwise fact-check would always end here)
        if not is_final:
            classes.add(Label.NEI)

        prompt = JudgePrompt(doc, classes, self.class_definitions, self.extra_rules)
        return self._generate_verdict(prompt)

    def judge_naively(self, doc: Report) -> Label:
        prompt = JudgeNaively(doc.claim, self.classes, self.class_definitions)
        return self._generate_verdict(prompt)

    def judge_minimally(self, doc: Report) -> Label:
        prompt = JudgeMinimal(doc.claim, self.classes, self.class_definitions)
        return self._generate_verdict(prompt)

    def _generate_verdict(self, prompt: Prompt) -> Label:
        response = self.llm.generate(prompt)

        # 处理 response 为 None 的情况
        if response is None:
            logger.warning(f"Error while generating verdict. Defaulting to REFUSED.")
            self.latest_reasoning = ""
            return Label.REFUSED_TO_ANSWER

        # 处理 response 为字符串的情况
        if isinstance(response, str):
            # 尝试将字符串解析为JSON，或者根据你的业务逻辑构建一个字典
            try:
                # 假设你的LLM有时会返回JSON字符串
                response_dict = json.loads(response)
            except json.JSONDecodeError:
                # 如果解析失败，说明是普通文本，可能需要根据文本内容推断verdict
                # 这里需要你根据实际情况设计逻辑，例如：
                response_dict = {
                    "verdict": Label.REFUSED_TO_ANSWER, # 或者根据response文本内容进行判断
                    "response": response
                }
            # 将response替换为处理后的字典
            response = response_dict

        # 现在 response 应该是一个字典了
        if not response.get("verdict"): # 使用 .get 方法避免 KeyError
            logger.warning(f"Error while generating verdict or verdict is empty. Defaulting to REFUSED.")
            self.latest_reasoning = response.get("response", "")
            return Label.REFUSED_TO_ANSWER

        self.latest_reasoning = response.get("response", "")
        return response["verdict"]

    def get_latest_reasoning(self) -> str:
        return self.latest_reasoning
