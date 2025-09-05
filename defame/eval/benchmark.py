import random
from abc import ABC
from pathlib import Path
from typing import MutableSequence, Iterable, Iterator

from defame.common import logger, Label, Action
from defame.utils.console import bold, green, red
from config.globals import random_seed, data_root_dir


class Benchmark(ABC, Iterable):
    """Abstract class for all benchmarks. Inherit from this class when you want to add
    a new benchmark."""
    name: str
    shorthand: str  # Used for naming files/directories

    data: MutableSequence[dict]  # Each element is of the form {"id": ..., "content": ..., "label": ...}

    class_mapping: dict[str, Label]  # Maps the benchmark-specific class/label to the standard Label class
    class_definitions: dict[Label, str]  # Explains (to the LLM) the meaning of each class/label

    file_path: Path

    available_actions: list[Action]  # If none, all actions are allowed

    extra_prepare_rules: str = None  # Additional, benchmark-specific instructions to guide LLM's initial reasoning
    extra_plan_rules: str = None  # Additional, benchmark-specific instructions to guide LLM's action planning
    extra_judge_rules: str = None  # Additional, benchmark-specific instructions to guide LLM's verdict prediction

    def __init__(self, variant: str, file_path: Path | str = None):
        """
        @param variant: The split to use, usually one of 'train', 'dev', 'test', or `val`
        @param file_path: The path to the file (relative to the base data dir) that contains
            the data of the specified split.
        """
        self.variant = variant
        self.full_name = f"{self.name} ({variant})"

        if file_path:
            self.file_path = data_root_dir / file_path
            if not self.file_path.exists():
                raise ValueError(f"Unable to locate {self.name} at '{self.file_path.as_posix()}'. "
                                 f"See README.md for setup instructions.")

        self.data = self._load_data()

    def _load_data(self) -> MutableSequence[dict]:
        """Reads the data from the disk and turns them into ready-to-use instances."""
        raise NotImplementedError()

    @property
    def labels(self) -> list[Label]:
        """Returns the ground truth labels of this dataset as a list."""
        labels = []
        for instance in self:
            labels.append(instance["label"])
        return labels

    def get_classes(self) -> list[Label]:
        """Returns a list of distinct labels representing the classes occurring in this dataset."""
        return list(self.class_definitions.keys())

    def shuffle(self):
        """Reorders the samples randomly."""
        random.seed(random_seed)
        random.shuffle(self.data)

    def get_by_id(self, claim_id: str):
        """Returns the instance with the given ID (different from the instance's index)."""
        # 尝试不同的ID匹配方式
        for instance in self:
            instance_id = instance["id"]
            logger.debug(f"Comparing claim_id '{claim_id}' (type: {type(claim_id)}) with instance_id '{instance_id}' (type: {type(instance_id)})")
            # 直接匹配
            if instance_id == claim_id:
                return instance
            # 字符串匹配
            if str(instance_id) == str(claim_id):
                return instance
            # 如果都是数字，尝试数值匹配
            try:
                if int(instance_id) == int(claim_id):
                    return instance
            except (ValueError, TypeError):
                pass
        
        # 如果找不到，提供更详细的错误信息
        available_ids = [str(instance["id"]) for instance in self.data[:10]]  # 只显示前10个
        if len(self.data) > 10:
            available_ids.append("...")
        raise ValueError(f"Benchmark does not contain any instance with ID '{claim_id}' (type: {type(claim_id)}). "
                        f"Available IDs (first 10): {available_ids}")

    def get_class_name(self, label: Label) -> str:
        """Returns the original class name for the given standard Label."""
        for name, cls in self.class_mapping.items():
            if cls == label:
                return name
        raise ValueError(f"Unknown label {label}.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def __iter__(self) -> Iterator[dict]:
        return iter(self.data)

    def process_output(self, output):
        """Handles the model's output and evaluates whether it is correct."""
        doc, meta = output
        claim = doc.claim
        try:
            instance = self.get_by_id(claim.id)
            prediction = doc.verdict
            self._save_prediction(doc, meta, claim, prediction, instance.get("label"), instance.get("justification"))
        except ValueError as e:
            # 记录错误但不中断程序
            logger.log(f"Warning: {e}")
            # 使用None作为默认值继续处理
            prediction = doc.verdict
            self._save_prediction(doc, meta, claim, prediction, None, None)

    def _save_prediction(self, doc, meta, claim, prediction, target_label=None, gt_justification=None):
        logger.save_next_prediction(
            sample_index=claim.id,
            claim=str(doc.claim),
            target=target_label,
            justification=doc.justification,
            predicted=prediction,
            gt_justification=gt_justification
        )
        logger.save_next_instance_stats(meta["Statistics"], claim.id)

        if target_label:
            prediction_is_correct = target_label == prediction
            if prediction_is_correct:
                logger.log(bold(green("✅ CORRECT\n")))
            else:
                logger.log(bold(red(f"❌ WRONG - Ground truth: {target_label.value}\n")))
