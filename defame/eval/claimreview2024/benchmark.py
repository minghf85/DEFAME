import json
import os
from datetime import datetime
from pathlib import Path

from ezmm import Image
from huggingface_hub import snapshot_download

from config.globals import data_root_dir
from defame.common import Label, Claim
from defame.eval.benchmark import Benchmark
from defame.evidence_retrieval.tools import Geolocate, Search
from defame.extension.classify import Classifier


class ClaimReview2024(Benchmark):
    name = "ClaimReview2024+"
    shorthand = "claimreview2024"

    is_multimodal = True

    class_mapping = {
        "refuted": Label.REFUTED,
        "supported": Label.SUPPORTED,
        "not enough information": Label.NEI,
        "misleading": Label.CHERRY_PICKING,
    }

    class_definitions = {
        Label.SUPPORTED: "The claim is accurate based on evidence.",
        Label.REFUTED: "A claim is considered refuted when the evidence contradicts the claim.",
        Label.CHERRY_PICKING: "The claim is misleading or requires additional context.",
        Label.NEI: "The claim does not have enough information to be verified.",
    }

    extra_plan_rules = """Always suggest the use of geolocation!"""

    available_actions = [Search, Geolocate]

    def __init__(self, variant="test"):
        super().__init__(variant, "ClaimReview2024plus/test.json")

    def _load_data(self) -> list[dict]:
        if not self.file_path.exists():
            # Download the dataset from Hugging Face:
            # Ensure you are logged in via `huggingface-cli login` and have
            # got access to the dataset
            snapshot_download(repo_id="MAI-Lab/ClaimReview2024plus",
                              repo_type="dataset",
                              local_dir=self.file_path.parent)

        with open(self.file_path, "r") as f:
            raw_data = json.load(f)

        data = []
        for i, entry in enumerate(raw_data):
            image_path = Path(data_root_dir / "ClaimReview2024plus" / entry["image"]) if entry["image"] else None
            # print("image_path:", image_path)
            image = Image(image_path) if (image_path and os.path.exists(image_path)) else None
            claim_text = f"{image.reference} {entry['text']}" if image else f"{entry['text']}"
            print(claim_text)
            label_text = entry.get("label")
            date_str = entry.get("date")
            date = datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%SZ") if date_str else None
            claim_entry = {
                "id": i,
                "input": Claim(claim_text,
                               id=i,
                               author=entry.get("author"),
                               date=date),
                "label": self.class_mapping.get(label_text),
                "justification": "",
            }
            data.append(claim_entry)

        return data


class ClaimReview2024_with_difficulty(Benchmark):
    name = "ClaimReview2024+"
    shorthand = "claimreview2024"

    is_multimodal = True

    class_mapping = {
        "refuted": Label.REFUTED,
        "supported": Label.SUPPORTED,
        "not enough information": Label.NEI,
        "misleading": Label.CHERRY_PICKING,
    }

    class_definitions = {
        Label.SUPPORTED: "The claim is accurate based on evidence.",
        Label.REFUTED: "A claim is considered refuted when the evidence contradicts the claim.",
        Label.CHERRY_PICKING: "The claim is misleading or requires additional context.",
        Label.NEI: "The claim does not have enough information to be verified.",
    }

    extra_plan_rules = """Always suggest the use of geolocation!"""

    available_actions = [Search, Geolocate]

    def __init__(self, variant="test"):
        super().__init__(variant, "ClaimReview2024plus/test.json")

    def _load_data(self) -> list[dict]:
        if not self.file_path.exists():
            # Download the dataset from Hugging Face:
            # Ensure you are logged in via `huggingface-cli login` and have
            # got access to the dataset
            snapshot_download(repo_id="MAI-Lab/ClaimReview2024plus",
                              repo_type="dataset",
                              local_dir=self.file_path.parent)

        with open(self.file_path, "r") as f:
            raw_data = json.load(f)

        data = []
        for i, entry in enumerate(raw_data):
            image_path = Path(data_root_dir / "ClaimReview2024plus" / entry["image"]) if entry["image"] else None
            # print("image_path:", image_path)
            image = Image(image_path) if (image_path and os.path.exists(image_path)) else None
            claim_text = f"{image.reference} {entry['text']}" if image else f"{entry['text']}"
            print(claim_text)
            label_text = entry.get("label")
            date_str = entry.get("date")
            date = datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%SZ") if date_str else None
            claim = Claim(claim_text,
                               id=i,
                               author=entry.get("author"),
                               date=date)
            
            # 处理难度分类
            difficulty = "unknown"  # 默认值
            if "difficulty" in entry:
                difficulty = entry["difficulty"]
                if difficulty not in ["easy", "medium", "hard"]:
                    print(f"Warning: Unknown difficulty level '{difficulty}' in entry {i}. Skipping difficulty assignment.")
                    difficulty = "unknown"
            else:
                # 生成分类结果并写回原json
                print(f"Classifying difficulty for claim {i}: {claim_text[:100]}...")
                
                # 延迟初始化分类器，避免重复创建
                if not hasattr(self, 'classifier') or self.classifier is None:
                    self.classifier = Classifier()
                
                try:
                    difficulty_label = self.classifier.classify_difficulty(claim)
                    
                    if difficulty_label is not None:
                        difficulty = difficulty_label.value
                        entry["difficulty"] = difficulty
                        print(f"Classified as: {difficulty}")
                    else:
                        difficulty = "unknown"
                        print(f"Classification failed for claim {i}")
                    
                    # 立即写回JSON文件以保存进度
                    with open(self.file_path, "w", encoding="utf-8") as f:
                        json.dump(raw_data, f, ensure_ascii=False, indent=2)
                    print(f"Updated JSON file with difficulty classification for claim {i}")
                    
                except Exception as e:
                    difficulty = "unknown"
                    print(f"Error classifying claim {i}: {e}")
                    # 即使出错也要保存当前状态
                    entry["difficulty"] = difficulty
                    with open(self.file_path, "w", encoding="utf-8") as f:
                        json.dump(raw_data, f, ensure_ascii=False, indent=2)

            claim_entry = {
                "id": i,
                "input": claim,
                "label": self.class_mapping.get(label_text),
                "difficulty": difficulty,
                "justification": ""
            }
            data.append(claim_entry)

        return data

    def classify_missing_difficulties(self):
        """
        批量分类所有缺失难度标签的条目
        """
        if not self.file_path.exists():
            print("JSON文件不存在，无法进行分类")
            return
        
        with open(self.file_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        
        missing_indices = []
        for i, entry in enumerate(raw_data):
            if "difficulty" not in entry or entry["difficulty"] is None:
                missing_indices.append(i)
        
        if not missing_indices:
            print("所有条目都已有难度分类")
            return
        
        print(f"发现 {len(missing_indices)} 个条目需要难度分类")
        
        # 初始化分类器
        if not hasattr(self, 'classifier') or self.classifier is None:
            self.classifier = Classifier()
        
        # 批量处理
        for count, i in enumerate(missing_indices, 1):
            entry = raw_data[i]
            
            # 构建claim文本
            image_path = Path(data_root_dir / "ClaimReview2024plus" / entry["image"]) if entry["image"] else None
            image = Image(image_path) if (image_path and os.path.exists(image_path)) else None
            claim_text = f"{image.reference} {entry['text']}" if image else f"{entry['text']}"
            
            # 创建Claim对象
            date_str = entry.get("date")
            date = datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%SZ") if date_str else None
            claim = Claim(claim_text,
                         id=i,
                         author=entry.get("author"),
                         date=date)
            
            print(f"[{count}/{len(missing_indices)}] 分类条目 {i}: {claim_text[:100]}...")
            
            try:
                difficulty_label = self.classifier.classify_difficulty(claim)
                
                if difficulty_label is not None:
                    difficulty = difficulty_label.value
                    entry["difficulty"] = difficulty
                    print(f"分类结果: {difficulty}")
                else:
                    entry["difficulty"] = "unknown"
                    print(f"分类失败")
                
                # 每处理10个条目保存一次，避免丢失进度
                if count % 10 == 0:
                    with open(self.file_path, "w", encoding="utf-8") as f:
                        json.dump(raw_data, f, ensure_ascii=False, indent=2)
                    print(f"已保存进度 ({count}/{len(missing_indices)})")
                    
            except Exception as e:
                entry["difficulty"] = "unknown"
                print(f"处理条目 {i} 时出错: {e}")
        
        # 最终保存
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(raw_data, f, ensure_ascii=False, indent=2)
        print(f"完成所有分类，已保存到 {self.file_path}")

if __name__ == "__main__":
    benchmark = ClaimReview2024_with_difficulty()
    # 批量分类缺失的难度标签
    benchmark.classify_missing_difficulties()
    # for claim in benchmark:
    #     print(claim)
