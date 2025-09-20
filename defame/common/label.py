from enum import Enum


class Label(Enum):
    SUPPORTED = "supported"
    NEI = "not enough information"
    REFUTED = "refuted"
    CONFLICTING = "conflicting evidence"
    CHERRY_PICKING = "cherry-picking"
    REFUSED_TO_ANSWER = "error: refused to answer"
    OUT_OF_CONTEXT = "out of context"
    MISCAPTIONED = "miscaptioned"


DEFAULT_LABEL_DEFINITIONS = {
    Label.SUPPORTED: "The knowledge from the fact-check supports or at least strongly implies the Claim. "
                     "Mere plausibility is not enough for this decision.",
    Label.NEI: "The fact-check does not contain sufficient information to come to a conclusion. For example, "
               "there is substantial lack of evidence. In this case, state which information exactly "
               "is missing. In particular, if no RESULTS or sources are available, pick this decision.",
    Label.REFUTED: "The knowledge from the fact-check clearly refutes the Claim. The mere absence or lack of "
                   "supporting evidence is not enough reason for being refuted (argument from ignorance).",
    Label.CONFLICTING: "The knowledge from the fact-check contains conflicting evidence from multiple "
                       "RELIABLE sources. Even trying to resolve the conflicting sources through additional "
                       "investigation was not successful.",
    Label.OUT_OF_CONTEXT: "The image is used out of context. This means that while the caption may be factually"
                          "correct, the image does not relate to the caption or is used in a misleading way to "
                          "convey a false narrative.",
    Label.MISCAPTIONED: "The claim has a true image, but the caption does not accurately describe the image, "
                        "providing incorrect information.",
}

class DifficultyLabel(Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"

DEFAULT_DIFFICULTY_DEFINITIONS_V1 = {
    DifficultyLabel.EASY: "Verifiable through single reliable source or simple method. Content (text/image/video) is clear and straightforward, requiring no specialized knowledge.",
    DifficultyLabel.MEDIUM: "Requires cross-referencing multiple sources. Content may need detailed analysis, temporal/spatial context verification, or multimodal content correlation.",
    DifficultyLabel.HARD: "Lacks reliable sources, has conflicting information, or requires specialized knowledge to analyze complex content relationships across text, visual, and contextual elements.",
}

DEFAULT_DIFFICULTY_DEFINITIONS = {
    DifficultyLabel.EASY: """Claims with objective, factual content that can be verified through direct observation or single authoritative source. Requires basic reading comprehension and simple fact-checking skills.
**Characteristics:**  
- Directly observable facts (time, location, numbers, names)  
- Single-modality verification (text-only or image-only)  
- No background knowledge or specialized skills required  
- Clear and unambiguous answers, black-and-white distinctions
""",
    DifficultyLabel.MEDIUM: """Claims requiring basic cross-referencing between multiple sources or simple multimodal content comparison. May need elementary reasoning or common knowledge context.
**Characteristics:**  
- Requires cross-verification from 2-3 information sources  
- Simple text-image consistency checks  
- Basic chronological or logical reasoning  
- Requires common sense knowledge but no specialized skills  
- May involve minor information incompleteness
""",
    DifficultyLabel.HARD: """Claims involving complex reasoning, conflicting evidence, deep domain expertise, or sophisticated manipulation detection. Requires advanced analytical skills and comprehensive evaluation frameworks.
**Characteristics:**  
- Requires in-depth knowledge of specialized fields  
- Involves contradictory or severely incomplete evidence  
- Complex multimodal content tampering detection  
- Requires integration of multiple analytical dimensions and evaluation frameworks  
- Involves complex causal chains or reasoning networks
"""
}