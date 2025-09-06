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

DEFAULT_DIFFICULTY_DEFINITIONS = {
    DifficultyLabel.EASY: "The claim can be verified with high confidence using a single reliable source or "
                          "a straightforward fact-checking method.",
    DifficultyLabel.MEDIUM: "The claim requires cross-referencing multiple reliable sources or involves "
                            "some complexity in verification, but can still be resolved with moderate effort.",
    DifficultyLabel.HARD: "The claim is difficult to verify due to lack of reliable sources, conflicting information, "
                         "or requires specialized knowledge or expertise to assess its validity.",
}