"""Minimal running example for a multimodal fact-check."""

from typing import Text
from ezmm import Image

from defame.fact_checker import FactChecker

# Configure tools: only use Searcher and TextAnalyzer
tools_config = {
    "searcher": {},  # Use default searcher configuration
    "text_analyzer": {}  # Use default text analyzer configuration
}

fact_checker = FactChecker(
    llm="gpt_4o",
    tools_config=tools_config
)

claim = [Image("in/example/us_prison.webp"),
           "look at this! 😱 all because of the democrats!!! 😡 if they continue, half of the US will be ",
           "woke and the other half in jail! Stop the mass incarceration!!!"]

report, _ = fact_checker.verify_claim(claim)
report.save_to("out/fact-check")
