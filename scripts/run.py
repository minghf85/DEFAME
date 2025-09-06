"""Minimal running example for a multimodal fact-check."""

from typing import Text
from ezmm import Image

from defame.fact_checker import FactChecker

# Configure tools: only use Searcher and TextAnalyzer
tools_config = {
    "searcher": {},  # Use default searcher configuration
    "text_extractor": {},
    "geolocator": {}  # Use default text analyzer configuration
}

fact_checker = FactChecker(
    llm="gpt_4o"
)

claim = [Image("in/example/2025-01-18_298.png"), "Image shows a 'Die in' climate protest in Austria, where protesters got inside of body bags to signify the catastrophic impact that current climate policy could have on the world."]

report, _ = fact_checker.verify_claim(claim)
report.save_to("out/fact-check")
