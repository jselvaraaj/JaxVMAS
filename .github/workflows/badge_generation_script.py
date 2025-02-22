import json
import os

import pybadges

os.makedirs("badges", exist_ok=True)
with open("coverage.json") as f:
    data = json.load(f)
coverage = data.get("totals", {}).get("percent_covered", 0)
badge_svg = pybadges.badge(
    left_text="coverage", right_text=f"{round(coverage)}%", right_color="green"
)
with open("badges/coverage.svg", "w") as f:
    f.write(badge_svg)
