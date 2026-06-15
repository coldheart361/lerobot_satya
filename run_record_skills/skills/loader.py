# skills/loader.py
import json
import os

def load_failure_skills(skills_dir="skills/failures"):
    failures = []
    for fname in sorted(os.listdir(skills_dir)):
        if fname.endswith(".json"):
            with open(os.path.join(skills_dir, fname)) as f:
                failures.append(json.load(f))
    return failures

def format_failures_for_prompt(failures):
    if not failures:
        return ""
    lines = ["## KNOWN FAILURE PATTERNS (avoid these):\n"]
    for f in failures:
        lines.append(f"### {f['id']} ({f['category']})")
        lines.append(f"Problem: {f['description']}")
        if "bad_example" in f:
            bad = f["bad_example"]
            if "what_happened" in bad:
                lines.append(f"What happened: {bad['what_happened']}")
        if "correct_pattern" in f:
            lines.append(f"Rule: {f['correct_pattern']['rule']}")
        lines.append("")
    return "\n".join(lines)