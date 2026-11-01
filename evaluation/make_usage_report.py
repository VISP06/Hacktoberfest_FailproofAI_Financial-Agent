"""Build the markdown model-usage report from local JSONL usage records."""

import json
from collections import defaultdict
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
USAGE_PATH = ROOT_DIR / "logs" / "usage.jsonl"
REPORT_PATH = ROOT_DIR / "evaluation" / "usage_report.md"


def summarize(path):
    summaries = {"purpose": defaultdict(lambda: [0, 0, 0, 0]),
                 "model": defaultdict(lambda: [0, 0, 0, 0])}
    if not path.exists():
        return summaries
    with path.open(encoding="utf-8") as usage_file:
        for line_number, line in enumerate(usage_file, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}: {exc}") from exc
            success = bool(record.get("success", False))
            for group, value in (("purpose", record.get("purpose", "unknown")),
                                 ("model", record.get("model", "unknown"))):
                totals = summaries[group][str(value)]
                totals[0] += 1
                totals[1] += int(record.get("input_tokens") or 0)
                totals[2] += int(record.get("output_tokens") or 0)
                totals[3] += int(not success)
    return summaries


def _markdown_table(title, values):
    lines = [f"## {title}", "", "| Name | Calls | Input tokens | Output tokens | Failures |",
             "| --- | ---: | ---: | ---: | ---: |"]
    if values:
        for name, totals in sorted(values.items()):
            lines.append(f"| {name} | {totals[0]} | {totals[1]} | {totals[2]} | {totals[3]} |")
    else:
        lines.append("| No usage recorded | 0 | 0 | 0 | 0 |")
    return "\n".join(lines)


def main():
    summaries = summarize(USAGE_PATH)
    report = "# Usage report\n\n"
    report += _markdown_table("Per purpose", summaries["purpose"])
    report += "\n\n" + _markdown_table("Per model", summaries["model"]) + "\n"
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"Wrote usage report to {REPORT_PATH}")


if __name__ == "__main__":
    main()
