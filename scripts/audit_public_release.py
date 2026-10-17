# 檢查 Git 待公開檔案中的敏感資訊。

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {".py", ".md", ".yaml", ".yml", ".json", ".txt", ".js", ".html"}
CONTENT_RULES = {
    "OpenAI-style key": re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
    "Google-style key": re.compile(r"AIza[0-9A-Za-z_-]{20,}"),
    "GitHub token": re.compile(r"gh[pousr]_[A-Za-z0-9_]{20,}"),
    "private key": re.compile(r"BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY"),
    "local absolute path": re.compile(r"/Users/"),
    "questionnaire identifier": re.compile(r"(?:班級|座號|answer_key|rater_assignment)", re.I),
}
PATH_RULES = (
    re.compile(r"(?:^|/)(?:models|checkpoints|logs|results|training_runs|questionnaire_app)(?:/|$)", re.I),
    re.compile(r"(?:^|/)(?:data/(?:raw|processed|private)|generated_questionnaire|human_evaluation_results)(?:/|$)", re.I),
    re.compile(r"(?:\.pt|\.pth|\.bin|\.safetensors|\.jsonl|\.xlsx|\.pdf|\.pptx|\.docx|\.mp4)$", re.I),
)
CONTENT_EXEMPTIONS = {
    "README.md",
    "scripts/audit_public_release.py",
    "tests/test_public_config.py",
}


def tracked_files() -> list[Path]:
    # 取得已加入 Git 及尚未追蹤、但未被忽略的候選檔案。
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return [ROOT / line for line in result.stdout.splitlines() if line]


def audit() -> list[str]:
    # 先查禁用路徑，再依副檔名與例外名單檢查文字內容。
    findings: list[str] = []
    for path in tracked_files():
        relative = path.relative_to(ROOT).as_posix()
        if any(rule.search(relative) for rule in PATH_RULES):
            findings.append(f"forbidden path: {relative}")
            continue
        if relative in CONTENT_EXEMPTIONS or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for label, rule in CONTENT_RULES.items():
            if rule.search(text):
                # 只回報檔名與問題類型，不印出疑似敏感文字。
                findings.append(f"{label}: {relative}")
    return findings


if __name__ == "__main__":
    problems = audit()
    if problems:
        print("Public-release audit failed:", file=sys.stderr)
        print("\n".join(f"- {problem}" for problem in problems), file=sys.stderr)
        raise SystemExit(1)
    print("Public-release audit passed.")
