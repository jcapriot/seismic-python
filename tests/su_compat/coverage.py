"""Which of the SU programs that README.md says are ported have a comparison against SU.

    python tests/su_compat/coverage.py        # prints the programs without a case
"""
import pathlib
import re
import sys

README = pathlib.Path(__file__).resolve().parents[2] / "README.md"


def readme_programs():
    """The SU programs of the `Available programs` table, in README order"""
    text = README.read_text()
    section = text.split("## Available programs", 1)[1].split("```python", 1)[0]
    programs = []
    for line in section.splitlines():
        if not line.startswith("|") or line.startswith("| seispy") or line.startswith("|---"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) >= 2:
            programs += re.findall(r"`([a-z0-9_]+)`", cells[1])
    return list(dict.fromkeys(programs))


def main():
    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    from cases import CASES

    covered = {c.program for c in CASES}
    programs = readme_programs()
    missing = [p for p in programs if p not in covered]
    print(f"{len(covered & set(programs))}/{len(programs)} ported programs are compared against SU")
    if missing:
        print("not yet compared:", " ".join(missing))


if __name__ == "__main__":
    main()
