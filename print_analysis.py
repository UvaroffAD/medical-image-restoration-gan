import json
import sys
from pathlib import Path


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

path = Path("analysis_outputs") / "analysis.json"
data = json.loads(path.read_text(encoding="utf-8"))

print("TITLE:", data["core_properties"].get("title"))
print("CREATOR:", data["core_properties"].get("creator"))
print("APP PAGES:", data["app_properties"].get("Pages"))
print("APP WORDS:", data["app_properties"].get("Words"))
print()

print("HEADINGS:")
for heading in data["headings"][:160]:
    indent = "  " * max(0, heading["level"] - 1)
    print(f"{indent}L{heading['level']} {heading['text']}")

print()
print("SECTIONS:")
for section in data["sections"][:40]:
    print(f"--- {section['heading']} | words={section['word_count']}")
    preview = section["preview"].replace("\n", " ")
    print(preview[:900])
    print()

print("TABLES:")
for table in data["tables"][:30]:
    print(f"- Table {table['index']}: {table['rows']} x {table['cols']}")
    if table["preview"]:
        print(f"  {table['preview']}")
