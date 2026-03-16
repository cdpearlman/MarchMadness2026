"""Parse Yahoo ownership MD files and generate ownership_2026.json."""
import json
import re
from pathlib import Path

# Yahoo name -> bracket_2026.json name mapping
YAHOO_TO_BRACKET = {
    "N. Carolina": "North Carolina",
    "Miami (FL)": "Miami FL",
    "California Baptist": "Cal Baptist",
    "N. Dak. St.": "North Dakota St.",
    "Pennsylvania": "Penn",
    "Queens University": "Queens",
    "LIU Brooklyn": "LIU",
    "McNeese": "McNeese St.",
    "Michigan St.": "Michigan St.",
    "St. Mary's": "Saint Mary's",
}

# Play-in combined entries -> both individual team names
# Assign combined ownership to both teams (whoever wins inherits the public's pick)
PLAYIN_MAP = {
    "MOH/SMU": ["Miami OH", "SMU"],
    "TX/NCST": ["Texas", "N.C. State"],
    "PV/LEH": ["Prairie View A&M", "Lehigh"],
    "UMBC/HOW": ["UMBC", "Howard"],
}

def parse_md(path: Path) -> dict[str, float]:
    """Parse an ownership MD file -> {yahoo_team_name: pct}."""
    text = path.read_text(encoding="utf-8")
    teams = {}
    lines = text.strip().split("\n")
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        # Look for percentage pattern
        pct_match = re.search(r"(\d+\.?\d*)%", line)
        if pct_match:
            pct = float(pct_match.group(1)) / 100.0
            # Team name is 2 lines before the percentage line (name, then seed, then pct)
            # But format is: rank line, blank line, team name, (seed), pct%
            # Let's look backwards for the team name
            for j in range(i - 1, max(i - 4, -1), -1):
                candidate = lines[j].strip()
                if candidate and not candidate.startswith("(") and not re.match(r"^\d+(st|nd|rd|th)$", candidate) and "Rank" not in candidate and "Round" not in candidate and "National" not in candidate:
                    teams[candidate] = pct
                    break
        i += 1
    return teams


def map_name(yahoo_name: str) -> list[str]:
    """Map a Yahoo team name to one or more bracket names."""
    if yahoo_name in PLAYIN_MAP:
        return PLAYIN_MAP[yahoo_name]
    mapped = YAHOO_TO_BRACKET.get(yahoo_name, yahoo_name)
    return [mapped]


def main():
    base = Path(".")

    # Map file -> JSON key
    # Keys must match internal round names used in bracket_gen.py
    files = {
        "ownership_64.md": "r64",
        "ownership_32.md": "r32",
        "ownership_16.md": "s16",
        "ownership_8.md": "e8",
        "ownership_4.md": "f4",
        "ownership_final.md": "champ",
    }

    result = {}

    for fname, key in files.items():
        path = base / fname
        if not path.exists():
            print(f"[!] Missing {fname}")
            continue

        raw = parse_md(path)
        mapped = {}
        for yahoo_name, pct in raw.items():
            bracket_names = map_name(yahoo_name)
            for bn in bracket_names:
                mapped[bn] = round(pct, 6)

        # Sort by ownership descending
        mapped = dict(sorted(mapped.items(), key=lambda x: x[1], reverse=True))
        result[key] = mapped
        print(f"[OK] {key}: {len(mapped)} teams parsed from {fname}")

    # Write output
    out_path = base / "data" / "ownership_2026.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[OK] Written to {out_path}")

    # Quick sanity check: print top 5 champion picks
    print("\nTop 5 champion ownership:")
    for team, pct in list(result.get("champion", {}).items())[:5]:
        print(f"  {team}: {pct:.2%}")


if __name__ == "__main__":
    main()
