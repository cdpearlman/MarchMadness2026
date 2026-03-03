"""
Download Barttorvik pre-tournament team stats CSVs for all seasons.

Opens each season's CSV URL in the default browser. The browser handles
Cloudflare JS verification and shows a save dialog. Save each file to
data/barttorvik/YYYY.csv when prompted.

Usage:
    python download_barttorvik.py          # all seasons 2008-2026
    python download_barttorvik.py 2024     # single season
    python download_barttorvik.py --test   # open 2024 only as a test
"""

import sys
import webbrowser
import time

SELECTION_SUNDAYS = {
    2008: "20080316",
    2009: "20090315",
    2010: "20100314",
    2011: "20110313",
    2012: "20120311",
    2013: "20130317",
    2014: "20140316",
    2015: "20150315",
    2016: "20160313",
    2017: "20170312",
    2018: "20180311",
    2019: "20190317",
    2020: "20200315",
    2021: "20210314",
    2022: "20220313",
    2023: "20230312",
    2024: "20240317",
    2025: "20250316",
    2026: "20260315",
}

URL_TEMPLATE = (
    "https://barttorvik.com/team-tables_each.php"
    "?year={year}"
    "&begin={begin}"
    "&end={end}"
    "&conlimit=All&state=All&top=0&quad=5"
    "&venue=All&type=All&mingames=0&csv=1"
)


def build_url(year: int) -> str:
    begin = f"{year - 1}1101"
    end = SELECTION_SUNDAYS[year]
    return URL_TEMPLATE.format(year=year, begin=begin, end=end)


def download_seasons(seasons: list[int]):
    total = len(seasons)
    for i, year in enumerate(seasons, 1):
        url = build_url(year)
        print(f"\n[{i}/{total}] Season {year}")
        print(f"  URL: {url}")
        print(f"  Save as: data/barttorvik/{year}.csv")
        webbrowser.open(url)
        if i < total:
            input("  Press Enter when saved to continue to next season...")
        else:
            input("  Press Enter when saved to finish.")
    print(f"\nDone. {total} season(s) downloaded.")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg == "--test":
            download_seasons([2024])
        else:
            year = int(arg)
            if year not in SELECTION_SUNDAYS:
                print(f"Unknown season {year}. Valid: {min(SELECTION_SUNDAYS)}-{max(SELECTION_SUNDAYS)}")
                sys.exit(1)
            download_seasons([year])
    else:
        all_seasons = sorted(SELECTION_SUNDAYS.keys())
        print(f"Downloading {len(all_seasons)} seasons: {all_seasons[0]}-{all_seasons[-1]}")
        print("Save each file to data/barttorvik/YYYY.csv when the save dialog appears.\n")
        download_seasons(all_seasons)
