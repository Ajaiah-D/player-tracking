"""Download a Metrica Sports sample game (tracking + events) into
data/metrica/Sample_Game_<n>/.

    python scripts/download_metrica.py --game 1
"""

import argparse
import urllib.request
from pathlib import Path

BASE = "https://raw.githubusercontent.com/metrica-sports/sample-data/master/data"
FILES = ["RawEventsData.csv", "RawTrackingData_Home_Team.csv", "RawTrackingData_Away_Team.csv"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--game", type=int, choices=(1, 2), default=1,
                        help="games 1 and 2 are CSV; game 3 is EPTS/FIFA format and not supported")
    parser.add_argument("--root", type=Path, default=Path("data/metrica"))
    args = parser.parse_args()

    folder = args.root / f"Sample_Game_{args.game}"
    folder.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        filename = f"Sample_Game_{args.game}_{name}"
        target = folder / filename
        if target.exists():
            print(f"have {target}")
            continue
        print(f"downloading {filename} (tracking files are ~33 MB)")
        urllib.request.urlretrieve(f"{BASE}/Sample_Game_{args.game}/{filename}", target)
    print(f"done -> {folder}")


if __name__ == "__main__":
    main()
