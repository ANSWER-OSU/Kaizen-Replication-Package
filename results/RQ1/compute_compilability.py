import sys
import csv

csv_file = sys.argv[1]

data = {}

with open(csv_file, "r") as f:
    reader = csv.DictReader(f)

    for row in reader:
        app = row["dataset"].split("-seed")[0]

        total_variants = int(row["total_variants"])
        passed = int(row["pass"])

        if app not in data:
            data[app] = {
                "total_variants_sum": 0,
                "pass_sum": 0,
                "count": 0
            }

        data[app]["total_variants_sum"] += total_variants
        data[app]["pass_sum"] += passed
        data[app]["count"] += 1

for app in sorted(data):
    avg_total_variants = round(
        data[app]["total_variants_sum"] / data[app]["count"]
    )
    avg_pass = round(
        data[app]["pass_sum"] / data[app]["count"]
    )

    pass_fraction = avg_pass / avg_total_variants if avg_total_variants > 0 else 0

    print(
        f"{app}: "
        f"avg_total_variants={avg_total_variants}, "
        f"avg_pass={avg_pass}, "
        f"pass_fraction={pass_fraction:.4f}"
    )
