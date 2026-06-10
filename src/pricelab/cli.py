from __future__ import annotations

import argparse
from pathlib import Path

from pricelab.data.demo_generator import save_demo_dataset
from pricelab.data.importers import load_and_standardize
from pricelab.data.quality import quality_score, scan_quality


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pricelab", description="PriceLab local pricing intelligence utilities.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser("generate-demo", help="Generate the synthetic demo dataset.")
    demo.add_argument("--out", default="data/demo/pricelab_demo.csv", help="Output CSV path.")
    demo.add_argument("--seed", type=int, default=42, help="Random seed.")

    validate = subparsers.add_parser("validate", help="Validate and scan a pricing CSV.")
    validate.add_argument("csv_path", help="CSV file to validate.")

    args = parser.parse_args(argv)
    if args.command == "generate-demo":
        out_path = save_demo_dataset(Path(args.out), seed=args.seed)
        print(f"Demo dataset written to {out_path}")
        return 0
    if args.command == "validate":
        df = load_and_standardize(args.csv_path)
        report = scan_quality(df)
        print(f"Rows: {report.row_count}")
        print(f"Products: {report.product_count}")
        print(f"Quality score: {quality_score(report):.0f}/100")
        for issue in report.issues:
            product = f" [{issue.product_id}]" if issue.product_id else ""
            print(f"{issue.severity.value.upper()} {issue.code}{product}: {issue.message} ({issue.metric})")
        return 1 if report.error_count else 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

