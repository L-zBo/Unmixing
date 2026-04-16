from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from demixing.data.dataset import RamanSpectrumDataset
from demixing.models.unified_unmixing import UnifiedRamanUnmixingNet, UnifiedUnmixingConfig
from demixing.training.trainer import TrainConfig, train_model


def load_anchor_tensor(data_root: Path, manifest_csv: Path) -> torch.Tensor | None:
    with manifest_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        manifest_rows = list(csv.DictReader(handle))
    family_order = {"pure_pp": 0, "pure_pe": 1, "pure_starch": 2}
    pure_rows = [row for row in manifest_rows if row.get("family") in family_order]
    pure_rows = sorted(pure_rows, key=lambda row: family_order[row["family"]])

    anchors = []
    for row in pure_rows:
        path = data_root / row["relative_path"]
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            spectrum_rows = list(csv.DictReader(handle))
        anchors.append([float(item["Intensity_corrected"]) for item in spectrum_rows])
    return torch.tensor(anchors, dtype=torch.float32) if anchors else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Train unified Raman unmixing model.")
    parser.add_argument("--data-root", type=Path, default=ROOT / "outputs/preprocessing/dataset_preprocessed_v1")
    parser.add_argument("--manifest-csv", type=Path, default=ROOT / "outputs/preprocessing/dataset_preprocessed_v1/_reports/sample_manifest.csv")
    parser.add_argument("--mode", type=str, default="semi", choices=["blind", "fixed", "semi", "weak", "full"])
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "outputs/training/unified_unmixing_baseline")
    parser.add_argument("--min-quality-tier", type=str, default="C", choices=["A", "B", "C"])
    parser.add_argument("--only-source-kind", action="append", choices=["raw", "average", "pure"])
    parser.add_argument("--require-weak-label", action="store_true")
    args = parser.parse_args()

    allowed_source_kinds = set(args.only_source_kind) if args.only_source_kind else None
    dataset = RamanSpectrumDataset(
        args.data_root,
        args.manifest_csv,
        use_normalized=False,
        min_quality_tier=args.min_quality_tier,
        allowed_source_kinds=allowed_source_kinds,
        require_weak_label=args.require_weak_label,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    anchor_tensor = load_anchor_tensor(args.data_root, args.manifest_csv)
    model = UnifiedRamanUnmixingNet(UnifiedUnmixingConfig(mode=args.mode), endmember_anchors=anchor_tensor)

    label_weight = 0.0 if args.mode in {"blind", "fixed", "semi"} else 0.2
    anchor_weight = 0.0 if args.mode == "blind" else 0.1
    history = train_model(
        model,
        dataloader,
        TrainConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            label_weight=label_weight,
            anchor_weight=anchor_weight,
        ),
        args.run_dir,
    )

    args.run_dir.mkdir(parents=True, exist_ok=True)
    (args.run_dir / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.run_dir / "config.json").write_text(
        json.dumps(
            {
                "mode": args.mode,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "min_quality_tier": args.min_quality_tier,
                "only_source_kind": sorted(allowed_source_kinds) if allowed_source_kinds else None,
                "require_weak_label": args.require_weak_label,
                "num_samples": len(dataset),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    torch.save(model.state_dict(), args.run_dir / "model.pt")
    print({"run_dir": str(args.run_dir), "epochs": args.epochs, "num_samples": len(dataset), "history": history})


if __name__ == "__main__":
    main()
