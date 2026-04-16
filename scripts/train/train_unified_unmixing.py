from __future__ import annotations

import argparse
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Train unified Raman unmixing skeleton model.")
    parser.add_argument("--data-root", type=Path, default=ROOT / "outputs/preprocessing/dataset_preprocessed_v1")
    parser.add_argument("--manifest-csv", type=Path, default=ROOT / "outputs/preprocessing/dataset_preprocessed_v1/_reports/sample_manifest.csv")
    parser.add_argument("--mode", type=str, default="semi", choices=["blind", "fixed", "semi", "weak", "full"])
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "outputs/training/unified_unmixing_baseline")
    parser.add_argument("--min-quality-tier", type=str, default="C", choices=["A", "B", "C"])
    args = parser.parse_args()

    dataset = RamanSpectrumDataset(args.data_root, args.manifest_csv, use_normalized=False, min_quality_tier=args.min_quality_tier)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    anchors = []
    pure_candidates = [
        args.data_root / "pp纯谱/采集图谱.csv",
        args.data_root / "pe纯谱/采集图谱.csv",
        args.data_root / "淀粉纯谱/玉米淀粉/DATA-105635-X0-Y30-8884.csv",
    ]
    for path in pure_candidates:
        if path.exists():
            rows = []
            import csv

            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            anchors.append([float(row["Intensity_corrected"]) for row in rows])
    anchor_tensor = torch.tensor(anchors, dtype=torch.float32) if anchors else None

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
    torch.save(model.state_dict(), args.run_dir / "model.pt")
    print({"run_dir": str(args.run_dir), "epochs": args.epochs, "history": history})


if __name__ == "__main__":
    main()
