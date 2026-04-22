#!/usr/bin/env python3
"""
End-to-end Re-entry Curriculum Demo

1. Generate synthetic storm shards (easy → storm)
2. Curriculum training: train on each stage, validate on next-stage held-out
3. Export ONNX
4. Validation on synthetic storms (storm-stage held-out)
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.gen_reentry_storms import generate_storm_decay, generate_shard
from pkg.reentry.curriculum_trainer import ReentryCurriculumTrainer


def main():
    print("=" * 70)
    print(" Re-entry Forecaster — Curriculum Training + Storm Validation")
    print("=" * 70)

    out_dir = Path("data/synthetic/reentry")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Generate curriculum shards
    # ------------------------------------------------------------------
    print("\n[1] Generating curriculum shards...")
    shard_paths = {}
    for stage in range(4):
        path = generate_shard(stage=stage, n=50, output_dir=str(out_dir))
        shard_paths[stage] = path

    # ------------------------------------------------------------------
    # 2. Curriculum training
    # ------------------------------------------------------------------
    print("\n[2] Curriculum training (easy → medium → hard → storm)...")
    trainer = ReentryCurriculumTrainer(device="cpu")

    for stage in range(4):
        cfg = trainer.STAGE_CONFIG[stage]
        print(f"\n--- Stage {stage}: {cfg['name'].upper()} ---")

        # Train on current stage
        train_result = trainer.train_stage(shard_paths[stage], stage=stage, batch_size=8)

        # Validate on next-stage held-out (or same stage if last)
        if stage < 3:
            val_stage = stage + 1
        else:
            val_stage = stage

        # Generate a small held-out validation shard
        print(f"[VALID] Generating held-out {trainer.STAGE_CONFIG[val_stage]['name']} shard...")
        val_path = generate_shard(stage=val_stage, n=20, output_dir=str(out_dir / "val"))
        val_metrics = trainer.validate_on_shard(val_path, batch_size=8)
        print(f"[VALID] MSE={val_metrics['mse']:.4f}  "
              f"TTD_MSE={val_metrics['ttd_mse']:.4f}  "
              f"LAT_MSE={val_metrics['lat_mse']:.4f}  "
              f"LON_MSE={val_metrics['lon_mse']:.4f}")

        # Advance decision
        if stage < 3:
            threshold = cfg["threshold_mse"]
            if trainer.should_advance(stage, val_metrics):
                print(f"[PASS] Validation MSE {val_metrics['mse']:.2f} < threshold {threshold:.2f}. Advancing.")
                trainer.current_stage = stage + 1
            else:
                print(f"[WARN] Validation MSE {val_metrics['mse']:.2f} >= threshold {threshold:.2f}. Staying at stage {stage}.")
                trainer.current_stage = stage
                break
        else:
            print(f"[FINAL] Storm stage complete.")

    # ------------------------------------------------------------------
    # 3. Final validation on storm held-out
    # ------------------------------------------------------------------
    print("\n[3] Final validation on storm-stage held-out (30 samples)...")
    final_val_path = generate_shard(stage=3, n=30, output_dir=str(out_dir / "final_val"))
    final_metrics = trainer.validate_on_shard(final_val_path, batch_size=8)
    print(f"   Final Storm MSE: {final_metrics['mse']:.4f}")
    print(f"   TTD_MSE:         {final_metrics['ttd_mse']:.4f}  hr²")
    print(f"   LAT_MSE:         {final_metrics['lat_mse']:.4f}  deg²")
    print(f"   LON_MSE:         {final_metrics['lon_mse']:.4f}  deg²")

    # Summary statistics
    preds = final_metrics["preds"]
    targs = final_metrics["targets"]
    ttd_err = preds[:, 0] - targs[:, 0]
    lat_err = preds[:, 1] - targs[:, 1]
    lon_err = preds[:, 2] - targs[:, 2]
    print(f"\n   TTD bias:  {ttd_err.mean():+.3f} hr,  MAE: {np.abs(ttd_err).mean():.3f} hr")
    print(f"   LAT bias:  {lat_err.mean():+.3f} deg,  MAE: {np.abs(lat_err).mean():.3f} deg")
    print(f"   LON bias:  {lon_err.mean():+.3f} deg,  MAE: {np.abs(lon_err).mean():.3f} deg")

    # Skip-out detection analysis
    data = np.load(final_val_path)
    skip_flags = data["skip_out"]
    if skip_flags.any():
        skip_preds = preds[skip_flags]
        skip_targs = targs[skip_flags]
        print(f"\n   Skip-out samples: {skip_flags.sum()}/{len(skip_flags)}")
        print(f"   Skip-out TTD MAE: {np.abs(skip_preds[:, 0] - skip_targs[:, 0]).mean():.3f} hr")

    # ------------------------------------------------------------------
    # 4. Export ONNX
    # ------------------------------------------------------------------
    print("\n[4] Exporting ONNX...")
    onnx_path = trainer.export_onnx("ai/training/models/reentry_forecaster_v1.onnx")
    size_kb = Path(onnx_path).stat().st_size / 1024
    print(f"   ONNX: {onnx_path} ({size_kb:.1f} KB)")

    # ------------------------------------------------------------------
    # 5. Save curriculum checkpoint
    # ------------------------------------------------------------------
    ckpt_path = "ai/training/models/reentry_curriculum.ckpt"
    trainer.save_checkpoint(ckpt_path)
    print(f"   Checkpoint: {ckpt_path}")

    print("\n" + "=" * 70)
    print(" Curriculum Training + Storm Validation Complete")
    print("=" * 70)

    # Print curriculum history
    print("\nCurriculum progression:")
    for h in trainer.history:
        print(f"  Stage {h['stage']} ({h['name']}): final_loss={h['final_loss']:.4f}")


if __name__ == "__main__":
    main()
