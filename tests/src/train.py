"""Script de experimento para W&B.

Ejemplo:
    wandb login
    python -m src.train --data data/raw --project albaranes-mlops --max-images 10
"""
import argparse
import os
from pathlib import Path

import pandas as pd
import wandb

from src.pipeline import analizar_albaran


def iter_images(folder: Path):
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
        yield from folder.glob(ext)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/raw", help="Carpeta con imágenes de albaranes")
    parser.add_argument("--project", default="albaranes-mlops", help="Nombre del proyecto en W&B")
    parser.add_argument("--max-images", type=int, default=20)
    args = parser.parse_args()

    run = wandb.init(project=args.project, job_type="experiment", config=vars(args))
    rows = []

    for path in list(iter_images(Path(args.data)))[: args.max_images]:
        result = analizar_albaran(path.read_bytes())
        row = {
            "archivo": path.name,
            "fecha": result["fecha"],
            "total": result["total"],
            "num_items": len(result["items"]),
            "num_anomalias": result["numero_anomalias"],
        }
        rows.append(row)
        wandb.log({"num_items": row["num_items"], "num_anomalias": row["num_anomalias"]})

    df = pd.DataFrame(rows)
    os.makedirs("data/processed", exist_ok=True)
    output = "data/processed/resultados_albaranes.csv"
    df.to_csv(output, index=False)
    if len(df):
        wandb.log({
            "dataset_resultados": wandb.Table(dataframe=df),
            "media_items": float(df["num_items"].mean()),
            "media_anomalias": float(df["num_anomalias"].mean()),
        })
    wandb.save(output)
    run.finish()


if __name__ == "__main__":
    main()
