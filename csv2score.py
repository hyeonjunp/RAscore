import argparse
import os
import sys
from pathlib import Path

# Disable TensorFlow GPU detection before importing TensorFlow-dependent modules.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Suppress TensorFlow INFO and WARNING messages.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import pandas as pd
from rdkit import Chem, RDConfig
from RAscore import RAscore_NN
from RAscore import RAscore_XGB


def import_sa_scorer():
    """Import the RDKit synthetic accessibility scorer.

    Args:
        None.

    Returns:
        module: RDKit SA Score module.

    Raises:
        ImportError: If the RDKit SA Score module cannot be found.
    """
    try:
        from rdkit.Contrib.SA_Score import sascorer

        return sascorer
    except ImportError:
        sa_score_path = Path(RDConfig.RDContribDir) / "SA_Score"

        if not sa_score_path.exists():
            raise ImportError(
                f"RDKit SA_Score directory was not found: {sa_score_path}"
            )

        sys.path.append(str(sa_score_path))
        import sascorer

        return sascorer


def get_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        None.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Calculate RDKit SA Score, RAscore(NN) and RAscore(XGB) "
        "for every molecule in an input CSV file."
    )

    parser.add_argument(
        "--csv_path",
        type=str,
        required=True,
        help="Path to the input CSV file, containing ID and SMILES columns.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save the output CSV file to.",
    )

    return parser.parse_args()


def score_csv(
    csv_path: str,
    output: str,
    sa_scorer,
    nn_scorer: RAscore_NN.RAScorerNN,
    xgb_scorer: RAscore_XGB.RAScorerXGB,
) -> Path:
    """Score every molecule in a CSV file and save the results to a new CSV.

    Args:
        csv_path: Path to the input CSV file. Must contain ID and SMILES columns.
        output: Path to save the output CSV file to.
        sa_scorer: RDKit SA Score module.
        nn_scorer: Loaded RAscore NN scorer.
        xgb_scorer: Loaded RAscore XGB scorer.

    Returns:
        Path: Path to the saved output CSV file.

    Raises:
        ValueError: If the input CSV is missing the ID or SMILES columns.
    """
    df = pd.read_csv(csv_path)

    if "ID" not in df.columns or "SMILES" not in df.columns:
        raise ValueError(f"{csv_path} must contain 'ID' and 'SMILES' columns.")

    records = []

    for mol_id, smiles in zip(df["ID"], df["SMILES"]):
        molecule = Chem.MolFromSmiles(smiles)

        if molecule is None:
            print(f"Warning: invalid SMILES for {mol_id}, skipping: {smiles}")
            records.append(
                {
                    "id": mol_id,
                    "SAScore": float("nan"),
                    "NN Score": float("nan"),
                    "XGB Score": float("nan"),
                }
            )
            continue

        sa_score = round(float(sa_scorer.calculateScore(molecule)), 3)
        nn_score = round(float(nn_scorer.predict(smiles)), 3)
        xgb_score = round(float(xgb_scorer.predict(smiles)), 3)

        records.append(
            {
                "id": mol_id,
                "SAScore": sa_score,
                "NN Score": nn_score,
                "XGB Score": xgb_score,
            }
        )

    out_df = pd.DataFrame(records, columns=["id", "SAScore", "NN Score", "XGB Score"])

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False, sep='\t')

    return out_path


def main() -> None:
    """Score molecules in an input CSV file and save the results.

    Args:
        None.

    Returns:
        None.
    """
    args = get_args()

    sa_scorer = import_sa_scorer()
    nn_scorer = RAscore_NN.RAScorerNN()
    xgb_scorer = RAscore_XGB.RAScorerXGB()

    try:
        print(f"Scoring: {args.csv_path}")
        out_path = score_csv(
            args.csv_path, args.output, sa_scorer, nn_scorer, xgb_scorer
        )
        print(f"Saved: {out_path}")
    except (ValueError, FileNotFoundError) as error:
        print(f"Error processing {args.csv_path}: {error}")
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
