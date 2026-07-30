import argparse
import os
import sys
from pathlib import Path

# Disable TensorFlow GPU detection before importing TensorFlow-dependent modules.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Suppress TensorFlow INFO and WARNING messages.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

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
        description="Calculate RAscore and RDKit SA Score."
    )

    parser.add_argument(
        "--smiles",
        type=str,
        required=True,
        help="SMILES string of the molecule to score.",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["NN", "XGB", "both"],
        default="both",
        help="RAscore model to use for scoring.",
    )

    return parser.parse_args()


def calculate_sa_score(smiles: str) -> float:
    """Calculate the RDKit synthetic accessibility score.

    Args:
        smiles: Input molecule represented as a SMILES string.

    Returns:
        float: Synthetic accessibility score ranging approximately from 1 to 10.

    Raises:
        ValueError: If the input SMILES string is invalid.
    """
    molecule = Chem.MolFromSmiles(smiles)

    if molecule is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    sa_scorer = import_sa_scorer()
    return float(sa_scorer.calculateScore(molecule))


def calculate_nn_score(smiles: str) -> float:
    """Calculate the neural-network-based RAscore.

    Args:
        smiles: Input molecule represented as a SMILES string.

    Returns:
        float: Neural-network-based RAscore.
    """
    scorer = RAscore_NN.RAScorerNN()
    return float(scorer.predict(smiles))


def calculate_xgb_score(smiles: str) -> float:
    """Calculate the XGBoost-based RAscore.

    Args:
        smiles: Input molecule represented as a SMILES string.

    Returns:
        float: XGBoost-based RAscore.
    """
    scorer = RAscore_XGB.RAScorerXGB()
    return float(scorer.predict(smiles))


def main() -> None:
    """Calculate and print RDKit SA Score and selected RAscore predictions.

    Args:
        None.

    Returns:
        None.
    """
    args = get_args()

    try:
        # Calculate SA Score independently of the selected RAscore model.
        sa_score = calculate_sa_score(args.smiles)
        print(f"SA Score: {sa_score:.3f}")

        if args.model in {"NN", "both"}:
            nn_score = calculate_nn_score(args.smiles)
            print(f"NN Score: {nn_score:.3f}")

        if args.model in {"XGB", "both"}:
            xgb_score = calculate_xgb_score(args.smiles)
            print(f"XGB Score: {xgb_score:.3f}")

    except (ValueError, ImportError) as error:
        print(f"Error: {error}")
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()