import argparse
import os

import torch
import torch.nn.functional as F

# ************************* General *************************
DATASET_MAP = {
    "aids": "AIDS",
    "bzr": "BZR_MD",
    "cifar": "CIFAR10",
    "mnist": "MNIST",
    "mutag": "MUTAG",
    "synthetic": "synthetic",
}

BEST_DATASET_PARAMETERS = {
    "AIDS": { 		
        "lr": 0.01,
        "wd": 0.001,
        "epochs": 300,
        "spc": 2,
        "spd": 1,
        "edo": 0.1,
        "ddo": 0.0,
    },
    "BZR_MD": {
        "lr": 0.015,
        "wd": 0.0025,
        "epochs": 400,
        "spc": 2,
        "spd": 4,
        "edo": 0.02,
        "ddo": 0.03,
    },
    "MNIST": {
        "lr": 0.001,
        "wd": 0.01,
        "epochs": 100,
        "spc": 3,
        "spd": 1,
        "edo": 0.0,
        "ddo": 0.3,
    },
    "CIFAR10": {
        "lr": 0.001,
        "wd": 0.0001,
        "epochs": 100,
        "spc": 3,
        "spd": 1,
        "edo": 0.1,
        "ddo": 0.3,
    },
    "MUTAG": {
        "lr": 0.015,
        "wd": 0.02,
        "epochs": 300,
        "spc": 3,
        "spd": 2,
        "edo": 0.0,
        "ddo": 0.6,
    },
}

ROOT_PATH = os.path.dirname(os.path.abspath(__file__)) + "/"

def parse_args() -> argparse.Namespace:  # pragma: no cover
    parser = argparse.ArgumentParser(
        "Global Counterfactual-based Self-explainable GNNs"
    )

    # General arguments
    parser.add_argument(
        "-d",
        "--dataset",
        default="aids",
        type=str,
        choices=list(DATASET_MAP.keys()),
        help="dataset name",
    )
    parser.add_argument("--device", type=int, default=0, help="CUDA GPU device")
    parser.add_argument(
        "-et",
        "--exp_type",
        default="gcsgnn",
        choices=["gcsgnn", "baseline", "ablation", "param"],
        help="type of experiment",
    )
    parser.add_argument("-bs", "--batch_size", type=int, default=500, help="Batch size")
    parser.add_argument(
        "-ne", "--num_exp", type=int, default=5, help="Number of experiments"
    )
    parser.add_argument(
        "-bt",
        "--baseline_type",
        default="gnn_explainer",
        choices=[
            "protgnn",
            "gnn_explainer",
            "induce",
            "gcf_explainer",
            "cf_gnnexplainer",
            "rcexplainer",
        ],
        help="Baseline method",
    )
    parser.add_argument(
        "-at",
        "--ablation_type",
        default=None,
        choices=["np", "ne", "nc", "ns", "nm", "nd"],
        help="Ablation method",
    )
    parser.add_argument(
        "-ed",
        "--embedding_dim",
        default=32,
        type=int,
        help="Embedding (hidden) dimension",
    )

    # GNN arguments
    parser.add_argument(
        "-glr", "--gnn_lr", type=float, default=1e-3, help="GNN learning rate"
    )
    parser.add_argument(
        "-gwd", "--gnn_wd", type=float, default=1e-3, help="GNN weight decay"
    )
    parser.add_argument(
        "-gdd", "--decoder_dropout", type=float, default=0.5, help="GNN decoder dropout"
    )
    parser.add_argument(
        "-ged", "--encoder_dropout", type=float, default=0.5, help="GNN encoder dropout"
    )
    parser.add_argument(
        "-ge", "--gnn_epochs", type=int, default=100, help="GNN training epochs"
    )
    parser.add_argument(
        "-spc",
        "--sub_patterns_per_class",
        type=int,
        default=3,
        help="Number of sub-patterns per class label",
    )
    parser.add_argument(
        "-spd",
        "--sub_pattern_dim",
        type=int,
        default=1,
        help="Dimension of sub-patterns",
    )

    parser.add_argument("-s", "--size", type=int, default=20, help="Reduced image size")

    # Baseline explainer arguments
    parser.add_argument(
        "-ee", "--exp_epochs", type=int, default=100, help="Explainer training epochs"
    )
    parser.add_argument(
        "-elr", "--exp_lr", type=float, default=1e-3, help="Explainer learning rate"
    )
    parser.add_argument(
        "-ewd",
        "--exp_weight_decay",
        type=float,
        default=1e-3,
        help="Explainer weight decay",
    )
    return parser.parse_args()


# ************************* Files *************************
def get_file_paths(args: argparse.Namespace) -> tuple[str, str, str, str]:
    if args.exp_type == "param":
        cfs_path = f"{ROOT_PATH}saved_results/gcsgnn/{args.dataset}/cfs_lr={args.gnn_lr}_wd={args.gnn_wd}_epochs={args.gnn_epochs}_spc={args.sub_patterns_per_class}_spd={args.sub_pattern_dim}_edo={args.encoder_dropout}_ddo={args.decoder_dropout}_s={args.size}_exp=exp-#"
        model = cfs_path.replace("/cfs", "/model") + ".pt"
        model_results = cfs_path.replace("/cfs", "/training_results") + ".txt"
        exp_results = cfs_path.replace("/cfs", "/exp_results").replace(
            "_exp=exp-#", ".txt"
        )
    elif args.exp_type in ["gcsgnn", "ablation"]:
        best_gnn_params = BEST_DATASET_PARAMETERS[args.dataset]
        cfs_path = f"{ROOT_PATH}saved_results/{args.exp_type}/{args.ablation_type}/{args.dataset}/cfs_lr={best_gnn_params['lr']}_wd={best_gnn_params['wd']}_epochs={best_gnn_params['epochs']}_spc={best_gnn_params['spc']}_spd={best_gnn_params['spd']}_edo={best_gnn_params['edo']}_ddo={best_gnn_params['ddo']}_s={args.size}_exp=exp-#"
        if args.exp_type == "gcsgnn":
            cfs_path = cfs_path.replace("None/", "")  # noqa: E701
        model = cfs_path.replace("/cfs", "/model") + ".pt"
        model_results = cfs_path.replace("/cfs", "/training_results") + ".txt"
        exp_results = cfs_path.replace("/cfs", "/exp_results").replace(
            "_exp=exp-#", ".txt"
        )
    else:
        best_gnn_params = BEST_DATASET_PARAMETERS[args.dataset]
        model = f"{ROOT_PATH}saved_results/gcsgnn/{args.dataset}/model_lr={best_gnn_params['lr']}_wd={best_gnn_params['wd']}_epochs={best_gnn_params['epochs']}_spc={best_gnn_params['spc']}_spd={best_gnn_params['spd']}_edo={best_gnn_params['edo']}_ddo={best_gnn_params['ddo']}_s={args.size}_exp=exp-#.pt"
        cfs_path = f"{ROOT_PATH}saved_results/baselines/{args.baseline_type}/{args.dataset}/cfs_glr={args.gnn_lr}_gwd={args.gnn_wd}_ge={best_gnn_params['epochs']}_exp=exp-#"
        model_results = cfs_path.replace("/cfs_", "/training_results_") + ".txt"
        exp_results = (
            cfs_path.replace("cfs_", "exp_results_").replace("exp=exp-#", "") + ".txt"
        )
    os.makedirs(os.path.dirname(exp_results), exist_ok=True)
    return model, model_results, cfs_path, exp_results


def update_file_paths(
    model: str, model_results: str, cfs_path: str, exp_num: int
) -> tuple[str, str, str]:
    model = model.replace("exp=exp-#", f"exp=exp-{exp_num}")
    model_results = model_results.replace("exp=exp-#", f"exp=exp-{exp_num}")
    cfs_path = cfs_path.replace("exp=exp-#", f"exp=exp-{exp_num}")
    return model, model_results, cfs_path


# ************************* Model Functions *************************
def vector_to_symm_matrix(
    vector: torch.Tensor, n_rows: int, n_rows_pad: int = 0
) -> torch.Tensor:
    """
    Converts a vector into a matrix by using values of the vector to fill in the lower
    triangle of the matrix and adds the transpose to create a symmetric matrix. Add n_rows_pad
    to create a matrix of shape (n_rows + n_rows_pad, n_rows + n_rows_pad) where the padding
    is added to the ends of the matrix (bottom and right sides)\n
    See example in symm_matrix_to_vector
    """
    matrix = torch.zeros(n_rows, n_rows).to(vector.device)
    idx = torch.tril_indices(n_rows, n_rows)
    matrix[idx[0], idx[1]] = vector
    symm_matrix = torch.tril(matrix) + torch.tril(matrix, -1).t()
    return F.pad(symm_matrix, (0, n_rows_pad, 0, n_rows_pad))


def symm_matrix_to_vector(matrix: torch.Tensor) -> torch.Tensor:
    """
    Converts a symmetric matrix into a vector by taking the values in the lower
    triangle of the matrix (including the diagonal values) from top to bottom. \n
    Ex:\n
    [
        [0, 1, 1, 1],\n
        [1, 0, 1, 1],\n
        [1, 1, 0, 0],\n
        [1, 1, 0, 0],\n
    ]\n
    becomes [0, 1, 0, 1, 1, 0, 1, 1, 0, 0] where the indices for each element are \n
    [(0,0), (1,0), (1,1), (2,0), (2, 1), (2,2), (3, 0), (3,1), (3,2), (3,3)]
    """
    idx = torch.tril_indices(matrix.shape[0], matrix.shape[0])
    vector = matrix[idx[0], idx[1]]
    return vector
