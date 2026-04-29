import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import get_dataset
from .utils.data_utils import normalize_labels
from .utils.hydra import compose
from .utils.model_utils import ConvEncoder, ConvEncoderViTTiny


LABEL_STATS = {
    "active_matter": {
        "means": [-3.0, 9.0],  # alpha, zeta
        "stds": [1.41, 5.16],
        "names": ["alpha", "zeta"],
    },
    "shear_flow": {
        "means": [4.85, 2.69],
        "stds": [0.61, 3.38],
        "compression": ["log", None],
        "names": ["rayleigh", "schmidt"],
    },
    "rayleigh_benard": {
        "means": [2.69, 8.0],
        "stds": [3.38, 1.41],
        "compression": [None, "log"],
        "names": ["prandtl", "rayleigh"],
    },
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="Dataset name under THE_WELL_DATA_DIR, e.g. active_matter",
    )
    parser.add_argument(
        "--encoder_checkpoint",
        type=str,
        required=True,
        help="Path to a saved encoder state_dict checkpoint",
    )
    parser.add_argument(
        "--model_config",
        type=str,
        required=True,
        help="Path to the training config used to build the encoder",
    )
    parser.add_argument(
        "--probe_type",
        type=str,
        default="both",
        choices=["linear", "knn", "both"],
        help="Which frozen probe(s) to run",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Directory where metrics, predictions, and features will be written",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for frozen feature extraction",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of DataLoader workers for frozen feature extraction",
    )
    parser.add_argument(
        "--feature_reduction",
        type=str,
        default="avg",
        choices=["avg", "flatten"],
        help="How to convert encoder outputs into a fixed-size feature vector",
    )
    parser.add_argument(
        "--knn_k",
        type=int,
        default=5,
        help="Number of neighbors for kNN regression",
    )
    parser.add_argument(
        "--knn_weights",
        type=str,
        default="distance",
        choices=["uniform", "distance"],
        help="Weighting rule for kNN regression",
    )
    parser.add_argument(
        "--linear_alpha",
        type=float,
        default=0.0,
        help="Ridge regularization strength. Use 0.0 for ordinary linear regression.",
    )
    parser.add_argument(
        "--train_split",
        type=str,
        default="train",
        help="Split used to fit frozen probes",
    )
    parser.add_argument(
        "--val_split",
        type=str,
        default="valid",
        help="Validation split name on disk",
    )
    parser.add_argument(
        "--test_split",
        type=str,
        default="test",
        help="Test split name on disk",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic evaluation",
    )
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def build_encoder(cfg):
    in_chans = cfg.dataset.num_chans
    if cfg.model.get("vit_equivalency", None) == "tiny":
        encoder = ConvEncoderViTTiny(
            in_chans=in_chans,
            num_res_blocks=cfg.model.num_res_blocks,
            dims=cfg.model.dims,
        )
    else:
        encoder = ConvEncoder(
            in_chans=in_chans,
            num_res_blocks=cfg.model.num_res_blocks,
            dims=cfg.model.dims,
            num_frames=cfg.dataset.num_frames,
        )
    return encoder


def load_encoder(checkpoint_path: Path, cfg, device: torch.device):
    encoder = build_encoder(cfg)
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    encoder.load_state_dict(state_dict, strict=True)
    encoder.eval()
    encoder.to(device)
    return encoder


def make_dataloader(cfg, dataset_name: str, split: str, batch_size: int, num_workers: int):
    dataset = get_dataset(
        dataset_name=dataset_name,
        num_frames=cfg.dataset.num_frames,
        split=split,
        include_labels=True,
        resolution=cfg.dataset.get("resolution", None),
        offset=cfg.dataset.get("offset", None),
        subset_config_path=cfg.dataset.get("subset_config_path", None),
        noise_std=0.0,
    )
    loader_kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": False,
        "drop_last": False,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
    return DataLoader(**loader_kwargs)


def reduce_features(x: torch.Tensor, reduction: str) -> torch.Tensor:
    if reduction == "flatten":
        return x.flatten(1)

    if x.ndim == 5:
        return x.mean(dim=(2, 3, 4))
    if x.ndim == 4:
        return x.mean(dim=(2, 3))
    if x.ndim == 3:
        return x.mean(dim=1)
    if x.ndim == 2:
        return x
    raise ValueError(f"Unsupported encoder output shape for reduction: {tuple(x.shape)}")


def extract_split_features(
    encoder,
    loader,
    label_stats,
    device: torch.device,
    feature_reduction: str,
    split_name: str,
):
    all_features = []
    all_labels_z = []
    all_labels_raw = []
    output_shape = None

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"{split_name} features"):
            ctx = batch["context"].to(device, non_blocking=True)
            enc = encoder(ctx)
            feats = reduce_features(enc, feature_reduction).float().cpu().numpy()

            labels_raw = batch["physical_params"].float()
            labels_z = normalize_labels(labels_raw.clone(), stats=label_stats).numpy()

            all_features.append(feats)
            all_labels_z.append(labels_z)
            all_labels_raw.append(labels_raw.numpy())
            if output_shape is None:
                output_shape = list(enc.shape[1:])

    features = np.concatenate(all_features, axis=0)
    labels_z = np.concatenate(all_labels_z, axis=0)
    labels_raw = np.concatenate(all_labels_raw, axis=0)
    return features, labels_z, labels_raw, output_shape


def feature_stats(features: np.ndarray):
    centered = features - features.mean(axis=0, keepdims=True)
    variances = np.var(features, axis=0)
    stats = {
        "feature_dim": int(features.shape[1]),
        "num_examples": int(features.shape[0]),
        "mean_feature_variance": float(variances.mean()),
        "min_feature_variance": float(variances.min()),
        "max_feature_variance": float(variances.max()),
    }

    max_svs = min(10, min(centered.shape))
    if max_svs > 0:
        singular_values = np.linalg.svd(centered, full_matrices=False, compute_uv=False)[:max_svs]
        stats["top_singular_values"] = [float(x) for x in singular_values]
    else:
        stats["top_singular_values"] = []
    return stats


def unnormalize_labels(x: np.ndarray, stats: dict) -> np.ndarray:
    means = np.asarray(stats["means"], dtype=np.float32)
    stds = np.asarray(stats["stds"], dtype=np.float32)
    return x * stds + means


def mse_dict(y_true: np.ndarray, y_pred: np.ndarray, label_names):
    per_label = ((y_true - y_pred) ** 2).mean(axis=0)
    metrics = {}
    for idx, label_name in enumerate(label_names):
        metrics[f"mse_{label_name}"] = float(per_label[idx])
    metrics["mse_mean"] = float(per_label.mean())
    metrics["mse_global"] = float(mean_squared_error(y_true, y_pred))
    return metrics


def fit_linear_probe(train_x: np.ndarray, train_y: np.ndarray, alpha: float):
    if alpha > 0:
        model = Ridge(alpha=alpha, fit_intercept=True, random_state=0)
    else:
        model = LinearRegression()
    model.fit(train_x, train_y)
    return model


def fit_knn_probe(train_x: np.ndarray, train_y: np.ndarray, k: int, weights: str):
    model = KNeighborsRegressor(n_neighbors=k, weights=weights)
    model.fit(train_x, train_y)
    return model


def save_probe_outputs(
    results_dir: Path,
    prefix: str,
    split_name: str,
    y_pred_z: np.ndarray,
    y_true_z: np.ndarray,
    y_pred_raw: np.ndarray,
    y_true_raw: np.ndarray,
):
    np.save(results_dir / f"{prefix}_{split_name}_predictions.npy", y_pred_z)
    np.save(results_dir / f"{prefix}_{split_name}_targets.npy", y_true_z)
    np.save(results_dir / f"{prefix}_{split_name}_predictions_raw.npy", y_pred_raw)
    np.save(results_dir / f"{prefix}_{split_name}_targets_raw.npy", y_true_raw)


def main():
    args = parse_args()
    set_seed(args.seed)

    cfg = compose(args.model_config, args.overrides)
    checkpoint_path = Path(args.encoder_checkpoint).resolve()
    results_dir = Path(args.results_dir).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset_name not in LABEL_STATS:
        raise ValueError(f"Unsupported dataset for frozen regression: {args.dataset_name}")
    label_stats = LABEL_STATS[args.dataset_name]
    label_names = label_stats["names"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = load_encoder(checkpoint_path, cfg, device)

    train_loader = make_dataloader(cfg, args.dataset_name, args.train_split, args.batch_size, args.num_workers)
    val_loader = make_dataloader(cfg, args.dataset_name, args.val_split, args.batch_size, args.num_workers)
    test_loader = make_dataloader(cfg, args.dataset_name, args.test_split, args.batch_size, args.num_workers)

    train_x, train_y_z, train_y_raw, encoder_output_shape = extract_split_features(
        encoder, train_loader, label_stats, device, args.feature_reduction, args.train_split
    )
    val_x, val_y_z, val_y_raw, _ = extract_split_features(
        encoder, val_loader, label_stats, device, args.feature_reduction, args.val_split
    )
    test_x, test_y_z, test_y_raw, _ = extract_split_features(
        encoder, test_loader, label_stats, device, args.feature_reduction, args.test_split
    )

    np.save(results_dir / f"{args.train_split}_features.npy", train_x)
    np.save(results_dir / f"{args.val_split}_features.npy", val_x)
    np.save(results_dir / f"{args.test_split}_features.npy", test_x)
    np.save(results_dir / f"{args.train_split}_labels.npy", train_y_z)
    np.save(results_dir / f"{args.val_split}_labels.npy", val_y_z)
    np.save(results_dir / f"{args.test_split}_labels.npy", test_y_z)

    with open(results_dir / "feature_stats.json", "w") as f:
        json.dump(
            {
                "train": feature_stats(train_x),
                "encoder_output_shape": encoder_output_shape,
                "feature_reduction": args.feature_reduction,
            },
            f,
            indent=2,
        )

    metrics = {
        "dataset_name": args.dataset_name,
        "encoder_checkpoint": str(checkpoint_path),
        "model_config": str(Path(args.model_config).resolve()),
        "feature_reduction": args.feature_reduction,
        "knn_k": args.knn_k,
        "knn_weights": args.knn_weights,
        "linear_alpha": args.linear_alpha,
        "label_order": label_names,
    }

    if args.probe_type in {"linear", "both"}:
        linear_model = fit_linear_probe(train_x, train_y_z, args.linear_alpha)

        val_pred_z = linear_model.predict(val_x)
        test_pred_z = linear_model.predict(test_x)
        val_pred_raw = unnormalize_labels(val_pred_z, label_stats)
        test_pred_raw = unnormalize_labels(test_pred_z, label_stats)

        metrics["linear"] = {
            "val": mse_dict(val_y_z, val_pred_z, label_names),
            "test": mse_dict(test_y_z, test_pred_z, label_names),
        }
        save_probe_outputs(results_dir, "linear", "val", val_pred_z, val_y_z, val_pred_raw, val_y_raw)
        save_probe_outputs(results_dir, "linear", "test", test_pred_z, test_y_z, test_pred_raw, test_y_raw)

    if args.probe_type in {"knn", "both"}:
        knn_model = fit_knn_probe(train_x, train_y_z, args.knn_k, args.knn_weights)

        val_pred_z = knn_model.predict(val_x)
        test_pred_z = knn_model.predict(test_x)
        val_pred_raw = unnormalize_labels(val_pred_z, label_stats)
        test_pred_raw = unnormalize_labels(test_pred_z, label_stats)

        metrics["knn"] = {
            "val": mse_dict(val_y_z, val_pred_z, label_names),
            "test": mse_dict(test_y_z, test_pred_z, label_names),
        }
        save_probe_outputs(results_dir, "knn", "val", val_pred_z, val_y_z, val_pred_raw, val_y_raw)
        save_probe_outputs(results_dir, "knn", "test", test_pred_z, test_y_z, test_pred_raw, test_y_raw)

    with open(results_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
