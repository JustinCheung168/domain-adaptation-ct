import argparse
import numpy as np
from sklearn.model_selection import StratifiedKFold


def make_5_folds_domain_files(
    data,
    key_source, 
    key_target,         
    n_splits=5,
    shuffle=True,
    random_state=42,
    prefix="fold"
):

    X_src = data[key_source]
    X_tgt = data[key_target]
    y = data["label"]  

    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_state
    )

    for fold_idx, (_, fold_indices) in enumerate(skf.split(np.zeros(len(y)), y)):
        src_split = X_src[fold_indices]
        tgt_split = X_tgt[fold_indices]
        cls_split = y[fold_indices]

        images = np.concatenate([src_split, tgt_split], axis=0)
        labels1 = np.concatenate([cls_split, cls_split], axis=0)

        labels2 = np.concatenate([
            np.zeros(len(cls_split), dtype=np.int64),
            np.ones(len(cls_split), dtype=np.int64),
        ])

        filename = f"{prefix}{fold_idx}.npz"
        np.savez(filename, images=images, labels1=labels1, labels2=labels2)
        print(f"Saved {filename}")


def parse_args():
    parser = argparse.ArgumentParser(description="Create 5-fold domain adaptation files.")

    parser.add_argument("--data", required=True, help="Path to input .npz file")
    parser.add_argument("--source", required=True, help="Key for source domain in npz")
    parser.add_argument("--target", required=True, help="Key for target domain in npz")
    parser.add_argument("--splits", type=int, default=5, help="Number of folds")
    parser.add_argument("--prefix", default="fold_", help="Prefix for output filenames")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    data = np.load(args.data)

    make_5_folds_domain_files(
        data=data,
        key_source=args.source,
        key_target=args.target,
        n_splits=args.splits,
        shuffle=True,
        random_state=args.seed,
        prefix=args.prefix
    )
