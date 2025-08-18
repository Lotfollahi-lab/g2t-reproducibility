
import argparse, torch, sys
from pathlib import Path

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--repo", required=True)
    p.add_argument("--dataset", default="proteins")
    p.add_argument("--ds_filename", required=True)
    p.add_argument("--k", type=int, default=16)
    p.add_argument("--smallest", action="store_true")
    p.add_argument("--scaler", default="standard")
    p.add_argument("--n_train", type=int, default=512)
    p.add_argument("--n_val",   type=int, default=256)
    p.add_argument("--steps",   type=int, default=200)
    p.add_argument("--device",  default="cuda")
    p.add_argument("--out",     required=True)
    args = p.parse_args()

    repo = Path(args.repo)
    sys.path.append(str(repo))

    from models.diffusion import SpectralDiffusion
    from dataset.load_data_generated import LaplacianDatasetNX

    device = args.device if torch.cuda.is_available() else "cpu"
    model = SpectralDiffusion.load_from_checkpoint(args.ckpt, strict=False, map_location="cpu").eval().to(device)

    train_set = LaplacianDatasetNX(args.dataset, args.ds_filename, point_dim=args.k,
                                   smallest=args.smallest, split='train_train',
                                   scaler=args.scaler, nodefeatures=False)

    def sample_pairs(n):
        sizes = list(train_set.sample_n_nodes(max(1, n-1))) + [train_set.n_max]
        with torch.no_grad():
            X, Y = model.sample_eigs(
                max_nodes=sizes,
                num_eigs=args.k + getattr(model.hparams, "feature_size", 0),
                scale_xy=train_set.scale_xy,
                unscale_xy=train_set.unscale_xy,
                num_graphs=len(sizes),
                oversample_mult=1,
                device=device,
                sampling_steps=args.steps,
                reproject=True,
            )
        return X.cpu(), Y.cpu()

    gx_tr, gy_tr = sample_pairs(args.n_train)
    gx_va, gy_va = sample_pairs(args.n_val)
    torch.save({"gen_x_train": gx_tr, "gen_y_train": gy_tr, "gen_x_val": gx_va, "gen_y_val": gy_va}, args.out)
    print("Saved:", args.out, "train:", tuple(gx_tr.shape), "val:", tuple(gx_va.shape))

if __name__ == "__main__":
    main()
