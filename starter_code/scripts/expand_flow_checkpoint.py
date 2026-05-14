import argparse
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--dst", required=True)
    args = parser.parse_args()

    os.environ.setdefault("HOME", "/data/qiaojiaxuan")
    os.environ.setdefault("JITTOR_HOME", "/data/qiaojiaxuan/jittor_home")

    import jittor as jt

    ckpt = jt.load(args.src)

    mlp_key = "encoder.conv1.mlp.0.weight"
    lin_key = "encoder.conv1.lin.0.weight"

    old_mlp = ckpt[mlp_key]
    old_lin = ckpt[lin_key]

    new_mlp = jt.zeros((old_mlp.shape[0], 10), dtype=old_mlp.dtype)
    new_mlp[:, 0:3] = old_mlp[:, 0:3]
    new_mlp[:, 5:8] = old_mlp[:, 3:6]

    new_lin = jt.zeros((old_lin.shape[0], 5), dtype=old_lin.dtype)
    new_lin[:, 0:3] = old_lin[:, 0:3]

    ckpt[mlp_key] = new_mlp
    ckpt[lin_key] = new_lin

    os.makedirs(os.path.dirname(args.dst), exist_ok=True)
    jt.save(ckpt, args.dst)
    print(f"wrote {args.dst}")


if __name__ == "__main__":
    main()
