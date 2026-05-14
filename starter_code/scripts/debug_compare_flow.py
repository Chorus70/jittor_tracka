import os
import sys

os.environ.setdefault("HOME", "/data/qiaojiaxuan")
os.environ.setdefault("JITTOR_HOME", "/data/qiaojiaxuan/jittor_home")
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import jittor as jt
from omegaconf import OmegaConf

from src.model.parse import get_model


def load_model(cfg_path, ckpt):
    cfg = OmegaConf.to_container(OmegaConf.load(cfg_path))
    transform_cfg = OmegaConf.to_container(OmegaConf.load("configs/transform/predict.yaml"))
    model = get_model(model_config=cfg, transform_config=transform_cfg)
    model.load(ckpt)
    return model


def main():
    base = load_model("configs/model/vm_robust_ft.yaml", "experiments/vm_finetune/checkpoint_19.pkl")
    flow = load_model("configs/model/vm_flow_ft.yaml", "experiments/vm_flow_ft/checkpoint_19_expanded.pkl")

    np.random.seed(0)
    x = jt.array((np.random.randn(1, 128, 3) * 0.02).astype("float32"))

    y_base, _ = base.denoise_langevin_dynamics(x)
    y_flow, _ = flow.denoise_langevin_dynamics(x, noise_std=0.0125)
    print("out_diff", float((y_base - y_flow).abs().mean().data), float((y_base - y_flow).abs().max().data))

    f_base = base.encoder(x)
    cond = flow._encoder_input(
        x,
        pc_t=jt.zeros((1, 128, 1)),
        pc_noise_std=jt.ones((1, 128, 1)) * 0.0125,
    )
    f_flow = flow.encoder(cond)
    print("feat_diff", float((f_base - f_flow).abs().mean().data), float((f_base - f_flow).abs().max().data))
    print("shapes", f_base.shape, f_flow.shape)


if __name__ == "__main__":
    main()
