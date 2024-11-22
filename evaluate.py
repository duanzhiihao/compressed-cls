import argparse
import torch

from models.ccls import VCMClassify
from datasets import imcls_evaluate, get_valloader


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--s1",      type=str, default="cheng5")
    parser.add_argument("--s2",      type=str, default="")
    parser.add_argument("--s3",      type=str, default="res50-aa")
    parser.add_argument("--device",  type=str, default="cuda:0")
    parser.add_argument("--workers", type=int, default=4)
    cfg = parser.parse_args()

    device = torch.device(cfg.device)

    model = VCMClassify(stage1=cfg.s1, stage2=cfg.s2, stage3=cfg.s3, num_cls=1000)
    model.eval()
    model = model.to(device=device)

    url = "https://huggingface.co/duanzh0/my-model-weights/resolve/main/vcm/cheng5__res50-aa_l2_0.pt"
    checkpoint = torch.hub.load_state_dict_from_url(url, weights_only=True)
    model.stage3.load_state_dict(checkpoint["model"])

    # Evaluate the model
    dataloader = get_valloader(split='val',
        img_size=224, crop_ratio=1, interp='bicubic', input_norm=False,
        batch_size=128, workers=cfg.workers
    )
    results = imcls_evaluate(model, testloader=dataloader)
    print(results)


if __name__ == "__main__":
    main()
