import torch


def main():
    model = torch.hub.load(
        "mateuszbuda/brain-segmentation-pytorch",
        "unet",
        n_channels=1,
        out_channels=1,
        init_features=32,
        pretrained=True,
    )


if __name__ is "__main__":
    main()
