import random
import numpy as np
import torch

from torchvision import transforms


def augment_sample(sample, augment_label=True, label_name="label"):
    # random params
    move_x = random.randint(-20, 20)
    move_y = random.randint(-3, 3)
    noise_level = random.expovariate(20)
    zoom = random.random() * 0.5 + 1
    adjust_brightness = random.random() * 8

    frame = sample["frame"]
    transformed_frame = np.zeros(frame.shape)
    height = frame.shape[0]
    width = frame.shape[1]

    trns = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.CenterCrop((int(height / zoom), int(width / zoom))),
            transforms.Resize((height, width)),
        ]
    )

    transformed_frame[
        max(0, move_y) : min(height, height + move_y),
        max(0, move_x) : min(width, width + move_x),
    ] = frame[
        max(0, -move_y) : min(height, height - move_y),
        max(0, -move_x) : min(height, height - move_x),
    ]

    transformed_frame = trns(transformed_frame).numpy()
    transformed_frame *= adjust_brightness

    transformed_frame += np.random.random(transformed_frame.shape) * noise_level
    transformed_frame[transformed_frame > 1] = 1

    out_dict = {}
    out_dict["frame"] = transformed_frame
    out_dict["frames"] = sample["frames"]

    if augment_label:
        label = sample[label_name]
        transformed_label = np.zeros(label.shape)

        transformed_label[
            max(0, move_y) : min(height, height + move_y),
            max(0, move_x) : min(width, width + move_x),
        ] = label[
            max(0, -move_y) : min(height, height - move_y),
            max(0, -move_x) : min(height, height - move_x),
        ]

        transformed_label = trns(transformed_label).numpy()

        out_dict[label_name] = transformed_label
    # print(sample["box"].shape)
    # transformed_box = np.zeros(frame.shape[1:], dtype=bool)
    # transformed_box[
    #     max(0, move_y) : min(height, height + move_y),
    #     max(0, move_x) : min(width, width + move_x),
    # ] = sample["box"][
    #     max(0, -move_y) : min(height, height - move_y),
    #     max(0, -move_x) : min(height, height - move_x),
    # ]

    # transformed_box = trns(transformed_box).numpy()[0]

    # out_dict["box"] = transformed_box
    # out_dict["label"] = sample["label"]
    return out_dict


def augment_transfrom(
    samples, size=(256, 256), is_batched=False, has_label=True, has_box=True
):
    """

    Args:
        - samples: list of dicts with keys "frame", "label", "box" depending on the has_label and has_box
        - size: size of the output image
        - is_batched: if True, the samples are batched together i.e. [batch, channel, height, width]
        - has_label: if True, the sample has a label
        - has_box: if True, the sample has a box

    Augment image using the following transforms:
        - ColorJitter
        - GaussianBlur
        - Random Equalize

    Augment image, label and box using the following transforms:
        - RandomAffine

    Input samples i.e. fames, labels, boxes should be in the following shape:
    [channel, height, width]

    """
    # Get random params for the transforms
    random_affine = transforms.RandomAffine.get_params(
        degrees=[-15.0, 15.0],  # degrees
        translate=(0.2, 0.2),
        scale_ranges=(0.8, 1.2),
        shears=[-10, 10],
        img_size=size,
    )

    # Define the transforms that only need to be applied to the image
    image_transforms = transforms.Compose(
        [
            transforms.ColorJitter(  # brightness, contrast, saturation, hue
                brightness=0.5,
                contrast=0.3,
                saturation=0.3,
                hue=0.3,
            ),
            transforms.GaussianBlur(
                kernel_size=(3, 3),
                sigma=(0.1, 2.0),
            ),
            transforms.RandomEqualize(p=0.5),
        ]
    )
    if is_batched:
        batch_tensor = image_transforms(
            torch.concat(
                [sample["frame"].type(torch.uint8) for sample in samples], dim=0
            )
        )
    else:
        batch_tensor = image_transforms(
            torch.stack([sample["frame"].type(torch.uint8) for sample in samples])
        )
    batch_tensor = transforms.functional.affine(batch_tensor, *random_affine)

    batches = [t["frame"].shape[0] for t in samples]

    if is_batched:
        for i in range(len(samples)):
            samples[i]["frame"] = batch_tensor[
                sum(batches[:i]) : sum(batches[: (i + 1)])
            ].float()
    else:
        for i in range(len(samples)):
            samples[i]["frame"] = batch_tensor[i].float()

    out_dicts = samples
    for sample, out_dict in zip(samples, out_dicts):
        # Transform labels and box with the random params
        if has_label:
            label = sample["label"]
            label = transforms.functional.affine(label, *random_affine)
            out_dict["label"] = label

        if has_box:
            box = sample["box"]
            box = transforms.functional.affine(box, *random_affine)
            out_dict["box"] = box

        return out_dicts
