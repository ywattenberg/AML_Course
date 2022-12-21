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


def augment_transfrom(samples, size=(256, 256), has_label=True, has_box=True):
    """

    Args:
        - sample: list of dicts with keys "frame", "label", "box" depending on the has_label and has_box
        - size: size of the output image
        - has_label: if True, the sample has a label
        - has_box: if True, the sample has a box

    Augment image using the following transforms:
        - ColorJitter
        - GaussianBlur
        - Random Equalize

    Augment image, label and box using the following transforms:
        - RandomAffine

    Input samples i.e. fames, labels, boxes should be in the following shape:
    [channels, height, width]

    """

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

    # Get random params for the transforms
    random_affine = transforms.RandomAffine.get_params(
        degrees=[-15.0, 15.0],  # degrees
        translate=(0.2, 0.2),
        scale_ranges=(0.8, 1.2),
        shears=[-10, 10],
        img_size=size,
    )
    out_dict = samples
    for sample in samples:
        # Transform image and label with the random params
        frame = sample["frame"]
        frame = image_transforms(frame)
        frame = transforms.functional.affine(frame, *random_affine)
        out_dict["frame"] = frame

        if has_label:
            label = sample["label"]
            label = transforms.functional.affine(label, *random_affine)
            out_dict["label"] = label

        if has_box:
            box = sample["box"]
            box = transforms.functional.affine(box, *random_affine)
            out_dict["box"] = box

        return out_dict
