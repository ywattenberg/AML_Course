import random
import numpy as np

from torchvision import transforms


def augment_sample(sample, augment_label=True):
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
        label = sample["label"]
        transformed_label = np.zeros(label.shape)

        transformed_label[
            max(0, move_y) : min(height, height + move_y),
            max(0, move_x) : min(width, width + move_x),
        ] = label[
            max(0, -move_y) : min(height, height - move_y),
            max(0, -move_x) : min(height, height - move_x),
        ]

        transformed_label = trns(transformed_label).numpy()

        out_dict["label"] = transformed_label
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
