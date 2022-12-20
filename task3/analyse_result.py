import utils
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

ORIGINAL_DATA = "data/test.pkl"
SEGMENTATION = "submission_256_1_200.pkl"
BOX = None
FOLDER = "results"


def main():
    if not os.path.exists(FOLDER):
        os.mkdir(FOLDER)

    original_data = utils.load_zipped_pickle(ORIGINAL_DATA)
    segmentation = utils.load_zipped_pickle(SEGMENTATION)
    
    
    for i in range(1):
        orig_vid = original_data[i]["video"]
        seg_vid = segmentation[i]["prediction"]
        assert(orig_vid.shape == seg_vid.shape)

        if not os.path.exists(f"{FOLDER}/{i}"):
            os.mkdir(f"{FOLDER}/{i}")

        if BOX is not None:
            box = original_data[i]["box"]

        for j in range(orig_vid.shape[2]):
            if BOX is None:
                utils.overlay_segmentation(orig_vid[:, :, j ], seg_vid[:, :, j], f"{FOLDER}/{i}/segmentation_{j}")
            else:
                utils.overlay_segmentation(orig_vid[:, :, j], seg_vid[:, :, j], f"{FOLDER}/{i}/segmentation_{j}", box[:, :, i])



            
            


if __name__ == "__main__":
    main()