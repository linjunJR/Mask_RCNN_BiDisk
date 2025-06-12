import os
import numpy as np
import cv2
import re

from mrcnn.config import Config
from mrcnn.utils import Dataset


############################################################
#  Dataset
############################################################

class DiskDataset(Dataset):
    def load_disks(self, dataset_dir, subset):
        self.add_class("disk", 1, "small_disk")
        self.add_class("disk", 2, "large_disk")

        # Train or validation dataset?
        assert subset in ["train", "val"]
        
        images_dir = os.path.join(dataset_dir, subset, "images")
        masks_dir = os.path.join(dataset_dir, subset, "masks")

        for filename in os.listdir(images_dir):
            image_id = os.path.splitext(filename)[0]
            mask_files = sorted([f for f in os.listdir(masks_dir) if f.startswith(image_id + "_object_")])
            image = cv2.imread(os.path.join(images_dir, filename))
            height, width = image.shape[:2]

            self.add_image(
                "disk",
                image_id=image_id,
                path=os.path.join(images_dir, filename),
                width=width,
                height=height,
                mask_files=[os.path.join(masks_dir, mf) for mf in mask_files]
            )

    def load_mask(self, image_id):
        info = self.image_info[image_id]
        mask_files = info["mask_files"]

        masks = []
        class_ids = []

        for path in mask_files:
            # Extract class_id from filename
            match = re.search(r"class_(\d+)", path)
            class_id = int(match.group(1)) if match else 1

            mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

            if mask.shape != (info['height'], info['width']):
                raise ValueError(f"Mask shape {mask.shape} doesn't match image shape {info['height'], info['width']}")
            
            mask = (mask > 128).astype(np.uint8)
            masks.append(mask)
            class_ids.append(class_id)

        if masks:
            masks = np.stack(masks, axis=-1)
            return masks, np.array(class_ids, dtype=np.int32)
        else:
            return super().load_mask(image_id)

