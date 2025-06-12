import os
import numpy as np
import cv2
import re

from mrcnn.config import Config
from mrcnn.utils import Dataset

############################################################
#  Configurations
############################################################

class DiskConfig(Config):
    NAME = "disk" # Name the configurations. For example, 'COCO', 'Experiment 3', ...etc.
    NUM_CLASSES = 1 + 2  # background + binary disk
    GPU_COUNT = 1     # NUMBER OF GPUs to use. When using only a CPU, this needs to be set to 1.
    IMAGES_PER_GPU = 1 # Number of images to train with on each GPU. A 12GB GPU can typically handle 2 images of 1024x1024px.
    STEPS_PER_EPOCH = 100
    DETECTION_MIN_CONFIDENCE = 0.8
    VALIDATION_STEPS = 50
    
    # Backbone network architecture
    # Supported values are: resnet50, resnet101.
    # You can also provide a callable that should have the signature
    # of model.resnet_graph. If you do so, you need to supply a callable
    # to COMPUTE_BACKBONE_SHAPE as well
    BACKBONE = "resnet101"

    # Only useful if you supply a callable to BACKBONE. Should compute
    # the shape of each layer of the FPN Pyramid.
    # See model.compute_backbone_shapes
    COMPUTE_BACKBONE_SHAPE = None

############################################################
#  Dataset
############################################################

class DiskDataset(Dataset):
    def load_disks(self, dataset_dir, subset):
        self.add_class("disk", 1, "small_disk")
        self.add_class("disk", 2, "large_disk")

        # Train or validation dataset?
        assert subset in ["train", "val"]
        
        images_dir = os.path.join(dataset_dir, "images", subset)
        masks_dir = os.path.join(dataset_dir, "masks", subset)

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

