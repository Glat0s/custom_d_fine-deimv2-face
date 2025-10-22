# src/dl/torchvision_transforms.py

import random
import albumentations as A
import numpy as np

class RandomZoomOut(A.DualTransform):
    """
    Randomly zooms out an image and its bounding boxes.

    This transform creates a larger canvas, places the original image randomly on it,
    and then takes a random crop of the original size from the canvas. This is a
    native Albumentations implementation using NumPy/cv2, ensuring that the same
    random transformation is applied to both the image and its corresponding bounding boxes.

    Args:
        side_range (tuple, optional): The range of scaling factors for zooming out.
            A factor of 1.0 means no zoom. A factor of 2.0 would double the canvas size.
            Defaults to (1.0, 4.0).
        fill_value (int, optional): The value used to fill the area outside the original
            image on the new canvas. Defaults to 114 (a common value for grey padding).
        p (float, optional): The probability of applying the transform. Defaults to 0.5.
    """
    def __init__(self, side_range=(1.0, 4.0), fill_value=114, always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        if not (isinstance(side_range, (list, tuple)) and len(side_range) == 2):
            raise ValueError("side_range must be a tuple or list of two floats.")
        if not side_range[0] <= side_range[1]:
            raise ValueError("side_range[0] must be less than or equal to side_range[1].")
        if not side_range[0] >= 1.0:
            raise ValueError("RandomZoomOut only supports zooming out, so side_range[0] must be >= 1.0.")

        self.side_range = side_range
        self.fill_value = fill_value

    def get_params_dependent_on_data(self, params, data):
        """
        Calculates the random parameters for the transformation once per call.
        """
        img = data["image"]
        h, w = img.shape[:2]

        scale = random.uniform(self.side_range[0], self.side_range[1])
        canvas_h = int(h * scale)
        canvas_w = int(w * scale)
        
        # Where to paste the original image on the larger canvas
        paste_x = random.randint(0, canvas_w - w)
        paste_y = random.randint(0, canvas_h - h)

        # Where to crop the final image from the canvas
        crop_x = random.randint(0, canvas_w - w)
        crop_y = random.randint(0, canvas_h - h)
        
        return {
            "scale": scale,
            "orig_h": h,
            "orig_w": w,
            "canvas_h": canvas_h,
            "canvas_w": canvas_w,
            "paste_x": paste_x,
            "paste_y": paste_y,
            "crop_x": crop_x,
            "crop_y": crop_y,
        }

    def apply(self, img, canvas_h, canvas_w, paste_x, paste_y, crop_x, crop_y, orig_h, orig_w, **params):
        """
        Applies the transformation to the image.
        """
        # Create a new canvas with the fill value
        canvas = np.full((canvas_h, canvas_w, img.shape[2]), self.fill_value, dtype=img.dtype)
        
        # Paste the original image onto the canvas at a random position
        canvas[paste_y : paste_y + orig_h, paste_x : paste_x + orig_w] = img
        
        # Crop a random region of the original size from the canvas
        zoomed_out_img = canvas[crop_y : crop_y + orig_h, crop_x : crop_x + orig_w]
        
        return zoomed_out_img

    def apply_to_bboxes(self, bboxes, paste_x, paste_y, crop_x, crop_y, orig_h, orig_w, **params):
        """
        Applies the same transformation to the bounding boxes.
        """
        if not bboxes:
            return []

        # Calculate the net translation
        delta_x = paste_x - crop_x
        delta_y = paste_y - crop_y
        
        new_bboxes = []
        for bbox in bboxes:
            x_min, y_min, x_max, y_max = bbox[:4]
            
            # Apply the translation
            new_x_min = x_min + delta_x
            new_y_min = y_min + delta_y
            new_x_max = x_max + delta_x
            new_y_max = y_max + delta_y
            
            # Clip the bounding box to the final image dimensions (0, 0, orig_w, orig_h)
            clipped_x_min = max(0, new_x_min)
            clipped_y_min = max(0, new_y_min)
            clipped_x_max = min(orig_w, new_x_max)
            clipped_y_max = min(orig_h, new_y_max)
            
            # Filter out boxes that are completely outside the crop or have zero area
            if clipped_x_max > clipped_x_min and clipped_y_max > clipped_y_min:
                new_bboxes.append((clipped_x_min, clipped_y_min, clipped_x_max, clipped_y_max) + tuple(bbox[4:]))
                
        return new_bboxes

    def get_transform_init_args_names(self):
        return ("side_range", "fill_value")

    @property
    def targets_as_params(self):
        # We need the original image dimensions to calculate parameters
        return ["image"]
