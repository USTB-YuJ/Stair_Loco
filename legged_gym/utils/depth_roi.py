"""Shared depth ROI helpers.

Crop pixels are ordered as [left, top, right, bottom], matching the existing
depth config convention. Returned windows are (top, bottom, left, right) with
exclusive bottom/right bounds for direct tensor/array slicing.
"""


def crop_window_from_pixels(image_shape, crop_pixels):
    height, width = int(image_shape[0]), int(image_shape[1])
    left, top, right, bottom = [int(v) for v in crop_pixels]

    if min(left, top, right, bottom) < 0:
        raise ValueError(f"Invalid crop {crop_pixels}: crop values must be non-negative.")

    bottom_exclusive = height - bottom
    right_exclusive = width - right

    if top >= bottom_exclusive or left >= right_exclusive:
        raise ValueError(
            f"Invalid crop {crop_pixels} for image shape {image_shape}: "
            f"resulting ROI is empty."
        )

    return top, bottom_exclusive, left, right_exclusive
