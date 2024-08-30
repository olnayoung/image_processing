import numpy as np
import PIL
from PIL import Image, ImageFilter, ImageOps


def crop_image_mask(
    image,
    image_mask,
    width=512,
    height=512,
    mask_blur=4,
    inpaint_full_res_padding=32,
    preprocess_type="fill",
):
    image_mask = image_mask.convert("L")

    if mask_blur > 0:
        image_mask = image_mask.filter(ImageFilter.GaussianBlur(mask_blur))

    mask_for_overlay = image_mask
    mask = image_mask.convert("L")

    paste_to = 0, 0, *image.size
    if inpaint_full_res_padding > 0:
        crop_region = get_crop_region(np.array(mask), inpaint_full_res_padding)
        crop_region = expand_crop_region(
            crop_region, width, height, mask.width, mask.height
        )

        x1, y1, x2, y2 = crop_region

        mask = mask.crop(crop_region)
        image_mask = resize_image(2, mask, width, height)
        paste_to = (x1, y1, x2 - x1, y2 - y1)

    overlay_images = []

    latent_mask = image_mask

    imgs = []
    for img in [image]:
        image = flatten(img, "#ffffff")

        image_masked = Image.new("RGBa", (image.width, image.height))
        image_masked.paste(
            image.convert("RGBA").convert("RGBa"),
            mask=ImageOps.invert(mask_for_overlay.convert("L")),
        )

        overlay_images.append(image_masked.convert("RGBA"))

        if inpaint_full_res_padding > 0:
            # crop_region is not None if we are doing inpaint full res
            image = image.crop(crop_region)
            image = resize_image(2, image, width, height)

        if preprocess_type == "fill":
            image = fill(image, latent_mask)

        imgs.append(image)

    return imgs, mask, paste_to, overlay_images


def flatten(img, bgcolor):
    """replaces transparency with bgcolor (example: "#ffffff"), returning an RGB mode image with no transparency"""

    if img.mode == "RGBA":
        background = Image.new("RGBA", img.size, bgcolor)
        background.paste(img, mask=img)
        img = background

    return img.convert("RGB")


def resize_image(resize_mode, im, width, height, upscaler_name=None):
    def resize(im, w, h):
        LANCZOS = (
            Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
        )
        return im.resize((w, h), resample=LANCZOS)

    if resize_mode == 0:
        res = resize(im, width, height)

    elif resize_mode == 1:
        ratio = width / height
        src_ratio = im.width / im.height

        src_w = width if ratio > src_ratio else im.width * height // im.height
        src_h = height if ratio <= src_ratio else im.height * width // im.width

        resized = resize(im, src_w, src_h)
        res = Image.new("RGB", (width, height))
        res.paste(resized, box=(width // 2 - src_w // 2, height // 2 - src_h // 2))

    else:
        ratio = width / height
        src_ratio = im.width / im.height

        src_w = width if ratio < src_ratio else im.width * height // im.height
        src_h = height if ratio >= src_ratio else im.height * width // im.width

        resized = resize(im, src_w, src_h)
        res = Image.new("RGB", (width, height))
        res.paste(resized, box=(width // 2 - src_w // 2, height // 2 - src_h // 2))

        if ratio < src_ratio:
            fill_height = height // 2 - src_h // 2
            res.paste(
                resized.resize((width, fill_height), box=(0, 0, width, 0)), box=(0, 0)
            )
            res.paste(
                resized.resize(
                    (width, fill_height), box=(0, resized.height, width, resized.height)
                ),
                box=(0, fill_height + src_h),
            )
        elif ratio > src_ratio:
            fill_width = width // 2 - src_w // 2
            res.paste(
                resized.resize((fill_width, height), box=(0, 0, 0, height)), box=(0, 0)
            )
            res.paste(
                resized.resize(
                    (fill_width, height), box=(resized.width, 0, resized.width, height)
                ),
                box=(fill_width + src_w, 0),
            )

    return res


def get_crop_region(mask, pad=0):
    """finds a rectangular region that contains all masked ares in an image. Returns (x1, y1, x2, y2) coordinates of the rectangle.
    For example, if a user has painted the top-right part of a 512x512 image", the result may be (256, 0, 512, 256)
    """

    h, w = mask.shape

    crop_left = 0
    for i in range(w):
        if not (mask[:, i] == 0).all():
            break
        crop_left += 1

    crop_right = 0
    for i in reversed(range(w)):
        if not (mask[:, i] == 0).all():
            break
        crop_right += 1

    crop_top = 0
    for i in range(h):
        if not (mask[i] == 0).all():
            break
        crop_top += 1

    crop_bottom = 0
    for i in reversed(range(h)):
        if not (mask[i] == 0).all():
            break
        crop_bottom += 1

    return (
        int(max(crop_left - pad, 0)),
        int(max(crop_top - pad, 0)),
        int(min(w - crop_right + pad, w)),
        int(min(h - crop_bottom + pad, h)),
    )


def expand_crop_region(
    crop_region, processing_width, processing_height, image_width, image_height
):
    """expands crop region get_crop_region() to match the ratio of the image the region will processed in; returns expanded region
    for example, if user drew mask in a 128x32 region, and the dimensions for processing are 512x512, the region will be expanded to 128x128.
    """

    x1, y1, x2, y2 = crop_region

    ratio_crop_region = (x2 - x1) / (y2 - y1)
    ratio_processing = processing_width / processing_height

    if ratio_crop_region > ratio_processing:
        desired_height = (x2 - x1) / ratio_processing
        desired_height_diff = int(desired_height - (y2 - y1))
        y1 -= desired_height_diff // 2
        y2 += desired_height_diff - desired_height_diff // 2
        if y2 >= image_height:
            diff = y2 - image_height
            y2 -= diff
            y1 -= diff
        if y1 < 0:
            y2 -= y1
            y1 -= y1
        if y2 >= image_height:
            y2 = image_height
    else:
        desired_width = (y2 - y1) * ratio_processing
        desired_width_diff = int(desired_width - (x2 - x1))
        x1 -= desired_width_diff // 2
        x2 += desired_width_diff - desired_width_diff // 2
        if x2 >= image_width:
            diff = x2 - image_width
            x2 -= diff
            x1 -= diff
        if x1 < 0:
            x2 -= x1
            x1 -= x1
        if x2 >= image_width:
            x2 = image_width

    return x1, y1, x2, y2


def apply_overlay(image, paste_loc, overlay):
    if paste_loc is not None:
        x, y, w, h = paste_loc
        base_image = Image.new("RGBA", (overlay.width, overlay.height))
        image = resize_image(1, image, w, h)
        base_image.paste(image, (x, y))
        image = base_image

    image = image.convert("RGBA")
    image.alpha_composite(overlay)
    image = image.convert("RGB")

    return image


def fill(image: PIL.Image.Image, mask: PIL.Image.Image):
    """fills masked regions with colors from image using blur. Not extremely effective."""

    image_mod = Image.new("RGBA", (image.width, image.height))

    image_masked = Image.new("RGBa", (image.width, image.height))
    image_masked.paste(
        image.convert("RGBA").convert("RGBa"), mask=ImageOps.invert(mask.convert("L"))
    )

    image_masked = image_masked.convert("RGBa")

    for radius, repeats in [(256, 1), (64, 1), (16, 2), (4, 4), (2, 2), (0, 1)]:
        blurred = image_masked.filter(ImageFilter.GaussianBlur(radius)).convert("RGBA")
        for _ in range(repeats):
            image_mod.alpha_composite(blurred)

    return image_mod.convert("RGB")
