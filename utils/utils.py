from PIL import Image


def resize_and_center(image, target_width, target_height, fill_color=(255, 255, 255)):
    """
    Resize the image to fit within (target_width, target_height) while maintaining aspect ratio,
    and center it with padding to match the exact target size.

    Parameters:
    - image: PIL.Image object
    - target_width: Desired width of the final image
    - target_height: Desired height of the final image
    - fill_color: Background color used for padding

    Returns:
    - A resized and centered PIL.Image object
    """
    # Resize the image while maintaining the aspect ratio
    image.thumbnail((target_width, target_height), Image.Resampling.LANCZOS)

    # Create a new image with the desired size and fill color
    new_image = Image.new("RGB", (target_width, target_height), fill_color)

    # Calculate the position to center the resized image
    x_offset = (target_width - image.width) // 2
    y_offset = (target_height - image.height) // 2

    # Paste the resized image onto the new image with padding
    new_image.paste(image, (x_offset, y_offset))

    return new_image
