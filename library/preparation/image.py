import io
from typing import Tuple, NamedTuple
import tensorflow as tf
import numpy as np
from PIL import Image


def resize_area(size: Tuple[int, int], target_area: float) -> Tuple[int, int]:
    """
    Calculates new dimensions with the same aspect ratio as the input
    such that the total number of pixels is close to the specified target are.

    :param size: The original size of the image.
    :param target_area: The targeted number of pixels (e.g. 512*512).
    :return: The new image size.
    """
    w, h = size
    if (w * h) <= target_area:
        return size
    aspect_ratio = w / h
    w = np.sqrt(target_area * aspect_ratio)
    h = w / aspect_ratio
    return int(w), int(h)


ImageData = NamedTuple('ImageData', [('width', int), ('height', int), ('channels', int),
                                     ('format', str), ('bytes', bytes)])


def get_image_bytes(file_path: str, max_width: int, max_height: int):
    # tf.gfile API abstracts different file system providers, such
    # as Google Cloud Storage, HDFS, etc.
    # See: https://stackoverflow.com/questions/42922948/why-use-tensorflow-gfile-for-file-i-o/
    image_data = tf.gfile.FastGFile(file_path, 'rb').read()
    bytes_io = io.BytesIO(image_data)

    img = Image.open(bytes_io)  # type: Image.Image
    img.thumbnail(resize_area(img.size, max_width * max_height))

    new = Image.new('RGB', img.size, (255, 255, 255))
    new.paste(img, mask=None)

    # The pixel information can be stored directly using
    # img.tobytes(). For sake of completeness, we're instead storing
    # the JPEG compressed image here instead.
    # See http://pillow.readthedocs.io/en/5.1.x/handbook/image-file-formats.html#jpeg
    bytes_io = io.BytesIO()
    new.save(bytes_io, format='JPEG', quality=80, optimize=True, progressive=False)
    data = bytes_io.getvalue()

    return ImageData(width=img.width, height=img.height, channels=3, format='jpg', bytes=data)
