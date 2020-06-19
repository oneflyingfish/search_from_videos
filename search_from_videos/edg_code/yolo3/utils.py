"""Miscellaneous utility functions."""

from functools import reduce

from PIL import Image

def compose(*funcs):
    """Compose arbitrarily many functions, evaluated left to right.

    Reference: https://mathieularose.com/function-composition-in-python/
    """
    # return lambda x: reduce(lambda v, f: f(v), funcs, x)
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')

def letterbox_image(image, size):
    '''resize image with unchanged aspect ratio using padding'''

    # 等比例条件下尽可能放缩图像
    image_w, image_h = image.size
    w, h = size
    new_w = int(image_w * min(w*1.0/image_w, h*1.0/image_h))
    new_h = int(image_h * min(w*1.0/image_w, h*1.0/image_h))
    resized_image = image.resize((new_w,new_h), Image.BICUBIC)

    # 放缩后图像居中放入size大小背景框纯色背景框中RGB（128,128,128）
    boxed_image = Image.new('RGB', size, (128,128,128))             # 生成size大小的灰色图片
    boxed_image.paste(resized_image, ((w - new_w) // 2, (h - new_h) // 2))
    
    return boxed_image
