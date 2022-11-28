import pycuda.autoinit
from .emboss import emboss


def print_info() -> None:
    dev = pycuda.autoinit.device
    attrs = dev.get_attributes().items()
    print(dev.name())

    for name, value in attrs:
        print(f'  {repr(name).lstrip("pycuda._driver.device_attribute")}={value}')
