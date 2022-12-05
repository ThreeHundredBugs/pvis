# Загрузить цветное изображение.
# Выполнить свертку с фильтром, придающим рельеф с каждым из цветовых
# каналов изображения. Ядро преобразования можно найти по ссылке:
# https://docs.gimp.org/2.8/ru/plug-in-convmatrix.html 
# Сохранить результат в файл.


import logging
import argparse
import sys
from mygpu import print_info, convert


def main() -> None:
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--input', required=True, help='Path to source PNG image')
    args_parser.add_argument('--output', required=True, help='Path to save filtered PNG image')
    args_parser.add_argument(
        '--log-level',
        choices=[
            'CRITICAL',
            'FATAL',
            'ERROR',
            'WARNING',
            'WARN',
            'INFO',
            'DEBUG',
        ],
        default='INFO',
    )
    args = args_parser.parse_args(sys.argv[1:])

    logging.basicConfig(
        level=args.log_level,
        format='%(asctime)s.%(msecs)03d %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    print_info()
    convert(args.input, args.output)


if __name__ == '__main__':
    main()
