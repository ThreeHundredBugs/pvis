# Python 3.10

import subprocess
import statistics
import re
import glob
import sys

def main() -> None:
    images = sorted(
        glob.glob('../images/*.png'),
        key=lambda i: (len(i), i),
    )
    for image in images:
        run_tests(image)


def run_tests(image_path: str) -> None:
    pattern = re.compile(r'convolution_time=(\d+) ')
    repeats = 10

    for threads_num in range(1, 33):
        time = [
            int(re.findall(pattern, r)[0])
            for r in run_n_times(
                f'./lab1.exe --threads-num {threads_num} --input-image {image_path} --kernel-size 12 --theta 1.5707963268',
                repeats,
            )
        ]
        mean_time = statistics.mean(time)
        print(f'image="{image_path}" threads={threads_num} repeats={repeats} mean_time={mean_time}')
        sys.stderr.write(f'{image_path},{threads_num},{repeats},{mean_time}\n')


def run_n_times(command: str, n: int) -> list[str]:
    result = []

    for _ in range(n):
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
        )
        process.wait()
        stdout, _ = process.communicate()
        result.append(stdout.decode('utf-8'))

    return result


if __name__ == '__main__':
    main()
