import json
from collections import defaultdict
from statistics import mean

with open('test_result.txt') as f:
    lines = f.read().split('\n')
    lines.sort()

agg = defaultdict(list)

for line in lines:
    image, name, time = line.split(': ')
    time = float(time[:-len('ns')])
    agg[(image, name)].append(time)

stats = {
    str(k): mean(v) / 1_000_000
    for k, v in agg.items()
}

print(json.dumps(stats, indent=2))
