"""
这里求解标签量化问题：
输入：stats.json(所有音符的频率统计)，量化个数
输出：量化阶级

求解如何设置这N个量化阶数，使得量化损失最小
KMeans
"""
from typing import Dict, Union
import json

import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from tqdm import tqdm


def round_quantize(value, granularity=0.01):
    return np.round(value / granularity) * granularity


def quantize(stats_data: Dict[Union[float, str], int], n=32, max_threshold=50):
    stats_data = {k: v for k, v in stats_data.items() if v > max_threshold}
    values = np.array(list(map(float, stats_data.keys())))
    frequencies = np.array(list(stats_data.values()))

    kmeans = KMeans(n_clusters=n, random_state=0).fit(values.reshape(-1, 1), sample_weight=frequencies)
    quantized_values = kmeans.cluster_centers_.flatten()

    loss = 0.0
    for value, freq in stats_data.items():
        value = float(value)
        closest_value = quantized_values[np.argmin(np.abs(quantized_values - value))]
        loss += (value - closest_value) ** 2 * freq
    loss = (loss / sum(values)) ** 0.5
    return np.sort(quantized_values), loss


if __name__ == '__main__':
    with open(f'../figs/stats.json', 'r') as f:
        data = json.load(f)

    duration = data['duration']
    quantized_durations, loss = quantize(duration, 32, max_threshold=50)
    print(round_quantize(quantized_durations))

    interval = data['interval']
    quantized_interval, loss2 = quantize(interval, 32, max_threshold=50)
    print(round_quantize(quantized_interval))
    with open('../figs/quantize_rules.json', 'w') as f:
        quantized_data = {'duration': round_quantize(quantized_durations).tolist(),
                          'interval': round_quantize(quantized_interval).tolist()}
        json.dump(quantized_data, f)
    # n_values = range(3, 128)
    # losses = []
    #
    # for n in tqdm(n_values):
    #     _, loss = quantize(duration, n)
    #     losses.append(loss)
    #
    # plt.figure(figsize=(10, 6))
    # plt.plot(n_values, losses, marker='o')
    # plt.xlabel('Number of Clusters (n)')
    # plt.ylabel('Quantization Loss')
    # plt.title('Elbow Method for Optimal Number of Clusters')
    # plt.grid(True)
    # plt.show()
