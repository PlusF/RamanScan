import numpy as np


def remove_cosmic_ray_1d(spectrum: np.ndarray, width: int, threshold: float):
    spectrum = spectrum.copy()
    intensity = np.diff(spectrum)
    median_int = np.median(intensity)
    mad_int = np.median([np.abs(intensity - median_int)])
    if mad_int == 0:
        mad_int = 1e-4
    modified_scores = 0.6745 * (intensity - median_int) / mad_int
    spikes = abs(modified_scores) > threshold

    for i in np.arange(len(spikes)):
        if spikes[i]:
            w = np.arange(i - width, i + 1 + width)  # スパイク周りの2 m + 1個のデータを取り出す
            w = w[(0 <= w) & (w < (spectrum.shape[0] - 1))]  # 範囲を超えないようトリミング
            w2 = w[spikes[w] == False]  # スパイクでない値を抽出し，
            if len(w2) > 0:
                spectrum[i] = np.mean(spectrum[w2])  # 平均を計算し補完
    return spectrum


def remove_cosmic_ray(spectrum: np.ndarray, width: int = 3, threshold: float = 7):
    if len(spectrum.shape) == 1:
        return remove_cosmic_ray_1d(spectrum, width, threshold)

    if len(spectrum.shape) == 2:
        data_removed = []
        for spec in spectrum:
            data_removed.append(remove_cosmic_ray_1d(spec, width, threshold))
        return np.array(data_removed)


def subtract_baseline(xdata: np.ndarray, ydata: np.ndarray, map_range_1: float, map_range_2: float):
    map_range_idx = (map_range_1 <= xdata) & (xdata <= map_range_2)
    ydata = ydata[:, :, map_range_idx]
    if ydata.shape[2] == 0:
        return None

    def sub(arr):
        baseline = np.linspace(arr[0], arr[-1], arr.shape[0])
        return ydata - baseline

    ydata = np.array([[sub(d).sum() for d in dat] for dat in ydata])
    return ydata


def generate_fake_data(size):
    spec = np.expand_dims(np.sin(np.linspace(-np.pi, np.pi, size)), axis=0) * np.random.randint(1, 10)
    noise = np.random.random(size) * 10
    cosmic_ray = np.zeros(size)
    cosmic_ray[np.random.randint(0, size)] = 100
    spec += noise + cosmic_ray
    return spec
