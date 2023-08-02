import numpy as np


def adaptive_noise_reduce(
    signal, sr, segment_len, estimate_noise_method, reduce_noise_method, smoothing=0.0, audio_len=None
):
    if audio_len:
        audio_div = audio_len / segment_len
    segment_len = int(segment_len * sr)
    div_points = np.arange(0, len(signal), segment_len).astype(int).tolist()
    div_points.append(len(signal))

    filtered_sig = np.ndarray((len(signal)))
    for i in range(len(div_points) - 1):
        start = div_points[i]
        end = div_points[i + 1]
        curr_segment = signal[start:end]

        if audio_len and i % audio_div == 0:
            temp_start = int(start + 5 * sr)
            temp_curr_segment = signal[temp_start:end]
            noise_estimated = estimate_noise_method(temp_curr_segment, sr)
        else:
            noise_estimated = estimate_noise_method(curr_segment, sr)

        if i == 0:
            noise_clip = noise_estimated
        else:
            noise_clip = smoothing * noise_clip + (1.0 - smoothing) * noise_estimated
        filtered_segment = reduce_noise_method(curr_segment, noise_clip)
        filtered_sig[start:end] = filtered_segment

    # Make sure data is of type np.float32
    filtered_sig = filtered_sig.astype(np.float32)

    return filtered_sig
