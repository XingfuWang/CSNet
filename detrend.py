import numpy as np


def detrend(data, window_length, step_size):
    """
    Remove drift trend along the time axis using a sliding window approach.

    Parameters:
    - data: numpy array of shape [samples, channels, timepoints]
    - window_length: int, length of the sliding window
    - step_size: int, step size for the sliding window

    Returns:
    - detrended_data: numpy array of the same shape as input data
    """
    samples, channels, timepoints = data.shape
    detrended_data = np.zeros_like(data)

    for sample in range(samples):
        for channel in range(channels):
            for start in range(0, timepoints - window_length + 1, step_size):
                end = start + window_length
                window = data[sample, channel, start:end]
                mean_value = np.mean(window)
                detrended_data[sample, channel, start:end] = window - mean_value

            # Handle the end part if it was not covered by the sliding windows
            if end < timepoints:
                window = data[sample, channel, end:]
                mean_value = np.mean(window)
                detrended_data[sample, channel, end:] = window - mean_value

    return detrended_data


# Example usage for baseline correction:
train_data = np.random.randn(3, 62, 1000)  # example train data, shape (batch_size, channels, timepoints)
test_data = np.random.randn(3, 62, 1000)  # example test data, shape (batch_size, channels, timepoints)
# reference:
train_data = train_data - train_data.mean(axis=2)[:, :, np.newaxis]
test_data = test_data - test_data.mean(axis=2)[:, :, np.newaxis]
# remove drift
train_data_corrected = detrend(train_data, window_length=100, step_size=2)
test_data_corrected = detrend(test_data, window_length=100, step_size=2)
