import numpy as np


def basic(ground_truth_gr, ground_truth_freq, alternative_gr, alternative_freq):
    """
    Calculate the difference metric between two sets of ground truth and alternative data.
    Parameters:
    ground_truth_gr (xarray): Ground truth growth rates.
    ground_truth_freq (xarray): Frequencies associated with ground truth growth rates.
    alternative_gr (xarray): Alternative growth rates.
    alternative_freq (xarray): Frequencies associated with alternative growth rates.
    Returns:
    float: The calculated difference metric.
    """
    # Calculate the weighted absolute differences
    gr_diff = np.abs(ground_truth_gr - alternative_gr)
    freq_diff = np.abs(ground_truth_freq - alternative_freq)
    # Sum the weighted differences to get the final metric
    difference_metric_value = np.sum(gr_diff.data) + numpy.sum(freq_diff.data)
    return difference_metric_value


def averaged(ground_truth_gr, ground_truth_freq, alternative_gr, alternative_freq):
    """
    Calculate the average absolute error between two sets of data.

    Returns
    -------
    float
        Average absolute error over all dimensions.
    """
    gr_diff = np.abs(ground_truth_gr - alternative_gr)
    freq_diff = np.abs(ground_truth_freq - alternative_freq)

    difference_metric_value = gr_diff.mean().item() + freq_diff.mean().item()

    return difference_metric_value


def stabalized(ground_truth_gr, ground_truth_freq, alternative_gr, alternative_freq):
    """
    Calculate the difference metric between two sets of ground truth and alternative data.
    Parameters:
    ground_truth_gr (xarray): Ground truth growth rates.
    ground_truth_freq (xarray): Frequencies associated with ground truth growth rates.
    alternative_gr (xarray): Alternative growth rates.
    alternative_freq (xarray): Frequencies associated with alternative growth rates.
    Returns:
    float: The calculated difference metric.
    """
    # removes all growth rates lower than 0.01 and all corresponding frequencies from both the ground truth and alternative values as these modes are staalised
    threshold = 0.01 * ground_truth_gr.data.units
    ground_truth_gr = ground_truth_gr.where(ground_truth_gr >= threshold, drop=True)
    ground_truth_freq = ground_truth_freq.where(ground_truth_gr >= threshold, drop=True)
    alternative_gr = alternative_gr.where(alternative_gr >= threshold, drop=True)
    alternative_freq = alternative_freq.where(alternative_gr >= threshold, drop=True)
    # Calculate the weighted absolute differences
    gr_diff = np.abs(ground_truth_gr - alternative_gr)
    freq_diff = np.abs(ground_truth_freq - alternative_freq)
    # Sum the weighted differences to get the final metric
    difference_metric_value = np.sum(gr_diff.data) + numpy.sum(freq_diff.data)
    return difference_metric_value
