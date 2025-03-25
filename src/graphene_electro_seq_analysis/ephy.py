import sys
from scipy import signal
from scipy.signal import find_peaks, correlate
from scipy.ndimage import gaussian_filter1d
import pandas as pd
import time
import os
import glob
import numpy as np
import pywt
from scipy.signal import savgol_filter
import spikeinterface.preprocessing as spre
import spikeinterface.extractors as se

sys.path.append('../src/')
from graphene_electro_seq_analysis.importrhdutilities import load_file
import shutil
from tqdm import tqdm
import numpy as np


# RHD file loading for figure 2
def process_rhd_files_longitudinal(recording_dates, channel_indices, data_path, output_path, 
                      cutoff_frequency=60, window_length=101, polyorder=3):
    """
    Process multiple .rhd files, extract traces from specified channels, 
    and apply filtering and smoothing.
    
    Parameters:
    -----------
    recording_dates : list
        List of strings representing recording dates (e.g., ['220922', '220926'])
    channel_indices : list of lists
        List of channel indices to extract for each recording date
    data_path : str
        Base path where recording date folders are located
    output_path : str
        Path where results will be saved
    cutoff_frequency : int, optional
        Cutoff frequency for lowpass filter in Hz (default: 60)
    window_length : int, optional
        Window length for Savitzky-Golay filter (default: 101)
    polyorder : int, optional
        Polynomial order for Savitzky-Golay filter (default: 3)
        
    Returns:
    --------
    dict
        Dictionary with recording dates as keys and filtered/smoothed traces as values
    """
    # Dictionary to store traces for each recording date
    recording_traces = {}
    
    # Process each recording date
    for i in range(len(recording_dates)):
        # Get paths for current recording date
        dat_path = os.path.join(data_path, str(recording_dates[i]))
        search_pattern = os.path.join(dat_path, '*.rhd')
        rhd_paths = sorted(glob.glob(search_pattern))
        
        # Get channel indices for current recording date
        channel_idx = channel_indices[i]
        
        # List to store traces from all files for this recording date
        all_traces = []
        
        # Process each .rhd file
        for rhd_path in rhd_paths:
            raw_data, data_present = load_file(rhd_path)
            if not data_present:
                continue
                
            # Collect traces for all specified channels
            traces = [raw_data['amplifier_data'][j] for j in channel_idx]
            traces = np.vstack(traces)
            all_traces.append(traces)
        
        # Combine all traces horizontally if data is present
        if all_traces:
            all_traces = np.hstack(all_traces)
            recording_traces[recording_dates[i]] = all_traces
    
    # Apply filtering and smoothing
    filtered_and_smoothed_traces = {}
    
    # Get sampling frequency from the last processed raw data
    # (Assuming all files have the same sampling frequency)
    sampling_frequency = int(raw_data['frequency_parameters']['amplifier_sample_rate'])
    
    # Process each recording
    for key, data in recording_traces.items():
        # Initialize array for processed data
        processed_data = np.zeros_like(data)
        
        # Apply filtering and smoothing to each channel
        for channel in range(data.shape[0]):
            # Apply lowpass filter
            y_filtered = butter_lowpass_filter(data[channel, :], cutoff_frequency, sampling_frequency)
            
            # Apply Savitzky-Golay filter for smoothing
            y_smoothed = signal.savgol_filter(y_filtered, window_length=window_length, polyorder=polyorder)
            
            # Store the processed channel data
            processed_data[channel, :] = y_smoothed
            
        # Store the processed data in the dictionary
        filtered_and_smoothed_traces[key] = processed_data
    
    return filtered_and_smoothed_traces


def butter_lowpass_filter(data, cutoff_frequency, sampling_frequency, order=3):
    """
    Apply a Butterworth lowpass filter to the data.
    
    Parameters:
    -----------
    data : array_like
        The data to filter
    cutoff_frequency : float
        The cutoff frequency in Hz
    sampling_frequency : float
        The sampling frequency in Hz
    order : int, optional
        The order of the filter (default: 4)
        
    Returns:
    --------
    array_like
        Filtered data
    """
    nyquist = 0.5 * sampling_frequency
    normalized_cutoff = cutoff_frequency / nyquist
    b, a = signal.butter(order, normalized_cutoff, btype='low', analog=False)
    y = signal.filtfilt(b, a, data)
    return y

# Load and process singledate recording
def process_rhd_files_single_recording(rhd_paths, active_channel_names=None, channel_indices=None,
                            freq_min=0.3, freq_max=150, window_length=101, polyorder=3):
    """
    Process a single recording session from multiple .rhd files.
    
    Parameters:
    -----------
    rhd_paths : list
        List of paths to .rhd files for a single recording session
    active_channel_names : list, optional
        List of channel names to extract. If provided, will use these to find indices.
    channel_indices : list, optional
        List of channel indices to extract. If active_channel_names is provided, this is ignored.
    freq_min : float, optional
        Minimum frequency for bandpass filter in Hz (default: 0.3)
    freq_max : float, optional
        Maximum frequency for bandpass filter in Hz (default: 150)
    window_length : int, optional
        Window length for Savitzky-Golay filter (default: 101)
    polyorder : int, optional
        Polynomial order for Savitzky-Golay filter (default: 3)
        
    Returns:
    --------
    numpy.ndarray
        Processed traces data
    float
        Sampling frequency
    """
    # List to store traces from all files
    all_traces = []
    
    # Process each .rhd file
    for rhd_path in rhd_paths:
        raw_data, data_present = load_file(rhd_path)
        if not data_present:
            continue
        
        # Extract channel data based on either names or indices
        if active_channel_names is not None:
            # Get channel names and find their indices
            channel_names = [channel['native_channel_name'] for channel in raw_data['amplifier_channels']]
            active_channel_indices = [channel_names.index(name) for name in active_channel_names]
            all_traces.append(raw_data['amplifier_data'][active_channel_indices])
        elif channel_indices is not None:
            # Use directly provided channel indices
            traces = [raw_data['amplifier_data'][j] for j in channel_indices]
            all_traces.append(np.vstack(traces))
        else:
            raise ValueError("Either active_channel_names or channel_indices must be provided")
    
    # Combine all traces horizontally if data is present
    if not all_traces:
        return None, None
    
    all_traces = np.hstack(all_traces)
    
    # Get sampling frequency
    sampling_frequency = int(raw_data['frequency_parameters']['amplifier_sample_rate'])
    
    # Create NumpyRecording object for spikeinterface processing
    rec = se.NumpyRecording(traces_list=all_traces.T, sampling_frequency=sampling_frequency)
    
    # Apply band-pass filter
    rec = spre.bandpass_filter(rec, freq_min=freq_min, freq_max=freq_max)
    
    # Remove common noise
    rec = spre.common_reference(rec, reference='global', operator='median')
    
    # Get filtered traces
    filtered_traces = rec.get_traces().T
    
    # Apply Savitzky-Golay filtering
    filtered_traces = savgol_filter(filtered_traces, window_length, polyorder)
    
    return filtered_traces, sampling_frequency


def peak_detection(recording_data, target_channel, negative=True, threshold=5, distance=5000):
    ''' 
    Find peaks from recording data
    Args:
        recording_data (Dataframe): Recording data.
        target_channel (string): Target channel.(e.g., 'ch0')
        threshold (int): peak detection threshold (e.g., 1,2,3,4,5) 
        distance (int): distance between each peak (e.g., 5000, 10000)
    Returns:
        peaks (list): Waveforms of the spikes from the target channel.
    '''
     
    sig = recording_data[target_channel]
    thres = threshold * np.std(sig)
    mean_targeted_ch = np.mean(sig)
    
    if negative:
        peaks, _ = signal.find_peaks(-sig, height= thres+mean_targeted_ch, distance=distance) # Multiplied -1 to detect negative peaks 
    else:
        peaks, _ = signal.find_peaks(sig, height= thres+mean_targeted_ch, distance=distance ) 
    
    return (peaks)


def spike_waveform(recording_data, target_channel, peaks, pre=0.6, post=1.0):
    ''' 
    Extract and align spike waveforms
    Args:
        recording_data (Dataframe): Recording data
        target_channel (string): target channel (e.g., 'ch0')
        peaks (dict): Dictionary of the peak lists (e.g., 'ch0': [1,2,3,4,5])
        
    Returns:
        spike_waveforms (list): Waveforms of the spikes from the target channel.
    '''

    # Set windows before and after spike peak
    pre_spike_window = int(pre * 10000)  # window length before spike peak (2.5s before)
    post_spike_window = int(post * 10000)  # window length after spike peak (1.5s after)
    
    spike_waveforms = []  # to store waveforms

    if isinstance(recording_data, pd.DataFrame):
        signal_data = recording_data[target_channel].values  # getting the signal data as numpy array
    else:
        signal_data = recording_data[target_channel]  # getting the signal data as numpy array if it's a numpy array itself
      
    peak_idx = peaks[target_channel]
    
    for idx in peak_idx:
        if idx - pre_spike_window >= 0 and idx + post_spike_window < len(signal_data):
        # To ensure that we're not exceeding the bounds of the data array:
            peak_waveform = signal_data[idx - pre_spike_window : idx + post_spike_window]
            spike_waveforms.append(peak_waveform)
    
    return (spike_waveforms)

# Modified codes for waveform align (20240315)
def preprocess_waveforms(peak_waveforms):
    # Apply Gaussian smoothing to each waveform
    return [gaussian_filter1d(waveform, sigma=1) for waveform in peak_waveforms]

def create_template_waveform(waveforms_aligned):
    # Create a template waveform by averaging all aligned waveforms
    return np.mean(waveforms_aligned, axis=0)

def align_to_template(waveform, template):
    # Align a waveform to a template using cross-correlation
    correlation = correlate(waveform, template, mode='full')
    max_corr_index = np.argmax(correlation)
    shift = max_corr_index - len(template) + 1
    return np.roll(waveform, -shift), shift, correlation[max_corr_index]

def filter_waveforms(aligned_waveforms, correlation_scores, quality_threshold):
    # Filter out waveforms with a correlation score below the threshold
    good_waveforms = [waveform for waveform, score in zip(aligned_waveforms, correlation_scores) if score > quality_threshold]
    return good_waveforms

def align_and_filter_waveforms(peak_waveforms, quality_threshold):
    # Preprocess the waveforms
    preprocessed_waveforms = preprocess_waveforms(peak_waveforms)

    # Create the initial template waveform
    template_waveform = create_template_waveform(preprocessed_waveforms)

    # Align waveforms to the template
    aligned_waveforms = []
    correlation_scores = []
    for waveform in preprocessed_waveforms:
        aligned_waveform, _, score = align_to_template(waveform, template_waveform)
        aligned_waveforms.append(aligned_waveform)
        correlation_scores.append(score)

    # Filter out waveforms with low correlation scores
    good_waveforms = filter_waveforms(aligned_waveforms, correlation_scores, quality_threshold)

    return good_waveforms

def spike_align(peak_waveforms, pre=0.6, post=1.0):
    ''' 
    Align spike waveforms from a single channel based on the maximum dV/dt point. 
    
    Args:
        spike_waveforms (List): Extracted spike waveforms.
        pre: length of the window before the maximum dV/dt.
        post: length of the window after the maximum dV/dt.  
        
    Returns:
        waveforms_aligned (list): aligned spike waveforms based on the maximum dV/dt. 
    '''
    window_pre = int(pre * 10000)  # window length before max dV/dt
    window_post = int(post * 10000)  # window length after max dV/dt
    total_length = window_pre + window_post #Total length of the waveform

    waveforms_aligned = []
    
    for i in range(len(peak_waveforms)):        
        data = peak_waveforms[i]
        # Compute the slopes of each waveform
        slopes = np.diff(data)
    
        # Find the index of the minimum slope (maximum in negative)
        idx_min_slope = np.argmin(slopes)
    
        # Define the exact starting and ending indices based on the detected slope peak
        start_idx = idx_min_slope - window_pre + 1
        end_idx = idx_min_slope + window_post + 1

        if start_idx >= 0 and end_idx <= len(data):
            new_waveform = data[start_idx:end_idx]
            waveforms_aligned.append(new_waveform)
    
    return waveforms_aligned

def denoise_signal(target_sig, wavelet='db6', level=3, factor=5):
    '''
    Denoise waveform using PyWavelets
    
    Args: 
        target_sig (list): meaned and pre-processed waveform from a single channel 
        wavelet (str, optional): type of wavelet to use. Default is 'db6'.
        level (int, optional): level of decomposition. Default is 3.
        factor (int, optional): factor to adjust the threshold for denoising. Default is 5.
    
    Returns:
        denoised (list): denoised spike waveform. 
    '''
    # Wavelet transform
    coeffs = pywt.wavedec(target_sig, wavelet, level=level)  
    
    # Noise estimation via the median of the absolute values of the wavelet coefficients
    sigma_est = (1 / 0.6745) * np.median(np.abs(coeffs[-1]))
    threshold = factor * sigma_est * np.sqrt(2 * np.log(len(target_sig)))
    
    # Apply the threshold to all the detail coefficients
    coeffs[1:] = [pywt.threshold(i, value=threshold, mode='soft') for i in coeffs[1:]]
    
    # Inverse wavelet transform
    denoised = pywt.waverec(coeffs, wavelet)

    return denoised


def downsample_signal(target_sig):
    #Downsampling using median

    '''
    Downsampling signal 
    whole signal to 22 points
    0.1 s narrow signal to 40 points
    
    Arg:
        target_sig (list): meaned and pre-processed waveform from a single channel
    
    Return: 
        downsampled_signal (list): list of values (amplitude) of the downsampled points (from zoomed-out binning)
        downsampled_signal_narrow (list): list of values (amplitude) of the downsampled points (from zoomed-in binning) 
    '''
    
    sig_narrow = target_sig[3500:4500] # narrow 0.1 s waveform near max dv/dt (change values based on peak alignment)
    
    num_bins = 21
    bin_length = len(target_sig) // num_bins
    
    num_bins_narrow = 39
    bin_length_narrow = len(sig_narrow) // num_bins_narrow 
    
    # Downsampling (used median)
    downsampled_signal = [np.median(target_sig[i:i+bin_length]) for i in range(0, len(target_sig), bin_length)]
    downsampled_signal_narrow = [np.median(sig_narrow[i:i+bin_length_narrow]) for i in range(0, len(sig_narrow), bin_length_narrow)]
    downsampled = downsampled_signal + downsampled_signal_narrow

    return downsampled
    

def normalize_amplitude(target_sig):
    '''
    downsample target signal and combine zoomed-out and zoomed-in binning 
    normalize the amplitude
    
    Arg:
        target_sig (list): meaned and pre-processed waveform from a single channel
    Return:
        ephy_org (list): list of 62 downsampled points without normalized amplitude
        ephy_normalized (list): list of 62 downsampled points with normalized amplitude 
     
    '''
    
    ephy_org = downsample_signal(target_sig)
    
    min_val = min(ephy_org)
    max_val = max(ephy_org)

    ephy_normalized = [(x - min_val) / (max_val - min_val) for x in ephy_org]
    
    return (ephy_org, ephy_normalized)


def normalize_or_scale_signal(target_sig):
    '''
    normalized signal before downsampling
    
    Arg:
        target_sig (list): meaned and denoised waveform from a single channel
    Return:
        ephy_normalized (list): normalized signal
        ephy_standardized (list): standardized signal
    '''
    # Normalize target signal from a single channel
    min_val = min(target_sig)
    max_val = max(target_sig)

    ephy_normalized = [(x - min_val) / (max_val - min_val) for x in target_sig]
    
    # Standardize target signal from a single channel
    mean_val = np.mean(target_sig)
    std_val = np.std(target_sig)
    ephy_standardized = (target_sig - mean_val) / std_val
    
    return (ephy_normalized, ephy_standardized)
    
    