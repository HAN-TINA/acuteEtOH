import numpy as np
import pandas as pd
from mne.time_frequency import psd_array_multitaper
from scipy.signal import butter, filtfilt, iirnotch, decimate

def bandpass_filter(signal, lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

def notch_filter(signal, freq, fs, Q=30):
    nyquist = 0.5 * fs
    w0 = freq / nyquist
    b, a = iirnotch(w0, Q)
    return filtfilt(b, a, signal)

def preprocess_data(signal, fs, lowcut=1, highcut=200, notch_freqs=[60, 1.5], decimate_factor=1):
    """
    Preprocess the signal:
      1. Apply a notch filter for each frequency in notch_freqs.
      2. Apply a bandpass filter between lowcut and highcut.
      3. Optionally decimate the signal.
    """
    for nf in notch_freqs:
        signal = notch_filter(signal, freq=nf, fs=fs)
    
    signal = bandpass_filter(signal, lowcut=lowcut, highcut=highcut, fs=fs)
    
    if decimate_factor > 1:
        signal = decimate(signal, decimate_factor)
        fs = fs / decimate_factor
        
    return signal, fs

def preprocess_epoch(epoch, fs, lowcut, highcut):
    """
    Preprocess an epoch array of shape (n_channels, n_times) by applying our preprocessing
    function to each channel.
    """
    n_channels, n_times = epoch.shape
    preprocessed = np.empty_like(epoch)
    for c in range(n_channels):
        preprocessed[c], _ = preprocess_data(epoch[c], fs, lowcut=lowcut, highcut=highcut, notch_freqs=[60, 1.5], decimate_factor=1)
    return preprocessed

# --- Outlier Detection Functions ---
def mad(arr):
    return np.median(np.abs(arr - np.median(arr)))

def rolling_mad_outliers(data: np.ndarray, window: int = 1000, threshold: float = 5.0) -> np.ndarray:
    """Detect outliers using a rolling median + MAD."""
    outliers = np.zeros_like(data, dtype=bool)
    half_win = window // 2
    for i in range(half_win, len(data) - half_win):
        window_data = data[i - half_win:i + half_win]
        med = np.median(window_data)
        mad_val = mad(window_data)
        if mad_val == 0:
            continue
        if np.abs(data[i] - med) > threshold * mad_val:
            outliers[i] = True
    return outliers

def clean_mne_raw_channel(raw, ch_name, window=1000, threshold=5.0):
    """Apply MAD-based outlier detection and interpolate for one channel."""
    ch_idx = raw.ch_names.index(ch_name)
    signal = raw.get_data(picks=[ch_name])[0]
    mask = rolling_mad_outliers(signal, window, threshold)
    signal_clean = pd.Series(signal)
    signal_clean[mask] = np.nan
    signal_clean = signal_clean.interpolate().fillna(method='bfill').fillna(method='ffill')
    raw._data[ch_idx] = signal_clean.values

def clean_mne_raw(raw, channels, window=1000, threshold=5.0):
    for ch in channels:
        clean_mne_raw_channel(raw, ch, window=window, threshold=threshold)

# --- Normalization Functions ---
def normalize_psd_by_mean(psd_group_data):
    normalized_data = {}
    for group, dfs in psd_group_data.items():
        normalized_dfs = []
        for df in dfs:
            df = df.copy()
            df['Power'] = df['Power'] / df['Power'].mean()
            normalized_dfs.append(df)
        normalized_data[group] = normalized_dfs
    return normalized_data

def normalize_to_baseline(psd_group_data, condition='Injected', baseline='Mouse In'):
    """
    Normalize power by dividing the PSD of the specified condition by the baseline PSD,
    matching on Frequency and Channel.
    
    Parameters
    ----------
    psd_group_data : dict of group -> list of DataFrames
    condition : str, condition name to normalize (e.g., 'Injected')
    baseline : str, baseline condition name (e.g., 'Mouse In')
    
    Returns
    -------
    output : dict, with normalized DataFrames per group
    """
    output = {}
    for group, dfs in psd_group_data.items():
        df_all = pd.concat(dfs)
        if not all(c in df_all['Condition'].values for c in [condition, baseline]):
            continue
        # Compute baseline PSD: one value per Frequency and Channel
        base = df_all[df_all['Condition'] == baseline].groupby(['Frequency', 'Channel'])['Power'].mean().reset_index()
        base = base.rename(columns={'Power': 'BaselinePower'})
        # Get the comparison data
        comp = df_all[df_all['Condition'] == condition].copy()
        # Merge baseline power into comp so that each row gets its matching baseline value
        comp = pd.merge(comp, base, on=['Frequency', 'Channel'], how='left')
        # Divide the power by the baseline power (row-wise)
        comp['Power'] = comp['Power'] / comp['BaselinePower']
        # Safely drop the baseline column (ignore if it doesn't exist)
        comp = comp.drop(columns=['BaselinePower'], errors='ignore')
        output[group] = comp
    return output

# --- Simulation for PSD reference (if needed) ---
def simulate_psd_reference(freqs=[4, 8, 30], amps=[5, 3, 2], sfreq=500, duration=10):
    t = np.arange(0, duration, 1 / sfreq)
    signal = np.zeros_like(t)
    for f, a in zip(freqs, amps):
        signal += np.sin(2 * np.pi * f * t) * a

    # Compute PSD using same method in 1-s chunks
    psds = []
    for start in range(0, len(t) - sfreq, sfreq):
        segment = signal[start:start + sfreq]
        psd, freqs_out = psd_array_multitaper(segment[np.newaxis, :], sfreq=sfreq, fmin=1, fmax=100, verbose=False)
        psds.append(psd)
    psd_mean = np.mean(psds, axis=0).squeeze()
    return freqs_out, psd_mean



def butter_bandpass_filter(data, lowcut, highcut, fs, order=2):
    """
    Bandpass a 1D signal using a Butterworth filter.

    Parameters
    ----------
    data : array-like, 1D signal
    lowcut : float, lower frequency cutoff in Hz
    highcut : float, upper frequency cutoff in Hz
    fs : int or float, sampling rate in Hz
    order : int, filter order (default is 2)

    Returns
    -------
    filt_data : array-like, 1D filtered signal
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='bandpass')
    filt_data = signal.filtfilt(b, a, data)
    return filt_data

def normalize_coherence_to_baseline(conn_group_data, baseline_condition="Baseline"):
    """
    For each group, compute the mean baseline connectivity (across all baseline recordings).
    Then, for each condition, normalize connectivity by that group's mean baseline.
    
    Parameters
    ----------
    conn_group_data : dict
        Dictionary keyed by (group_label, condition). Each value is a list of arrays, where
        each array has shape [n_epochs, 2, 2, n_freqs].
    baseline_condition : str
        Name of the condition that should act as baseline (e.g. "Baseline").

    Returns
    -------
    normalized_conn : dict
        Dictionary in the same format as conn_group_data, but with values normalized
        so that the baseline condition is ~1.0 across all frequencies.
    """
    # 1) Compute mean baseline connectivity for each group across all baseline recordings
    baseline_map = {}
    for (group_label, cond_label), arr_list in conn_group_data.items():
        if cond_label == baseline_condition:
            # Combine all baseline arrays for this group, shape -> [total_epochs, 2, 2, n_freqs]
            combined = np.concatenate(arr_list, axis=0)
            # Average across epochs to get a single [2, 2, n_freqs] baseline array
            baseline_mean = np.mean(combined, axis=0)
            baseline_map[group_label] = baseline_mean

    # 2) Normalize each condition by the group's baseline
    normalized_conn = {}
    for (group_label, cond_label), arr_list in conn_group_data.items():
        if group_label in baseline_map:
            base = baseline_map[group_label]  # shape [2, 2, n_freqs]
            new_arrays = []
            for arr in arr_list:
                # arr has shape [n_epochs, 2, 2, n_freqs]
                # Divide each epoch by the baseline array. Broadcasting handles the epoch dimension.
                new_arrays.append(arr / base)
            normalized_conn[(group_label, cond_label)] = new_arrays
        else:
            # If we have no baseline for that group, we cannot normalize
            normalized_conn[(group_label, cond_label)] = arr_list

    return normalized_conn
