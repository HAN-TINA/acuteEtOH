import os
import re
import numpy as np
import pandas as pd
import mne
from mne_connectivity import spectral_connectivity_time
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from joblib import Parallel, delayed
from scipy.signal import welch, savgol_filter, decimate
from scipy.ndimage import gaussian_filter1d
from scipy.stats import ttest_ind
from utils import clean_mne_raw, normalize_coherence_to_baseline, normalize_to_baseline

###############################################
# Custom Preprocessing Functions
###############################################
from scipy.signal import butter, filtfilt, iirnotch

def bandpass_filter(signal, lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal

def notch_filter(signal, freq, fs, Q=30):
    nyquist = 0.5 * fs
    w0 = freq / nyquist
    b, a = iirnotch(w0, Q)
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal

def preprocess_data(signal, fs, lowcut=1, highcut=200, notch_freqs=[60, 1.5, 1, 1.75], decimate_factor=1):
    """
    Preprocess the signal:
      1. Apply a notch filter for each frequency in notch_freqs.
      2. Apply a bandpass filter.
      3. Optionally decimate the signal.
    """
    for nf in notch_freqs:
        signal = notch_filter(signal, freq=nf, fs=fs)
    
    signal = bandpass_filter(signal, lowcut=lowcut, highcut=highcut, fs=fs)
    
    if decimate_factor > 1:
        signal = decimate(signal, decimate_factor)
        fs = fs / decimate_factor
    
    return signal, fs

###############################################
# Directories and Basic Setup
###############################################
path = r'C:\Users\melonteam\OneDrive - wesleyan.edu (1)\Shared Documents - Melón Lab\Experiments\EXP240904'
data_dir = os.path.join(path, "iEEG_data")
output_dir = os.path.join(path, "results")
os.makedirs(output_dir, exist_ok=True)

# Define processing parameters
event_dict = {
    "Mouse In": 1,
    "EtOH Injected": 2,
    "Saline Injected": 2,
    "Juvenile In": 3
}
channels = ['EEG VTAA-B', 'EEG PFCA-B']
fmin, fmax = 1, 100    # For our custom bandpass filtering
sfreq = 500
freqs_conn = np.linspace(1, 100, 100)

# --- Cache paths ---
cached_epochs_path = os.path.join(output_dir, "cached_epochs.pkl")
cached_psd_path = os.path.join(output_dir, "cached_psd.pkl")
cached_conn_path = os.path.join(output_dir, "cached_conn.pkl")

###############################################
# Step 1: Preload All Data and Preprocess Using Custom Functions
###############################################
all_epochs = {}

if os.path.exists(cached_epochs_path):
    print("Loading cached epochs from disk...")
    all_epochs = joblib.load(cached_epochs_path)
    print("Loaded.")
else:
    edf_files = [f for f in os.listdir(data_dir) if f.endswith(".edf")]
    for edf_file in edf_files:
        match = re.search(r'_(M|F)_(EtOH|Saline)', edf_file)
        if not match:
            print(f"Could not parse group info from {edf_file}")
            continue
        sex, treatment = match.groups()
        group_label = f"{sex}_{treatment}"

        file_path = os.path.join(data_dir, edf_file)
        print(f"Reading {edf_file}...")
        
        # Read the data from EDF
        raw = mne.io.read_raw_edf(file_path, preload=True)
        
        # After processing each channel:
        for ch in channels:
            if ch in raw.ch_names:
                idx = raw.ch_names.index(ch)
                channel_data = raw.get_data(picks=[ch])[0]
                processed_data, new_fs = preprocess_data(channel_data, fs=sfreq, lowcut=fmin, highcut=fmax,
                                                        notch_freqs=[60, 1.5, 1.75, 1], decimate_factor=1)
                raw._data[idx, :] = processed_data

        # Only update sfreq if decimation was applied.
        if new_fs != sfreq:
            raw.resample(new_fs, method='polyphase')

        
        # Clean the raw data using your existing function.
        clean_mne_raw(raw, channels, window=1000, threshold=4.0)

        # Extract events and epoch the data using the event dictionary.
        events, _ = mne.events_from_annotations(raw, event_id=event_dict)
        epochs = mne.Epochs(raw, events, event_id=event_dict, tmin=0, tmax=500, baseline=None, preload=True)
        all_epochs[(edf_file, group_label)] = epochs

    joblib.dump(all_epochs, cached_epochs_path)
    print("Cached all epochs to disk.")

###############################################
# Step 2: Process PSD and Connectivity Using Welch
###############################################
if os.path.exists(cached_psd_path) and os.path.exists(cached_conn_path):
    print("Loading cached PSD and connectivity data...")
    psd_group_data = joblib.load(cached_psd_path)
    conn_group_data = joblib.load(cached_conn_path)
    print("Cached PSD and connectivity loaded.")
else:
    psd_group_data = {}
    conn_group_data = {}
    # Map raw annotation labels to unified condition names.
    condition_map = {
        "Mouse In": "Baseline",
        "EtOH Injected": "Injected",
        "Saline Injected": "Injected",
        "Juvenile In": "Natural Reward"
    }
    nperseg = 1024  # Parameter for Welch's method

    for (edf_file, group_label), epochs in all_epochs.items():
        for raw_label, unified_label in condition_map.items():
            if raw_label not in epochs.event_id:
                continue
            epochs_condition = epochs[raw_label]
            if len(epochs_condition) == 0:
                continue

            # --- PSD Processing Using Welch's Method ---
            # Get epoch data: shape (n_epochs, n_channels, n_times)
            data_epochs = epochs_condition.get_data(picks=channels)
            print("go")
            # Compute Welch PSD for each epoch and each channel in parallel.
            results = Parallel(n_jobs=-1)(
                delayed(lambda epoch: np.array([welch(epoch[c], fs=sfreq, nperseg=nperseg, noverlap=nperseg//2)[1]
                                                 for c in range(epoch.shape[0])]))(epoch)
                for epoch in data_epochs
            )
            # results is a list of arrays of shape (n_channels, n_freqs); stack them: shape (n_epochs, n_channels, n_freqs)
            psds = np.array(results)
            # Use frequencies from the first channel of the first epoch (assumed constant)
            freqs = welch(data_epochs[0][0], fs=sfreq, nperseg=nperseg, noverlap=nperseg//2)[0]

            # Average across epochs to obtain a representative PSD per channel.
            psds_mean = np.mean(psds, axis=0)  # shape: (n_channels, n_freqs)

            # Smooth the PSD using Savitzky-Golay and Gaussian filters.
            smoothed_psds = savgol_filter(psds_mean, window_length=11, polyorder=3, axis=1)
            smoothed_psds = gaussian_filter1d(smoothed_psds, sigma=3, axis=1)

            # Create a DataFrame for the PSD and append to group data.
            df = pd.DataFrame(psds_mean.T, columns=channels)
            df['Frequency'] = freqs
            df['Mouse'] = edf_file
            df['Group'] = group_label
            df['Condition'] = unified_label
            df = df.melt(id_vars=['Frequency', 'Mouse', 'Group', 'Condition'],
                         var_name='Channel', value_name='Power')
            psd_group_data.setdefault(group_label, []).append(df)

            # --- Connectivity Processing (unchanged) ---
            data = epochs_condition.get_data(picks=channels)
            if data.shape[0] == 0:
                continue
            con = spectral_connectivity_time(
                data=data,
                sfreq=sfreq,
                freqs=freqs_conn,
                fmin=1,
                fmax=100,
                mode="multitaper",
                indices=(np.array([0]), np.array([1])),
                verbose=False,
            )
            con_data = con.get_data(output="dense")  # shape: (n_epochs, 2, 2, n_freqs)
            conn_group_data.setdefault((group_label, unified_label), []).append(con_data)
    
    # Normalize connectivity so that baseline values equal 1.
    #conn_group_data = normalize_coherence_to_baseline(conn_group_data, baseline_condition="Baseline")
    
    # Cache processed PSD and connectivity results.
    joblib.dump(psd_group_data, cached_psd_path)
    joblib.dump(conn_group_data, cached_conn_path)
    print("Processed and cached PSD and connectivity data.")

# ###############################################
# # Step 3: Plot PSD by Group using seaborn (Line plots)
# ###############################################
# # Combine all PSD DataFrames into one
# df_psd_all = pd.concat([pd.concat(dfs) for dfs in psd_group_data.values()], ignore_index=True)
# df_plot = df_psd_all.query(
#     "1 <= Frequency <= 10 and Group in ['M_EtOH', 'F_EtOH']"
# )

# g = sns.relplot(
#     data=df_plot, x='Frequency', y='Power',
#     hue='Group', col='Condition', row=None,
#     kind="line", estimator="mean", errorbar="se",          # shaded SEM like SAKE
#     height=2.5, aspect=6/4
# )

# g.set_axis_labels("Frequency (Hz)", "Power (a.u.)")
# g.fig.suptitle("Welch PSD (custom‑filtered data)", y=1.04)
# plt.show()


# # For each channel, filter the data to a desired frequency range (e.g., 2-5 Hz) and specified groups.
# for ch in channels:
#     df_plot = df_psd_all[(df_psd_all['Frequency'] >= 1) & (df_psd_all['Frequency'] <= 10) &
#                          (df_psd_all['Channel'] == ch) & (df_psd_all['Group'].isin(["F_EtOH", "M_EtOH"]))]
    
#     # Create a FacetGrid: separate panels by Condition and different hues for Groups.



#     g = sns.FacetGrid(df_plot, col="Condition", hue="Group", sharey=True, height=6, aspect=1.2,
#                       hue_order=["M_EtOH", "F_EtOH"])
#     g.map(sns.lineplot, "Frequency", "Power")
#     g.add_legend(title="Group")
#     g.set_axis_labels("Frequency (Hz)", "Power (a.u.)")
#     g.fig.suptitle(f"Normalized PSD Comparison - {ch}", y=1.05)
#     plt.savefig(os.path.join(output_dir, f"sake_PSD_Comparison_{ch}.png"))
#     plt.close()

# print("Batch processing complete! Smoothed PSD and connectivity results saved.")

# ###############################################
# # Step 4: Compare Female vs Male EtOH (Baseline Normalized) with Statistics 
# ###############################################
plt.figure(figsize=(10, 6))
male_group = 'M_EtOH'
female_group = 'F_EtOH'
if male_group in psd_group_data and female_group in psd_group_data:
    # Normalize EtOH conditions to their Mouse In baseline
    baseline_norm = normalize_to_baseline(psd_group_data, condition='Injected', baseline='Baseline')
    df_m = baseline_norm.get(male_group, pd.DataFrame())
    if df_m.empty or 'Frequency' not in df_m.columns:
        print("No normalized data available for", male_group)
    else:
        df_m = df_m.reset_index(drop=True)
        df_m = df_m[(df_m['Frequency'] >= 2) & (df_m['Frequency'] <= 5) & (df_m['Channel'] == 'EEG VTAA-B')]  # Focus on the VTA channel (EEG VTAA-B) and frequency range 0.5-10 Hz
    df_f = baseline_norm.get(female_group, pd.DataFrame())
    if df_f.empty or 'Frequency' not in df_f.columns:
        print("No normalized data available for", female_group)
    else:
        df_f = df_f[(df_f['Frequency'] >= 2) & (df_f['Frequency'] <= 5) & (df_f['Channel'] == 'EEG VTAA-B')]

    # Compute subject-level means (per mouse) for each frequency
    mouse_dfs_m = [d.groupby('Frequency')['Power'].mean() for _, d in df_m.groupby('Mouse')]
    mouse_dfs_f = [d.groupby('Frequency')['Power'].mean() for _, d in df_f.groupby('Mouse')]
    df_mouse_m = pd.DataFrame(mouse_dfs_m)
    df_mouse_f = pd.DataFrame(mouse_dfs_f)
    mean_m = df_mouse_m.mean(axis=0)
    std_m = df_mouse_m.std(axis=0)/np.sqrt(df_mouse_m.shape[0])
    mean_f = df_mouse_f.mean(axis=0)
    std_f = df_mouse_f.std(axis=0)/np.sqrt(df_mouse_f.shape[0])
    plt.plot(mean_m.index, mean_m.values, label='M_EtOH')
    plt.fill_between(mean_m.index, mean_m - std_m, mean_m + std_m, alpha=0.2)
    plt.plot(mean_f.index, mean_f.values, label='F_EtOH')
    plt.fill_between(mean_f.index, mean_f - std_f, mean_f + std_f, alpha=0.2)
    # # T-test per frequency (using subject-level data)
    # sig_freqs = []
    # for freq in mean_m.index:
    #     if freq in df_mouse_m.columns and freq in df_mouse_f.columns:
    #         vals_m = df_mouse_m[freq].dropna()
    #         vals_f = df_mouse_f[freq].dropna()
    #         if len(vals_m) > 1 and len(vals_f) > 1:
    #             stat, pval = ttest_ind(vals_m, vals_f)
    #             if pval < 0.05:
    #                 sig_freqs.append(freq)
    # if sig_freqs:
    #     ylim = plt.ylim()
    #     y = ylim[1] * 0.95
    #     plt.scatter(sig_freqs, [y] * len(sig_freqs), marker='*', color='black', label='p < 0.05')
plt.axhline(1.0, linestyle='--', color='gray', label='Baseline')
plt.title("EtOH Injection VTA Power (Normalized to Baseline) with Stats")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Normalized Power")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, "EtOH_VTA_Sex_Comp_with_stats.png"))
plt.close()

# -------------
# Step 5: Plot Connectivity by Group (Line Plots)
# -------------
# conditions = ["Baseline", "Injected", "Natural Reward"]
# desired_order = ["M_EtOH", "F_EtOH", "M_Saline", "F_Saline"]
# for condition in conditions:
#     plt.figure(figsize=(10, 6))
#     for group_label in desired_order:
#         con_list = conn_group_data[(group_label, condition)]
#         con_all = np.concatenate(con_list, axis=0)
#         con_mean = np.mean(con_all, axis=0)
#         # Plot the normalized connectivity for channel pair VTA-PFC (assumed at index [0,1])
#         plt.plot(freqs_conn, con_mean[0, 1, :], label=group_label)
#     plt.title(f"VTA-PFC Normalized Connectivity - {condition}")
#     plt.xlabel("Frequency (Hz)")
#     plt.ylabel("Normalized Connectivity (Ratio to Baseline)")
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.savefig(os.path.join(output_dir, f"Coherence_Comparison_{condition}.png"))
#     plt.close()
# print("Connectivity line plots saved.")

# Build a DataFrame from your normalized connectivity data.
# rows = []
# # conn_group_data is keyed by (group_label, condition); each value is a list of arrays.
# # Each connectivity array has shape: [n_epochs, 2, 2, n_freqs] (for VTA-PFC, we use index 0,1).
# conn_group_data = normalize_coherence_to_baseline(conn_group_data, baseline_condition="Baseline")
# for (group_label, condition), arr_list in conn_group_data.items():
#     for arr in arr_list:
#         n_epochs = arr.shape[0]
#         for epoch in range(n_epochs):
#             # Extract the connectivity values for the VTA-PFC pair
#             conn_vals = arr[epoch, 0, 1, :]  # shape: (n_freqs,)
#             # Loop over frequencies
#             for i, freq in enumerate(freqs_conn):
#                 rows.append({
#                     "Group": group_label,
#                     "Condition": condition,
#                     "Frequency": freq,
#                     "Connectivity": conn_vals[i]
#                 })

# df_conn = pd.DataFrame(rows)
# # Check the DataFrame structure.
# print(df_conn.head())
# print(df_conn.columns)

# sns.set(style="whitegrid")
# g = sns.FacetGrid(df_conn, col="Condition", hue="Group", height=6, aspect=1.2, hue_order=["M_EtOH","F_EtOH","M_Saline","F_Saline"])
# g.map(sns.lineplot, "Frequency", "Connectivity")
# g.add_legend(title="Group")
# g.set_axis_labels("Frequency (Hz)", "Normalized Connectivity")
# g.fig.suptitle("VTA-PFC Connectivity by Frequency", y=1.05)
# plt.savefig(os.path.join(output_dir, "Normalized_Coherence_Comparison_by_Condition.png"))
# plt.close()


# -------------
# Step 6: Box Plot Connectivity by Frequency Bands
# -------------
freq_bands = { "2-5 Hz": (2, 5), "6-12 Hz": (6, 12), "15-30 Hz": (15, 30), "40-70 Hz": (40, 70), "70-100 Hz": (70, 100) }
boxplot_rows = []
for (group_label, condition), con_list in conn_group_data.items():
    for con_data in con_list:
        n_epochs = con_data.shape[0]
        for epoch in range(n_epochs):
            # Extract connectivity values for the VTA-PFC pair (assumed stored at [epoch, 0, 1, :])
            conn_vals = con_data[epoch, 0, 1, :]  # array of shape (n_freqs,)
            for band_name, (f_low, f_high) in freq_bands.items():
                band_indices = np.where((freqs_conn >= f_low) & (freqs_conn <= f_high))[0]
                if len(band_indices) > 0:
                    band_mean = np.mean(conn_vals[band_indices])
                    boxplot_rows.append({
                        "Group": group_label,
                        "Condition": condition,
                        "Band": band_name,
                        "Coherence": band_mean
                    })

conn_box_df = pd.DataFrame(boxplot_rows)
print(conn_box_df.columns)

sns.set(style="whitegrid")
g = sns.catplot(data=conn_box_df, x="Band", y="Coherence", col="Group", hue="Condition", kind="box", 
                height=5, aspect=1)
g.fig.subplots_adjust(top=0.85)
g.fig.suptitle("VTA-PFC Connectivity by Frequency Band", fontsize=16)
plt.savefig(os.path.join(output_dir, "Connectivity_Boxplot_by_Frequency_Band.png"))
plt.close()

print("All processing complete: PSD, connectivity line plots, and connectivity box plots saved.")
