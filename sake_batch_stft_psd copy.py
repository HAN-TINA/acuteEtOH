import os, re, joblib
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
import mne
from joblib import Parallel, delayed
from scipy.signal import butter, filtfilt, iirnotch, decimate, stft as scipy_stft, savgol_filter  # Import Savitzky-Golay filter
from scipy.ndimage import gaussian_filter1d
from utils import clean_mne_raw            # <- your own helper
# ------------------------------------------------------------------
# 1.  custom filters ------------------------------------------------
def bandpass_filter(sig, low, high, fs, order=5):
    nyq = 0.5 * fs
    b, a = butter(order, [low/nyq, high/nyq], btype="band")
    return filtfilt(b, a, sig)

def notch_filter(sig, f0, fs, Q=30):
    nyq = 0.5 * fs
    b, a = iirnotch(f0/nyq, Q)
    return filtfilt(b, a, sig)

def preprocess(sig, fs, low=1, high=100,
               notch=[60, 1.5, 1.75, 1], dec=1):
    for f0 in notch:
        sig = notch_filter(sig, f0, fs)
    sig = bandpass_filter(sig, low, high, fs)
    if dec > 1:
        sig = decimate(sig, dec, ftype="fir")
        fs /= dec
    return sig, fs
# ------------------------------------------------------------------
# 2.  SAKE‑style STFT helper ---------------------------------------
class Stft:
    """Minimal SAKE transformer: power matrix + mains‑interpolation."""
    def __init__(self, fs, win_s=1.0, band=(1, 100),
                 overlap=0.5, mains=(55, 65)):
        self.fs  = fs
        self.nw  = int(fs*win_s)
        self.no  = int(self.nw*overlap)
        self.fmin, self.fmax = band
        self.keep = slice(int(self.fmin*self.nw/fs),
                          int(self.fmax*self.nw/fs)+1)
        self.freq = np.arange(self.fmin, self.fmax+1/win_s, 1/win_s)
        self.m0 = int(mains[0]*self.nw/fs)
        self.m1 = int(mains[1]*self.nw/fs)+1

    def power(self, x):
        _, _, Z = scipy_stft(x, self.fs, nperseg=self.nw,
                             noverlap=self.no, padded=False)
        p = np.abs(Z[self.keep, :])**2            # freq × time
        # blank mains rows & interpolate in freq‑axis
        p[self.m0:self.m1, :] = np.nan
        p = pd.DataFrame(p).interpolate("nearest", axis=0).values
        return p                                   # freq × time
# ------------------------------------------------------------------
# 3.  paths & constants --------------------------------------------
root   = r"C:\Users\melonteam\OneDrive - wesleyan.edu (1)\Shared Documents - Melón Lab\Experiments\EXP240904"
data_d = os.path.join(root, r"iEEG_data\NR")
out_d  = os.path.join(root, "results"); os.makedirs(out_d, exist_ok=True)

sfreq  = 500
channels = ['EEG VTAA-B', 'EEG PFCA-B']
event_id = {"Mouse In":1, "Juvenile In":2}

cache_epochs = os.path.join(out_d, "j_epochs_stft.pkl")
cache_psd    = os.path.join(out_d, "j_psd_stft.pkl")
# ------------------------------------------------------------------
# 4.  load / preprocess / epoch ------------------------------------
if os.path.exists(cache_epochs):
    all_epochs = joblib.load(cache_epochs)
else:
    all_epochs = {}
    for fname in [f for f in os.listdir(data_d) if f.endswith(".edf")]:
        m = re.search(r"_(M|F)", fname)
        if not m: continue
        sex = m.group(1)          # << grab the string, not the tuple
        grp = sex 
        raw = mne.io.read_raw_edf(os.path.join(data_d, fname), preload=True)
        scale = float(raw.info['chs'][0]['unit_mul'] or 1)   # Sirenia puts µV scale here
        raw._data *= scale 
        for ch in channels:
            idx = raw.ch_names.index(ch)
            data, _ = preprocess(raw[ch][0][0], fs=sfreq,
                                 low=1, high=100,
                                 notch=[60,1.5,1.75,1,2], dec=1)
            raw._data[idx] = data

        #clean_mne_raw(raw, channels, window=1000, threshold=4.0)
        ev, _ = mne.events_from_annotations(raw, event_id=event_id)
        ep = mne.Epochs(raw, ev, event_id, tmin=0, tmax=500,
                        baseline=None, preload=True)
        all_epochs[(fname, grp)] = ep
    joblib.dump(all_epochs, cache_epochs)
# ------------------------------------------------------------------
# 5.  PSD via STFT --------------------------------------------------
if os.path.exists(cache_psd):
    psd_group = joblib.load(cache_psd)
else:
    stft_engine = Stft(fs=sfreq, win_s=1.0, band=(1,100),
                       overlap=0.5, mains=(55,65))
    cond_map = {"Mouse In":"Baseline",
                "Juvenile In":"Natural Reward"}
    psd_group = {}
    for (fname, grp), ep in all_epochs.items():
        for raw_lab, cond in cond_map.items():
            if raw_lab not in ep.event_id: continue
            e = ep[raw_lab]
            if len(e)==0: continue
            # epoch → (n_epochs, n_ch, n_times)
            def epoch_psd(epoch):
                return np.array([ stft_engine.power(epoch[c]).mean(axis=1)
                                   for c in range(len(channels)) ])
            psd_epochs = Parallel(-1)(delayed(epoch_psd)(x) for x in e.get_data(picks=channels))
            psd_mean = np.mean(psd_epochs, axis=0)            # ch × freq
           # psd_mean /= psd_mean.mean(axis=1, keepdims=True)  # normalise
            smoothed_psds = savgol_filter(psd_mean, window_length=11,
                              polyorder=3, axis=1)
            smoothed_psds = gaussian_filter1d(smoothed_psds, sigma=3, axis=1)
            df = pd.DataFrame(smoothed_psds.T, columns=channels)
            df["Frequency"] = stft_engine.freq
            df["Mouse"]     = fname
            df["Group"]     = grp
            df["Condition"] = cond
            df = df.melt(id_vars=["Frequency","Mouse","Group","Condition"],
                         var_name="Channel", value_name="Power")
            psd_group.setdefault(grp, []).append(df)
    joblib.dump(psd_group, cache_psd)
# ------------------------------------------------------------------
# 6.  Plot 1‑10 Hz for M/F EtOH ------------------------------------
df_psd = pd.concat([pd.concat(v) for v in psd_group.values()],
                   ignore_index=True)
plot_df = df_psd.query("1<=Frequency<=10 and Group in ['M','F']")

sns.set(style="whitegrid")
g = sns.relplot(data=plot_df, x="Frequency", y="Power",
                hue="Group", col="Condition",
                kind="line", estimator="mean", errorbar="se",
                hue_order=["M","F"])
g.set_axis_labels("Frequency (Hz)", "Power (μV² Hz⁻¹)")
g.fig.suptitle("SAKE‑style STFT PSD (1–10 Hz) — Male vs Female Juvenile",
               y=1.04)
plt.tight_layout()
plt.savefig(os.path.join(out_d, "SAKE_graph_Juvenile.png"))
plt.close()

import statsmodels.api as sm
from statsmodels.formula.api import ols
import pingouin as pg

# 2. Preprocess: average across 1–4 Hz
df_band = df_psd.query("1 <= Frequency <= 4")  # bandpass select
df_mean = df_band.groupby(["Mouse", "Group", "Condition", "Channel"]).agg({'Power':'mean'}).reset_index()

# 3. Filter for VTA channel only
df_vta = df_mean.query("Channel == 'EEG VTAA-B'")

# 4. Pivot to wide format for RM ANOVA
pivot = df_vta.pivot_table(index=["Mouse","Group"],
                           columns="Condition",
                           values="Power").reset_index()

# 5. Run repeated measures ANOVA
anova = pg.mixed_anova(dv="Power", within="Condition", between="Group", subject="Mouse", data=df_vta)
print(anova)
