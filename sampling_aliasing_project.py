"""
Sampling Theorem, Aliasing and Reconstruction
Signals and Systems project (Unit 6)
Author: Swarit Chauhan

Run:  python sampling_aliasing_project.py
It saves the figures in the folder figs/ and the numbers used in the
report in results.json.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, sosfiltfilt

os.makedirs("figs", exist_ok=True)
results = {}
plt.rcParams.update({"font.size": 8, "axes.grid": True, "grid.alpha": 0.3})


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------
def sinc_reconstruct(samples, ts, t):
    """Ideal reconstruction: x(t) = sum x[n] sinc((t - nTs)/Ts)."""
    Ts = ts[1] - ts[0]
    # np.sinc(x) = sin(pi x)/(pi x), so this is the usual sinc
    return np.array([np.sum(samples * np.sinc((tt - ts) / Ts)) for tt in t])


def zoh_reconstruct(samples, ts, t):
    """Zero-order hold: keep the last sample value."""
    Ts = ts[1] - ts[0]
    idx = np.clip(np.floor((t - ts[0]) / Ts).astype(int), 0, len(ts) - 1)
    return samples[idx]


def linear_reconstruct(samples, ts, t):
    """First-order hold: join the samples by straight lines."""
    return np.interp(t, ts, samples)


def alias_freq(f, fs):
    """Apparent frequency of a cosine of frequency f sampled at fs."""
    return np.abs(f - fs * np.round(f / fs))


def rms(a):
    return float(np.sqrt(np.mean(a ** 2)))


# ---------------------------------------------------------------------
# Experiment 1: sampling above the Nyquist rate and sinc reconstruction
# x(t) = cos(2 pi 4 t) + 0.5 cos(2 pi 9 t)   -> f_max = 9 Hz
# ---------------------------------------------------------------------
def x1(t):
    return np.cos(2 * np.pi * 4 * t) + 0.5 * np.cos(2 * np.pi * 9 * t)


fs1 = 40.0
ts1 = np.arange(-6, 10, 1 / fs1)          # samples on a longer interval
t_dense = np.linspace(0, 2, 1600)         # we look at 0..2 s
xr1 = sinc_reconstruct(x1(ts1), ts1, t_dense)
err1 = x1(t_dense) - xr1

fig, ax = plt.subplots(2, 1, figsize=(6.5, 4.2))
tz1 = t_dense <= 0.6
ax[0].plot(t_dense[tz1], x1(t_dense)[tz1], "b", lw=1, label="original x(t)")
ax[0].plot(t_dense[tz1], xr1[tz1], "r--", lw=1, label="sinc reconstruction")
m = (ts1 >= 0) & (ts1 <= 0.6)
ax[0].stem(ts1[m], x1(ts1[m]), linefmt="g-", markerfmt="go", basefmt=" ",
           label="samples (fs = 40 Hz)")
ax[0].set_ylim(-1.6, 2.3)
ax[0].set_ylabel("x(t)")
ax[0].set_xlabel("t (seconds)")
ax[0].set_title("x(t) = cos(2π·4t) + 0.5 cos(2π·9t), sampled at 40 Hz (first 0.6 s)")
ax[0].legend(loc="upper center", fontsize=6.5, ncol=3)
ax[1].plot(t_dense, err1, "k", lw=0.8)
ax[1].set_ylabel("error")
ax[1].set_xlabel("t (seconds)")
ax[1].set_title("Reconstruction error over 0 to 2 s")
plt.tight_layout()
plt.savefig("figs/fig1_sampling_reconstruction.png", dpi=170)
plt.close()
results["exp1_fmax"] = 9
results["exp1_nyquist_rate"] = 18
results["exp1_fs"] = fs1
results["exp1_max_error"] = float(np.max(np.abs(err1)))
results["exp1_rms_error"] = rms(err1)

# ---------------------------------------------------------------------
# Experiment 2: aliasing in the time domain
# 9 Hz cosine sampled at 12 Hz -> looks like 3 Hz
# ---------------------------------------------------------------------
f_true, fs2 = 9.0, 12.0
f_al = float(alias_freq(f_true, fs2))
n = np.arange(0, 25)
tn = n / fs2
t = np.linspace(0, 2, 2000)
s_true = np.cos(2 * np.pi * f_true * tn)
s_alias = np.cos(2 * np.pi * f_al * tn)

fig, ax = plt.subplots(1, 1, figsize=(6.5, 2.8))
ax.plot(t, np.cos(2 * np.pi * f_true * t), "b", lw=0.9, label="true signal, 9 Hz")
ax.plot(t, np.cos(2 * np.pi * f_al * t), "r--", lw=1.1, label="alias, 3 Hz")
ax.stem(tn, s_true, linefmt="g-", markerfmt="go", basefmt=" ",
        label="samples at fs = 12 Hz")
ax.set_xlim(0, 2)
ax.set_xlabel("t (seconds)")
ax.set_ylabel("amplitude")
ax.set_title("A 9 Hz cosine sampled at 12 Hz looks like a 3 Hz cosine")
ax.set_ylim(-1.7, 1.15)
ax.legend(loc="lower center", fontsize=6.5, ncol=3)
plt.tight_layout()
plt.savefig("figs/fig2_aliasing_time.png", dpi=170)
plt.close()
results["exp2_alias_freq"] = f_al
results["exp2_sample_difference"] = float(np.max(np.abs(s_true - s_alias)))

# ---------------------------------------------------------------------
# Experiment 3: apparent frequency against true frequency (folding)
# ---------------------------------------------------------------------
f = np.linspace(0, 36, 1441)
fig, ax = plt.subplots(1, 2, figsize=(6.5, 2.7))
ax[0].plot(f, alias_freq(f, 12), "b", lw=1.2)
ax[0].axvline(6, color="gray", ls=":")
ax[0].set_xlabel("true frequency f (Hz)")
ax[0].set_ylabel("apparent frequency (Hz)")
ax[0].set_title("fs = 12 Hz (fs/2 = 6 Hz)")
ax[1].plot(f, alias_freq(f, 20), "r", lw=1.2)
ax[1].axvline(10, color="gray", ls=":")
ax[1].set_xlabel("true frequency f (Hz)")
ax[1].set_title("fs = 20 Hz (fs/2 = 10 Hz)")
plt.tight_layout()
plt.savefig("figs/fig3_alias_map.png", dpi=170)
plt.close()
results["exp3_examples"] = {
    "fs12_f9": float(alias_freq(9, 12)),
    "fs12_f15": float(alias_freq(15, 12)),
    "fs12_f21": float(alias_freq(21, 12)),
    "fs20_f14": float(alias_freq(14, 20)),
    "fs20_f26": float(alias_freq(26, 20)),
}

# ---------------------------------------------------------------------
# Experiment 4: spectrum of the sampled signal (copies of the spectrum)
# two tones at 50 Hz and 120 Hz  -> f_max = 120 Hz, Nyquist rate = 240 Hz
# ---------------------------------------------------------------------
fine_fs = 4000
N = 8000
tt = np.arange(N) / fine_fs
xa = np.cos(2 * np.pi * 50 * tt) + 0.6 * np.cos(2 * np.pi * 120 * tt)
freqs = np.fft.rfftfreq(N, 1 / fine_fs)


def sampled_spectrum(M):
    """Keep every M-th point and put zeros in between (impulse train)."""
    xs = np.zeros(N)
    xs[::M] = xa[::M]
    X = np.abs(np.fft.rfft(xs)) / (N / M) * 0.5   # so a cosine of amplitude A -> A/2
    return X


fig, ax = plt.subplots(2, 1, figsize=(6.5, 4.0), sharex=True)
for a, M, title in [(ax[0], 8, "fs = 500 Hz (above 240 Hz): copies do not overlap"),
                    (ax[1], 20, "fs = 200 Hz (below 240 Hz): copies overlap")]:
    X = sampled_spectrum(M)
    a.plot(freqs, X, "b", lw=0.8)
    a.set_xlim(0, 1000)
    a.set_ylabel("|X_s(f)|")
    a.set_title(title)
ax[0].axvline(250, color="r", ls=":")
ax[1].axvline(100, color="r", ls=":")
ax[1].set_xlabel("frequency (Hz)")
plt.tight_layout()
plt.savefig("figs/fig4_sampled_spectrum.png", dpi=170)
plt.close()


def peak_list(X, thr=0.1, fmax=500):
    out = []
    for k in range(1, len(X) - 1):
        if X[k] > thr and X[k] >= X[k - 1] and X[k] >= X[k + 1] and freqs[k] <= fmax:
            out.append(float(freqs[k]))
    return out


results["exp4_peaks_fs500"] = peak_list(sampled_spectrum(8), fmax=1000)
results["exp4_peaks_fs200"] = peak_list(sampled_spectrum(20), fmax=100)

# ---------------------------------------------------------------------
# Experiment 5: sinc, zero-order hold and linear reconstruction
# x(t) = cos(2 pi 2 t) + 0.6 sin(2 pi 5 t) + 0.3 cos(2 pi 8 t); f_max = 8 Hz
# ---------------------------------------------------------------------
def x5(t):
    return (np.cos(2 * np.pi * 2 * t) + 0.6 * np.sin(2 * np.pi * 5 * t)
            + 0.3 * np.cos(2 * np.pi * 8 * t))


def recon_errors(fs, t_eval):
    ts = np.arange(-8, 12, 1 / fs)
    s = x5(ts)
    out = {}
    out["sinc"] = sinc_reconstruct(s, ts, t_eval)
    out["zoh"] = zoh_reconstruct(s, ts, t_eval)
    out["linear"] = linear_reconstruct(s, ts, t_eval)
    return ts, s, out


t_eval = np.linspace(0.5, 3.5, 3000)
fs5 = 40.0
ts5, s5, rec5 = recon_errors(fs5, t_eval)
errs5 = {k: rms(x5(t_eval) - v) for k, v in rec5.items()}
results["exp5_fs"] = fs5
results["exp5_rms"] = errs5

tz = np.linspace(0.5, 1.5, 1000)
_, _, recz = recon_errors(fs5, tz)
fig, ax = plt.subplots(2, 1, figsize=(6.5, 4.3), sharex=True)
ax[0].plot(tz, x5(tz), "k", lw=1.2, label="original")
ax[0].plot(tz, recz["zoh"], "b", lw=0.9, label="zero-order hold")
ax[0].plot(tz, recz["linear"], "g", lw=0.9, label="linear")
ax[0].plot(tz, recz["sinc"], "r--", lw=1, label="sinc (ideal)")
mm = (ts5 >= 0.5) & (ts5 <= 1.5)
ax[0].plot(ts5[mm], s5[mm], "ko", ms=3)
ax[0].set_title("Three ways to rebuild the signal from samples (fs = 40 Hz)")
ax[0].set_ylim(-1.5, 2.1)
ax[0].legend(fontsize=6.5, ncol=4, loc="upper center")
ax[1].plot(tz, x5(tz) - recz["zoh"], "b", lw=0.8, label="zero-order hold")
ax[1].plot(tz, x5(tz) - recz["linear"], "g", lw=0.8, label="linear")
ax[1].plot(tz, x5(tz) - recz["sinc"], "r", lw=0.8, label="sinc")
ax[1].set_title("Reconstruction error")
ax[1].set_xlabel("t (seconds)")
ax[1].legend(fontsize=6.5, ncol=3, loc="upper right")
plt.tight_layout()
plt.savefig("figs/fig5_reconstruction_methods.png", dpi=170)
plt.close()

# ---------------------------------------------------------------------
# Experiment 6: reconstruction error against the sampling rate
# ---------------------------------------------------------------------
fs_list = [10, 12, 14, 16, 18, 20, 25, 30, 40, 60, 80, 100]
t_eval6 = np.linspace(0.5, 3.5, 1500)
E = {"sinc": [], "zoh": [], "linear": []}
for fs in fs_list:
    _, _, r = recon_errors(float(fs), t_eval6)
    for k in E:
        E[k].append(rms(x5(t_eval6) - r[k]))

fig, ax = plt.subplots(1, 1, figsize=(6.0, 3.1))
ax.semilogy(fs_list, E["zoh"], "s-", color="b", ms=3.5, label="zero-order hold")
ax.semilogy(fs_list, E["linear"], "^-", color="g", ms=3.5, label="linear")
ax.semilogy(fs_list, E["sinc"], "o-", color="r", ms=3.5, label="sinc (ideal)")
ax.axvline(16, color="gray", ls=":")
ax.text(16.6, ax.get_ylim()[0] * 1.6, "Nyquist rate 16 Hz", fontsize=6.5)
ax.set_xlabel("sampling rate fs (Hz)")
ax.set_ylabel("RMS error")
ax.set_title("Reconstruction error against sampling rate")
ax.legend(fontsize=6.5)
plt.tight_layout()
plt.savefig("figs/fig6_error_vs_fs.png", dpi=170)
plt.close()
results["exp6_fs"] = fs_list
results["exp6_errors"] = E

# ---------------------------------------------------------------------
# Experiment 7: anti-aliasing filter
# x(t) = sin(2 pi 3 t) + sin(2 pi 14 t)   sampled at fs = 20 Hz
# The 14 Hz part is above fs/2 = 10 Hz and folds to 6 Hz.
# ---------------------------------------------------------------------
fine = 2000
Tfin = 4.0
tf = np.arange(0, Tfin, 1 / fine)
xin = np.sin(2 * np.pi * 3 * tf) + np.sin(2 * np.pi * 14 * tf)
sos = butter(6, 8, btype="low", fs=fine, output="sos")
xfilt = sosfiltfilt(sos, xin)
fs7 = 20
step = fine // fs7
xs_no = xin[::step]
xs_yes = xfilt[::step]


def spec(xs):
    X = np.abs(np.fft.rfft(xs)) / len(xs) * 2
    fr = np.fft.rfftfreq(len(xs), 1 / fs7)
    return fr, X


fr, X_no = spec(xs_no)
_, X_yes = spec(xs_yes)

fig, ax = plt.subplots(1, 2, figsize=(6.5, 2.9), sharey=True)
ax[0].stem(fr, X_no, basefmt=" ", markerfmt="bo", linefmt="b-")
ax[0].set_title("Without anti-aliasing filter")
ax[0].set_xlabel("frequency (Hz)")
ax[0].set_ylabel("amplitude")
ax[1].stem(fr, X_yes, basefmt=" ", markerfmt="ro", linefmt="r-")
ax[1].set_title("With low-pass filter (8 Hz) before sampling")
ax[1].set_xlabel("frequency (Hz)")
plt.tight_layout()
plt.savefig("figs/fig7_antialiasing.png", dpi=170)
plt.close()

k3 = int(np.argmin(np.abs(fr - 3)))
k6 = int(np.argmin(np.abs(fr - 6)))
results["exp7_no_filter_3Hz"] = float(X_no[k3])
results["exp7_no_filter_6Hz"] = float(X_no[k6])
results["exp7_with_filter_3Hz"] = float(X_yes[k3])
results["exp7_with_filter_6Hz"] = float(X_yes[k6])

with open("results.json", "w") as fh:
    json.dump(results, fh, indent=2)

print(json.dumps(results, indent=2))
