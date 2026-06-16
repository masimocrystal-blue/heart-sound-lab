import numpy as np
import wave
from pathlib import Path
import json

# Fast, deterministic 60s synthetic stethoscope heart sound
sr = 44100
duration = 60.0
n = int(sr * duration)
rng = np.random.default_rng(13579)
out_dir = Path("outputs")
out_dir.mkdir(exist_ok=True)

out_wav = out_dir / "gpt_realistic_apex_heart_sound_60s.wav"

try:
    from scipy import signal
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

y = np.zeros(n, dtype=np.float32)

def add_event(t0, freqs, amps, dur, tau, attack, gain):
    """Add a multi-part damped low-frequency heart sound."""
    start = int(t0 * sr)
    if start >= n:
        return
    m = int(dur * sr)
    end = min(n, start + m)
    m = end - start
    if m <= 8:
        return

    tt = np.arange(m, dtype=np.float32) / sr
    env = (1.0 - np.exp(-tt / attack)) * np.exp(-tt / tau)

    # subtle organic envelope irregularity
    env *= 1.0 + 0.018 * np.sin(2*np.pi*(7.0 + rng.normal(0, 0.3))*tt + rng.uniform(0, 2*np.pi))

    # fade out click prevention
    fade = min(m, int(0.014 * sr))
    if fade > 4:
        env[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)

    s = np.zeros(m, dtype=np.float32)
    for f, a in zip(freqs, amps):
        f = f * (1.0 + rng.normal(0, 0.006))
        phase = rng.uniform(0, 2*np.pi)
        drift = 1.0 + 0.0025 * np.sin(2*np.pi*(2.5 + rng.random()*2.0)*tt + rng.uniform(0, 2*np.pi))
        s += np.float32(a) * np.sin(2*np.pi*f*drift*tt + phase).astype(np.float32)

    y[start:end] += np.float32(gain) * env.astype(np.float32) * s

def moving_average(x, L):
    if L <= 1:
        return x
    c = np.cumsum(np.insert(x, 0, 0.0))
    out = (c[L:] - c[:-L]) / L
    pad_left = L // 2
    pad_right = len(x) - len(out) - pad_left
    return np.pad(out, (pad_left, pad_right), mode="edge").astype(np.float32)

def add_noise_pulse(t0, dur, gain, tau=0.030):
    """Short chest-wall / valve closure noise burst."""
    start = int(t0 * sr)
    if start >= n:
        return
    m = int(dur * sr)
    end = min(n, start + m)
    m = end - start
    if m <= 8:
        return

    tt = np.arange(m, dtype=np.float32) / sr
    env = (1.0 - np.exp(-tt / 0.0045)) * np.exp(-tt / tau)
    fade = min(m, int(0.010 * sr))
    if fade > 4:
        env[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)

    z = rng.normal(0, 1, m).astype(np.float32)
    # rough 35-180 Hz band shaping by subtracting smoothed versions
    low = moving_average(z, max(3, int(sr/180)))
    vlow = moving_average(z, max(3, int(sr/38)))
    band = low - 0.78 * vlow
    y[start:end] += np.float32(gain) * env.astype(np.float32) * band.astype(np.float32)

# Beat train: normal adult at apex, respiratory sinus arrhythmia
beat_times = []
t = 0.28
rr_state = 0.0
while t < duration + 0.8:
    resp = np.sin(2*np.pi*0.215*t + 0.7)  # about 12.9/min
    slow = np.sin(2*np.pi*0.031*t + 1.3)
    hr = 70.8 + 4.2*resp + 1.2*slow
    rr_state = 0.86*rr_state + rng.normal(0, 0.007)
    rr = 60.0/hr + rr_state
    rr = float(np.clip(rr, 0.735, 1.02))
    beat_times.append(t)
    t += rr

# Body/stethoscope bed
if HAS_SCIPY:
    bed = rng.normal(0, 1, n).astype(np.float32)
    sos_lp180 = signal.butter(3, 180, "lowpass", fs=sr, output="sos")
    bed = signal.sosfilt(sos_lp180, bed).astype(np.float32)
    bed *= 0.0019

    rumble = rng.normal(0, 1, n).astype(np.float32)
    sos_rumble = signal.butter(2, [12, 55], "bandpass", fs=sr, output="sos")
    rumble = signal.sosfilt(sos_rumble, rumble).astype(np.float32)
    rumble *= 0.0011
else:
    bed = moving_average(rng.normal(0, 1, n).astype(np.float32), 160)
    bed *= 0.0012
    rumble = moving_average(rng.normal(0, 1, n).astype(np.float32), 400)
    rumble *= 0.0010

time = np.arange(n, dtype=np.float32) / sr
breath_env = 0.5 + 0.5*np.sin(2*np.pi*0.215*time - 0.9)
y += bed + rumble * (0.65 + 0.35*breath_env).astype(np.float32)

# Heart sounds
for bt in beat_times:
    if bt >= duration:
        break

    resp = np.sin(2*np.pi*0.215*bt + 0.7)
    amp = 1.0 + 0.070*rng.normal() + 0.050*resp
    amp = float(np.clip(amp, 0.78, 1.24))

    # Systolic interval varies with HR / respiration
    systole = 0.318 - 0.012*max(resp, -0.4) + rng.normal(0, 0.008)
    systole = float(np.clip(systole, 0.285, 0.350))

    # S1: apex/mitral area, M1 dominant; T1 later and softer
    s1 = bt + rng.normal(0, 0.002)
    add_event(
        s1,
        freqs=[32.5, 48.5, 68.0, 91.0, 124.0],
        amps=[0.92, 0.84, 0.55, 0.25, 0.08],
        dur=0.155,
        tau=0.045,
        attack=0.010,
        gain=0.64*amp
    )
    add_event(
        s1 + 0.024 + rng.normal(0, 0.003),
        freqs=[38.0, 58.0, 82.0, 111.0],
        amps=[0.46, 0.38, 0.19, 0.07],
        dur=0.108,
        tau=0.036,
        attack=0.007,
        gain=0.31*amp
    )
    add_noise_pulse(s1 + 0.001, 0.073, 0.020*amp, tau=0.026)

    # tiny post-S1 chest rebound, below obvious S3 level
    add_event(
        s1 + 0.078 + rng.normal(0, 0.004),
        freqs=[25.5, 35.5],
        amps=[0.36, 0.18],
        dur=0.125,
        tau=0.062,
        attack=0.020,
        gain=0.090*amp
    )

    # S2: A2/P2 split, softer at apex
    s2 = bt + systole
    split = 0.026 + 0.013*max(resp, -0.1) + rng.normal(0, 0.0035)
    split = float(np.clip(split, 0.017, 0.049))

    add_event(
        s2,
        freqs=[45.0, 67.0, 96.0, 132.0, 165.0],
        amps=[0.55, 0.58, 0.35, 0.14, 0.04],
        dur=0.092,
        tau=0.026,
        attack=0.005,
        gain=0.42*amp
    )
    add_noise_pulse(s2 + 0.001, 0.052, 0.015*amp, tau=0.020)

    add_event(
        s2 + split,
        freqs=[41.0, 62.0, 88.0, 120.0],
        amps=[0.40, 0.35, 0.17, 0.05],
        dur=0.078,
        tau=0.023,
        attack=0.006,
        gain=0.19*amp
    )

    # Rare micro-rub / stethoscope contact shifts
    if rng.random() < 0.15:
        add_noise_pulse(bt + rng.uniform(0.12, 0.62), rng.uniform(0.030, 0.085), rng.uniform(0.0035, 0.0085), tau=rng.uniform(0.020, 0.050))

# Stethoscope/body transfer shaping
if HAS_SCIPY:
    # 18-215 Hz band, then subtle resonant boosts
    sos_bp = signal.butter(4, [18, 215], "bandpass", fs=sr, output="sos")
    y = signal.sosfilt(sos_bp, y).astype(np.float32)

    # Add gentle EQ via FFT-domain curve
    freq = np.fft.rfftfreq(n, 1/sr)
    Y = np.fft.rfft(y)
    curve = (
        1.0
        + 0.24*np.exp(-0.5*((freq-62)/24)**2)
        + 0.08*np.exp(-0.5*((freq-112)/36)**2)
        - 0.10*np.exp(-0.5*((freq-170)/45)**2)
    )
    Y *= curve
    y = np.fft.irfft(Y, n=n).astype(np.float32)
else:
    # fallback: simple FFT high/low rolloff
    freq = np.fft.rfftfreq(n, 1/sr)
    Y = np.fft.rfft(y)
    hp = 1/(1+(18/np.maximum(freq, 1e-9))**6)
    lp = 1/(1+(freq/210)**5)
    curve = hp*lp*(1+0.22*np.exp(-0.5*((freq-62)/24)**2))
    curve[0] = 0
    y = np.fft.irfft(Y*curve, n=n).astype(np.float32)

# Gentle tape-ish saturation / clinical recording level
y = np.tanh(y * 2.05).astype(np.float32) / 2.05

# Start/end fades
fade_in = int(0.12 * sr)
fade_out = int(0.18 * sr)
y[:fade_in] *= np.linspace(0, 1, fade_in, dtype=np.float32)
y[-fade_out:] *= np.linspace(1, 0, fade_out, dtype=np.float32)

# Normalize to conservative peak
peak = float(np.max(np.abs(y)))
if peak == 0:
    raise RuntimeError("Generated silence.")
y = y / peak * 0.78
pcm = np.int16(np.clip(y, -1, 1) * 32767)

with wave.open(str(out_wav), "wb") as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(sr)
    wf.writeframes(pcm.tobytes())


# Confirm file exists and print useful info
beats_in_file = [b for b in beat_times if b < duration]
rr = np.diff(beats_in_file)

mean_bpm = None
if len(rr) > 0:
    mean_bpm = round(float(60 / np.mean(rr)), 2)

info = {
    "wav_path": str(out_wav),
    "wav_size_bytes": out_wav.stat().st_size,
    "duration_sec": duration,
    "sample_rate": sr,
    "channels": 1,
    "bit_depth": 16,
    "beats": len(beats_in_file),
    "mean_bpm": mean_bpm,
    "scipy_used": HAS_SCIPY,
}

print(json.dumps(info, ensure_ascii=False, indent=2))
