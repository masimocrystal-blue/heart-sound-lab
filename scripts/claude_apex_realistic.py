import numpy as np
from scipy.signal import butter, sosfiltfilt
import soundfile as sf

sr = 48000
rng = np.random.default_rng(42)

# ---------------------------------------------------------------
# 単一弁音の合成：帯域制限ノイズバースト + 減衰サイン（少量）に
# 急峻立ち上がり/指数減衰エンベロープをかける
#  -> 弁閉鎖が心血管系（cardiohemic system）を叩いて生じる
#     減衰振動を模す。ノイズ主体で「ピー」ではなく「ドッ」にする。
# ---------------------------------------------------------------
def bandpass_noise(n, centers, amps, sr, order=2):
    noise = rng.standard_normal(n)
    out = np.zeros(n)
    for fc, a in zip(centers, amps):
        low = max(fc * 0.6, 12.0)
        high = min(fc * 1.5, sr * 0.45)
        sos = butter(order, [low, high], btype='band', fs=sr, output='sos')
        out += a * sosfiltfilt(sos, noise)
    return out

def valve_sound(centers, amps, total_dur, attack, decay_tau, sr,
                noise_mix=1.0, sine_mix=0.35):
    n = int(total_dur * sr)
    t = np.arange(n) / sr
    env = (1.0 - np.exp(-t / attack)) * np.exp(-t / decay_tau)
    env /= (env.max() + 1e-12)
    noise = bandpass_noise(n, centers, amps, sr)
    noise /= (np.max(np.abs(noise)) + 1e-12)
    sine = np.zeros(n)
    for fc, a in zip(centers, amps):
        sine += a * np.sin(2 * np.pi * fc * t + rng.uniform(0, 2 * np.pi))
    sine /= (np.max(np.abs(sine)) + 1e-12)
    return env * (noise_mix * noise + sine_mix * sine)

# ---------------------------------------------------------------
T = 60.0
pad = sr
buf = np.zeros(int(T * sr) + pad)

def add(sound, t0, gain):
    i0 = int(t0 * sr)
    buf[i0:i0 + len(sound)] += gain * sound

# 呼吸性洞性不整脈(RSA)：吸気で心拍↑。心拍ごとにRR間隔を更新。
HR0 = 68.0          # bpm
HRamp = 5.0         # 吸気↔呼気の心拍スイング
resp = 15.0 / 60.0  # 呼吸 ~15回/分 (Hz)

beats = []
t = 0.45
while t < T - 0.6:
    insp = np.sin(2 * np.pi * resp * t)            # +1 吸気ピーク
    hr = HR0 + HRamp * insp
    rr = 60.0 / hr * (1 + rng.uniform(-0.015, 0.015))  # ±1.5%ゆらぎ
    beats.append(t)
    t += rr

for tb in beats:
    insp_level = (np.sin(2 * np.pi * resp * tb) + 1) / 2   # 0..1
    ampmod = 1.0 + 0.08 * (insp_level - 0.5) * 2           # 呼吸性の音量うねり ±8%
    t1 = tb + rng.uniform(-0.003, 0.003)                   # microタイミングjitter

    # --- S1 : M1(僧帽弁) + T1(三尖弁) 約20-25ms遅れ・やや小さい ---
    M1 = valve_sound([35, 55, 80], [1.0, 0.8, 0.4], 0.140, 0.008, 0.034, sr)
    T1 = valve_sound([40, 62],     [1.0, 0.6],      0.120, 0.008, 0.030, sr)
    a1 = 1.00 * ampmod * (1 + rng.uniform(-0.05, 0.05))
    add(M1, t1, a1)
    add(T1, t1 + 0.022, a1 * 0.55)

    # --- S2 : A2(大動脈弁) + P2(肺動脈弁) 吸気で分裂が開く ---
    qs2 = 0.300 + rng.uniform(-0.008, 0.008)               # 電気機械的収縮期 ~0.30s
    ts2 = t1 + qs2
    split = 0.012 + insp_level * 0.030                     # 呼気12ms→吸気42ms
    A2 = valve_sound([55, 90, 130], [1.0, 0.7, 0.35], 0.090, 0.005, 0.020, sr)
    P2 = valve_sound([60, 100],     [1.0, 0.6],       0.080, 0.005, 0.018, sr)
    a2 = 0.72 * ampmod * (1 + rng.uniform(-0.05, 0.05))    # 心尖部:S1>S2
    add(A2, ts2, a2)
    add(P2, ts2 + split, a2 * 0.5)                         # 心尖部ではP2は減弱

# ---------------------------------------------------------------
# 背景：胸腔の低域ランブル + ごく微かな呼吸音(吸気で増)
n = len(buf)
tt = np.arange(n) / sr

rumble = rng.standard_normal(n)
sos = butter(2, 35, btype='low', fs=sr, output='sos')
rumble = sosfiltfilt(sos, rumble); rumble /= np.max(np.abs(rumble))
buf += 0.012 * rumble

breath = rng.standard_normal(n)
sos = butter(2, [160, 420], btype='band', fs=sr, output='sos')
breath = sosfiltfilt(sos, breath); breath /= np.max(np.abs(breath))
respenv = 0.5 + 0.5 * np.sin(2 * np.pi * resp * tt)
buf += 0.016 * breath * respenv

# ---------------------------------------------------------------
# 聴診器(ベル)的な最終整形：18Hz HPF + 360Hz LPF
sos = butter(2, 18, btype='high', fs=sr, output='sos'); buf = sosfiltfilt(sos, buf)
sos = butter(2, 360, btype='low', fs=sr, output='sos'); buf = sosfiltfilt(sos, buf)

buf = buf[:int(T * sr)]
buf /= np.max(np.abs(buf)); buf *= 0.84   # peak ~ -1.5 dBFS

sf.write('outputs/heart_apex_60s.wav', buf, sr, subtype='PCM_24')
print("beats:", len(beats), " mean HR:", round(60*len(beats)/T,1), "bpm")
print("dur(s):", round(len(buf)/sr,2), " peak:", round(np.max(np.abs(buf)),3))
print("RMS dBFS:", round(20*np.log10(np.sqrt(np.mean(buf**2))),1))
