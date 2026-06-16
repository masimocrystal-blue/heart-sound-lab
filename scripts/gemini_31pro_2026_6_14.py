import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, filtfilt

# ==========================================
# 1. 基本設定 (Global Parameters)
# ==========================================
SAMPLE_RATE = 44100
DURATION_SEC = 60
T_TOTAL = np.linspace(0, DURATION_SEC, DURATION_SEC * SAMPLE_RATE, endpoint=False)

# ==========================================
# 2. 呼吸と心拍の数理モデル (Biomathematical Model)
# ==========================================
BREATH_CYCLE_SEC = 8.0  # 4秒吸気、4秒呼気の8秒周期
BREATH_FREQ = 1.0 / BREATH_CYCLE_SEC
# 呼吸フェーズ: 0~1 (0~0.5が吸気、0.5~1.0が呼気を想定)
breath_phase = (T_TOTAL % BREATH_CYCLE_SEC) / BREATH_CYCLE_SEC

# 呼吸性不整脈 (Respiratory Sinus Arrhythmia)
# 吸気時にBPM65へ上昇、呼気時にBPM55へ下降
BPM_BASE = 60
BPM_MOD = 5 * np.sin(2 * np.pi * BREATH_FREQ * T_TOTAL)
INSTANTANEOUS_BPM = BPM_BASE + BPM_MOD

# 心拍のタイミングを計算 (Beat times)
beat_times = []
current_time = 0.0
while current_time < DURATION_SEC:
    beat_times.append(current_time)
    # 現在時刻のBPMを取得し、次の拍までの時間を計算
    idx = int(current_time * SAMPLE_RATE)
    if idx >= len(INSTANTANEOUS_BPM): break
    current_bpm = INSTANTANEOUS_BPM[idx]
    current_time += 60.0 / current_bpm

# ==========================================
# 3. エンベロープ＆オシレーター生成関数
# ==========================================
def apply_envelope(length, attack_sec, decay_sec, sr=SAMPLE_RATE):
    """指数関数的ディケイを持つ非対称エンベロープ"""
    t = np.linspace(0, length / sr, length, endpoint=False)
    attack_samples = int(attack_sec * sr)
    
    env = np.zeros(length)
    if attack_samples > 0:
        env[:attack_samples] = np.linspace(0, 1, attack_samples)
    
    decay_t = t[attack_samples:] - t[attack_samples]
    # 時定数(tau)を調整してディケイカーブを作る
    tau = decay_sec / 3.0 
    env[attack_samples:] = np.exp(-decay_t / tau)
    return env

def generate_s1():
    """I音 (僧帽弁・三尖弁閉鎖音)"""
    duration = 0.15
    samples = int(duration * SAMPLE_RATE)
    t = np.linspace(0, duration, samples, endpoint=False)
    
    # OSC 1: Heart Muscle Thud (筋肉の低い振動)
    # 50Hzから30Hzへのピッチドロップ
    freqs = np.linspace(50, 30, samples)
    phase = np.cumsum(freqs) / SAMPLE_RATE * 2 * np.pi
    thud = np.sin(phase)
    thud_env = apply_envelope(samples, 0.01, 0.08)
    
    # OSC 2: Valve Snap (弁が閉じる高音成分)
    noise = np.random.normal(0, 1, samples)
    # 簡易的なローパス(移動平均)でノイズを少し丸める
    noise = np.convolve(noise, np.ones(5)/5, mode='same')
    snap_env = apply_envelope(samples, 0.002, 0.015)
    
    s1 = (thud * thud_env) * 0.8 + (noise * snap_env) * 0.15
    return s1

def generate_s2():
    """II音 (大動脈弁・肺動脈弁閉鎖音)"""
    duration = 0.1
    samples = int(duration * SAMPLE_RATE)
    t = np.linspace(0, duration, samples, endpoint=False)
    
    # OSC 1: A2/P2 Valve (少し高めでタイトな音)
    freq = 120
    # サイン波と少しの倍音(三角波的アプローチ)
    wave = np.sin(2 * np.pi * freq * t) + 0.1 * np.sin(2 * np.pi * freq * 3 * t)
    env = apply_envelope(samples, 0.005, 0.04)
    
    s2 = wave * env * 0.6
    return s2

# ==========================================
# 4. タイムラインへのレンダリング
# ==========================================
audio_buffer = np.zeros_like(T_TOTAL)

s1_wave = generate_s1()
s2_wave = generate_s2()

for b_time in beat_times:
    if b_time >= DURATION_SEC: break
    
    # 吸気時にS2が分裂するギミック (生理的分裂)
    # 呼吸フェーズ(0~1)から、吸気時(sin波の頂点付近)に最大0.035秒の遅延を作る
    b_idx = int(b_time * SAMPLE_RATE)
    split_delay_sec = 0.035 * max(0, np.sin(2 * np.pi * BREATH_FREQ * b_time))
    
    # 収縮期(Systole)の長さ（S1とS2の間隔）は約0.35秒
    systole_duration = 0.35
    
    # タイミング計算 (サンプル単位)
    idx_s1 = b_idx
    idx_s2_a2 = int((b_time + systole_duration) * SAMPLE_RATE)
    idx_s2_p2 = int((b_time + systole_duration + split_delay_sec) * SAMPLE_RATE)
    
    # バッファへの加算 (S1)
    if idx_s1 + len(s1_wave) < len(audio_buffer):
        audio_buffer[idx_s1:idx_s1+len(s1_wave)] += s1_wave
        
    # バッファへの加算 (S2 - A2成分)
    if idx_s2_a2 + len(s2_wave) < len(audio_buffer):
        audio_buffer[idx_s2_a2:idx_s2_a2+len(s2_wave)] += s2_wave
        
    # バッファへの加算 (S2 - P2成分、呼吸で分裂幅が変わる)
    if idx_s2_p2 + len(s2_wave) < len(audio_buffer):
        audio_buffer[idx_s2_p2:idx_s2_p2+len(s2_wave)] += s2_wave * 0.8 # P2は少し音量小さめ

# ==========================================
# 5. FXチェーン (DSPフィルタリング)
# ==========================================
# 5.1 聴診器のダイアフラム（膜）と体壁の特性を模倣するローパスフィルター
def butter_lowpass_filter(data, cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

# 250Hz以上の高域をカットしてくぐもった音に
audio_filtered = butter_lowpass_filter(audio_buffer, cutoff=250.0, fs=SAMPLE_RATE, order=4)

# 5.2 聴診器チューブ内の共鳴（非常に短いディレイによるコムフィルター効果）
delay_samples = int(0.005 * SAMPLE_RATE) # 5msの遅延
tube_resonance = np.zeros_like(audio_filtered)
tube_resonance[delay_samples:] = audio_filtered[:-delay_samples] * 0.3
audio_filtered += tube_resonance

# 5.3 環境ノイズ（検者の指の擦れ、部屋の空調などのフロアノイズ）
noise_floor = np.random.normal(0, 0.005, len(audio_filtered))
noise_floor = butter_lowpass_filter(noise_floor, cutoff=50.0, fs=SAMPLE_RATE, order=2)
audio_filtered += noise_floor

# ==========================================
# 6. ノーマライズとWAV書き出し
# ==========================================
# ピークを0.9にノーマライズ
audio_normalized = audio_filtered / np.max(np.abs(audio_filtered)) * 0.9

# 16-bit PCMに変換
audio_pcm = np.int16(audio_normalized * 32767)

# ファイル保存
output_filename = "outputs/gemini_apex_synth_model.wav"
wavfile.write(output_filename, SAMPLE_RATE, audio_pcm)
print(f"Exported: {output_filename}")