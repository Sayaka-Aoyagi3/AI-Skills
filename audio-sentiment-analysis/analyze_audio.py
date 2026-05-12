"""
音声特徴量抽出・可視化
ピッチ（F0）・音量（RMS）・話速（ZCR）を時系列で分析してグラフを生成する

Usage:
    WAV_PATH と OUT_PATH を変更して実行
    python analyze_audio.py
"""

import librosa
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import warnings
warnings.filterwarnings("ignore")

# ---- 設定 ----
WAV_PATH    = r"your_audio.wav"   # 分析する音声ファイル
OUT_PATH    = r"audio_analysis.png"  # 出力グラフのパス
WINDOW_SEC  = 10                  # 分析ウィンドウ幅（秒）
TITLE       = "音響特徴量の時系列分析"
# --------------

matplotlib.rcParams['font.family'] = 'MS Gothic'

print("音声ファイル読み込み中...")
y, sr = librosa.load(WAV_PATH, sr=16000, mono=True)
duration = librosa.get_duration(y=y, sr=sr)
print(f"  長さ: {duration:.1f}秒 ({duration/60:.1f}分)")

hop = int(WINDOW_SEC * sr)
times, pitches_mean, pitches_var, energies, zcrs = [], [], [], [], []

print("音響特徴量を抽出中...")
for start in range(0, len(y) - hop, hop // 2):
    chunk = y[start:start + hop]
    t = (start + hop / 2) / sr
    times.append(t)

    f0, voiced_flag, _ = librosa.pyin(
        chunk, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'),
        sr=sr, frame_length=2048
    )
    voiced = f0[voiced_flag]
    pitches_mean.append(np.mean(voiced) if len(voiced) > 0 else 0)
    pitches_var.append(np.std(voiced) if len(voiced) > 0 else 0)

    rms = librosa.feature.rms(y=chunk)[0]
    energies.append(float(np.mean(rms)))

    zcr = librosa.feature.zero_crossing_rate(chunk)[0]
    zcrs.append(float(np.mean(zcr)))

times        = np.array(times)
pitches_mean = np.array(pitches_mean)
pitches_var  = np.array(pitches_var)
energies     = np.array(energies)
zcrs         = np.array(zcrs)

def norm(x):
    mn, mx = np.min(x), np.max(x)
    return (x - mn) / (mx - mn + 1e-8)

print("グラフ描画中...")
fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
fig.suptitle(TITLE, fontsize=14, fontweight='bold')

for ax, data, color, label in zip(
    axes,
    [pitches_mean, pitches_var, energies, zcrs],
    ['royalblue', 'tomato', 'seagreen', 'darkorchid'],
    ['ピッチ（平均F0）\n正規化', 'ピッチ変動\n（感情表現の代理）', '音量（RMS）\n正規化', 'ZCR\n（話速の代理）']
):
    ax.plot(times / 60, norm(data), color=color, linewidth=1.2)
    ax.set_ylabel(label, fontsize=9)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel("時間（分）", fontsize=10)
plt.tight_layout()
plt.savefig(OUT_PATH, dpi=150, bbox_inches='tight')
print(f"[完了] {OUT_PATH}")

print(f"\n--- 数値サマリー ---")
print(f"ピッチ平均: {np.mean(pitches_mean[pitches_mean>0]):.1f} Hz")
print(f"ピッチ最大: {np.max(pitches_mean):.1f} Hz")
print(f"音量ピーク時刻: {times[np.argmax(energies)]/60:.1f}分")
print(f"ピッチ変動最大時刻: {times[np.argmax(pitches_var)]/60:.1f}分")
