"""
テキスト感情分析 + 音声特徴量のマルチモーダル比較
docxの文字起こし（話者ターン単位）に感情分析をかけ、音声特徴量と比較グラフを生成する

前提：
    docxを事前にunpackしてXMLを展開しておくこと
    unpack方法: python scripts/office/unpack.py your.docx unpacked/

Usage:
    WAV_PATH / DOCX_XML / OUT_PATH / SPEAKER_MAP を変更して実行
    python analyze_text_sentiment.py
"""

import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Patch
from transformers import pipeline
import librosa
import warnings
warnings.filterwarnings("ignore")

# ---- 設定 ----
WAV_PATH   = r"your_audio.wav"
DOCX_XML   = r"unpacked/word/document.xml"
OUT_PATH   = r"multimodal_analysis.png"
WINDOW_SEC = 10

# 話者ラベルの表示名マッピング（"話者 2" -> "ハヤカワ" など）
SPEAKER_MAP = {
    "話者 2": ("ハヤカワ", "royalblue"),
    "話者 3": ("usutaku",  "tomato"),
}
# --------------

matplotlib.rcParams['font.family'] = 'MS Gothic'

# --- docx XMLから話者ターンを抽出 ---
NS = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
tree = ET.parse(DOCX_XML)
root = tree.getroot()

turns = []
current_speaker, current_text = None, ""

for p in root.findall('.//w:p', NS):
    texts = [t.text for t in p.findall('.//w:t', NS) if t.text]
    line = "".join(texts).strip()
    if not line:
        continue
    if line.startswith("話者"):
        if current_speaker and current_text.strip():
            turns.append((current_speaker, current_text.strip()))
        current_speaker = line
        current_text = ""
    else:
        current_text += line + " "

if current_speaker and current_text.strip():
    turns.append((current_speaker, current_text.strip()))

print(f"話者ターン数: {len(turns)}")

# --- テキスト感情分析 ---
print("感情分析モデルをロード中...")
sentiment = pipeline(
    "sentiment-analysis",
    model="lxyuan/distilbert-base-multilingual-cased-sentiments-student",
    truncation=True, max_length=512
)

print("テキスト感情分析中...")
scores, speakers = [], []
for sp, tx in turns:
    result = sentiment(tx[:400])[0]
    label, score = result['label'], result['score']
    val = score if label.lower() == 'positive' else (-score if label.lower() == 'negative' else 0)
    scores.append(val)
    speakers.append(sp)

# --- 音声特徴量抽出 ---
print("音声特徴量を取得中...")
y, sr = librosa.load(WAV_PATH, sr=16000, mono=True)
duration = librosa.get_duration(y=y, sr=sr)

hop = int(WINDOW_SEC * sr)
times, energies, pitch_vars = [], [], []

for start in range(0, len(y) - hop, hop // 2):
    chunk = y[start:start + hop]
    times.append((start + hop / 2) / sr / 60)
    rms = librosa.feature.rms(y=chunk)[0]
    energies.append(float(np.mean(rms)))
    f0, voiced_flag, _ = librosa.pyin(chunk, fmin=80, fmax=800, sr=sr)
    voiced = f0[voiced_flag]
    pitch_vars.append(np.std(voiced) if len(voiced) > 0 else 0)

def norm(x):
    mn, mx = np.min(x), np.max(x)
    return (x - mn) / (mx - mn + 1e-8)

times      = np.array(times)
energies   = norm(np.array(energies))
pitch_vars = norm(np.array(pitch_vars))

# ターンを時間軸に等分割マッピング
turn_times = np.linspace(0, duration / 60, len(scores))
scores_arr = np.array(scores)
bar_colors = [SPEAKER_MAP.get(sp, ("不明", "gray"))[1] for sp in speakers]

# --- 可視化 ---
fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
fig.suptitle("マルチモーダル分析（音声 + テキスト感情）", fontsize=13, fontweight='bold')

axes[0].plot(times, pitch_vars, color='darkorange', linewidth=1.2, label='ピッチ変動（感情表現）')
axes[0].set_ylabel("ピッチ変動\n（正規化）", fontsize=9)
axes[0].set_ylim(0, 1); axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)

axes[1].plot(times, energies, color='seagreen', linewidth=1.2, label='音量（RMS）')
axes[1].set_ylabel("音量\n（正規化）", fontsize=9)
axes[1].set_ylim(0, 1); axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)

axes[2].bar(turn_times, scores_arr, width=duration / 60 / len(scores) * 0.8,
            color=bar_colors, alpha=0.7)
axes[2].axhline(0, color='black', linewidth=0.8)
axes[2].set_ylabel("テキスト感情価\n(+ポジ / -ネガ)", fontsize=9)
axes[2].set_xlabel("時間（分）", fontsize=10)
axes[2].set_ylim(-1, 1); axes[2].grid(True, alpha=0.3)
axes[2].legend(handles=[
    Patch(color=c, label=n) for sp, (n, c) in SPEAKER_MAP.items()
], fontsize=8)

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=150, bbox_inches='tight')
print(f"[完了] {OUT_PATH}")

# --- サマリー ---
print("\n--- テキスト感情サマリー ---")
for sp, (name, _) in SPEAKER_MAP.items():
    vals = [s for s, speaker in zip(scores, speakers) if speaker == sp]
    if vals:
        print(f"{name} 平均感情価: {np.mean(vals):.3f}")
print(f"全体 ポジティブ率: {sum(1 for s in scores if s > 0) / len(scores) * 100:.0f}%")
