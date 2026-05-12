"""
VTT字幕 + 音声 マルチモーダル感情分析
YouTube自動字幕（VTT）のタイムスタンプを使って音声特徴量と正確にアライメントし、
テキスト感情と音響特徴量を同一タイムラインで可視化する。

必要なライブラリ:
    pip install librosa transformers torch fugashi ipadic unidic-lite matplotlib

使い方:
    WAV_PATH / VTT_PATH / OUT_PATH / TITLE を書き換えて実行する。
"""

import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from transformers import pipeline
import librosa
import warnings
warnings.filterwarnings("ignore")

matplotlib.rcParams['font.family'] = 'MS Gothic'

# --- 設定 ---
WAV_PATH   = r"input.wav"          # 音声ファイルのパス
VTT_PATH   = r"input.ja.vtt"       # YouTube自動字幕VTTのパス
OUT_PATH   = r"output.png"         # 出力グラフのパス
TITLE      = "マルチモーダル分析"   # グラフタイトル
WINDOW_SEC = 30  # テキスト集約ウィンドウ（秒）

# --- VTT解析：タイムスタンプ付きテキストを抽出 ---
def parse_vtt(path):
    with open(path, encoding='utf-8') as f:
        content = f.read()
    # <c>タグ・インラインタイムスタンプを除去
    content = re.sub(r'<[^>]+>', '', content)
    blocks = re.split(r'\n\n+', content)
    segments = []
    for block in blocks:
        lines = block.strip().split('\n')
        time_line = next((l for l in lines if '-->' in l), None)
        if not time_line:
            continue
        m = re.match(r'(\d+:\d+:\d+\.\d+)\s*-->', time_line)
        if not m:
            continue
        t_str = m.group(1)
        h, mn, s = t_str.split(':')
        t_sec = int(h)*3600 + int(mn)*60 + float(s)
        text_lines = [l.strip() for l in lines if '-->' not in l and l.strip() and l.strip() != 'WEBVTT']
        text = ' '.join(text_lines)
        # 空行・音楽タグを除去
        text = re.sub(r'\[.*?\]', '', text).strip()
        if text:
            segments.append((t_sec, text))
    return segments

print("VTT字幕を解析中...")
segments = parse_vtt(VTT_PATH)
print(f"  字幕セグメント数: {len(segments)}")

# 30秒ウィンドウに集約
print("30秒ウィンドウに集約中...")
y_dummy, sr_dummy = librosa.load(WAV_PATH, sr=16000, mono=True)
duration = librosa.get_duration(y=y_dummy, sr=sr_dummy)

window_texts = []
window_times = []
for w_start in np.arange(0, duration, WINDOW_SEC):
    w_end = w_start + WINDOW_SEC
    texts = [t for (ts, t) in segments if w_start <= ts < w_end]
    combined = ' '.join(texts).strip()
    # 重複テキストを削除（VTTの重複行）
    words = []
    seen = set()
    for w in combined.split():
        if w not in seen:
            words.append(w)
            seen.add(w)
    window_texts.append(' '.join(words))
    window_times.append((w_start + w_end) / 2 / 60)

# --- テキスト感情分析 ---
print("感情分析モデルをロード中...")
sentiment = pipeline(
    "sentiment-analysis",
    model="koheiduck/bert-japanese-finetuned-sentiment",
    truncation=True, max_length=512
)

print("テキスト感情分析中...")
scores = []
for i, text in enumerate(window_texts):
    if not text.strip():
        scores.append(0)
        continue
    result = sentiment(text[:400])[0]
    label, score = result['label'], result['score']
    val = score if 'POSITIVE' in label.upper() else (-score if 'NEGATIVE' in label.upper() else 0)
    scores.append(val)
    print(f"  {window_times[i]:.1f}分: {label}({score:.2f}) | {text[:40]}...")

# --- 音声特徴量 ---
print("音声特徴量を抽出中...")
WINDOW_AUDIO = 10
y, sr = y_dummy, sr_dummy
hop = int(WINDOW_AUDIO * sr)
times_a, energies, pitch_vars = [], [], []

for start in range(0, len(y) - hop, hop // 2):
    chunk = y[start:start + hop]
    times_a.append((start + hop / 2) / sr / 60)
    rms = librosa.feature.rms(y=chunk)[0]
    energies.append(float(np.mean(rms)))
    f0, voiced_flag, _ = librosa.pyin(chunk, fmin=80, fmax=800, sr=sr)
    voiced = f0[voiced_flag]
    pitch_vars.append(np.std(voiced) if len(voiced) > 0 else 0)

def norm(x):
    mn, mx = np.min(x), np.max(x)
    return (x - mn) / (mx - mn + 1e-8)

times_a   = np.array(times_a)
energies  = norm(np.array(energies))
pitch_vars = norm(np.array(pitch_vars))
window_times = np.array(window_times)
scores_arr = np.array(scores)

# --- 可視化 ---
fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
fig.suptitle(f"{TITLE} - マルチモーダル分析（音声 + 字幕感情）", fontsize=13, fontweight='bold')

axes[0].plot(times_a, pitch_vars, color='darkorange', linewidth=1.2, label='ピッチ変動（感情表現）')
axes[0].set_ylabel("ピッチ変動\n（正規化）", fontsize=9)
axes[0].set_ylim(0, 1); axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)

axes[1].plot(times_a, energies, color='seagreen', linewidth=1.2, label='音量（RMS）')
axes[1].set_ylabel("音量\n（正規化）", fontsize=9)
axes[1].set_ylim(0, 1); axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)

bar_colors = ['royalblue' if s >= 0 else 'tomato' for s in scores_arr]
axes[2].bar(window_times, scores_arr, width=WINDOW_SEC/60*0.8, color=bar_colors, alpha=0.7)
axes[2].axhline(0, color='black', linewidth=0.8)
axes[2].set_ylabel("テキスト感情価\n(+ポジ / -ネガ)", fontsize=9)
axes[2].set_xlabel("時間（分）", fontsize=10)
axes[2].set_ylim(-1, 1); axes[2].grid(True, alpha=0.3)

from matplotlib.patches import Patch
axes[2].legend(handles=[
    Patch(color='royalblue', label='ポジティブ'),
    Patch(color='tomato', label='ネガティブ'),
], fontsize=8)

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=150, bbox_inches='tight')
print(f"\n[完了] {OUT_PATH}")
print(f"\n--- テキスト感情サマリー ---")
print(f"平均感情価: {np.mean(scores_arr):.3f}")
print(f"ポジティブ率: {sum(1 for s in scores_arr if s > 0)/len(scores_arr)*100:.0f}%")
print(f"感情ピーク時刻: {window_times[np.argmax(np.abs(scores_arr))]:.1f}分")
