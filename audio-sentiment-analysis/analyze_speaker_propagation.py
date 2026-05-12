"""
話者分離 + 感情伝播可視化
pyannote-audio で話者分離 → VTTとマージ → 感情スコア計算 → 相互相関ヒートマップ

必要な環境変数: HF_TOKEN
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Patch
import torch
import librosa
from transformers import pipeline
import warnings
warnings.filterwarnings("ignore")

matplotlib.rcParams['font.family'] = 'MS Gothic'

# --- 設定 ---
WAV_PATH  = r"C:\Users\55kas\Downloads\mtg_commander.wav"
VTT_PATH  = r"C:\Users\55kas\Downloads\mtg_commander.ja.vtt"
OUT_PATH  = r"C:\Users\55kas\NightAru\mtg_speaker_propagation.png"
HF_TOKEN  = os.environ.get("HF_TOKEN")
WINDOW_SEC = 60   # 話者ごとの感情集約ウィンドウ（秒）
N_SPEAKERS = 4    # コマンダー戦は4人


# --- VTT解析 ---
def parse_vtt(path):
    with open(path, encoding='utf-8') as f:
        content = f.read()
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
        text = re.sub(r'\[.*?\]', '', ' '.join(text_lines)).strip()
        if text:
            segments.append((t_sec, text))
    return segments


# --- 話者分離 ---
def run_diarization(wav_path, hf_token, n_speakers):
    print("音声を読み込み中...")
    y, sr = librosa.load(wav_path, sr=16000, mono=True)
    waveform = torch.tensor(y).unsqueeze(0)  # (1, samples)

    print("話者分離モデルをロード中...")
    from pyannote.audio import Pipeline
    pipe = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=hf_token
    )

    print(f"話者分離を実行中（{n_speakers}人）...")
    diarization = pipe(
        {"waveform": waveform, "sample_rate": sr},
        num_speakers=n_speakers
    )

    # (start, end, speaker_label) のリストに変換
    # pyannote 4.x は DiarizeOutput を返す → .speaker_diarization でAnnotationを取得
    annotation = diarization.speaker_diarization
    turns = []
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        turns.append((turn.start, turn.end, speaker))
    print(f"  検出ターン数: {len(turns)}")
    return turns


# --- VTTセグメントに話者ラベルを付与 ---
def assign_speakers(segments, turns):
    labeled = []
    for t_sec, text in segments:
        speaker = "UNKNOWN"
        for start, end, spk in turns:
            if start <= t_sec <= end:
                speaker = spk
                break
        labeled.append((t_sec, speaker, text))
    return labeled


# --- 話者ごとに感情スコアを時系列化 ---
def compute_speaker_sentiments(labeled, duration, sentiment_model, window_sec, speakers):
    windows = np.arange(0, duration, window_sec)
    scores = {spk: [] for spk in speakers}
    times = []

    for w_start in windows:
        w_end = w_start + window_sec
        times.append((w_start + w_end) / 2 / 60)
        for spk in speakers:
            texts = [t for (ts, s, t) in labeled if s == spk and w_start <= ts < w_end]
            combined = ' '.join(texts).strip()
            if not combined:
                scores[spk].append(np.nan)
                continue
            result = sentiment_model(combined[:400])[0]
            label, score = result['label'], result['score']
            val = score if 'POSITIVE' in label.upper() else (-score if 'NEGATIVE' in label.upper() else 0)
            scores[spk].append(val)

    return np.array(times), {spk: np.array(v) for spk, v in scores.items()}


# --- 相互相関ヒートマップ ---
def plot_cross_correlation(scores, speakers):
    n = len(speakers)
    corr_matrix = np.zeros((n, n))
    lag_matrix = np.zeros((n, n))

    for i, spk_a in enumerate(speakers):
        for j, spk_b in enumerate(speakers):
            a = scores[spk_a]
            b = scores[spk_b]
            # NaNを0で補完
            a = np.where(np.isnan(a), 0, a)
            b = np.where(np.isnan(b), 0, b)
            corr = np.correlate(a - a.mean(), b - b.mean(), mode='full')
            lags = np.arange(-(len(a)-1), len(a))
            best_lag = lags[np.argmax(np.abs(corr))]
            best_corr = corr[np.argmax(np.abs(corr))] / (len(a) * a.std() * b.std() + 1e-8)
            corr_matrix[i, j] = best_corr
            lag_matrix[i, j] = best_lag

    return corr_matrix, lag_matrix


# --- メイン ---
print("VTT字幕を解析中...")
segments = parse_vtt(VTT_PATH)
print(f"  字幕セグメント数: {len(segments)}")

turns = run_diarization(WAV_PATH, HF_TOKEN, N_SPEAKERS)

speakers = sorted(set(spk for _, _, spk in turns))
print(f"  検出話者: {speakers}")

print("話者ラベルをVTTに付与中...")
labeled = assign_speakers(segments, turns)

print("感情分析モデルをロード中...")
sentiment = pipeline(
    "sentiment-analysis",
    model="koheiduck/bert-japanese-finetuned-sentiment",
    truncation=True, max_length=512
)

y_tmp, sr_tmp = librosa.load(WAV_PATH, sr=16000, mono=True)
duration = librosa.get_duration(y=y_tmp, sr=sr_tmp)

print("話者ごとに感情スコアを計算中...")
times, scores = compute_speaker_sentiments(labeled, duration, sentiment, WINDOW_SEC, speakers)

print("相互相関を計算中...")
corr_matrix, lag_matrix = plot_cross_correlation(scores, speakers)

# --- 可視化 ---
colors = ['royalblue', 'tomato', 'seagreen', 'darkorange']
spk_colors = {spk: colors[i % len(colors)] for i, spk in enumerate(speakers)}

fig, axes = plt.subplots(2, 1, figsize=(14, 10))
fig.suptitle("MTGコマンダー戦 - 話者別感情伝播分析", fontsize=13, fontweight='bold')

# 上段：話者別感情時系列
ax = axes[0]
for spk in speakers:
    s = scores[spk]
    valid = ~np.isnan(s)
    ax.plot(times[valid], s[valid], color=spk_colors[spk], linewidth=1.5,
            marker='o', markersize=3, label=spk)
ax.axhline(0, color='black', linewidth=0.8)
ax.set_ylabel("感情価\n(+ポジ / -ネガ)", fontsize=9)
ax.set_ylim(-1.1, 1.1)
ax.legend(fontsize=8, loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_title("話者別 感情スコアの時系列", fontsize=10)

# 下段：相互相関ヒートマップ
ax2 = axes[1]
im = ax2.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
ax2.set_xticks(range(len(speakers)))
ax2.set_yticks(range(len(speakers)))
ax2.set_xticklabels(speakers, fontsize=9)
ax2.set_yticklabels(speakers, fontsize=9)
ax2.set_title("感情伝播 相互相関ヒートマップ\n（行→列の影響度、正=同期・負=逆相関）", fontsize=10)
for i in range(len(speakers)):
    for j in range(len(speakers)):
        lag = int(lag_matrix[i, j])
        ax2.text(j, i, f"{corr_matrix[i,j]:.2f}\nlag={lag}", ha='center', va='center',
                 fontsize=7, color='black')
plt.colorbar(im, ax=ax2, label='相関係数')

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=150, bbox_inches='tight')
print(f"\n[完了] {OUT_PATH}")

# サマリー
print("\n--- 感情伝播サマリー ---")
for i, spk_a in enumerate(speakers):
    for j, spk_b in enumerate(speakers):
        if i != j and abs(corr_matrix[i, j]) > 0.3:
            direction = "正相関（同期）" if corr_matrix[i, j] > 0 else "逆相関"
            print(f"  {spk_a} → {spk_b}: {corr_matrix[i,j]:.2f} ({direction}, lag={int(lag_matrix[i,j])})")
