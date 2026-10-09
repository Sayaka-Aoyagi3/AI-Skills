"""
話者分離 + 感情伝播可視化
pyannote-audio で話者分離 → VTTとマージ → 感情スコア計算 → 相互相関ヒートマップ

必要な環境変数: HF_TOKEN（pyannote/speaker-diarization-3.1 の利用規約に同意したHugging Faceアカウントのトークン）

Usage:
    python analyze_speaker_propagation.py input.wav input.ja.vtt -o propagation.png \
        --speakers 4 --title "MTGコマンダー戦"
"""

import argparse
import os
import warnings

import numpy as np
import matplotlib.pyplot as plt
import librosa

from common import setup_japanese_font, parse_vtt, load_sentiment_model, sentiment_value

warnings.filterwarnings("ignore")


# --- 話者分離 ---
def run_diarization(y, sr, hf_token, n_speakers):
    import torch
    from pyannote.audio import Pipeline

    waveform = torch.tensor(y).unsqueeze(0)  # (1, samples)

    print("話者分離モデルをロード中...")
    pipe = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=hf_token
    )

    print(f"話者分離を実行中（{f'{n_speakers}人' if n_speakers else '人数は自動推定'}）...")
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
            scores[spk].append(sentiment_value(sentiment_model(combined[:400])[0]))

    return np.array(times), {spk: np.array(v) for spk, v in scores.items()}


# --- 相互相関 ---
def compute_cross_correlation(scores, speakers):
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
def main():
    ap = argparse.ArgumentParser(description="話者分離 + 感情伝播の可視化")
    ap.add_argument("wav", help="音声ファイル")
    ap.add_argument("vtt", help="タイムスタンプ付き字幕（VTT）")
    ap.add_argument("-o", "--out", default="speaker_propagation.png", help="出力グラフのパス")
    ap.add_argument("--speakers", type=int, default=None, help="話者数（省略時は自動推定）")
    ap.add_argument("--window", type=float, default=60, help="話者ごとの感情集約ウィンドウ（秒）")
    ap.add_argument("--title", default="話者別分析", help="グラフタイトル")
    args = ap.parse_args()

    setup_japanese_font()

    print("VTT字幕を解析中...")
    segments = parse_vtt(args.vtt)
    print(f"  字幕セグメント数: {len(segments)}")

    print("音声を読み込み中...")
    y, sr = librosa.load(args.wav, sr=16000, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)

    turns = run_diarization(y, sr, os.environ.get("HF_TOKEN"), args.speakers)

    speakers = sorted(set(spk for _, _, spk in turns))
    print(f"  検出話者: {speakers}")

    print("話者ラベルをVTTに付与中...")
    labeled = assign_speakers(segments, turns)

    print("感情分析モデルをロード中...")
    sentiment = load_sentiment_model()

    print("話者ごとに感情スコアを計算中...")
    times, scores = compute_speaker_sentiments(labeled, duration, sentiment, args.window, speakers)

    print("相互相関を計算中...")
    corr_matrix, lag_matrix = compute_cross_correlation(scores, speakers)

    # --- 可視化 ---
    colors = ['royalblue', 'tomato', 'seagreen', 'darkorange']
    spk_colors = {spk: colors[i % len(colors)] for i, spk in enumerate(speakers)}

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle(f"{args.title} - 話者別感情伝播分析", fontsize=13, fontweight='bold')

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
    plt.savefig(args.out, dpi=150, bbox_inches='tight')
    print(f"\n[完了] {args.out}")

    # サマリー
    print("\n--- 感情伝播サマリー ---")
    for i, spk_a in enumerate(speakers):
        for j, spk_b in enumerate(speakers):
            if i != j and abs(corr_matrix[i, j]) > 0.3:
                direction = "正相関（同期）" if corr_matrix[i, j] > 0 else "逆相関"
                print(f"  {spk_a} → {spk_b}: {corr_matrix[i,j]:.2f} ({direction}, lag={int(lag_matrix[i,j])})")


if __name__ == "__main__":
    main()
