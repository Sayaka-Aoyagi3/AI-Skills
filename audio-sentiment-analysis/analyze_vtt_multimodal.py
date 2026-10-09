"""
VTT字幕 + 音声 マルチモーダル感情分析
YouTube自動字幕（VTT）のタイムスタンプを使って音声特徴量と正確にアライメントし、
テキスト感情と音響特徴量を同一タイムラインで可視化する。

Usage:
    python analyze_vtt_multimodal.py input.wav input.ja.vtt -o output.png --title "番組名"
"""

import argparse
import warnings

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import librosa

from common import setup_japanese_font, parse_vtt, load_sentiment_model, sentiment_value, norm, extract_audio_features

warnings.filterwarnings("ignore")


def aggregate_windows(segments, duration, window_sec):
    """字幕セグメントをウィンドウごとに連結する。返り値は (中心時刻[分]のリスト, テキストのリスト)"""
    window_texts, window_times = [], []
    for w_start in np.arange(0, duration, window_sec):
        w_end = w_start + window_sec
        texts = [t for (ts, t) in segments if w_start <= ts < w_end]
        # 重複テキストを削除（VTTの重複行）
        words = list(dict.fromkeys(' '.join(texts).split()))
        window_texts.append(' '.join(words))
        window_times.append((w_start + w_end) / 2 / 60)
    return window_times, window_texts


def main():
    ap = argparse.ArgumentParser(description="VTT字幕 + 音声のマルチモーダル感情分析")
    ap.add_argument("wav", help="音声ファイル")
    ap.add_argument("vtt", help="YouTube自動字幕などのVTTファイル")
    ap.add_argument("-o", "--out", default="output.png", help="出力グラフのパス")
    ap.add_argument("--title", default="マルチモーダル分析", help="グラフタイトル")
    ap.add_argument("--window", type=float, default=30, help="テキスト集約ウィンドウ（秒）")
    ap.add_argument("--audio-window", type=float, default=10, help="音声分析ウィンドウ（秒）")
    args = ap.parse_args()

    setup_japanese_font()

    print("VTT字幕を解析中...")
    segments = parse_vtt(args.vtt)
    print(f"  字幕セグメント数: {len(segments)}")

    y, sr = librosa.load(args.wav, sr=16000, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)

    print(f"{args.window:g}秒ウィンドウに集約中...")
    window_times, window_texts = aggregate_windows(segments, duration, args.window)

    print("感情分析モデルをロード中...")
    sentiment = load_sentiment_model()

    print("テキスト感情分析中...")
    scores = []
    for t, text in zip(window_times, window_texts):
        if not text.strip():
            scores.append(0)
            continue
        result = sentiment(text[:400])[0]
        scores.append(sentiment_value(result))
        print(f"  {t:.1f}分: {result['label']}({result['score']:.2f}) | {text[:40]}...")

    print("音声特徴量を抽出中...")
    f = extract_audio_features(y, sr, args.audio_window)
    times_a = f["times"] / 60
    energies = norm(f["rms"])
    pitch_vars = norm(f["pitch_std"])
    window_times = np.array(window_times)
    scores_arr = np.array(scores)

    # --- 可視化 ---
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    fig.suptitle(f"{args.title} - マルチモーダル分析（音声 + 字幕感情）", fontsize=13, fontweight='bold')

    axes[0].plot(times_a, pitch_vars, color='darkorange', linewidth=1.2, label='ピッチ変動（感情表現）')
    axes[0].set_ylabel("ピッチ変動\n（正規化）", fontsize=9)
    axes[0].set_ylim(0, 1); axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)

    axes[1].plot(times_a, energies, color='seagreen', linewidth=1.2, label='音量（RMS）')
    axes[1].set_ylabel("音量\n（正規化）", fontsize=9)
    axes[1].set_ylim(0, 1); axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)

    bar_colors = ['royalblue' if s >= 0 else 'tomato' for s in scores_arr]
    axes[2].bar(window_times, scores_arr, width=args.window/60*0.8, color=bar_colors, alpha=0.7)
    axes[2].axhline(0, color='black', linewidth=0.8)
    axes[2].set_ylabel("テキスト感情価\n(+ポジ / -ネガ)", fontsize=9)
    axes[2].set_xlabel("時間（分）", fontsize=10)
    axes[2].set_ylim(-1, 1); axes[2].grid(True, alpha=0.3)
    axes[2].legend(handles=[
        Patch(color='royalblue', label='ポジティブ'),
        Patch(color='tomato', label='ネガティブ'),
    ], fontsize=8)

    plt.tight_layout()
    plt.savefig(args.out, dpi=150, bbox_inches='tight')
    print(f"\n[完了] {args.out}")
    print("\n--- テキスト感情サマリー ---")
    print(f"平均感情価: {np.mean(scores_arr):.3f}")
    print(f"ポジティブ率: {sum(1 for s in scores_arr if s > 0)/len(scores_arr)*100:.0f}%")
    print(f"感情ピーク時刻: {window_times[np.argmax(np.abs(scores_arr))]:.1f}分")


if __name__ == "__main__":
    main()
