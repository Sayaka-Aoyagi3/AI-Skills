"""
音声特徴量抽出・可視化
ピッチ（F0）・音量（RMS）・話速（ZCR）を時系列で分析してグラフを生成する

Usage:
    python analyze_audio.py your_audio.wav -o audio_analysis.png
"""

import argparse
import warnings

import numpy as np
import matplotlib.pyplot as plt
import librosa

from common import setup_japanese_font, norm, extract_audio_features

warnings.filterwarnings("ignore")


def main():
    ap = argparse.ArgumentParser(description="音響特徴量の時系列分析")
    ap.add_argument("wav", help="分析する音声ファイル")
    ap.add_argument("-o", "--out", default="audio_analysis.png", help="出力グラフのパス")
    ap.add_argument("--window", type=float, default=10, help="分析ウィンドウ幅（秒）")
    ap.add_argument("--title", default="音響特徴量の時系列分析")
    args = ap.parse_args()

    setup_japanese_font()

    print("音声ファイル読み込み中...")
    y, sr = librosa.load(args.wav, sr=16000, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)
    print(f"  長さ: {duration:.1f}秒 ({duration/60:.1f}分)")

    print("音響特徴量を抽出中...")
    f = extract_audio_features(y, sr, args.window,
                               fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    times = f["times"]

    print("グラフ描画中...")
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(args.title, fontsize=14, fontweight='bold')

    for ax, data, color, label in zip(
        axes,
        [f["pitch_mean"], f["pitch_std"], f["rms"], f["zcr"]],
        ['royalblue', 'tomato', 'seagreen', 'darkorchid'],
        ['ピッチ（平均F0）\n正規化', 'ピッチ変動\n（感情表現の代理）', '音量（RMS）\n正規化', 'ZCR\n（話速の代理）']
    ):
        ax.plot(times / 60, norm(data), color=color, linewidth=1.2)
        ax.set_ylabel(label, fontsize=9)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("時間（分）", fontsize=10)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150, bbox_inches='tight')
    print(f"[完了] {args.out}")

    pitches_mean = f["pitch_mean"]
    print("\n--- 数値サマリー ---")
    print(f"ピッチ平均: {np.mean(pitches_mean[pitches_mean>0]):.1f} Hz")
    print(f"ピッチ最大: {np.max(pitches_mean):.1f} Hz")
    print(f"音量ピーク時刻: {times[np.argmax(f['rms'])]/60:.1f}分")
    print(f"ピッチ変動最大時刻: {times[np.argmax(f['pitch_std'])]/60:.1f}分")


if __name__ == "__main__":
    main()
