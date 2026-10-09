"""
テキスト感情分析 + 音声特徴量のマルチモーダル比較
docxの文字起こし（話者ターン単位）に感情分析をかけ、音声特徴量と比較グラフを生成する

docxはそのまま渡せる（展開済みの word/document.xml を渡してもよい）。
文字起こしは「話者 N」で始まる段落を話者ターンの区切りとして扱う。

Usage:
    python analyze_text_sentiment.py your_audio.wav transcript.docx -o multimodal_analysis.png \
        --speaker "話者 2=ハヤカワ" --speaker "話者 3=usutaku"
"""

import argparse
import xml.etree.ElementTree as ET
import zipfile
import warnings

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import librosa

from common import setup_japanese_font, load_sentiment_model, sentiment_value, norm, extract_audio_features

warnings.filterwarnings("ignore")

NS = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
COLORS = ['royalblue', 'tomato', 'seagreen', 'darkorange', 'darkorchid', 'goldenrod']


def parse_docx_turns(path):
    """docx（または document.xml）から (話者ラベル, テキスト) のリストを返す"""
    if path.lower().endswith(".docx"):
        with zipfile.ZipFile(path) as z, z.open("word/document.xml") as f:
            root = ET.parse(f).getroot()
    else:
        root = ET.parse(path).getroot()

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
    return turns


def main():
    ap = argparse.ArgumentParser(description="docx文字起こし + 音声のマルチモーダル感情分析")
    ap.add_argument("wav", help="音声ファイル")
    ap.add_argument("transcript", help="文字起こしの .docx（または展開済み document.xml）")
    ap.add_argument("-o", "--out", default="multimodal_analysis.png", help="出力グラフのパス")
    ap.add_argument("--window", type=float, default=10, help="音声分析ウィンドウ幅（秒）")
    ap.add_argument("--speaker", action="append", default=[], metavar="LABEL=NAME",
                    help='話者ラベルの表示名（例: "話者 2=ハヤカワ"）。複数指定可')
    args = ap.parse_args()

    setup_japanese_font()

    turns = parse_docx_turns(args.transcript)
    print(f"話者ターン数: {len(turns)}")

    # 話者ラベル -> (表示名, 色)。--speaker 未指定ならラベルをそのまま使う
    names = dict(s.split("=", 1) for s in args.speaker)
    labels = list(names) or sorted({sp for sp, _ in turns})
    speaker_map = {lb: (names.get(lb, lb), COLORS[i % len(COLORS)]) for i, lb in enumerate(labels)}

    print("感情分析モデルをロード中...")
    sentiment = load_sentiment_model()

    print("テキスト感情分析中...")
    scores = [sentiment_value(sentiment(tx[:400])[0]) for _, tx in turns]
    speakers = [sp for sp, _ in turns]

    print("音声特徴量を取得中...")
    y, sr = librosa.load(args.wav, sr=16000, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)
    f = extract_audio_features(y, sr, args.window)
    times = f["times"] / 60
    energies = norm(f["rms"])
    pitch_vars = norm(f["pitch_std"])

    # ターンを時間軸に等分割マッピング
    turn_times = np.linspace(0, duration / 60, len(scores))
    scores_arr = np.array(scores)
    bar_colors = [speaker_map.get(sp, ("不明", "gray"))[1] for sp in speakers]

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
    axes[2].legend(handles=[Patch(color=c, label=n) for n, c in speaker_map.values()], fontsize=8)

    plt.tight_layout()
    plt.savefig(args.out, dpi=150, bbox_inches='tight')
    print(f"[完了] {args.out}")

    # --- サマリー ---
    print("\n--- テキスト感情サマリー ---")
    for sp, (name, _) in speaker_map.items():
        vals = [s for s, speaker in zip(scores, speakers) if speaker == sp]
        if vals:
            print(f"{name} 平均感情価: {np.mean(vals):.3f}")
    print(f"全体 ポジティブ率: {sum(1 for s in scores if s > 0) / len(scores) * 100:.0f}%")


if __name__ == "__main__":
    main()
