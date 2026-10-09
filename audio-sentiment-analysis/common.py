"""
各分析スクリプトで共通の処理
フォント設定・VTT解析・感情スコア変換・音響特徴量抽出
"""

import re
import warnings

import numpy as np
import matplotlib
from matplotlib import font_manager

SENTIMENT_MODEL = "koheiduck/bert-japanese-finetuned-sentiment"

# 上から順に、インストール済みの最初の日本語フォントを使う（Windows / macOS / Linux）
JAPANESE_FONTS = [
    "MS Gothic", "Yu Gothic", "Meiryo",
    "Hiragino Sans", "Hiragino Kaku Gothic ProN",
    "Noto Sans CJK JP", "Noto Sans JP", "IPAexGothic", "IPAGothic", "TakaoGothic",
]


def setup_japanese_font():
    """日本語フォントを設定し、使ったフォント名を返す（見つからなければ None）"""
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in JAPANESE_FONTS:
        if name in installed:
            matplotlib.rcParams["font.family"] = name
            return name
    warnings.warn("日本語フォントが見つかりません。グラフの日本語が文字化けします。")
    return None


def parse_vtt(path):
    """VTT字幕から (開始秒, テキスト) のリストを返す"""
    with open(path, encoding="utf-8") as f:
        content = f.read()
    # <c>タグ・インラインタイムスタンプを除去
    content = re.sub(r"<[^>]+>", "", content)
    blocks = re.split(r"\n\n+", content)
    segments = []
    for block in blocks:
        lines = block.strip().split("\n")
        time_line = next((l for l in lines if "-->" in l), None)
        if not time_line:
            continue
        m = re.match(r"(\d+:\d+:\d+\.\d+)\s*-->", time_line)
        if not m:
            continue
        h, mn, s = m.group(1).split(":")
        t_sec = int(h) * 3600 + int(mn) * 60 + float(s)
        text_lines = [l.strip() for l in lines if "-->" not in l and l.strip() and l.strip() != "WEBVTT"]
        # 空行・音楽タグを除去
        text = re.sub(r"\[.*?\]", "", " ".join(text_lines)).strip()
        if text:
            segments.append((t_sec, text))
    return segments


def load_sentiment_model():
    from transformers import pipeline
    return pipeline("sentiment-analysis", model=SENTIMENT_MODEL, truncation=True, max_length=512)


def sentiment_value(result):
    """pipelineの結果 {'label', 'score'} を感情価（+1=ポジ, -1=ネガ）に変換"""
    label, score = result["label"].upper(), result["score"]
    if "POSITIVE" in label:
        return score
    if "NEGATIVE" in label:
        return -score
    return 0


def norm(x):
    mn, mx = np.min(x), np.max(x)
    return (x - mn) / (mx - mn + 1e-8)


def extract_audio_features(y, sr, window_sec, fmin=80, fmax=800):
    """半分ずつ重ねたウィンドウごとに音響特徴量を計算する

    返り値の各配列はウィンドウ数の長さ。times は中心時刻（秒）。
    """
    import librosa

    hop = int(window_sec * sr)
    feats = {k: [] for k in ("times", "pitch_mean", "pitch_std", "rms", "zcr")}
    for start in range(0, len(y) - hop, hop // 2):
        chunk = y[start:start + hop]
        feats["times"].append((start + hop / 2) / sr)

        f0, voiced_flag, _ = librosa.pyin(chunk, fmin=fmin, fmax=fmax, sr=sr, frame_length=2048)
        voiced = f0[voiced_flag]
        feats["pitch_mean"].append(np.mean(voiced) if len(voiced) > 0 else 0)
        feats["pitch_std"].append(np.std(voiced) if len(voiced) > 0 else 0)

        feats["rms"].append(float(np.mean(librosa.feature.rms(y=chunk)[0])))
        feats["zcr"].append(float(np.mean(librosa.feature.zero_crossing_rate(chunk)[0])))
    return {k: np.array(v) for k, v in feats.items()}
