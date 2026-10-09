"""モデル（Hugging Face）不要の部分のテスト"""

import zipfile

import numpy as np
import pytest

from common import parse_vtt, sentiment_value, norm, extract_audio_features, setup_japanese_font
from analyze_text_sentiment import parse_docx_turns
from analyze_vtt_multimodal import aggregate_windows
from analyze_speaker_propagation import assign_speakers, compute_cross_correlation

VTT = """WEBVTT
Kind: captions
Language: ja

00:00:01.000 --> 00:00:03.000
<c>こんにちは</c>

00:00:05.500 --> 00:00:07.000
[音楽]

01:00:10.250 --> 01:00:12.000
今日は 楽しい
"""


def test_parse_vtt(tmp_path):
    p = tmp_path / "a.vtt"
    p.write_text(VTT, encoding="utf-8")
    assert parse_vtt(p) == [(1.0, "こんにちは"), (3610.25, "今日は 楽しい")]


@pytest.mark.parametrize("label,score,expected", [
    ("POSITIVE", 0.9, 0.9), ("negative", 0.8, -0.8), ("NEUTRAL", 0.7, 0),
])
def test_sentiment_value(label, score, expected):
    assert sentiment_value({"label": label, "score": score}) == expected


def test_norm_range():
    x = norm(np.array([2.0, 4.0, 6.0]))
    assert x[0] == 0 and x[-1] == pytest.approx(1)


def test_extract_audio_features_sine():
    sr = 16000
    t = np.arange(sr * 4) / sr
    y = 0.5 * np.sin(2 * np.pi * 220 * t).astype(np.float32)
    f = extract_audio_features(y, sr, window_sec=1)
    # 1秒窓を0.5秒ずつずらす。末尾の窓は切り捨てるので (4-1)/0.5 = 6 個
    assert len(f["times"]) == len(f["rms"]) == 6
    # 220Hzの純音なのでF0推定はほぼ220Hz、揺れはほぼ0
    assert np.median(f["pitch_mean"]) == pytest.approx(220, rel=0.03)
    assert np.median(f["pitch_std"]) < 5


def test_parse_docx_turns(tmp_path):
    w = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'
    paras = ["話者 1", "おはよう", "ございます", "話者 2", "どうも"]
    body = "".join(f"<w:p><w:r><w:t>{t}</w:t></w:r></w:p>" for t in paras)
    p = tmp_path / "t.docx"
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("word/document.xml", f"<w:document {w}><w:body>{body}</w:body></w:document>")
    assert parse_docx_turns(str(p)) == [("話者 1", "おはよう ございます"), ("話者 2", "どうも")]


def test_aggregate_windows_dedup():
    segs = [(1, "あ い"), (2, "い う"), (31, "え")]
    times, texts = aggregate_windows(segs, duration=60, window_sec=30)
    assert texts == ["あ い う", "え"]
    assert times == [0.25, 0.75]


def test_assign_speakers():
    turns = [(0, 5, "A"), (5, 10, "B")]
    labeled = assign_speakers([(1, "x"), (7, "y"), (20, "z")], turns)
    assert [s for _, s, _ in labeled] == ["A", "B", "UNKNOWN"]


def test_cross_correlation_self_is_one():
    a = np.array([0.1, -0.5, 0.8, np.nan, 0.3])
    corr, lag = compute_cross_correlation({"A": a, "B": -a}, ["A", "B"])
    assert corr[0, 0] == pytest.approx(1, abs=1e-6) and lag[0, 0] == 0
    assert corr[0, 1] == pytest.approx(-1, abs=1e-6)


def test_setup_japanese_font_does_not_crash():
    setup_japanese_font()
