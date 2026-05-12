# 音声マルチモーダル感情分析

音声ファイルの音響特徴量（ピッチ・音量・話速）と、文字起こしテキストの感情分析を組み合わせたマルチモーダル分析ツール。

## できること

- 音声の時系列分析（ピッチ変動・音量・ZCR）
- 話者ターン単位のテキスト感情分類（Positive / Neutral / Negative）
- 音声とテキスト感情の比較グラフ生成

## ファイル構成

```
audio-sentiment-analysis/
├── analyze_audio.py                 # 音声特徴量抽出・可視化
├── analyze_text_sentiment.py        # docx文字起こし + テキスト感情分析
├── analyze_vtt_multimodal.py        # YouTube VTT字幕 + 音声 マルチモーダル分析
├── analyze_speaker_propagation.py   # 話者分離 + 感情伝播可視化（4話者対応）
└── requirements.txt
```

## セットアップ

```bash
pip install -r requirements.txt
```

## 使い方

### 1. 音声のみ分析

```python
# analyze_audio.py の WAV_PATH を変更して実行
WAV_PATH = r"your_audio.wav"
python analyze_audio.py
```

出力：`nagara_ai_analysis.png`（ピッチ・音量・ZCR の時系列グラフ）

### 2. マルチモーダル分析（音声 + テキスト）

文字起こしdocxが必要。docxのXMLを展開してから実行。

```python
# analyze_text_sentiment.py の以下を変更
WAV_PATH  = r"your_audio.wav"
DOCX_XML  = r"unpacked_docx/word/document.xml"
python analyze_text_sentiment.py
```

出力：`nagara_ai_multimodal.png`（ピッチ変動・音量・感情バーの比較グラフ）

## 使用モデル

| モデル | 用途 | 言語 |
|--------|------|------|
| `koheiduck/bert-japanese-finetuned-sentiment` | テキスト感情分類 | 日本語特化BERT |
| librosa pYIN | ピッチ（F0）推定 | 言語非依存 |

## 出力の読み方

| 指標 | 感情との対応 |
|------|------------|
| ピッチ変動（std F0） | 大きいほど感情表現が豊か |
| RMSエネルギー | 大きいほど発話が強い・興奮している |
| ZCR | 大きいほど話速が速い・子音密度が高い |
| 感情価 | +1=強いポジティブ、0=ニュートラル、-1=強いネガティブ |

## 制限事項

- 音声とテキストの時間アライメントは等分割推定（粗い）
- 話者分離未対応（混合音声では精度低下）

## 今後の改善候補

- [ ] WhisperX による強制アライメント（タイムスタンプ付き文字起こし）
- [ ] ~~pyannote-audio による話者分離~~ → **完了**（analyze_speaker_propagation.py）
- [ ] ~~複数話者の感情伝播の可視化~~ → **完了**（相互相関ヒートマップ実装済み）

## 初出・実績

- 2026-05-12：ながらAI #1（AIポッドキャスト）を対象に初回分析実施
- 参照レポート：`nagara_ai_analysis_report.md`
