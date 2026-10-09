# 音声マルチモーダル感情分析

音声ファイルの音響特徴量（ピッチ・音量・話速）と、文字起こしテキストの感情分析を組み合わせたマルチモーダル分析ツール。

## できること

- 音声の時系列分析（ピッチ変動・音量・ZCR）
- 話者ターン単位のテキスト感情分類（Positive / Neutral / Negative）
- 音声とテキスト感情の比較グラフ生成

## ファイル構成

```
audio-sentiment-analysis/
├── SKILL.md                         # Claude用スキル定義（Claudeがこのツールを使うときの手順）
├── common.py                        # 共通処理（日本語フォント・VTT解析・音響特徴量）
├── analyze_audio.py                 # 音声特徴量抽出・可視化
├── analyze_text_sentiment.py        # docx文字起こし + テキスト感情分析
├── analyze_vtt_multimodal.py        # YouTube VTT字幕 + 音声 マルチモーダル分析
├── analyze_speaker_propagation.py   # 話者分離 + 感情伝播可視化
├── requirements.txt                 # 実行用
├── requirements-dev.txt             # テスト用（モデル不要の軽量セット）
└── tests/
```

## セットアップ

```bash
pip install -r requirements.txt
```

## 使い方

パスはすべてコマンドライン引数で渡す（スクリプトの書き換えは不要）。詳しいオプションは各スクリプトの `--help` を参照。
日本語フォントはOSに入っているものを自動で選ぶ（MS Gothic / Yu Gothic / Hiragino / Noto Sans CJK / IPAゴシックなど）。

```bash
# 1. 音声のみ
python analyze_audio.py your_audio.wav -o audio_analysis.png

# 2. 音声 + docx文字起こし（docxはそのまま渡せる。展開は不要）
python analyze_text_sentiment.py your_audio.wav transcript.docx -o multimodal.png \
    --speaker "話者 2=ハヤカワ" --speaker "話者 3=usutaku"

# 3. 音声 + VTT字幕
python analyze_vtt_multimodal.py input.wav input.ja.vtt -o output.png --title "番組名"

# 4. 話者分離 + 感情伝播（環境変数 HF_TOKEN が必要）
python analyze_speaker_propagation.py input.wav input.ja.vtt -o propagation.png --speakers 4 --title "MTGコマンダー戦"
```

## テスト

```bash
pip install -r requirements-dev.txt
pytest tests
```

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
- 話者分離に対応しているのは analyze_speaker_propagation.py のみ（他は混合音声で精度が落ちる）

## 今後の改善候補

- [ ] WhisperX による強制アライメント（タイムスタンプ付き文字起こし）
- [ ] ~~pyannote-audio による話者分離~~ → **完了**（analyze_speaker_propagation.py）
- [ ] ~~複数話者の感情伝播の可視化~~ → **完了**（相互相関ヒートマップ実装済み）

## 初出・実績

- 2026-05-12：ながらAI #1（AIポッドキャスト）を対象に初回分析実施
- 参照レポート：`nagara_ai_analysis_report.md`
