---
name: audio-sentiment-analysis
description: 日本語の音声（ポッドキャスト・会議・対戦配信など）を、音響特徴量（ピッチ・音量・話速）と文字起こしのテキスト感情（日本語BERT）でマルチモーダルに分析し、時系列グラフを作る。話者分離（pyannote）による話者間の感情伝播（相互相関）の可視化にも対応。wav/mp3 と、VTT字幕または話者付きdocx文字起こしがあるときに使う。
---

# 音声マルチモーダル感情分析

スクリプトはこのファイルと同じフォルダにある。作業ディレクトリをこのフォルダにして実行する（`common.py` を import するため）。

## どれを使うか

| 手元にあるもの | スクリプト | 出力 |
|---|---|---|
| 音声だけ | `analyze_audio.py WAV` | ピッチ・ピッチ変動・音量・ZCRの時系列 |
| 音声 + 話者付きdocx文字起こし | `analyze_text_sentiment.py WAV DOCX` | 音響特徴量 + 話者ターンごとの感情価 |
| 音声 + VTT字幕（YouTube自動字幕など） | `analyze_vtt_multimodal.py WAV VTT` | 音響特徴量 + 時間窓ごとの感情価 |
| 上に加えて話者ごとの感情の連動を見たい | `analyze_speaker_propagation.py WAV VTT` | 話者別感情時系列 + 相互相関ヒートマップ |

各スクリプトは `-o 出力.png` で出力先を指定できる。その他のオプションは `--help` で確認する。

```bash
python analyze_vtt_multimodal.py ep1.wav ep1.ja.vtt -o ep1.png --title "ながらAI #1"
python analyze_text_sentiment.py ep1.wav ep1.docx --speaker "話者 2=ハヤカワ" --speaker "話者 3=usutaku"
HF_TOKEN=... python analyze_speaker_propagation.py game.wav game.ja.vtt --speakers 4 --title "MTGコマンダー戦"
```

## 前提

- `pip install -r requirements.txt`
- 初回実行時にHugging Faceからモデルをダウンロードする（`huggingface.co` への通信が必要）
  - 感情分析: `koheiduck/bert-japanese-finetuned-sentiment`
  - 話者分離: `pyannote/speaker-diarization-3.1`（Hugging Faceで利用規約に同意し、環境変数 `HF_TOKEN` を設定する）
- 話者分離はCPUでも動くが遅い。長い音声はGPUのある環境で動かす

## 結果を解釈するときの注意

結果を報告するときは、次の限界も一緒に伝える。

- 「ピッチ変動＝感情表現」「ZCR＝話速」はあくまで代理指標。話者の声質や録音環境の影響を受ける
- 感情価は「テキストの」感情で、皮肉や冗談の口調は反映されない。1回の推論に使うのは先頭400文字まで
- `analyze_text_sentiment.py` は話者ターンを時間軸に等間隔で並べるだけで、実際の発話時刻とは対応していない
- 相互相関の値が大きくても因果（誰が誰に影響したか）は示さない。話者のいない区間は0で埋めている
- 各特徴量は音声ごとに0〜1へ正規化しているので、別々の音声の値を直接比べてはいけない
