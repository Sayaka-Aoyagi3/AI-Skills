# AI Skills

再利用可能なAI分析スキル集。研究・ビジネス・個人プロジェクト横断で使えるPythonコードをまとめています。

## スキル一覧

| フォルダ | 内容 | 主要ライブラリ |
|---------|------|-------------|
| [audio-sentiment-analysis](./audio-sentiment-analysis/) | 音声の音響特徴量抽出 + テキスト感情分析のマルチモーダル分析 | librosa, transformers |

## Claudeのスキルとして使う

各フォルダの `SKILL.md` は、Claudeがそのフォルダのツールを使うときに読む手順書です。

- **Claude Code:** フォルダごと `~/.claude/skills/` にコピーすると、すべてのプロジェクトで使えます
- **claude.ai:** フォルダをzipにして「設定 → 機能 → スキル」からアップロードします

## 動作環境

- Python 3.10+
- 各フォルダの `requirements.txt` を参照

## 開発者

青柳明佳 / 七夕研究所
