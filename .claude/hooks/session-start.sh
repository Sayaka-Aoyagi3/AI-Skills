#!/bin/bash
# クラウドセッション開始時に、テストとlintに必要な依存関係を入れる
set -euo pipefail

if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

cd "$CLAUDE_PROJECT_DIR"
pip install --quiet --disable-pip-version-check -r audio-sentiment-analysis/requirements-dev.txt
