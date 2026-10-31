#!/usr/bin/env bash
# Считает все таблицы заново и сверяет вывод с эталоном expected.txt.
# Код выхода 1, если результаты разошлись.
set -euo pipefail
cd "$(dirname "$0")"

# Поток случайных чисел закреплён только для версии numpy из requirements.txt
python3 -m venv .venv
.venv/bin/pip -q install -r requirements.txt

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

{
  .venv/bin/python simulate.py
  echo
  echo "ПРИМЕР retrodesign.py"
  python3 retrodesign.py --users 120000 --base 0.05 --lift 0.02 --observed 0.06 --tau 0.01
} > "$tmp/actual.txt"

if diff -u expected.txt "$tmp/actual.txt"; then
  echo "OK: результаты совпадают с expected.txt"
else
  echo "Результаты разошлись с expected.txt" >&2
  exit 1
fi
