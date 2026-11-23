#!/usr/bin/env bash
# Rebuild static/dashboard.css with the real Tailwind CLI.
# DEV MACHINE ONLY. The Pi never runs Node/npm; it just serves the built file.
#
# Usage: bash scripts/build_tailwind.sh
set -euo pipefail
cd "$(dirname "$0")/.."

TMP_INPUT=".tailwind.build.$$.css"
TMP_CONFIG=".tailwind.build.$$.js"
trap 'rm -f "$TMP_INPUT" "$TMP_CONFIG"' EXIT

cat > "$TMP_INPUT" <<'CSS'
@tailwind base;
@tailwind components;
@tailwind utilities;

@layer components {
  #chart { min-height: 260px; }
  .uplot { font-family: inherit; }
}
CSS

cat > "$TMP_CONFIG" <<'JS'
module.exports = {
  content: [
    "./app/templates/**/*.html",
    "./app/server.py"
  ],
  theme: { extend: {} },
  plugins: []
};
JS

npx --yes tailwindcss@3.4.17 \
  -c "$TMP_CONFIG" \
  -i "$TMP_INPUT" \
  -o static/dashboard.css \
  --minify

# Flask serves these precompressed; keep the .gz siblings in sync.
for f in static/dashboard.css static/*.js static/uplot.min.css; do
  gzip -9 -c "$f" > "$f.gz"
done

echo "Built static/dashboard.css"
ls -l static/dashboard.css static/*.gz
