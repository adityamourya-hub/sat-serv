#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_FILE="${1:-${ROOT_DIR}/dist/sbom/ssis_go_modules.json}"

mkdir -p "$(dirname "${OUT_FILE}")"

if command -v cyclonedx-gomod >/dev/null 2>&1; then
  (cd "${ROOT_DIR}" && cyclonedx-gomod mod -json -licenses -output "${OUT_FILE}")
else
  {
    echo "{"
    echo "  \"schema\": \"go-list-modules-v1\","
    echo "  \"generated_at_utc\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\","
    echo "  \"modules\": ["
    first=1
    while IFS= read -r line; do
      if [[ $first -eq 0 ]]; then
        echo "    ,${line}"
      else
        echo "    ${line}"
        first=0
      fi
    done < <(cd "${ROOT_DIR}" && go list -m -json all | tr '\n' ' ' | sed 's/} {/}\n{/g')
    echo "  ]"
    echo "}"
  } > "${OUT_FILE}"
fi

if command -v shasum >/dev/null 2>&1; then
  shasum -a 256 "${OUT_FILE}" > "${OUT_FILE}.sha256"
else
  sha256sum "${OUT_FILE}" > "${OUT_FILE}.sha256"
fi

echo "SBOM written to ${OUT_FILE}"
