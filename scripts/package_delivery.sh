#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIST_ROOT="${1:-${ROOT_DIR}/dist}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
PACKAGE_DIR="${DIST_ROOT}/SSIS_ADITI4_PS24_${STAMP}"
BIN_PATH="${ROOT_DIR}/build/ssa_ops_runner"

mkdir -p "${PACKAGE_DIR}" "${PACKAGE_DIR}/bin" "${PACKAGE_DIR}/assets" "${PACKAGE_DIR}/docs" "${PACKAGE_DIR}/models" "${PACKAGE_DIR}/runtime" "${PACKAGE_DIR}/scripts"

if [[ ! -x "${BIN_PATH}" ]]; then
  if ! (cd "${ROOT_DIR}" && CGO_ENABLED=0 go build -trimpath -ldflags "-s -w -buildid=" -o "${BIN_PATH}" ./cmd/ssa_ops_runner); then
    echo "WARNING: static build unavailable (likely ONNX CGO dependency); building with CGO_ENABLED=1"
    (cd "${ROOT_DIR}" && CGO_ENABLED=1 go build -trimpath -ldflags "-s -w -buildid=" -o "${BIN_PATH}" ./cmd/ssa_ops_runner)
  fi
fi

cp "${BIN_PATH}" "${PACKAGE_DIR}/bin/ssa_ops_runner"
cp -R "${ROOT_DIR}/assets/." "${PACKAGE_DIR}/assets/"

if [[ -d "${ROOT_DIR}/pkg/ml/models" ]]; then
  mkdir -p "${PACKAGE_DIR}/models/ml"
  cp -R "${ROOT_DIR}/pkg/ml/models/." "${PACKAGE_DIR}/models/ml/"
fi

if [[ -d "${ROOT_DIR}/models/rl" ]]; then
  mkdir -p "${PACKAGE_DIR}/models/rl"
  cp -R "${ROOT_DIR}/models/rl/." "${PACKAGE_DIR}/models/rl/"
fi

if [[ -d "${ROOT_DIR}/models/llm" ]]; then
  mkdir -p "${PACKAGE_DIR}/models/llm"
  cp -R "${ROOT_DIR}/models/llm/." "${PACKAGE_DIR}/models/llm/"
fi

cp "${ROOT_DIR}/README.md" "${PACKAGE_DIR}/docs/README.md"
cp "${ROOT_DIR}/docs/architecture.md" "${PACKAGE_DIR}/docs/architecture.md"
cp "${ROOT_DIR}/docs/OPERATOR_MANUAL.md" "${PACKAGE_DIR}/docs/OPERATOR_MANUAL.md"
cp "${ROOT_DIR}/docs/HARDENING_COMPLIANCE_MATRIX.md" "${PACKAGE_DIR}/docs/HARDENING_COMPLIANCE_MATRIX.md"
cp "${ROOT_DIR}/docs/DEPLOYMENT_CHECKLIST.md" "${PACKAGE_DIR}/docs/DEPLOYMENT_CHECKLIST.md"

cp "${ROOT_DIR}/scripts/generate_sbom.sh" "${PACKAGE_DIR}/scripts/generate_sbom.sh"
chmod +x "${PACKAGE_DIR}/scripts/generate_sbom.sh"

"${ROOT_DIR}/scripts/generate_sbom.sh" "${PACKAGE_DIR}/docs/SBOM_go_modules.json"

cat > "${PACKAGE_DIR}/runtime/one_click_demo.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
"${DIR}/bin/ssa_ops_runner" -demo indian-catalogue -full -assistant -steering -out "${DIR}/runtime/demo_output"
EOF
chmod +x "${PACKAGE_DIR}/runtime/one_click_demo.sh"

pushd "${PACKAGE_DIR}" >/dev/null
if command -v shasum >/dev/null 2>&1; then
  find . -type f -print0 | sort -z | xargs -0 shasum -a 256 > SHA256SUMS
else
  find . -type f -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS
fi
popd >/dev/null

ZIP_PATH="${PACKAGE_DIR}.zip"
rm -f "${ZIP_PATH}"
(
  cd "${DIST_ROOT}"
  zip -qr "$(basename "${ZIP_PATH}")" "$(basename "${PACKAGE_DIR}")"
)

echo "Delivery package prepared: ${PACKAGE_DIR}"
echo "Delivery zip prepared: ${ZIP_PATH}"
