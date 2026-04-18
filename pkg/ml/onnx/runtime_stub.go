//go:build !onnx

package onnx

import (
	"fmt"
)

/**
 * NewNativeInferrer is a stub for the CGO-bound ONNX runtime when built without the 'onnx' tag.
 * It returns an error to force the caller into the MockInferrer fallback logic.
 */
func NewNativeInferrer(modelPath, modelID, version string, inputNames, outputNames []string) (Inferrer, error) {
	return nil, fmt.Errorf("ONNX native runtime not available in this build (missing 'onnx' tag)")
}
