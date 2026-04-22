"""ONNX Export Pipeline for Sovereign SDA Models"""

import torch
import json
from pathlib import Path

class ONNXExporter:
    def __init__(self, output_dir="/Users/adi/idex/sda-train/export/onnx_models"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def export_model(self, model, name, input_shape, opset=17):
        model.eval()
        dummy = torch.randn(*input_shape)
        path = self.output_dir / f"{name}.onnx"
        torch.onnx.export(model, dummy, path, input_names=['input'], 
                         output_names=['output'], opset_version=opset)
        
        meta = {'model_name': name, 'input_shape': list(input_shape), 'opset': opset}
        with open(self.output_dir / f"{name}_meta.json", 'w') as f:
            json.dump(meta, f)
        
        print(f"[ONNX] Exported {name} -> {path}")
        return str(path)
    
    def quantize_dynamic(self, model_name):
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType
            inp = self.output_dir / f"{model_name}.onnx"
            out = self.output_dir / f"{model_name}_int8.onnx"
            if inp.exists():
                quantize_dynamic(str(inp), str(out), weight_type=QuantType.QInt8)
                reduction = (1 - out.stat().st_size / inp.stat().st_size) * 100
                print(f"[ONNX] Quantized {name}: {reduction:.1f}% smaller")
                return str(out)
        except ImportError:
            print("[ONNX] onnxruntime not available")
        return None
    
    def export_all(self, denoiser=None, regressor=None, segmenter=None):
        results = {}
        if denoiser:
            results['denoiser'] = self.export_model(denoiser, 'cosmic_denoiser', (1, 6, 128))
            self.quantize_dynamic('cosmic_denoiser')
        if regressor:
            results['regressor'] = self.export_model(regressor, 'orbital_regressor', (1, 6))
            self.quantize_dynamic('orbital_regressor')
        return results

if __name__ == "__main__":
    print("ONNX Export Pipeline ready. Load models and call export_all()")
