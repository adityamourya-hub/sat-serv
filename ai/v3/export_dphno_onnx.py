"""
Export DPHNO model to ONNX format for Go integration

This script exports the trained DPHNO model to ONNX format
so it can be loaded by the Go binding using ONNX Runtime.

Usage:
    python ai/v3/export_dphno_onnx.py --model_path models/dphno.pth --output_path models/dphno.onnx
"""

import argparse
import torch
from dphno import DPHNO


def export_dphno_to_onnx(model_path: str, output_path: str):
    """
    Export DPHNO model to ONNX format
    
    Args:
        model_path: Path to trained PyTorch model (.pth)
        output_path: Output ONNX file path (.onnx)
    """
    # Load model
    model = DPHNO(state_dim=6, hidden_dim=128)
    
    if model_path:
        model.load_state_dict(torch.load(model_path))
        print(f"Loaded model from {model_path}")
    
    model.eval()
    
    # Create dummy inputs
    dummy_state = torch.randn(1, 6)
    dummy_control = torch.randn(1, 1)
    
    # Export to ONNX
    torch.onnx.export(
        model,
        (dummy_state, dummy_control),
        output_path,
        input_names=['state', 'control'],
        output_names=['dxdt'],
        dynamic_axes={
            'state': {0: 'batch'},
            'control': {0: 'batch'},
            'dxdt': {0: 'batch'}
        },
        opset_version=14,
        do_constant_folding=True,
        export_params=True,
        verbose=True
    )
    
    print(f"Exported DPHNO model to {output_path}")
    
    # Verify the exported model
    import onnx
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model validation passed")


def export_hypernetwork_atlas_to_onnx(encoder_path: str, decoder_path: str, 
                                      output_encoder: str, output_decoder: str):
    """
    Export Hypernetwork Atlas components to ONNX
    
    Args:
        encoder_path: Path to trained encoder model
        decoder_path: Path to trained decoder model
        output_encoder: Output ONNX file for encoder
        output_decoder: Output ONNX file for decoder
    """
    from hypernetwork_atlas import AtlasEncoder, AtlasDecoder
    
    # Export encoder
    encoder = AtlasEncoder(state_dim=6, latent_dim=256)
    if encoder_path:
        encoder.load_state_dict(torch.load(encoder_path))
    encoder.eval()
    
    dummy_state = torch.randn(1, 6)
    dummy_atmos = torch.randn(1, 64)
    dummy_vehicle = torch.tensor([0], dtype=torch.long)
    
    torch.onnx.export(
        encoder,
        (dummy_state, dummy_atmos, dummy_vehicle),
        output_encoder,
        input_names=['state', 'atmosphere', 'vehicle_class'],
        output_names=['z'],
        dynamic_axes={
            'state': {0: 'batch'},
            'atmosphere': {0: 'batch'},
            'vehicle_class': {0: 'batch'},
            'z': {0: 'batch'}
        },
        opset_version=14
    )
    
    print(f"Exported Atlas Encoder to {output_encoder}")
    
    # Export decoder
    decoder = AtlasDecoder(latent_dim=256)
    if decoder_path:
        decoder.load_state_dict(torch.load(decoder_path))
    decoder.eval()
    
    dummy_z = torch.randn(1, 256)
    
    torch.onnx.export(
        decoder,
        dummy_z,
        output_decoder,
        input_names=['z'],
        output_names=['W_H', 'W_R', 'W_g'],
        dynamic_axes={
            'z': {0: 'batch'},
            'W_H': {0: 'batch'},
            'W_R': {0: 'batch'},
            'W_g': {0: 'batch'}
        },
        opset_version=14
    )
    
    print(f"Exported Atlas Decoder to {output_decoder}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export DPHNO models to ONNX')
    parser.add_argument('--model_path', type=str, default='models/dphno.pth',
                        help='Path to trained DPHNO model')
    parser.add_argument('--output_path', type=str, default='models/dphno.onnx',
                        help='Output ONNX file path')
    parser.add_argument('--export_atlas', action='store_true',
                        help='Also export Hypernetwork Atlas components')
    parser.add_argument('--encoder_path', type=str, default='models/atlas_encoder.pth',
                        help='Path to trained Atlas encoder')
    parser.add_argument('--decoder_path', type=str, default='models/atlas_decoder.pth',
                        help='Path to trained Atlas decoder')
    parser.add_argument('--output_encoder', type=str, default='models/atlas_encoder.onnx',
                        help='Output ONNX file for encoder')
    parser.add_argument('--output_decoder', type=str, default='models/atlas_decoder.onnx',
                        help='Output ONNX file for decoder')
    
    args = parser.parse_args()
    
    # Export DPHNO
    export_dphno_to_onnx(args.model_path, args.output_path)
    
    # Optionally export Hypernetwork Atlas
    if args.export_atlas:
        export_hypernetwork_atlas_to_onnx(
            args.encoder_path, args.decoder_path,
            args.output_encoder, args.output_decoder
        )
