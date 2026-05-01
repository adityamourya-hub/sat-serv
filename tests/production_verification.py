"""
Production Verification Suite — Trained Model Readiness
Tests ONNX model loading, inference shapes, and physics constraints.
"""
import torch, numpy as np, onnxruntime as ort, sys, json, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ai.v3.navic_anomaly_detector import NavICAnomalyDetector
try:
    from ai.v3.neural_density import NeuralAtmosphere
    JAX_AVAILABLE = True
except Exception:
    JAX_AVAILABLE = False

MODEL_DIR = "pkg/ml/models"
RESULTS = {}


def test_maneuver_onnx():
    path = f"{MODEL_DIR}/maneuver_detector.onnx"
    assert os.path.exists(path), f"Missing {path}"
    sess = ort.InferenceSession(path)
    assert len(sess.get_inputs()) == 1
    assert sess.get_inputs()[0].name == "maneuver_window"
    # Test inference with multiple batch sizes
    for bs in [1, 4]:
        dummy = np.random.randn(bs, 10, 2).astype(np.float32)
        out = sess.run(None, {"maneuver_window": dummy})
        assert len(out) == 4
        assert out[0].shape == (bs, 1)  # probability
        assert out[1].shape == (bs, 2)  # type logits
    names = [o.name for o in sess.get_outputs()]
    assert "maneuver_probability" in names
    RESULTS["maneuver_detector"] = "PASS"
    print("[PASS] Maneuver detector ONNX load + inference (dynamic batch)")


def test_collision_onnx():
    path = f"{MODEL_DIR}/collision_risk.onnx"
    assert os.path.exists(path), f"Missing {path}"
    sess = ort.InferenceSession(path)
    assert len(sess.get_inputs()) == 3
    dummy = {
        "nodes": np.random.randn(1, 10, 26).astype(np.float32),
        "adj": np.eye(10, dtype=np.float32)[None, ...],
        "iono": np.random.randn(1, 16).astype(np.float32),
    }
    out = sess.run(None, dummy)
    assert len(out) == 2
    RESULTS["collision_risk"] = "PASS"
    print("[PASS] Collision risk ONNX load + inference")


def test_navic_onnx():
    path = f"{MODEL_DIR}/navic_anomaly_detector.onnx"
    assert os.path.exists(path), f"Missing {path}"
    sess = ort.InferenceSession(path)
    dummy = np.random.randn(1, 60, 12).astype(np.float32)
    out = sess.run(None, {"residual_history": dummy})
    logits, conf = out
    assert logits.shape == (1, 3)
    assert conf.shape == (1, 1)
    RESULTS["navic_anomaly"] = "PASS"
    print("[PASS] NavIC anomaly detector ONNX load + inference")


def test_navic_pytorch_equivalence():
    """Verify ONNX export matches PyTorch model."""
    model = NavICAnomalyDetector()
    model.load_state_dict(
        torch.load("ai/training/models/navic_anomaly_detector_best.pt", map_location="cpu")
    )
    model.eval()
    dummy = torch.randn(1, 60, 12)
    pt_logits, pt_conf = model(dummy)
    sess = ort.InferenceSession(f"{MODEL_DIR}/navic_anomaly_detector.onnx")
    onnx_logits, onnx_conf = sess.run(None, {"residual_history": dummy.numpy()})
    np.testing.assert_allclose(
        pt_logits.detach().numpy(), onnx_logits, rtol=1e-3, atol=1e-4
    )
    np.testing.assert_allclose(
        pt_conf.detach().numpy(), onnx_conf, rtol=1e-3, atol=1e-4
    )
    RESULTS["navic_pytorch_onnx_parity"] = "PASS"
    print("[PASS] NavIC PyTorch/ONNX parity (<1e-3)")


def test_atmosphere_fno():
    """Test Neural Atmosphere JAX model loads and produces atmospheric density."""
    if not JAX_AVAILABLE:
        RESULTS["atmosphere_fno"] = "SKIP (JAX not installed)"
        print("[SKIP] Atmosphere FNO — JAX not available")
        return
    model = NeuralAtmosphere()
    # Test inference with sample inputs
    result = model(alt_km=80.0, lat_deg=26.0, lon_deg=88.5, f107=150.0, ap=15.0, rho_jr=1e-12)
    assert 'rho_total' in result
    assert result['rho_total'] > 0, "Density must be positive"
    RESULTS["atmosphere_fno"] = "PASS"
    print("[PASS] Atmosphere FNO load + positive density")


def test_sensor_tasking_onnx():
    """Test ATeena Sensor Tasking PPO ONNX model."""
    path = f"{MODEL_DIR}/sensor_tasking.onnx"
    assert os.path.exists(path), f"Missing {path}"
    sess = ort.InferenceSession(path)
    inputs = sess.get_inputs()
    outputs = sess.get_outputs()
    assert len(inputs) >= 1, "Expected at least 1 input"
    # Run inference with dummy input matching expected shape
    input_name = inputs[0].name
    input_shape = inputs[0].shape
    # Replace dynamic dims with 1
    shape = [s if isinstance(s, int) and s > 0 else 1 for s in input_shape]
    dummy = np.random.randn(*shape).astype(np.float32)
    out = sess.run(None, {input_name: dummy})
    assert len(out) >= 1, "Expected at least 1 output"
    output_names = [o.name for o in outputs]
    RESULTS["sensor_tasking"] = "PASS"
    print(f"[PASS] Sensor tasking ONNX load + inference ({len(outputs)} outputs: {output_names})")


def test_rapid_ranker_onnx():
    """Test Rapid Ranker neural evidential ONNX model."""
    path = f"{MODEL_DIR}/rapid_ranker.onnx"
    assert os.path.exists(path), f"Missing {path}"
    sess = ort.InferenceSession(path)
    inputs = sess.get_inputs()
    outputs = sess.get_outputs()
    input_name = inputs[0].name
    input_shape = inputs[0].shape
    shape = [s if isinstance(s, int) and s > 0 else 1 for s in input_shape]
    dummy = np.random.randn(*shape).astype(np.float32)
    out = sess.run(None, {input_name: dummy})
    assert len(out) >= 4, f"Expected 4 outputs (score, confidence, unc_low, unc_high), got {len(out)}"
    output_names = [o.name for o in outputs]
    RESULTS["rapid_ranker"] = "PASS"
    print(f"[PASS] Rapid ranker ONNX load + inference (4 outputs: {output_names})")


def test_hnode_onnx():
    """Test Axiom-HNODE Port-Hamiltonian ONNX model."""
    path = f"{MODEL_DIR}/hnode_core.onnx"
    assert os.path.exists(path), f"Missing {path}"
    sess = ort.InferenceSession(path)
    inputs = sess.get_inputs()
    outputs = sess.get_outputs()
    input_name = inputs[0].name
    input_shape = inputs[0].shape
    shape = [s if isinstance(s, int) and s > 0 else 1 for s in input_shape]
    dummy = np.random.randn(*shape).astype(np.float32)
    out = sess.run(None, {input_name: dummy})
    assert len(out) >= 1, "Expected at least 1 output (dynamics)"
    output_names = [o.name for o in outputs]
    RESULTS["axiom_hnode"] = "PASS"
    print(f"[PASS] Axiom-HNODE ONNX load + inference ({output_names})")


def main():
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    print("=" * 60)
    print("SSIS Production Verification Suite")
    print("=" * 60)
    try:
        test_maneuver_onnx()
    except Exception as e:
        RESULTS["maneuver_detector"] = f"FAIL: {e}"
        print(f"[FAIL] Maneuver detector: {e}")
    try:
        test_collision_onnx()
    except Exception as e:
        RESULTS["collision_risk"] = f"FAIL: {e}"
        print(f"[FAIL] Collision risk: {e}")
    try:
        test_navic_onnx()
    except Exception as e:
        RESULTS["navic_anomaly"] = f"FAIL: {e}"
        print(f"[FAIL] NavIC anomaly: {e}")
    try:
        test_navic_pytorch_equivalence()
    except Exception as e:
        RESULTS["navic_pytorch_onnx_parity"] = f"FAIL: {e}"
        print(f"[FAIL] NavIC parity: {e}")
    try:
        test_atmosphere_fno()
    except Exception as e:
        RESULTS["atmosphere_fno"] = f"FAIL: {e}"
        print(f"[FAIL] Atmosphere FNO: {e}")
    try:
        test_sensor_tasking_onnx()
    except Exception as e:
        RESULTS["sensor_tasking"] = f"FAIL: {e}"
        print(f"[FAIL] Sensor tasking: {e}")
    try:
        test_rapid_ranker_onnx()
    except Exception as e:
        RESULTS["rapid_ranker"] = f"FAIL: {e}"
        print(f"[FAIL] Rapid ranker: {e}")
    try:
        test_hnode_onnx()
    except Exception as e:
        RESULTS["axiom_hnode"] = f"FAIL: {e}"
        print(f"[FAIL] Axiom-HNODE: {e}")

    print("=" * 60)
    passed = sum(1 for v in RESULTS.values() if v == "PASS")
    total = len(RESULTS)
    print(f"Results: {passed}/{total} passed")
    print(json.dumps(RESULTS, indent=2))
    with open("tests/verification_results.json", "w") as f:
        json.dump(RESULTS, f, indent=2)
    print("Results written to tests/verification_results.json")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
