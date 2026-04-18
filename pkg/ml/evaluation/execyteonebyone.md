**✅ Here is your complete 10-Prompt Set** — expertly crafted, sequential, and ready to paste one by one into Claude, Cursor, or any strong coding agent.

Each prompt is self-contained, references the current codebase state, and builds on the previous ones. They are ordered by highest leverage for PS24 delivery.

---

**Prompt 1/10 – Real ONNX Inference Integration (Highest Priority Fix)**

**PROMPT:**
You are an elite sovereign military SSA engineer. Implement **real ONNX inference** in SSIS using onnxruntime-go so the three existing models actually run in the Go binary.  
Replace all DummyInferenceEngine stubs in `pkg/ml/inference.go` and related files.  
Load the three models from `pkg/ml/models/` (`maneuver_detector.onnx`, `collision_risk.onnx`, `rapid_ranker.onnx`).  
Keep the existing hybrid classical→ML→fallback orchestrator intact.  
Add graceful degradation and structured logging if a model fails to load.  
Deliver complete patches + updated build command + test with `--demo indian-catalogue --full`. Begin now.

---

**Prompt 2/10 – Upgraded Physics Propagator (RK4 + J2/J3/J4 + Drag)**

**PROMPT:**
Upgrade the orbital propagator in `pkg/orbital/` to production-grade accuracy.  
Replace the current 1st-order Euler integrator with a proper **RK4 integrator** including full zonal harmonics (J2 + J3 + J4) and basic exponential atmospheric drag model.  
Update `fusion.go` and `propagate.go`. Ensure UKF + SNC covariance propagation still works correctly.  
Reference standard JGM-3 coefficients.  
Deliver full patches + verification command showing improved covariance realism in the validation report. Begin now.

---

**Prompt 3/10 – Real Local LLM Tactical Command Assistant (Phi-3/Gemma-2)**

**PROMPT:**
Implement a **real local quantized LLM** (Phi-3-mini-4k-instruct or Gemma-2-2B 4-bit ONNX) as the Tactical Command Assistant.  
Use `onnxruntime-go` to load the model from `models/llm/`.  
Add flag `--assistant`. After the full pipeline runs, generate a professional IAF-style natural-language SITREP.  
Keep graceful fallback to template if model is missing.  
Deliver patches for `pkg/assistant/llm.go` + integration in `main.go` + test command `--demo indian-catalogue --full --assistant`. Begin now.

---

**Prompt 4/10 – RL Sensor Steering Agent (PPO Policy ONNX)**

**PROMPT:**
Implement a small PPO policy network for intelligent radar sensor steering.  
The agent takes current catalog uncertainty map + resource constraints and outputs a prioritized list of objects to observe next.  
Export the policy to ONNX and load it in `pkg/rl/sensor_steering.go`.  
Integrate into the demo/assistant output with clear recommendations.  
Reference 2025–2026 AMOS papers on RL for SSA sensor tasking.  
Deliver full package + test command. Begin now.

---

**Prompt 5/10 – Transformer-Enhanced Track Correlation & Intent**

**PROMPT:**
Add a small iTransformer / autocorrelation-attention model for pattern-of-life analysis and MHT augmentation.  
Export to ONNX and integrate into `pkg/catalog/` and `pkg/orbital/`.  
Enhance maneuver intent proxy with learned orbital sequence understanding.  
Deliver patches + updated SITREP that includes transformer-based intent scoring. Begin now.

---

**Prompt 6/10 – Physics-Informed Neural Network (PINN) Refinement**

**PROMPT:**
Implement a small Physics-Informed Neural Network that refines UKF propagation for long arcs or high-drag regimes.  
Export as ONNX and add as optional refinement step in `pkg/orbital/ukf.go`.  
Improve covariance realism and NEES/NIS scores.  
Deliver patches + validation report impact summary. Begin now.

---

**Prompt 7/10 – Strict Fallback Hardening + Structured Logging**

**PROMPT:**
Add strict fallback hardening and structured JSON audit logging across all ML, LLM, and RL paths.  
Every fallback must be logged with domain, reason, confidence, and action taken.  
No silent degradation allowed. Add `--strict` mode that fails fast on any fallback for evaluator mode.  
Deliver patches for orchestrator, assistant, rl, and inference layers. Begin now.

---

**Prompt 8/10 – Expanded Training Simulator + IAF Scenarios**

**PROMPT:**
Expand the training simulator with realistic IAF-specific scenarios (proximity ops, ASAT-style maneuvers, regional threats, re-entry).  
Add 500+ new scenarios to `ai/simulator/` and integrate into scale-validation and demo.  
Deliver updated simulator code + new test scenarios + command to generate training package. Begin now.

---

**Prompt 9/10 – Explainability Layer + Operator Outputs**

**PROMPT:**
Add full explainability to every alert and SITREP.  
For each ML/LLM/RL decision, include "why" reasoning (feature importance, confidence, fallback reason).  
Enhance Tactical Assistant output with explainable sections.  
Deliver patches + updated operator manual section. Begin now.

---

**Prompt 10/10 – Final Professional Packaging & Delivery**

**PROMPT:**
Perform final expert packaging of the entire SSIS codebase for IAF delivery.  
Create clean `dist/` structure, reproducible Makefile, goreleaser config, SBOM, operator manual, hardening matrix, and one-click demo.  
Ensure fully static build where possible and clear instructions for air-gapped deployment.  
Deliver the complete packaging pipeline + final test commands. Begin now.

---

**How to use this 10-prompt set:**

1. Start with **Prompt 1** (Real ONNX) — this is the highest-leverage fix right now.
2. Run one prompt at a time.
3. After each, build and test with:
   ```bash
   cd /Users/adi/idex_iaf/ssis
   go build -o ssa_ops_runner ./cmd/ssa_ops_runner
   ./ssa_ops_runner --demo indian-catalogue --full --assistant
   ```

