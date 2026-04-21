# Full DSP Pipeline Integration Plan

This plan implements a complete analog→digital→DSP preprocessing chain for sensor inputs, verifies integration of recent sensor improvements (DPHNO propagation, VSC, Fractional Atmosphere), and creates a unified preprocessing framework supporting optical, RF, IR, and Lidar sensors.

## Phase 1: Verify Sensor Improvements Integration

### Tasks
1. **Check DPHNO-based Propagation**
   - Verify `PropagateSymplectic()` is called from `Propagate()`
   - Confirm `orbitalPHSModel` implements `physics.PHSModel` interface
   - Test Hamiltonian computation and storage in `TrackedObject`
   - Validate fractional Laplacian corrections for HGV objects

2. **Check VSC (Verification Symbolic Certificate)**
   - Verify `GenerateObservationReport()` creates VSC with Hamiltonian hash
   - Confirm `VerifyPhysicsConsistency()` grades A-D with self-invalidation
   - Test bootstrap CI computation from covariance matrix
   - Validate falsifiable 60s predictions

3. **Check Fractional Atmosphere**
   - Verify `FractionalAtmosphere` extends `StandardAtmosphere`
   - Confirm spectral multiplier computation for wake effects
   - Test HGV corridor (30-80km) detection and correction
   - Validate integration with `computeOpticalPoD()` and `computeIRPoD()`

### Files to Check
- `/Users/adi/idex/pkg/sensors/information_theoretic_tasking.go` - DPHNO propagation
- `/Users/adi/idex/pkg/sensors/space_based.go` - VSC and ObservationReport
- `/Users/adi/idex/pkg/sensors/pod.go` - Fractional atmosphere
- `/Users/adi/idex/pkg/physics/symplectic_engine.go` - PHS interface

## Phase 2: Design DSP Pipeline Architecture

### Architecture Overview
```
Raw Sensor Input → AnalogPreprocessor → DigitalPreprocessor → DSPPreprocessor → Processed Data
                     (Filter/AGC)        (Sample/Quantize)      (FFT/Filter/Extract)           ↓
                                                                                      ObservationReport
```

### Core Interfaces
1. **SensorPreprocessor** - Unified interface for all preprocessing stages
   - `Process(input []float64) ([]float64, error)`
   - `Reset()`
   - `GetConfig() PreprocessorConfig`

2. **AnalogPreprocessor** - Simulates analog front-end
   - Anti-aliasing filter (Butterworth/Chebyshev)
   - Automatic Gain Control (AGC)
   - Noise figure calculation
   - Saturation detection

3. **DigitalPreprocessor** - ADC simulation
   - Sampling rate conversion
   - Quantization (8/12/16/24-bit)
   - Dithering (TPDF, Gaussian)
   - Clipping detection

4. **DSPPreprocessor** - Digital signal processing
   - FFT/IFFT (using `physics.FFTProcessor`)
   - FIR/IIR filtering (Kalman, matched filter)
   - Wavelet denoising
   - Feature extraction (SNR, centroid, spectral peaks)

### Sensor-Specific Pipelines
- **Optical**: Speckle correction → Centroid extraction → Photometry
- **RF**: IQ demodulation → FFT → Spectral analysis → Modulation ID
- **IR**: Dark current subtraction → Flat fielding → Bad pixel correction
- **Lidar**: Range gating → Pulse shaping → Background subtraction

## Phase 3: Implement AnalogPreprocessor

### New File: `/Users/adi/idex/pkg/sensors/analog_preprocessor.go`

### Components
1. **AnalogConfig**
   - Cutoff frequency for anti-aliasing
   - Filter order (2nd/4th/8th)
   - AGC attack/release time
   - Noise figure

2. **AntiAliasingFilter**
   - Butterworth low-pass implementation
   - Configurable order and cutoff
   - Phase response analysis

3. **AGCController**
   - Logarithmic AGC
   - Attack/release parameters
   - Saturation protection

4. **NoiseModel**
   - Thermal noise (kTB)
   - Shot noise
   - 1/f noise

## Phase 4: Implement DigitalPreprocessor

### New File: `/Users/adi/idex/pkg/sensors/digital_preprocessor.go`

### Components
1. **ADCConfig**
   - Sampling rate (Hz)
   - Bit depth (8/12/16/24)
   - Dithering type (none/TPDF/Gaussian)
   - Full-scale voltage

2. **Sampler**
   - Resampling with interpolation
   - Decimation/interpolation
   - Anti-aliasing for resampling

3. **Quantizer**
   - Uniform quantization
   - Dithering injection
   - SQNR calculation

4. **ClippingDetector**
   - Peak detection
   - Clipping flag
   - Dynamic range analysis

## Phase 5: Implement DSPPreprocessor

### New File: `/Users/adi/idex/pkg/sensors/dsp_preprocessor.go`

### Components
1. **DSPConfig**
   - FFT size (256/512/1024/2048)
   - Window function (Hamming/Hanning/Blackman)
   - Filter type (FIR/IIR/Kalman)
   - Denoising method (wavelet/median)

2. **FFTProcessor** (wrapper around `physics.FFTProcessor`)
   - Power spectrum computation
   - Spectrogram generation
   - Peak detection

3. **FilterBank**
   - FIR filter design (window method)
   - IIR filter (Butterworth/Chebyshev)
   - Kalman filter implementation
   - Matched filter for known signals

4. **FeatureExtractor**
   - SNR estimation
   - Centroid calculation (for optical)
   - Spectral peak detection
   - Modulation classification (for RF)
   - Pulse width (for Lidar)

5. **WaveletDenoiser**
   - Daubechies wavelets
   - Threshold selection (universal/adaptive)
   - Multi-level decomposition

## Phase 6: Integration with SensorPlatform

### Modifications to `/Users/adi/idex/pkg/sensors/space_based.go`

1. **Add Preprocessor Chain to SensorPayload**
   ```go
   type SensorPayload struct {
       // ... existing fields ...
       PreprocessorChain []SensorPreprocessor
       AnalogPreprocessor *AnalogPreprocessor
       DigitalPreprocessor *DigitalPreprocessor
       DSPPreprocessor *DSPPreprocessor
   }
   ```

2. **Integrate into Observation Flow**
   - Add `PreprocessRawData()` method to SensorPlatform
   - Call preprocessing before `ComputeObservationGeometry()`
   - Include preprocessing metadata in `ObservationReport`

3. **Update ObservationReport**
   - Add `PreprocessingChain` field with stage-by-stage metrics
   - Include SNR, clipping, quantization error
   - Store raw vs processed data comparison

### Integration Points
- **Optical**: Integrate with existing `SpecklePreprocessor` in `command_bridge.go`
- **RF**: Integrate with existing `RFProcessor` in `sigint/rf_processor.go`
- **General**: Use `physics.FFTProcessor` from `physics/fft.go`

## Phase 7: Testing and Validation

### Test Coverage
1. **Unit Tests**
   - Each preprocessor stage independently
   - Filter response verification
   - Quantization accuracy
   - FFT correctness

2. **Integration Tests**
   - Full pipeline end-to-end
   - Multi-sensor scenarios
   - Performance benchmarks

3. **Physics Validation**
   - Energy conservation through pipeline
   - Noise figure consistency
   - Hamiltonian drift verification

### Benchmark Targets
- Analog preprocessing: < 10μs per sample
- Digital preprocessing: < 50μs per sample
- DSP preprocessing: < 100μs per FFT
- Total pipeline latency: < 1ms (for <100ms Go-reflex)

## Dependencies
- Existing: `physics.FFTProcessor`, `sigint.RFProcessor`, `telescope.SpecklePreprocessor`
- New: No external dependencies (pure Go implementation)

## Success Criteria
1. All sensor improvements (DPHNO, VSC, Fractional Atmosphere) verified and integrated
2. Full analog→digital→DSP pipeline implemented for all sensor types
3. Pipeline latency < 1ms per sample
4. Integration with `ObservationReport` with preprocessing metadata
5. Unit tests with >80% coverage
6. Benchmark targets met
