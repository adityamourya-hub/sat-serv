# 3D Matched Filtering for Satellite Streak Detection

**Module:** Rapid Processing (i) - IAF   
**Technology:** MPDAF (MUSE Python Data Analysis Framework)  
**Purpose:** Detect faint satellite streaks below single-frame noise floor using 3D convolution

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│ IAF Optical Sensor → Go Ingestor → PostgreSQL (raw_frames)     │
└──────────────────────────────────┬──────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────┐
│ Python Analytics Microservice (ai/v3/)                          │
│ ├─ streak_detection.py (3D Matched Filtering)                   │
│ ├─ Input: [X, Y, Time] cube from sensor frames                  │
│ ├─ Process: Cross-correlation with streak kernel bank           │
│ └─ Output: UCT (Uncorrelated Track) detections → JSON            │
└──────────────────────────────────┬──────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────┐
│ Go Catalog Correlation → Space-Track TLE matching               │
│ → Known object: Update orbit                                    │
│ → Unknown object: New UCT catalog entry                       │
└─────────────────────────────────────────────────────────────────┘
```

## 3D Matched Filtering Theory

### The Convolution Operation
For detecting linear streaks across time-series imagery:

```
y[n] = Σ h[k] * x[n-k]
```

Where:
- `x`: Input image cube [X, Y, Time]
- `h`: Streak kernel template (rotated line + Gaussian PSF)
- `y`: Filtered output (detection map)

### MPDAF Cube Structure
- **Axis 0 (X):** Spatial coordinate (RA/pixels)
- **Axis 1 (Y):** Spatial coordinate (Dec/pixels)  
- **Axis 2 (Time):** Temporal axis (exposure timestamps)

This maps to astronomical datacube where wavelength → time.

## Streak Kernel Bank

36 kernels at 5° angular sampling:
```
Kernel = Rotated_Line(length=15px, width=3px) × Gaussian(PSF)
```

Properties:
- Length: Matched to expected satellite motion
- Width: PSF approximation for optical system
- Rotation: Full 180° coverage (satellites move in straight lines)

## Detection Pipeline

1. **Background Subtraction**
   ```python
   background = cube.median(axis=0)  # Remove static stars
   cube_clean = cube - background
   ```

2. **3D Matched Filtering**
   ```python
   # Parallel convolution across time slices
   mpdaf.tools.loop_ima_multiprocessing(
       convolve_with_kernel_bank,
       time_slices,
       nthreads=MAX_THREADS
   )
   ```

3. **Peak Detection**
   - 5σ threshold for detection confidence
   - Local maxima identification
   - SNR calculation

4. **Temporal Linking**
   - Associate detections across frames
   - Velocity consistency check
   - Minimum 5-frame track length

5. **UCT Export**
   - JSON format for Go catalog ingestion
   - RA/Dec, velocity, confidence metrics

## Usage

### Standalone Detection
```python
from streak_detection import StreakDetector3D

detector = StreakDetector3D(
    kernel_length=15,      # pixels
    kernel_width=3,        # PSF size
    sn_threshold=5.0,      # 5-sigma
    min_streak_length=5    # frames
)

# From IAF optical sensor
detections = detector.detect_streaks(
    image_stack=sensor_frames,  # [N_frames, Y, X]
    timestamps=exposure_times
)

export_to_uct_catalog(detections, "ucts.json")
```

### Go Integration (via PostgreSQL)
```go
// Go service stores sensor frames
query := `
    INSERT INTO sensor_frames (timestamp, image_data, sensor_id)
    VALUES ($1, $2, $3)
`

// Python microservice polls and processes
// Results written to uct_detections table
```

## Performance

**M2 MacBook Air Optimization:**
- Parallel kernel processing (MPDAF multiprocessing)
- FFT-based convolution (scipy.signal.fftconvolve)
- Typical throughput: 30 frames (1024×1024) in <2 seconds

## Detection Capabilities

| Target Type | Detection SNR | Example |
|-------------|---------------|---------|
| LEO satellites (bright) | >20σ | ISS, Starlink |
| GEO satellites (dim) | 5-10σ | Comsats |
| Debris (faint) | 3-5σ | Rocket bodies |
| Hypersonic streaks | >15σ | Sub-orbital trajectories |

## Output: UCT Catalog Format

```json
[
  {
    "norad_cat_id": null,
    "object_name": "UCT_20240115_143022",
    "detection_time": "2024-01-15T14:30:22Z",
    "ra_deg": 125.456,
    "dec_deg": -27.892,
    "velocity_ra": 0.5,
    "velocity_dec": 0.2,
    "peak_sn": 8.5,
    "confidence": 0.85,
    "detection_source": "MPDAF_3D_STREAK",
    "sensor_type": "OPTICAL"
  }
]
```

## Dependencies

```bash
pip install mpdaf astropy numpy scipy
```

**Note:** MPDAF requires scientific Python stack. Install in virtual environment or container for IAF deployment.

## Integration with Go-Native SSA

The Python module operates as an **Analytics Microservice**:

1. **Operational Layer (Go)**
   - Real-time sensor ingestion
   - Frame storage to PostgreSQL
   - UCT catalog correlation

2. **Analytics Layer (Python/MPDAF)**
   - Batch processing of frame sequences
   - Deep signal extraction
   - Hyperspectral analysis (future)

3. **Data Bridge**
   - PostgreSQL `sensor_frames` table
   - JSON UCT export for Go ingestion

## Future Enhancements

- **Hyperspectral Detection:** Use MPDAF Spectrum for material identification
- **PSF Modeling:** Adaptive kernel based on seeing conditions
- **ML Integration:** CNN-based streak detection (YOLO-style)
- **Real-time Stream:** Kafka integration for live processing

## References

- MPDAF Documentation: https://mpdaf.readthedocs.io
- Matched Filtering Theory: DSP in radio astronomy
- Satellite Streak Detection: IAF Optical Sensor Network
