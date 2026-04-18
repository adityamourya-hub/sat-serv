package data

import (
	"fmt"
	"math"
	"math/rand"
	"ssis/pkg/orbital"
	"ssis/pkg/physics"
	"time"
)

func dayOfYearFraction(t time.Time) float64 {
	start := time.Date(t.Year(), t.Month(), t.Day(), 0, 0, 0, 0, t.Location())
	frac := t.Sub(start).Seconds() / 86400.0
	return float64(t.YearDay()) + frac
}

/**
 * SyntheticDataGenerator generates realistic training data via TLE + J2 propagation + noise.
 * Hardened to use the physics substrate schema for ADITI 4.0 data fidelity.
 */
type SyntheticDataGenerator struct {
	config   DataConfig
	rng      *rand.Rand
	registry *VersionRegistry
	tles     []*physics.TLE
	samples  []Sample
	version  DataVersion
}

/**
 * NewSyntheticDataGenerator creates a new data generator with the given configuration.
 */
func NewSyntheticDataGenerator(config DataConfig) *SyntheticDataGenerator {
	rng := rand.New(rand.NewSource(config.Seed))
	return &SyntheticDataGenerator{
		config:   config,
		rng:      rng,
		registry: NewVersionRegistry(),
		samples:  make([]Sample, 0),
	}
}

/**
 * GenerateTLEs creates random but realistic TLE samples.
 */
func (g *SyntheticDataGenerator) GenerateTLEs(count int) ([]*physics.TLE, error) {
	fmt.Printf("[TRC] Entering GenerateTLEs for count=%d\n", count)
	defer fmt.Printf("[TRC] Exiting GenerateTLEs\n")

	tles := make([]*physics.TLE, 0, count)

	for i := 0; i < count; i++ {
		inclination := 30.0 + g.rng.Float64()*120.0  // 30-150 degrees
		raan := g.rng.Float64() * 360.0              // 0-360 degrees
		argp := g.rng.Float64() * 360.0              // 0-360 degrees
		ma := g.rng.Float64() * 360.0                // 0-360 degrees
		eccentricity := g.rng.Float64()*0.15 + 0.001 // 0.001-0.151

		altitudeKm := 400.0 + g.rng.Float64()*1600.0 // 400-2000 km altitude
		ae := physics.EARTH_R_EQUATORIAL
		a := ae + altitudeKm
		n := 2.0 * math.Pi / (math.Sqrt(a*a*a/physics.EARTH_MU_IAU) * 60.0)
		n = n * 1440.0 / (2.0 * math.Pi) // Convert to revolutions per day

		epoch := time.Now().Add(-time.Duration(g.rng.Intn(365*24)) * time.Hour)
		twoDigitYear := epoch.Year() % 100
		epochDay := dayOfYearFraction(epoch)
		satNum := 40000 + i

		line1 := fmt.Sprintf("1 %05dU 24001A   %02d%012.8f  .00000000  00000-0  00000-0 0  0000", satNum, twoDigitYear, epochDay)
		eccStr := fmt.Sprintf("%07d", int(eccentricity*1e7))
		line2 := fmt.Sprintf("2 %05d %8.4f %8.4f %s %8.4f %8.4f %11.8f00000", satNum, inclination, raan, eccStr, argp, ma, n)

		tle := &physics.TLE{
			SatelliteNumber: satNum,
			EpochYear:       2000 + twoDigitYear,
			EpochDay:        epochDay,
			Inclination:     inclination,
			RAAN:            raan,
			Eccentricity:    eccentricity,
			ArgOfPerigee:    argp,
			MeanAnomaly:     ma,
			MeanMotion:      n,
			RawL1:           line1,
			RawL2:           line2,
		}
		tles = append(tles, tle)
	}

	g.tles = tles
	return tles, nil
}

/**
 * GenerateDataset produces a complete training dataset.
 */
func (g *SyntheticDataGenerator) GenerateDataset() ([]Sample, error) {
	fmt.Printf("[TRC] Entering GenerateDataset\n")
	defer fmt.Printf("[TRC] Exiting GenerateDataset\n")

	if len(g.tles) == 0 {
		return nil, fmt.Errorf("no TLEs generated; call GenerateTLEs first")
	}

	samples := make([]Sample, 0)

	for tleIdx, tle := range g.tles {
		prop, err := orbital.NewPropagator(tle)
		if err != nil {
			fmt.Printf("WARNING: Failed to create propagator for TLE %d: %v\n", tleIdx, err)
			continue
		}

		for sampleIdx := 0; sampleIdx < g.config.SamplesPerTLE; sampleIdx++ {
			minuteOffset := float64(sampleIdx) * g.config.SampleIntervalMin
			state, err := prop.Propagate(minuteOffset)
			if err != nil {
				continue
			}

			// Add observation noise
			obsX := state.Pos[0] + g.rng.NormFloat64()*g.config.NoiseSTDKm
			obsY := state.Pos[1] + g.rng.NormFloat64()*g.config.NoiseSTDKm
			obsZ := state.Pos[2] + g.rng.NormFloat64()*g.config.NoiseSTDKm

			obsVX := state.Vel[0] + g.rng.NormFloat64()*g.config.NoiseSTDVel
			obsVY := state.Vel[1] + g.rng.NormFloat64()*g.config.NoiseSTDVel
			obsVZ := state.Vel[2] + g.rng.NormFloat64()*g.config.NoiseSTDVel

			trackLost := g.rng.Float64() < g.config.TrackingErrorRate
			if trackLost {
				obsX += g.rng.NormFloat64() * g.config.NoiseSTDKm * 100.0 
				obsY += g.rng.NormFloat64() * g.config.NoiseSTDKm * 100.0
				obsZ += g.rng.NormFloat64() * g.config.NoiseSTDKm * 100.0
			}

			residualX := obsX - state.Pos[0]
			residualY := obsY - state.Pos[1]
			residualZ := obsZ - state.Pos[2]

			covX := g.config.NoiseSTDKm * g.config.NoiseSTDKm * g.config.CovarianceScale
			covY := g.config.NoiseSTDKm * g.config.NoiseSTDKm * g.config.CovarianceScale
			covZ := g.config.NoiseSTDKm * g.config.NoiseSTDKm * g.config.CovarianceScale
			covVX := g.config.NoiseSTDVel * g.config.NoiseSTDVel * g.config.CovarianceScale
			covVY := g.config.NoiseSTDVel * g.config.NoiseSTDVel * g.config.CovarianceScale
			covVZ := g.config.NoiseSTDVel * g.config.NoiseSTDVel * g.config.CovarianceScale

			risk := g.computeRiskProxy(state)
			threatClass := g.classifyThreat(risk)
			confidence := 0.8 + g.rng.Float64()*0.2

			epochSec := int64((minuteOffset + float64(sampleIdx)) * 60.0)
			sample := Sample{
				ID:            fmt.Sprintf("s-%d-%d-%d", tleIdx, sampleIdx, g.rng.Int63()),
				TLEID:         fmt.Sprintf("tle-%d", tleIdx),
				EpochUnix:     time.Now().Unix() + epochSec,
				PropagateMins: minuteOffset,

				TrueX:  state.Pos[0],
				TrueY:  state.Pos[1],
				TrueZ:  state.Pos[2],
				TrueVX: state.Vel[0],
				TrueVY: state.Vel[1],
				TrueVZ: state.Vel[2],

				ObsX:  obsX,
				ObsY:  obsY,
				ObsZ:  obsZ,
				ObsVX: obsVX,
				ObsVY: obsVY,
				ObsVZ: obsVZ,

				ResidualX: residualX,
				ResidualY: residualY,
				ResidualZ: residualZ,
				CovX:      covX,
				CovY:      covY,
				CovZ:      covZ,
				CovVX:     covVX,
				CovVY:     covVY,
				CovVZ:     covVZ,

				Risk:        risk,
				Confidence:  confidence,
				ThreatClass: threatClass,
			}
			samples = append(samples, sample)
		}
	}

	g.assignSplits(samples)
	g.samples = samples

	g.version = DataVersion{
		ID:               GenerateVersionID(g.config.Purpose, g.config.Model),
		Name:             fmt.Sprintf("%s-data-%s", g.config.Model, time.Now().Format("20060102-150405")),
		GeneratedAt:      time.Now(),
		Purpose:          g.config.Purpose,
		Model:            g.config.Model,
		SampleCount:      int64(len(samples)),
		TLECount:         len(g.tles),
		PropagationSteps: g.config.PropagationSteps,
		NoiseSTD:         g.config.NoiseSTDKm,
		CovarianceScale:  g.config.CovarianceScale,
		Description:      g.config.Description,
		Splits: SplitInfo{
			TrainFraction: g.config.TrainFraction,
			ValFraction:   g.config.ValFraction,
			TestFraction:  g.config.TestFraction,
		},
	}

	trainCount := int64(float64(len(samples)) * g.config.TrainFraction)
	valCount := int64(float64(len(samples)) * g.config.ValFraction)
	testCount := int64(len(samples)) - trainCount - valCount

	g.version.Splits.TrainCount = trainCount
	g.version.Splits.ValCount = valCount
	g.version.Splits.TestCount = testCount

	return samples, nil
}

/**
 * assignSplits assigns train/val/test splits to samples deterministically.
 */
func (g *SyntheticDataGenerator) assignSplits(samples []Sample) {
	trainCount := int64(float64(len(samples)) * g.config.TrainFraction)
	valCount := int64(float64(len(samples)) * g.config.ValFraction)

	for i := range samples {
		idx := int64(i)
		if idx < trainCount {
			samples[i].Split = "train"
		} else if idx < trainCount+valCount {
			samples[i].Split = "val"
		} else {
			samples[i].Split = "test"
		}
	}
}

/**
 * computeRiskProxy computes a synthetic threat risk score based on altitude.
 */
func (g *SyntheticDataGenerator) computeRiskProxy(state physics.State) float64 {
	rMag := math.Sqrt(state.Pos[0]*state.Pos[0] + state.Pos[1]*state.Pos[1] + state.Pos[2]*state.Pos[2])
	altitudeKm := rMag - physics.EARTH_R_EQUATORIAL

	risk := 1.0 - math.Min(1.0, altitudeKm/2000.0)
	return math.Min(1.0, math.Max(0.0, risk+g.rng.NormFloat64()*0.1))
}

/**
 * classifyThreat assigns threat class based on risk score.
 */
func (g *SyntheticDataGenerator) classifyThreat(risk float64) string {
	if risk > 0.7 {
		return "high"
	}
	if risk > 0.4 {
		return "medium"
	}
	return "low"
}

func (g *SyntheticDataGenerator) Version() DataVersion {
	return g.version
}

func (g *SyntheticDataGenerator) Samples() []Sample {
	return g.samples
}
