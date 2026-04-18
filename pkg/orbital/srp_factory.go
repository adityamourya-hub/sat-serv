package orbital

import (
	"ssis/pkg/catalog"
	"ssis/pkg/physics"
)

/**
 * ADITI 4.0 SRP Engine Factory
 *
 * Provides intelligent model selection based on target type:
 * - Own Assets (GSAT, INSAT, Cartosat): Hybrid (Box-Wing + ECOM1)
 * - Known Satellites: ECOM1 (5-param)
 * - Unknown/Adversary: ECOM2 (9-param) with BLS fitting
 * - Debris/Dead: Cannonball (simplified)
 */

/**
 * TargetClassification determines SRP model selection.
 */
type TargetClassification int

const (
	OwnAsset         TargetClassification = iota // IAF/ISRO satellites
	KnownCommercial                              // Known commercial GEO
	UnknownAdversary                             // Surveillance targets
	Debris                                       // Dead satellites/debris
	RocketBody                                   // Upper stages
)

/**
 * SatelliteDatabase holds engineering data for known assets.
 * In production, this would query a configuration database.
 */
type SatelliteDatabase struct {
	// Known IAF assets with Box-Wing specs
	KnownAssets map[int]BoxWingModel
}

/**
 * NewSatelliteDatabase creates a database with IAF satellite specs.
 */
func NewSatelliteDatabase() *SatelliteDatabase {
	db := &SatelliteDatabase{
		KnownAssets: make(map[int]BoxWingModel),
	}

	// Populate known IAF assets
	db.registerIAFAssets()

	return db
}

/**
 * registerIAFAssets populates the database with IAF satellite specifications.
 * These are approximate values based on public ISRO documentation.
 */
func (db *SatelliteDatabase) registerIAFAssets() {
	// GSAT series (communication satellites)
	// Typical: 2-ton bus, 2 solar panel wings ~30 m² each
	gsatModel := BoxWingModel{
		BusArea:         [6]float64{8.0, 8.0, 7.0, 7.0, 6.0, 6.0}, // ~2.2m x 2.0m x 3.5m
		PanelArea:       30.0,                                     // 3m x 10m
		NumPanels:       2,
		Mass:            2100.0, // kg (dry)
		Cr:              1.3,    // Typical white MLI
		PanelCr:         1.1,    // Solar cells
		HasAttitudeData: true,
	}

	// Register known GSAT satellites
	gsatIDs := []int{25504, 28903, 32050, 333736, 343113, 393643, 410289}
	for _, id := range gsatIDs {
		db.KnownAssets[id] = gsatModel
	}

	// INSAT series (older generation)
	// Slightly different configuration
	insatModel := BoxWingModel{
		BusArea:         [6]float64{7.0, 7.0, 6.0, 6.0, 5.5, 5.5},
		PanelArea:       25.0,
		NumPanels:       2,
		Mass:            1800.0,
		Cr:              1.3,
		PanelCr:         1.1,
		HasAttitudeData: true,
	}

	insatIDs := []int{234395, 23767, 24714, 25424, 256262, 264982}
	for _, id := range insatIDs {
		db.KnownAssets[id] = insatModel
	}

	// Cartosat series (Earth observation)
	// Smaller, different configuration
	cartosatModel := BoxWingModel{
		BusArea:         [6]float64{4.0, 4.0, 3.5, 3.5, 3.0, 3.0},
		PanelArea:       15.0,
		NumPanels:       2,
		Mass:            650.0,
		Cr:              1.4,
		PanelCr:         1.1,
		HasAttitudeData: true,
	}

	cartosatIDs := []int{286492, 297895, 31135, 327858, 401113}
	for _, id := range cartosatIDs {
		db.KnownAssets[id] = cartosatModel
	}
}

/**
 * ClassifyTarget determines the classification from NORAD ID and data.
 */
func (db *SatelliteDatabase) ClassifyTarget(noradID int, obj catalog.SDAObject) TargetClassification {
	// Check if it's a known IAF asset
	if _, isOwn := db.KnownAssets[noradID]; isOwn {
		return OwnAsset
	}

	// Check if it's Indian by country code
	if obj.CountryCode == "IND" || obj.CountryCode == "IN" {
		return OwnAsset
	}

	// Check if it's known commercial (would query commercial DB)
	// if isKnownCommercial(noradID) {
	//     return KnownCommercial
	// }

	// Check if it's debris (high area-to-mass ratio, no maneuvers)
	// if isDebris(obj) {
	//     return Debris
	// }

	// Default: unknown adversary
	return UnknownAdversary
}

/**
 * GetBoxWingModel retrieves the physical model for a known asset.
 */
func (db *SatelliteDatabase) GetBoxWingModel(noradID int) (BoxWingModel, bool) {
	model, exists := db.KnownAssets[noradID]
	return model, exists
}

/**
 * SRPFactory creates appropriate SRP engines based on target classification.
 */
type SRPFactory struct {
	DB            *SatelliteDatabase
	Estimator     *BLSEstimator
	UseEstimation bool
}

/**
 * NewSRPFactory creates a factory with default configuration.
 */
func NewSRPFactory() *SRPFactory {
	return &SRPFactory{
		DB:            NewSatelliteDatabase(),
		Estimator:     NewBLSEstimator(),
		UseEstimation: true,
	}
}

/**
 * CreateEngine builds the appropriate SRP engine for a target.
 *
 * @param noradID: Satellite NORAD catalog number
 * @param obj: Catalog object data
 * @param tleHistory: Optional TLE history for BLS fitting
 * @return: Configured SRPEngine
 */
func (f *SRPFactory) CreateEngine(
	noradID int,
	obj catalog.SDAObject,
	tleHistory []*physics.TLE,
) *SRPEngine {

	classification := f.DB.ClassifyTarget(noradID, obj)

	switch classification {
	case OwnAsset:
		return f.createOwnAssetEngine(noradID, tleHistory)

	case KnownCommercial:
		return f.createKnownCommercialEngine(noradID, tleHistory)

	case UnknownAdversary:
		return f.createAdversaryEngine(tleHistory)

	case Debris:
		return f.createDebrisEngine()

	case RocketBody:
		return f.createRocketBodyEngine()

	default:
		return f.createDefaultEngine()
	}
}

/**
 * createOwnAssetEngine builds Hybrid model (Box-Wing + ECOM1) for IAF satellites.
 */
func (f *SRPFactory) createOwnAssetEngine(noradID int, tleHistory []*physics.TLE) *SRPEngine {
	// Get physical model
	boxWing, exists := f.DB.GetBoxWingModel(noradID)
	if !exists {
		// Fallback to default
		boxWing = NewStandardGEOSat(2000.0)
	}

	// Fit ECOM1 residuals if history available
	var residuals ECOM1Params
	if len(tleHistory) >= 5 && f.UseEstimation {
		// Use quick estimation for real-time
		residuals = f.estimateECOM1Residuals(tleHistory)
	} else {
		residuals = DefaultECOM1Params()
		// Scale down for residuals only
		residuals.D0 *= 0.1
		residuals.Y0 *= 0.1
		residuals.B0 *= 0.1
	}

	engine := NewSRPEngine(SRPHybrid)
	engine.SetHybrid(boxWing, residuals)
	engine.UseConicalShadow = true

	return engine
}

/**
 * createKnownCommercialEngine builds ECOM1 for known commercial sats.
 */
func (f *SRPFactory) createKnownCommercialEngine(noradID int, tleHistory []*physics.TLE) *SRPEngine {
	var params ECOM1Params

	if len(tleHistory) >= 5 && f.UseEstimation {
		// Quick estimation
		ecom2, _ := QuickEstimateECOM2(tleHistory)
		params = ECOM2ToECOM1(ecom2)
	} else {
		params = DefaultECOM1Params()
	}

	engine := NewSRPEngine(SRPECOM1)
	engine.SetECOM1(params)
	engine.UseConicalShadow = true

	return engine
}

/**
 * createAdversaryEngine builds ECOM2 (9-parameter) for surveillance targets.
 */
func (f *SRPFactory) createAdversaryEngine(tleHistory []*physics.TLE) *SRPEngine {
	var params ECOM2Params

	if len(tleHistory) >= 7 && f.UseEstimation {
		// Full BLS estimation for high-value targets
		result, err := f.Estimator.EstimateECOM2(tleHistory, DefaultECOM2Params())
		if err == nil && result.Converged {
			params = result.ECOM2Params
		} else {
			// Fallback to quick estimate
			params, _ = QuickEstimateECOM2(tleHistory)
		}
	} else if len(tleHistory) >= 5 {
		// Quick estimation
		params, _ = QuickEstimateECOM2(tleHistory)
	} else {
		// Default guess based on typical GEO
		params = DefaultECOM2Params()
	}

	engine := NewSRPEngine(SRPECOM2)
	engine.SetECOM2(params)
	engine.UseConicalShadow = true

	return engine
}

/**
 * createDebrisEngine builds simplified model for dead satellites.
 */
func (f *SRPFactory) createDebrisEngine() *SRPEngine {
	// Debris: use cannonball with high Cr (often metallic)
	engine := NewSRPEngine(SRPCannonball)
	// No additional configuration needed
	return engine
}

/**
 * createRocketBodyEngine builds model for upper stages.
 */
func (f *SRPFactory) createRocketBodyEngine() *SRPEngine {
	// Rocket bodies: typically cylindrical, use ECOM1
	params := DefaultECOM1Params()
	params.D0 = 2.0e-9 // Higher A/m ratio

	engine := NewSRPEngine(SRPECOM1)
	engine.SetECOM1(params)
	engine.UseConicalShadow = true

	return engine
}

/**
 * createDefaultEngine builds standard model for unknown cases.
 */
func (f *SRPFactory) createDefaultEngine() *SRPEngine {
	engine := NewSRPEngine(SRPECOM1)
	engine.SetECOM1(DefaultECOM1Params())
	engine.UseConicalShadow = true
	return engine
}

/**
 * estimateECOM1Residuals fits ECOM1 parameters to TLE history.
 * Simplified version for hybrid model residuals.
 */
func (f *SRPFactory) estimateECOM1Residuals(tleHistory []*physics.TLE) ECOM1Params {
	if len(tleHistory) < 5 {
		return DefaultECOM1Params()
	}

	// Use quick estimate
	ecom2, err := QuickEstimateECOM2(tleHistory)
	if err != nil {
		return DefaultECOM1Params()
	}

	// Convert to ECOM1 (take only main terms)
	return ECOM2ToECOM1(ecom2)
}

/**
 * ECOM2ToECOM1 converts 9-parameter to 5-parameter model.
 */
func ECOM2ToECOM1(ecom2 ECOM2Params) ECOM1Params {
	return ECOM1Params{
		D0: ecom2.D0,
		Y0: ecom2.Y0,
		B0: ecom2.B0,
		Bc: ecom2.Bc,
		Bs: ecom2.Bs,
	}
}

/**
 * ECOM1ToECOM2 converts 5-parameter to 9-parameter model.
 * Periodic terms are set to small defaults.
 */
func ECOM1ToECOM2(ecom1 ECOM1Params) ECOM2Params {
	return ECOM2Params{
		D0:  ecom1.D0,
		D2c: 0.1 * ecom1.D0, // 10% of D0
		D2s: 0.05 * ecom1.D0,
		D4c: 0.02 * ecom1.D0,
		D4s: 0.01 * ecom1.D0,
		Y0:  ecom1.Y0,
		B0:  ecom1.B0,
		Bc:  ecom1.Bc,
		Bs:  ecom1.Bs,
	}
}

/**
 * GetModelDescription returns human-readable description of SRP model.
 */
func GetModelDescription(modelType SRPModelType) string {
	switch modelType {
	case SRPCannonball:
		return "Cannonball (simplified sphere)"
	case SRPBoxWing:
		return "Box-Wing (physical model)"
	case SRPECOM1:
		return "ECOM1 (5-parameter empirical)"
	case SRPECOM2:
		return "ECOM2 (9-parameter empirical)"
	case SRPHybrid:
		return "Hybrid (Box-Wing + ECOM1 residuals)"
	default:
		return "Unknown"
	}
}

/**
 * GetTierDescription returns strategic description.
 */
func GetTierDescription(classification TargetClassification) string {
	switch classification {
	case OwnAsset:
		return "Tier 1: Own Assets - Hybrid (Box-Wing + ECOM1) for maximum precision"
	case KnownCommercial:
		return "Tier 1: Known Commercial - ECOM1 with fitted parameters"
	case UnknownAdversary:
		return "Tier 2: Unknown/Adversary - ECOM2 (9-param) with BLS estimation"
	case Debris:
		return "Tier 3: Debris - Cannonball (sufficient accuracy)"
	case RocketBody:
		return "Tier 3: Rocket Body - Simplified ECOM1"
	default:
		return "Default: ECOM1"
	}
}
