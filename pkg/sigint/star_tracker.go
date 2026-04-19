// Package sigint provides Star Tracker SSA capabilities
// Implements background subtraction, streak detection, and non-catalog object identification
package sigint

import (
	"fmt"
	"math"
	"time"
)

const (
	// Star tracker thresholds
	StreakLengthPixels   = 10.0 // Minimum streak length in pixels
	StreakVelocityPixels = 5.0  // Minimum velocity in pixels/frame
	BackgroundThreshold  = 3.0  // Sigma threshold for background subtraction
	StarCatalogAccuracy  = 0.1  // Arcsecond accuracy for star catalog
	FrameExposureTime    = 0.1  // 100ms exposure time
	FrameRate            = 10.0 // 10 Hz frame rate
)

// StarCatalogEntry represents a star in the reference catalog
type StarCatalogEntry struct {
	RightAscension  float64 // Degrees
	Declination     float64 // Degrees
	Magnitude       float64 // Visual magnitude
	ProperMotionRA  float64 // mas/year
	ProperMotionDec float64 // mas/year
}

// StarField represents the expected star field for a given attitude
type StarField struct {
	Stars        []StarCatalogEntry
	AttitudeQuat [4]float64 // Quaternion [w, x, y, z]
	Timestamp    time.Time
}

// StarFrame represents a captured frame from the star tracker
type StarFrame struct {
	Pixels       [][]float64 // Pixel intensities
	Width        int
	Height       int
	Timestamp    time.Time
	ExposureTime float64
	AttitudeQuat [4]float64
}

// DetectedObject represents an object detected in the star field
type DetectedObject struct {
	CenterX      float64 // Pixel coordinates
	CenterY      float64
	Width        float64
	Height       float64
	Intensity    float64
	IsStreak     bool
	StreakLength float64
	StreakAngle  float64 // Degrees
	VelocityX    float64 // pixels/frame
	VelocityY    float64 // pixels/frame
	IsCatalog    bool
	CatalogID    string
}

// StarTrackerSSA manages star tracker-based SSA
type StarTrackerSSA struct {
	starCatalog     []StarCatalogEntry
	currentField    *StarField
	backgroundModel [][]float64
	detectedObjects []DetectedObject
	trackHistory    map[int][]DetectedObject
}

// NewStarTrackerSSA creates a new star tracker SSA system
func NewStarTrackerSSA() *StarTrackerSSA {
	return &StarTrackerSSA{
		starCatalog:     make([]StarCatalogEntry, 0),
		currentField:    nil,
		backgroundModel: make([][]float64, 0),
		detectedObjects: make([]DetectedObject, 0),
		trackHistory:    make(map[int][]DetectedObject),
	}
}

// LoadStarCatalog loads the reference star catalog
func (st *StarTrackerSSA) LoadStarCatalog(catalog []StarCatalogEntry) {
	st.starCatalog = catalog
}

// GenerateExpectedField generates the expected star field for current attitude
func (st *StarTrackerSSA) GenerateExpectedField(attitude [4]float64, timestamp time.Time) *StarField {
	// Calculate expected star positions based on attitude and catalog
	// In production, use precise star catalog (e.g., Hipparcos, Tycho-2)

	expectedStars := make([]StarCatalogEntry, 0)

	for _, star := range st.starCatalog {
		// Rotate star position by attitude quaternion
		ra, dec := st.rotateByQuaternion(star.RightAscension, star.Declination, attitude)

		// Check if star is in field of view
		if st.isInFieldOfView(ra, dec) {
			expectedStars = append(expectedStars, StarCatalogEntry{
				RightAscension:  ra,
				Declination:     dec,
				Magnitude:       star.Magnitude,
				ProperMotionRA:  star.ProperMotionRA,
				ProperMotionDec: star.ProperMotionDec,
			})
		}
	}

	return &StarField{
		Stars:        expectedStars,
		AttitudeQuat: attitude,
		Timestamp:    timestamp,
	}
}

// rotateByQuaternion rotates celestial coordinates by quaternion
func (st *StarTrackerSSA) rotateByQuaternion(ra, dec float64, quat [4]float64) (float64, float64) {
	// Convert to Cartesian coordinates
	raRad := ra * math.Pi / 180.0
	decRad := dec * math.Pi / 180.0

	x := math.Cos(decRad) * math.Cos(raRad)
	y := math.Cos(decRad) * math.Sin(raRad)
	z := math.Sin(decRad)

	// Apply quaternion rotation (simplified)
	w, qx, qy, qz := quat[0], quat[1], quat[2], quat[3]

	// Rotate vector
	rotatedX := (1-2*qy*qy-2*qz*qz)*x + 2*(qx*qy-qz*w)*y + 2*(qx*qz+qy*w)*z
	rotatedY := 2*(qx*qy+qz*w)*x + (1-2*qx*qx-2*qz*qz)*y + 2*(qy*qz-qx*w)*z
	rotatedZ := 2*(qx*qz-qy*w)*x + 2*(qy*qz+qx*w)*y + (1-2*qx*qx-2*qy*qy)*z

	// Convert back to spherical coordinates
	newDec := math.Asin(rotatedZ / math.Sqrt(rotatedX*rotatedX+rotatedY*rotatedY+rotatedZ*rotatedZ))
	newRa := math.Atan2(rotatedY, rotatedX)

	return newRa * 180.0 / math.Pi, newDec * 180.0 / math.Pi
}

// isInFieldOfView checks if a star is in the field of view
func (st *StarTrackerSSA) isInFieldOfView(ra, dec float64) bool {
	// Simplified: assume 20x20 degree field of view
	fov := 20.0
	return math.Abs(ra) < fov/2 && math.Abs(dec) < fov/2
}

// ProcessFrame processes a captured frame for non-catalog objects
func (st *StarTrackerSSA) ProcessFrame(frame *StarFrame, expectedField *StarField) []DetectedObject {
	// 1. Background subtraction
	subtracted := st.subtractBackground(frame)

	// 2. Detect objects
	objects := st.detectObjects(subtracted)

	// 3. Match against star catalog
	for i := range objects {
		catalogID, isCatalog := st.matchToCatalog(objects[i], expectedField)
		objects[i].CatalogID = catalogID
		objects[i].IsCatalog = isCatalog
	}

	// 4. Identify non-catalog objects
	nonCatalog := make([]DetectedObject, 0)
	for _, obj := range objects {
		if !obj.IsCatalog {
			nonCatalog = append(nonCatalog, obj)
		}
	}

	st.detectedObjects = nonCatalog
	return nonCatalog
}

// subtractBackground subtracts the expected background from the frame
func (st *StarTrackerSSA) subtractBackground(frame *StarFrame) [][]float64 {
	// Calculate mean background level
	mean, stddev := st.calculateBackgroundStats(frame)

	subtracted := make([][]float64, frame.Height)
	for y := 0; y < frame.Height; y++ {
		subtracted[y] = make([]float64, frame.Width)
		for x := 0; x < frame.Width; x++ {
			// Subtract mean and apply threshold
			value := frame.Pixels[y][x] - mean
			if value > BackgroundThreshold*stddev {
				subtracted[y][x] = value
			} else {
				subtracted[y][x] = 0.0
			}
		}
	}

	return subtracted
}

// calculateBackgroundStats calculates mean and standard deviation of background
func (st *StarTrackerSSA) calculateBackgroundStats(frame *StarFrame) (float64, float64) {
	sum := 0.0
	count := 0

	for y := 0; y < frame.Height; y++ {
		for x := 0; x < frame.Width; x++ {
			sum += frame.Pixels[y][x]
			count++
		}
	}

	mean := sum / float64(count)

	variance := 0.0
	for y := 0; y < frame.Height; y++ {
		for x := 0; x < frame.Width; x++ {
			diff := frame.Pixels[y][x] - mean
			variance += diff * diff
		}
	}

	stddev := math.Sqrt(variance / float64(count))

	return mean, stddev
}

// detectObjects identifies bright objects in the frame
func (st *StarTrackerSSA) detectObjects(pixels [][]float64) []DetectedObject {
	objects := make([]DetectedObject, 0)
	visited := make([][]bool, len(pixels))
	for i := range visited {
		visited[i] = make([]bool, len(pixels[0]))
	}

	for y := 0; y < len(pixels); y++ {
		for x := 0; x < len(pixels[0]); x++ {
			if pixels[y][x] > 0 && !visited[y][x] {
				obj := st.growObject(pixels, x, y, visited)
				if obj.Intensity > 0 {
					objects = append(objects, obj)
				}
			}
		}
	}

	// Detect streaks
	for i := range objects {
		objects[i].IsStreak, objects[i].StreakLength, objects[i].StreakAngle = st.detectStreak(objects[i])
	}

	return objects
}

// growObject grows an object region using connected components
func (st *StarTrackerSSA) growObject(pixels [][]float64, startX, startY int, visited [][]bool) DetectedObject {
	// Simplified connected component labeling
	minX, maxX := startX, startX
	minY, maxY := startY, startY
	totalIntensity := 0.0

	// Simple 3x3 neighborhood expansion
	for dy := -1; dy <= 1; dy++ {
		for dx := -1; dx <= 1; dx++ {
			ny := startY + dy
			nx := startX + dx

			if ny >= 0 && ny < len(pixels) && nx >= 0 && nx < len(pixels[0]) {
				if pixels[ny][nx] > 0 {
					if nx < minX {
						minX = nx
					}
					if nx > maxX {
						maxX = nx
					}
					if ny < minY {
						minY = ny
					}
					if ny > maxY {
						maxY = ny
					}
					totalIntensity += pixels[ny][nx]
					visited[ny][nx] = true
				}
			}
		}
	}

	return DetectedObject{
		CenterX:   float64(minX+maxX) / 2.0,
		CenterY:   float64(minY+maxY) / 2.0,
		Width:     float64(maxX - minX + 1),
		Height:    float64(maxY - minY + 1),
		Intensity: totalIntensity,
	}
}

// detectStreak determines if an object is a streak (moving object)
func (st *StarTrackerSSA) detectStreak(obj DetectedObject) (bool, float64, float64) {
	// Calculate aspect ratio
	aspectRatio := obj.Width / obj.Height
	if aspectRatio < 1.0 {
		aspectRatio = 1.0 / aspectRatio
	}

	// Check if elongated (streak)
	if aspectRatio > 2.0 {
		// Calculate streak angle
		angle := math.Atan2(obj.Height, obj.Width) * 180.0 / math.Pi
		length := math.Sqrt(obj.Width*obj.Width + obj.Height*obj.Height)
		return true, length, angle
	}

	return false, 0.0, 0.0
}

// matchToCatalog matches detected object to star catalog
func (st *StarTrackerSSA) matchToCatalog(obj DetectedObject, field *StarField) (string, bool) {
	// Convert pixel coordinates to celestial coordinates
	ra, dec := st.pixelToCelestial(obj.CenterX, obj.CenterY, field.AttitudeQuat)

	// Search for matching star in catalog
	for _, star := range field.Stars {
		// Calculate angular separation
		raDiff := star.RightAscension - ra
		decDiff := star.Declination - dec
		separation := math.Sqrt(raDiff*raDiff + decDiff*decDiff)

		if separation < StarCatalogAccuracy {
			return fmt.Sprintf("HIP_%d", int(star.Magnitude*1000)), true
		}
	}

	return "", false
}

// pixelToCelestial converts pixel coordinates to celestial coordinates
func (st *StarTrackerSSA) pixelToCelestial(x, y float64, quat [4]float64) (float64, float64) {
	// Simplified conversion
	// In production, use precise calibration

	// Convert to normalized coordinates
	normX := (x - 512.0) / 512.0 // Assume 1024x1024 sensor
	normY := (y - 512.0) / 512.0

	// Convert to angular coordinates
	ra := normX * 10.0 // 10 degrees field of view
	dec := normY * 10.0

	return ra, dec
}

// TrackMotion tracks motion of non-catalog objects across frames
func (st *StarTrackerSSA) TrackMotion(frameNumber int) map[int][2]float64 {
	velocities := make(map[int][2]float64)

	for objID, history := range st.trackHistory {
		if len(history) >= 2 {
			// Calculate velocity from last two frames
			last := history[len(history)-1]
			prev := history[len(history)-2]

			vx := last.CenterX - prev.CenterX
			vy := last.CenterY - prev.CenterY

			velocities[objID] = [2]float64{vx, vy}
		}
	}

	return velocities
}

// AnalyzeMotion analyzes motion patterns for threat assessment
func (st *StarTrackerSSA) AnalyzeMotion(velocities map[int][2]float64) []MotionAnalysis {
	analyses := make([]MotionAnalysis, 0)

	for objID, vel := range velocities {
		speed := math.Sqrt(vel[0]*vel[0] + vel[1]*vel[1])

		analysis := MotionAnalysis{
			ObjectID:    objID,
			VelocityX:   vel[0],
			VelocityY:   vel[1],
			Speed:       speed,
			IsManeuver:  speed > StreakVelocityPixels,
			ThreatLevel: st.calculateThreatLevel(speed),
		}

		analyses = append(analyses, analysis)
	}

	return analyses
}

// calculateThreatLevel determines threat level based on motion
func (st *StarTrackerSSA) calculateThreatLevel(speed float64) string {
	if speed > 20.0 {
		return "CRITICAL"
	} else if speed > 10.0 {
		return "HIGH"
	} else if speed > 5.0 {
		return "MEDIUM"
	}
	return "LOW"
}

// MotionAnalysis represents motion analysis for an object
type MotionAnalysis struct {
	ObjectID    int
	VelocityX   float64
	VelocityY   float64
	Speed       float64
	IsManeuver  bool
	ThreatLevel string
}
