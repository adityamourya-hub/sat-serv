// Package handlers provides sovereign volume monitoring API for IAF strategic areas
package handlers

import (
	"net/http"

	"github.com/gin-gonic/gin"
	"ssis/internal/db/repository"
)

// SovereignHandler handles strategic volume monitoring requests
type SovereignHandler struct {
	volumeRepo *repository.SovereignVolumeRepo
}

// NewSovereignHandler creates handler instance
func NewSovereignHandler(volumeRepo *repository.SovereignVolumeRepo) *SovereignHandler {
	return &SovereignHandler{volumeRepo: volumeRepo}
}

// StrategicVolumeResponse represents a strategic monitoring zone
type StrategicVolumeResponse struct {
	Name        string `json:"name"`
	Description string `json:"description"`
	Bounds      struct {
		MinLon   float64 `json:"min_lon"`
		MaxLon   float64 `json:"max_lon"`
		MinLat   float64 `json:"min_lat"`
		MaxLat   float64 `json:"max_lat"`
		MinAltKm float64 `json:"min_alt_km"`
		MaxAltKm float64 `json:"max_alt_km"`
	} `json:"bounds"`
	AlertLevel string `json:"alert_level"`
}

// IntrusionAlert represents an adversary satellite in sovereign volume
type IntrusionAlert struct {
	VolumeName   string    `json:"volume_name"`
	NoradCatID   int32     `json:"norad_cat_id"`
	ObjectName   string    `json:"object_name"`
	CountryCode  string    `json:"country_code"`
	Longitude    float64   `json:"longitude"`
	Latitude     float64   `json:"latitude"`
	AltitudeKm   float64   `json:"altitude_km"`
	Velocity     []float64 `json:"velocity_kms"`
	DetectedAt   string    `json:"detected_at"`
	ThreatLevel  string    `json:"threat_level"`
}

// GetStrategicVolumes handles GET /api/sovereign/volumes
// Returns all IAF strategic monitoring zones
func (h *SovereignHandler) GetStrategicVolumes(c *gin.Context) {
	volumes := repository.StrategicVolumes()

	response := make([]StrategicVolumeResponse, 0, len(volumes))
	for _, v := range volumes {
		svr := StrategicVolumeResponse{
			Name:        v.Name,
			Description: v.Description,
			AlertLevel:  v.AlertLevel,
		}
		svr.Bounds.MinLon = v.MinLon
		svr.Bounds.MaxLon = v.MaxLon
		svr.Bounds.MinLat = v.MinLat
		svr.Bounds.MaxLat = v.MaxLat
		svr.Bounds.MinAltKm = v.MinAltKm
		svr.Bounds.MaxAltKm = v.MaxAltKm

		response = append(response, svr)
	}

	c.JSON(http.StatusOK, gin.H{
		"volumes": response,
		"count":   len(response),
	})
}

// GetVolumeIntrusions handles GET /api/sovereign/intrusions
// Returns all adversary satellites currently in sovereign volumes
func (h *SovereignHandler) GetVolumeIntrusions(c *gin.Context) {
	results, err := h.volumeRepo.ScanAllVolumes(c.Request.Context())
	if err != nil {
		c.JSON(http.StatusInternalServerError, ErrorResponse{
			Error:   "Failed to scan volumes",
			Details: err.Error(),
		})
		return
	}

	// Flatten results into alerts
	var alerts []IntrusionAlert
	totalCount := 0

	for volumeName, intrusions := range results {
		totalCount += len(intrusions)
		for _, i := range intrusions {
			alerts = append(alerts, IntrusionAlert{
				VolumeName:   volumeName,
				NoradCatID:   i.NoradCatID,
				ObjectName:   i.ObjectName,
				CountryCode:  i.CountryCode,
				Longitude:    i.Longitude,
				Latitude:     i.Latitude,
				AltitudeKm:   i.AltitudeKm,
				Velocity:     []float64{i.Velocity[0], i.Velocity[1], i.Velocity[2]},
				DetectedAt:   i.DetectedAt.Format("2006-01-02T15:04:05Z"),
				ThreatLevel:  i.ThreatLevel,
			})
		}
	}

	// Log intrusion detection for audit
	if totalCount > 0 {
		logSovereignAlert(c, totalCount)
	}

	c.JSON(http.StatusOK, gin.H{
		"intrusions": alerts,
		"total_count": totalCount,
		"volumes_scanned": len(results),
	})
}

// GetVolumeIntrusionsByName handles GET /api/sovereign/intrusions/:volume
// Returns intrusions for a specific volume
func (h *SovereignHandler) GetVolumeIntrusionsByName(c *gin.Context) {
	volumeName := c.Param("volume")

	// Find the volume configuration
	var targetVolume *repository.SovereignVolume
	for _, v := range repository.StrategicVolumes() {
		if v.Name == volumeName {
			targetVolume = &v
			break
		}
	}

	if targetVolume == nil {
		c.JSON(http.StatusNotFound, ErrorResponse{
			Error: "Volume not found",
			Details: "Valid volumes: Siliguri_Corridor, Galwan_Valley, IAF_Test_Range, Sriharikota_Range",
		})
		return
	}

	intrusions, err := h.volumeRepo.ScanVolumeForAdversaries(c.Request.Context(), *targetVolume)
	if err != nil {
		c.JSON(http.StatusInternalServerError, ErrorResponse{
			Error:   "Failed to scan volume",
			Details: err.Error(),
		})
		return
	}

	// Convert to response format
	var alerts []IntrusionAlert
	for _, i := range intrusions {
		alerts = append(alerts, IntrusionAlert{
			VolumeName:   i.VolumeName,
			NoradCatID:   i.NoradCatID,
			ObjectName:   i.ObjectName,
			CountryCode:  i.CountryCode,
			Longitude:    i.Longitude,
			Latitude:     i.Latitude,
			AltitudeKm:   i.AltitudeKm,
			Velocity:     []float64{i.Velocity[0], i.Velocity[1], i.Velocity[2]},
			DetectedAt:   i.DetectedAt.Format("2006-01-02T15:04:05Z"),
			ThreatLevel:  i.ThreatLevel,
		})
	}

	c.JSON(http.StatusOK, gin.H{
		"volume":     volumeName,
		"intrusions": alerts,
		"count":      len(alerts),
	})
}

// logSovereignAlert logs intrusion detection for audit (IAF requirement)
func logSovereignAlert(c *gin.Context, count int) {
	clientIP := c.ClientIP()
	userAgent := c.GetHeader("User-Agent")
	_ = clientIP
	_ = userAgent
	_ = count
	// TODO: Write to audit log table
}
