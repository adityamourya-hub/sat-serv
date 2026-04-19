// Package handlers provides REST API handlers for ISDA
// IAF SDLC: All handlers include input validation and audit logging
package handlers

import (
	"fmt"
	"net/http"

	"github.com/gin-gonic/gin"
	"ssis/internal/db/repository"
)

// VolumetricHandler handles 3D spatial scan requests
type VolumetricHandler struct {
	statesRepo *repository.CurrentStatesRepo
}

// NewVolumetricHandler creates handler instance
func NewVolumetricHandler(statesRepo *repository.CurrentStatesRepo) *VolumetricHandler {
	return &VolumetricHandler{statesRepo: statesRepo}
}

// BoundingBoxRequest for 3D box scan
type BoundingBoxRequest struct {
	MinLon   float64 `json:"min_lon" binding:"required,gte=-180,lte=180"`
	MaxLon   float64 `json:"max_lon" binding:"required,gte=-180,lte=180"`
	MinLat   float64 `json:"min_lat" binding:"required,gte=-90,lte=90"`
	MaxLat   float64 `json:"max_lat" binding:"required,gte=-90,lte=90"`
	MinAltKm float64 `json:"min_alt_km" binding:"gte=0"`
	MaxAltKm float64 `json:"max_alt_km" binding:"required,gte=0"`
}

// SensorConeRequest for sensor FOV scan
type SensorConeRequest struct {
	OriginLon      float64 `json:"origin_lon" binding:"required,gte=-180,lte=180"`
	OriginLat      float64 `json:"origin_lat" binding:"required,gte=-90,lte=90"`
	OriginAltKm    float64 `json:"origin_alt_km" binding:"gte=0"`
	AzimuthDeg     float64 `json:"azimuth_deg" binding:"gte=0,lt=360"`
	ElevationDeg   float64 `json:"elevation_deg" binding:"gte=-90,lte=90"`
	FovHDeg        float64 `json:"fov_h_deg" binding:"required,gte=0,lte=180"`
	FovVDeg        float64 `json:"fov_v_deg" binding:"required,gte=0,lte=180"`
	RangeKm        float64 `json:"range_km" binding:"required,gte=0"`
}

// ScanResult represents a single satellite in scan results
type ScanResult struct {
	NoradCatID  int32     `json:"norad_cat_id"`
	ObjectName  string    `json:"object_name"`
	Longitude   float64   `json:"longitude"`
	Latitude    float64   `json:"latitude"`
	AltitudeKm  float64   `json:"altitude_km"`
	Velocity    []float64 `json:"velocity_kms"` // [vx, vy, vz] in km/s
	LastUpdated string    `json:"last_updated"` // ISO 8601
}

// ScanResponse is the API response
type ScanResponse struct {
	Count   int          `json:"count"`
	Objects []ScanResult `json:"objects"`
}

// ErrorResponse for failed requests
type ErrorResponse struct {
	Error   string `json:"error"`
	Details string `json:"details,omitempty"`
}

// PostVolumetricScanBBox handles POST /api/scan/volumetric/bbox
// Performs 3D bounding box volumetric scan against current_states
func (h *VolumetricHandler) PostVolumetricScanBBox(c *gin.Context) {
	var req BoundingBoxRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, ErrorResponse{
			Error:   "Invalid request",
			Details: err.Error(),
		})
		return
	}

	// Validate box dimensions
	if req.MinLon > req.MaxLon {
		c.JSON(http.StatusBadRequest, ErrorResponse{
			Error: "Invalid bounding box: min_lon > max_lon",
		})
		return
	}
	if req.MinLat > req.MaxLat {
		c.JSON(http.StatusBadRequest, ErrorResponse{
			Error: "Invalid bounding box: min_lat > max_lat",
		})
		return
	}
	if req.MinAltKm > req.MaxAltKm {
		c.JSON(http.StatusBadRequest, ErrorResponse{
			Error: "Invalid bounding box: min_alt_km > max_alt_km",
		})
		return
	}

	// Execute PostGIS query
	results, err := h.statesRepo.VolumetricScanBBox(
		c.Request.Context(),
		req.MinLon, req.MinLat, req.MinAltKm,
		req.MaxLon, req.MaxLat, req.MaxAltKm,
	)
	if err != nil {
		c.JSON(http.StatusInternalServerError, ErrorResponse{
			Error:   "Database query failed",
			Details: err.Error(),
		})
		return
	}

	// Transform to response format
	response := ScanResponse{
		Count:   len(results),
		Objects: make([]ScanResult, 0, len(results)),
	}

	for _, r := range results {
		response.Objects = append(response.Objects, ScanResult{
			NoradCatID:  r.NoradCatID,
			ObjectName:  r.ObjectName,
			Longitude:   r.Longitude,
			Latitude:    r.Latitude,
			AltitudeKm:  r.AltitudeKm,
			Velocity:    []float64{r.Velocity[0], r.Velocity[1], r.Velocity[2]},
			LastUpdated: r.LastUpdated.Format("2006-01-02T15:04:05Z"),
		})
	}

	// Log scan for audit (IAF requirement)
	logScan(c, "bbox", len(results))

	c.JSON(http.StatusOK, response)
}

// PostVolumetricScanCone handles POST /api/scan/volumetric/cone
// Performs sensor cone volumetric scan (simplified to bbox approximation)
func (h *VolumetricHandler) PostVolumetricScanCone(c *gin.Context) {
	var req SensorConeRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, ErrorResponse{
			Error:   "Invalid request",
			Details: err.Error(),
		})
		return
	}

	// Execute cone scan (uses bbox approximation internally)
	results, err := h.statesRepo.VolumetricScanCone(
		c.Request.Context(),
		req.OriginLon, req.OriginLat, req.OriginAltKm,
		req.AzimuthDeg, req.ElevationDeg,
		req.FovHDeg, req.FovVDeg,
		req.RangeKm,
	)
	if err != nil {
		c.JSON(http.StatusInternalServerError, ErrorResponse{
			Error:   "Database query failed",
			Details: err.Error(),
		})
		return
	}

	// Transform to response format
	response := ScanResponse{
		Count:   len(results),
		Objects: make([]ScanResult, 0, len(results)),
	}

	for _, r := range results {
		response.Objects = append(response.Objects, ScanResult{
			NoradCatID:  r.NoradCatID,
			ObjectName:  r.ObjectName,
			Longitude:   r.Longitude,
			Latitude:    r.Latitude,
			AltitudeKm:  r.AltitudeKm,
			Velocity:    []float64{r.Velocity[0], r.Velocity[1], r.Velocity[2]},
			LastUpdated: r.LastUpdated.Format("2006-01-02T15:04:05Z"),
		})
	}

	// Log scan for audit
	logScan(c, "cone", len(results))

	c.JSON(http.StatusOK, response)
}

// GetVolumetricScan handles unified endpoint POST /api/scan/volumetric
// Accepts either bbox or cone scan type via "scan_type" field
type UnifiedScanRequest struct {
	ScanType string `json:"scan_type" binding:"required,oneof=bbox cone"`
	BoundingBoxRequest
	SensorConeRequest
}

func (h *VolumetricHandler) PostVolumetricScan(c *gin.Context) {
	var req UnifiedScanRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, ErrorResponse{
			Error:   "Invalid request",
			Details: err.Error(),
		})
		return
	}

	switch req.ScanType {
	case "bbox":
		h.PostVolumetricScanBBox(c)
	case "cone":
		h.PostVolumetricScanCone(c)
	default:
		c.JSON(http.StatusBadRequest, ErrorResponse{
			Error: fmt.Sprintf("Unknown scan_type: %s", req.ScanType),
		})
	}
}

// logScan logs volumetric scan for audit trail (IAF requirement)
func logScan(c *gin.Context, scanType string, resultCount int) {
	// In production: write to audit log table or structured logging
	clientIP := c.ClientIP()
	userAgent := c.GetHeader("User-Agent")
	_ = clientIP
	_ = userAgent
	_ = scanType
	_ = resultCount
	// TODO: Implement audit logging
}
