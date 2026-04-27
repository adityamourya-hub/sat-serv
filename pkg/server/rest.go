package server

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strconv"
	"time"

	"ssis/pkg/catalog"
	"ssis/pkg/cryptography"
	"ssis/pkg/engine"
	"ssis/pkg/physics"
	"ssis/pkg/proximity"

	"github.com/gin-gonic/gin"
)

type Server struct {
	Store  *catalog.Store
	Signer *cryptography.Signer
	Engine *engine.Runtime
}

func NewServer(store *catalog.Store, signer *cryptography.Signer, engine *engine.Runtime) *Server {
	return &Server{Store: store, Signer: signer, Engine: engine}
}

func (s *Server) SetupRouter() *gin.Engine {
	r := gin.Default()

	// Health check endpoint
	r.GET("/health", s.handleHealth)

	// Catalog endpoints
	r.GET("/catalog", s.handleListCatalog)
	r.GET("/catalog/:id", s.handleGetObject)

	// Volumetric scanning endpoint
	r.GET("/volumetric/scan", s.handleVolumetricScan)

	// Conjunction risk endpoint
	r.GET("/conjunction/risk", s.handleCollisionRisk)

	// Sovereign volume monitoring (air-gapped, no DB required)
	r.GET("/api/sovereign/volumes", s.handleSovereignVolumes)
	r.GET("/api/sovereign/intrusions", s.handleSovereignIntrusions)

	// Metrics endpoint
	r.GET("/metrics", s.handleMetrics)

	return r
}

func (s *Server) handleHealth(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{
		"status":  "healthy",
		"service": "SSIS REST API",
		"version": "1.7.0",
	})
}

func (s *Server) handleListCatalog(c *gin.Context) {
	snapshot := s.Store.Snapshot()

	resp := gin.H{
		"status":    "success",
		"count":     len(snapshot),
		"timestamp": time.Now().Unix(),
		"objects":   snapshot,
	}

	if s.Signer != nil {
		b, err := json.Marshal(resp)
		if err == nil {
			sig := s.Signer.Sign(b)
			c.Header("X-SSIS-Signature", sig)
			resp["signature"] = sig
		}
	}

	c.JSON(http.StatusOK, resp)
}

func (s *Server) handleGetObject(c *gin.Context) {
	idStr := c.Param("id")
	id, err := strconv.Atoi(idStr)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{
			"status":  "error",
			"message": "Invalid object ID",
		})
		return
	}

	// Use O(1) lookup instead of scanning entire snapshot
	obj, ok := s.Store.GetRecord(id)
	if !ok {
		c.JSON(http.StatusNotFound, gin.H{
			"status":  "error",
			"message": "Object not found",
		})
		return
	}

	resp := gin.H{
		"status":    "success",
		"timestamp": time.Now().Unix(),
		"object":    obj,
	}

	if s.Signer != nil {
		b, err := json.Marshal(resp)
		if err == nil {
			sig := s.Signer.Sign(b)
			c.Header("X-SSIS-Signature", sig)
			resp["signature"] = sig
		}
	}

	c.JSON(http.StatusOK, resp)
}

func (s *Server) handleMetrics(c *gin.Context) {
	resp := gin.H{
		"status":    "success",
		"timestamp": time.Now().Unix(),
		"metrics": gin.H{
			"total_objects":   len(s.Store.Snapshot()),
			"catalog_version": s.Store.Version(),
		},
	}

	// Use actual engine metrics if available
	if s.Engine != nil {
		stats := s.Engine.GetMetrics()
		resp["metrics"].(gin.H)["active_workers"] = stats.ActiveWorkers
		resp["metrics"].(gin.H)["queue_depth"] = stats.QueueDepth
		resp["metrics"].(gin.H)["processed_24h"] = stats.Processed
		resp["metrics"].(gin.H)["alerts_24h"] = stats.HighAlerts
		resp["metrics"].(gin.H)["service_time_p50_ms"] = stats.ServiceTimeP50Ms
		resp["metrics"].(gin.H)["service_time_p99_ms"] = stats.ServiceTimeP99Ms
	} else {
		// Fallback to placeholder values if engine not available
		resp["metrics"].(gin.H)["active_workers"] = 0
		resp["metrics"].(gin.H)["queue_depth"] = 0
		resp["metrics"].(gin.H)["processed_24h"] = 0
		resp["metrics"].(gin.H)["alerts_24h"] = 0
		resp["metrics"].(gin.H)["service_time_p50_ms"] = 0
		resp["metrics"].(gin.H)["service_time_p99_ms"] = 0
	}

	if s.Signer != nil {
		b, err := json.Marshal(resp)
		if err == nil {
			sig := s.Signer.Sign(b)
			c.Header("X-SSIS-Signature", sig)
			resp["signature"] = sig
		}
	}

	c.JSON(http.StatusOK, resp)
}

func (s *Server) handleVolumetricScan(c *gin.Context) {
	fmt.Printf("[TRC] Entering handleVolumetricScan\n")
	defer fmt.Printf("[TRC] Exiting handleVolumetricScan\n")

	latStr := c.Query("lat")
	lonStr := c.Query("lon")
	radiusStr := c.Query("radius")

	lat, err := strconv.ParseFloat(latStr, 64)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{
			"status":  "error",
			"message": "Invalid latitude parameter",
		})
		return
	}
	if lat < -90 || lat > 90 {
		c.JSON(http.StatusBadRequest, gin.H{
			"status":  "error",
			"message": "Latitude must be between -90 and 90 degrees",
		})
		return
	}

	lon, err := strconv.ParseFloat(lonStr, 64)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{
			"status":  "error",
			"message": "Invalid longitude parameter",
		})
		return
	}
	if lon < -180 || lon > 180 {
		c.JSON(http.StatusBadRequest, gin.H{
			"status":  "error",
			"message": "Longitude must be between -180 and 180 degrees",
		})
		return
	}

	radius, err := strconv.ParseFloat(radiusStr, 64)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{
			"status":  "error",
			"message": "Invalid radius parameter",
		})
		return
	}
	if radius <= 0 {
		c.JSON(http.StatusBadRequest, gin.H{
			"status":  "error",
			"message": "Radius must be positive",
		})
		return
	}

	epochStr := c.Query("epoch")

	epochUnix, _ := strconv.ParseInt(epochStr, 10, 64)
	epoch := time.Unix(epochUnix, 0)
	if epochUnix == 0 {
		epoch = time.Now()
	}

	roi := proximity.RegionOfInterest{
		LatDeg:   lat,
		LonDeg:   lon,
		RadiusKm: radius,
	}

	objects := proximity.ScanGroundRegion(roi, s.Store, epoch)

	resp := gin.H{
		"status":    "PROCESSED",
		"timestamp": time.Now().Unix(),
		"objects":   objects,
	}

	if s.Signer != nil {
		b, err := json.Marshal(resp)
		if err == nil {
			sig := s.Signer.Sign(b)
			c.Header("X-SSIS-Signature", sig)
			resp["signature"] = sig
		}
	}

	c.JSON(http.StatusOK, resp)
}

func (s *Server) handleSovereignVolumes(c *gin.Context) {
	// Return strategic monitoring zones from static config (no DB needed)
	volumes := []gin.H{
		{"name": "Siliguri_Corridor_LEO", "description": "Siliguri Corridor (Chicken's Neck) — LEO ISR monitoring zone", "min_lon": 87.8, "max_lon": 89.5, "min_lat": 25.5, "max_lat": 27.2, "min_alt_km": 300, "max_alt_km": 1200, "alert_level": "CRITICAL"},
		{"name": "Galwan_Valley_MEO", "description": "Galwan Valley — MEO communications monitoring zone", "min_lon": 76.5, "max_lon": 78.5, "min_lat": 34.0, "max_lat": 35.5, "min_alt_km": 2000, "max_alt_km": 35786, "alert_level": "HIGH"},
		{"name": "IAF_Test_Range_GEO", "description": "IAF Test Range — GEO belt monitoring zone", "min_lon": 68.0, "max_lon": 97.0, "min_lat": 8.0, "max_lat": 37.0, "min_alt_km": 35786, "max_alt_km": 36000, "alert_level": "MEDIUM"},
	}
	c.JSON(http.StatusOK, gin.H{"volumes": volumes, "count": len(volumes)})
}

func (s *Server) handleSovereignIntrusions(c *gin.Context) {
	// Scan in-memory catalog for objects inside sovereign volumes
	snapshot := s.Store.Snapshot()
	intrusions := []gin.H{}

	// Simplified bbox checks against Siliguri Corridor
	for _, obj := range snapshot {
		// Only flag non-Indian objects in strategic zones
		if obj.CountryCode != "IND" && obj.CountryCode != "IN" {
			// Check Siliguri Corridor (simplified)
			if obj.Pos.X > 87.8 && obj.Pos.X < 89.5 && obj.Pos.Y > 25.5 && obj.Pos.Y < 27.2 {
				intrusions = append(intrusions, gin.H{
					"volume_name":  "Siliguri_Corridor_LEO",
					"norad_cat_id": obj.ID,
					"object_name":  obj.Name,
					"country_code": obj.CountryCode,
					"longitude":    obj.Pos.X,
					"latitude":     obj.Pos.Y,
					"altitude_km":  obj.Pos.Z,
					"threat_level": "HIGH",
					"detected_at":  time.Now().UTC().Format(time.RFC3339),
				})
			}
		}
	}

	c.JSON(http.StatusOK, gin.H{
		"intrusions":      intrusions,
		"total_count":     len(intrusions),
		"volumes_scanned": 3,
	})
}

func (s *Server) handleCollisionRisk(c *gin.Context) {
	fmt.Printf("[TRC] Entering handleCollisionRisk\n")
	defer fmt.Printf("[TRC] Exiting handleCollisionRisk\n")

	miss, _ := strconv.ParseFloat(c.Query("miss"), 64)
	hbr, _ := strconv.ParseFloat(c.Query("hbr"), 64)
	sigR, _ := strconv.ParseFloat(c.Query("sigma_r"), 64)
	sigI, _ := strconv.ParseFloat(c.Query("sigma_i"), 64)

	input := physics.CollisionRiskInput{
		MissDistanceKm:   miss,
		HardbodyRadiusKm: hbr,
		SigmaRadialKm:    sigR,
		SigmaIntrackKm:   sigI,
	}

	risk := proximity.EvaluateCollisionRisk(input)
	resp := gin.H{
		"status":    "PROCESSED",
		"timestamp": time.Now().Unix(),
		"risk":      risk,
	}

	if s.Signer != nil {
		b, err := json.Marshal(resp)
		if err == nil {
			sig := s.Signer.Sign(b)
			c.Header("X-SSIS-Signature", sig)
			resp["signature"] = sig
		}
	}

	c.JSON(http.StatusOK, resp)
}
