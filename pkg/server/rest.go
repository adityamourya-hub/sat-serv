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
	Store   *catalog.Store
	Signer  *cryptography.Signer
	Engine  *engine.Runtime
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
