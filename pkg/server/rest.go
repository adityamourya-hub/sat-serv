package server

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strconv"
	"time"

	"ssis/pkg/catalog"
	"ssis/pkg/crypto"
	"ssis/pkg/physics"
	"ssis/pkg/proximity"

	"github.com/gin-gonic/gin"
)

type Server struct {
	Store  *catalog.Store
	Signer *crypto.Signer
}

func NewServer(store *catalog.Store, signer *crypto.Signer) *Server {
	return &Server{Store: store, Signer: signer}
}

func (s *Server) SetupRouter() *gin.Engine {
	r := gin.Default()

	r.GET("/volumetric/scan", s.handleVolumetricScan)
	r.GET("/conjunction/risk", s.handleCollisionRisk)

	return r
}

func (s *Server) handleVolumetricScan(c *gin.Context) {
	fmt.Printf("[TRC] Entering handleVolumetricScan\n")
	defer fmt.Printf("[TRC] Exiting handleVolumetricScan\n")

	lat, _ := strconv.ParseFloat(c.Query("lat"), 64)
	lon, _ := strconv.ParseFloat(c.Query("lon"), 64)
	radius, _ := strconv.ParseFloat(c.Query("radius"), 64)
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
		b, _ := json.Marshal(resp)
		sig := s.Signer.Sign(b)
		c.Header("X-SSIS-Signature", sig)
		resp["signature"] = sig
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
		b, _ := json.Marshal(resp)
		sig := s.Signer.Sign(b)
		c.Header("X-SSIS-Signature", sig)
		resp["signature"] = sig
	}

	c.JSON(http.StatusOK, resp)
}
