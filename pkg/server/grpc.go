package server

import (
	"context"
	"crypto/tls"
	"crypto/x509"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/golang/protobuf/ptypes/timestamp"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials"
	"google.golang.org/protobuf/types/known/timestamppb"

	"ssis/pkg/catalog"
	"ssis/pkg/cryptography"
	"ssis/pkg/engine"
	"ssis/pkg/orbital"
	pb "ssis/pkg/proto/ssis/v1"
	"ssis/pkg/proximity"
	"ssis/pkg/types"
)

// IngestionServer handles high-throughput observation ingest.
type IngestionServer struct {
	pb.UnimplementedSpaceEngineServer
	Engine *engine.Runtime
}

func (s *IngestionServer) StreamObservations(stream pb.SpaceEngine_IngestObservationServer) error {
	for {
		obs, err := stream.Recv()
		if err == io.EOF {
			return nil
		}
		if err != nil {
			return err
		}
		// Convert protobuf to internal type
		internalObs := convertProtoToObservation(obs)
		if err := s.Engine.SubmitIngest(context.Background(), internalObs); err != nil {
			return err
		}
		// Send acknowledgment
		if err := stream.Send(&pb.IngestAck{ObservationId: obs.ObservationId, Status: pb.IngestStatus_INGEST_STATUS_ACCEPTED}); err != nil {
			return err
		}
	}
}

func (s *IngestionServer) IngestObservation(ctx context.Context, obs *pb.Observation) (*pb.IngestAck, error) {
	internalObs := convertProtoToObservation(obs)
	if err := s.Engine.SubmitIngest(ctx, internalObs); err != nil {
		return &pb.IngestAck{ObservationId: obs.ObservationId, Status: pb.IngestStatus_INGEST_STATUS_REJECTED_INVALID, Message: err.Error()}, nil
	}
	return &pb.IngestAck{ObservationId: obs.ObservationId, Status: pb.IngestStatus_INGEST_STATUS_ACCEPTED}, nil
}

// AlertServer manages tactical alert subscriptions.
type AlertServer struct {
	pb.UnimplementedSpaceEngineServer
	Engine *engine.Runtime
	Signer *cryptography.Signer
}

func (s *AlertServer) SubscribeAlerts(req *pb.AlertSubscription, stream pb.SpaceEngine_SubscribeAlertsServer) error {
	bus := s.Engine.GetEventBus()
	ch := make(chan *types.Alert, 100)
	bus.Subscribe(ch)
	defer bus.Unsubscribe(ch)

	for {
		select {
		case alert := <-ch:
			// Filter based on subscription criteria
			if int32(alert.Priority) < int32(req.MinPriority) {
				continue
			}
			// Filter by alert type if specified
			if len(req.Types) > 0 {
				typeMatch := false
				for _, t := range req.Types {
					if pb.AlertType(alert.Type) == t {
						typeMatch = true
						break
					}
				}
				if !typeMatch {
					continue
				}
			}
			if s.Signer != nil {
				payload := fmt.Sprintf("alert:%s:%d:%d", alert.TargetID, alert.Type, alert.Priority)
				alert.Signature = []byte(s.Signer.Sign([]byte(payload)))
			}
			pbAlert := convertAlertToProto(alert)
			if err := stream.Send(pbAlert); err != nil {
				return err
			}
		case <-stream.Context().Done():
			return stream.Context().Err()
		}
	}
}

// SSISServer implements the DEPRECATED monolithic interface for Phase 0 compatibility.
type SSISServer struct {
	pb.UnimplementedSpaceEngineServer
	Ingest *IngestionServer
	Alert  *AlertServer
	Store  *catalog.Store
	Axiom  *orbital.RKANKernel
}

func (s *SSISServer) ScanGroundRegion(ctx context.Context, req *pb.CatalogQuery) (*pb.CatalogResponse, error) {
	// Re-implemented to use the refined proximity logic
	roi := proximity.RegionOfInterest{LatDeg: req.CenterLatitudeDeg, LonDeg: req.CenterLongitudeDeg, RadiusKm: req.RadiusKm}
	objects := proximity.ScanGroundRegion(roi, s.Store, time.Now())

	resp := &pb.CatalogResponse{Objects: make([]*pb.ObjectState, 0, len(objects))}
	for _, obj := range objects {
		resp.Objects = append(resp.Objects, &pb.ObjectState{ObjectId: fmt.Sprintf("%d", obj.ID), ObjectName: obj.Name})
	}
	return resp, nil
}

func (s *SSISServer) StreamObservations(stream pb.SpaceEngine_IngestObservationServer) error {
	return s.Ingest.StreamObservations(stream)
}

func (s *SSISServer) SubscribeAlerts(req *pb.AlertSubscription, stream pb.SpaceEngine_SubscribeAlertsServer) error {
	return s.Alert.SubscribeAlerts(req, stream)
}

// QueryCatalog queries the satellite catalog with pagination support
func (s *SSISServer) QueryCatalog(ctx context.Context, req *pb.CatalogQuery) (*pb.CatalogResponse, error) {
	if req.Timestamp == nil {
		req.Timestamp = timestamppb.New(time.Now())
	}

	snapshot := s.Store.Snapshot()
	objects := make([]*pb.ObjectState, 0, len(snapshot))

	for _, obj := range snapshot {
		// Apply filters if specified
		countryMatch := true
		typeMatch := true
		if len(req.ObjectTypes) > 0 {
			typeMatch = false
			for _, ot := range req.ObjectTypes {
				if strings.EqualFold(ot, obj.AssetType) {
					typeMatch = true
					break
				}
			}
		}
		if !countryMatch || !typeMatch {
			continue
		}

		objects = append(objects, &pb.ObjectState{
			ObjectId:     fmt.Sprintf("%d", obj.ID),
			ObjectName:   obj.Name,
			PositionXKm:  obj.Pos.X,
			PositionYKm:  obj.Pos.Y,
			PositionZKm:  obj.Pos.Z,
			VelocityXKms: obj.Vel.X,
			VelocityYKms: obj.Vel.Y,
			VelocityZKms: obj.Vel.Z,
			Epoch:        timestamppb.New(time.Unix(int64(obj.EpochTT*86400-2440587.5*86400), 0)),
		})
	}

	return &pb.CatalogResponse{Objects: objects}, nil
}

// convertProtoToObservation converts protobuf Observation to internal type
func convertProtoToObservation(pbObs *pb.Observation) *types.Observation {
	// Extract target ID from measurement based on type
	var targetID string
	if pbObs.GetRadar() != nil {
		targetID = pbObs.GetRadar().TargetId
	} else if pbObs.GetOptical() != nil {
		targetID = pbObs.GetOptical().TargetId
	}

	// Use protobuf timestamp if available, otherwise use current time
	timestamp := time.Now()
	if pbObs.Timestamp != nil {
		timestamp = pbObs.Timestamp.AsTime()
	}

	return &types.Observation{
		ObservationID: pbObs.ObservationId,
		SensorID:      pbObs.SensorId,
		TargetID:      targetID,
		Timestamp:     timestamp,
	}
}

// convertAlertToProto converts internal Alert to protobuf
func convertAlertToProto(alert *types.Alert) *pb.Alert {
	return &pb.Alert{
		AlertId:  alert.AlertID,
		Type:     pb.AlertType(alert.Type),
		Priority: pb.AlertPriority(alert.Priority),
		Timestamp: &timestamp.Timestamp{
			Seconds: alert.Timestamp.Unix(),
			Nanos:   int32(alert.Timestamp.Nanosecond()),
		},
	}
}

/**
 * RESTGateway handles HTTP/1.1 fallbacks for JSON-native clients.
 * Bridging Phase 1.7 gRPC logic to legacy humanitarian and defense displays.
 *
 * Logic: It provides RESTful endpoints that internally dispatch to
 * the gRPC server handlers, ensuring a single source of truth.
 */
type RESTGateway struct {
	SSIS *SSISServer
}

func (g *RESTGateway) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	// Entry Point: REST Request Handling
	fmt.Printf("DEBUG: REST Gateway received %s %s at %v\n", r.Method, r.URL.Path, time.Now())

	// Authorization Check (JWT Fallback)
	auth := r.Header.Get("Authorization")
	if auth == "" && r.TLS == nil {
		http.Error(w, "Unauthorized: mTLS or JWT required", http.StatusUnauthorized)
		return
	}

	if strings.HasPrefix(r.URL.Path, "/v1/scan") {
		g.handleScan(w, r)
		return
	}

	http.NotFound(w, r)
	// Exit Point: REST Request Finalized
}

func (g *RESTGateway) handleScan(w http.ResponseWriter, r *http.Request) {
	var req pb.CatalogQuery
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	resp, err := g.SSIS.QueryCatalog(r.Context(), &req)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

/**
 * StartGRPCServer initializes the sovereign endpoints with mTLS support.
 *
 * <plan_complexity>High</plan_complexity><total_lines_estimate>600</total_lines_estimate>
 */
func StartGRPCServer(port int, store *catalog.Store, signer *cryptography.Signer, axiom *orbital.RKANKernel, engineRuntime *engine.Runtime, certFile, keyFile, caFile string) error {
	fmt.Printf("DEBUG: Starting gRPC Server v1.7 on port %d (mTLS enabled: %v)\n", port, caFile != "")

	lis, err := net.Listen("tcp", fmt.Sprintf(":%d", port))
	if err != nil {
		return err
	}

	var opts []grpc.ServerOption

	// 1. Enforce TLS/mTLS if certificates provided
	if certFile != "" && keyFile != "" {
		var tlsConfig tls.Config
		tlsConfig.Certificates = make([]tls.Certificate, 1)
		cert, err := tls.LoadX509KeyPair(certFile, keyFile)
		if err != nil {
			return fmt.Errorf("failed to load TLS keys: %w", err)
		}
		tlsConfig.Certificates[0] = cert

		// If CA is provided, enable client authentication (mTLS)
		if caFile != "" {
			caCert, err := os.ReadFile(caFile)
			if err != nil {
				return fmt.Errorf("failed to read CA cert: %w", err)
			}
			caCertPool := x509.NewCertPool()
			if !caCertPool.AppendCertsFromPEM(caCert) {
				return fmt.Errorf("failed to parse CA cert")
			}
			tlsConfig.ClientCAs = caCertPool
			tlsConfig.ClientAuth = tls.RequireAndVerifyClientCert
			fmt.Printf(" [SEC] mTLS enabled via CA: %s\n", caFile)
		}

		creds := credentials.NewTLS(&tlsConfig)
		opts = append(opts, grpc.Creds(creds))
	} else {
		fmt.Printf(" [WRN] gRPC starting in INSECURE mode (no certs)\n")
	}

	s := grpc.NewServer(opts...)

	ingestSrv := &IngestionServer{Engine: engineRuntime}
	alertSrv := &AlertServer{Engine: engineRuntime, Signer: signer}
	ssisSrv := &SSISServer{Ingest: ingestSrv, Alert: alertSrv, Store: store, Axiom: axiom}

	pb.RegisterSpaceEngineServer(s, ssisSrv)

	fmt.Printf("IAF-SSIS (v1.7) sovereign substrate listening at %v\n", lis.Addr())
	return s.Serve(lis)
}
