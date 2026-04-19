package engine

import (
	"context"
	"crypto/tls"
	"fmt"
	"io"
	"net"
	"strconv"
	"sync"
	"time"

	"github.com/golang/protobuf/proto"
	"github.com/golang/protobuf/ptypes/timestamp"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/peer"
	"google.golang.org/grpc/status"

	"ssis/pkg/catalog"
	"ssis/pkg/common"
	"ssis/pkg/cryptography"
	pb "ssis/pkg/proto/ssis/v1"
	"ssis/pkg/types"
)

// GRPCServer implements the SpaceEngine gRPC service
type GRPCServer struct {
	pb.UnimplementedSpaceEngineServer

	runtime     *Runtime
	signer      *cryptography.Signer
	auditLog    *common.AuditLog
	subscribers sync.Map // map[string]*AlertSubscriber
	mu          sync.RWMutex
}

// AlertSubscriber tracks a connected alert client
type AlertSubscriber struct {
	clientID     string
	subscription *pb.AlertSubscription
	stream       pb.SpaceEngine_SubscribeAlertsServer
	done         chan struct{}
}

// NewGRPCServer creates a new gRPC server with mTLS
func NewGRPCServer(runtime *Runtime, signer *cryptography.Signer, auditLog *common.AuditLog) *GRPCServer {
	return &GRPCServer{
		runtime:  runtime,
		signer:   signer,
		auditLog: auditLog,
	}
}

// IngestObservation handles bidirectional streaming of sensor observations
func (s *GRPCServer) IngestObservation(stream pb.SpaceEngine_IngestObservationServer) error {
	ctx := stream.Context()

	// Extract client identity from mTLS certificate
	clientID, err := s.extractClientID(ctx)
	if err != nil {
		return status.Error(codes.Unauthenticated, "invalid client certificate")
	}

	// Process incoming observations
	for {
		obs, err := stream.Recv()
		if err == io.EOF {
			return nil
		}
		if err != nil {
			s.auditLog.LogError(clientID, "ingest_receive_failed", err)
			return status.Error(codes.Internal, "failed to receive observation")
		}

		// Validate and process
		ack := s.processObservation(ctx, clientID, obs)

		if err := stream.Send(ack); err != nil {
			s.auditLog.LogError(clientID, "ingest_send_failed", err)
			return status.Error(codes.Internal, "failed to send ack")
		}
	}
}

func (s *GRPCServer) processObservation(ctx context.Context, clientID string, pbObs *pb.Observation) *pb.IngestAck {
	start := time.Now()

	// Convert protobuf to internal format
	obs := convertProtoToObservation(pbObs)

	// Verify signature if present
	if len(pbObs.Signature) > 0 {
		// Convert protobuf to bytes for signature verification
		obsBytes, err := proto.Marshal(pbObs)
		sigStr := string(pbObs.Signature)
		if err != nil || !s.signer.Verify(obsBytes, sigStr) {
			s.auditLog.LogError(clientID, "invalid_signature", fmt.Errorf("signature verification failed"))
			return &pb.IngestAck{
				ObservationId: pbObs.ObservationId,
				Status:        pb.IngestStatus_INGEST_STATUS_REJECTED_UNAUTHORIZED,
				Message:       "invalid signature",
			}
		}
	}

	// Submit to runtime queue
	if err := s.runtime.SubmitIngest(ctx, obs); err != nil {
		s.auditLog.LogError(clientID, "queue_full", err)
		return &pb.IngestAck{
			ObservationId:  pbObs.ObservationId,
			Status:         pb.IngestStatus_INGEST_STATUS_REJECTED_OVERFLOW,
			Message:        err.Error(),
			QueueLatencyMs: int32(time.Since(start).Milliseconds()),
		}
	}

	s.auditLog.LogSuccess(clientID, pbObs.ObservationId)

	return &pb.IngestAck{
		ObservationId:     pbObs.ObservationId,
		ReceivedTimestamp: timeToProto(time.Now()),
		Status:            pb.IngestStatus_INGEST_STATUS_ACCEPTED,
		Message:           "queued for processing",
		QueueLatencyMs:    int32(time.Since(start).Milliseconds()),
	}
}

// SubscribeAlerts provides server-streaming alerts to clients
func (s *GRPCServer) SubscribeAlerts(req *pb.AlertSubscription, stream pb.SpaceEngine_SubscribeAlertsServer) error {
	clientID := req.ClientId
	if clientID == "" {
		return status.Error(codes.InvalidArgument, "client_id required")
	}

	// Register subscriber
	sub := &AlertSubscriber{
		clientID:     clientID,
		subscription: req,
		stream:       stream,
		done:         make(chan struct{}),
	}
	s.subscribers.Store(clientID, sub)
	defer s.subscribers.Delete(clientID)

	// Subscribe to event bus
	alertChan := make(chan *types.Alert, 100)
	s.runtime.bus.Subscribe(alertChan)
	defer s.runtime.bus.Unsubscribe(alertChan)

	ctx := stream.Context()

	for {
		select {
		case <-ctx.Done():
			return nil
		case <-sub.done:
			return nil
		case alert := <-alertChan:
			if s.shouldSendAlert(alert, req) {
				pbAlert := convertAlertToProto(alert)
				if err := stream.Send(pbAlert); err != nil {
					s.auditLog.LogError(clientID, "alert_send_failed", err)
					return err
				}
			}
		}
	}
}

func (s *GRPCServer) shouldSendAlert(alert *types.Alert, sub *pb.AlertSubscription) bool {
	// Check type filter
	typeMatch := false
	for _, t := range sub.Types {
		if int32(t) == int32(alert.Type) {
			typeMatch = true
			break
		}
	}
	if !typeMatch {
		return false
	}

	// Check priority threshold
	if int32(alert.Priority) > int32(sub.MinPriority) {
		return false
	}

	// Check object filter
	if len(sub.ObjectIds) > 0 {
		objMatch := false
		for _, id := range sub.ObjectIds {
			if id == alert.TargetID || id == alert.SecondaryID {
				objMatch = true
				break
			}
		}
		if !objMatch {
			return false
		}
	}

	return true
}

// QueryCatalog queries the satellite catalog
func (s *GRPCServer) QueryCatalog(ctx context.Context, req *pb.CatalogQuery) (*pb.CatalogResponse, error) {
	// Validate query parameters
	if req.Timestamp == nil {
		req.Timestamp = timeToProto(time.Now())
	}

	// Execute query through catalog store
	// For now, return all objects via Snapshot
	objects := s.runtime.store.Snapshot()

	// Convert to protobuf
	pbObjects := make([]*pb.ObjectState, len(objects))
	for i, obj := range objects {
		pbObjects[i] = convertObjectToProto(&obj)
	}

	return &pb.CatalogResponse{
		Objects:       pbObjects,
		TotalCount:    int32(len(pbObjects)),
		NextPageToken: "", // Pagination TODO
	}, nil
}

// GetObjectState retrieves state for a specific object
func (s *GRPCServer) GetObjectState(ctx context.Context, req *pb.ObjectRequest) (*pb.ObjectState, error) {
	if req.ObjectId == "" {
		return nil, status.Error(codes.InvalidArgument, "object_id required")
	}

	// Convert string ID to int
	objID, err := strconv.Atoi(req.ObjectId)
	if err != nil {
		return nil, status.Error(codes.InvalidArgument, "invalid object ID format")
	}

	obj, ok := s.runtime.store.GetRecord(objID)
	if !ok {
		return nil, status.Error(codes.NotFound, "object not found")
	}

	return convertObjectToProto(&obj), nil
}

// OperatorSession provides bidirectional operator communication
func (s *GRPCServer) OperatorSession(stream pb.SpaceEngine_OperatorSessionServer) error {
	ctx := stream.Context()
	clientID, _ := s.extractClientID(ctx)

	// Send initial system state
	if err := stream.Send(s.getSystemState()); err != nil {
		return err
	}

	// Handle incoming commands
	for {
		cmd, err := stream.Recv()
		if err == io.EOF {
			return nil
		}
		if err != nil {
			return err
		}

		// Process command
		state := s.processOperatorCommand(clientID, cmd)

		if err := stream.Send(state); err != nil {
			return err
		}
	}
}

func (s *GRPCServer) processOperatorCommand(clientID string, cmd *pb.OperatorCommand) *pb.SystemState {
	s.auditLog.LogCommand(clientID, fmt.Sprintf("%+v", cmd))

	switch c := cmd.Command.(type) {
	case *pb.OperatorCommand_Query:
		// Handle natural language or structured query
		// TODO: Implement query processing

	case *pb.OperatorCommand_Control:
		// Handle control actions
		switch c.Control.Action {
		case pb.ControlAction_CONTROL_ACTION_FORCE_UPDATE:
			// Force immediate catalog update
		case pb.ControlAction_CONTROL_ACTION_GENERATE_REPORT:
			// Generate and queue report
		}

	case *pb.OperatorCommand_Config:
		// Update runtime configuration
		// TODO: Implement config update
	}

	return s.getSystemState()
}

func (s *GRPCServer) getSystemState() *pb.SystemState {
	m := s.runtime.metrics
	elapsed := time.Since(m.startTime).Seconds()

	var obsPerSec float64
	if elapsed > 0 {
		obsPerSec = float64(m.Completions) / elapsed
	}

	return &pb.SystemState{
		Timestamp:          timeToProto(time.Now()),
		Health:             &pb.SystemHealth{Overall: pb.HealthStatus_HEALTH_STATUS_HEALTHY},
		CatalogSize:        int32(len(s.runtime.store.Snapshot())),
		ObservationsPerSec: obsPerSec,
		AvgLatencyMs:       0, // TODO: Implement latency tracking
		P99LatencyMs:       0, // TODO: Implement latency tracking
		CriticalAlerts:     0, // TODO
		HighAlerts:         0, // TODO
	}
}

func (s *GRPCServer) extractClientID(ctx context.Context) (string, error) {
	// Extract from mTLS peer certificate
	peer, ok := peer.FromContext(ctx)
	if !ok {
		return "", fmt.Errorf("no peer info")
	}

	tlsInfo, ok := peer.AuthInfo.(credentials.TLSInfo)
	if !ok {
		return "", fmt.Errorf("no TLS info")
	}

	if len(tlsInfo.State.PeerCertificates) == 0 {
		return "", fmt.Errorf("no client certificate")
	}

	// Use certificate subject CN as client ID
	cert := tlsInfo.State.PeerCertificates[0]
	return cert.Subject.CommonName, nil
}

// StartGRPCServer initializes and starts the gRPC server
func StartGRPCServer(addr string, runtime *Runtime, signer *cryptography.Signer, auditLog *common.AuditLog, tlsConfig *tls.Config) (*grpc.Server, error) {
	server := NewGRPCServer(runtime, signer, auditLog)

	opts := []grpc.ServerOption{
		grpc.Creds(credentials.NewTLS(tlsConfig)),
		grpc.StreamInterceptor(serverStreamInterceptor(auditLog)),
		grpc.UnaryInterceptor(serverUnaryInterceptor(auditLog)),
	}

	grpcServer := grpc.NewServer(opts...)
	pb.RegisterSpaceEngineServer(grpcServer, server)

	lis, err := net.Listen("tcp", addr)
	if err != nil {
		return nil, fmt.Errorf("failed to listen: %w", err)
	}

	go func() {
		if err := grpcServer.Serve(lis); err != nil {
			auditLog.LogError("server", "serve_failed", err)
		}
	}()

	return grpcServer, nil
}

func serverStreamInterceptor(auditLog *common.AuditLog) grpc.StreamServerInterceptor {
	return func(srv interface{}, ss grpc.ServerStream, info *grpc.StreamServerInfo, handler grpc.StreamHandler) error {
		start := time.Now()
		err := handler(srv, ss)
		duration := time.Since(start)

		auditLog.LogRPC(info.FullMethod, duration, err)
		return err
	}
}

func serverUnaryInterceptor(auditLog *common.AuditLog) grpc.UnaryServerInterceptor {
	return func(ctx context.Context, req interface{}, info *grpc.UnaryServerInfo, handler grpc.UnaryHandler) (interface{}, error) {
		start := time.Now()
		resp, err := handler(ctx, req)
		duration := time.Since(start)

		auditLog.LogRPC(info.FullMethod, duration, err)
		return resp, err
	}
}

// Helper conversion functions
func convertProtoToObservation(pbObs *pb.Observation) *types.Observation {
	obs := &types.Observation{
		ObservationID: pbObs.ObservationId,
		SensorID:      pbObs.SensorId,
		TargetID:      "",
		Timestamp:     protoToTime(pbObs.Timestamp),
	}

	switch m := pbObs.Measurement.(type) {
	case *pb.Observation_Radar:
		obs.TargetID = m.Radar.TargetId
		obs.Type = types.MeasurementTypeRadar
		obs.Radar = &types.RadarMeasurement{
			TargetID:     m.Radar.TargetId,
			RangeKm:      m.Radar.RangeKm,
			RangeRateKms: m.Radar.RangeRateKms,
			AzimuthDeg:   m.Radar.AzimuthDeg,
			ElevationDeg: m.Radar.ElevationDeg,
			RcsDbm:       m.Radar.RcsDbm,
			SnrDb:        m.Radar.SnrDb,
		}
		// TODO: Handle other measurement types
	}

	return obs
}

func convertAlertToProto(alert *types.Alert) *pb.Alert {
	return &pb.Alert{
		AlertId:   alert.AlertID,
		Type:      pb.AlertType(alert.Type),
		Priority:  pb.AlertPriority(alert.Priority),
		Timestamp: timeToProto(alert.Timestamp),
	}
}

func convertObjectToProto(obj *catalog.ObjectRecord) *pb.ObjectState {
	return &pb.ObjectState{
		ObjectId:     fmt.Sprintf("%d", obj.ID),
		ObjectName:   obj.Name,
		Epoch:        timeToProto(time.Unix(obj.LastUpdateUnix, 0)),
		PositionXKm:  obj.Pos.X,
		PositionYKm:  obj.Pos.Y,
		PositionZKm:  obj.Pos.Z,
		VelocityXKms: obj.Vel.X,
		VelocityYKms: obj.Vel.Y,
		VelocityZKms: obj.Vel.Z,
	}
}

func timeToProto(t time.Time) *timestamp.Timestamp {
	return &timestamp.Timestamp{
		Seconds: t.Unix(),
		Nanos:   int32(t.Nanosecond()),
	}
}

func protoToTime(ts *timestamp.Timestamp) time.Time {
	if ts == nil {
		return time.Now()
	}
	return time.Unix(ts.Seconds, int64(ts.Nanos))
}
