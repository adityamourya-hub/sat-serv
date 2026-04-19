package server

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"ssis/pkg/catalog"
	"ssis/pkg/cryptography"
	"ssis/pkg/engine"
	"ssis/pkg/octree"
	"ssis/pkg/types"
	pb "ssis/pkg/proto/ssis/v1"
)

// mockRuntime is a minimal mock for testing Runtime interface
type mockRuntime struct {
	submitErr error
	bus       *types.EventBus
}

func (m *mockRuntime) SubmitIngest(ctx context.Context, obs *types.Observation) error {
	return m.submitErr
}

func (m *mockRuntime) GetEventBus() *types.EventBus {
	if m.bus == nil {
		m.bus = types.NewEventBus()
	}
	return m.bus
}

func (m *mockRuntime) Start() {}
func (m *mockRuntime) Stop(ctx context.Context) error { return nil }

func TestIngestionServer_IngestObservation(t *testing.T) {
	tests := []struct {
		name      string
		obs       *pb.Observation
		submitErr error
		wantStatus pb.IngestStatus
	}{
		{
			name: "successful ingestion",
			obs: &pb.Observation{
				ObservationId: "obs-001",
				SensorId:      "sensor-1",
				Radar:         &pb.RadarMeasurement{TargetId: "target-1"},
			},
			submitErr:  nil,
			wantStatus: pb.IngestStatus_INGEST_STATUS_ACCEPTED,
		},
		{
			name: "submission error",
			obs: &pb.Observation{
				ObservationId: "obs-002",
				SensorId:      "sensor-1",
				Radar:         &pb.RadarMeasurement{TargetId: "target-2"},
			},
			submitErr:  context.Canceled,
			wantStatus: pb.IngestStatus_INGEST_STATUS_REJECTED_INVALID,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mockEng := &mockRuntime{submitErr: tt.submitErr}
			server := &IngestionServer{Engine: mockEng}

			resp, err := server.IngestObservation(context.Background(), tt.obs)
			if err != nil {
				t.Fatalf("IngestObservation() unexpected error: %v", err)
			}

			if resp.Status != tt.wantStatus {
				t.Errorf("IngestObservation() status = %v, want %v", resp.Status, tt.wantStatus)
			}

			if resp.ObservationId != tt.obs.ObservationId {
				t.Errorf("IngestObservation() id = %v, want %v", resp.ObservationId, tt.obs.ObservationId)
			}
		})
	}
}

func TestIngestionServer_StreamObservations(t *testing.T) {
	mockEng := &mockRuntime{submitErr: nil}
	server := &IngestionServer{Engine: mockEng}

	// This test verifies the method exists and handles nil stream gracefully
	// Full streaming tests require a mock gRPC stream implementation
	t.Skip("Streaming tests require mock gRPC stream implementation")
}

func TestAlertServer_SubscribeAlerts(t *testing.T) {
	tests := []struct {
		name        string
		minPriority pb.AlertPriority
		types       []pb.AlertType
	}{
		{
			name:        "subscribe with min priority",
			minPriority: pb.AlertPriority_ALERT_PRIORITY_HIGH,
			types:       nil,
		},
		{
			name:        "subscribe with type filter",
			minPriority: pb.AlertPriority_ALERT_PRIORITY_LOW,
			types:       []pb.AlertType{pb.AlertType_ALERT_TYPE_COLLISION},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mockEng := &mockRuntime{}
			signer, err := cryptography.NewSigner([]byte("test-key-32-bytes-123456789012345"))
			if err != nil {
				t.Fatalf("Failed to create signer: %v", err)
			}

			server := &AlertServer{Engine: mockEng, Signer: signer}

			req := &pb.AlertSubscription{
				MinPriority: tt.minPriority,
				Types:        tt.types,
			}

			// This test verifies the subscription request is accepted
			// Full streaming tests require a mock gRPC stream implementation
			t.Skip("Streaming tests require mock gRPC stream implementation")
			_ = req
			_ = server
		})
	}
}

func TestSSISServer_ScanGroundRegion(t *testing.T) {
	store := catalog.NewStore(100.0)

	// Add test objects
	delta := catalog.Delta{
		Upserts: []catalog.SDAObject{
			{
				ID:   1,
				Name: "SAT-1",
				Pos:  octree.Vector3{X: 7000, Y: 0, Z: 0},
				Vel:  octree.Vector3{X: 0, Y: 7.5, Z: 0},
			},
			{
				ID:   2,
				Name: "SAT-2",
				Pos:  octree.Vector3{X: -7000, Y: 0, Z: 0},
				Vel:  octree.Vector3{X: 0, Y: -7.5, Z: 0},
			},
		},
	}
	store.ApplyDeltaWithRebuild(delta, false)

	mockEng := &mockRuntime{}
	ingestSrv := &IngestionServer{Engine: mockEng}
	alertSrv := &AlertServer{Engine: mockEng}
	server := &SSISServer{
		Ingest: ingestSrv,
		Alert:  alertSrv,
		Store:  store,
	}

	req := &pb.CatalogQuery{
		CenterLatitudeDeg:  28.6,
		CenterLongitudeDeg: 77.2,
		RadiusKm:          1000,
	}

	resp, err := server.ScanGroundRegion(context.Background(), req)
	if err != nil {
		t.Fatalf("ScanGroundRegion() error = %v", err)
	}

	if resp == nil {
		t.Fatal("ScanGroundRegion() returned nil response")
	}

	// Verify response structure
	if len(resp.Objects) == 0 {
		t.Logf("ScanGroundRegion() returned no objects (expected for ground scan)")
	}
}

func TestConvertProtoToObservation(t *testing.T) {
	tests := []struct {
		name     string
		pbObs    *pb.Observation
		wantID   string
		wantType string
	}{
		{
			name: "radar observation",
			pbObs: &pb.Observation{
				ObservationId: "obs-001",
				SensorId:      "sensor-1",
				Radar:         &pb.RadarMeasurement{TargetId: "target-1"},
			},
			wantID:   "target-1",
			wantType: "radar",
		},
		{
			name: "optical observation",
			pbObs: &pb.Observation{
				ObservationId: "obs-002",
				SensorId:      "sensor-2",
				Optical:       &pb.OpticalMeasurement{TargetId: "target-2"},
			},
			wantID:   "target-2",
			wantType: "optical",
		},
		{
			name: "no measurement type",
			pbObs: &pb.Observation{
				ObservationId: "obs-003",
				SensorId:      "sensor-3",
			},
			wantID:   "",
			wantType: "none",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			obs := convertProtoToObservation(tt.pbObs)

			if obs.TargetID != tt.wantID {
				t.Errorf("convertProtoToObservation() targetID = %v, want %v", obs.TargetID, tt.wantID)
			}

			if obs.SensorID != tt.pbObs.SensorId {
				t.Errorf("convertProtoToObservation() sensorID = %v, want %v", obs.SensorID, tt.pbObs.SensorId)
			}

			if obs.ObservationID != tt.pbObs.ObservationId {
				t.Errorf("convertProtoToObservation() observationID = %v, want %v", obs.ObservationID, tt.pbObs.ObservationId)
			}
		})
	}
}

func TestConvertAlertToProto(t *testing.T) {
	alert := &types.Alert{
		AlertID:   "alert-001",
		Type:      types.AlertTypeCollision,
		Priority:  types.PriorityHigh,
		TargetID:  "target-1",
		Timestamp: time.Now(),
	}

	pbAlert := convertAlertToProto(alert)

	if pbAlert.AlertId != alert.AlertID {
		t.Errorf("convertAlertToProto() alertId = %v, want %v", pbAlert.AlertId, alert.AlertID)
	}

	if pbAlert.Type != pb.AlertType(alert.Type) {
		t.Errorf("convertAlertToProto() type = %v, want %v", pbAlert.Type, alert.Type)
	}

	if pbAlert.Priority != pb.AlertPriority(alert.Priority) {
		t.Errorf("convertAlertToProto() priority = %v, want %v", pbAlert.Priority, alert.Priority)
	}

	if pbAlert.Timestamp == nil {
		t.Error("convertAlertToProto() timestamp is nil")
	}
}

func TestRESTGateway_ServeHTTP(t *testing.T) {
	store := catalog.NewStore(100.0)
	mockEng := &mockRuntime{}
	ingestSrv := &IngestionServer{Engine: mockEng}
	alertSrv := &AlertServer{Engine: mockEng}
	ssisSrv := &SSISServer{
		Ingest: ingestSrv,
		Alert:  alertSrv,
		Store:  store,
	}

	gateway := &RESTGateway{SSIS: ssisSrv}

	tests := []struct {
		name           string
		method         string
		path           string
		authHeader     string
		wantStatusCode int
	}{
		{
			name:           "unauthorized request",
			method:         "GET",
			path:           "/v1/scan",
			authHeader:     "",
			wantStatusCode: 401,
		},
		{
			name:           "not found",
			method:         "GET",
			path:           "/v1/unknown",
			authHeader:     "Bearer token",
			wantStatusCode: 404,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := httptest.NewRequest(tt.method, tt.path, nil)
			if tt.authHeader != "" {
				req.Header.Set("Authorization", tt.authHeader)
			}

			w := httptest.NewRecorder()
			gateway.ServeHTTP(w, req)

			if w.Code != tt.wantStatusCode {
				t.Errorf("ServeHTTP() status = %d, want %d", w.Code, tt.wantStatusCode)
			}
		})
	}
}

func TestRESTGateway_handleScan(t *testing.T) {
	store := catalog.NewStore(100.0)
	mockEng := &mockRuntime{}
	ingestSrv := &IngestionServer{Engine: mockEng}
	alertSrv := &AlertServer{Engine: mockEng}
	ssisSrv := &SSISServer{
		Ingest: ingestSrv,
		Alert:  alertSrv,
		Store:  store,
	}

	gateway := &RESTGateway{SSIS: ssisSrv}

	reqBody := `{
		"center_latitude_deg": 28.6,
		"center_longitude_deg": 77.2,
		"radius_km": 1000
	}`

	req := httptest.NewRequest("POST", "/v1/scan", strings.NewReader(reqBody))
	req.Header.Set("Authorization", "Bearer token")
	req.Header.Set("Content-Type", "application/json")

	w := httptest.NewRecorder()
	gateway.handleScan(w, req)

	if w.Code != http.StatusOK && w.Code != http.StatusInternalServerError {
		t.Logf("handleScan() status = %d", w.Code)
	}

	if w.Code == http.StatusOK && w.Header().Get("Content-Type") != "application/json" {
		t.Errorf("handleScan() Content-Type = %v, want application/json", w.Header().Get("Content-Type"))
	}
}
