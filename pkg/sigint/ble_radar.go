// Package sigint provides BLE radar for physical node security
// Detects unauthorized Bluetooth devices near Reticulum mesh nodes
package sigint

import (
	"fmt"
	"math"
	"time"
)

const (
	// BLE radar thresholds
	ProximityThresholdMeters = 20.0  // 20 meters detection radius
	AlertThresholdSeconds    = 10.0  // 10 seconds presence triggers alert
	MaxRSSI                  = -30.0 // dBm - very close
	MinRSSI                  = -90.0 // dBm - far away
	WhitelistTTL             = 24 * time.Hour
)

// BLEDevice represents a detected Bluetooth device
type BLEDevice struct {
	MACAddress    string
	Name          string
	RSSI          float64 // dBm
	LastSeen      time.Time
	FirstSeen     time.Time
	IsWhitelisted bool
	DeviceType    string // PHONE, HEADSET, SENSOR, UNKNOWN
	Proximity     string // IMMEDIATE, NEAR, FAR
}

// BLEScanResult represents a BLE scan result
type BLEScanResult struct {
	Devices      []*BLEDevice
	ScanDuration time.Duration
	Timestamp    time.Time
}

// BLERadar performs Bluetooth device detection for physical security
type BLERadar struct {
	whitelist       map[string]*BLEDevice
	detectedDevices map[string]*BLEDevice
	alertHistory    []*BLESecurityAlert
	isScanning      bool
	scanInterval    time.Duration
	maxProximity    float64
}

// BLESecurityAlert represents a security alert from BLE radar
type BLESecurityAlert struct {
	AlertID     string
	Device      *BLEDevice
	AlertType   string
	Severity    string
	Proximity   string
	Timestamp   time.Time
	ActionTaken string
}

// NewBLERadar creates a new BLE radar
func NewBLERadar() *BLERadar {
	return &BLERadar{
		whitelist:       make(map[string]*BLEDevice),
		detectedDevices: make(map[string]*BLEDevice),
		alertHistory:    make([]*BLESecurityAlert, 0),
		isScanning:      false,
		scanInterval:    5 * time.Second,
		maxProximity:    ProximityThresholdMeters,
	}
}

// StartScanning starts the BLE radar scanning
func (br *BLERadar) StartScanning() error {
	// TODO: Implement actual BLE scanning using noble/macOS CoreBluetooth
	// For now, mark as scanning
	br.isScanning = true
	return nil
}

// StopScanning stops the BLE radar scanning
func (br *BLERadar) StopScanning() {
	br.isScanning = false
}

// IsScanning checks if the radar is scanning
func (br *BLERadar) IsScanning() bool {
	return br.isScanning
}

// Scan performs a BLE scan
func (br *BLERadar) Scan() (*BLEScanResult, error) {
	if !br.isScanning {
		return nil, fmt.Errorf("radar not scanning")
	}

	// TODO: Implement actual BLE scan
	// For now, return empty result
	return &BLEScanResult{
		Devices:      make([]*BLEDevice, 0),
		ScanDuration: br.scanInterval,
		Timestamp:    time.Now(),
	}, nil
}

// ProcessScanResult processes a scan result
func (br *BLERadar) ProcessScanResult(result *BLEScanResult) []*BLESecurityAlert {
	alerts := make([]*BLESecurityAlert, 0)

	for _, device := range result.Devices {
		// Check if whitelisted
		if _, exists := br.whitelist[device.MACAddress]; exists {
			continue
		}

		// Update detected devices
		if existing, exists := br.detectedDevices[device.MACAddress]; exists {
			existing.RSSI = device.RSSI
			existing.LastSeen = device.LastSeen
		} else {
			br.detectedDevices[device.MACAddress] = device
		}

		// Check for alert conditions
		alert := br.checkForAlert(device)
		if alert != nil {
			alerts = append(alerts, alert)
			br.alertHistory = append(br.alertHistory, alert)
		}
	}

	return alerts
}

// checkForAlert checks if a device should trigger an alert
func (br *BLERadar) checkForAlert(device *BLEDevice) *BLESecurityAlert {
	// Check proximity
	if device.Proximity != "IMMEDIATE" {
		return nil
	}

	// Check how long device has been present
	presenceDuration := time.Since(device.FirstSeen).Seconds()
	if presenceDuration < AlertThresholdSeconds {
		return nil
	}

	// Generate alert
	severity := "LOW"
	if device.RSSI > MaxRSSI-10.0 {
		severity = "CRITICAL"
	} else if device.RSSI > MaxRSSI-20.0 {
		severity = "HIGH"
	} else if device.RSSI > MaxRSSI-30.0 {
		severity = "MEDIUM"
	}

	action := "Continue monitoring"
	if severity == "CRITICAL" {
		action = "IMMEDIATE: Air-gap systems, wipe Plasma buffer"
	} else if severity == "HIGH" {
		action = "Alert operators, verify device identity"
	}

	return &BLESecurityAlert{
		AlertID:     fmt.Sprintf("BLE-ALERT-%d", time.Now().Unix()),
		Device:      device,
		AlertType:   "UNAUTHORIZED_DEVICE",
		Severity:    severity,
		Proximity:   device.Proximity,
		Timestamp:   time.Now(),
		ActionTaken: action,
	}
}

// AddToWhitelist adds a device to the whitelist
func (br *BLERadar) AddToWhitelist(macAddress string, name string) {
	br.whitelist[macAddress] = &BLEDevice{
		MACAddress:    macAddress,
		Name:          name,
		IsWhitelisted: true,
		FirstSeen:     time.Now(),
		LastSeen:      time.Now(),
	}
}

// RemoveFromWhitelist removes a device from the whitelist
func (br *BLERadar) RemoveFromWhitelist(macAddress string) {
	delete(br.whitelist, macAddress)
}

// IsWhitelisted checks if a device is whitelisted
func (br *BLERadar) IsWhitelisted(macAddress string) bool {
	_, exists := br.whitelist[macAddress]
	return exists
}

// GetDetectedDevices returns all detected devices
func (br *BLERadar) GetDetectedDevices() map[string]*BLEDevice {
	return br.detectedDevices
}

// GetDeviceCount returns the number of detected devices
func (br *BLERadar) GetDeviceCount() int {
	return len(br.detectedDevices)
}

// GetAlertHistory returns the alert history
func (br *BLERadar) GetAlertHistory() []*BLESecurityAlert {
	return br.alertHistory
}

// ClearAlertHistory clears the alert history
func (br *BLERadar) ClearAlertHistory() {
	br.alertHistory = make([]*BLESecurityAlert, 0)
}

// ClearOldDevices removes devices not seen recently
func (br *BLERadar) ClearOldDevices(maxAge time.Duration) int {
	cleared := 0
	cutoff := time.Now().Add(-maxAge)

	for mac, device := range br.detectedDevices {
		if device.LastSeen.Before(cutoff) {
			delete(br.detectedDevices, mac)
			cleared++
		}
	}

	return cleared
}

// SetScanInterval sets the scan interval
func (br *BLERadar) SetScanInterval(interval time.Duration) {
	br.scanInterval = interval
}

// SetMaxProximity sets the maximum proximity threshold
func (br *BLERadar) SetMaxProximity(meters float64) {
	br.maxProximity = meters
}

// EstimateDistance estimates distance from RSSI
func (br *BLERadar) EstimateDistance(rssi float64) float64 {
	// Simplified RSSI to distance conversion
	// In production, use proper path loss model
	// Distance = 10^((RSSI - A) / (10 * n))
	// Where A is RSSI at 1 meter, n is path loss exponent (typically 2-4)

	const A = -40.0 // RSSI at 1 meter
	const n = 2.0   // Path loss exponent (free space)

	distance := math.Pow(10, (rssi-A)/(10*n))
	return distance
}

// DetermineProximity determines proximity category from RSSI
func (br *BLERadar) DetermineProximity(rssi float64) string {
	distance := br.EstimateDistance(rssi)

	if distance < 1.0 {
		return "IMMEDIATE"
	} else if distance < 5.0 {
		return "NEAR"
	} else if distance < 10.0 {
		return "FAR"
	} else {
		return "OUT_OF_RANGE"
	}
}

// GetImmediateThreats returns devices in immediate proximity
func (br *BLERadar) GetImmediateThreats() []*BLEDevice {
	threats := make([]*BLEDevice, 0)

	for _, device := range br.detectedDevices {
		if device.Proximity == "IMMEDIATE" && !device.IsWhitelisted {
			threats = append(threats, device)
		}
	}

	return threats
}

// GenerateSecurityReport generates a security report
func (br *BLERadar) GenerateSecurityReport() *BLESecurityReport {
	immediateThreats := br.GetImmediateThreats()

	return &BLESecurityReport{
		TotalDevices:       br.GetDeviceCount(),
		WhitelistedDevices: len(br.whitelist),
		ImmediateThreats:   len(immediateThreats),
		RecentAlerts:       len(br.alertHistory),
		IsScanning:         br.isScanning,
		Timestamp:          time.Now(),
	}
}

// BLESecurityReport represents a security report
type BLESecurityReport struct {
	TotalDevices       int
	WhitelistedDevices int
	ImmediateThreats   int
	RecentAlerts       int
	IsScanning         bool
	Timestamp          time.Time
}

// WipePlasmaBuffer simulates wiping the Plasma buffer on security alert
func (br *BLERadar) WipePlasmaBuffer() error {
	// TODO: Implement actual Plasma buffer wipe
	// This would clear the shared memory buffer used for Go-Python communication
	return fmt.Errorf("Plasma buffer wipe not implemented")
}

// AirgapSystems simulates air-gapping systems on security alert
func (br *BLERadar) AirgapSystems() error {
	// TODO: Implement actual air-gap procedure
	// This would disconnect network interfaces and enable air-gap mode
	return fmt.Errorf("Air-gap procedure not implemented")
}

// AutoResponse performs automatic response to critical alerts
func (br *BLERadar) AutoResponse(alert *BLESecurityAlert) error {
	if alert.Severity == "CRITICAL" {
		// Wipe Plasma buffer
		if err := br.WipePlasmaBuffer(); err != nil {
			return fmt.Errorf("failed to wipe Plasma buffer: %w", err)
		}

		// Air-gap systems
		if err := br.AirgapSystems(); err != nil {
			return fmt.Errorf("failed to air-gap systems: %w", err)
		}
	}

	return nil
}
