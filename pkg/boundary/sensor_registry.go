// pkg/boundary/sensor_registry.go - Sensor Registry and Factory
//
// Manages dynamic sensor registration and instantiation.
// Enables adding new sensors without modifying the core reflex kernel.

package boundary

import (
	"fmt"
	"sync"

	"ssis/pkg/config"
	"ssis/pkg/sensors"
)

// SensorRegistry manages all registered sensors
type SensorRegistry struct {
	sensors map[uint32]sensors.SensorObserver // deviceID -> sensor
	mu      sync.RWMutex
}

// Global sensor registry instance
var globalRegistry = &SensorRegistry{
	sensors: make(map[uint32]sensors.SensorObserver),
}

// RegisterSensor adds a sensor to the registry
func RegisterSensor(sensor sensors.SensorObserver) {
	globalRegistry.mu.Lock()
	defer globalRegistry.mu.Unlock()
	globalRegistry.sensors[sensor.GetDeviceID()] = sensor
}

// GetSensor retrieves a sensor by device ID
func GetSensor(deviceID uint32) (sensors.SensorObserver, bool) {
	globalRegistry.mu.RLock()
	defer globalRegistry.mu.RUnlock()
	sensor, ok := globalRegistry.sensors[deviceID]
	return sensor, ok
}

// GetAllSensors returns all registered sensors
func GetAllSensors() []sensors.SensorObserver {
	globalRegistry.mu.RLock()
	defer globalRegistry.mu.RUnlock()
	sensorList := make([]sensors.SensorObserver, 0, len(globalRegistry.sensors))
	for _, sensor := range globalRegistry.sensors {
		sensorList = append(sensorList, sensor)
	}
	return sensorList
}

// InitializeSensorsFromConfig loads sensors from configuration
func InitializeSensorsFromConfig(cfg *config.Config) error {
	for _, sensorCfg := range cfg.Sensors {
		if !sensorCfg.Enabled {
			continue
		}

		var sensor sensors.SensorObserver
		var err error

		switch sensorCfg.Type {
		case "optical":
			sensor, err = sensors.NewHanleOptical(sensorCfg)
		case "radar":
			sensor, err = sensors.NewNetraRadar(sensorCfg)
		case "rf":
			sensor, err = sensors.NewPassiveRF(sensorCfg)
		default:
			return fmt.Errorf("unknown sensor type: %s", sensorCfg.Type)
		}

		if err != nil {
			return fmt.Errorf("failed to initialize sensor %s: %w", sensorCfg.Name, err)
		}

		RegisterSensor(sensor)
	}

	return nil
}

// ShutdownSensors closes all sensor connections
func ShutdownSensors() {
	for _, sensor := range GetAllSensors() {
		if closer, ok := sensor.(interface{ Close() error }); ok {
			_ = closer.Close()
		}
	}
}
