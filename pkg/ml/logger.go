// pkg/ml/logger.go
// ADITI 4.0: Structured logging for ML inference pipeline
// Replaces fmt.Printf with leveled JSON logging for correlation with training metrics
//
// Benefits of using zap over standard library:
// - Performance: ~10x faster than standard library due to zero-allocation design
// - Structured fields: Key-value pairs for easy parsing by log aggregators (ELK, Loki, Datadog)
// - Level control: Debug, Info, Warn, Error levels for filtering
// - Sampling: Configurable sampling for high-volume logs
// - Stack traces: Automatic stack trace capture on errors
// - Correlation: Request IDs, trace IDs for distributed tracing

package ml

import (
	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"
)

var (
	// Global logger instance
	logger *zap.Logger
)

// InitLogger initializes the structured logger with JSON output
func InitLogger(level string) error {
	var zapLevel zapcore.Level
	switch level {
	case "debug":
		zapLevel = zapcore.DebugLevel
	case "info":
		zapLevel = zapcore.InfoLevel
	case "warn":
		zapLevel = zapcore.WarnLevel
	case "error":
		zapLevel = zapcore.ErrorLevel
	default:
		zapLevel = zapcore.InfoLevel
	}

	config := zap.Config{
		Level:            zap.NewAtomicLevelAt(zapLevel),
		Development:      false,
		Encoding:         "json",
		EncoderConfig:    zap.NewProductionEncoderConfig(),
		OutputPaths:      []string{"stdout"},
		ErrorOutputPaths: []string{"stderr"},
	}

	var err error
	logger, err = config.Build()
	if err != nil {
		return err
	}

	return nil
}

// GetLogger returns the global logger instance
func GetLogger() *zap.Logger {
	if logger == nil {
		// Fallback to production logger if not initialized
		logger, _ = zap.NewProduction()
	}
	return logger
}

// Sync flushes any buffered log entries (call before application shutdown)
func Sync() error {
	if logger != nil {
		return logger.Sync()
	}
	return nil
}

// LogInference logs model inference with tier and metrics
func LogInference(modelName string, tier int, r2 float64, conflict float64) {
	log := GetLogger()
	log.Info("model_inference",
		zap.String("model", modelName),
		zap.Int("tier", tier),
		zap.Float64("r2", r2),
		zap.Float64("conflict_metric", conflict),
	)
}

// LogTierShift logs tier switching events
func LogTierShift(fromTier, toTier int, reason string) {
	log := GetLogger()
	log.Info("tier_shift",
		zap.Int("from_tier", fromTier),
		zap.Int("to_tier", toTier),
		zap.String("reason", reason),
	)
}

// LogThermalEvent logs thermal governor events
func LogThermalEvent(level int, temperature float64, action string) {
	log := GetLogger()
	log.Warn("thermal_event",
		zap.Int("thermal_level", level),
		zap.Float64("temperature_c", temperature),
		zap.String("action", action),
	)
}

// LogModelLoad logs model loading events
func LogModelLoad(modelName string, path string, success bool, durationMs float64) {
	log := GetLogger()
	if success {
		log.Info("model_loaded",
			zap.String("model", modelName),
			zap.String("path", path),
			zap.Float64("load_duration_ms", durationMs),
		)
	} else {
		log.Error("model_load_failed",
			zap.String("model", modelName),
			zap.String("path", path),
			zap.Float64("load_duration_ms", durationMs),
		)
	}
}

// LogDebug logs debug-level messages
func LogDebug(msg string, fields ...zap.Field) {
	log := GetLogger()
	log.Debug(msg, fields...)
}

// LogInfo logs info-level messages
func LogInfo(msg string, fields ...zap.Field) {
	log := GetLogger()
	log.Info(msg, fields...)
}

// LogWarn logs warning-level messages
func LogWarn(msg string, fields ...zap.Field) {
	log := GetLogger()
	log.Warn(msg, fields...)
}

// LogError logs error-level messages
func LogError(msg string, fields ...zap.Field) {
	log := GetLogger()
	log.Error(msg, fields...)
}
