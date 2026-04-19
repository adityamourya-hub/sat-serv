// Package postgis provides PostgreSQL/PostGIS connection management for ISDA
// IAF SDLC: All database operations use parameterized queries to prevent SQL injection
package postgis

import (
	"context"
	"fmt"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

// Config holds database connection parameters
type Config struct {
	Host     string
	Port     int
	Database string
	User     string
	Password string
	SSLMode  string

	// Pool settings
	MaxConns          int32         // Maximum connections in pool
	MinConns          int32         // Minimum idle connections
	MaxConnLifetime   time.Duration // Maximum lifetime of a connection
	MaxConnIdleTime   time.Duration // Maximum idle time
	HealthCheckPeriod time.Duration
}

// DefaultConfig returns IAF production-ready defaults
func DefaultConfig() *Config {
	return &Config{
		Host:              "localhost",
		Port:              5432,
		Database:          "isda",
		User:              "isda_user",
		Password:          "", // Set from environment variable
		SSLMode:           "require",
		MaxConns:          25,
		MinConns:          5,
		MaxConnLifetime:   time.Hour,
		MaxConnIdleTime:   30 * time.Minute,
		HealthCheckPeriod: 5 * time.Minute,
	}
}

// Pool wraps pgxpool with ISDA-specific configuration
type Pool struct {
	*pgxpool.Pool
	config *Config
}

// NewPool creates a connection pool with IAF hardening
func NewPool(cfg *Config) (*Pool, error) {
	if cfg == nil {
		cfg = DefaultConfig()
	}

	connString := fmt.Sprintf(
		"host=%s port=%d dbname=%s user=%s password=%s sslmode=%s",
		cfg.Host, cfg.Port, cfg.Database, cfg.User, cfg.Password, cfg.SSLMode,
	)

	poolConfig, err := pgxpool.ParseConfig(connString)
	if err != nil {
		return nil, fmt.Errorf("parse connection string: %w", err)
	}

	// Apply IAF security and performance settings
	poolConfig.MaxConns = cfg.MaxConns
	poolConfig.MinConns = cfg.MinConns
	poolConfig.MaxConnLifetime = cfg.MaxConnLifetime
	poolConfig.MaxConnIdleTime = cfg.MaxConnIdleTime
	poolConfig.HealthCheckPeriod = cfg.HealthCheckPeriod

	// Enable prepared statement cache for performance
	poolConfig.ConnConfig.DefaultQueryExecMode = pgx.QueryExecModeCacheDescribe

	pool, err := pgxpool.NewWithConfig(context.Background(), poolConfig)
	if err != nil {
		return nil, fmt.Errorf("create connection pool: %w", err)
	}

	// Verify connectivity
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	if err := pool.Ping(ctx); err != nil {
		pool.Close()
		return nil, fmt.Errorf("database ping failed: %w", err)
	}

	return &Pool{Pool: pool, config: cfg}, nil
}

// Close gracefully shuts down the connection pool
func (p *Pool) Close() {
	if p.Pool != nil {
		p.Pool.Close()
	}
}

// HealthCheck performs a comprehensive database health check
func (p *Pool) HealthCheck(ctx context.Context) error {
	// Basic connectivity
	if err := p.Ping(ctx); err != nil {
		return fmt.Errorf("ping failed: %w", err)
	}

	// Verify PostGIS extension
	var postgisVersion string
	err := p.QueryRow(ctx, "SELECT PostGIS_Version()").Scan(&postgisVersion)
	if err != nil {
		return fmt.Errorf("postgis extension not available: %w", err)
	}

	// Verify TimescaleDB extension
	var timescaleVersion string
	err = p.QueryRow(ctx, "SELECT extversion FROM pg_extension WHERE extname = 'timescaledb'").Scan(&timescaleVersion)
	if err != nil {
		return fmt.Errorf("timescaledb extension not available: %w", err)
	}

	// Check core tables exist
	tables := []string{"current_states", "orbital_trends", "asset_metadata", "threat_vector"}
	for _, table := range tables {
		var exists bool
		err := p.QueryRow(ctx,
			"SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = $1)",
			table,
		).Scan(&exists)
		if err != nil || !exists {
			return fmt.Errorf("required table %s not found", table)
		}
	}

	return nil
}

// Stats returns pool statistics for monitoring
func (p *Pool) Stats() *pgxpool.Stat {
	return p.Pool.Stat()
}
