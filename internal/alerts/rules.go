// Package alerts provides the rules engine for the sovereign alerting system.
package alerts

import (
	"fmt"
	"os"
	"os/signal"
	"sync"
	"syscall"

	"gopkg.in/yaml.v3"
)

// Rule defines a single alert rule loaded from YAML.
type Rule struct {
	ID          string     `yaml:"id"`
	Category    Category   `yaml:"category"`
	Severity    Severity   `yaml:"severity"`
	Enabled     bool       `yaml:"enabled"`
	Description string     `yaml:"description"`
	Condition   Condition  `yaml:"condition"`
	Escalation  Escalation `yaml:"escalation"`
}

// Condition specifies the metric and threshold that triggers the rule.
type Condition struct {
	Metric    string  `yaml:"metric"`
	Operator  string  `yaml:"operator"` // ">=", "<=", ">", "<", "=="
	Threshold float64 `yaml:"threshold"`
}

// Escalation defines auto-escalation behavior.
type Escalation struct {
	TimeoutMin  int      `yaml:"escalation_timeout_min"`
	EscalateTo  Severity `yaml:"escalate_to"`
}

// RuleConfig is the top-level structure of the YAML file.
type RuleConfig struct {
	Rules     []Rule `yaml:"rules"`
	Settings  Settings `yaml:"settings"`
	Metadata  Metadata `yaml:"metadata"`
}

// Settings contains global configuration.
type Settings struct {
	EvaluationIntervalSec int    `yaml:"evaluation_interval_sec"`
	MaxAlertsPerMinute   int    `yaml:"max_alerts_per_minute"`
	DedupWindowSec       int    `yaml:"dedup_window_sec"`
	DefaultEscalationMin int    `yaml:"default_escalation_min"`
	AuditEveryAlert      bool   `yaml:"audit_every_alert"`
	SignEveryAlert       bool   `yaml:"sign_every_alert"`
}

// Metadata tracks version and authority.
type Metadata struct {
	Version     string `yaml:"version"`
	LastUpdated string `yaml:"last_updated"`
	Authority   string `yaml:"authority"`
}

// RulesEngine loads and evaluates alert rules from YAML.
type RulesEngine struct {
	mu          sync.RWMutex
	config      *RuleConfig
	configPath  string
	rulesByCategory map[Category][]*Rule
}

// NewRulesEngine creates a new rules engine and loads the initial config.
func NewRulesEngine(configPath string) (*RulesEngine, error) {
	engine := &RulesEngine{
		configPath:      configPath,
		rulesByCategory: make(map[Category][]*Rule),
	}

	if err := engine.Load(); err != nil {
		return nil, err
	}

	// Set up hot-reload on SIGHUP
	go engine.watchForReload()

	return engine, nil
}

// Load reads and parses the YAML config file.
func (r *RulesEngine) Load() error {
	r.mu.Lock()
	defer r.mu.Unlock()

	data, err := os.ReadFile(r.configPath)
	if err != nil {
		return fmt.Errorf("failed to read rules config: %w", err)
	}

	var config RuleConfig
	if err := yaml.Unmarshal(data, &config); err != nil {
		return fmt.Errorf("failed to parse rules config: %w", err)
	}

	r.config = &config

	// Index rules by category for faster lookup
	r.rulesByCategory = make(map[Category][]*Rule)
	for i := range config.Rules {
		rule := &config.Rules[i]
		r.rulesByCategory[rule.Category] = append(r.rulesByCategory[rule.Category], rule)
	}

	return nil
}

// watchForReload triggers a reload on SIGHUP signal.
func (r *RulesEngine) watchForReload() {
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGHUP)

	for range sigChan {
		if err := r.Load(); err != nil {
			fmt.Printf("[WARN] Failed to reload rules: %v\n", err)
		} else {
			fmt.Printf("[INFO] Rules reloaded successfully from %s\n", r.configPath)
		}
	}
}

// Evaluate checks an incoming alert against all enabled rules and returns the highest-severity match.
func (r *RulesEngine) Evaluate(category Category, metrics map[string]float64) (*Rule, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	if r.config == nil {
		return nil, false
	}

	var bestMatch *Rule
	var bestSeverityOrder = -1

	for _, rule := range r.rulesByCategory[category] {
		if !rule.Enabled {
			continue
		}

		// Check if condition matches
		if r.conditionMatches(rule.Condition, metrics) {
			severityOrder := SeverityOrder(rule.Severity)
			if severityOrder > bestSeverityOrder {
				bestMatch = rule
				bestSeverityOrder = severityOrder
			}
		}
	}

	return bestMatch, bestMatch != nil
}

// conditionMatches checks if the rule condition is satisfied by the metrics.
func (r *RulesEngine) conditionMatches(cond Condition, metrics map[string]float64) bool {
	value, ok := metrics[cond.Metric]
	if !ok {
		return false
	}

	switch cond.Operator {
	case ">=":
		return value >= cond.Threshold
	case "<=":
		return value <= cond.Threshold
	case ">":
		return value > cond.Threshold
	case "<":
		return value < cond.Threshold
	case "==":
		return value == cond.Threshold
	default:
		return false
	}
}

// GetRules returns all rules (for admin dashboard).
func (r *RulesEngine) GetRules() []Rule {
	r.mu.RLock()
	defer r.mu.RUnlock()

	if r.config == nil {
		return nil
	}

	rules := make([]Rule, len(r.config.Rules))
	copy(rules, r.config.Rules)
	return rules
}

// GetSettings returns global settings.
func (r *RulesEngine) GetSettings() Settings {
	r.mu.RLock()
	defer r.mu.RUnlock()

	if r.config == nil {
		return Settings{}
	}
	return r.config.Settings
}

// GetMetadata returns version info.
func (r *RulesEngine) GetMetadata() Metadata {
	r.mu.RLock()
	defer r.mu.RUnlock()

	if r.config == nil {
		return Metadata{}
	}
	return r.config.Metadata
}
