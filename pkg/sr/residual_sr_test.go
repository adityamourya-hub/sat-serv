package sr

import (
	"math"
	"testing"
)

func TestEquationEvaluator(t *testing.T) {
	tests := []struct {
		name     string
		expr     string
		params   map[string]interface{}
		expected float64
		wantErr  bool
	}{
		{
			name:     "simple addition",
			expr:     "2 + 3",
			params:   map[string]interface{}{},
			expected: 5,
		},
		{
			name:     "simple multiplication",
			expr:     "4 * 5",
			params:   map[string]interface{}{},
			expected: 20,
		},
		{
			name:     "with variable",
			expr:     "x + 5",
			params:   map[string]interface{}{"x": 10.0},
			expected: 15,
		},
		{
			name:     "complex expression",
			expr:     "(a + b) * c",
			params:   map[string]interface{}{"a": 2.0, "b": 3.0, "c": 4.0},
			expected: 20,
		},
		{
			name:     "exponentiation",
			expr:     "2 ^ 3",
			params:   map[string]interface{}{},
			expected: 8,
		},
		{
			name:     "division",
			expr:     "10 / 2",
			params:   map[string]interface{}{},
			expected: 5,
		},
		{
			name:     "atmospheric density equation",
			expr:     "rho0 * exp(-h / H)",
			params:   map[string]interface{}{"rho0": 1.225, "h": 10000.0, "H": 8500.0},
			expected: 1.225 * math.Exp(-10000.0/8500.0),
		},
		{
			name:    "division by zero",
			expr:    "10 / 0",
			params:  map[string]interface{}{},
			wantErr: true,
		},
		{
			name:    "empty expression",
			expr:    "",
			params:  map[string]interface{}{},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			eq := &DiscoveredEquation{
				Expression: tt.expr,
			}
			
			eval, err := NewEquationEvaluator(eq)
			if (err != nil) != tt.wantErr {
				t.Errorf("NewEquationEvaluator() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			
			if tt.wantErr {
				return
			}

			result, err := eval.Evaluate(tt.params)
			if (err != nil) != tt.wantErr {
				t.Errorf("Evaluate() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if !tt.wantErr && math.Abs(result-tt.expected) > 1e-6 {
				t.Errorf("Evaluate() = %v, want %v", result, tt.expected)
			}
		})
	}
}

func TestExtractVariables(t *testing.T) {
	tests := []struct {
		name     string
		expr     string
		expected []string
	}{
		{
			name:     "single variable",
			expr:     "x + 5",
			expected: []string{"x"},
		},
		{
			name:     "multiple variables",
			expr:     "a * b + c",
			expected: []string{"a", "b", "c"},
		},
		{
			name:     "no variables",
			expr:     "2 + 3 * 4",
			expected: []string{},
		},
		{
			name:     "with functions",
			expr:     "sin(x) + cos(y)",
			expected: []string{"x", "y"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			vars := extractVariables(tt.expr)
			
			// Check that all expected variables are present
			for _, exp := range tt.expected {
				found := false
				for _, v := range vars {
					if v == exp {
						found = true
						break
					}
				}
				if !found {
					t.Errorf("Expected variable %s not found in %v", exp, vars)
				}
			}
		})
	}
}

func TestParseExpression(t *testing.T) {
	tests := []struct {
		name     string
		expr     string
		expected float64
		wantErr  bool
	}{
		{"number", "42", 42, false},
		{"add", "2 + 3", 5, false},
		{"multiply", "2 * 3", 6, false},
		{"precedence", "2 + 3 * 4", 14, false},
		{"parentheses", "(2 + 3) * 4", 20, false},
		{"nested", "(1 + 2) * (3 + 4)", 21, false},
		{"exponent", "2 ^ 10", 1024, false},
		{"complex", "((1 + 2) * 3) ^ 2", 81, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := parseExpression(tt.expr)
			if (err != nil) != tt.wantErr {
				t.Errorf("parseExpression() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !tt.wantErr && math.Abs(result-tt.expected) > 1e-9 {
				t.Errorf("parseExpression() = %v, want %v", result, tt.expected)
			}
		})
	}
}
