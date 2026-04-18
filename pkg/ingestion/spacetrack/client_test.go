package spacetrack

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
)

/**
 * TestClientLogin verifies that the Space-Track client correctly handles
 * the cookie-based authentication flow and rejection of invalid credentials.
 * <plan_complexity>Low</plan_complexity><total_lines_estimate>60</total_lines_estimate>
 */
func TestClientLogin(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/ajaxauth/login" {
			t.Errorf("expected login path, got %s", r.URL.Path)
		}
		
		username := r.FormValue("identity")
		password := r.FormValue("password")

		if username == "iaf_evaluator" && password == "secure_password" {
			http.SetCookie(w, &http.Cookie{Name: "session", Value: "valid_token", Path: "/"})
			w.WriteHeader(http.StatusOK)
			fmt.Fprint(w, "success")
		} else {
			w.WriteHeader(http.StatusOK)
			fmt.Fprint(w, "failed: invalid credentials")
		}
	}))
	defer ts.Close()

	client, _ := NewClient()
	client.baseURL = ts.URL

	// 1. Success case
	err := client.Login("iaf_evaluator", "secure_password")
	if err != nil {
		t.Errorf("expected success, got error: %v", err)
	}

	// 2. Failure case
	err = client.Login("wrong", "user")
	if err == nil {
		t.Error("expected failure for wrong credentials, got success")
	}
}

/**
 * TestQueryTLE verifies that the client can correctly parse a raw TLE string
 * response into a slice of orbital.TLE objects.
 */
func TestQueryTLE(t *testing.T) {
	// Generate valid TLE strings dynamically to satisfy checksum guards
	l1 := "1 25544U 98067A   24110.53673735  .00015569  00000-0  28135-3 0  999"
	l2 := "2 25544  51.6402 173.3400 0004554  58.2045  78.2913 15.4985651444983"
	
	l1 += fmt.Sprintf("%d", calculateTLEChecksum(l1))
	l2 += fmt.Sprintf("%d", calculateTLEChecksum(l2))
	mockTLE := l1 + "\n" + l2
	
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		fmt.Fprint(w, mockTLE)
	}))
	defer ts.Close()

	client, _ := NewClient()
	client.baseURL = ts.URL

	results, err := client.QueryTLE("NORAD_CAT_ID/25544")
	if err != nil {
		t.Fatalf("QueryTLE failed: %v", err)
	}

	if len(results) == 0 {
		t.Fatalf("expected 1 TLE, got 0")
	}
	if results[0].SatelliteNumber != 25544 {
		t.Errorf("expected satellite 25544, got %d", results[0].SatelliteNumber)
	}
}

func calculateTLEChecksum(line string) int {
	checksum := 0
	for i := 0; i < len(line) && i < 68; i++ {
		if line[i] >= '0' && line[i] <= '9' {
			checksum += int(line[i] - '0')
		} else if line[i] == '-' {
			checksum++
		}
	}
	return checksum % 10
}
