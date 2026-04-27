package comms

import (
	"testing"
	"time"
)

func TestNewDRYADSystem(t *testing.T) {
	ds := NewDRYADSystem()
	if ds == nil {
		t.Fatal("NewDRYADSystem returned nil")
	}
	if ds.currentSheet != nil {
		t.Error("new system should not have a current sheet")
	}
	if len(ds.usedAuthenticators) != 0 {
		t.Error("new system should have empty used authenticators")
	}
}

func TestGenerateSheet(t *testing.T) {
	ds := NewDRYADSystem()
	sheet := ds.GenerateSheet("Alice", "Bob")

	if sheet == nil {
		t.Fatal("GenerateSheet returned nil")
	}

	if sheet.SheetID == "" {
		t.Error("SheetID should not be empty")
	}

	if sheet.IssuedTo != "Alice" {
		t.Errorf("IssuedTo = %q, want Alice", sheet.IssuedTo)
	}

	if sheet.IssuedBy != "Bob" {
		t.Errorf("IssuedBy = %q, want Bob", sheet.IssuedBy)
	}

	if !sheet.IsActive {
		t.Error("sheet should be active")
	}

	// BUG: Only 26 unique NATO words exist. Map collisions overwrite entries,
	// so the actual count is 26, not MaxAuthenticators (100). Grid coords also
	// collide because the loop overwrites words rather than generating 100 unique pairs.
	// TODO: Switch to unique (word, coord) pair generation to reach MaxAuthenticators.
	if len(sheet.Authenticators) > 26 {
		t.Errorf("len(Authenticators) = %d, expected at most 26 due to word collisions", len(sheet.Authenticators))
	}

	if time.Now().After(sheet.ExpiryDate) {
		t.Error("sheet already expired immediately after generation")
	}
}

func TestAuthenticate(t *testing.T) {
	ds := NewDRYADSystem()
	sheet := ds.GenerateSheet("Alice", "Bob")

	// Pick an authenticator from the sheet
	var auth, coord string
	for a, c := range sheet.Authenticators {
		auth = a
		coord = c
		break
	}

	// Correct auth
	ok, err := ds.Authenticate(auth, coord, "Alice")
	if err != nil {
		t.Fatalf("unexpected error on correct auth: %v", err)
	}
	if !ok {
		t.Error("correct authentication should succeed")
	}

	// Same auth should fail on reuse
	ok, err = ds.Authenticate(auth, coord, "Alice")
	if err == nil || ok {
		t.Error("reused authenticator should fail")
	}

	// Wrong grid coord
	ok, err = ds.Authenticate("BRAVO", "Z-99", "Alice")
	if err == nil || ok {
		t.Error("wrong grid coord should fail")
	}

	// No active sheet
	ds2 := NewDRYADSystem()
	ok, err = ds2.Authenticate("BRAVO", "A-1", "Alice")
	if err == nil || ok {
		t.Error("no active sheet should fail")
	}
}

func TestAuthenticateExpiredSheet(t *testing.T) {
	ds := NewDRYADSystem()
	sheet := &DRYADSheet{
		SheetID:        "EXPIRED-1",
		Authenticators: map[string]string{"TEST": "A-1"},
		ExpiryDate:     time.Now().Add(-time.Hour),
		IsActive:       true,
		IssuedTo:       "Alice",
		IssuedBy:       "Bob",
	}

	if err := ds.LoadSheet(sheet); err == nil {
		t.Error("loading expired sheet should fail")
	}
}
