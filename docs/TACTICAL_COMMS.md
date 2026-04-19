# Tactical Communications Integration for ADITI 4.0

## Overview

This document describes the integration of tactical communication protocols from the Signals Handbook for Small Teams (Volume 1) into the ADITI 4.0 sovereign stack, providing human-in-the-loop communication capabilities for air-gapped defense systems.

## Architecture

### 1. C.R.A.P.S.H.O.O.T. Protocol (pkg/comms/tactical_messaging.go)

Implements the deployment routine for clear transmission discipline under fire.

**Protocol Steps:**
- **C**ompose: Mental rehearsal to keep messages short
- **R**elax: Message shortening and verification
- **A**ctivate: Keying the radio
- **P**ause: 1-second delay to prevent clipping
- **S**peak: Using "THIS IS" and callsigns in "You Are, I Am" format
- **H**ail: Delivering the actual message
- **O**ver/Out/Traffic: Strict proword usage

**Key Components:**
- `CRAPSHOOTProtocol`: State machine for transmission routine
- `CallSign`: Unit identifier with hierarchy support
- `TacticalMessage`: Radio transmission with proword
- `TacticalNet`: Network management
- `HierarchicalComms`: Squad-to-command structure

**Prowords:**
- `OVER`: Expecting reply
- `OUT`: No reply expected
- `TRAFFIC`: More messages to follow
- `ROGER`: Received and understood
- `WILCO`: Received and will comply
- `STANDBY`: Wait for instructions
- `SAYAGAIN`: Repeat transmission

**Thresholds:**
- `KeyupDelaySeconds`: 1.0
- `MaxMessageLength`: 160 characters
- `CallSignMaxLength`: 8 characters
- `TransmissionTimeout`: 10 seconds

### 2. DRYAD Authentication (pkg/comms/dryad_auth.go)

Implements paper-based two-factor authentication for infrastructure-independent security.

**Key Components:**
- `DRYADSheet`: Authentication sheet with authenticators
- `DRYADAuthenticator`: Single authenticator with grid coordinate
- `DRYADSystem`: Authentication management
- `AuthRecord`: Authentication history

**Authentication Flow:**
1. Challenge: "AUTHENTICATE: APPLE A-3"
2. Response: Verify authenticator matches grid coordinate
3. Validation: Check if authenticator already used
4. Logging: Record authentication attempt

**DRYAD Words (Standard Set):**
- APPLE, BRAVO, CHARLIE, DELTA, ECHO
- FOXTROT, GOLF, HOTEL, INDIA, JULIET
- KILO, LIMA, MIKE, NOVEMBER, OSCAR
- PAPA, QUEBEC, ROMEO, SIERRA, TANGO
- UNIFORM, VICTOR, WHISKEY, XRAY, YANKEE, ZULU

**Grid Coordinates:**
- Format: "A-3", "B-12" (row A-E, column 1-5)
- Each authenticator used only once
- Sheet expires after 24 hours

**Two-Factor Auth:**
- First factor: Password
- Second factor: DRYAD authenticator

### 3. Codebook and OTP Encryption (pkg/comms/codebook.go)

Implements numerical codes, OPCODEs, and One-Time Pads for message security.

**Key Components:**
- `Codebook`: Numerical/OPCODE codebook
- `CodebookEntry`: Single code-meaning pair
- `OTP`: One-Time Pad with random digits
- `EncryptionStack`: Multi-layer encryption

**Codebook Types:**
- `NUMERICAL`: 3-digit codes (e.g., 837 = "soldiers")
- `OPCODE`: 3-letter codes (e.g., DIM = "Direction of Movement")
- `MIXED`: Combined codes

**Standard Codes (Numerical):**
- `001-005`: Personnel (soldiers, officers, civilians, casualties, prisoners)
- `100-104`: Equipment (weapons, ammunition, vehicles, aircraft, comms)
- `200-204`: Actions (attack, defend, move, recon, withdraw)
- `300-304`: Status (ok, danger, unknown, confirmed, denied)
- `400-404`: Locations (base, forward, rear, airport, port)
- `500-504`: Time (now, soon, later, today, tomorrow)
- `600-604`: Space-specific (satellite, orbit, launch, debris, intercept)

**Standard OPCODEs:**
- `DIM`: Direction of Movement
- `SIT`: Situation Report
- `SPOT`: Spot Report
- `MED`: Medevac Request
- `LOG`: Logistics Report
- `INT`: Intelligence Report
- `FIRE`: Fire Mission
- `MOV`: Movement Order
- `SEC`: Security Report
- `COM`: Communications Report

**OTP Encryption:**
- Key length: 1000 digits
- Encryption: `ciphertext = (plaintext + key) mod 10`
- Decryption: `plaintext = (ciphertext - key + 10) mod 10`
- Mathematically unbreakable when handled properly

**Encryption Stack:**
- Supports multiple encryption layers
- Order: Codebook → OTP → Digital (optional)
- Reverse order for decryption

### 4. Standard Tactical Formats (pkg/comms/standard_formats.go)

Implements standardized reporting formats for structured data ingestion.

**SALUTE Report (Spot Report):**
- **S**ize: Unit size (S1=Squad, S2=Platoon, S3=Company)
- **A**ctivity: What the enemy is doing
- **L**ocation: Grid coordinates or description
- **U**nit: Enemy unit identification
- **T**ime: When observed
- **E**quipment: Weapons, vehicles, etc.

**SITREP (Situation Report):**
- **Line 1**: Enemy situation
- **Line 2**: Friendly situation
- **Line 3**: Logistics status
- **Line 4**: Personnel status
- **Line 5**: Weather/terrain
- **Line 6**: Recommendations
- **Line 7**: Next report time

**9-Line Medevac:**
- **Line 1**: Location
- **Line 2**: Radio frequency, callsign
- **Line 3**: Patients precedence (URGENT, PRIORITY, ROUTINE, TACTICAL IMMEDIATE)
- **Line 4**: Special equipment (NONE, VENTILATOR, BLOOD, LITTER, SUCTION)
- **Line 5**: Patients type (LITTER, AMBULATORY)
- **Line 6**: Security (NO ENEMY, ENEMY IN AREA, ENEMY APPROACHING, ENEMY FIRING)
- **Line 7**: Marking method
- **Line 8**: Patient nationality
- **Line 9**: CBRN contamination

**Additional Formats:**
- `SpotReport`: Generic observation report
- `IntelligenceReport`: Source, reliability, information, assessment
- `LogisticsReport`: Fuel, ammo, food, water, equipment, medical status

### 5. PACE Planning (pkg/comms/pace_planning.go)

Implements four-level redundancy for resilient communications.

**PACE Levels:**

| Level | Description | ADITI 4.0 Example | Success Rate | Latency |
|-------|-------------|------------------|--------------|---------|
| **Primary** | Main comms method | Reticulum Mesh (LXMF) | 95% | 100ms |
| **Alternate** | Secondary method | Handheld VHF Radio | 85% | 500ms |
| **Contingency** | Third method | Satellite Phone | 75% | 2s |
| **Emergency** | Last resort | Physical Signal | 50% | 30min |

**Space DSA PACE:**
- **Primary**: Reticulum Mesh (LXMF) - 10 km range
- **Alternate**: HF Radio with ALE - 500 km range
- **Contingency**: Satellite Uplink - Global
- **Emergency**: Morse Code/Signal Lamp - 50 km visual

**Key Features:**
- Automatic failover
- Manual failover support
- Health checking for all levels
- Redundancy calculator
- Expected latency computation

**Reliability Calculation:**
```
Overall Reliability = 1 - (1-P₁)(1-P₂)(1-P₃)(1-Pₑ)
```

Where P₁, P₂, P₃, Pₑ are success rates for each level.

## Integration Flow

### Complete Tactical Communication Flow

```
[Message Composition] → [C.R.A.P.S.H.O.O.T.] → [DRYAD Auth] → 
[Codebook Encode] → [OTP Encrypt] → [PACE Selection] → 
[Transmission] → [Proword] → [Receipt Confirmation]
```

### Data Flow

```
[Operator Input] → [TacticalMessage] → [EncryptionStack] → 
[PACEMethod] → [Reticulum Mesh] → [Receiver] → 
[Decryption] → [StandardFormat] → [TimescaleDB]
```

## Usage Examples

### C.R.A.P.S.H.O.O.T. Transmission

```go
// Create protocol handler
callsign := &comms.CallSign{
    Unit:   "ALPHA",
    Number: "1",
    Parent: "COMMAND",
}
protocol := comms.NewCRAPSHOOTProtocol(callsign)

// Create recipient
to := &comms.CallSign{
    Unit:   "BRAVO",
    Number: "2",
}

// Transmit message
message, err := protocol.Transmit(to, "Enemy squad at grid 123-456", "OVER")
if err != nil {
    log.Fatal(err)
}

fmt.Printf("Transmitted: %s\n", comms.FormatMessage(message))
```

### DRYAD Authentication

```go
// Create DRYAD system
dryad := comms.NewDRYADSystem()

// Generate sheet
sheet := dryad.GenerateSheet("ALPHA-1", "COMMAND")

// Authenticate
valid, err := dryad.Authenticate("APPLE", "A-3", "ALPHA-1")
if err != nil {
    log.Fatal(err)
}

if valid {
    fmt.Println("Authentication successful")
}
```

### Codebook Encoding

```go
// Generate standard codebook
cb := comms.GenerateStandardCodebook("CB-001", "ALPHA-1", "COMMAND")

// Encode message
encoded, err := cb.Encode("SOLDIERS ATTACK BASE")
if err != nil {
    log.Fatal(err)
}

fmt.Printf("Encoded: %s\n", encoded) // "001 200 400"
```

### OTP Encryption

```go
// Create OTP
otp, err := comms.NewOTP("OTP-001", 1000)
if err != nil {
    log.Fatal(err)
}

// Encrypt numerical message
ciphertext, err := otp.Encrypt("12345")
if err != nil {
    log.Fatal(err)
}

fmt.Printf("Encrypted: %s\n", ciphertext)
```

### SALUTE Report

```go
// Create SALUTE report
salute := comms.NewSALUTE("ALPHA-1")
salute.Size = "S2"
salute.Activity = "Patrolling north"
salute.Location = "Grid 123-456"
salute.Unit = "Unknown"
salute.Time = "1500Z"
salute.Equipment = "Rifles, vehicles"
salute.Priority = 3

fmt.Printf("%s\n", salute.Format())
```

### SITREP

```go
// Create SITREP
sitrep := comms.NewSITREP("ALPHA-1")
sitrep.Line1 = "Enemy squad 2km north"
sitrep.Line2 = "Alpha company holding position"
sitrep.Line3 = "Ammo 80%, fuel 70%"
sitrep.Line4 = "All personnel accounted for"
sitrep.Line5 = "Clear, visibility 10km"
sitrep.Line6 = "Request reinforcements"
sitrep.Line7 = "1800Z"

fmt.Printf("%s\n", sitrep.Format())
```

### 9-Line Medevac

```go
// Create Medevac request
medevac := comms.NewMedevacLine()
medevac.Line1 = "Grid 123-456"
medevac.Line2 = "51.0 MHz ALPHA-1"
medevac.Line3 = comms.PrecedenceUrgent
medevac.Line4 = comms.EquipVentilator
medevac.Line5 = comms.PatientLitter
medevac.Line6 = comms.SecurityEnemyPresent
medevac.Line7 = "Orange smoke"
medevac.Line8 = "Indian"
medevac.Line9 = "None"

if err := medevac.Validate(); err != nil {
    log.Fatal(err)
}

fmt.Printf("%s\n", medevac.Format())
```

### PACE Planning

```go
// Create ADITI 4.0 PACE plan
plan := comms.CreateDefaultADITI4PACE()

// Test all methods
results := plan.TestAllMethods()
fmt.Printf("Primary: %v, Alternate: %v\n", results[comms.PACEPrimary], results[comms.PACEAlternate])

// Perform failover
level, method, err := plan.Failover()
if err != nil {
    log.Fatal(err)
}

fmt.Printf("Failed over to %s: %s\n", level, method.Name)
```

## Strategic Payoff

### 1. Infrastructure Independence
- **DRYAD**: Paper-based authentication works without electronics
- **Codebooks**: Manual encoding/decoding possible
- **PACE**: Physical fallback methods (whistle, flare, courier)

### 2. Human-in-the-Loop
- **C.R.A.P.S.H.O.O.T.**: Standardized transmission discipline
- **Standard Formats**: Structured reporting for clear communication
- **Prowords**: Unambiguous procedural language

### 3. Air-Gapped Security
- **OTP**: Mathematically unbreakable encryption
- **Codebooks**: Obscures meaning without electronic dependency
- **DRYAD**: Two-factor authentication without digital infrastructure

### 4. Resilience
- **PACE**: Four-level redundancy ensures communication in any scenario
- **Auto-Failover**: Automatic switching to backup methods
- **Health Checking**: Continuous monitoring of communication paths

## File Structure

```
pkg/comms/
├── tactical_messaging.go  # C.R.A.P.S.H.O.O.T. protocol
├── dryad_auth.go          # DRYAD authentication
├── codebook.go            # Codebook and OTP encryption
├── standard_formats.go    # SALUTE, SITREP, 9-Line Medevac
└── pace_planning.go       # PACE redundancy
```

## Constants Reference

### C.R.A.P.S.H.O.O.T.
```go
KeyupDelaySeconds     = 1.0
MaxMessageLength      = 160
CallSignMaxLength    = 8
TransmissionTimeout  = 10 * time.Second
```

### DRYAD
```go
AuthenticatorLength = 6
GridSize           = 25
MaxAuthenticators  = 100
AuthTTL           = 24 * time.Hour
```

### Codebook
```go
CodeLength      = 3
OPCODELength    = 3
OTPLength       = 1000
MaxCodebookSize = 1000
```

### PACE
```go
PACEPrimary     = "PRIMARY"
PACEAlternate   = "ALTERNATE"
PACEContingency = "CONTINGENCY"
PACEEmergency   = "EMERGENCY"
```

## Dependencies

- Go: `time`, `fmt`, `strings`, `math/rand`, `crypto/rand`
- External: None (all Go-native)
- Hardware: RTL-SDR (optional), VHF radios (optional), satellite phones (optional)

## Notes

- All tactical comms are Go-native for real-time performance
- DRYAD authentication is paper-based and infrastructure-independent
- OTP encryption is mathematically unbreakable when handled properly
- Standard formats integrate with TimescaleDB for structured data
- PACE planning ensures communication resilience in all scenarios
- C.R.A.P.S.H.O.O.T. protocol ensures clear transmission under fire

## Integration with Existing Systems

### Reticulum Mesh
- **Primary PACE**: Reticulum Mesh (LXMF)
- **Integration**: Use existing `pkg/mesh/rns_bridge.go`
- **Encryption**: LXMF + DRYAD + Codebook + OTP

### TimescaleDB
- **Data Ingestion**: SALUTE, SITREP, Spot Reports
- **Structured Storage**: Standardized formats for querying
- **Historical Analysis**: Communication patterns and trends

### Fire-Control Handover
- **Tactical Alert**: Use C.R.A.P.S.H.O.O.T. for FCH alerts
- **Encryption**: OTP for critical FCH data
- **PACE**: Ensure FCH delivery via backup methods

## Next Steps

1. **Field Testing**: Deploy at Siliguri Corridor for real-world validation
2. **Training**: Train operators on C.R.A.P.S.H.O.O.T. protocol
3. **DRYAD Distribution**: Print and distribute authentication sheets
4. **Codebook Distribution**: Issue standard codebooks to units
5. **PACE Validation**: Test all four levels in field conditions
6. **Integration**: Connect with existing Reticulum mesh and TimescaleDB

## Summary

The Tactical Communications integration transforms ADITI 4.0 into a complete air-gappable defense system:

- **C.R.A.P.S.H.O.O.T.**: Standardized transmission discipline
- **DRYAD**: Infrastructure-independent authentication
- **Codebook/OTP**: Mathematically unbreakable encryption
- **Standard Formats**: Structured reporting (SALUTE, SITREP, 9-Line Medevac)
- **PACE**: Four-level redundancy for resilience

This provides the IAF with sovereign tactical communication capabilities that work in the most austere environments, from electronic warfare scenarios to complete infrastructure failure.
