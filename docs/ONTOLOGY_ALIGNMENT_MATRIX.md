# BOS-4.0 Ontology Alignment Matrix

This matrix executes Phase A of OPEN_SOURCE_COMSPOC_ALIGNMENT.md by mapping BOS ontology entities to Orbital Space Ontology style classes.

| BOS Type | Canonical Class | Required Fields | Validation Status |
|---|---|---|---|
| SpaceObject | SpaceObject | id, source, timestamp_utc, state[6], covariance[6x6] | Enforced via `ValidateSpaceObjectSchema` |
| Conjunction | Event.Conjunction | id, event_type, source, timestamp_utc, object1, object2, tca, miss_distance_km, pc | Enforced via `ValidateEventSchema` + type checks |
| Manoeuvre | Event.Maneuver | id, event_type, source, timestamp_utc, object_id, delta_v, burn_time | Enforced via `ValidateEventSchema` |
| RPOManeuver | Event.RPO | id, event_type, source, timestamp_utc, object_id, relative_state, proximity_km, risk_level | Enforced via `ValidateEventSchema` |
| ReEntryFootprint | Event.ReEntry | id, event_type, source, timestamp_utc, object_id, impact_time, footprint_polygon | Enforced via `ValidateEventSchema` |
| CoverageGap | Event.CoverageGap | id, event_type, source, timestamp_utc, region, start_time, end_time | Enforced via `ValidateEventSchema` |
| EOPData | Environment.EOPSnapshot | date_utc, polar_motion_x_arcsec, polar_motion_y_arcsec, ut1_utc_seconds, source, timestamp_utc | Loader/runtime checks |

## Alias Policy

- Maneuver spelling aliases: `Manoeuvre` (BOS existing) and `Maneuver` (external naming) are treated as semantic equivalents in integration docs.
- RPO terminology aliases: `RPOManeuver` and `ProximityOps` events are treated as equivalent event family labels.

## Implemented Validators

- `pkg/ontology/schema_validation.go`
  - `ValidateSpaceObjectSchema`
  - `ValidateEventSchema`
- `pkg/ontology/schema_validation_test.go`

## Notes

- Validation is intentionally minimal and strict-mode-safe: it enforces required contracts without over-constraining optional metadata fields.
- Timestamp normalization is required in UTC for all canonical ontology objects and events.
