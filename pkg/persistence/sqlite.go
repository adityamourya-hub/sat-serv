package persistence

import (
	"bytes"
	"database/sql"
	"encoding/gob"
	"fmt"
	"time"

	"ssis/pkg/catalog"

	_ "modernc.org/sqlite"
)

type SQLiteBackend struct {
	db *sql.DB
}

func NewSQLiteBackend(path string) (*SQLiteBackend, error) {
	if path == "" {
		path = "ssis_catalog_sql.db"
	}

	db, err := sql.Open("sqlite", path)
	if err != nil {
		return nil, err
	}

	// Create tables with indexes
	queries := []string{
		`CREATE TABLE IF NOT EXISTS objects (
			id INTEGER PRIMARY KEY,
			name TEXT,
			country_code TEXT,
			asset_type TEXT,
			status TEXT,
			pos_x REAL,
			pos_y REAL,
			pos_z REAL,
			vel_x REAL,
			vel_y REAL,
			vel_z REAL,
			epoch_tt REAL,
			mass_kg REAL,
			drag_area REAL,
			srp_area REAL,
			od_quality REAL,
			sensor_id TEXT,
			last_update_unix INTEGER,
			version INTEGER,
			tle_line1 TEXT,
			tle_line2 TEXT
		)`,
		`CREATE INDEX IF NOT EXISTS idx_objects_epoch ON objects(epoch_tt)`,
		`CREATE TABLE IF NOT EXISTS history (
			id INTEGER,
			timestamp REAL,
			data BLOB,
			PRIMARY KEY (id, timestamp)
		)`,
	}

	for _, q := range queries {
		if _, err := db.Exec(q); err != nil {
			db.Close()
			return nil, err
		}
	}

	return &SQLiteBackend{db: db}, nil
}

func (b *SQLiteBackend) Close() error {
	return b.db.Close()
}

func (b *SQLiteBackend) UpsertObject(r catalog.SDAObject) error {
	tx, err := b.db.Begin()
	if err != nil {
		return err
	}
	defer tx.Rollback()

	if _, err := tx.Exec(`
		INSERT INTO objects (id, name, country_code, asset_type, status, pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, epoch_tt, mass_kg, drag_area, srp_area, od_quality, sensor_id, last_update_unix, version, tle_line1, tle_line2)
		VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
		ON CONFLICT(id) DO UPDATE SET
			name=excluded.name,
			country_code=excluded.country_code,
			asset_type=excluded.asset_type,
			status=excluded.status,
			pos_x=excluded.pos_x, pos_y=excluded.pos_y, pos_z=excluded.pos_z,
			vel_x=excluded.vel_x, vel_y=excluded.vel_y, vel_z=excluded.vel_z,
			epoch_tt=excluded.epoch_tt,
			mass_kg=excluded.mass_kg,
			drag_area=excluded.drag_area,
			srp_area=excluded.srp_area,
			od_quality=excluded.od_quality,
			sensor_id=excluded.sensor_id,
			last_update_unix=excluded.last_update_unix,
			version=excluded.version,
			tle_line1=excluded.tle_line1,
			tle_line2=excluded.tle_line2
	`, r.ID, r.Name, r.CountryCode, r.AssetType, string(r.Status), r.Pos.X, r.Pos.Y, r.Pos.Z, r.Vel.X, r.Vel.Y, r.Vel.Z, r.EpochTT, r.MassKg, r.DragArea, r.SRPArea, r.ODQualityScore, r.SensorID, r.LastUpdateUnix, r.Version, r.TLELine1, r.TLELine2); err != nil {
		return err
	}

	var buf bytes.Buffer
	if err := gob.NewEncoder(&buf).Encode(r); err != nil {
		return err
	}
	if _, err := tx.Exec(`
		INSERT INTO history (id, timestamp, data)
		VALUES (?, ?, ?)
		ON CONFLICT(id, timestamp) DO UPDATE SET data=excluded.data
	`, r.ID, r.EpochTT, buf.Bytes()); err != nil {
		return err
	}

	return tx.Commit()
}

func (b *SQLiteBackend) DeleteObject(id int) error {
	_, err := b.db.Exec("DELETE FROM objects WHERE id=?", id)
	return err
}

func (b *SQLiteBackend) GetLatestSnapshot() ([]catalog.SDAObject, error) {
	rows, err := b.db.Query("SELECT id, name, country_code, asset_type, status, pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, epoch_tt, mass_kg, drag_area, srp_area, od_quality, sensor_id, last_update_unix, version, tle_line1, tle_line2 FROM objects")
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var records []catalog.SDAObject
	for rows.Next() {
		var r catalog.SDAObject
		var status string
		err := rows.Scan(&r.ID, &r.Name, &r.CountryCode, &r.AssetType, &status, &r.Pos.X, &r.Pos.Y, &r.Pos.Z, &r.Vel.X, &r.Vel.Y, &r.Vel.Z, &r.EpochTT, &r.MassKg, &r.DragArea, &r.SRPArea, &r.ODQualityScore, &r.SensorID, &r.LastUpdateUnix, &r.Version, &r.TLELine1, &r.TLELine2)
		if err == nil {
			r.Status = catalog.ObjectStatus(status)
			records = append(records, r)
		}
	}
	return records, nil
}

func (b *SQLiteBackend) GetObjectAtTime(id int, t time.Time) (catalog.SDAObject, error) {
	row := b.db.QueryRow(`
		SELECT data
		FROM history
		WHERE id = ? AND timestamp <= ?
		ORDER BY timestamp DESC
		LIMIT 1
	`, id, float64(t.Unix()))

	var blob []byte
	if err := row.Scan(&blob); err != nil {
		if err == sql.ErrNoRows {
			return catalog.SDAObject{}, fmt.Errorf("no historical record found for ID %d at time %v", id, t)
		}
		return catalog.SDAObject{}, err
	}

	var rec catalog.SDAObject
	if err := gob.NewDecoder(bytes.NewReader(blob)).Decode(&rec); err != nil {
		return catalog.SDAObject{}, err
	}
	return rec, nil
}

func (b *SQLiteBackend) QueryObjectsInWindow(start, end time.Time) ([]catalog.SDAObject, error) {
	rows, err := b.db.Query(`
		SELECT data
		FROM history
		WHERE timestamp >= ? AND timestamp <= ?
		ORDER BY timestamp ASC
	`, float64(start.Unix()), float64(end.Unix()))
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	records := make([]catalog.SDAObject, 0)
	for rows.Next() {
		var blob []byte
		if err := rows.Scan(&blob); err != nil {
			continue
		}
		var rec catalog.SDAObject
		if err := gob.NewDecoder(bytes.NewReader(blob)).Decode(&rec); err == nil {
			records = append(records, rec)
		}
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	return records, nil
}

func (b *SQLiteBackend) PersistHypothesis(hyp *catalog.GlobalHypothesis) error {
	// MHT SQL table creation and insert logic...
	return nil
}
