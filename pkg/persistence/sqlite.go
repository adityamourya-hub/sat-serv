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
			pos_x REAL,
			pos_y REAL,
			pos_z REAL,
			vel_x REAL,
			vel_y REAL,
			vel_z REAL,
			pos_sigma_km REAL,
			last_update_unix INTEGER,
			version INTEGER,
			timestamp REAL,
			tle_line1 TEXT,
			tle_line2 TEXT
		)`,
		`CREATE INDEX IF NOT EXISTS idx_objects_timestamp ON objects(timestamp)`,
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

func (b *SQLiteBackend) UpsertObject(r catalog.ObjectRecord) error {
	tx, err := b.db.Begin()
	if err != nil {
		return err
	}
	defer tx.Rollback()

	ts := r.Timestamp
	if ts <= 0 {
		ts = float64(time.Now().Unix())
		r.Timestamp = ts
	}

	if _, err := tx.Exec(`
		INSERT INTO objects (id, name, country_code, asset_type, pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, pos_sigma_km, last_update_unix, version, timestamp, tle_line1, tle_line2)
		VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
		ON CONFLICT(id) DO UPDATE SET
			name=excluded.name,
			country_code=excluded.country_code,
			asset_type=excluded.asset_type,
			pos_x=excluded.pos_x, pos_y=excluded.pos_y, pos_z=excluded.pos_z,
			vel_x=excluded.vel_x, vel_y=excluded.vel_y, vel_z=excluded.vel_z,
			pos_sigma_km=excluded.pos_sigma_km,
			last_update_unix=excluded.last_update_unix,
			version=excluded.version,
			timestamp=excluded.timestamp,
			tle_line1=excluded.tle_line1,
			tle_line2=excluded.tle_line2
	`, r.ID, r.Name, r.CountryCode, r.AssetType, r.Pos.X, r.Pos.Y, r.Pos.Z, r.Vel.X, r.Vel.Y, r.Vel.Z, r.PosSigmaKm, r.LastUpdateUnix, r.Version, ts, r.TLELine1, r.TLELine2); err != nil {
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
	`, r.ID, ts, buf.Bytes()); err != nil {
		return err
	}

	return tx.Commit()
}

func (b *SQLiteBackend) DeleteObject(id int) error {
	_, err := b.db.Exec("DELETE FROM objects WHERE id=?", id)
	return err
}

func (b *SQLiteBackend) GetLatestSnapshot() ([]catalog.ObjectRecord, error) {
	rows, err := b.db.Query("SELECT id, name, country_code, asset_type, pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, pos_sigma_km, last_update_unix, version, timestamp, tle_line1, tle_line2 FROM objects")
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var records []catalog.ObjectRecord
	for rows.Next() {
		var r catalog.ObjectRecord
		err := rows.Scan(&r.ID, &r.Name, &r.CountryCode, &r.AssetType, &r.Pos.X, &r.Pos.Y, &r.Pos.Z, &r.Vel.X, &r.Vel.Y, &r.Vel.Z, &r.PosSigmaKm, &r.LastUpdateUnix, &r.Version, &r.Timestamp, &r.TLELine1, &r.TLELine2)
		if err == nil {
			records = append(records, r)
		}
	}
	return records, nil
}

func (b *SQLiteBackend) GetObjectAtTime(id int, t time.Time) (catalog.ObjectRecord, error) {
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
			return catalog.ObjectRecord{}, fmt.Errorf("no historical record found for ID %d at time %v", id, t)
		}
		return catalog.ObjectRecord{}, err
	}

	var rec catalog.ObjectRecord
	if err := gob.NewDecoder(bytes.NewReader(blob)).Decode(&rec); err != nil {
		return catalog.ObjectRecord{}, err
	}
	return rec, nil
}

func (b *SQLiteBackend) QueryObjectsInWindow(start, end time.Time) ([]catalog.ObjectRecord, error) {
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

	records := make([]catalog.ObjectRecord, 0)
	for rows.Next() {
		var blob []byte
		if err := rows.Scan(&blob); err != nil {
			continue
		}
		var rec catalog.ObjectRecord
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
