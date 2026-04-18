package persistence

import (
	"bytes"
	"encoding/gob"
	"fmt"
	"strconv"
	"strings"
	"time"

	"ssis/pkg/catalog"
	"go.etcd.io/bbolt"
)

type BoltDBBackend struct {
	db *bbolt.DB
}

func NewBoltDBBackend(path string) (*BoltDBBackend, error) {
	if path == "" {
		path = "ssis_catalog.db"
	}

	db, err := bbolt.Open(path, 0600, &bbolt.Options{Timeout: 1 * time.Second})
	if err != nil {
		return nil, err
	}

	// Create buckets
	err = db.Update(func(tx *bbolt.Tx) error {
		buckets := []string{"objects", "history", "mht", "audit"}
		for _, b := range buckets {
			_, err := tx.CreateBucketIfNotExists([]byte(b))
			if err != nil {
				return err
			}
		}
		return nil
	})

	if err != nil {
		db.Close()
		return nil, err
	}

	return &BoltDBBackend{db: db}, nil
}

func (b *BoltDBBackend) Close() error {
	return b.db.Close()
}

func (b *BoltDBBackend) UpsertObject(record catalog.SDAObject) error {
	return b.db.Update(func(tx *bbolt.Tx) error {
		objBucket := tx.Bucket([]byte("objects"))
		histBucket := tx.Bucket([]byte("history"))

		var buf bytes.Buffer
		if err := gob.NewEncoder(&buf).Encode(record); err != nil {
			return err
		}

		key := []byte(fmt.Sprintf("%d", record.ID))
		if err := objBucket.Put(key, buf.Bytes()); err != nil {
			return err
		}

		// Historical entry: ID:EpochTT
		histKey := []byte(fmt.Sprintf("%d:%.6f", record.ID, record.EpochTT))
		return histBucket.Put(histKey, buf.Bytes())
	})
}

func (b *BoltDBBackend) DeleteObject(id int) error {
	return b.db.Update(func(tx *bbolt.Tx) error {
		return tx.Bucket([]byte("objects")).Delete([]byte(fmt.Sprintf("%d", id)))
	})
}

func (b *BoltDBBackend) GetLatestSnapshot() ([]catalog.SDAObject, error) {
	var records []catalog.SDAObject
	err := b.db.View(func(tx *bbolt.Tx) error {
		bucket := tx.Bucket([]byte("objects"))
		return bucket.ForEach(func(k, v []byte) error {
			var rec catalog.SDAObject
			if err := gob.NewDecoder(bytes.NewReader(v)).Decode(&rec); err == nil {
				records = append(records, rec)
			}
			return nil
		})
	})
	return records, err
}

func (b *BoltDBBackend) GetObjectAtTime(id int, t time.Time) (catalog.SDAObject, error) {
	var rec catalog.SDAObject
	targetTS := t.Unix()
	
	err := b.db.View(func(tx *bbolt.Tx) error {
		bucket := tx.Bucket([]byte("history"))
		prefix := []byte(fmt.Sprintf("%d:", id))
		c := bucket.Cursor()

		var bestVal []byte
		bestTS := int64(-1)
		for k, v := c.Seek(prefix); k != nil && bytes.HasPrefix(k, prefix); k, v = c.Next() {
			ts, ok := parseHistoryTS(k)
			if !ok {
				continue
			}
			if ts <= targetTS && ts >= bestTS {
				bestTS = ts
				bestVal = append(bestVal[:0], v...)
			}
		}

		if bestVal != nil {
			return gob.NewDecoder(bytes.NewReader(bestVal)).Decode(&rec)
		}
		
		return fmt.Errorf("no historical record found for ID %d at time %v", id, t)
	})
	
	return rec, err
}

func parseHistoryTS(key []byte) (int64, bool) {
	parts := strings.SplitN(string(key), ":", 2)
	if len(parts) != 2 {
		return 0, false
	}
	ts, err := strconv.ParseInt(parts[1], 10, 64)
	if err != nil {
		return 0, false
	}
	return ts, true
}

func (b *BoltDBBackend) QueryObjectsInWindow(start, end time.Time) ([]catalog.SDAObject, error) {
	var records []catalog.SDAObject
	err := b.db.View(func(tx *bbolt.Tx) error {
		bucket := tx.Bucket([]byte("history"))
		c := bucket.Cursor()
		
		for k, v := c.First(); k != nil; k, v = c.Next() {
			var rec catalog.SDAObject
			if err := gob.NewDecoder(bytes.NewReader(v)).Decode(&rec); err == nil {
				ts := rec.EpochTT
				if ts >= float64(start.Unix())/86400.0+2440587.5 && ts <= float64(end.Unix())/86400.0+2440587.5 {
					records = append(records, rec)
				}
			}
		}
		return nil
	})
	return records, err
}

func (b *BoltDBBackend) PersistHypothesis(hyp *catalog.GlobalHypothesis) error {
	return b.db.Update(func(tx *bbolt.Tx) error {
		bucket := tx.Bucket([]byte("mht"))
		var buf bytes.Buffer
		if err := gob.NewEncoder(&buf).Encode(hyp); err != nil {
			return err
		}
		key := []byte(fmt.Sprintf("%d:%d", hyp.ID, hyp.ScanIndex))
		return bucket.Put(key, buf.Bytes())
	})
}
