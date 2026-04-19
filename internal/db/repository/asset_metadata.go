// Package repository provides asset metadata data access
package repository

import (
	"context"
	"fmt"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

// AssetMetadata represents static satellite information
type AssetMetadata struct {
	NoradCatID            int32
	IntlDesignator        string
	ObjectName            string
	ObjectType            string
	CountryCode           string
	LaunchDate            string // ISO 8601 date format
	LaunchSite            string
	Owner                 string
	MassKg                float64
	DragAreaM2            float64
	SRPAreaM2             float64
	ReflectivityCoeff     float64
	IsIndianAsset         bool
	PriorityLevel         int32
	MissionType           string
	ITUFilingRef          string
	OrbitalSlotDeg        float64
}

// AssetMetadataRepo provides data access for asset_metadata table
type AssetMetadataRepo struct {
	pool *pgxpool.Pool
}

// NewAssetMetadataRepo creates repository instance
func NewAssetMetadataRepo(pool *pgxpool.Pool) *AssetMetadataRepo {
	return &AssetMetadataRepo{pool: pool}
}

// GetByNoradID retrieves asset metadata by NORAD catalog ID
func (r *AssetMetadataRepo) GetByNoradID(ctx context.Context, noradID int32) (*AssetMetadata, error) {
	query := `
		SELECT 
			norad_cat_id, intl_designator, object_name, object_type,
			country_code, launch_date::TEXT, launch_site, owner,
			mass_kg, drag_area_m2, srp_area_m2, reflectivity_coefficient,
			is_indian_asset, priority_level, mission_type,
			itu_filing_ref, orbital_slot_deg
		FROM asset_metadata
		WHERE norad_cat_id = $1
	`

	var am AssetMetadata
	err := r.pool.QueryRow(ctx, query, noradID).Scan(
		&am.NoradCatID, &am.IntlDesignator, &am.ObjectName, &am.ObjectType,
		&am.CountryCode, &am.LaunchDate, &am.LaunchSite, &am.Owner,
		&am.MassKg, &am.DragAreaM2, &am.SRPAreaM2, &am.ReflectivityCoeff,
		&am.IsIndianAsset, &am.PriorityLevel, &am.MissionType,
		&am.ITUFilingRef, &am.OrbitalSlotDeg,
	)
	if err != nil {
		return nil, fmt.Errorf("query asset_metadata: %w", err)
	}

	return &am, nil
}

// Upsert creates or updates asset metadata
func (r *AssetMetadataRepo) Upsert(ctx context.Context, am *AssetMetadata) error {
	query := `
		INSERT INTO asset_metadata (
			norad_cat_id, intl_designator, object_name, object_type,
			country_code, launch_date, launch_site, owner,
			mass_kg, drag_area_m2, srp_area_m2, reflectivity_coefficient,
			is_indian_asset, priority_level, mission_type,
			itu_filing_ref, orbital_slot_deg
		) VALUES (
			$1, $2, $3, $4,
			$5, $6::DATE, $7, $8,
			$9, $10, $11, $12,
			$13, $14, $15,
			$16, $17
		)
		ON CONFLICT (norad_cat_id) DO UPDATE SET
			intl_designator = EXCLUDED.intl_designator,
			object_name = EXCLUDED.object_name,
			object_type = EXCLUDED.object_type,
			country_code = EXCLUDED.country_code,
			launch_date = EXCLUDED.launch_date,
			launch_site = EXCLUDED.launch_site,
			owner = EXCLUDED.owner,
			mass_kg = EXCLUDED.mass_kg,
			drag_area_m2 = EXCLUDED.drag_area_m2,
			srp_area_m2 = EXCLUDED.srp_area_m2,
			reflectivity_coefficient = EXCLUDED.reflectivity_coefficient,
			is_indian_asset = EXCLUDED.is_indian_asset,
			priority_level = EXCLUDED.priority_level,
			mission_type = EXCLUDED.mission_type,
			itu_filing_ref = EXCLUDED.itu_filing_ref,
			orbital_slot_deg = EXCLUDED.orbital_slot_deg,
			updated_at = NOW()
	`

	_, err := r.pool.Exec(ctx, query,
		am.NoradCatID, am.IntlDesignator, am.ObjectName, am.ObjectType,
		am.CountryCode, am.LaunchDate, am.LaunchSite, am.Owner,
		am.MassKg, am.DragAreaM2, am.SRPAreaM2, am.ReflectivityCoeff,
		am.IsIndianAsset, am.PriorityLevel, am.MissionType,
		am.ITUFilingRef, am.OrbitalSlotDeg,
	)
	if err != nil {
		return fmt.Errorf("upsert asset_metadata for NORAD %d: %w", am.NoradCatID, err)
	}

	return nil
}

// GetIndianAssets retrieves all Indian assets (priority protection)
func (r *AssetMetadataRepo) GetIndianAssets(ctx context.Context) ([]*AssetMetadata, error) {
	query := `
		SELECT 
			norad_cat_id, intl_designator, object_name, object_type,
			country_code, launch_date::TEXT, launch_site, owner,
			mass_kg, drag_area_m2, srp_area_m2, reflectivity_coefficient,
			is_indian_asset, priority_level, mission_type,
			itu_filing_ref, orbital_slot_deg
		FROM asset_metadata
		WHERE is_indian_asset = TRUE
		ORDER BY priority_level ASC, norad_cat_id ASC
	`

	rows, err := r.pool.Query(ctx, query)
	if err != nil {
		return nil, fmt.Errorf("query indian assets: %w", err)
	}
	defer rows.Close()

	return r.scanAssets(rows)
}

// GetByPriority retrieves assets filtered by priority level
func (r *AssetMetadataRepo) GetByPriority(ctx context.Context, maxPriority int32) ([]*AssetMetadata, error) {
	query := `
		SELECT 
			norad_cat_id, intl_designator, object_name, object_type,
			country_code, launch_date::TEXT, launch_site, owner,
			mass_kg, drag_area_m2, srp_area_m2, reflectivity_coefficient,
			is_indian_asset, priority_level, mission_type,
			itu_filing_ref, orbital_slot_deg
		FROM asset_metadata
		WHERE priority_level <= $1
		ORDER BY priority_level ASC, norad_cat_id ASC
	`

	rows, err := r.pool.Query(ctx, query, maxPriority)
	if err != nil {
		return nil, fmt.Errorf("query by priority: %w", err)
	}
	defer rows.Close()

	return r.scanAssets(rows)
}

// GetAll retrieves all asset metadata (paginated)
func (r *AssetMetadataRepo) GetAll(ctx context.Context, limit, offset int32) ([]*AssetMetadata, error) {
	query := `
		SELECT 
			norad_cat_id, intl_designator, object_name, object_type,
			country_code, launch_date::TEXT, launch_site, owner,
			mass_kg, drag_area_m2, srp_area_m2, reflectivity_coefficient,
			is_indian_asset, priority_level, mission_type,
			itu_filing_ref, orbital_slot_deg
		FROM asset_metadata
		ORDER BY norad_cat_id ASC
		LIMIT $1 OFFSET $2
	`

	rows, err := r.pool.Query(ctx, query, limit, offset)
	if err != nil {
		return nil, fmt.Errorf("query all assets: %w", err)
	}
	defer rows.Close()

	return r.scanAssets(rows)
}

// Count returns total number of assets
func (r *AssetMetadataRepo) Count(ctx context.Context) (int32, error) {
	var count int32
	err := r.pool.QueryRow(ctx, "SELECT COUNT(*) FROM asset_metadata").Scan(&count)
	if err != nil {
		return 0, fmt.Errorf("count assets: %w", err)
	}
	return count, nil
}

// Helper function to scan asset metadata from rows
func (r *AssetMetadataRepo) scanAssets(rows pgx.Rows) ([]*AssetMetadata, error) {
	var assets []*AssetMetadata

	for rows.Next() {
		var am AssetMetadata

		err := rows.Scan(
			&am.NoradCatID, &am.IntlDesignator, &am.ObjectName, &am.ObjectType,
			&am.CountryCode, &am.LaunchDate, &am.LaunchSite, &am.Owner,
			&am.MassKg, &am.DragAreaM2, &am.SRPAreaM2, &am.ReflectivityCoeff,
			&am.IsIndianAsset, &am.PriorityLevel, &am.MissionType,
			&am.ITUFilingRef, &am.OrbitalSlotDeg,
		)
		if err != nil {
			continue // Skip malformed rows
		}

		assets = append(assets, &am)
	}

	return assets, rows.Err()
}
