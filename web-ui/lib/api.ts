import axios from 'axios';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8080';

export const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  validateStatus: function (status) {
    // Resolve for all status codes, handle errors manually
    return true;
  },
});

// API Endpoints
export const apiEndpoints = {
  health: '/health',
  catalog: '/catalog',
  volumetricScan: '/volumetric/scan',
  conjunctionRisk: '/conjunction/risk',
  sovereignIntrusions: '/api/sovereign/intrusions',
  alerts: '/api/alerts',
  metrics: '/metrics',
};

// Types
export interface Satellite {
  id: string;
  name: string;
  position: { x: number; y: number; z: number };
  velocity: { vx: number; vy: number; vz: number };
  epoch: string;
}

export interface VolumetricScanResult {
  scan_id: string;
  timestamp: string;
  region: { center: [number, number, number]; radius: number };
  objects: Satellite[];
  anomalies: number;
}

export interface ConjunctionRisk {
  primary_id: string;
  secondary_id: string;
  time_of_closest_approach: string;
  miss_distance: number;
  probability: number;
}

export interface Alert {
  id: string;
  type: 'collision' | 'maneuver' | 'anomaly' | 'intrusion';
  severity: 'low' | 'medium' | 'high' | 'critical';
  message: string;
  timestamp: string;
  metadata?: Record<string, unknown>;
}

export interface DashboardMetrics {
  timestamp: string;
  total_objects: number;
  active_alerts: number;
  conjunction_events: number;
  anomaly_detections: number;
  system_health: string;
}

// API Functions
export async function getHealth() {
  try {
    const response = await api.get(apiEndpoints.health);
    return response.data;
  } catch (error) {
    console.error('Failed to fetch health:', error);
    return { status: 'error', service: 'SSIS REST API', version: 'unknown' };
  }
}

export async function getCatalog() {
  try {
    const response = await api.get(apiEndpoints.catalog);
    return response.data;
  } catch (error) {
    console.error('Failed to fetch catalog:', error);
    return { objects: [], count: 0 };
  }
}

export async function runVolumetricScan(params: {
  region_center: [number, number, number];
  radius: number;
}) {
  try {
    const response = await api.post(apiEndpoints.volumetricScan, params);
    return response.data as VolumetricScanResult;
  } catch (error) {
    console.error('Failed to run volumetric scan:', error);
    return { objects: [], scan_id: '', timestamp: '', region: { center: [0, 0, 0], radius: 0 }, anomalies: 0 };
  }
}

export async function getConjunctionRisks() {
  try {
    const response = await api.get(apiEndpoints.conjunctionRisk);
    return response.data as ConjunctionRisk[];
  } catch (error) {
    console.error('Failed to fetch conjunction risks:', error);
    return [];
  }
}

export async function getSovereignIntrusions() {
  try {
    const response = await api.get(apiEndpoints.sovereignIntrusions);
    return response.data;
  } catch (error) {
    console.error('Failed to fetch sovereign intrusions:', error);
    return { intrusions: [], total_count: 0, volumes_scanned: 0 };
  }
}

export async function getAlerts() {
  try {
    const response = await api.get(apiEndpoints.alerts);
    // Check if response status indicates an error
    if (response.status >= 400) {
      console.error('Alert API returned error status:', response.status);
      return [];
    }
    const data = response.data;
    // Go API returns { alerts: [], total: number, count: number } or error response
    if (data.status === 'error') {
      console.error('Alert API error:', data.message);
      return [];
    }
    return data.alerts || [];
  } catch (error) {
    console.error('Failed to fetch alerts:', error);
    return [];
  }
}

export async function getMetrics() {
  try {
    const response = await api.get(apiEndpoints.metrics);
    const data = response.data;
    // Go API returns { metrics: { total_objects, active_workers, etc }, status, timestamp }
    const metrics = data.metrics || {};
    return {
      timestamp: String(data.timestamp || Date.now()),
      total_objects: metrics.total_objects || 0,
      active_alerts: metrics.alerts_24h || 0,
      conjunction_events: 0, // Not in current metrics response
      anomaly_detections: 0, // Not in current metrics response
      system_health: data.status || 'unknown',
    };
  } catch (error) {
    console.error('Failed to fetch metrics:', error);
    // Return default values on error
    return {
      timestamp: String(Date.now()),
      total_objects: 0,
      active_alerts: 0,
      conjunction_events: 0,
      anomaly_detections: 0,
      system_health: 'error',
    };
  }
}
