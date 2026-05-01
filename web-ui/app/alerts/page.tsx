'use client';

import { useEffect, useState } from 'react';
import { AlertTriangle, Filter } from 'lucide-react';

export default function AlertsPage() {
  const [alerts, setAlerts] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState('all');
  const [severityFilter, setSeverityFilter] = useState('all');

  useEffect(() => {
    async function fetchAlerts() {
      setLoading(true);
      try {
        const response = await fetch('http://localhost:8080/api/alerts');
        if (!response.ok) {
          console.error('Alert API returned error:', response.status);
          setAlerts([]);
          return;
        }
        const data = await response.json();
        if (data.status === 'error') {
          console.error('Alert API error:', data.message);
          setAlerts([]);
          return;
        }
        setAlerts(data.alerts || []);
      } catch (error) {
        console.error('Failed to fetch alerts:', error);
        setAlerts([]);
      } finally {
        setLoading(false);
      }
    }

    fetchAlerts();
  }, []);

  const filteredAlerts = alerts.filter(alert => {
    if (filter !== 'all' && alert.category !== filter) return false;
    if (severityFilter !== 'all' && alert.severity !== severityFilter) return false;
    return true;
  });

  const handleAcknowledge = async (alertId: string) => {
    try {
      await fetch(`http://localhost:8080/api/alerts/${alertId}/ack`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          operator_id: 'OPERATOR',
          action: 'MONITOR',
          notes: 'Acknowledged via dashboard'
        })
      });
      setAlerts(alerts.filter(a => a.alert_id !== alertId));
    } catch (error) {
      console.error('Failed to acknowledge alert:', error);
    }
  };

  const severityColors = {
    INFO: 'bg-blue-100 text-blue-800 border-blue-300',
    WARNING: 'bg-yellow-100 text-yellow-800 border-yellow-300',
    CRITICAL: 'bg-red-100 text-red-800 border-red-300',
    EMERGENCY: 'bg-red-600 text-white border-red-700',
  };

  return (
    <div className="p-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-slate-900">Alerts</h1>
        <p className="text-slate-600 mt-2">System-wide alerts and notifications</p>
      </div>

      <div className="bg-white rounded-lg shadow p-6 mb-6">
        <div className="flex items-center gap-2 mb-4">
          <Filter className="w-5 h-5 text-slate-600" />
          <h2 className="text-lg font-semibold text-slate-900">Filters</h2>
        </div>
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-1">Category</label>
            <select
              value={filter}
              onChange={(e) => setFilter(e.target.value)}
              className="w-full px-3 py-2 border border-slate-300 rounded-lg"
            >
              <option value="all">All Categories</option>
              <option value="MANEUVER">Maneuver</option>
              <option value="NAVIC">NavIC</option>
              <option value="COLLISION">Collision</option>
              <option value="INTRUSION">Intrusion</option>
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-1">Severity</label>
            <select
              value={severityFilter}
              onChange={(e) => setSeverityFilter(e.target.value)}
              className="w-full px-3 py-2 border border-slate-300 rounded-lg"
            >
              <option value="all">All Severities</option>
              <option value="INFO">Info</option>
              <option value="WARNING">Warning</option>
              <option value="CRITICAL">Critical</option>
              <option value="EMERGENCY">Emergency</option>
            </select>
          </div>
        </div>
      </div>

      <div className="bg-white rounded-lg shadow p-6 mb-6">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <AlertTriangle className="w-5 h-5 text-orange-500" />
            <span className="text-lg font-semibold text-slate-900">Total Alerts</span>
          </div>
          <span className="text-2xl font-bold text-slate-900">{filteredAlerts.length}</span>
        </div>
      </div>

      <div className="bg-white rounded-lg shadow p-6">
        {loading ? (
          <p className="text-slate-500">Loading alerts...</p>
        ) : filteredAlerts.length > 0 ? (
          <div className="space-y-3">
            {filteredAlerts.map((alert) => (
              <div
                key={alert.alert_id}
                className={`p-4 rounded border ${
                  severityColors[alert.severity as keyof typeof severityColors] || 'bg-slate-50 border-slate-200'
                }`}
              >
                <div className="flex justify-between items-start">
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-1">
                      <span className="font-medium">{alert.title || alert.category}</span>
                      <span className={`px-2 py-0.5 rounded text-xs border ${
                        severityColors[alert.severity as keyof typeof severityColors] || 'bg-slate-100 text-slate-800 border-slate-300'
                      }`}>
                        {alert.severity}
                      </span>
                    </div>
                    <p className="text-sm mb-2">{alert.description || alert.message}</p>
                    {alert.metadata && (
                      <div className="text-sm opacity-80">
                        <p>Category: {alert.category}</p>
                        <p>Created: {alert.created_at ? new Date(alert.created_at).toLocaleString() : 'N/A'}</p>
                      </div>
                    )}
                  </div>
                  <button
                    onClick={() => handleAcknowledge(alert.alert_id)}
                    className="px-3 py-1 bg-white text-slate-900 text-sm rounded hover:bg-slate-100 border border-slate-300"
                  >
                    Acknowledge
                  </button>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="p-8 bg-green-50 rounded border border-green-200">
            <p className="text-green-800 font-medium">No alerts</p>
            <p className="text-green-600 text-sm mt-1">All systems nominal</p>
          </div>
        )}
      </div>
    </div>
  );
}
