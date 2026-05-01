'use client';

import { useEffect, useState } from 'react';
import { Activity } from 'lucide-react';

export default function ManeuverPage() {
  const [alerts, setAlerts] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function fetchAlerts() {
      try {
        const response = await fetch('http://localhost:8080/api/alerts?category=MANEUVER');
        const data = await response.json();
        if (data.alerts) {
          setAlerts(data.alerts);
        }
      } catch (error) {
        console.error('Failed to fetch maneuver alerts:', error);
      }
      setLoading(false);
    }

    fetchAlerts();
  }, []);

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
      // Refresh alerts
      setAlerts(alerts.filter(a => a.alert_id !== alertId));
    } catch (error) {
      console.error('Failed to acknowledge alert:', error);
    }
  };

  return (
    <div className="p-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-slate-900">Maneuver Detection</h1>
        <p className="text-slate-600 mt-2">Satellite maneuver alerts</p>
      </div>

      <div className="bg-white rounded-lg shadow p-6">
        <div className="flex items-center gap-2 mb-4">
          <Activity className="w-5 h-5 text-yellow-600" />
          <h2 className="text-xl font-semibold text-slate-900">Maneuver Alerts</h2>
        </div>

        {loading ? (
          <p className="text-slate-500">Loading...</p>
        ) : alerts.length > 0 ? (
          <div className="space-y-3">
            {alerts.map((alert) => (
              <div key={alert.alert_id} className="p-4 bg-yellow-50 rounded border-l-4 border-yellow-500">
                <div className="flex justify-between items-start">
                  <div>
                    <div className="font-medium text-yellow-900">{alert.title}</div>
                    <div className="text-sm text-yellow-700 mt-1">{alert.description}</div>
                    {alert.metadata && (
                      <div className="text-sm text-slate-600 mt-2">
                        <p>Satellite: {alert.metadata.satellite_id || 'N/A'}</p>
                        <p>Δv: {alert.metadata.delta_v?.toFixed(3) || 'N/A'} m/s</p>
                        <p>Detected: {alert.created_at ? new Date(alert.created_at).toLocaleString() : 'N/A'}</p>
                      </div>
                    )}
                  </div>
                  <button
                    onClick={() => handleAcknowledge(alert.alert_id)}
                    className="px-3 py-1 bg-yellow-600 text-white text-sm rounded hover:bg-yellow-700"
                  >
                    Acknowledge
                  </button>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="p-8 bg-green-50 rounded border border-green-200">
            <p className="text-green-800 font-medium">No maneuver alerts</p>
            <p className="text-green-600 text-sm mt-1">All satellites are nominal</p>
          </div>
        )}
      </div>
    </div>
  );
}
