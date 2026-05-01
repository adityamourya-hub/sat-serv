'use client';

import { useEffect, useState } from 'react';
import { Globe } from 'lucide-react';

export default function NavicPage() {
  const [alerts, setAlerts] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function fetchAlerts() {
      try {
        const response = await fetch('http://localhost:8080/api/alerts?category=NAVIC');
        const data = await response.json();
        if (data.alerts) {
          setAlerts(data.alerts);
        }
      } catch (error) {
        console.error('Failed to fetch NavIC alerts:', error);
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
        <h1 className="text-3xl font-bold text-slate-900">NavIC Anomaly</h1>
        <p className="text-slate-600 mt-2">NavIC navigation signal anomalies</p>
      </div>

      <div className="bg-white rounded-lg shadow p-6">
        <div className="flex items-center gap-2 mb-4">
          <Globe className="w-5 h-5 text-purple-600" />
          <h2 className="text-xl font-semibold text-slate-900">NavIC Anomaly Alerts</h2>
        </div>

        {loading ? (
          <p className="text-slate-500">Loading...</p>
        ) : alerts.length > 0 ? (
          <div className="space-y-3">
            {alerts.map((alert) => (
              <div key={alert.alert_id} className="p-4 bg-purple-50 rounded border-l-4 border-purple-500">
                <div className="flex justify-between items-start">
                  <div>
                    <div className="font-medium text-purple-900">{alert.title}</div>
                    <div className="text-sm text-purple-700 mt-1">{alert.description}</div>
                    {alert.metadata && (
                      <div className="text-sm text-slate-600 mt-2">
                        <p>Satellite: {alert.metadata.satellite_id || 'N/A'}</p>
                        <p>Signal: {alert.metadata.signal_strength || 'N/A'} dB</p>
                        <p>Anomaly type: {alert.metadata.anomaly_type || 'N/A'}</p>
                        <p>Detected: {alert.created_at ? new Date(alert.created_at).toLocaleString() : 'N/A'}</p>
                      </div>
                    )}
                  </div>
                  <button
                    onClick={() => handleAcknowledge(alert.alert_id)}
                    className="px-3 py-1 bg-purple-600 text-white text-sm rounded hover:bg-purple-700"
                  >
                    Acknowledge
                  </button>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="p-8 bg-green-50 rounded border border-green-200">
            <p className="text-green-800 font-medium">No NavIC anomalies</p>
            <p className="text-green-600 text-sm mt-1">All navigation signals nominal</p>
          </div>
        )}
      </div>
    </div>
  );
}
