'use client';

import { useEffect, useState } from 'react';
import { Radar } from 'lucide-react';

export default function NeighborhoodPage() {
  const [intrusions, setIntrusions] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function fetchIntrusions() {
      try {
        const response = await fetch('http://localhost:8080/api/sovereign/intrusions');
        const data = await response.json();
        setIntrusions(data);
      } catch (error) {
        console.error('Failed to fetch intrusions:', error);
      }
      setLoading(false);
    }

    fetchIntrusions();
  }, []);

  return (
    <div className="p-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-slate-900">Neighborhood Watch</h1>
        <p className="text-slate-600 mt-2">Sovereign volume intrusion monitoring</p>
      </div>

      <div className="bg-white rounded-lg shadow p-6">
        <div className="flex items-center gap-2 mb-4">
          <Radar className="w-5 h-5 text-blue-600" />
          <h2 className="text-xl font-semibold text-slate-900">Sovereign Intrusions</h2>
        </div>

        {loading ? (
          <p className="text-slate-500">Loading...</p>
        ) : intrusions ? (
          <div>
            <div className="text-sm text-slate-600 mb-4">
              <p>Total intrusions: {intrusions.total_count || 0}</p>
              <p>Volumes scanned: {intrusions.volumes_scanned || 0}</p>
            </div>
            {intrusions.intrusions && intrusions.intrusions.length > 0 ? (
              <div className="space-y-3">
                {intrusions.intrusions.map((intrusion: any, idx: number) => (
                  <div key={idx} className="p-4 bg-red-50 rounded border-l-4 border-red-500">
                    <div className="font-medium text-red-900">{intrusion.volume_name}</div>
                    <div className="text-sm text-red-700 mt-1">
                      {intrusion.object_name} (NORAD: {intrusion.norad_cat_id})
                    </div>
                    <div className="text-sm text-slate-600 mt-2">
                      <p>Country: {intrusion.country_code}</p>
                      <p>Position: ({intrusion.longitude.toFixed(2)}°, {intrusion.latitude.toFixed(2)}°, {intrusion.altitude_km.toFixed(1)} km)</p>
                      <p>Threat level: <span className="font-bold text-red-600">{intrusion.threat_level}</span></p>
                      <p>Detected: {new Date(intrusion.detected_at).toLocaleString()}</p>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="p-8 bg-green-50 rounded border border-green-200">
                <p className="text-green-800 font-medium">No intrusions detected</p>
                <p className="text-green-600 text-sm mt-1">All sovereign volumes are clear</p>
              </div>
            )}
          </div>
        ) : (
          <p className="text-slate-500">Failed to load intrusion data</p>
        )}
      </div>
    </div>
  );
}
