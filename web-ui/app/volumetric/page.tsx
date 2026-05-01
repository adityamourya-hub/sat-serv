'use client';

import { useState } from 'react';
import { Scan } from 'lucide-react';

export default function VolumetricPage() {
  const [lat, setLat] = useState('26.0');
  const [lon, setLon] = useState('88.5');
  const [radius, setRadius] = useState('100');
  const [results, setResults] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  const handleScan = async () => {
    setLoading(true);
    try {
      const response = await fetch(
        `http://localhost:8080/volumetric/scan?lat=${lat}&lon=${lon}&radius=${radius}`
      );
      const data = await response.json();
      setResults(data);
    } catch (error) {
      console.error('Scan failed:', error);
    }
    setLoading(false);
  };

  return (
    <div className="p-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-slate-900">Volumetric Scan</h1>
        <p className="text-slate-600 mt-2">Scan a region for space objects</p>
      </div>

      <div className="bg-white rounded-lg shadow p-6 mb-6">
        <div className="grid grid-cols-3 gap-4 mb-4">
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-1">Latitude</label>
            <input
              type="number"
              value={lat}
              onChange={(e) => setLat(e.target.value)}
              className="w-full px-3 py-2 border border-slate-300 rounded-lg"
              placeholder="26.0"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-1">Longitude</label>
            <input
              type="number"
              value={lon}
              onChange={(e) => setLon(e.target.value)}
              className="w-full px-3 py-2 border border-slate-300 rounded-lg"
              placeholder="88.5"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-1">Radius (km)</label>
            <input
              type="number"
              value={radius}
              onChange={(e) => setRadius(e.target.value)}
              className="w-full px-3 py-2 border border-slate-300 rounded-lg"
              placeholder="100"
            />
          </div>
        </div>
        <button
          onClick={handleScan}
          disabled={loading}
          className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-slate-400"
        >
          <Scan className="w-4 h-4" />
          {loading ? 'Scanning...' : 'Scan Region'}
        </button>
      </div>

      {results && (
        <div className="bg-white rounded-lg shadow p-6">
          <h2 className="text-xl font-semibold text-slate-900 mb-4">Scan Results</h2>
          <div className="text-sm text-slate-600 mb-4">
            <p>Status: {results.status}</p>
            <p>Objects found: {results.objects?.length || 0}</p>
          </div>
          {results.objects && results.objects.length > 0 ? (
            <div className="space-y-2">
              {results.objects.map((obj: any, idx: number) => (
                <div key={idx} className="p-3 bg-slate-50 rounded border border-slate-200">
                  <p className="font-medium text-slate-900">{obj.name || `Object ${obj.object_id}`}</p>
                  <p className="text-sm text-slate-600">ID: {obj.object_id}</p>
                  <p className="text-sm text-slate-600">
                    Position: ({obj.position?.X?.toFixed(1)}, {obj.position?.Y?.toFixed(1)}, {obj.position?.Z?.toFixed(1)}) km
                  </p>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-slate-500">No objects found in region</p>
          )}
        </div>
      )}
    </div>
  );
}
