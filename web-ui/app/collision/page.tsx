'use client';

import { useState } from 'react';
import { Shield } from 'lucide-react';

export default function CollisionPage() {
  const [miss, setMiss] = useState('1.0');
  const [hbr, setHbr] = useState('0.1');
  const [sigmaR, setSigmaR] = useState('0.5');
  const [sigmaI, setSigmaI] = useState('0.5');
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  const handleCalculate = async () => {
    setLoading(true);
    try {
      const response = await fetch(
        `http://localhost:8080/conjunction/risk?miss=${miss}&hbr=${hbr}&sigma_r=${sigmaR}&sigma_i=${sigmaI}`
      );
      const data = await response.json();
      setResult(data);
    } catch (error) {
      console.error('Calculation failed:', error);
    }
    setLoading(false);
  };

  return (
    <div className="p-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-slate-900">Collision Risk</h1>
        <p className="text-slate-600 mt-2">Calculate conjunction probability</p>
      </div>

      <div className="bg-white rounded-lg shadow p-6 mb-6">
        <div className="grid grid-cols-4 gap-4 mb-4">
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-1">Miss Distance (km)</label>
            <input
              type="number"
              step="0.1"
              value={miss}
              onChange={(e) => setMiss(e.target.value)}
              className="w-full px-3 py-2 border border-slate-300 rounded-lg"
              placeholder="1.0"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-1">Hard Body Radius (km)</label>
            <input
              type="number"
              step="0.1"
              value={hbr}
              onChange={(e) => setHbr(e.target.value)}
              className="w-full px-3 py-2 border border-slate-300 rounded-lg"
              placeholder="0.1"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-1">Sigma Radial (km)</label>
            <input
              type="number"
              step="0.1"
              value={sigmaR}
              onChange={(e) => setSigmaR(e.target.value)}
              className="w-full px-3 py-2 border border-slate-300 rounded-lg"
              placeholder="0.5"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-1">Sigma In-track (km)</label>
            <input
              type="number"
              step="0.1"
              value={sigmaI}
              onChange={(e) => setSigmaI(e.target.value)}
              className="w-full px-3 py-2 border border-slate-300 rounded-lg"
              placeholder="0.5"
            />
          </div>
        </div>
        <button
          onClick={handleCalculate}
          disabled={loading}
          className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-slate-400"
        >
          <Shield className="w-4 h-4" />
          {loading ? 'Calculating...' : 'Calculate Risk'}
        </button>
      </div>

      {result && result.risk && (
        <div className="bg-white rounded-lg shadow p-6">
          <h2 className="text-xl font-semibold text-slate-900 mb-4">Risk Assessment</h2>
          <div className="grid grid-cols-2 gap-4">
            <div className="p-4 bg-slate-50 rounded">
              <div className="text-sm text-slate-600">Collision Probability (Pc)</div>
              <div className="text-2xl font-bold text-slate-900">{(result.risk.pc * 100).toFixed(6)}%</div>
            </div>
            <div className="p-4 bg-slate-50 rounded">
              <div className="text-sm text-slate-600">Maximum Miss Distance</div>
              <div className="text-2xl font-bold text-slate-900">{result.risk.max_miss_km?.toFixed(3) || 'N/A'} km</div>
            </div>
            <div className="p-4 bg-slate-50 rounded">
              <div className="text-sm text-slate-600">Combined Sigma</div>
              <div className="text-2xl font-bold text-slate-900">{result.risk.combined_sigma_km?.toFixed(3) || 'N/A'} km</div>
            </div>
            <div className="p-4 bg-slate-50 rounded">
              <div className="text-sm text-slate-600">Risk Level</div>
              <div className={`text-2xl font-bold ${
                result.risk.risk_level === 'HIGH' ? 'text-red-600' :
                result.risk.risk_level === 'MEDIUM' ? 'text-yellow-600' :
                'text-green-600'
              }`}>
                {result.risk.risk_level || 'LOW'}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
