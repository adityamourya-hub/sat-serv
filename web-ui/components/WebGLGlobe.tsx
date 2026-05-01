'use client';

import { useEffect, useRef, useState } from 'react';
import { getCatalog, getAlerts } from '@/lib/api';

declare global {
  interface Window {
    initGlobe?: any;
  }
}

interface Satellite {
  id: string;
  name: string;
  position: { x: number; y: number; z: number };
  velocity: { vx: number; vy: number; vz: number };
  epoch: string;
  type?: 'indian' | 'tracked' | 'risk' | 'maneuver' | 'anomaly';
  alt?: number;
  i?: number;
  e?: number;
  period?: number;
  status?: string;
  norad?: number;
}

export default function WebGLGlobe() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [sats, setSats] = useState<Satellite[]>([]);
  const [alerts, setAlerts] = useState<any[]>([]);
  const [layers, setLayers] = useState({
    indian: true,
    tracked: true,
    risk: true,
    maneuver: true,
    anomaly: true,
    trails: true,
    grid: true,
  });
  const [paused, setPaused] = useState(false);
  const [timeSpeed, setTimeSpeed] = useState(1.0);
  const [simT, setSimT] = useState(0);

  useEffect(() => {
    async function fetchData() {
      try {
        const catalogData = await getCatalog();
        const alertsData = await getAlerts();
        
        // Transform catalog data to satellite format
        const transformedSats = (catalogData.satellites || []).map((sat: any) => ({
          ...sat,
          type: sat.type || 'tracked',
          alt: sat.alt || 500,
          i: sat.i || 45,
          e: sat.e || 0.01,
          period: sat.period || 90,
          status: sat.status || 'Normal operation',
          norad: sat.norad || 25000 + Math.floor(Math.random() * 1000),
        }));
        
        setSats(transformedSats);
        setAlerts(alertsData || []);
      } catch (error) {
        console.error('Failed to fetch data:', error);
        // Use mock data as fallback
        setSats(generateMockSatellites());
      }
    }

    fetchData();
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return;

    // Load globe.html script
    const script = document.createElement('script');
    script.src = '/globe.js';
    script.async = true;
    document.body.appendChild(script);

    script.onload = () => {
      // Initialize globe with data
      if (window.initGlobe) {
        window.initGlobe(canvas, sats, layers, paused, timeSpeed, simT);
      }
    };

    return () => {
      document.body.removeChild(script);
    };
  }, [sats, layers, paused, timeSpeed, simT]);

  const toggleLayer = (layer: keyof typeof layers) => {
    setLayers(prev => ({ ...prev, [layer]: !prev[layer] }));
  };

  return (
    <div ref={containerRef} className="relative w-full h-screen bg-slate-900">
      <canvas ref={canvasRef} className="w-full h-full cursor-grab" />
      
      {/* Layer Controls */}
      <div className="absolute top-4 left-4 flex flex-col gap-2 z-10">
        {Object.entries(layers).map(([key, enabled]) => (
          <button
            key={key}
            onClick={() => toggleLayer(key as keyof typeof layers)}
            className={`px-3 py-2 text-xs font-medium rounded ${
              enabled ? 'bg-blue-600 text-white' : 'bg-slate-700 text-slate-300'
            }`}
          >
            {key.charAt(0).toUpperCase() + key.slice(1)}
          </button>
        ))}
      </div>

      {/* Time Controls */}
      <div className="absolute bottom-4 left-1/2 transform -translate-x-1/2 flex gap-2 z-10">
        <button
          onClick={() => setPaused(!paused)}
          className="px-4 py-2 bg-blue-600 text-white rounded"
        >
          {paused ? '▶ Play' : '⏸ Pause'}
        </button>
        {[1, 5, 20].map(speed => (
          <button
            key={speed}
            onClick={() => setTimeSpeed(speed)}
            className={`px-3 py-2 rounded ${timeSpeed === speed ? 'bg-blue-600 text-white' : 'bg-slate-700 text-slate-300'}`}
          >
            {speed}×
          </button>
        ))}
      </div>

      {/* Satellite Count */}
      <div className="absolute top-4 right-4 bg-slate-800 px-4 py-2 rounded z-10">
        <span className="text-slate-300">Catalogue: </span>
        <span className="text-white font-bold">{sats.length}</span>
        <span className="text-slate-300"> objects</span>
      </div>

      {/* Alerts Panel */}
      <div className="absolute top-16 right-4 w-80 bg-slate-800 rounded-lg p-4 max-h-96 overflow-y-auto z-10">
        <h2 className="text-sm font-bold text-blue-400 mb-2">Active Alerts</h2>
        {alerts.length === 0 ? (
          <p className="text-slate-400 text-xs">No alerts</p>
        ) : (
          <div className="space-y-2">
            {alerts.slice(0, 5).map((alert) => (
              <div key={alert.id} className="p-2 bg-slate-700 rounded border-l-2 border-red-500">
                <div className="text-xs font-medium text-white">{alert.type}</div>
                <div className="text-xs text-slate-400">{alert.message}</div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function generateMockSatellites(): Satellite[] {
  const sats: Satellite[] = [];
  const names = ['IRS-1C', 'IRS-1D', 'Cartosat-2', 'RISAT-2', 'GSAT-30', 'INSAT-3A', 'Oceansat-2', 'EOS-01'];
  
  for (let i = 0; i < 50; i++) {
    const types: Array<'indian' | 'tracked' | 'risk' | 'maneuver' | 'anomaly'> = 
      ['indian', 'tracked', 'tracked', 'tracked', 'tracked', 'risk', 'maneuver', 'anomaly'];
    const type = types[Math.floor(Math.random() * types.length)];
    
    sats.push({
      id: `SAT-${25000 + i}`,
      name: names[i % names.length] || `SAT-${25000 + i}`,
      position: {
        x: (Math.random() - 0.5) * 20000,
        y: (Math.random() - 0.5) * 20000,
        z: (Math.random() - 0.5) * 20000,
      },
      velocity: {
        vx: (Math.random() - 0.5) * 10,
        vy: (Math.random() - 0.5) * 10,
        vz: (Math.random() - 0.5) * 10,
      },
      epoch: new Date().toISOString(),
      type,
      alt: 400 + Math.random() * 1000,
      i: 20 + Math.random() * 60,
      e: 0.001 + Math.random() * 0.01,
      period: 90 + Math.random() * 30,
      status: type === 'risk' ? 'CONJUNCTION RISK' : 'Normal operation',
      norad: 25000 + i,
    });
  }
  
  return sats;
}
