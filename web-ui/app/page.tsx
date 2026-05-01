'use client';

import { useEffect, useState } from 'react';
import { getMetrics, getAlerts } from '@/lib/api';
import { Activity, AlertTriangle, Shield, Satellite, TrendingUp } from 'lucide-react';

interface DashboardState {
  timestamp: string;
  total_objects: number;
  active_alerts: number;
  conjunction_events: number;
  anomaly_detections: number;
  system_health: string;
}

export default function Dashboard() {
  const [metrics, setMetrics] = useState<DashboardState | null>(null);
  const [alerts, setAlerts] = useState<any[]>([]);

  useEffect(() => {
    // Fetch initial data
    async function fetchData() {
      try {
        const metricsData = await getMetrics();
        setMetrics(metricsData);
        const alertsData = await getAlerts();
        setAlerts(alertsData);
      } catch (error) {
        console.error('Failed to fetch initial data:', error);
      }
    }

    fetchData();

    // Poll for updates every 5 seconds instead of using WebSocket
    const interval = setInterval(fetchData, 5000);

    return () => {
      clearInterval(interval);
    };
  }, []);

  return (
    <div className="p-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-slate-900">Dashboard</h1>
        <p className="text-slate-600 mt-2">Real-time space situational awareness</p>
      </div>

      {/* Status Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        <MetricCard
          title="Total Objects"
          value={metrics?.total_objects || 0}
          icon={Satellite}
          color="blue"
        />
        <MetricCard
          title="Active Alerts"
          value={metrics?.active_alerts || 0}
          icon={AlertTriangle}
          color="red"
        />
        <MetricCard
          title="Conjunction Events"
          value={metrics?.conjunction_events || 0}
          icon={Shield}
          color="orange"
        />
        <MetricCard
          title="Anomaly Detections"
          value={metrics?.anomaly_detections || 0}
          icon={Activity}
          color="purple"
        />
      </div>

      {/* Recent Alerts */}
      <div className="bg-white rounded-lg shadow p-6">
        <h2 className="text-xl font-semibold text-slate-900 mb-4">Recent Alerts</h2>
        {alerts.length === 0 ? (
          <p className="text-slate-500">No alerts</p>
        ) : (
          <div className="space-y-3">
            {alerts.slice(0, 5).map((alert) => (
              <AlertCard key={alert.id} alert={alert} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function MetricCard({ title, value, icon: Icon, color }: {
  title: string;
  value: number;
  icon: React.ElementType;
  color: 'blue' | 'red' | 'orange' | 'purple';
}) {
  const colorClasses = {
    blue: 'bg-blue-500',
    red: 'bg-red-500',
    orange: 'bg-orange-500',
    purple: 'bg-purple-500',
  };

  return (
    <div className="bg-white rounded-lg shadow p-6">
      <div className="flex items-center justify-between mb-4">
        <Icon className="w-6 h-6 text-slate-400" />
        <div className={`w-2 h-2 rounded-full ${colorClasses[color]}`} />
      </div>
      <div className="text-3xl font-bold text-slate-900">{value}</div>
      <div className="text-sm text-slate-600 mt-1">{title}</div>
    </div>
  );
}

function AlertCard({ alert }: { alert: any }) {
  const severityColors = {
    low: 'bg-green-100 text-green-800',
    medium: 'bg-yellow-100 text-yellow-800',
    high: 'bg-orange-100 text-orange-800',
    critical: 'bg-red-100 text-red-800',
  };

  return (
    <div className="flex items-start gap-4 p-4 bg-slate-50 rounded-lg">
      <AlertTriangle className="w-5 h-5 text-orange-500 mt-0.5" />
      <div className="flex-1">
        <div className="flex items-center gap-2 mb-1">
          <span className="font-medium text-slate-900">{alert.type}</span>
          <span className={`px-2 py-0.5 rounded text-xs ${severityColors[alert.severity as keyof typeof severityColors]}`}>
            {alert.severity}
          </span>
        </div>
        <p className="text-sm text-slate-600">{alert.message}</p>
        <p className="text-xs text-slate-400 mt-1">{new Date(alert.timestamp).toLocaleString()}</p>
      </div>
    </div>
  );
}
