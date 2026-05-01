'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { 
  LayoutDashboard, 
  Scan, 
  Shield, 
  AlertTriangle, 
  Activity,
  Globe,
  Radar,
  Satellite
} from 'lucide-react';

const navItems = [
  { href: '/', label: 'Dashboard', icon: LayoutDashboard },
  { href: '/volumetric', label: 'Volumetric Scan', icon: Scan },
  { href: '/neighborhood', label: 'Neighborhood Watch', icon: Radar },
  { href: '/collision', label: 'Collision Risk', icon: Shield },
  { href: '/maneuver', label: 'Maneuver Detection', icon: Activity },
  { href: '/navic', label: 'NavIC Anomaly', icon: Globe },
  { href: '/trajectories', label: 'Trajectories', icon: Satellite },
  { href: '/alerts', label: 'Alerts', icon: AlertTriangle },
];

export default function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="w-64 bg-slate-900 text-white h-screen fixed left-0 top-0 flex flex-col">
      <div className="p-6 border-b border-slate-700">
        <h1 className="text-xl font-bold">SSIS Dashboard</h1>
        <p className="text-slate-400 text-sm">Space Situational Awareness</p>
      </div>
      
      <nav className="flex-1 p-4">
        <ul className="space-y-2">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = pathname === item.href;
            
            return (
              <li key={item.href}>
                <Link
                  href={item.href}
                  className={`flex items-center gap-3 px-4 py-3 rounded-lg transition-colors ${
                    isActive
                      ? 'bg-blue-600 text-white'
                      : 'text-slate-300 hover:bg-slate-800'
                  }`}
                >
                  <Icon className="w-5 h-5" />
                  <span>{item.label}</span>
                </Link>
              </li>
            );
          })}
        </ul>
      </nav>
      
      <div className="p-4 border-t border-slate-700">
        <div className="text-xs text-slate-400">
          <p>API: {process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8080'}</p>
          <p>Status: Connected</p>
        </div>
      </div>
    </aside>
  );
}
