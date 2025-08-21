
import React from "react";
import { Link, useLocation } from "react-router-dom";
import { createPageUrl } from "@/utils";
import { Shield, LayoutDashboard, AlertOctagon, Settings, BarChart } from "lucide-react";

export default function Layout({ children, currentPageName }) {
  const location = useLocation();

  const navItems = [
    { name: "Dashboard", icon: LayoutDashboard, path: createPageUrl("Dashboard") },
    { name: "Detections", icon: AlertOctagon, path: createPageUrl("Detections") },
    { name: "Settings", icon: Settings, path: createPageUrl("Settings") },
  ];

  return (
    <div className="min-h-screen w-full flex bg-gray-100 dark:bg-gray-900">
      <aside className="w-64 bg-white dark:bg-gray-950 border-r border-gray-200 dark:border-gray-800 flex flex-col">
        <div className="p-4 border-b border-gray-200 dark:border-gray-800 flex items-center gap-3">
          <div className="bg-blue-600 p-2 rounded-lg">
            <Shield className="text-white" />
          </div>
          <div>
            <h1 className="text-lg font-bold text-gray-900 dark:text-white">WATCHDOG</h1>
            <p className="text-xs text-gray-500 dark:text-gray-400">Verbal Abuse Detection</p>
          </div>
        </div>
        <nav className="flex-1 p-4 space-y-2">
          {navItems.map((item) => (
            <Link
              key={item.name}
              to={item.path}
              className={`flex items-center gap-3 px-3 py-2 rounded-lg transition-colors ${
                location.pathname === item.path
                  ? "bg-blue-50 text-blue-600 dark:bg-blue-900/50 dark:text-blue-400"
                  : "text-gray-600 hover:bg-gray-50 hover:text-gray-900 dark:text-gray-400 dark:hover:bg-gray-800 dark:hover:text-white"
              }`}
            >
              <item.icon className="w-5 h-5" />
              <span>{item.name}</span>
            </Link>
          ))}
        </nav>
        <div className="p-4 border-t border-gray-200 dark:border-gray-800">
           <div className="bg-gray-100 dark:bg-gray-800/50 p-4 rounded-lg">
              <div className="flex items-center gap-3 mb-2">
                <BarChart className="w-5 h-5 text-gray-500"/>
                <h3 className="font-semibold text-sm text-gray-800 dark:text-gray-200">System Status</h3>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse"></div>
                <span className="text-xs text-green-600 dark:text-green-400 font-medium">All systems operational</span>
              </div>
           </div>
        </div>
      </aside>
      <main className="flex-1 overflow-auto">
        <div className="p-4 sm:p-6 lg:p-8">
            {children}
        </div>
      </main>
    </div>
  );
}
