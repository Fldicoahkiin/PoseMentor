import React from 'react';
import { BrowserRouter, Routes, Route, Link, useLocation } from 'react-router-dom';
import { Terminal, LayoutDashboard, Settings, User } from 'lucide-react';
import { cn } from './lib/utils';
import DemoPage from './pages/DemoPage';
import AdminPage from './pages/AdminPage';

function Sidebar() {
  const location = useLocation();

  const links = [
    { name: '在线演示Demo', href: '/', icon: User },
    { name: '系统管理后台', href: '/admin', icon: LayoutDashboard },
  ];

  return (
    <div className="w-64 h-screen border-r bg-zinc-50 border-zinc-200 flex flex-col items-start px-4 py-8 overflow-y-auto shrink-0 select-none">
      <div className="flex items-center gap-3 px-2 mb-10 w-full">
        <div className="w-8 h-8 rounded-lg bg-zinc-900 text-white flex items-center justify-center">
          <Terminal size={18} strokeWidth={2.5} />
        </div>
        <span className="font-bold text-lg tracking-tight">PoseMentor</span>
      </div>

      <div className="px-2 mb-4 text-xs font-semibold text-zinc-500 uppercase tracking-wider w-full">
        模块导航
      </div>

      <nav className="flex flex-col gap-2 w-full flex-1">
        {links.map((link) => {
          const isActive = location.pathname === link.href;
          const Icon = link.icon;
          return (
            <Link
              key={link.name}
              to={link.href}
              className={cn(
                "flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-all duration-200",
                isActive
                  ? "bg-zinc-900 text-zinc-50 shadow-sm shadow-zinc-200"
                  : "text-zinc-600 hover:bg-zinc-200/50 hover:text-zinc-900"
              )}
            >
              <Icon size={18} strokeWidth={isActive ? 2.5 : 2} />
              {link.name}
            </Link>
          );
        })}
      </nav>

      <div className="mt-8 px-2 w-full">
        <div className="glass-card rounded-xl p-4 text-xs text-zinc-500 text-center">
          AIST++ <br />
          AI舞蹈矫正平台
        </div>
      </div>
    </div>
  );
}

function App() {
  return (
    <BrowserRouter>
      <div className="flex h-screen bg-white text-zinc-900 font-sans antialiased overflow-hidden">
        <Sidebar />
        <main className="flex-1 h-screen overflow-y-auto bg-zinc-50/50">
          <React.Suspense fallback={<div className="p-8 text-zinc-500 flex items-center justify-center h-full">加载中...</div>}>
            <div className="max-w-6xl mx-auto px-8 py-10">
              <Routes>
                <Route path="/" element={<DemoPage />} />
                <Route path="/admin" element={<AdminPage />} />
              </Routes>
            </div>
          </React.Suspense>
        </main>
      </div>
    </BrowserRouter>
  );
}

export default App;
