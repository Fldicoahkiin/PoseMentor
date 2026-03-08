import { Suspense, useState } from 'react';
import { BrowserRouter, Routes, Route, Link, useLocation } from 'react-router-dom';
import { PanelLeftClose, PanelLeftOpen, UserRoundCog } from 'lucide-react';
import { cn } from './lib/utils';
import DemoPage from './pages/DemoPage';

type SidebarProps = {
  collapsed: boolean;
  onToggle: () => void;
};

function Sidebar({ collapsed, onToggle }: SidebarProps) {
  const location = useLocation();

  const links = [
    { name: '训练工作台', href: '/', icon: UserRoundCog },
  ];

  return (
    <div
      className={cn(
        "h-screen border-r bg-zinc-50 border-zinc-200 flex flex-col overflow-y-auto shrink-0 select-none transition-all duration-200",
        collapsed ? "w-20 items-center px-2 py-6" : "w-64 items-start px-4 py-8",
      )}
    >
      <div className={cn("mb-8 w-full", collapsed ? "flex flex-col items-center gap-3" : "flex items-center gap-3 px-2")}>
        <button
          type="button"
          onClick={onToggle}
          className={cn(
            "rounded-md border border-zinc-200 bg-white text-zinc-600 hover:bg-zinc-100",
            collapsed ? "h-8 w-8" : "h-8 w-8",
          )}
          title={collapsed ? "展开导航" : "折叠导航"}
        >
          {collapsed ? <PanelLeftOpen size={15} className="mx-auto" /> : <PanelLeftClose size={15} className="mx-auto" />}
        </button>
        <div className="w-8 h-8 rounded-lg overflow-hidden border border-zinc-700 bg-zinc-900 flex items-center justify-center">
          <img src="/posementor-favicon.svg" alt="PoseMentor" className="h-8 w-8 object-cover" />
        </div>
        {!collapsed && <span className="font-bold text-lg tracking-tight">PoseMentor</span>}
      </div>

      {!collapsed && (
        <div className="px-2 mb-4 text-xs font-semibold text-zinc-500 uppercase tracking-wider w-full">
          模块导航
        </div>
      )}

      <nav className="flex flex-col gap-2 w-full flex-1">
        {links.map((link) => {
          const isActive = location.pathname === link.href;
          const Icon = link.icon;
          return (
            <Link
              key={link.name}
              to={link.href}
              title={collapsed ? link.name : undefined}
              className={cn(
                "flex items-center rounded-lg text-sm font-medium transition-all duration-200",
                collapsed ? "justify-center px-0 py-3" : "gap-3 px-3 py-2.5",
                isActive
                  ? "bg-zinc-900 text-zinc-50 shadow-sm shadow-zinc-200"
                  : "text-zinc-600 hover:bg-zinc-200/50 hover:text-zinc-900"
              )}
            >
              <Icon size={18} strokeWidth={isActive ? 2.5 : 2} />
              {!collapsed && link.name}
            </Link>
          );
        })}
      </nav>

      {!collapsed && (
        <div className="mt-8 px-2 w-full">
          <div className="glass-card rounded-xl p-4 text-xs text-zinc-500 text-center">
            PoseMentor <br />
            单摄像头动作教学系统
          </div>
        </div>
      )}
    </div>
  );
}

function Shell() {
  const location = useLocation();
  const isDemoPage = location.pathname === '/';
  const [sidebarCollapsed, setSidebarCollapsed] = useState(true);

  return (
    <div className="flex h-screen bg-white text-zinc-900 font-sans antialiased overflow-hidden">
      <Sidebar collapsed={sidebarCollapsed} onToggle={() => setSidebarCollapsed((value) => !value)} />
      <main className="flex-1 h-screen overflow-y-auto bg-zinc-50/50">
        <Suspense fallback={<div className="p-8 text-zinc-500 flex items-center justify-center h-full">加载中...</div>}>
          <div className={cn("mx-auto py-10", isDemoPage ? "max-w-none px-6" : "max-w-6xl px-8")}>
            <Routes>
              <Route path="/" element={<DemoPage />} />
            </Routes>
          </div>
        </Suspense>
      </main>
    </div>
  );
}

function App() {
  return (
    <BrowserRouter>
      <Shell />
    </BrowserRouter>
  );
}

export default App;
