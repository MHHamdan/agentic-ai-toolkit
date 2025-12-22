import { Link, useLocation } from 'react-router-dom'
import {
  LayoutDashboard,
  PlayCircle,
  DollarSign,
  AlertTriangle,
  Shield,
  Settings,
  Bot,
} from 'lucide-react'

const navItems = [
  { icon: LayoutDashboard, label: 'Dashboard', path: '/' },
  { icon: PlayCircle, label: 'Evaluations', path: '/evaluations' },
  { icon: DollarSign, label: 'Cost Analysis', path: '/costs' },
  { icon: AlertTriangle, label: 'Incidents', path: '/incidents' },
  { icon: Shield, label: 'Safety', path: '/safety' },
  { icon: Settings, label: 'Settings', path: '/settings' },
]

export default function Sidebar() {
  const location = useLocation()

  return (
    <aside className="w-64 bg-card border-r border-border flex flex-col">
      {/* Logo */}
      <div className="h-16 flex items-center px-6 border-b border-border">
        <Bot className="w-8 h-8 text-primary mr-3" />
        <div>
          <h1 className="font-bold text-foreground">Agentic AI</h1>
          <p className="text-xs text-muted-foreground">Toolkit Dashboard</p>
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 p-4 space-y-1">
        {navItems.map((item) => {
          const isActive = location.pathname === item.path
          return (
            <Link
              key={item.path}
              to={item.path}
              className={`flex items-center px-4 py-3 rounded-lg transition-colors ${
                isActive
                  ? 'bg-primary text-primary-foreground'
                  : 'text-muted-foreground hover:bg-accent hover:text-accent-foreground'
              }`}
            >
              <item.icon className="w-5 h-5 mr-3" />
              {item.label}
            </Link>
          )
        })}
      </nav>

      {/* Status */}
      <div className="p-4 border-t border-border">
        <div className="flex items-center text-sm">
          <div className="w-2 h-2 rounded-full bg-green-500 mr-2" />
          <span className="text-muted-foreground">Ollama Connected</span>
        </div>
      </div>
    </aside>
  )
}
