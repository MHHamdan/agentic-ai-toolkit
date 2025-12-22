import { useLocation } from 'react-router-dom'
import { Sun, Moon, Bell, RefreshCw } from 'lucide-react'
import { useThemeStore } from '../../store/themeStore'

const pageTitles: Record<string, string> = {
  '/': 'Dashboard',
  '/evaluations': 'Evaluations',
  '/costs': 'Cost Analysis',
  '/incidents': 'Incidents & Pathologies',
  '/safety': 'Safety Compliance',
  '/settings': 'Settings',
}

export default function Header() {
  const location = useLocation()
  const { theme, toggleTheme } = useThemeStore()
  const title = pageTitles[location.pathname] || 'Dashboard'

  return (
    <header className="h-16 bg-card border-b border-border flex items-center justify-between px-6">
      <h2 className="text-xl font-semibold text-foreground">{title}</h2>

      <div className="flex items-center space-x-4">
        {/* Refresh */}
        <button
          className="p-2 rounded-lg hover:bg-accent transition-colors"
          title="Refresh data"
        >
          <RefreshCw className="w-5 h-5 text-muted-foreground" />
        </button>

        {/* Notifications */}
        <button
          className="p-2 rounded-lg hover:bg-accent transition-colors relative"
          title="Notifications"
        >
          <Bell className="w-5 h-5 text-muted-foreground" />
          <span className="absolute top-1 right-1 w-2 h-2 bg-destructive rounded-full" />
        </button>

        {/* Theme toggle */}
        <button
          onClick={toggleTheme}
          className="p-2 rounded-lg hover:bg-accent transition-colors"
          title={`Switch to ${theme === 'light' ? 'dark' : 'light'} mode`}
        >
          {theme === 'light' ? (
            <Moon className="w-5 h-5 text-muted-foreground" />
          ) : (
            <Sun className="w-5 h-5 text-muted-foreground" />
          )}
        </button>
      </div>
    </header>
  )
}
