import { ReactNode } from 'react'
import { TrendingUp, TrendingDown, Minus } from 'lucide-react'

interface MetricCardProps {
  title: string
  value: string | number
  subtitle?: string
  trend?: 'up' | 'down' | 'stable'
  trendValue?: string
  icon?: ReactNode
  color?: 'blue' | 'green' | 'yellow' | 'red' | 'purple'
}

const colorClasses = {
  blue: 'bg-blue-500/10 text-blue-500',
  green: 'bg-green-500/10 text-green-500',
  yellow: 'bg-yellow-500/10 text-yellow-500',
  red: 'bg-red-500/10 text-red-500',
  purple: 'bg-purple-500/10 text-purple-500',
}

export default function MetricCard({
  title,
  value,
  subtitle,
  trend,
  trendValue,
  icon,
  color = 'blue',
}: MetricCardProps) {
  return (
    <div className="bg-card rounded-xl border border-border p-6">
      <div className="flex items-start justify-between">
        <div>
          <p className="text-sm text-muted-foreground">{title}</p>
          <p className="text-3xl font-bold mt-2 text-foreground">{value}</p>
          {subtitle && (
            <p className="text-sm text-muted-foreground mt-1">{subtitle}</p>
          )}
        </div>
        {icon && (
          <div className={`p-3 rounded-lg ${colorClasses[color]}`}>
            {icon}
          </div>
        )}
      </div>
      {trend && trendValue && (
        <div className="mt-4 flex items-center text-sm">
          {trend === 'up' && (
            <TrendingUp className="w-4 h-4 text-green-500 mr-1" />
          )}
          {trend === 'down' && (
            <TrendingDown className="w-4 h-4 text-red-500 mr-1" />
          )}
          {trend === 'stable' && (
            <Minus className="w-4 h-4 text-muted-foreground mr-1" />
          )}
          <span
            className={
              trend === 'up'
                ? 'text-green-500'
                : trend === 'down'
                ? 'text-red-500'
                : 'text-muted-foreground'
            }
          >
            {trendValue}
          </span>
          <span className="text-muted-foreground ml-1">vs last period</span>
        </div>
      )}
    </div>
  )
}
