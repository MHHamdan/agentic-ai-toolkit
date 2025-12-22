interface CNSRGaugeProps {
  value: number
  size?: 'sm' | 'md' | 'lg'
}

export default function CNSRGauge({ value, size = 'lg' }: CNSRGaugeProps) {
  const sizeClasses = {
    sm: 'w-24 h-24',
    md: 'w-36 h-36',
    lg: 'w-48 h-48',
  }

  const textSizes = {
    sm: 'text-xl',
    md: 'text-2xl',
    lg: 'text-4xl',
  }

  // Normalize value for display (assuming CNSR typically 0-50)
  const normalizedValue = Math.min(value, 50)
  const percentage = (normalizedValue / 50) * 100
  const strokeDasharray = `${percentage} ${100 - percentage}`

  // Color based on CNSR value
  const getColor = () => {
    if (value >= 25) return '#22c55e' // green
    if (value >= 15) return '#eab308' // yellow
    return '#ef4444' // red
  }

  return (
    <div className="flex flex-col items-center">
      <div className={`relative ${sizeClasses[size]}`}>
        <svg className="w-full h-full transform -rotate-90" viewBox="0 0 36 36">
          {/* Background circle */}
          <circle
            cx="18"
            cy="18"
            r="15.5"
            fill="none"
            stroke="currentColor"
            strokeWidth="3"
            className="text-muted"
          />
          {/* Progress circle */}
          <circle
            cx="18"
            cy="18"
            r="15.5"
            fill="none"
            stroke={getColor()}
            strokeWidth="3"
            strokeDasharray={strokeDasharray}
            strokeLinecap="round"
            className="transition-all duration-500"
          />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className={`font-bold ${textSizes[size]} text-foreground`}>
            {value.toFixed(1)}
          </span>
          <span className="text-xs text-muted-foreground">CNSR</span>
        </div>
      </div>
      <p className="mt-2 text-sm text-muted-foreground">
        Cost-Normalized Success Rate
      </p>
    </div>
  )
}
