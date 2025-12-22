import { PieChart, Pie, Cell, ResponsiveContainer, Legend, Tooltip } from 'recharts'

interface CostBreakdownProps {
  data: {
    inference: number
    tools: number
    latency: number
    human: number
  }
}

const COLORS = ['#3b82f6', '#22c55e', '#eab308', '#ef4444']

export default function CostBreakdownChart({ data }: CostBreakdownProps) {
  const chartData = [
    { name: 'Inference', value: data.inference, color: COLORS[0] },
    { name: 'Tools', value: data.tools, color: COLORS[1] },
    { name: 'Latency', value: data.latency, color: COLORS[2] },
    { name: 'Human', value: data.human, color: COLORS[3] },
  ]

  const total = chartData.reduce((sum, item) => sum + item.value, 0)

  return (
    <div className="bg-card rounded-xl border border-border p-6">
      <h3 className="text-lg font-semibold mb-4 text-foreground">
        Cost Breakdown (4-Component Model)
      </h3>
      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <PieChart>
            <Pie
              data={chartData}
              cx="50%"
              cy="50%"
              innerRadius={60}
              outerRadius={80}
              paddingAngle={2}
              dataKey="value"
            >
              {chartData.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={entry.color} />
              ))}
            </Pie>
            <Tooltip
              formatter={(value: number) => [`$${value.toFixed(4)}`, '']}
              contentStyle={{
                backgroundColor: 'hsl(var(--card))',
                border: '1px solid hsl(var(--border))',
                borderRadius: '8px',
              }}
            />
            <Legend />
          </PieChart>
        </ResponsiveContainer>
      </div>
      <div className="mt-4 text-center">
        <p className="text-sm text-muted-foreground">Total Cost</p>
        <p className="text-2xl font-bold text-foreground">${total.toFixed(4)}</p>
      </div>
    </div>
  )
}
