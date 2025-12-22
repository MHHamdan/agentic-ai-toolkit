import { useState, useEffect } from 'react'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from 'recharts'
import { DollarSign, TrendingDown, Calculator, Cpu } from 'lucide-react'
import MetricCard from '../components/cards/MetricCard'
import CostBreakdownChart from '../components/charts/CostBreakdownChart'
import { api, CostBreakdown } from '../api/client'

export default function CostAnalysis() {
  const [costs, setCosts] = useState<CostBreakdown | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    async function fetchCosts() {
      try {
        const data = await api.getCostBreakdown()
        setCosts(data)
      } catch {
        // Demo data
        setCosts({
          inference_cost: 0.0234,
          tool_cost: 0.0056,
          latency_cost: 0.0045,
          human_cost: 0.0012,
          total_cost: 0.0347,
        })
      } finally {
        setLoading(false)
      }
    }
    fetchCosts()
  }, [])

  // Demo trend data
  const trendData = [
    { day: 'Mon', inference: 0.021, tools: 0.005, latency: 0.004, human: 0.001 },
    { day: 'Tue', inference: 0.023, tools: 0.006, latency: 0.005, human: 0.001 },
    { day: 'Wed', inference: 0.022, tools: 0.005, latency: 0.004, human: 0.002 },
    { day: 'Thu', inference: 0.025, tools: 0.007, latency: 0.005, human: 0.001 },
    { day: 'Fri', inference: 0.024, tools: 0.006, latency: 0.004, human: 0.001 },
    { day: 'Sat', inference: 0.020, tools: 0.004, latency: 0.003, human: 0.001 },
    { day: 'Sun', inference: 0.023, tools: 0.006, latency: 0.005, human: 0.001 },
  ]

  // Model comparison data
  const modelComparison = [
    { model: 'gemma2:2b', cost: 0.0234, success: 85, cnsr: 24.5 },
    { model: 'llama3.2:3b', cost: 0.0312, success: 88, cnsr: 22.1 },
    { model: 'qwen2:1.5b', cost: 0.0189, success: 79, cnsr: 26.8 },
    { model: 'phi3:mini', cost: 0.0267, success: 82, cnsr: 21.3 },
  ]

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary"></div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-foreground">Cost Analysis</h1>
        <p className="text-muted-foreground">
          4-Component Cost Model: C<sub>total</sub> = C<sub>inference</sub> + C<sub>tools</sub> + C
          <sub>latency</sub> + C<sub>human</sub>
        </p>
      </div>

      {/* Cost Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <MetricCard
          title="Total Cost"
          value={`$${(costs?.total_cost ?? 0).toFixed(4)}`}
          subtitle="Per task average"
          icon={<DollarSign className="w-6 h-6" />}
          color="blue"
        />
        <MetricCard
          title="Inference Cost"
          value={`$${(costs?.inference_cost ?? 0).toFixed(4)}`}
          subtitle={`${(((costs?.inference_cost ?? 0) / (costs?.total_cost ?? 1)) * 100).toFixed(0)}% of total`}
          icon={<Cpu className="w-6 h-6" />}
          color="purple"
        />
        <MetricCard
          title="Tool Cost"
          value={`$${(costs?.tool_cost ?? 0).toFixed(4)}`}
          subtitle={`${(((costs?.tool_cost ?? 0) / (costs?.total_cost ?? 1)) * 100).toFixed(0)}% of total`}
          icon={<Calculator className="w-6 h-6" />}
          color="green"
        />
        <MetricCard
          title="Cost Efficiency"
          value="-5.2%"
          subtitle="vs last period"
          trend="down"
          trendValue="Improving"
          icon={<TrendingDown className="w-6 h-6" />}
          color="green"
        />
      </div>

      {/* Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Cost Breakdown Pie */}
        <CostBreakdownChart
          data={{
            inference: costs?.inference_cost ?? 0,
            tools: costs?.tool_cost ?? 0,
            latency: costs?.latency_cost ?? 0,
            human: costs?.human_cost ?? 0,
          }}
        />

        {/* Cost Trend Line */}
        <div className="bg-card rounded-xl border border-border p-6">
          <h3 className="text-lg font-semibold mb-4 text-foreground">Cost Trend (7 Days)</h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={trendData}>
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                <XAxis dataKey="day" stroke="hsl(var(--muted-foreground))" fontSize={12} />
                <YAxis
                  stroke="hsl(var(--muted-foreground))"
                  fontSize={12}
                  tickFormatter={(v) => `$${v}`}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: 'hsl(var(--card))',
                    border: '1px solid hsl(var(--border))',
                    borderRadius: '8px',
                  }}
                  formatter={(value: number) => [`$${value.toFixed(4)}`, '']}
                />
                <Legend />
                <Line
                  type="monotone"
                  dataKey="inference"
                  stroke="#3b82f6"
                  strokeWidth={2}
                  dot={false}
                  name="Inference"
                />
                <Line
                  type="monotone"
                  dataKey="tools"
                  stroke="#22c55e"
                  strokeWidth={2}
                  dot={false}
                  name="Tools"
                />
                <Line
                  type="monotone"
                  dataKey="latency"
                  stroke="#eab308"
                  strokeWidth={2}
                  dot={false}
                  name="Latency"
                />
                <Line
                  type="monotone"
                  dataKey="human"
                  stroke="#ef4444"
                  strokeWidth={2}
                  dot={false}
                  name="Human"
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Model Comparison */}
      <div className="bg-card rounded-xl border border-border p-6">
        <h3 className="text-lg font-semibold mb-4 text-foreground">Model Cost Comparison</h3>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-border">
                <th className="text-left py-3 text-muted-foreground font-medium">Model</th>
                <th className="text-left py-3 text-muted-foreground font-medium">Avg Cost</th>
                <th className="text-left py-3 text-muted-foreground font-medium">Success Rate</th>
                <th className="text-left py-3 text-muted-foreground font-medium">CNSR</th>
                <th className="text-left py-3 text-muted-foreground font-medium">Efficiency</th>
              </tr>
            </thead>
            <tbody>
              {modelComparison.map((model) => (
                <tr key={model.model} className="border-b border-border/50 hover:bg-muted/30">
                  <td className="py-3">
                    <span className="font-medium text-foreground">{model.model}</span>
                  </td>
                  <td className="py-3 text-foreground">${model.cost.toFixed(4)}</td>
                  <td className="py-3">
                    <span className="text-green-500">{model.success}%</span>
                  </td>
                  <td className="py-3">
                    <span className="font-semibold text-foreground">{model.cnsr.toFixed(1)}</span>
                  </td>
                  <td className="py-3">
                    <div className="w-24 h-2 bg-muted rounded-full overflow-hidden">
                      <div
                        className="h-full bg-primary"
                        style={{ width: `${(model.cnsr / 30) * 100}%` }}
                      />
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Cost Formula */}
      <div className="bg-card rounded-xl border border-border p-6">
        <h3 className="text-lg font-semibold mb-4 text-foreground">4-Component Cost Model</h3>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
          <div className="p-4 bg-blue-500/10 rounded-lg border border-blue-500/20">
            <h4 className="font-medium text-blue-500">C_inference</h4>
            <p className="text-sm text-muted-foreground mt-2">
              LLM inference costs based on token usage. Calculated as: tokens × rate_per_token
            </p>
            <p className="text-xs mt-2 font-mono bg-muted/50 p-2 rounded">
              = input_tokens × $0.001 + output_tokens × $0.002
            </p>
          </div>
          <div className="p-4 bg-green-500/10 rounded-lg border border-green-500/20">
            <h4 className="font-medium text-green-500">C_tools</h4>
            <p className="text-sm text-muted-foreground mt-2">
              Cost of tool/API calls. Varies by tool type and usage frequency.
            </p>
            <p className="text-xs mt-2 font-mono bg-muted/50 p-2 rounded">
              = Σ(tool_calls × tool_rate)
            </p>
          </div>
          <div className="p-4 bg-yellow-500/10 rounded-lg border border-yellow-500/20">
            <h4 className="font-medium text-yellow-500">C_latency</h4>
            <p className="text-sm text-muted-foreground mt-2">
              Time cost representing compute opportunity cost during waiting.
            </p>
            <p className="text-xs mt-2 font-mono bg-muted/50 p-2 rounded">
              = execution_time × $0.001/sec
            </p>
          </div>
          <div className="p-4 bg-red-500/10 rounded-lg border border-red-500/20">
            <h4 className="font-medium text-red-500">C_human</h4>
            <p className="text-sm text-muted-foreground mt-2">
              Human oversight cost for reviews, approvals, and corrections.
            </p>
            <p className="text-xs mt-2 font-mono bg-muted/50 p-2 rounded">
              = review_time × $0.50/min
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}
