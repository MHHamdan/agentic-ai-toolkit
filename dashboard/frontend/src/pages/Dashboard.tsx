import { useEffect, useState } from 'react'
import { DollarSign, CheckCircle, AlertTriangle } from 'lucide-react'
import MetricCard from '../components/cards/MetricCard'
import CNSRGauge from '../components/charts/CNSRGauge'
import CostBreakdownChart from '../components/charts/CostBreakdownChart'
import SuccessTrendChart from '../components/charts/SuccessTrendChart'
import { api, CNSRMetrics, Incident } from '../api/client'

export default function Dashboard() {
  const [metrics, setMetrics] = useState<CNSRMetrics | null>(null)
  const [incidents, setIncidents] = useState<Incident[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    async function fetchData() {
      try {
        const [metricsData, incidentsData] = await Promise.all([
          api.getCNSR(),
          api.getIncidents(),
        ])
        setMetrics(metricsData)
        setIncidents(incidentsData)
        setError(null)
      } catch (err) {
        setError('Failed to fetch data. Using demo data.')
        // Use demo data for display
        setMetrics({
          cnsr: 24.5,
          success_rate: 0.85,
          mean_cost: 0.0347,
          total_tasks: 121,
          total_successes: 103,
          cost_breakdown: {
            inference_cost: 0.0234,
            tool_cost: 0.0056,
            latency_cost: 0.0045,
            human_cost: 0.0012,
            total_cost: 0.0347,
          },
        })
        setIncidents([
          {
            id: '1',
            type: 'hallucination',
            severity: 'medium',
            description: 'Model generated unverified information',
            task_id: 'task-42',
            timestamp: new Date().toISOString(),
            resolved: false,
            resolved_at: null,
          },
          {
            id: '2',
            type: 'goal_drift',
            severity: 'low',
            description: 'Slight deviation from original objective',
            task_id: 'task-56',
            timestamp: new Date(Date.now() - 3600000).toISOString(),
            resolved: true,
            resolved_at: new Date().toISOString(),
          },
        ])
      } finally {
        setLoading(false)
      }
    }

    fetchData()
    const interval = setInterval(fetchData, 30000) // Refresh every 30s
    return () => clearInterval(interval)
  }, [])

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary"></div>
      </div>
    )
  }

  const demoTrendData = [0.78, 0.82, 0.79, 0.85, 0.83, 0.87, 0.85, 0.88, 0.86, 0.85]
  const activeIncidents = incidents.filter((i) => !i.resolved)

  return (
    <div className="space-y-6">
      {error && (
        <div className="bg-yellow-500/10 border border-yellow-500/20 rounded-lg p-4">
          <p className="text-yellow-500 text-sm">{error}</p>
        </div>
      )}

      {/* Top Row: CNSR Gauge + Metric Cards */}
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* CNSR Gauge - Takes 1 column */}
        <div className="bg-card rounded-xl border border-border p-6 flex items-center justify-center">
          <CNSRGauge value={metrics?.cnsr ?? 0} />
        </div>

        {/* Metric Cards - Take 3 columns */}
        <div className="lg:col-span-3 grid grid-cols-1 md:grid-cols-3 gap-6">
          <MetricCard
            title="Success Rate"
            value={`${((metrics?.success_rate ?? 0) * 100).toFixed(1)}%`}
            subtitle={`${metrics?.total_successes ?? 0} of ${metrics?.total_tasks ?? 0} tasks`}
            trend="up"
            trendValue="+2.3%"
            icon={<CheckCircle className="w-6 h-6" />}
            color="green"
          />
          <MetricCard
            title="Mean Cost"
            value={`$${(metrics?.mean_cost ?? 0).toFixed(4)}`}
            subtitle="Per task average"
            trend="down"
            trendValue="-5.2%"
            icon={<DollarSign className="w-6 h-6" />}
            color="blue"
          />
          <MetricCard
            title="Active Incidents"
            value={activeIncidents.length}
            subtitle={`${incidents.length} total incidents`}
            trend={activeIncidents.length > 0 ? 'up' : 'stable'}
            trendValue={activeIncidents.length > 0 ? 'Needs attention' : 'All clear'}
            icon={<AlertTriangle className="w-6 h-6" />}
            color={activeIncidents.length > 0 ? 'red' : 'green'}
          />
        </div>
      </div>

      {/* Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <CostBreakdownChart
          data={{
            inference: metrics?.cost_breakdown.inference_cost ?? 0,
            tools: metrics?.cost_breakdown.tool_cost ?? 0,
            latency: metrics?.cost_breakdown.latency_cost ?? 0,
            human: metrics?.cost_breakdown.human_cost ?? 0,
          }}
        />
        <SuccessTrendChart data={demoTrendData} />
      </div>

      {/* Bottom Row: Task Overview + Recent Incidents */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Task Overview */}
        <div className="bg-card rounded-xl border border-border p-6">
          <h3 className="text-lg font-semibold mb-4 text-foreground">
            Evaluation Overview
          </h3>
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-muted-foreground">Total Tasks Evaluated</span>
              <span className="font-semibold text-foreground">{metrics?.total_tasks ?? 0}</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-muted-foreground">Successful Tasks</span>
              <span className="font-semibold text-green-500">{metrics?.total_successes ?? 0}</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-muted-foreground">Failed Tasks</span>
              <span className="font-semibold text-red-500">
                {(metrics?.total_tasks ?? 0) - (metrics?.total_successes ?? 0)}
              </span>
            </div>
            <div className="h-px bg-border my-4" />
            <div className="flex items-center justify-between">
              <span className="text-muted-foreground">4-Component Cost Model</span>
              <span className="font-semibold text-foreground">Active</span>
            </div>
            <div className="text-xs text-muted-foreground">
              C<sub>total</sub> = C<sub>inference</sub> + C<sub>tools</sub> + C<sub>latency</sub> + C<sub>human</sub>
            </div>
          </div>
        </div>

        {/* Recent Incidents */}
        <div className="bg-card rounded-xl border border-border p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-foreground">Recent Incidents</h3>
            <span className="text-xs bg-muted px-2 py-1 rounded-full text-muted-foreground">
              10 Pathology Types
            </span>
          </div>
          <div className="space-y-3">
            {incidents.slice(0, 5).map((incident) => (
              <div
                key={incident.id}
                className="flex items-center justify-between p-3 bg-muted/50 rounded-lg"
              >
                <div className="flex items-center space-x-3">
                  <div
                    className={`w-2 h-2 rounded-full ${
                      incident.severity === 'critical'
                        ? 'bg-red-500'
                        : incident.severity === 'high'
                        ? 'bg-orange-500'
                        : incident.severity === 'medium'
                        ? 'bg-yellow-500'
                        : 'bg-blue-500'
                    }`}
                  />
                  <div>
                    <p className="text-sm font-medium text-foreground capitalize">
                      {incident.type.replace('_', ' ')}
                    </p>
                    <p className="text-xs text-muted-foreground truncate max-w-[200px]">
                      {incident.description}
                    </p>
                  </div>
                </div>
                <span
                  className={`text-xs px-2 py-1 rounded-full ${
                    incident.resolved
                      ? 'bg-green-500/10 text-green-500'
                      : 'bg-red-500/10 text-red-500'
                  }`}
                >
                  {incident.resolved ? 'Resolved' : 'Active'}
                </span>
              </div>
            ))}
            {incidents.length === 0 && (
              <p className="text-center text-muted-foreground py-4">No incidents recorded</p>
            )}
          </div>
        </div>
      </div>

      {/* Autonomy Levels Info */}
      <div className="bg-card rounded-xl border border-border p-6">
        <h3 className="text-lg font-semibold mb-4 text-foreground">
          Autonomy Framework (5 Levels)
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
          {[
            { level: 1, name: 'Human-in-the-Loop', desc: 'Full human control' },
            { level: 2, name: 'Human-on-the-Loop', desc: 'Human oversight' },
            { level: 3, name: 'Human-out-of-Loop', desc: 'Supervised autonomy' },
            { level: 4, name: 'Bounded Autonomy', desc: 'Limited independence' },
            { level: 5, name: 'Full Autonomy', desc: 'Complete independence' },
          ].map((item) => (
            <div
              key={item.level}
              className="text-center p-4 bg-muted/30 rounded-lg border border-border"
            >
              <div className="w-10 h-10 mx-auto mb-2 rounded-full bg-primary/10 flex items-center justify-center">
                <span className="text-lg font-bold text-primary">{item.level}</span>
              </div>
              <p className="text-sm font-medium text-foreground">{item.name}</p>
              <p className="text-xs text-muted-foreground mt-1">{item.desc}</p>
            </div>
          ))}
        </div>
        <div className="mt-4 text-xs text-muted-foreground">
          <strong>4 Classification Criteria:</strong> Action Scope Freedom (ASF), Goal Definition Power (GDP),
          Decision Timing (DT), Error Recovery (ER)
        </div>
      </div>
    </div>
  )
}
