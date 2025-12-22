import { useState, useEffect } from 'react'
import { CheckCircle, Clock, Filter, Search } from 'lucide-react'
import { api, Incident } from '../api/client'

// 10 Failure Pathology Types
const PATHOLOGY_TYPES = [
  { type: 'hallucination', label: 'Hallucination', color: 'red' },
  { type: 'goal_drift', label: 'Goal Drift', color: 'orange' },
  { type: 'infinite_loop', label: 'Infinite Loop', color: 'red' },
  { type: 'resource_exhaustion', label: 'Resource Exhaustion', color: 'yellow' },
  { type: 'tool_misuse', label: 'Tool Misuse', color: 'orange' },
  { type: 'context_overflow', label: 'Context Overflow', color: 'yellow' },
  { type: 'permission_violation', label: 'Permission Violation', color: 'red' },
  { type: 'data_leakage', label: 'Data Leakage', color: 'red' },
  { type: 'incomplete_task', label: 'Incomplete Task', color: 'blue' },
  { type: 'incorrect_output', label: 'Incorrect Output', color: 'orange' },
]

export default function Incidents() {
  const [incidents, setIncidents] = useState<Incident[]>([])
  const [loading, setLoading] = useState(true)
  const [filter, setFilter] = useState<'all' | 'active' | 'resolved'>('all')
  const [typeFilter, setTypeFilter] = useState<string>('all')
  const [searchQuery, setSearchQuery] = useState('')

  useEffect(() => {
    fetchIncidents()
  }, [])

  async function fetchIncidents() {
    try {
      const data = await api.getIncidents()
      setIncidents(data)
    } catch {
      // Demo data with all 10 pathology types
      setIncidents([
        {
          id: '1',
          type: 'hallucination',
          severity: 'high',
          description: 'Model generated unverified claims about API responses',
          task_id: 'task-042',
          timestamp: new Date(Date.now() - 1800000).toISOString(),
          resolved: false,
          resolved_at: null,
        },
        {
          id: '2',
          type: 'goal_drift',
          severity: 'medium',
          description: 'Agent deviated from original task objective during execution',
          task_id: 'task-056',
          timestamp: new Date(Date.now() - 3600000).toISOString(),
          resolved: true,
          resolved_at: new Date(Date.now() - 1800000).toISOString(),
        },
        {
          id: '3',
          type: 'tool_misuse',
          severity: 'medium',
          description: 'Incorrect parameters passed to file system tool',
          task_id: 'task-078',
          timestamp: new Date(Date.now() - 7200000).toISOString(),
          resolved: true,
          resolved_at: new Date(Date.now() - 3600000).toISOString(),
        },
        {
          id: '4',
          type: 'context_overflow',
          severity: 'low',
          description: 'Context window exceeded, early messages truncated',
          task_id: 'task-091',
          timestamp: new Date(Date.now() - 10800000).toISOString(),
          resolved: true,
          resolved_at: new Date(Date.now() - 9000000).toISOString(),
        },
        {
          id: '5',
          type: 'incomplete_task',
          severity: 'medium',
          description: 'Task terminated before all subtasks completed',
          task_id: 'task-102',
          timestamp: new Date(Date.now() - 14400000).toISOString(),
          resolved: false,
          resolved_at: null,
        },
        {
          id: '6',
          type: 'permission_violation',
          severity: 'critical',
          description: 'Attempted to access restricted system directory',
          task_id: 'task-115',
          timestamp: new Date(Date.now() - 18000000).toISOString(),
          resolved: true,
          resolved_at: new Date(Date.now() - 16200000).toISOString(),
        },
      ])
    } finally {
      setLoading(false)
    }
  }

  async function handleResolve(id: string) {
    try {
      await api.resolveIncident(id)
      setIncidents(
        incidents.map((i) =>
          i.id === id ? { ...i, resolved: true, resolved_at: new Date().toISOString() } : i
        )
      )
    } catch {
      setIncidents(
        incidents.map((i) =>
          i.id === id ? { ...i, resolved: true, resolved_at: new Date().toISOString() } : i
        )
      )
    }
  }

  const filteredIncidents = incidents.filter((i) => {
    if (filter === 'active' && i.resolved) return false
    if (filter === 'resolved' && !i.resolved) return false
    if (typeFilter !== 'all' && i.type !== typeFilter) return false
    if (searchQuery && !i.description.toLowerCase().includes(searchQuery.toLowerCase())) return false
    return true
  })

  const getSeverityColor = (severity: Incident['severity']) => {
    const colors = {
      critical: 'bg-red-500',
      high: 'bg-orange-500',
      medium: 'bg-yellow-500',
      low: 'bg-blue-500',
    }
    return colors[severity]
  }

  const activeCount = incidents.filter((i) => !i.resolved).length
  const criticalCount = incidents.filter((i) => i.severity === 'critical' && !i.resolved).length

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
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-2xl font-bold text-foreground">Incidents & Pathologies</h1>
          <p className="text-muted-foreground">
            Track and resolve the 10 agent failure pathology types
          </p>
        </div>
        <div className="flex items-center space-x-4">
          <div className="text-center px-4 py-2 bg-red-500/10 rounded-lg">
            <p className="text-2xl font-bold text-red-500">{activeCount}</p>
            <p className="text-xs text-muted-foreground">Active</p>
          </div>
          {criticalCount > 0 && (
            <div className="text-center px-4 py-2 bg-red-500/20 rounded-lg border border-red-500/30">
              <p className="text-2xl font-bold text-red-500">{criticalCount}</p>
              <p className="text-xs text-red-500">Critical</p>
            </div>
          )}
        </div>
      </div>

      {/* 10 Pathology Types Overview */}
      <div className="bg-card rounded-xl border border-border p-6">
        <h3 className="text-lg font-semibold mb-4 text-foreground">10 Failure Pathology Types</h3>
        <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
          {PATHOLOGY_TYPES.map((pathology) => {
            const count = incidents.filter((i) => i.type === pathology.type && !i.resolved).length
            return (
              <button
                key={pathology.type}
                onClick={() => setTypeFilter(typeFilter === pathology.type ? 'all' : pathology.type)}
                className={`p-3 rounded-lg border transition-colors ${
                  typeFilter === pathology.type
                    ? 'bg-primary/10 border-primary'
                    : 'bg-muted/30 border-border hover:border-primary/50'
                }`}
              >
                <div className="flex items-center justify-between mb-1">
                  <span
                    className={`w-2 h-2 rounded-full ${
                      pathology.color === 'red'
                        ? 'bg-red-500'
                        : pathology.color === 'orange'
                        ? 'bg-orange-500'
                        : pathology.color === 'yellow'
                        ? 'bg-yellow-500'
                        : 'bg-blue-500'
                    }`}
                  />
                  {count > 0 && (
                    <span className="text-xs bg-red-500/10 text-red-500 px-1.5 rounded-full">
                      {count}
                    </span>
                  )}
                </div>
                <p className="text-xs font-medium text-foreground">{pathology.label}</p>
              </button>
            )
          })}
        </div>
      </div>

      {/* Filters */}
      <div className="flex flex-wrap items-center gap-4">
        <div className="relative flex-1 max-w-md">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
          <input
            type="text"
            placeholder="Search incidents..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full pl-10 pr-4 py-2 bg-background border border-border rounded-lg text-foreground focus:outline-none focus:ring-2 focus:ring-primary"
          />
        </div>
        <div className="flex items-center space-x-2">
          <Filter className="w-4 h-4 text-muted-foreground" />
          {(['all', 'active', 'resolved'] as const).map((f) => (
            <button
              key={f}
              onClick={() => setFilter(f)}
              className={`px-3 py-1.5 text-sm rounded-lg transition-colors ${
                filter === f
                  ? 'bg-primary text-primary-foreground'
                  : 'bg-muted text-muted-foreground hover:bg-muted/80'
              }`}
            >
              {f.charAt(0).toUpperCase() + f.slice(1)}
            </button>
          ))}
        </div>
      </div>

      {/* Incidents List */}
      <div className="space-y-3">
        {filteredIncidents.length === 0 ? (
          <div className="bg-card rounded-xl border border-border p-12 text-center">
            <CheckCircle className="w-12 h-12 mx-auto text-green-500 mb-4" />
            <p className="text-lg font-medium text-foreground">No incidents found</p>
            <p className="text-muted-foreground">
              {filter === 'active' ? 'All incidents have been resolved' : 'No matching incidents'}
            </p>
          </div>
        ) : (
          filteredIncidents.map((incident) => (
            <div
              key={incident.id}
              className="bg-card rounded-xl border border-border p-6 hover:border-primary/30 transition-colors"
            >
              <div className="flex items-start justify-between">
                <div className="flex items-start space-x-4">
                  <div className={`w-3 h-3 rounded-full mt-1.5 ${getSeverityColor(incident.severity)}`} />
                  <div>
                    <div className="flex items-center space-x-3 mb-1">
                      <h4 className="font-medium text-foreground capitalize">
                        {incident.type.replace('_', ' ')}
                      </h4>
                      <span
                        className={`text-xs px-2 py-0.5 rounded-full ${
                          incident.severity === 'critical'
                            ? 'bg-red-500/10 text-red-500'
                            : incident.severity === 'high'
                            ? 'bg-orange-500/10 text-orange-500'
                            : incident.severity === 'medium'
                            ? 'bg-yellow-500/10 text-yellow-500'
                            : 'bg-blue-500/10 text-blue-500'
                        }`}
                      >
                        {incident.severity}
                      </span>
                      {incident.task_id && (
                        <span className="text-xs text-muted-foreground font-mono">
                          {incident.task_id}
                        </span>
                      )}
                    </div>
                    <p className="text-muted-foreground">{incident.description}</p>
                    <div className="flex items-center space-x-4 mt-3 text-xs text-muted-foreground">
                      <span className="flex items-center">
                        <Clock className="w-3 h-3 mr-1" />
                        {new Date(incident.timestamp).toLocaleString()}
                      </span>
                      {incident.resolved && incident.resolved_at && (
                        <span className="flex items-center text-green-500">
                          <CheckCircle className="w-3 h-3 mr-1" />
                          Resolved {new Date(incident.resolved_at).toLocaleString()}
                        </span>
                      )}
                    </div>
                  </div>
                </div>
                <div className="flex items-center space-x-2">
                  {!incident.resolved ? (
                    <button
                      onClick={() => handleResolve(incident.id)}
                      className="px-3 py-1.5 text-sm bg-green-500/10 text-green-500 rounded-lg hover:bg-green-500/20 transition-colors"
                    >
                      Resolve
                    </button>
                  ) : (
                    <span className="px-3 py-1.5 text-sm bg-green-500/10 text-green-500 rounded-lg">
                      Resolved
                    </span>
                  )}
                </div>
              </div>
            </div>
          ))
        )}
      </div>

      {/* Incident Response Info */}
      <div className="bg-card rounded-xl border border-border p-6">
        <h3 className="text-lg font-semibold mb-4 text-foreground">Incident Response Protocol</h3>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="p-4 bg-red-500/10 rounded-lg">
            <div className="flex items-center space-x-2 mb-2">
              <div className="w-3 h-3 rounded-full bg-red-500" />
              <span className="font-medium text-red-500">Critical</span>
            </div>
            <p className="text-sm text-muted-foreground">
              Immediate halt, human review required. Auto-escalation enabled.
            </p>
          </div>
          <div className="p-4 bg-orange-500/10 rounded-lg">
            <div className="flex items-center space-x-2 mb-2">
              <div className="w-3 h-3 rounded-full bg-orange-500" />
              <span className="font-medium text-orange-500">High</span>
            </div>
            <p className="text-sm text-muted-foreground">
              Task suspended, notification sent. Review within 1 hour.
            </p>
          </div>
          <div className="p-4 bg-yellow-500/10 rounded-lg">
            <div className="flex items-center space-x-2 mb-2">
              <div className="w-3 h-3 rounded-full bg-yellow-500" />
              <span className="font-medium text-yellow-500">Medium</span>
            </div>
            <p className="text-sm text-muted-foreground">
              Logged for review, task continues with monitoring.
            </p>
          </div>
          <div className="p-4 bg-blue-500/10 rounded-lg">
            <div className="flex items-center space-x-2 mb-2">
              <div className="w-3 h-3 rounded-full bg-blue-500" />
              <span className="font-medium text-blue-500">Low</span>
            </div>
            <p className="text-sm text-muted-foreground">
              Recorded for analysis, no immediate action required.
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}
