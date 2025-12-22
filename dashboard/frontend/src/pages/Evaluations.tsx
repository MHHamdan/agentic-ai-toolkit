import { useState, useEffect } from 'react'
import { Play, CheckCircle, XCircle, Clock, RefreshCw } from 'lucide-react'
import { api, Evaluation } from '../api/client'

export default function Evaluations() {
  const [evaluations, setEvaluations] = useState<Evaluation[]>([])
  const [loading, setLoading] = useState(true)
  const [showNewForm, setShowNewForm] = useState(false)
  const [newEval, setNewEval] = useState({
    name: '',
    model: 'gemma2:2b',
    num_tasks: 10,
  })
  const [runningDemo, setRunningDemo] = useState(false)

  useEffect(() => {
    fetchEvaluations()
    const interval = setInterval(fetchEvaluations, 10000)
    return () => clearInterval(interval)
  }, [])

  async function fetchEvaluations() {
    try {
      const data = await api.getEvaluations()
      setEvaluations(data)
    } catch {
      // Use demo data
      setEvaluations([
        {
          id: 'eval-001',
          name: 'Benchmark Run #1',
          status: 'completed',
          progress: 100,
          total_tasks: 121,
          completed_tasks: 121,
          success_rate: 0.85,
          cnsr: 24.5,
          started_at: new Date(Date.now() - 3600000).toISOString(),
          completed_at: new Date().toISOString(),
        },
        {
          id: 'eval-002',
          name: 'Cost Analysis Test',
          status: 'running',
          progress: 45,
          total_tasks: 50,
          completed_tasks: 22,
          success_rate: null,
          cnsr: null,
          started_at: new Date(Date.now() - 1800000).toISOString(),
          completed_at: null,
        },
        {
          id: 'eval-003',
          name: 'Safety Compliance',
          status: 'pending',
          progress: 0,
          total_tasks: 30,
          completed_tasks: 0,
          success_rate: null,
          cnsr: null,
          started_at: new Date().toISOString(),
          completed_at: null,
        },
      ])
    } finally {
      setLoading(false)
    }
  }

  async function handleStartEvaluation() {
    try {
      await api.startEvaluation(newEval)
      setShowNewForm(false)
      fetchEvaluations()
    } catch {
      // Add demo evaluation
      setEvaluations([
        ...evaluations,
        {
          id: `eval-${Date.now()}`,
          name: newEval.name || 'New Evaluation',
          status: 'pending',
          progress: 0,
          total_tasks: newEval.num_tasks,
          completed_tasks: 0,
          success_rate: null,
          cnsr: null,
          started_at: new Date().toISOString(),
          completed_at: null,
        },
      ])
      setShowNewForm(false)
    }
  }

  async function handleRunDemo() {
    setRunningDemo(true)
    try {
      await api.runDemoEvaluation('gemma2:2b')
      fetchEvaluations()
    } catch (err) {
      console.error('Demo failed:', err)
    } finally {
      setRunningDemo(false)
    }
  }

  const getStatusIcon = (status: Evaluation['status']) => {
    switch (status) {
      case 'completed':
        return <CheckCircle className="w-5 h-5 text-green-500" />
      case 'running':
        return <RefreshCw className="w-5 h-5 text-blue-500 animate-spin" />
      case 'failed':
        return <XCircle className="w-5 h-5 text-red-500" />
      default:
        return <Clock className="w-5 h-5 text-muted-foreground" />
    }
  }

  const getStatusBadge = (status: Evaluation['status']) => {
    const colors = {
      completed: 'bg-green-500/10 text-green-500',
      running: 'bg-blue-500/10 text-blue-500',
      failed: 'bg-red-500/10 text-red-500',
      pending: 'bg-muted text-muted-foreground',
    }
    return colors[status]
  }

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
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-foreground">Evaluations</h1>
          <p className="text-muted-foreground">
            Run and monitor agent evaluations with the Long-Horizon Evaluator
          </p>
        </div>
        <div className="flex items-center space-x-3">
          <button
            onClick={handleRunDemo}
            disabled={runningDemo}
            className="flex items-center space-x-2 px-4 py-2 border border-primary text-primary rounded-lg hover:bg-primary/10 transition-colors disabled:opacity-50"
          >
            {runningDemo ? (
              <RefreshCw className="w-4 h-4 animate-spin" />
            ) : (
              <Play className="w-4 h-4" />
            )}
            <span>{runningDemo ? 'Running...' : 'Quick Demo'}</span>
          </button>
          <button
            onClick={() => setShowNewForm(true)}
            className="flex items-center space-x-2 px-4 py-2 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 transition-colors"
          >
            <Play className="w-4 h-4" />
            <span>New Evaluation</span>
          </button>
        </div>
      </div>

      {/* New Evaluation Form */}
      {showNewForm && (
        <div className="bg-card rounded-xl border border-border p-6">
          <h3 className="text-lg font-semibold mb-4 text-foreground">Start New Evaluation</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <label className="block text-sm text-muted-foreground mb-2">Name</label>
              <input
                type="text"
                value={newEval.name}
                onChange={(e) => setNewEval({ ...newEval, name: e.target.value })}
                placeholder="Evaluation name"
                className="w-full px-3 py-2 bg-background border border-border rounded-lg text-foreground focus:outline-none focus:ring-2 focus:ring-primary"
              />
            </div>
            <div>
              <label className="block text-sm text-muted-foreground mb-2">Model</label>
              <select
                value={newEval.model}
                onChange={(e) => setNewEval({ ...newEval, model: e.target.value })}
                className="w-full px-3 py-2 bg-background border border-border rounded-lg text-foreground focus:outline-none focus:ring-2 focus:ring-primary"
              >
                <option value="gemma2:2b">gemma2:2b</option>
                <option value="llama3.2:3b">llama3.2:3b</option>
                <option value="qwen2:1.5b">qwen2:1.5b</option>
                <option value="phi3:mini">phi3:mini</option>
              </select>
            </div>
            <div>
              <label className="block text-sm text-muted-foreground mb-2">Number of Tasks</label>
              <input
                type="number"
                value={newEval.num_tasks}
                onChange={(e) => setNewEval({ ...newEval, num_tasks: parseInt(e.target.value) })}
                min={1}
                max={100}
                className="w-full px-3 py-2 bg-background border border-border rounded-lg text-foreground focus:outline-none focus:ring-2 focus:ring-primary"
              />
            </div>
          </div>
          <div className="flex justify-end space-x-3 mt-6">
            <button
              onClick={() => setShowNewForm(false)}
              className="px-4 py-2 border border-border rounded-lg text-foreground hover:bg-muted transition-colors"
            >
              Cancel
            </button>
            <button
              onClick={handleStartEvaluation}
              className="px-4 py-2 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 transition-colors"
            >
              Start Evaluation
            </button>
          </div>
        </div>
      )}

      {/* Evaluations List */}
      <div className="bg-card rounded-xl border border-border overflow-hidden">
        <table className="w-full">
          <thead className="bg-muted/50">
            <tr>
              <th className="text-left px-6 py-4 text-sm font-medium text-muted-foreground">
                Status
              </th>
              <th className="text-left px-6 py-4 text-sm font-medium text-muted-foreground">
                Name
              </th>
              <th className="text-left px-6 py-4 text-sm font-medium text-muted-foreground">
                Progress
              </th>
              <th className="text-left px-6 py-4 text-sm font-medium text-muted-foreground">
                Success Rate
              </th>
              <th className="text-left px-6 py-4 text-sm font-medium text-muted-foreground">
                CNSR
              </th>
              <th className="text-left px-6 py-4 text-sm font-medium text-muted-foreground">
                Started
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-border">
            {evaluations.length === 0 && (
              <tr>
                <td colSpan={6} className="px-6 py-12 text-center">
                  <p className="text-muted-foreground mb-2">No evaluations yet</p>
                  <p className="text-sm text-muted-foreground">
                    Click "Quick Demo" to run a 5-task evaluation with real Ollama inference
                  </p>
                </td>
              </tr>
            )}
            {evaluations.map((evaluation) => (
              <tr key={evaluation.id} className="hover:bg-muted/30 transition-colors">
                <td className="px-6 py-4">
                  <div className="flex items-center space-x-2">
                    {getStatusIcon(evaluation.status)}
                    <span
                      className={`text-xs px-2 py-1 rounded-full ${getStatusBadge(
                        evaluation.status
                      )}`}
                    >
                      {evaluation.status}
                    </span>
                  </div>
                </td>
                <td className="px-6 py-4">
                  <span className="font-medium text-foreground">{evaluation.name}</span>
                </td>
                <td className="px-6 py-4">
                  <div className="flex items-center space-x-3">
                    <div className="flex-1 h-2 bg-muted rounded-full overflow-hidden max-w-[120px]">
                      <div
                        className="h-full bg-primary transition-all duration-300"
                        style={{ width: `${evaluation.progress}%` }}
                      />
                    </div>
                    <span className="text-sm text-muted-foreground">
                      {evaluation.completed_tasks}/{evaluation.total_tasks}
                    </span>
                  </div>
                </td>
                <td className="px-6 py-4">
                  <span className="text-foreground">
                    {evaluation.success_rate !== null
                      ? `${(evaluation.success_rate * 100).toFixed(1)}%`
                      : '-'}
                  </span>
                </td>
                <td className="px-6 py-4">
                  <span className="text-foreground">
                    {evaluation.cnsr !== null ? evaluation.cnsr.toFixed(1) : '-'}
                  </span>
                </td>
                <td className="px-6 py-4">
                  <span className="text-muted-foreground">
                    {new Date(evaluation.started_at).toLocaleString()}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Evaluation Info */}
      <div className="bg-card rounded-xl border border-border p-6">
        <h3 className="text-lg font-semibold mb-4 text-foreground">
          Long-Horizon Evaluation Framework
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="space-y-2">
            <h4 className="font-medium text-foreground">Checkpointing</h4>
            <p className="text-sm text-muted-foreground">
              Automatic state saves during long evaluations for resume capability
            </p>
          </div>
          <div className="space-y-2">
            <h4 className="font-medium text-foreground">Rolling Window</h4>
            <p className="text-sm text-muted-foreground">
              Configurable window size for tracking recent performance trends
            </p>
          </div>
          <div className="space-y-2">
            <h4 className="font-medium text-foreground">Real-time Metrics</h4>
            <p className="text-sm text-muted-foreground">
              Live CNSR, success rate, and cost tracking during evaluation
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}
