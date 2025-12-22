import { useState } from 'react'
import { Save, RefreshCw, Server, DollarSign, Clock, Shield } from 'lucide-react'

interface SettingsState {
  ollama: {
    host: string
    port: number
    defaultModel: string
  }
  costs: {
    inputTokenRate: number
    outputTokenRate: number
    toolCallRate: number
    latencyRate: number
    humanReviewRate: number
  }
  evaluation: {
    defaultWindowSize: number
    checkpointInterval: number
    maxConcurrentTasks: number
  }
  safety: {
    defaultAutonomyLevel: number
    requireApprovalAboveLevel: number
    incidentAutoEscalation: boolean
  }
}

export default function Settings() {
  const [settings, setSettings] = useState<SettingsState>({
    ollama: {
      host: 'localhost',
      port: 11434,
      defaultModel: 'gemma2:2b',
    },
    costs: {
      inputTokenRate: 0.001,
      outputTokenRate: 0.002,
      toolCallRate: 0.0001,
      latencyRate: 0.001,
      humanReviewRate: 0.50,
    },
    evaluation: {
      defaultWindowSize: 10,
      checkpointInterval: 5,
      maxConcurrentTasks: 4,
    },
    safety: {
      defaultAutonomyLevel: 2,
      requireApprovalAboveLevel: 3,
      incidentAutoEscalation: true,
    },
  })

  const [saving, setSaving] = useState(false)
  const [saved, setSaved] = useState(false)

  const handleSave = async () => {
    setSaving(true)
    // Simulate API call
    await new Promise((resolve) => setTimeout(resolve, 1000))
    setSaving(false)
    setSaved(true)
    setTimeout(() => setSaved(false), 3000)
  }

  const updateSetting = <K extends keyof SettingsState>(
    category: K,
    key: keyof SettingsState[K],
    value: SettingsState[K][keyof SettingsState[K]]
  ) => {
    setSettings((prev) => ({
      ...prev,
      [category]: {
        ...prev[category],
        [key]: value,
      },
    }))
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-foreground">Settings</h1>
          <p className="text-muted-foreground">Configure toolkit parameters and connections</p>
        </div>
        <button
          onClick={handleSave}
          disabled={saving}
          className="flex items-center space-x-2 px-4 py-2 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 transition-colors disabled:opacity-50"
        >
          {saving ? (
            <RefreshCw className="w-4 h-4 animate-spin" />
          ) : (
            <Save className="w-4 h-4" />
          )}
          <span>{saving ? 'Saving...' : saved ? 'Saved!' : 'Save Changes'}</span>
        </button>
      </div>

      {saved && (
        <div className="bg-green-500/10 border border-green-500/30 rounded-lg p-4">
          <p className="text-green-500">Settings saved successfully!</p>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Ollama Connection */}
        <div className="bg-card rounded-xl border border-border p-6">
          <div className="flex items-center space-x-3 mb-6">
            <div className="p-2 bg-purple-500/10 rounded-lg">
              <Server className="w-5 h-5 text-purple-500" />
            </div>
            <div>
              <h3 className="text-lg font-semibold text-foreground">Ollama Connection</h3>
              <p className="text-sm text-muted-foreground">Local LLM server configuration</p>
            </div>
          </div>

          <div className="space-y-4">
            <div>
              <label className="block text-sm text-muted-foreground mb-2">Host</label>
              <input
                type="text"
                value={settings.ollama.host}
                onChange={(e) => updateSetting('ollama', 'host', e.target.value)}
                className="w-full px-3 py-2 bg-background border border-border rounded-lg text-foreground focus:outline-none focus:ring-2 focus:ring-primary"
              />
            </div>
            <div>
              <label className="block text-sm text-muted-foreground mb-2">Port</label>
              <input
                type="number"
                value={settings.ollama.port}
                onChange={(e) => updateSetting('ollama', 'port', parseInt(e.target.value))}
                className="w-full px-3 py-2 bg-background border border-border rounded-lg text-foreground focus:outline-none focus:ring-2 focus:ring-primary"
              />
            </div>
            <div>
              <label className="block text-sm text-muted-foreground mb-2">Default Model</label>
              <select
                value={settings.ollama.defaultModel}
                onChange={(e) => updateSetting('ollama', 'defaultModel', e.target.value)}
                className="w-full px-3 py-2 bg-background border border-border rounded-lg text-foreground focus:outline-none focus:ring-2 focus:ring-primary"
              >
                <option value="gemma2:2b">gemma2:2b</option>
                <option value="llama3.2:3b">llama3.2:3b</option>
                <option value="qwen2:1.5b">qwen2:1.5b</option>
                <option value="phi3:mini">phi3:mini</option>
              </select>
            </div>
          </div>
        </div>

        {/* Cost Rates */}
        <div className="bg-card rounded-xl border border-border p-6">
          <div className="flex items-center space-x-3 mb-6">
            <div className="p-2 bg-green-500/10 rounded-lg">
              <DollarSign className="w-5 h-5 text-green-500" />
            </div>
            <div>
              <h3 className="text-lg font-semibold text-foreground">Cost Rates</h3>
              <p className="text-sm text-muted-foreground">4-component cost model parameters</p>
            </div>
          </div>

          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm text-muted-foreground mb-2">
                  Input Token Rate ($/token)
                </label>
                <input
                  type="number"
                  step="0.0001"
                  value={settings.costs.inputTokenRate}
                  onChange={(e) =>
                    updateSetting('costs', 'inputTokenRate', parseFloat(e.target.value))
                  }
                  className="w-full px-3 py-2 bg-background border border-border rounded-lg text-foreground focus:outline-none focus:ring-2 focus:ring-primary"
                />
              </div>
              <div>
                <label className="block text-sm text-muted-foreground mb-2">
                  Output Token Rate ($/token)
                </label>
                <input
                  type="number"
                  step="0.0001"
                  value={settings.costs.outputTokenRate}
                  onChange={(e) =>
                    updateSetting('costs', 'outputTokenRate', parseFloat(e.target.value))
                  }
                  className="w-full px-3 py-2 bg-background border border-border rounded-lg text-foreground focus:outline-none focus:ring-2 focus:ring-primary"
                />
              </div>
            </div>
            <div>
              <label className="block text-sm text-muted-foreground mb-2">
                Tool Call Rate ($/call)
              </label>
              <input
                type="number"
                step="0.0001"
                value={settings.costs.toolCallRate}
                onChange={(e) =>
                  updateSetting('costs', 'toolCallRate', parseFloat(e.target.value))
                }
                className="w-full px-3 py-2 bg-background border border-border rounded-lg text-foreground focus:outline-none focus:ring-2 focus:ring-primary"
              />
            </div>
            <div>
              <label className="block text-sm text-muted-foreground mb-2">
                Latency Rate ($/second)
              </label>
              <input
                type="number"
                step="0.0001"
                value={settings.costs.latencyRate}
                onChange={(e) => updateSetting('costs', 'latencyRate', parseFloat(e.target.value))}
                className="w-full px-3 py-2 bg-background border border-border rounded-lg text-foreground focus:outline-none focus:ring-2 focus:ring-primary"
              />
            </div>
            <div>
              <label className="block text-sm text-muted-foreground mb-2">
                Human Review Rate ($/minute)
              </label>
              <input
                type="number"
                step="0.01"
                value={settings.costs.humanReviewRate}
                onChange={(e) =>
                  updateSetting('costs', 'humanReviewRate', parseFloat(e.target.value))
                }
                className="w-full px-3 py-2 bg-background border border-border rounded-lg text-foreground focus:outline-none focus:ring-2 focus:ring-primary"
              />
            </div>
          </div>
        </div>

        {/* Evaluation Parameters */}
        <div className="bg-card rounded-xl border border-border p-6">
          <div className="flex items-center space-x-3 mb-6">
            <div className="p-2 bg-blue-500/10 rounded-lg">
              <Clock className="w-5 h-5 text-blue-500" />
            </div>
            <div>
              <h3 className="text-lg font-semibold text-foreground">Evaluation Parameters</h3>
              <p className="text-sm text-muted-foreground">Long-horizon evaluation settings</p>
            </div>
          </div>

          <div className="space-y-4">
            <div>
              <label className="block text-sm text-muted-foreground mb-2">
                Rolling Window Size
              </label>
              <input
                type="number"
                value={settings.evaluation.defaultWindowSize}
                onChange={(e) =>
                  updateSetting('evaluation', 'defaultWindowSize', parseInt(e.target.value))
                }
                className="w-full px-3 py-2 bg-background border border-border rounded-lg text-foreground focus:outline-none focus:ring-2 focus:ring-primary"
              />
              <p className="text-xs text-muted-foreground mt-1">
                Number of recent tasks for rolling metrics
              </p>
            </div>
            <div>
              <label className="block text-sm text-muted-foreground mb-2">
                Checkpoint Interval
              </label>
              <input
                type="number"
                value={settings.evaluation.checkpointInterval}
                onChange={(e) =>
                  updateSetting('evaluation', 'checkpointInterval', parseInt(e.target.value))
                }
                className="w-full px-3 py-2 bg-background border border-border rounded-lg text-foreground focus:outline-none focus:ring-2 focus:ring-primary"
              />
              <p className="text-xs text-muted-foreground mt-1">
                Save checkpoint every N tasks
              </p>
            </div>
            <div>
              <label className="block text-sm text-muted-foreground mb-2">
                Max Concurrent Tasks
              </label>
              <input
                type="number"
                value={settings.evaluation.maxConcurrentTasks}
                onChange={(e) =>
                  updateSetting('evaluation', 'maxConcurrentTasks', parseInt(e.target.value))
                }
                className="w-full px-3 py-2 bg-background border border-border rounded-lg text-foreground focus:outline-none focus:ring-2 focus:ring-primary"
              />
            </div>
          </div>
        </div>

        {/* Safety Settings */}
        <div className="bg-card rounded-xl border border-border p-6">
          <div className="flex items-center space-x-3 mb-6">
            <div className="p-2 bg-red-500/10 rounded-lg">
              <Shield className="w-5 h-5 text-red-500" />
            </div>
            <div>
              <h3 className="text-lg font-semibold text-foreground">Safety Settings</h3>
              <p className="text-sm text-muted-foreground">Autonomy and oversight configuration</p>
            </div>
          </div>

          <div className="space-y-4">
            <div>
              <label className="block text-sm text-muted-foreground mb-2">
                Default Autonomy Level
              </label>
              <select
                value={settings.safety.defaultAutonomyLevel}
                onChange={(e) =>
                  updateSetting('safety', 'defaultAutonomyLevel', parseInt(e.target.value))
                }
                className="w-full px-3 py-2 bg-background border border-border rounded-lg text-foreground focus:outline-none focus:ring-2 focus:ring-primary"
              >
                <option value={1}>Level 1 - Human-in-the-Loop</option>
                <option value={2}>Level 2 - Human-on-the-Loop</option>
                <option value={3}>Level 3 - Human-out-of-Loop</option>
                <option value={4}>Level 4 - Bounded Autonomy</option>
                <option value={5}>Level 5 - Full Autonomy</option>
              </select>
            </div>
            <div>
              <label className="block text-sm text-muted-foreground mb-2">
                Require Approval Above Level
              </label>
              <select
                value={settings.safety.requireApprovalAboveLevel}
                onChange={(e) =>
                  updateSetting('safety', 'requireApprovalAboveLevel', parseInt(e.target.value))
                }
                className="w-full px-3 py-2 bg-background border border-border rounded-lg text-foreground focus:outline-none focus:ring-2 focus:ring-primary"
              >
                <option value={2}>Level 2+</option>
                <option value={3}>Level 3+</option>
                <option value={4}>Level 4+</option>
                <option value={5}>Level 5 only</option>
              </select>
              <p className="text-xs text-muted-foreground mt-1">
                Human approval required for actions above this level
              </p>
            </div>
            <div className="flex items-center justify-between p-3 bg-muted/30 rounded-lg">
              <div>
                <p className="font-medium text-foreground">Auto-Escalation</p>
                <p className="text-sm text-muted-foreground">
                  Automatically escalate critical incidents
                </p>
              </div>
              <button
                onClick={() =>
                  updateSetting(
                    'safety',
                    'incidentAutoEscalation',
                    !settings.safety.incidentAutoEscalation
                  )
                }
                className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                  settings.safety.incidentAutoEscalation ? 'bg-primary' : 'bg-muted'
                }`}
              >
                <span
                  className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                    settings.safety.incidentAutoEscalation ? 'translate-x-6' : 'translate-x-1'
                  }`}
                />
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* API Info */}
      <div className="bg-card rounded-xl border border-border p-6">
        <h3 className="text-lg font-semibold mb-4 text-foreground">API Information</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div>
            <p className="text-sm text-muted-foreground">API Base URL</p>
            <p className="font-mono text-foreground bg-muted/50 px-3 py-2 rounded mt-1">
              http://localhost:8000/api/v1
            </p>
          </div>
          <div>
            <p className="text-sm text-muted-foreground">WebSocket URL</p>
            <p className="font-mono text-foreground bg-muted/50 px-3 py-2 rounded mt-1">
              ws://localhost:8000/ws/realtime
            </p>
          </div>
          <div>
            <p className="text-sm text-muted-foreground">API Documentation</p>
            <a
              href="/api/docs"
              target="_blank"
              className="block font-mono text-primary bg-muted/50 px-3 py-2 rounded mt-1 hover:underline"
            >
              /api/docs (Swagger UI)
            </a>
          </div>
        </div>
      </div>
    </div>
  )
}
