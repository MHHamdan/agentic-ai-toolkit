import { useState, useEffect } from 'react'
import { Shield, CheckCircle, AlertCircle, Clock, Users } from 'lucide-react'
import { api, SafetyStatus } from '../api/client'

// 5 Safety Requirements
const SAFETY_REQUIREMENTS = [
  {
    id: 'sr1',
    name: 'Human Override Capability',
    description: 'System must allow human operators to override or halt agent actions at any time',
    icon: Users,
  },
  {
    id: 'sr2',
    name: 'Action Transparency',
    description: 'All agent actions must be logged and explainable to human reviewers',
    icon: Shield,
  },
  {
    id: 'sr3',
    name: 'Bounded Resource Usage',
    description: 'Agent resource consumption must stay within defined limits',
    icon: AlertCircle,
  },
  {
    id: 'sr4',
    name: 'Goal Alignment Verification',
    description: 'Continuous verification that agent actions align with original objectives',
    icon: CheckCircle,
  },
  {
    id: 'sr5',
    name: 'Fail-Safe Mechanisms',
    description: 'Automatic safety responses when anomalies or failures are detected',
    icon: Shield,
  },
]

// 5 Autonomy Levels with 4 Criteria
const AUTONOMY_LEVELS = [
  {
    level: 1,
    name: 'Human-in-the-Loop',
    description: 'Human approval required for every action',
    criteria: { asf: 'None', gdp: 'Human-defined', dt: 'Pre-action', er: 'Human-led' },
  },
  {
    level: 2,
    name: 'Human-on-the-Loop',
    description: 'Human monitors and can intervene',
    criteria: { asf: 'Limited', gdp: 'Human-guided', dt: 'Real-time', er: 'Assisted' },
  },
  {
    level: 3,
    name: 'Human-out-of-Loop',
    description: 'Periodic human review',
    criteria: { asf: 'Moderate', gdp: 'Shared', dt: 'Post-action', er: 'Semi-auto' },
  },
  {
    level: 4,
    name: 'Bounded Autonomy',
    description: 'Agent operates within defined constraints',
    criteria: { asf: 'High', gdp: 'Agent-proposed', dt: 'Batch', er: 'Automatic' },
  },
  {
    level: 5,
    name: 'Full Autonomy',
    description: 'Complete agent independence',
    criteria: { asf: 'Full', gdp: 'Agent-defined', dt: 'Independent', er: 'Self-healing' },
  },
]

export default function Safety() {
  const [status, setStatus] = useState<SafetyStatus | null>(null)
  const [loading, setLoading] = useState(true)
  const [currentLevel, setCurrentLevel] = useState(2) // Default to Human-on-the-Loop

  useEffect(() => {
    fetchStatus()
  }, [])

  async function fetchStatus() {
    try {
      const data = await api.getSafetyStatus()
      setStatus(data)
      setCurrentLevel(data.autonomy_level)
    } catch {
      // Demo data
      setStatus({
        overall_compliant: true,
        autonomy_level: 2,
        requirements: [
          { name: 'Human Override Capability', status: 'compliant', details: 'All controls active' },
          { name: 'Action Transparency', status: 'compliant', details: 'Full logging enabled' },
          { name: 'Bounded Resource Usage', status: 'compliant', details: 'Within limits' },
          { name: 'Goal Alignment Verification', status: 'partial', details: 'Monitoring active' },
          { name: 'Fail-Safe Mechanisms', status: 'compliant', details: 'All safeguards operational' },
        ],
        pending_approvals: 3,
      })
    } finally {
      setLoading(false)
    }
  }

  const getStatusBadge = (reqStatus: 'compliant' | 'partial' | 'non_compliant') => {
    switch (reqStatus) {
      case 'compliant':
        return (
          <span className="flex items-center text-green-500 bg-green-500/10 px-2 py-1 rounded-full text-xs">
            <CheckCircle className="w-3 h-3 mr-1" /> Compliant
          </span>
        )
      case 'partial':
        return (
          <span className="flex items-center text-yellow-500 bg-yellow-500/10 px-2 py-1 rounded-full text-xs">
            <AlertCircle className="w-3 h-3 mr-1" /> Partial
          </span>
        )
      default:
        return (
          <span className="flex items-center text-red-500 bg-red-500/10 px-2 py-1 rounded-full text-xs">
            <AlertCircle className="w-3 h-3 mr-1" /> Non-Compliant
          </span>
        )
    }
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary"></div>
      </div>
    )
  }

  const compliantCount = status?.requirements.filter((r) => r.status === 'compliant').length ?? 0

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-2xl font-bold text-foreground">Safety Compliance</h1>
          <p className="text-muted-foreground">
            5 Safety Requirements & 5 Autonomy Levels with 4 Criteria
          </p>
        </div>
        <div className="flex items-center space-x-4">
          <div
            className={`text-center px-6 py-3 rounded-lg ${
              status?.overall_compliant
                ? 'bg-green-500/10 border border-green-500/30'
                : 'bg-red-500/10 border border-red-500/30'
            }`}
          >
            <p
              className={`text-2xl font-bold ${
                status?.overall_compliant ? 'text-green-500' : 'text-red-500'
              }`}
            >
              {compliantCount}/5
            </p>
            <p className="text-xs text-muted-foreground">Requirements Met</p>
          </div>
        </div>
      </div>

      {/* Overall Status Banner */}
      <div
        className={`rounded-xl p-6 ${
          status?.overall_compliant
            ? 'bg-green-500/10 border border-green-500/30'
            : 'bg-yellow-500/10 border border-yellow-500/30'
        }`}
      >
        <div className="flex items-center space-x-4">
          <Shield
            className={`w-12 h-12 ${status?.overall_compliant ? 'text-green-500' : 'text-yellow-500'}`}
          />
          <div>
            <h2
              className={`text-xl font-bold ${
                status?.overall_compliant ? 'text-green-500' : 'text-yellow-500'
              }`}
            >
              {status?.overall_compliant ? 'System Compliant' : 'Attention Required'}
            </h2>
            <p className="text-muted-foreground">
              {status?.overall_compliant
                ? 'All critical safety requirements are being met'
                : 'Some safety requirements need attention'}
            </p>
          </div>
          {(status?.pending_approvals ?? 0) > 0 && (
            <div className="ml-auto text-right">
              <p className="text-2xl font-bold text-foreground">{status?.pending_approvals}</p>
              <p className="text-xs text-muted-foreground">Pending Approvals</p>
            </div>
          )}
        </div>
      </div>

      {/* 5 Safety Requirements */}
      <div className="bg-card rounded-xl border border-border p-6">
        <h3 className="text-lg font-semibold mb-4 text-foreground">5 Safety Requirements</h3>
        <div className="space-y-4">
          {SAFETY_REQUIREMENTS.map((req, index) => {
            const reqStatus = status?.requirements[index]
            const Icon = req.icon
            return (
              <div
                key={req.id}
                className="flex items-center justify-between p-4 bg-muted/30 rounded-lg border border-border"
              >
                <div className="flex items-center space-x-4">
                  <div className="p-2 bg-primary/10 rounded-lg">
                    <Icon className="w-5 h-5 text-primary" />
                  </div>
                  <div>
                    <h4 className="font-medium text-foreground">{req.name}</h4>
                    <p className="text-sm text-muted-foreground">{req.description}</p>
                  </div>
                </div>
                <div className="flex items-center space-x-4">
                  <span className="text-sm text-muted-foreground hidden md:block">
                    {reqStatus?.details}
                  </span>
                  {reqStatus && getStatusBadge(reqStatus.status)}
                </div>
              </div>
            )
          })}
        </div>
      </div>

      {/* 5 Autonomy Levels */}
      <div className="bg-card rounded-xl border border-border p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-foreground">5 Autonomy Levels</h3>
          <span className="text-sm text-muted-foreground">
            Current Level: <strong className="text-primary">{currentLevel}</strong>
          </span>
        </div>

        {/* Level Selector */}
        <div className="flex items-center justify-between mb-6 p-4 bg-muted/30 rounded-lg">
          {AUTONOMY_LEVELS.map((level) => (
            <button
              key={level.level}
              onClick={() => setCurrentLevel(level.level)}
              className={`flex-1 py-3 mx-1 rounded-lg transition-all ${
                currentLevel === level.level
                  ? 'bg-primary text-primary-foreground shadow-lg scale-105'
                  : currentLevel > level.level
                  ? 'bg-primary/20 text-primary'
                  : 'bg-muted text-muted-foreground hover:bg-muted/80'
              }`}
            >
              <p className="text-lg font-bold">{level.level}</p>
              <p className="text-xs hidden md:block">{level.name.split(' ')[0]}</p>
            </button>
          ))}
        </div>

        {/* Selected Level Details */}
        <div className="p-6 bg-muted/30 rounded-lg border border-primary/30">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h4 className="text-xl font-bold text-foreground">
                Level {currentLevel}: {AUTONOMY_LEVELS[currentLevel - 1].name}
              </h4>
              <p className="text-muted-foreground">{AUTONOMY_LEVELS[currentLevel - 1].description}</p>
            </div>
          </div>

          {/* 4 Classification Criteria */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-4">
            <div className="p-3 bg-background rounded-lg">
              <p className="text-xs text-muted-foreground mb-1">Action Scope Freedom (ASF)</p>
              <p className="font-medium text-foreground">
                {AUTONOMY_LEVELS[currentLevel - 1].criteria.asf}
              </p>
            </div>
            <div className="p-3 bg-background rounded-lg">
              <p className="text-xs text-muted-foreground mb-1">Goal Definition Power (GDP)</p>
              <p className="font-medium text-foreground">
                {AUTONOMY_LEVELS[currentLevel - 1].criteria.gdp}
              </p>
            </div>
            <div className="p-3 bg-background rounded-lg">
              <p className="text-xs text-muted-foreground mb-1">Decision Timing (DT)</p>
              <p className="font-medium text-foreground">
                {AUTONOMY_LEVELS[currentLevel - 1].criteria.dt}
              </p>
            </div>
            <div className="p-3 bg-background rounded-lg">
              <p className="text-xs text-muted-foreground mb-1">Error Recovery (ER)</p>
              <p className="font-medium text-foreground">
                {AUTONOMY_LEVELS[currentLevel - 1].criteria.er}
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Pending Approvals */}
      {(status?.pending_approvals ?? 0) > 0 && (
        <div className="bg-card rounded-xl border border-border p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-foreground">Pending Approvals</h3>
            <span className="bg-yellow-500/10 text-yellow-500 px-3 py-1 rounded-full text-sm">
              {status?.pending_approvals} awaiting review
            </span>
          </div>
          <div className="space-y-3">
            {[1, 2, 3].slice(0, status?.pending_approvals ?? 0).map((i) => (
              <div
                key={i}
                className="flex items-center justify-between p-4 bg-muted/30 rounded-lg"
              >
                <div className="flex items-center space-x-4">
                  <Clock className="w-5 h-5 text-yellow-500" />
                  <div>
                    <p className="font-medium text-foreground">
                      Action Request #{String(100 + i).padStart(4, '0')}
                    </p>
                    <p className="text-sm text-muted-foreground">
                      File system modification requested
                    </p>
                  </div>
                </div>
                <div className="flex items-center space-x-2">
                  <button className="px-3 py-1.5 text-sm bg-red-500/10 text-red-500 rounded-lg hover:bg-red-500/20">
                    Deny
                  </button>
                  <button className="px-3 py-1.5 text-sm bg-green-500/10 text-green-500 rounded-lg hover:bg-green-500/20">
                    Approve
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
