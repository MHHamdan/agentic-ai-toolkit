const API_BASE = '/api/v1'

export interface CostBreakdown {
  inference_cost: number
  tool_cost: number
  latency_cost: number
  human_cost: number
  total_cost: number
}

export interface CNSRMetrics {
  cnsr: number
  success_rate: number
  mean_cost: number
  total_tasks: number
  total_successes: number
  cost_breakdown: CostBreakdown
}

export interface RollingMetrics {
  window_size: number
  current_window: number
  success_rates: number[]
  costs: number[]
  timestamps: string[]
}

export interface Evaluation {
  id: string
  name: string
  status: 'pending' | 'running' | 'completed' | 'failed'
  progress: number
  total_tasks: number
  completed_tasks: number
  success_rate: number | null
  cnsr: number | null
  started_at: string
  completed_at: string | null
}

// Backend response format
interface EvaluationResponse {
  evaluation_id: string
  name: string
  status: string
  progress: number
  tasks_total: number
  tasks_completed: number
  current_success_rate: number
  current_cnsr: number
  current_cost: number
  started_at: string
  model?: string
}

export interface EvaluationConfig {
  name: string
  model: string
  num_tasks: number
}

export interface Incident {
  id: string
  type: string
  severity: 'low' | 'medium' | 'high' | 'critical'
  description: string
  task_id: string | null
  timestamp: string
  resolved: boolean
  resolved_at: string | null
}

// Backend response format (different field names)
interface IncidentResponse {
  incident_id: string
  incident_type: string
  severity: string
  description: string
  context: { task_id?: string }
  created_at: string
  resolved: boolean
  resolved_at: string | null
}

export interface SafetyStatus {
  overall_compliant: boolean
  autonomy_level: number
  requirements: {
    name: string
    status: 'compliant' | 'partial' | 'non_compliant'
    details: string
  }[]
  pending_approvals: number
}

export interface OllamaStatus {
  connected: boolean
  models: string[]
  error: string | null
}

class ApiClient {
  private async fetch<T>(endpoint: string, options?: RequestInit): Promise<T> {
    const response = await fetch(`${API_BASE}${endpoint}`, {
      headers: {
        'Content-Type': 'application/json',
      },
      ...options,
    })

    if (!response.ok) {
      throw new Error(`API error: ${response.status}`)
    }

    return response.json()
  }

  // Health
  async getHealth(): Promise<{ status: string; ollama: OllamaStatus }> {
    return this.fetch('/health')
  }

  // Metrics
  async getCNSR(): Promise<CNSRMetrics> {
    return this.fetch('/metrics/cnsr')
  }

  async getRollingMetrics(windowSize = 10): Promise<RollingMetrics> {
    return this.fetch(`/metrics/rolling?window_size=${windowSize}`)
  }

  // Helper to transform evaluation response
  private transformEvaluation(item: EvaluationResponse): Evaluation {
    return {
      id: item.evaluation_id,
      name: item.name,
      status: item.status as Evaluation['status'],
      progress: item.progress * 100, // Convert 0-1 to 0-100
      total_tasks: item.tasks_total,
      completed_tasks: item.tasks_completed,
      success_rate: item.current_success_rate,
      cnsr: item.current_cnsr,
      started_at: item.started_at,
      completed_at: null,
    }
  }

  // Evaluations
  async getEvaluations(): Promise<Evaluation[]> {
    const data = await this.fetch<EvaluationResponse[]>('/evaluations')
    return data.map((item) => this.transformEvaluation(item))
  }

  async getEvaluation(id: string): Promise<Evaluation> {
    const data = await this.fetch<EvaluationResponse>(`/evaluations/${id}`)
    return this.transformEvaluation(data)
  }

  async startEvaluation(config: EvaluationConfig): Promise<Evaluation> {
    const data = await this.fetch<EvaluationResponse>('/evaluations', {
      method: 'POST',
      body: JSON.stringify(config),
    })
    return this.transformEvaluation(data)
  }

  // Run quick demo evaluation
  async runDemoEvaluation(model: string = 'gemma2:2b'): Promise<unknown> {
    return this.fetch(`/evaluations/demo?model=${model}`, {
      method: 'POST',
    })
  }

  // Incidents
  async getIncidents(): Promise<Incident[]> {
    const data = await this.fetch<IncidentResponse[]>('/incidents')
    return data.map((item) => ({
      id: item.incident_id,
      type: item.incident_type,
      severity: item.severity as Incident['severity'],
      description: item.description,
      task_id: item.context?.task_id || null,
      timestamp: item.created_at,
      resolved: item.resolved,
      resolved_at: item.resolved_at,
    }))
  }

  async resolveIncident(id: string): Promise<Incident> {
    return this.fetch(`/incidents/${id}/resolve`, {
      method: 'POST',
    })
  }

  // Safety
  async getSafetyStatus(): Promise<SafetyStatus> {
    return this.fetch('/safety/status')
  }

  // Costs
  async getCostBreakdown(): Promise<CostBreakdown> {
    return this.fetch('/costs/breakdown')
  }

  async getCostTrend(days = 7): Promise<{ dates: string[]; costs: number[] }> {
    return this.fetch(`/costs/trend?days=${days}`)
  }
}

export const api = new ApiClient()
