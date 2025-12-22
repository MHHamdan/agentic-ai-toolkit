import { create } from 'zustand'

interface CostBreakdown {
  inference_cost: number
  tool_cost: number
  latency_cost: number
  human_cost: number
  total_cost: number
}

interface MetricsState {
  cnsr: number
  successRate: number
  meanCost: number
  totalTasks: number
  costBreakdown: CostBreakdown
  rollingHistory: number[]
  lastUpdated: Date | null
  isLoading: boolean
  error: string | null

  setMetrics: (metrics: Partial<MetricsState>) => void
  setLoading: (loading: boolean) => void
  setError: (error: string | null) => void
}

export const useMetricsStore = create<MetricsState>((set) => ({
  cnsr: 0,
  successRate: 0,
  meanCost: 0,
  totalTasks: 0,
  costBreakdown: {
    inference_cost: 0,
    tool_cost: 0,
    latency_cost: 0,
    human_cost: 0,
    total_cost: 0,
  },
  rollingHistory: [],
  lastUpdated: null,
  isLoading: false,
  error: null,

  setMetrics: (metrics) => set((state) => ({ ...state, ...metrics, lastUpdated: new Date() })),
  setLoading: (isLoading) => set({ isLoading }),
  setError: (error) => set({ error }),
}))
