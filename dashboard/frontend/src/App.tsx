import { Routes, Route } from 'react-router-dom'
import { useEffect } from 'react'
import Layout from './components/layout/Layout'
import Dashboard from './pages/Dashboard'
import Evaluations from './pages/Evaluations'
import CostAnalysis from './pages/CostAnalysis'
import Incidents from './pages/Incidents'
import Safety from './pages/Safety'
import Settings from './pages/Settings'
import { useThemeStore } from './store/themeStore'

function App() {
  const { theme } = useThemeStore()

  useEffect(() => {
    document.documentElement.classList.toggle('dark', theme === 'dark')
  }, [theme])

  return (
    <Layout>
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/evaluations" element={<Evaluations />} />
        <Route path="/costs" element={<CostAnalysis />} />
        <Route path="/incidents" element={<Incidents />} />
        <Route path="/safety" element={<Safety />} />
        <Route path="/settings" element={<Settings />} />
      </Routes>
    </Layout>
  )
}

export default App
