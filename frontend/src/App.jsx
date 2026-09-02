import React from 'react'
import { Routes, Route, NavLink } from 'react-router-dom'
import UploadPage from './pages/UploadPage'
import HistoryPage from './pages/HistoryPage'
import HistoryDetailPage from './pages/HistoryDetailPage'

export default function App() {
  return (
    <div className="app">
      <header className="app-header">
        <div className="app-header-inner">
          <NavLink to="/" className="brand">
            <span className="brand-dot" />
            Clarity<span>AI</span>
          </NavLink>
          <nav className="app-nav">
            <NavLink to="/" end className={({ isActive }) => (isActive ? 'nav-link active' : 'nav-link')}>
              Analyze
            </NavLink>
            <NavLink to="/history" className={({ isActive }) => (isActive ? 'nav-link active' : 'nav-link')}>
              History
            </NavLink>
          </nav>
        </div>
      </header>

      <main className="app-main">
        <Routes>
          <Route path="/" element={<UploadPage />} />
          <Route path="/history" element={<HistoryPage />} />
          <Route path="/history/:id" element={<HistoryDetailPage />} />
        </Routes>
      </main>

      <footer className="app-footer">
        <span>ClarityAI — reliable binary usable / not-usable quality gate</span>
      </footer>
    </div>
  )
}
