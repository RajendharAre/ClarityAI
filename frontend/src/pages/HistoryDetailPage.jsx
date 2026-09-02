import React, { useEffect, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { fetchAnalysis } from '../services/api'
import ResultView from '../components/ResultView'

export default function HistoryDetailPage() {
  const { id } = useParams()
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    let active = true
    fetchAnalysis(id)
      .then((d) => active && setData(d))
      .catch((err) => active && setError(err.message))
      .finally(() => active && setLoading(false))
    return () => {
      active = false
    }
  }, [id])

  if (loading) return <div className="spinner" />
  if (error) return <div className="alert alert-error">Failed to load analysis: {error}</div>

  const result = data.result || data

  return (
    <section className="page">
      <Link to="/history" className="link back-link">
        ← Back to history
      </Link>
      <h1 className="page-title">Analysis #{data.analysis_id}</h1>
      <ResultView result={{ ...result, analysis_id: data.analysis_id, created_at: data.created_at }} />
    </section>
  )
}
