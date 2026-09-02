import React, { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { fetchHistory } from '../services/api'

const PAGE_SIZE = 10

const LABEL_META = {
  ACCEPTABLE: 'label-good',
  DEGRADED: 'label-warn',
  DEFECTIVE: 'label-bad',
}

export default function HistoryPage() {
  const [rows, setRows] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [page, setPage] = useState(1)

  useEffect(() => {
    let active = true
    fetchHistory()
      .then((data) => active && setRows(data))
      .catch((err) => active && setError(err.message))
      .finally(() => active && setLoading(false))
    return () => {
      active = false
    }
  }, [])

  const totalPages = Math.max(1, Math.ceil(rows.length / PAGE_SIZE))
  const current = rows.slice((page - 1) * PAGE_SIZE, page * PAGE_SIZE)

  if (loading) return <div className="spinner" />
  if (error) return <div className="alert alert-error">Failed to load history: {error}</div>

  return (
    <section className="page">
      <h1 className="page-title">Analysis history</h1>
      {rows.length === 0 ? (
        <p className="empty">No analyses yet. Upload an image to get started.</p>
      ) : (
        <>
          <table className="table">
            <thead>
              <tr>
                <th>#</th>
                <th>Filename</th>
                <th>Label</th>
                <th>Score</th>
                <th>Usable</th>
                <th>Created</th>
              </tr>
            </thead>
            <tbody>
              {current.map((r) => (
                <tr key={r.analysis_id}>
                  <td>{r.analysis_id}</td>
                  <td>
                    <Link to={`/history/${r.analysis_id}`} className="link">
                      {r.filename}
                    </Link>
                  </td>
                  <td>
                    <span className={`badge ${LABEL_META[r.quality_label] || 'label-warn'}`}>
                      {r.quality_label}
                    </span>
                  </td>
                  <td>{r.quality_score?.toFixed?.(1) ?? r.quality_score}</td>
                  <td>{r.usable ? 'Yes' : 'No'}</td>
                  <td>{new Date(r.created_at).toLocaleString()}</td>
                </tr>
              ))}
            </tbody>
          </table>

          {totalPages > 1 && (
            <div className="pagination">
              <button
                className="btn btn-ghost"
                disabled={page <= 1}
                onClick={() => setPage((p) => p - 1)}
              >
                Prev
              </button>
              <span className="page-info">
                {page} / {totalPages}
              </span>
              <button
                className="btn btn-ghost"
                disabled={page >= totalPages}
                onClick={() => setPage((p) => p + 1)}
              >
                Next
              </button>
            </div>
          )}
        </>
      )}
    </section>
  )
}
