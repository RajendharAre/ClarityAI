import React from 'react'
import ScoreGauge from './ScoreGauge'
import IssueCard from './IssueCard'

const LABEL_META = {
  ACCEPTABLE: { cls: 'label-good', label: 'Acceptable' },
  DEGRADED: { cls: 'label-warn', label: 'Degraded' },
  DEFECTIVE: { cls: 'label-bad', label: 'Defective' },
}

export default function ResultView({ result }) {
  const meta = LABEL_META[result.quality_label] || LABEL_META.DEGRADED
  const issues = result.issues || []

  return (
    <div className="result">
      <div className="result-head">
        <ScoreGauge score={result.quality_score} />
        <div className="result-summary">
          <span className={`badge ${meta.cls}`}>{meta.label}</span>
          <span className={`badge ${result.usable ? 'badge-usable' : 'badge-unusable'}`}>
            {result.usable ? 'Usable' : 'Not usable'}
          </span>
          <p className="confidence">Confidence: {(result.confidence * 100).toFixed(0)}%</p>
          {result.filename && <p className="filename">{result.filename}</p>}
        </div>
      </div>

      {result.reasoning && <p className="reasoning">{result.reasoning}</p>}

      {issues.length > 0 && (
        <div className="issues-grid">
          {issues.map((issue, i) => (
            <IssueCard key={i} issue={issue} />
          ))}
        </div>
      )}

      <div className="detail-meta">
        {typeof result.anomaly_score === 'number' && (
          <p>Anomaly score: {result.anomaly_score.toFixed(4)}</p>
        )}
        {result.created_at && <p>Created: {new Date(result.created_at).toLocaleString()}</p>}
        <p>Analysis ID: {result.analysis_id}</p>
      </div>
    </div>
  )
}
