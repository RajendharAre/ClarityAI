import React from 'react'

const TYPE_LABEL = {
  blur: 'Blur',
  exposure: 'Exposure',
  noise: 'Noise',
  jpeg: 'JPEG blocking',
  defect: 'Defect / anomaly',
}

function severityClass(sev) {
  if (sev < 0.3) return 'sev-low'
  if (sev < 0.6) return 'sev-mid'
  return 'sev-high'
}

export default function IssueCard({ issue }) {
  const name = TYPE_LABEL[issue.issue_type] || issue.issue_type
  const sevPct = Math.round((issue.severity || 0) * 100)
  return (
    <div className="issue-card">
      <div className="issue-head">
        <span className="issue-name">{name}</span>
        <span className={`issue-sev ${severityClass(issue.severity || 0)}`}>{sevPct}</span>
      </div>
      <div className="sev-bar">
        <div className="sev-fill" style={{ width: `${sevPct}%` }} />
      </div>
      <p className="issue-detail">{issue.detail || '—'}</p>
      {issue.present && <span className="tag tag-warn">present</span>}
    </div>
  )
}
