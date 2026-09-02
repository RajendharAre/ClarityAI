import React from 'react'

export default function ScoreGauge({ score }) {
  const clamped = Math.max(0, Math.min(100, score))
  const hue = clamped >= 80 ? 130 : clamped >= 50 ? 40 : 0

  return (
    <div className="gauge" style={{ '--score': `${clamped * 3.6}deg`, '--hue': hue }}>
      <div className="gauge-ring">
        <div className="gauge-center">
          <span className="gauge-value">{clamped.toFixed(0)}</span>
          <span className="gauge-unit">/ 100</span>
        </div>
      </div>
    </div>
  )
}
