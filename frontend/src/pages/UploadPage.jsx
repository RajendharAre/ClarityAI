import React, { useCallback, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { analyzeImage } from '../services/api'
import ResultView from '../components/ResultView'

const ACCEPTED = ['image/jpeg', 'image/png', 'image/webp']

export default function UploadPage() {
  const [image, setImage] = useState(null)
  const [previewUrl, setPreviewUrl] = useState(null)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [dragging, setDragging] = useState(false)
  const inputRef = useRef(null)
  const navigate = useNavigate()

  const handleFile = useCallback(
    (file) => {
      if (!file) return
      setError(null)
      setResult(null)
      if (!ACCEPTED.includes(file.type)) {
        setError(`Unsupported file type: ${file.type || 'unknown'}. Use JPEG, PNG or WebP.`)
        return
      }
      setImage(file)
      setPreviewUrl(URL.createObjectURL(file))
    },
    [],
  )

  const onDrop = useCallback(
    (e) => {
      e.preventDefault()
      setDragging(false)
      handleFile(e.dataTransfer.files?.[0])
    },
    [handleFile],
  )

  const runAnalyze = async () => {
    if (!image) return
    setLoading(true)
    setError(null)
    try {
      const data = await analyzeImage(image)
      setResult(data)
    } catch (err) {
      setError(err?.response?.data?.detail || err.message || 'Analysis failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <section className="page">
      <div
        className={`dropzone ${dragging ? 'dragging' : ''}`}
        onClick={() => inputRef.current?.click()}
        onDragOver={(e) => {
          e.preventDefault()
          setDragging(true)
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
      >
        <input
          ref={inputRef}
          type="file"
          accept="image/jpeg,image/png,image/webp"
          hidden
          onChange={(e) => handleFile(e.target.files?.[0])}
        />
        {previewUrl ? (
          <img src={previewUrl} alt="preview" className="preview-img" />
        ) : (
          <div className="dropzone-hint">
            <p className="dropzone-title">Drop an image here or click to browse</p>
            <p className="dropzone-sub">JPEG · PNG · WebP</p>
          </div>
        )}
      </div>

      {error && <div className="alert alert-error">{error}</div>}

      <div className="upload-actions">
        <button className="btn btn-ghost" onClick={() => inputRef.current?.click()}>
          {previewUrl ? 'Choose another' : 'Browse files'}
        </button>
        {image && (
          <button className="btn btn-primary" onClick={runAnalyze} disabled={loading}>
            {loading ? 'Analyzing…' : 'Analyze image'}
          </button>
        )}
        {result && (
          <button className="btn btn-ghost" onClick={() => navigate('/history')}>
            View history
          </button>
        )}
      </div>

      {loading && <div className="spinner" />}

      {result && <ResultView result={result} />}
    </section>
  )
}
