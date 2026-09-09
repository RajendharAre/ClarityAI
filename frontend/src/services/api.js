import axios from 'axios'

const BACKEND_URL = import.meta.env.VITE_API_BASE_URL || '/api/v1'

const api = axios.create({
  baseURL: BACKEND_URL,
  timeout: 300000,
})

export async function analyzeImage(file) {
  const form = new FormData()
  form.append('file', file)
  const { data } = await api.post('/analyze', form)
  return data
}

export async function fetchHistory() {
  const { data } = await api.get('/history')
  return data
}

export async function fetchAnalysis(id) {
  const { data } = await api.get(`/history/${id}`)
  return data
}

export default api
