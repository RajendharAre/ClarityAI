import axios from 'axios'

const api = axios.create({
  baseURL: '/api/v1',
  timeout: 60000,
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
