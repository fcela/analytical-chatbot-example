import React, { useState, useRef, useEffect, useCallback } from 'react'
import { A2UIProvider, A2UIRenderer } from '@a2ui-sdk/react/0.8'
import type { A2UIMessage, A2UIAction } from '@a2ui-sdk/react/0.8'
import { customCatalog } from './catalog'

const API_BASE = (() => {
  if (import.meta.env.DEV) return ''  // Vite proxy handles routing
  return window.location.origin
})()

interface FileInfo {
  filename: string
  rows: number
  columns: number
}

interface TableInfo {
  columns: Record<string, string>
  row_count: number
}

interface DatabaseInfo {
  available: boolean
  tables: Record<string, TableInfo>
}

function ChatApp() {
  const [files, setFiles] = useState<FileInfo[]>([])
  const [database, setDatabase] = useState<DatabaseInfo | null>(null)
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [contextId, setContextId] = useState<string>('')
  const [chatHistory, setChatHistory] = useState<Array<{ role: string; content: string }>>([])
  const [a2uiMessages, setA2uiMessages] = useState<A2UIMessage[]>([])
  const messagesEndRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    fetchDatabase()
  }, [])

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [chatHistory, a2uiMessages])

  async function fetchDatabase() {
    try {
      const res = await fetch(`${API_BASE}/database`)
      const data = await res.json()
      setDatabase(data)
    } catch (e) {
      console.error('Failed to fetch database info:', e)
    }
  }

  async function handleUpload(e: React.ChangeEvent<HTMLInputElement>) {
    if (!e.target.files?.length) return
    const file = e.target.files[0]
    const form = new FormData()
    form.append('file', file)

    try {
      const res = await fetch(`${API_BASE}/upload`, {
        method: 'POST',
        body: form,
        headers: contextId ? { 'X-Context-Id': contextId } : {},
      })
      if (res.ok) {
        const data = await res.json()
        setFiles(f => [...f, { filename: data.filename, rows: data.rows, columns: data.columns }])
        if (data.contextId) setContextId(data.contextId)
      } else {
        const err = await res.json()
        alert('Upload error: ' + (err.detail || res.statusText))
      }
    } catch (err) {
      alert('Upload failed: ' + err)
    }
    e.target.value = ''
  }

  const sendMessage = useCallback(async () => {
    if (!input.trim() || loading) return

    const userText = input
    setChatHistory(h => [...h, { role: 'user', content: userText }])
    setInput('')
    setLoading(true)

    try {
      const response = await fetch(`${API_BASE}/a2a/message/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: {
            messageId: crypto.randomUUID(),
            role: 'user',
            parts: [{ text: userText }],
            contextId: contextId || undefined,
          },
        }),
      })

      // Read context ID from response header
      const respContextId = response.headers.get('X-Context-Id')
      if (respContextId && !contextId) {
        setContextId(respContextId)
      }

      // Process SSE stream
      const reader = response.body?.getReader()
      const decoder = new TextDecoder()

      if (reader) {
        let buffer = ''
        while (true) {
          const { done, value } = await reader.read()
          if (done) break

          buffer += decoder.decode(value, { stream: true })
          const lines = buffer.split('\n')
          buffer = lines.pop() || ''

          for (const line of lines) {
            if (line.startsWith('data: ')) {
              try {
                const eventData = JSON.parse(line.slice(6))

                // Check for A2UI DataParts
                if (eventData.parts) {
                  for (const part of eventData.parts) {
                    const data = part?.root?.data
                    const mimeType = part?.root?.metadata?.mimeType
                    if (data && mimeType === 'application/json+a2ui') {
                      setA2uiMessages(prev => [...prev, data as A2UIMessage])
                    }
                  }
                }

                // Check for completion
                if (eventData.status?.state === 'completed') {
                  if (eventData.contextId) setContextId(eventData.contextId)
                }
              } catch {
                // Skip malformed lines
              }
            }
          }
        }
      }

      setChatHistory(h => [...h, { role: 'assistant', content: '(rendered via A2UI)' }])
      fetchDatabase()
    } catch (e) {
      console.error('Message send error:', e)
      setChatHistory(h => [...h, { role: 'assistant', content: `Error: ${e}` }])
    } finally {
      setLoading(false)
    }
  }, [input, loading, contextId])

  async function clearSession() {
    await fetch(`${API_BASE}/clear`, {
      method: 'POST',
      headers: contextId ? { 'X-Context-Id': contextId } : {},
    })
    setFiles([])
    setChatHistory([])
    setA2uiMessages([])
    setContextId('')
  }

  function handleAction(action: A2UIAction) {
    console.log('A2UI action:', action)
  }

  return (
    <div className="app">
      <header>
        <h1>Analytical Chatbot</h1>
        <p className="subtitle">Upload data, ask questions, get insights</p>
      </header>

      <section className="files-section">
        <div className="files-header">
          <h2>Data Files</h2>
          <label className="upload-btn">
            + Upload
            <input type="file" accept=".csv,.json" onChange={handleUpload} hidden />
          </label>
        </div>
        {files.length === 0 ? (
          <p className="no-files">No files uploaded. Upload a CSV or JSON file to analyze.</p>
        ) : (
          <div className="file-chips">
            {files.map(f => (
              <div className="file-chip" key={f.filename}>
                <span>{f.filename}</span>
                <span className="file-info">{f.rows} rows x {f.columns} cols</span>
              </div>
            ))}
          </div>
        )}
      </section>

      {database?.available && (
        <section className="database-section">
          <div className="database-header">
            <h2>Database Tables</h2>
          </div>
          <div className="table-chips">
            {Object.entries(database.tables).map(([name, info]) => (
              <div className="table-chip" key={name}>
                <span className="table-name">{name}</span>
                <span className="table-info">{info.row_count} rows</span>
              </div>
            ))}
          </div>
        </section>
      )}

      <section className="chat-section">
        <div className="messages">
          {chatHistory.length === 0 && (
            <div className="welcome">
              <p>Welcome! I can help you analyze data. Try:</p>
              <ul>
                <li>"Show employees with salary over 90k"</li>
                <li>"What are the total sales by region?"</li>
                <li>"Create a bar chart of products by category"</li>
                <li>"Join sales with products and show top sellers"</li>
              </ul>
            </div>
          )}

          {chatHistory.map((m, i) => (
            <div className={`msg ${m.role === 'user' ? 'user' : 'assistant'}`} key={i}>
              <div className="msg-content">
                {m.role === 'user' ? m.content : null}
              </div>
            </div>
          ))}

          {/* A2UI rendered content */}
          <A2UIRenderer onAction={handleAction} />

          {loading && (
            <div className="msg assistant">
              <div className="msg-content loading">
                <span className="dot"></span>
                <span className="dot"></span>
                <span className="dot"></span>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        <div className="composer">
          <input
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && !e.shiftKey && sendMessage()}
            placeholder="Ask a question or request analysis..."
            disabled={loading}
          />
          <button onClick={sendMessage} disabled={loading || !input.trim()}>
            Send
          </button>
        </div>

        <div className="actions">
          <button className="clear-btn" onClick={clearSession}>Clear Session</button>
        </div>
      </section>
    </div>
  )
}

export default function App() {
  const [a2uiMessages] = useState<A2UIMessage[]>([])

  return (
    <A2UIProvider messages={a2uiMessages} catalog={customCatalog}>
      <ChatApp />
    </A2UIProvider>
  )
}
