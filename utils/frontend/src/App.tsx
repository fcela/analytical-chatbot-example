import React, { useEffect, useState, useRef } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import mermaid from 'mermaid'

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000'

// Initialize mermaid
mermaid.initialize({
  startOnLoad: false,
  theme: 'dark',
  securityLevel: 'loose',
})

function MermaidChart({ chart }: { chart: string }) {
  const [svg, setSvg] = useState('')
  const id = useRef(`mermaid-${Math.random().toString(36).substr(2, 9)}`).current

  useEffect(() => {
    mermaid.render(id, chart).then(({ svg }) => {
      setSvg(svg)
    }).catch((error) => {
      console.error("Mermaid error:", error)
      // Basic error display, maybe improve later
      setSvg(`<div style="color: #ff6b6b; padding: 10px; border: 1px solid #ff6b6b; border-radius: 4px;">Failed to render diagram</div>`)
    })
  }, [chart, id])

  return <div className="mermaid-container" style={{ background: 'rgba(0,0,0,0.2)', padding: '10px', borderRadius: '8px', margin: '10px 0', textAlign: 'center' }} dangerouslySetInnerHTML={{ __html: svg }} />
}

function OutputBlock({ output }: { output: string }) {
  const [expanded, setExpanded] = useState(false)
  const lineCount = output.split('\n').length

  return (
    <div className={`output-block ${expanded ? 'expanded' : 'collapsed'}`}>
      <div 
        className="output-header" 
        onClick={() => setExpanded(!expanded)}
        style={{ cursor: 'pointer', userSelect: 'none', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}
      >
        <span>{expanded ? '▼' : '▶'} Output ({lineCount} lines)</span>
      </div>
      {expanded && (
        <pre>{output}</pre>
      )}
    </div>
  )
}

function CodeBlock({ code }: { code: string }) {
  const [expanded, setExpanded] = useState(false)
  const lineCount = code.split('\n').length

  return (
    <div className={`code-block ${expanded ? 'expanded' : 'collapsed'}`}>
      <div className="code-header" onClick={() => setExpanded(!expanded)}>
        <span className="code-toggle">
          {expanded ? '▼' : '▶'} Python ({lineCount} lines)
        </span>
        <button onClick={(e) => { e.stopPropagation(); navigator.clipboard.writeText(code) }}>
          Copy
        </button>
      </div>
      {expanded && (
        <pre><code>{code}</code></pre>
      )}
    </div>
  )
}

interface ChatResponse {
  message: string
  code?: string | null
  output?: string | null
  plots?: string[] | null
  tables?: string[] | null
  html?: string[] | null
  error?: string | null
}

interface Message {
  role: 'user' | 'assistant'
  content: string
  response?: ChatResponse
}

interface TableInfo {
  columns: Record<string, string>
  row_count: number
}

interface DatabaseInfo {
  available: boolean
  tables: Record<string, TableInfo>
  saved_tables: Record<string, TableInfo>
}

export default function App() {
  const [files, setFiles] = useState<any[]>([])
  const [database, setDatabase] = useState<DatabaseInfo | null>(null)
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    fetchFiles()
    fetchDatabase()
  }, [])

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  async function fetchFiles() {
    try {
      const res = await fetch(`${API_BASE}/files`, { credentials: 'include' })
      const data = await res.json()
      setFiles(data.files || [])
    } catch (e) {
      console.error('Failed to fetch files:', e)
    }
  }

  async function fetchDatabase() {
    try {
      const res = await fetch(`${API_BASE}/database`, { credentials: 'include' })
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
        credentials: 'include'
      })
      if (res.ok) {
        await fetchFiles()
      } else {
        const err = await res.json()
        alert('Upload error: ' + (err.detail || res.statusText))
      }
    } catch (e) {
      alert('Upload failed: ' + e)
    }
    e.target.value = ''
  }

  async function removeFile(filename: string) {
    await fetch(`${API_BASE}/upload/${encodeURIComponent(filename)}`, {
      method: 'DELETE',
      credentials: 'include'
    })
    await fetchFiles()
  }

  async function sendMessage() {
    if (!input.trim() || loading) return

    const userMsg: Message = { role: 'user', content: input }
    setMessages(m => [...m, userMsg])
    setInput('')
    setLoading(true)

    try {
      const res = await fetch(`${API_BASE}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: userMsg.content }),
        credentials: 'include'
      })

      const data = await res.json()

      if (!res.ok) {
        setMessages(m => [...m, {
          role: 'assistant',
          content: 'Error: ' + (data.detail || 'Request failed'),
          response: { message: data.detail || 'Request failed', error: data.detail }
        }])
        return
      }

      const response = data.response as ChatResponse
      setMessages(m => [...m, {
        role: 'assistant',
        content: response.message,
        response
      }])

      // Refresh database info in case tables were saved
      fetchDatabase()
    } catch (e) {
      setMessages(m => [...m, {
        role: 'assistant',
        content: 'Network error: ' + e,
        response: { message: 'Network error', error: String(e) }
      }])
    } finally {
      setLoading(false)
    }
  }

  async function clearSession() {
    await fetch(`${API_BASE}/clear`, { method: 'POST', credentials: 'include' })
    setFiles([])
    setMessages([])
  }

  function renderMessage(msg: Message, index: number) {
    if (msg.role === 'user') {
      return (
        <div className="msg user" key={index}>
          <div className="msg-content">{msg.content}</div>
        </div>
      )
    }

    const resp = msg.response
    return (
      <div className="msg assistant" key={index}>
        <div className="msg-content">
          {/* Message text with Mermaid support */}
          {msg.content && (
            <ReactMarkdown
              remarkPlugins={[remarkGfm]}
              components={{
                code(props) {
                  const {children, className, node, ...rest} = props
                  const match = /language-(\w+)/.exec(className || '')
                  if (match && match[1] === 'mermaid') {
                    return <MermaidChart chart={String(children).replace(/\n$/, '')} />
                  }
                  return (
                    <code {...rest} className={className}>
                      {children}
                    </code>
                  )
                }
              }}
            >
              {msg.content}
            </ReactMarkdown>
          )}

          {/* Tables */}
          {resp?.tables && resp.tables.length > 0 && (
            <div className="tables">
              {resp.tables.map((tableHtml, i) => (
                <div
                  key={i}
                  className="table-container"
                  dangerouslySetInnerHTML={{ __html: tableHtml }}
                />
              ))}
            </div>
          )}

          {/* Plots */}
          {resp?.plots && resp.plots.length > 0 && (
            <div className="plots">
              {resp.plots.map((plot, i) => (
                <img
                  key={i}
                  src={`data:image/svg+xml;base64,${plot}`}
                  alt={`Plot ${i + 1}`}
                  className="plot-image"
                />
              ))}
            </div>
          )}

          {/* HTML Dashboards */}
          {resp?.html && resp.html.length > 0 && (
            <div className="html-dashboards">
              {resp.html.map((htmlContent, i) => (
                <div key={i} className="dashboard-container">
                  <div className="dashboard-header">Interactive Dashboard {i + 1}</div>
                  <iframe
                    title={`Dashboard ${i + 1}`}
                    srcDoc={htmlContent}
                    style={{
                      width: '100%',
                      height: '500px',
                      border: 'none',
                      backgroundColor: 'white',
                      borderRadius: '4px'
                    }}
                    sandbox="allow-scripts"
                  />
                </div>
              ))}
            </div>
          )}

          {/* Code block (collapsible) */}
          {resp?.code && (
            <CodeBlock code={resp.code} />
          )}

          {/* Output */}
          {resp?.output && (
            <OutputBlock output={resp.output} />
          )}

          {/* Error */}
          {resp?.error && (
            <div className="error-block">
              <strong>Error:</strong> {resp.error}
            </div>
          )}
        </div>
      </div>
    )
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
                <span className="file-info">{f.rows} rows × {f.columns} cols</span>
                <button className="remove-btn" onClick={() => removeFile(f.filename)}>×</button>
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
          {Object.keys(database.saved_tables).length > 0 && (
            <>
              <div className="saved-tables-header">Saved Results</div>
              <div className="table-chips saved">
                {Object.entries(database.saved_tables).map(([name, info]) => (
                  <div className="table-chip saved" key={name}>
                    <span className="table-name">{name.replace('saved_', '')}</span>
                    <span className="table-info">{info.row_count} rows</span>
                  </div>
                ))}
              </div>
            </>
          )}
        </section>
      )}

      <section className="chat-section">
        <div className="messages">
          {messages.length === 0 && (
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
          {messages.map((m, i) => renderMessage(m, i))}
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
