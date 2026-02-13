import React from 'react'
import { useDataBinding } from '@a2ui-sdk/react/0.8'
import type { ValueSource } from '@a2ui-sdk/react/0.8'

interface HtmlViewerProps {
    surfaceId: string
    componentId: string
    html?: ValueSource
}

export function HtmlViewer({ surfaceId, html }: HtmlViewerProps) {
    const htmlContent = useDataBinding<string>(surfaceId, html, '')
    const iframeRef = React.useRef<HTMLIFrameElement>(null)

    // Auto-resize iframe to fit content
    React.useEffect(() => {
        const iframe = iframeRef.current
        if (!iframe) return
        const resize = () => {
            try {
                const doc = iframe.contentDocument || iframe.contentWindow?.document
                if (doc?.body) {
                    iframe.style.height = Math.max(600, doc.body.scrollHeight + 40) + 'px'
                }
            } catch { /* cross-origin — keep default height */ }
        }
        iframe.addEventListener('load', resize)
        return () => iframe.removeEventListener('load', resize)
    }, [htmlContent])

    if (!htmlContent) return null
    return (
        <div className="dashboard-container">
            <iframe
                ref={iframeRef}
                title="Dashboard"
                srcDoc={htmlContent}
                style={{
                    width: '100%',
                    minHeight: '600px',
                    border: 'none',
                    backgroundColor: 'white',
                    borderRadius: '8px',
                }}
                sandbox="allow-scripts"
            />
        </div>
    )
}
