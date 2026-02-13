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
    if (!htmlContent) return null
    return (
        <div className="dashboard-container">
            <div className="dashboard-header">Interactive Output</div>
            <iframe
                title="Dashboard"
                srcDoc={htmlContent}
                style={{
                    width: '100%',
                    height: '500px',
                    border: 'none',
                    backgroundColor: 'white',
                    borderRadius: '4px',
                }}
                sandbox="allow-scripts"
            />
        </div>
    )
}
