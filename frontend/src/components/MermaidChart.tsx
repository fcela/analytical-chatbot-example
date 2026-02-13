import React, { useEffect, useState, useRef } from 'react'
import mermaid from 'mermaid'
import { useDataBinding } from '@a2ui-sdk/react/0.8'
import type { ValueSource } from '@a2ui-sdk/react/0.8'

mermaid.initialize({
    startOnLoad: false,
    theme: 'dark',
    securityLevel: 'loose',
})

interface MermaidChartProps {
    surfaceId: string
    componentId: string
    definition?: ValueSource
}

export function MermaidChart({ surfaceId, definition }: MermaidChartProps) {
    const mermaidDef = useDataBinding<string>(surfaceId, definition, '')
    const [svg, setSvg] = useState('')
    const id = useRef(`mermaid-${Math.random().toString(36).substr(2, 9)}`).current

    useEffect(() => {
        let cleaned = mermaidDef.trim()
        cleaned = cleaned.replace(/^```mermaid\s*/i, '').replace(/\s*```$/, '').trim()

        setSvg('')
        if (!cleaned) return

        mermaid.render(id, cleaned).then(({ svg }) => {
            setSvg(svg)
        }).catch((error) => {
            console.error('Mermaid error:', error)
            setSvg(`<div style="color: #ff6b6b; padding: 10px;">Mermaid render error</div>`)
        })
    }, [mermaidDef, id])

    return (
        <div
            className="mermaid-container"
            style={{ background: 'rgba(0,0,0,0.2)', padding: '10px', borderRadius: '8px', margin: '10px 0', textAlign: 'center', overflowX: 'auto' }}
            dangerouslySetInnerHTML={{ __html: svg }}
        />
    )
}
