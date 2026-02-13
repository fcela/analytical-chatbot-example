import React, { useState } from 'react'
import { useDataBinding } from '@a2ui-sdk/react/0.8'
import type { ValueSource } from '@a2ui-sdk/react/0.8'

interface OutputBlockProps {
    surfaceId: string
    componentId: string
    output?: ValueSource
}

export function OutputBlock({ surfaceId, output }: OutputBlockProps) {
    const outputText = useDataBinding<string>(surfaceId, output, '')
    const [expanded, setExpanded] = useState(false)
    const lineCount = outputText.split('\n').length

    if (!outputText) return null

    return (
        <div className={`output-block ${expanded ? 'expanded' : 'collapsed'}`}>
            <div
                className="output-header"
                onClick={() => setExpanded(!expanded)}
                style={{ cursor: 'pointer', userSelect: 'none', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}
            >
                <span>{expanded ? '\u25BC' : '\u25B6'} Output ({lineCount} lines)</span>
            </div>
            {expanded && (
                <pre>{outputText}</pre>
            )}
        </div>
    )
}
