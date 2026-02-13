import React, { useState } from 'react'
import { useDataBinding } from '@a2ui-sdk/react/0.8'
import type { ValueSource } from '@a2ui-sdk/react/0.8'

interface CodeBlockProps {
    surfaceId: string
    componentId: string
    code?: ValueSource
    language?: ValueSource
}

export function CodeBlock({ surfaceId, code, language }: CodeBlockProps) {
    const codeText = useDataBinding<string>(surfaceId, code, '')
    const lang = useDataBinding<string>(surfaceId, language, 'python')
    const [expanded, setExpanded] = useState(false)
    const lineCount = codeText.split('\n').length

    if (!codeText) return null

    return (
        <div className={`code-block ${expanded ? 'expanded' : 'collapsed'}`}>
            <div className="code-header" onClick={() => setExpanded(!expanded)}>
                <span className="code-toggle">
                    {expanded ? '\u25BC' : '\u25B6'} {lang} ({lineCount} lines)
                </span>
                <button onClick={(e) => { e.stopPropagation(); navigator.clipboard.writeText(codeText) }}>
                    Copy
                </button>
            </div>
            {expanded && (
                <pre><code>{codeText}</code></pre>
            )}
        </div>
    )
}
