import React from 'react'
import { useDataBinding } from '@a2ui-sdk/react/0.8'
import type { ValueSource } from '@a2ui-sdk/react/0.8'

interface DataTableProps {
    surfaceId: string
    componentId: string
    html?: ValueSource
}

export function DataTable({ surfaceId, html }: DataTableProps) {
    const htmlContent = useDataBinding<string>(surfaceId, html, '')
    if (!htmlContent) return null
    return (
        <div className="table-container" dangerouslySetInnerHTML={{ __html: htmlContent }} />
    )
}
