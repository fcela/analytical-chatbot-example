import React from 'react'
import { useDataBinding } from '@a2ui-sdk/react/0.8'
import type { ValueSource } from '@a2ui-sdk/react/0.8'

interface PlotViewerProps {
    surfaceId: string
    componentId: string
    data?: ValueSource
}

export function PlotViewer({ surfaceId, data }: PlotViewerProps) {
    const svgData = useDataBinding<string>(surfaceId, data, '')
    if (!svgData) return null
    return (
        <div className="plots">
            <img
                src={`data:image/svg+xml;base64,${svgData}`}
                alt="Plot"
                className="plot-image"
            />
        </div>
    )
}
