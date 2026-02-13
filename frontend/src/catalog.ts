import { standardCatalog } from '@a2ui-sdk/react/0.8/standard-catalog'
import type { Catalog } from '@a2ui-sdk/react/0.8'
import { PlotViewer } from './components/PlotViewer'
import { CodeBlock } from './components/CodeBlock'
import { OutputBlock } from './components/OutputBlock'
import { DataTable } from './components/DataTable'
import { MermaidChart } from './components/MermaidChart'
import { HtmlViewer } from './components/HtmlViewer'

export const customCatalog: Catalog = {
    ...standardCatalog,
    components: {
        ...standardCatalog.components,
        PlotViewer: PlotViewer as any,
        CodeBlock: CodeBlock as any,
        OutputBlock: OutputBlock as any,
        DataTable: DataTable as any,
        MermaidChart: MermaidChart as any,
        HtmlViewer: HtmlViewer as any,
    },
}
