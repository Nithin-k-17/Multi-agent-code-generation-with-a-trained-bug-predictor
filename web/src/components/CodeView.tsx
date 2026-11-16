import Editor, { type OnMount } from '@monaco-editor/react'
import { useEffect, useRef } from 'react'
import type { HotLine, Severity } from '../lib/api'
import { usePalette, useTheme } from '../lib/theme'

type MonacoEditor = Parameters<OnMount>[0]
type Monaco = Parameters<OnMount>[1]

/** A highlighted line range with a hover message (used for detected bugs). */
export interface LineMark { line: number; end_line: number; severity: Severity; message: string }

/**
 * Monaco editor for Python. Read-only unless `onChange` is given.
 * `marks` highlights bug lines by severity; `hotLines` (legacy) highlights the classifier's attention lines.
 * `placeholder` is shown while the editor is empty; `onPaste` fires after the user pastes.
 */
export default function CodeView({ code, hotLines = [], marks = [], onChange, onPaste, onSubmit, height = 260, placeholder }: {
  code: string; hotLines?: HotLine[]; marks?: LineMark[]; onChange?: (v: string) => void; onPaste?: (v: string) => void
  onSubmit?: (v: string) => void
  height?: number | string; placeholder?: string
}) {
  const editorRef = useRef<MonacoEditor | null>(null)
  const monacoRef = useRef<Monaco | null>(null)
  const decoRef = useRef<string[]>([])
  const pasteRef = useRef(onPaste)
  pasteRef.current = onPaste
  const submitRef = useRef(onSubmit)
  submitRef.current = onSubmit
  const p = usePalette()
  const { theme } = useTheme()
  const themeName = `bugaware-${theme}`
  const h = (c: string) => c.replace('#', '')

  const applyDecorations = () => {
    const editor = editorRef.current, monaco = monacoRef.current
    if (!editor || !monaco) return
    const lines = editor.getModel()?.getLineCount() ?? 1
    const clamp = (n: number) => Math.max(1, Math.min(lines, n))
    decoRef.current = editor.deltaDecorations(decoRef.current, [
      ...marks.map(m => ({
        range: new monaco.Range(clamp(m.line), 1, clamp(m.end_line), 1),
        options: {
          isWholeLine: true, className: `mark-${m.severity.toLowerCase()}`, linesDecorationsClassName: 'mark-gutter',
          hoverMessage: { value: m.message },
        },
      })),
      ...hotLines.map((hl, i) => ({
        range: new monaco.Range(clamp(hl.line_start), 1, clamp(hl.line_end), 1),
        options: { isWholeLine: true, className: i === 0 ? 'hot-line-bg' : 'warm-line-bg', linesDecorationsClassName: 'hot-line-gutter' },
      })),
    ])
  }

  useEffect(applyDecorations, [hotLines, marks, code]) // eslint-disable-line react-hooks/exhaustive-deps

  // Re-colour the editor when the site theme changes
  useEffect(() => {
    const monaco = monacoRef.current
    if (!monaco) return
    defineTheme(monaco)
    monaco.editor.setTheme(themeName)
  }, [themeName]) // eslint-disable-line react-hooks/exhaustive-deps

  // Theme is registered before the editor mounts so there is no grey flash.
  const defineTheme = (monaco: Monaco) => {
    monaco.editor.defineTheme(themeName, {
      base: 'vs-dark', inherit: true,
      rules: [
        { token: '', foreground: h(p.panelText) },
        { token: 'keyword', foreground: h(p.keyword), fontStyle: 'bold' },
        { token: 'string', foreground: h(p.string) },
        { token: 'number', foreground: h(p.number) },
        { token: 'comment', foreground: h(p.comment), fontStyle: 'italic' },
        { token: 'delimiter', foreground: h(p.string) },
      ],
      colors: {
        'editor.background': p.panel, 'editor.foreground': p.panelText, 'editorLineNumber.foreground': p.lineNo,
        'editorLineNumber.activeForeground': p.panelText, 'editor.lineHighlightBackground': '#FFFFFF0a',
        'editorGutter.background': p.panel, 'editorCursor.foreground': p.string,
        'editor.selectionBackground': '#FFFFFF26', 'editorIndentGuide.background1': '#FFFFFF12',
      },
    })
  }

  const onMount: OnMount = (editor, monaco) => {
    editorRef.current = editor
    monacoRef.current = monaco
    monaco.editor.setTheme(themeName)
    editor.onDidPaste(() => pasteRef.current?.(editor.getValue()))
    // Ctrl/Cmd + Enter submits (Monaco would otherwise swallow it)
    editor.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyCode.Enter, () => submitRef.current?.(editor.getValue()))
    applyDecorations()
  }

  return (
    <div className="relative overflow-hidden rounded-xl ring-1 ring-line/10">
      <Editor
        height={height}
        language="python"
        value={code}
        onChange={v => onChange?.(v ?? '')}
        beforeMount={defineTheme}
        onMount={onMount}
        theme={themeName}
        loading={<div className="h-full w-full bg-blood-night p-4 font-mono text-xs text-sage">Loading editor…</div>}
        options={{
          readOnly: !onChange, minimap: { enabled: false }, fontSize: 13, fontFamily: 'JetBrains Mono, monospace',
          scrollBeyondLastLine: false, lineNumbersMinChars: 3, padding: { top: 12, bottom: 12 },
          renderLineHighlight: onChange ? 'line' : 'none', automaticLayout: true, tabSize: 4,
          scrollbar: { verticalScrollbarSize: 8, horizontalScrollbarSize: 8 },
        }}
      />
      {placeholder && !code && (
        <div className="pointer-events-none absolute left-[52px] top-3 whitespace-pre-line font-mono text-[13px] leading-[19px] text-white/35">
          {placeholder}
        </div>
      )}
    </div>
  )
}
