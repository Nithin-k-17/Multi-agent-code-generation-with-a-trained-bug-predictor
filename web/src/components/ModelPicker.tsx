import clsx from 'clsx'
import { AnimatePresence, motion } from 'framer-motion'
import { Check, ChevronDown, Cpu } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import type { ModelOption } from '../lib/api'

/** Model dropdown next to the Run button: Qwen2.5-Coder 1.5B by default, plus any model pulled in Ollama. */
export default function ModelPicker({ models, value, onChange, disabled }: {
  models: ModelOption[]; value: string; onChange: (id: string) => void; disabled?: boolean
}) {
  const [open, setOpen] = useState(false)
  const ref = useRef<HTMLDivElement>(null)
  const current = models.find(m => m.id === value) ?? models[0]

  useEffect(() => {
    if (!open) return
    const onDown = (e: MouseEvent) => { if (!ref.current?.contains(e.target as Node)) setOpen(false) }
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') setOpen(false) }
    document.addEventListener('mousedown', onDown)
    document.addEventListener('keydown', onKey)
    return () => { document.removeEventListener('mousedown', onDown); document.removeEventListener('keydown', onKey) }
  }, [open])

  if (!current) return null
  return (
    <div ref={ref} className="relative">
      <button type="button" disabled={disabled} onClick={() => setOpen(o => !o)} aria-haspopup="listbox" aria-expanded={open}
        title="Choose the model that writes the code"
        className="inline-flex items-center gap-2 rounded-xl px-3 py-2.5 text-sm font-medium text-ink-200 ring-1 ring-inset ring-line/15 transition hover:bg-line/5 hover:text-ink-100 disabled:cursor-not-allowed disabled:opacity-50">
        <Cpu className="h-4 w-4 text-ink-400" />
        {current.label}
        <ChevronDown className={clsx('h-4 w-4 text-ink-400 transition', open && 'rotate-180')} />
      </button>
      <AnimatePresence>
        {open && (
          <motion.ul role="listbox" initial={{ opacity: 0, y: 6, scale: 0.98 }} animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 6, scale: 0.98 }} transition={{ duration: 0.12 }}
            className="absolute bottom-full left-0 z-30 mb-2 w-72 max-w-[calc(100vw-2rem)] overflow-hidden rounded-xl bg-ink-900 p-1 shadow-2xl ring-1 ring-line/15">
            {models.map(m => (
              <li key={m.id}>
                <button type="button" role="option" aria-selected={m.id === current.id}
                  onClick={() => { onChange(m.id); setOpen(false) }}
                  className={clsx('flex w-full items-start gap-3 rounded-lg px-3 py-2.5 text-left transition hover:bg-line/5',
                    m.id === current.id && 'bg-line/5')}>
                  <span className="flex-1">
                    <span className="block text-sm font-medium text-ink-100">{m.label}</span>
                    <span className="block text-xs text-ink-400">{m.detail}{m.size_gb ? ` · ${m.size_gb} GB` : ''}</span>
                  </span>
                  {m.id === current.id && <Check className="mt-0.5 h-4 w-4 shrink-0 text-brand-400" />}
                </button>
              </li>
            ))}
            {models.length === 1 && (
              <li className="px-3 py-2 text-xs text-ink-400">More models appear here after <code className="font-mono">ollama pull</code>.</li>
            )}
          </motion.ul>
        )}
      </AnimatePresence>
    </div>
  )
}
