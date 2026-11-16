import clsx from 'clsx'
import { AnimatePresence, motion } from 'framer-motion'
import { Bot, Check, ChevronDown, Copy, Cpu, FlaskConical, Lightbulb, ScanSearch } from 'lucide-react'
import { useState, type ReactNode } from 'react'
import type { Finding, Severity } from '../lib/api'
import type { LineMark } from './CodeView'

const SEV_ORDER: Severity[] = ['Critical', 'High', 'Medium', 'Low']

export function SeverityBadge({ s }: { s: Severity }) {
  return <span className={clsx('sev', `sev-${s.toLowerCase()}`)}>{s}</span>
}

export function TypeBadge({ t }: { t: string }) {
  return <span className="inline-flex items-center rounded-md bg-line/[0.06] px-2 py-0.5 text-[11px] font-semibold text-ink-200 ring-1 ring-inset ring-line/10">{t}</span>
}

const SOURCE = {
  static: { icon: ScanSearch, label: 'Exact check', hint: 'Found by analysing the code structure; the line number is exact.' },
  execution: { icon: FlaskConical, label: 'Found by running the code', hint: 'Python raised this error when the file was loaded in a sandbox.' },
  ai: { icon: Bot, label: 'AI review', hint: 'The AI reviewer had to change these lines to fix the code. Check the explanation against your intent.' },
}

export const isBug = (f: Finding) => f.type !== 'Code quality'

export const marksFor = (findings: Finding[]): LineMark[] =>
  findings.map(f => ({ line: f.line, end_line: f.end_line, severity: f.severity, message: `**${f.type} · ${f.severity}**\n\n${f.title}` }))

export const lineLabel = (f: { line: number; end_line: number }) => (f.end_line > f.line ? `Lines ${f.line}–${f.end_line}` : `Line ${f.line}`)

/** Counts per severity, most severe first. */
export function SeveritySummary({ findings }: { findings: Finding[] }) {
  const counts = SEV_ORDER.map(s => [s, findings.filter(f => f.severity === s).length] as const).filter(([, n]) => n > 0)
  return (
    <div className="flex flex-wrap gap-1.5">
      {counts.map(([s, n]) => <span key={s} className={clsx('sev', `sev-${s.toLowerCase()}`)}>{n} {s}</span>)}
    </div>
  )
}

export function FindingCard({ f, index, onLocate }: { f: Finding; index: number; onLocate?: (f: Finding) => void }) {
  const src = SOURCE[f.source]
  return (
    <motion.li layout initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: Math.min(index, 6) * 0.04 }}
      className="rounded-2xl bg-ink-900/70 p-4 ring-1 ring-line/10">
      <div className="flex flex-wrap items-center gap-2">
        <span className="grid h-6 w-6 place-items-center rounded-full bg-brand-500 text-[11px] font-bold text-on-brand">{index + 1}</span>
        <button onClick={() => onLocate?.(f)} className="font-mono text-sm font-semibold text-ink-100 underline-offset-2 hover:underline">{lineLabel(f)}</button>
        <TypeBadge t={f.type} />
        <SeverityBadge s={f.severity} />
        <span className="ml-auto flex items-center gap-1 text-[11px] text-ink-400" title={src.hint}><src.icon className="h-3.5 w-3.5" /> {src.label}</span>
      </div>
      <p className="mt-2.5 text-sm leading-relaxed text-ink-200">{f.explanation}</p>
      {f.before && (
        <pre className="mt-2 overflow-x-auto rounded-lg bg-blood-night px-3 py-2 font-mono text-xs leading-relaxed text-vanilla/85">{f.before}</pre>
      )}
      <div className="mt-2.5 flex gap-2 rounded-xl bg-line/[0.04] p-3 text-sm text-ink-300">
        <Lightbulb className="mt-0.5 h-4 w-4 shrink-0 text-ink-100" />
        <span className="whitespace-pre-wrap leading-relaxed"><b className="text-ink-100">Suggestion: </b>{f.suggestion}</span>
      </div>
    </motion.li>
  )
}

export function CopyButton({ text, label = 'Copy code' }: { text: string; label?: string }) {
  const [done, setDone] = useState(false)
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(text)
    } catch {
      const ta = document.createElement('textarea')
      ta.value = text; document.body.appendChild(ta); ta.select(); document.execCommand('copy'); ta.remove()
    }
    setDone(true)
    setTimeout(() => setDone(false), 1600)
  }
  return (
    <button onClick={copy} className="inline-flex items-center gap-1.5 rounded-lg bg-brand-500 px-3 py-1.5 text-xs font-semibold text-on-brand transition hover:brightness-110">
      {done ? <Check className="h-3.5 w-3.5" /> : <Copy className="h-3.5 w-3.5" />} {done ? 'Copied' : label}
    </button>
  )
}

/** Collapsible section, closed by default (used for "Advanced Model Details" and "Execution Logs"). */
export function Disclosure({ title, icon, children, defaultOpen = false }: { title: string; icon?: ReactNode; children: ReactNode; defaultOpen?: boolean }) {
  const [open, setOpen] = useState(defaultOpen)
  return (
    <div className="glass rounded-2xl">
      <button onClick={() => setOpen(o => !o)} className="flex w-full items-center gap-2 px-5 py-4 text-left text-sm font-semibold text-ink-100" aria-expanded={open}>
        {icon ?? <Cpu className="h-4 w-4 text-ink-400" />} {title}
        <ChevronDown className={clsx('ml-auto h-4 w-4 text-ink-400 transition', open && 'rotate-180')} />
      </button>
      <AnimatePresence initial={false}>
        {open && (
          <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: 'auto', opacity: 1 }} exit={{ height: 0, opacity: 0 }} className="overflow-hidden">
            <div className="border-t border-line/10 px-5 py-4">{children}</div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

export function KV({ k, v }: { k: string; v: ReactNode }) {
  return (
    <div className="flex items-baseline justify-between gap-4 border-b border-line/5 py-1.5 text-sm last:border-0">
      <span className="text-ink-400">{k}</span><span className="text-right font-medium text-ink-100">{v}</span>
    </div>
  )
}

/** The bug-prediction model's facts, shown only inside "Advanced Model Details". */
export const MODEL_FACTS = {
  name: 'CodeBERT bug predictor (microsoft/codebert-base, fine-tuned)',
  version: 'v3 · run 3 · 512-token context · seed 123',
  mbpp: 'ROC-AUC 0.70 (95% CI 0.68–0.72) · recall 90% · precision 58% · accuracy 60%',
  humaneval: 'ROC-AUC 0.63 (zero-shot)',
}

export function riskLevel(score: number) {
  return score >= 0.795 ? 'High' : score >= 0.34 ? 'Medium' : 'Low'
}
