import clsx from 'clsx'
import { Loader2, Zap } from 'lucide-react'
import type { ButtonHTMLAttributes, ReactNode } from 'react'

export function Card({ className, children }: { className?: string; children: ReactNode }) {
  return <div className={clsx('glass rounded-2xl', className)}>{children}</div>
}

export function Badge({ tone = 'neutral', children, className }: {
  tone?: 'neutral' | 'brand' | 'good' | 'bad' | 'warn' | 'cyan'; children: ReactNode; className?: string
}) {
  const tones = {
    neutral: 'bg-line/5 text-ink-300 ring-line/10',
    brand: 'bg-brand-500/15 text-brand-400 ring-brand-500/30',
    good: 'bg-emerald-500/12 text-emerald-300 ring-emerald-500/30',
    bad: 'bg-rose-500/12 text-rose-300 ring-rose-500/30',
    warn: 'bg-amber-500/12 text-amber-300 ring-amber-500/30',
    cyan: 'bg-cyan-500/12 text-cyan-300 ring-cyan-500/30',
  }
  return (
    <span className={clsx('inline-flex items-center gap-1 rounded-full px-2.5 py-0.5 text-xs font-medium ring-1 ring-inset', tones[tone], className)}>
      {children}
    </span>
  )
}

export function Button({ variant = 'primary', className, children, ...props }:
  ButtonHTMLAttributes<HTMLButtonElement> & { variant?: 'primary' | 'ghost' | 'outline' }) {
  const styles = {
    primary: 'shine bg-brand-500 text-on-brand shadow-lg shadow-brand-500/25 hover:-translate-y-0.5 hover:shadow-xl',
    outline: 'ring-1 ring-inset ring-line/15 text-ink-100 hover:bg-line/5',
    ghost: 'text-ink-300 hover:text-ink-100 hover:bg-line/5',
  }
  return (
    <button
      className={clsx('inline-flex items-center justify-center gap-2 rounded-xl px-4 py-2.5 text-sm font-semibold transition disabled:cursor-not-allowed disabled:opacity-50', styles[variant], className)}
      {...props}
    >
      {children}
    </button>
  )
}

/** Friendly name of the code-generator engine, from the backend's LLM name (e.g. "ollama:qwen2.5-coder:1.5b"). */
export function engineLabel(name?: string | null, backend?: string | null, baseUrl?: string | null) {
  const n = `${backend ?? ''} ${name ?? ''} ${baseUrl ?? ''}`.toLowerCase()
  if (n.includes('ollama') || n.includes(':11434')) return 'Ollama'
  if (n.includes('+lookup')) return 'llama.cpp · fast'
  if (n.includes('gguf') || n.includes('llama.cpp')) return 'llama.cpp'
  if (n.includes('hf')) return 'PyTorch'
  return name ? 'OpenAI-compatible' : 'LLM'
}

export function EngineBadge({ name, backend, baseUrl }: { name?: string | null; backend?: string | null; baseUrl?: string | null }) {
  return (
    <span className="inline-flex items-center gap-1.5 rounded-full bg-line/5 px-2.5 py-0.5 text-xs font-medium text-ink-300 ring-1 ring-inset ring-line/10" title={name ?? ''}>
      <Zap className="h-3 w-3" /> {engineLabel(name, backend, baseUrl)}
    </span>
  )
}

export function Spinner({ className }: { className?: string }) {
  return <Loader2 className={clsx('h-4 w-4 animate-spin', className)} />
}

export function SectionHeading({ eyebrow, title, children }: { eyebrow: string; title: ReactNode; children?: ReactNode }) {
  return (
    <div className="mx-auto max-w-3xl text-center">
      <p className="text-xs font-semibold uppercase tracking-[0.2em] text-brand-400">{eyebrow}</p>
      <h2 className="mt-3 text-3xl font-bold tracking-tight text-ink-100 sm:text-4xl">{title}</h2>
      {children && <p className="mt-4 text-base leading-relaxed text-ink-300">{children}</p>}
    </div>
  )
}

export function Stat({ label, value, sub }: { label: string; value: ReactNode; sub?: string }) {
  return (
    <Card className="p-5">
      <p className="text-xs font-medium uppercase tracking-wider text-ink-400">{label}</p>
      <p className="mt-2 text-3xl font-bold tracking-tight text-ink-100">{value}</p>
      {sub && <p className="mt-1 text-xs text-ink-400">{sub}</p>}
    </Card>
  )
}

export function PageHeader({ eyebrow, title, children, right }: { eyebrow: string; title: string; children?: ReactNode; right?: ReactNode }) {
  return (
    <div className="flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
      <div>
        <p className="text-xs font-semibold uppercase tracking-[0.2em] text-brand-400">{eyebrow}</p>
        <h1 className="mt-2 text-3xl font-bold tracking-tight text-ink-100 sm:text-4xl">{title}</h1>
        {children && <p className="mt-2 max-w-2xl text-ink-300">{children}</p>}
      </div>
      {right}
    </div>
  )
}

export function Label({ children }: { children: ReactNode }) {
  return <label className="mb-1.5 block text-xs font-medium uppercase tracking-wider text-ink-400">{children}</label>
}

export const inputCls =
  'w-full rounded-xl bg-ink-900/80 px-3.5 py-2.5 text-sm text-ink-100 ring-1 ring-inset ring-line/10 placeholder:text-ink-400 focus:outline-none focus:ring-2 focus:ring-brand-500/60'
