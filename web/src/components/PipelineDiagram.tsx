import clsx from 'clsx'
import { motion } from 'framer-motion'
import { Fragment } from 'react'
import { AGENTS } from '../lib/agents'

/** The five-agent pipeline. `active` highlights the running agent; `skipBug` greys out Step 3 (baseline). */
export default function PipelineDiagram({ active, skipBug = false, compact = false }: {
  active?: string; skipBug?: boolean; compact?: boolean
}) {
  return (
    <div className="flex flex-col items-stretch gap-2 md:flex-row md:items-center md:gap-0">
      {AGENTS.map((a, i) => {
        const Icon = a.icon
        const isActive = active && active.startsWith(a.key)
        const disabled = skipBug && a.isNew
        return (
          <Fragment key={a.key}>
            <motion.div
              initial={{ opacity: 0, y: 12 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }}
              transition={{ delay: i * 0.08 }}
              className={clsx(
                'relative flex-1 rounded-2xl p-4 transition',
                a.isNew ? 'glass ring-1 ring-brand-500/40' : 'glass',
                isActive && 'ring-2 ring-offset-0',
                disabled && 'opacity-30 grayscale',
                compact ? 'md:p-3' : 'md:p-5',
              )}
              style={isActive ? { boxShadow: `0 0 0 2px ${a.accent}, 0 0 32px ${a.accent}55` } : a.isNew && !disabled ? undefined : undefined}
            >
              {a.isNew && !disabled && (
                <span className="absolute -top-2.5 right-3 rounded-full bg-brand-500 px-2 py-0.5 text-[10px] font-bold uppercase tracking-wider text-on-brand">New</span>
              )}
              <div className="flex items-center gap-3">
                <div className="grid h-10 w-10 shrink-0 place-items-center rounded-xl" style={{ background: `${a.accent}1f`, color: a.accent }}>
                  <Icon className={clsx('h-5 w-5', isActive && 'animate-pulse-soft')} />
                </div>
                <div className="min-w-0">
                  <p className="text-[10px] font-semibold uppercase tracking-widest text-ink-400">Step {a.step}</p>
                  <p className="truncate font-semibold text-ink-100">{a.name}</p>
                </div>
              </div>
              {!compact && <p className="mt-3 text-sm leading-relaxed text-ink-300">{a.blurb}</p>}
            </motion.div>
            {i < AGENTS.length - 1 && (
              <svg className="mx-auto h-6 w-6 shrink-0 rotate-90 text-brand-500 md:mx-0 md:h-6 md:w-8 md:rotate-0" viewBox="0 0 32 24" aria-hidden>
                <path d="M2 12 H28" stroke="currentColor" strokeWidth="2" strokeDasharray="4 4" className="animate-flow" />
                <path d="M24 7 L29 12 L24 17" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
              </svg>
            )}
          </Fragment>
        )
      })}
    </div>
  )
}
