import { motion } from 'framer-motion'
import { Zap } from 'lucide-react'
import { useEffect, useRef } from 'react'
import { agentLabel, metaFor } from '../lib/agents'
import { fmtSeconds, useNow } from '../lib/hooks'
import type { LiveOutput as Live } from '../lib/runState'

/** Terminal-style panel showing the LLM's output as it is generated, token by token. */
export default function LiveOutput({ live }: { live: Live }) {
  const ref = useRef<HTMLPreElement>(null)
  const now = useNow(true, 200)
  const meta = metaFor(live.agent)
  const Icon = meta.icon
  const elapsed = (now - live.startedAt) / 1000
  // tokens/second since the first token (cached answers arrive in one chunk, so need a few to be meaningful)
  const genSecs = live.firstTokenAt ? (now - live.firstTokenAt) / 1000 : 0
  const tps = live.chunks && live.chunks > 3 && genSecs > 0.3 ? live.chunks / genSecs : null

  useEffect(() => {
    const el = ref.current
    if (el) el.scrollTop = el.scrollHeight
  }, [live.text])

  return (
    <motion.div layout initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }}
      className="overflow-hidden rounded-2xl glass-dark ring-1" style={{ boxShadow: `0 0 0 1px ${meta.accent}55, 0 0 40px ${meta.accent}1a` }}>
      <div className="flex items-center gap-3 border-b border-vanilla/10 px-4 py-2.5">
        <span className="grid h-7 w-7 place-items-center rounded-lg" style={{ background: `${meta.accent}22`, color: meta.accent }}><Icon className="h-4 w-4" /></span>
        <span className="text-sm font-medium text-vanilla">
          {agentLabel[live.agent]}{live.iter ? ` · attempt ${live.iter}` : ''}
          
        </span>
        <span className="ml-auto flex items-center gap-3 font-mono text-xs text-sage">
          {tps !== null && (
            <span className="flex items-center gap-1 rounded-full bg-vanilla/10 px-2 py-0.5 tabular-nums text-vanilla/90" title="Generation speed">
              <Zap className="h-3 w-3" />{tps.toFixed(0)} tok/s
            </span>
          )}
          <span className="tabular-nums">{fmtSeconds(elapsed)}</span>
        </span>
      </div>
      <pre ref={ref} className="max-h-56 min-h-20 overflow-auto px-4 py-3 font-mono text-xs leading-relaxed whitespace-pre-wrap text-vanilla/90">
        {live.text || <span className="text-sage">{live.agent === 'bug_predictor' ? 'Checking the code for likely bugs…' : live.agent === 'tester' ? 'Running tests in the sandbox…' : 'Thinking…'}</span>}
        <span className="ml-0.5 inline-block h-3.5 w-1.5 translate-y-0.5 animate-pulse bg-sage" />
      </pre>
    </motion.div>
  )
}
