import { motion } from 'framer-motion'
import { ArrowRight, BarChart3, Bot, ClipboardPaste, Copy, Cpu, FlaskConical, ListChecks, Play, ScanSearch, Wrench, Zap } from 'lucide-react'
import { useEffect, useState } from 'react'
import { api } from '../lib/api'
import { Link } from 'react-router-dom'
import Model3D from '../components/Model3D'
import PipelineDiagram from '../components/PipelineDiagram'
import { Card } from '../components/ui'

const fade = { initial: { opacity: 0, y: 18 }, whileInView: { opacity: 1, y: 0 }, viewport: { once: true, margin: '-60px' }, transition: { duration: 0.5 } }

/** Live bug-model metrics from the API; the fallback labels show while the server is offline. */
function useStats() {
  const [stats, setStats] = useState<{ auc?: number } | null>(null)
  useEffect(() => {
    let alive = true
    api.model().then(m => { if (alive && m?.val_metrics) setStats({ auc: m.val_metrics.roc_auc }) }).catch(() => {})
    return () => { alive = false }
  }, [])
  return stats
}

export default function Home() {
  const stats = useStats()
  return (
    <div>
      {/* ------------------------------------------------------------ hero */}
      <section className="mx-auto grid max-w-7xl items-center gap-2 px-4 pt-8 sm:px-6 md:grid-cols-[1fr_1fr] lg:pt-14">
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.6 }}>
          <h1 className="text-4xl font-extrabold leading-[1.05] tracking-tight text-ink-100 sm:text-6xl">
            Find the bugs in your code <span className="text-gradient">before they find you.</span>
          </h1>
          <p className="mt-5 max-w-xl text-lg leading-relaxed text-ink-300">
            Paste Python code and see how many bugs it has, which lines they're on, how serious they are and how to fix them.
            Or describe a task and let five AI agents write, test and repair the code for you.
          </p>
          <div className="mt-7 flex flex-wrap gap-3">
            <Link to="/scanner" className="inline-flex items-center gap-2 shine rounded-xl bg-brand-500 px-5 py-3 font-semibold text-on-brand shadow-xl shadow-brand-500/25 transition hover:-translate-y-0.5">
              <ScanSearch className="h-4 w-4" /> Scan your code
            </Link>
            <Link to="/playground" className="inline-flex items-center gap-2 rounded-xl px-5 py-3 font-semibold text-ink-100 ring-1 ring-inset ring-line/20 hover:bg-line/5">
              <Play className="h-4 w-4" /> Run the agents
            </Link>
          </div>
        </motion.div>
        <Model3D scene="robot" className="h-[320px] sm:h-[440px] lg:h-[540px]" zoom={0.82} />
      </section>

      {/* ------------------------------------------------------------ at a glance */}
      <motion.section {...fade} className="mx-auto mt-6 max-w-7xl px-4 sm:px-6">
        <Card className="grid grid-cols-2 divide-line/10 sm:grid-cols-4 sm:divide-x">
          {[
            { icon: Bot, v: '5', l: 'AI agents working together' },
            { icon: BarChart3, v: stats?.auc != null ? stats.auc.toFixed(2) : 'CodeBERT', l: stats?.auc != null ? 'bug-model ROC-AUC' : 'trained bug predictor' },
            { icon: Zap, v: 'Ollama', l: 'fast local code generation' },
            { icon: Cpu, v: '100%', l: 'local, your code never leaves' },
          ].map(x => (
            <div key={x.l} className="flex items-center gap-3 p-5">
              <span className="grid h-10 w-10 shrink-0 place-items-center rounded-xl bg-line/5 text-ink-100 ring-1 ring-line/10"><x.icon className="h-5 w-5" /></span>
              <span><b className="block text-xl font-bold tracking-tight text-ink-100 tabular-nums">{x.v}</b><span className="text-xs text-ink-400">{x.l}</span></span>
            </div>
          ))}
        </Card>
      </motion.section>

      {/* ------------------------------------------------------------ bug scanner */}
      <section className="mx-auto mt-16 max-w-7xl px-4 sm:px-6">
        <div className="grid items-center gap-8 md:grid-cols-[1fr_1.1fr]">
          <Model3D scene="screen" className="order-2 h-[260px] sm:h-[380px] md:order-1" zoom={1.1} />
          <motion.div {...fade} className="order-1 md:order-2">
            <p className="text-xs font-semibold uppercase tracking-[0.2em] text-brand-400">Bug Scanner</p>
            <h2 className="mt-3 text-3xl font-bold tracking-tight text-ink-100 sm:text-4xl">A full bug report in seconds</h2>
            <ul className="mt-6 space-y-3">
              {[
                { icon: ClipboardPaste, t: 'Paste your code', d: 'Analysis starts as soon as you paste. There is nothing to configure.' },
                { icon: ListChecks, t: 'See every bug', d: 'Number of bugs, line numbers, type (syntax, runtime, logic, edge case) and severity.' },
                { icon: Wrench, t: 'Get the fix', d: 'A clear explanation for each bug and a corrected version of your code.' },
                { icon: Copy, t: 'Copy and go', d: 'One click copies the fixed code.' },
              ].map(s => (
                <li key={s.t} className="flex gap-3">
                  <span className="grid h-9 w-9 shrink-0 place-items-center rounded-xl bg-brand-500 text-on-brand"><s.icon className="h-4 w-4" /></span>
                  <span><b className="text-ink-100">{s.t}</b><span className="block text-sm text-ink-300">{s.d}</span></span>
                </li>
              ))}
            </ul>
            <Link to="/scanner" className="mt-7 inline-flex items-center gap-2 font-semibold text-ink-100 hover:underline">Open the Bug Scanner <ArrowRight className="h-4 w-4" /></Link>
          </motion.div>
        </div>
      </section>

      {/* ------------------------------------------------------------ agents */}
      <section className="mx-auto mt-16 max-w-7xl px-4 sm:px-6">
        <div className="grid items-center gap-8 md:grid-cols-[1.1fr_1fr]">
          <motion.div {...fade}>
            <p className="text-xs font-semibold uppercase tracking-[0.2em] text-brand-400">Agent Pipeline</p>
            <h2 className="mt-3 text-3xl font-bold tracking-tight text-ink-100 sm:text-4xl">Describe a task. Get tested code.</h2>
            <p className="mt-4 leading-relaxed text-ink-300">
              Five agents work as a team. One of them checks every new draft for likely bugs, so testing starts in the right
              place and fixes happen sooner. You watch each step live and get the final code, the test results and a summary
              of what was fixed.
            </p>
            <div className="mt-5 flex flex-wrap gap-2 text-sm text-ink-300">
              {[{ i: Bot, t: 'Live agent output' }, { i: FlaskConical, t: 'Sandboxed tests' }, { i: Wrench, t: 'Automatic fixes' }].map(x => (
                <span key={x.t} className="inline-flex items-center gap-1.5 rounded-full bg-line/[0.05] px-3 py-1 ring-1 ring-line/10"><x.i className="h-3.5 w-3.5" /> {x.t}</span>
              ))}
            </div>
            <Link to="/playground" className="mt-7 inline-flex items-center gap-2 shine rounded-xl bg-brand-500 px-5 py-3 font-semibold text-on-brand shadow-xl shadow-brand-500/25 transition hover:-translate-y-0.5">
              Open the Agent Pipeline <ArrowRight className="h-4 w-4" />
            </Link>
          </motion.div>
          <Model3D scene="hologram" className="h-[260px] sm:h-[380px]" />
        </div>
        <motion.div {...fade} className="mt-10">
          <Card className="p-5 sm:p-6"><PipelineDiagram /></Card>
        </motion.div>
      </section>
    </div>
  )
}
