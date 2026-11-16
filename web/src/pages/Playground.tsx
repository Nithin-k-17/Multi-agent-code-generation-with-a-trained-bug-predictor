import clsx from 'clsx'
import { AnimatePresence, motion } from 'framer-motion'
import {
  AlertTriangle, Bug, CheckCircle2, ChevronDown, CircleDashed, Clock, Code2, FileText, FlaskConical, ListTree, Loader2,
  MinusCircle, Play, PlayCircle, Plus, Rocket, Scale, Square, Trash2, Wrench, XCircle, Zap,
} from 'lucide-react'
import { useEffect, useMemo, useReducer, useRef, useState, type ReactNode } from 'react'
import { CopyButton, Disclosure, FindingCard, isBug, KV, marksFor, MODEL_FACTS, riskLevel } from '../components/BugReport'
import CodeView from '../components/CodeView'
import LiveOutput from '../components/LiveOutput'
import Model3D from '../components/Model3D'
import ModelPicker from '../components/ModelPicker'
import { useHealth } from '../components/Navbar'
import { useToast } from '../components/Toast'
import { Badge, Button, Card, EngineBadge, Label, Spinner, inputCls } from '../components/ui'
import { AGENTS, agentLabel } from '../lib/agents'
import { api, streamRun, type BenchmarkProblem, type Finding, type ModelOption, type PipelineEvent, type ScanResult, type TestOutcome, type Variant } from '../lib/api'
import { demoTask, playDemo } from '../lib/demo'
import { diffHunks } from '../lib/diff'
import { fmtSeconds, useNow, useRunShortcut } from '../lib/hooks'
import { initialRun, reduce, type RunState, type Step } from '../lib/runState'

const PRESETS = {
  fast: { label: 'Fast', icon: Zap, hint: '≈ 30 s', maxFix: 1, nBase: 3, nTargeted: 3, nCand: 1 },
  balanced: { label: 'Balanced', icon: Scale, hint: '≈ 1 min', maxFix: 2, nBase: 4, nTargeted: 4, nCand: 3 },
  thorough: { label: 'Thorough', icon: Rocket, hint: '≈ 2 min', maxFix: 3, nBase: 5, nTargeted: 5, nCand: 4 },
} as const
type PresetKey = keyof typeof PRESETS
const DEFAULT_MODEL: ModelOption = { id: 'default', label: 'Qwen2.5-Coder 1.5B', detail: 'Fastest, default', params: '1.5B', size_gb: 1.0, default: true }

const DOING: Record<string, string> = {
  requirement: 'Reading your task and writing a clear specification',
  developer: 'Writing the first version of the code',
  tester_base: 'Writing test cases from the specification',
  bug_predictor: 'Checking the new code for likely bugs',
  tester_targeted: 'Adding extra edge-case tests for risky code',
  tester: 'Running the tests in a sandbox',
  fixer: 'Issues detected – attempting correction',
}

type Action = { kind: 'event'; ev: PipelineEvent; maxFix: number } | { kind: 'reset' }
const reducer = (s: RunState, a: Action) => (a.kind === 'reset' ? initialRun() : reduce(s, a.ev, a.maxFix))
const IDENT = /^[A-Za-z_][A-Za-z0-9_]*$/

export default function Playground() {
  const { health, offline } = useHealth(5000)
  const toast = useToast()
  const [state, dispatch] = useReducer(reducer, undefined, initialRun)
  // Inputs always start empty on a fresh page load (nothing is pre-filled or restored).
  const [source, setSource] = useState<'custom' | 'benchmark'>('custom')
  const [description, setDescription] = useState('')
  const [entry, setEntry] = useState('')
  const [tests, setTests] = useState<string[]>([])
  const [split, setSplit] = useState('mbpp_test')
  const [taskId, setTaskId] = useState('')
  const [preset, setPreset] = useState<PresetKey>('fast')
  const [compareBaseline, setCompareBaseline] = useState(false)
  const [randomAblation, setRandomAblation] = useState(false)
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [problems, setProblems] = useState<BenchmarkProblem[]>([])
  const [isDemo, setIsDemo] = useState(false)
  const [touched, setTouched] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [draftFindings, setDraftFindings] = useState<Finding[] | null>(null)
  const [finalScan, setFinalScan] = useState<ScanResult | null>(null)
  const [showLive, setShowLive] = useState(false)
  const [models, setModels] = useState<ModelOption[]>([DEFAULT_MODEL])
  const [model, setModel] = useState('default')  // every page load starts on the 1.5B default
  const abortRef = useRef<() => void>(() => {})
  const liveRef = useRef<HTMLDivElement>(null)
  const knobs = PRESETS[preset]

  useEffect(() => {
    if (source !== 'benchmark' || offline) return
    api.benchmarks(split, 80).then(ps => {
      setProblems(ps)
      setTaskId(t => (ps.some(p => p.task_id === t) ? t : ps[0]?.task_id ?? ''))
    }).catch(e => setError(String(e.message ?? e)))
  }, [source, split, offline])

  useEffect(() => {
    if (offline) return
    api.models().then(r => {
      setModels(r.models.length ? r.models : [DEFAULT_MODEL])
      setModel(m => (r.models.some(x => x.id === m) ? m : 'default'))
    }).catch(() => {})
  }, [offline])

  // ------------------------------------------------------------ validation
  const errors = useMemo(() => {
    const e: Record<string, string> = {}
    if (source === 'custom') {
      if (description.trim().length < 10) e.description = 'Describe the task in at least a sentence.'
      if (!entry.trim()) e.entry = 'Enter the function name.'
      else if (!IDENT.test(entry.trim())) e.entry = 'Use a valid Python name: letters, digits and _, not starting with a digit.'
      tests.forEach((t, i) => {
        if (t.trim() && !t.trim().startsWith('assert ')) e[`test${i}`] = 'A test case must start with "assert".'
        else if (t.trim() && entry.trim() && !t.includes(entry.trim())) e[`test${i}`] = `This test does not call ${entry.trim()}().`
      })
    } else if (!taskId) e.task = 'Pick a problem.'
    return e
  }, [source, description, entry, tests, taskId])
  const valid = Object.keys(errors).length === 0

  const running = state.status === 'running'
  const warming = !!health && health.warmup !== 'ready' && health.warmup !== 'idle' && health.warmup !== 'error'
  const variants: Variant[] = ['proposed', ...(compareBaseline ? ['baseline' as Variant] : []), ...(randomAblation ? ['random' as Variant] : [])]

  const scrollToLive = () => setTimeout(() => liveRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' }), 80)

  const start = async () => {
    setTouched(true)
    if (running || offline || !valid) return
    setError(null); setIsDemo(false); setDraftFindings(null); setFinalScan(null)
    dispatch({ kind: 'reset' })
    const ctrl = new AbortController()
    abortRef.current = () => ctrl.abort()
    const base = { variants, max_fix_iters: knobs.maxFix, n_base_tests: knobs.nBase, n_targeted_tests: knobs.nTargeted, n_candidates: knobs.nCand, model }
    const req = source === 'benchmark'
      ? { ...base, benchmark_split: split, task_id: taskId }
      : { ...base, description: description.trim(), entry_point: entry.trim(), public_tests: tests.map(t => t.trim()).filter(Boolean) }
    dispatch({ kind: 'event', ev: { type: 'status', message: 'Starting…' }, maxFix: knobs.maxFix })
    scrollToLive()
    try {
      await streamRun(req, ev => dispatch({ kind: 'event', ev, maxFix: knobs.maxFix }), ctrl.signal)
    } catch (e) {
      if ((e as Error).name === 'AbortError') dispatch({ kind: 'event', ev: { type: 'error', message: 'Stopped by you' }, maxFix: knobs.maxFix })
      else { setError((e as Error).message); dispatch({ kind: 'event', ev: { type: 'error', message: (e as Error).message }, maxFix: knobs.maxFix }) }
    }
  }

  // The demo replays a scripted run; it never touches the input fields.
  const demo = () => {
    abortRef.current()
    setError(null); setIsDemo(true); setDraftFindings(null); setFinalScan(null)
    dispatch({ kind: 'reset' })
    abortRef.current = playDemo(ev => dispatch({ kind: 'event', ev, maxFix: 1 }))
    scrollToLive()
  }
  const stop = () => { abortRef.current(); if (isDemo) dispatch({ kind: 'event', ev: { type: 'error', message: 'Demo stopped' }, maxFix: 1 }) }

  useRunShortcut(() => (running ? undefined : start()), !offline)

  // Exact checks on the Developer's first draft (real line numbers for the results section).
  const draft = state.shared.developer.value
  useEffect(() => {
    if (!draft || offline) return
    let alive = true
    api.scan(draft).then(r => { if (alive) setDraftFindings(r.findings) }).catch(() => {})
    return () => { alive = false }
  }, [draft, offline])

  // Exact checks on the final code once the run is done (highlights + bug status in Final results).
  const doneVariant = state.variants.includes('proposed') ? 'proposed' : state.variants[0]
  const doneState = doneVariant ? state.byVariant[doneVariant] : undefined
  const finalCode = state.status === 'done' && doneState ? doneState.result?.final_code ?? doneState.currentCode : undefined
  useEffect(() => {
    if (!finalCode || offline) return
    let alive = true
    api.scan(finalCode).then(r => { if (alive) setFinalScan(r) })
      .catch(() => { if (alive) setFinalScan({ findings: [], score: 0, threshold: 0.34, mode: 'model', flagged: false, functions: [], risky_functions: [], hot_lines: [] }) })
    return () => { alive = false }
  }, [finalCode, offline])

  // Toast when a run ends
  const prev = useRef(state.status)
  const mv: Variant | undefined = state.variants.includes('proposed') ? 'proposed' : state.variants[0]
  const vs = mv ? state.byVariant[mv] : undefined
  const lastTest = vs?.steps.filter(s => s.agent === 'tester' && s.status === 'done').at(-1)
  useEffect(() => {
    if (prev.current === 'running' && state.status === 'done') {
      const r = lastTest?.results ?? []
      const failed = r.filter(x => !x.ok).length
      toast(failed || lastTest?.loadError
        ? { tone: 'bad', title: 'Pipeline finished with failing tests', body: `${failed} of ${r.length} tests still fail.` }
        : { tone: 'good', title: 'Pipeline finished', body: r.length ? `All ${r.length} tests passed.` : undefined })
    } else if (prev.current === 'running' && state.status === 'error') {
      toast({ tone: 'bad', title: 'Run stopped', body: state.error })
    }
    prev.current = state.status
  }, [state.status, state.error, lastTest, toast])

  return (
    <div className="mx-auto max-w-7xl px-4 py-8 sm:px-6">
      <div className="relative grid items-center gap-4 md:grid-cols-[1.2fr_1fr]">
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.2em] text-brand-400">Agent Pipeline</p>
          <h1 className="mt-2 text-3xl font-bold tracking-tight text-ink-100 sm:text-5xl">Describe it. <span className="text-gradient">Five agents build it.</span></h1>
          <p className="mt-3 max-w-xl text-ink-300">
            The agents write a spec, write the code, check it for likely bugs, test it and fix it, and you can watch each step as it happens.
          </p>
        </div>
        <Model3D scene="coder" className="h-64 sm:h-72 lg:h-80" zoom={0.92} />
      </div>

      {offline && (
        <Card className="mt-6 flex items-start gap-3 p-4">
          <AlertTriangle className="mt-0.5 h-5 w-5 shrink-0 text-ink-100" />
          <p className="text-sm text-ink-300"><b className="text-ink-100">The agent server is not running.</b> Start it with <code className="font-mono">python -m uvicorn server.app:app --port 8000</code> in the <code className="font-mono">ZIPFORLAP</code> folder, or press <b>Play demo</b> to watch a scripted example.</p>
        </Card>
      )}
      {warming && !offline && (
        <Card className="mt-6 flex items-center gap-3 p-4"><Spinner /><p className="text-sm text-ink-300">Loading the AI models. You can start a run now; it begins as soon as they're ready.</p></Card>
      )}

      {/* ------------------------------------------------------------ inputs */}
      <Card className="mt-6 p-6">
        <div className="mb-5 inline-flex rounded-xl bg-ink-900 p-1 ring-1 ring-line/10">
          {(['custom', 'benchmark'] as const).map(s => (
            <button key={s} onClick={() => setSource(s)}
              className={clsx('relative rounded-lg px-4 py-1.5 text-sm font-medium transition-colors', source === s ? 'text-ink-100' : 'text-ink-400 hover:text-ink-200')}>
              {source === s && <motion.span layoutId="srcTab" className="absolute inset-0 rounded-lg bg-line/10" transition={{ type: 'spring', stiffness: 500, damping: 38 }} />}
              <span className="relative">{s === 'custom' ? 'Your task' : 'Benchmark problem'}</span>
            </button>
          ))}
        </div>

        {source === 'custom' ? (
          <div className="grid gap-5 lg:grid-cols-[1.4fr_1fr]">
            <div>
              <Label>Task description</Label>
              <textarea className={clsx(inputCls, 'min-h-36 resize-y')} value={description} onChange={e => setDescription(e.target.value)}
                placeholder="Describe the coding task you want the agent to solve…" />
              {touched && errors.description && <FieldError>{errors.description}</FieldError>}
            </div>
            <div className="space-y-5">
              <div>
                <Label>Function name</Label>
                <input className={clsx(inputCls, 'font-mono')} value={entry} onChange={e => setEntry(e.target.value)} placeholder="Enter the function name…" spellCheck={false} />
                {touched && errors.entry && <FieldError>{errors.entry}</FieldError>}
              </div>
              <div>
                <Label>Example test cases <span className="normal-case tracking-normal text-ink-400">(optional)</span></Label>
                <div className="space-y-2">
                  {tests.map((t, i) => (
                    <div key={i}>
                      <div className="flex gap-2">
                        <input className={clsx(inputCls, 'font-mono text-xs')} value={t} spellCheck={false}
                          placeholder={`assert ${entry.trim() || 'my_function'}(2, 3) == 5`}
                          onChange={e => setTests(ts => ts.map((x, j) => (j === i ? e.target.value : x)))} />
                        <button className="rounded-lg px-2 text-ink-400 transition hover:text-ink-100" onClick={() => setTests(ts => ts.filter((_, j) => j !== i))} aria-label="Delete test case"><Trash2 className="h-4 w-4" /></button>
                      </div>
                      {errors[`test${i}`] && t.trim() && <FieldError>{errors[`test${i}`]}</FieldError>}
                    </div>
                  ))}
                  <button className="flex items-center gap-1 text-sm font-medium text-brand-400 transition hover:text-brand-300" onClick={() => setTests(ts => [...ts, ''])}><Plus className="h-4 w-4" /> Add test case</button>
                  <p className="text-xs leading-relaxed text-ink-400">One Python <code className="font-mono">assert</code> per case, e.g. <code className="font-mono">assert add(2, 3) == 5</code>. Leave empty and the agents write their own tests.</p>
                </div>
              </div>
            </div>
          </div>
        ) : (
          <BenchmarkPicker split={split} setSplit={setSplit} taskId={taskId} setTaskId={setTaskId} problems={problems} />
        )}

        {/* advanced settings */}
        <button onClick={() => setShowAdvanced(a => !a)} className="mt-6 flex items-center gap-1 text-xs font-medium uppercase tracking-wider text-ink-400 transition hover:text-ink-200">
          <ChevronDown className={clsx('h-4 w-4 transition', showAdvanced && 'rotate-180')} /> Advanced settings
          <span className="ml-1 normal-case tracking-normal">({knobs.label}{compareBaseline ? ' · comparison on' : ''})</span>
        </button>
        <AnimatePresence initial={false}>
          {showAdvanced && (
            <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: 'auto', opacity: 1 }} exit={{ height: 0, opacity: 0 }} className="overflow-hidden">
              <div className="grid gap-5 pt-4 md:grid-cols-2">
                <div>
                  <Label>Speed</Label>
                  <div className="grid grid-cols-3 gap-2">
                    {(Object.keys(PRESETS) as PresetKey[]).map(k => {
                      const P = PRESETS[k]
                      return (
                        <button key={k} onClick={() => setPreset(k)}
                          className={clsx('rounded-xl p-3 text-left ring-1 transition', preset === k ? 'bg-brand-500/10 ring-brand-500/50' : 'ring-line/10 hover:bg-line/5')}>
                          <P.icon className="h-4 w-4 text-ink-300" />
                          <p className="mt-1 text-sm font-semibold text-ink-100">{P.label}</p>
                          <p className="text-[11px] text-ink-400">{P.hint} · up to {P.maxFix} fix{P.maxFix > 1 ? 'es' : ''}</p>
                        </button>
                      )
                    })}
                  </div>
                </div>
                <div className="space-y-2">
                  <Label>Research comparison</Label>
                  <Check on={compareBaseline} set={setCompareBaseline} title="Also run without the Bug Predictor" desc="Same draft, no bug check: shows what the Bug Predictor changes." />
                  <Check on={randomAblation} set={setRandomAblation} title="Random-flag control" desc="Bug checks replaced by random flags (ablation)." />
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        <div className="mt-6 flex flex-wrap items-center gap-3 border-t border-line/10 pt-5">
          <ModelPicker models={models} value={model} onChange={setModel} disabled={running || offline} />
          {touched && !valid && <p className="flex items-center gap-1.5 text-sm text-ink-100"><AlertTriangle className="h-4 w-4" /> Fill in the highlighted fields to run.</p>}
          <div className="ml-auto flex gap-2">
            <Button variant="outline" onClick={demo} disabled={running && !isDemo}><PlayCircle className="h-4 w-4" /> Play demo</Button>
            {running
              ? <Button variant="outline" onClick={stop}><Square className="h-4 w-4" /> Stop</Button>
              : <Button onClick={start} disabled={offline}><Play className="h-4 w-4" /> Run pipeline</Button>}
          </div>
        </div>
      </Card>

      {error && <Card className="mt-6 p-4 text-sm text-ink-100">{error}</Card>}

      {/* ------------------------------------------------------------ run */}
      <div ref={liveRef} className="scroll-mt-20" />
      <AnimatePresence>
        {state.status !== 'idle' && (
          <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} className="mt-10 space-y-6">
            {isDemo && (
              <Card className="flex flex-wrap items-center gap-3 p-4">
                <Badge tone="warn">DEMO</Badge>
                <p className="text-sm text-ink-300">Scripted example, not a real run. Task: <i className="text-ink-100">{demoTask.description}</i></p>
              </Card>
            )}
            <Progress state={state} isDemo={isDemo} />
            <Stepper state={state} mv={mv} />
            {running && (
              <button onClick={() => setShowLive(v => !v)} className="flex items-center gap-1.5 text-sm font-medium text-ink-300 hover:text-ink-100">
                <ChevronDown className={clsx('h-4 w-4 transition', showLive && 'rotate-180')} /> {showLive ? 'Hide' : 'Show'} what the agent is writing
              </button>
            )}
            <AnimatePresence mode="popLayout">{running && showLive && state.live && <LiveOutput key="live" live={state.live} />}</AnimatePresence>
            {state.status === 'done' && mv && <FinalResults state={state} mv={mv} draftFindings={draftFindings} finalScan={finalScan} />}
            <Disclosure title="Step-by-step agent output" icon={<Code2 className="h-4 w-4 text-ink-400" />}>
              <AgentOutputs state={state} mv={mv} draftFindings={draftFindings} />
            </Disclosure>
            {(state.status === 'done' || state.status === 'error') && <Logs state={state} mv={mv} />}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

function FieldError({ children }: { children: ReactNode }) {
  return <p className="mt-1 flex items-center gap-1 text-xs font-medium text-ink-100"><AlertTriangle className="h-3.5 w-3.5" /> {children}</p>
}

function Check({ on, set, title, desc }: { on: boolean; set: (b: boolean) => void; title: string; desc: string }) {
  return (
    <button onClick={() => set(!on)} className={clsx('flex w-full items-start gap-3 rounded-xl p-3 text-left ring-1 transition', on ? 'bg-brand-500/10 ring-brand-500/40' : 'ring-line/10 hover:bg-line/5')}>
      <span className={clsx('mt-0.5 grid h-5 w-5 shrink-0 place-items-center rounded-md ring-1', on ? 'bg-brand-500 ring-brand-500' : 'ring-line/25')}>
        {on && <CheckCircle2 className="h-3.5 w-3.5 text-on-brand" />}
      </span>
      <span><span className="block text-sm font-semibold text-ink-100">{title}</span><span className="text-xs text-ink-400">{desc}</span></span>
    </button>
  )
}

function BenchmarkPicker({ split, setSplit, taskId, setTaskId, problems }: {
  split: string; setSplit: (s: string) => void; taskId: string; setTaskId: (s: string) => void; problems: BenchmarkProblem[]
}) {
  const selected = problems.find(p => p.task_id === taskId)
  return (
    <div className="space-y-4">
      <div className="grid gap-4 sm:grid-cols-[200px_1fr]">
        <div>
          <Label>Benchmark</Label>
          <select className={inputCls} value={split} onChange={e => setSplit(e.target.value)}>
            <option value="mbpp_test">MBPP (601–974)</option>
            <option value="mbpp_val">MBPP (511–600)</option>
            <option value="humaneval">HumanEval</option>
          </select>
        </div>
        <div>
          <Label>Problem</Label>
          <select className={inputCls} value={taskId} onChange={e => setTaskId(e.target.value)}>
            {problems.map(p => <option key={p.task_id} value={p.task_id}>{p.task_id}: {p.entry_point}</option>)}
          </select>
        </div>
      </div>
      {selected && (
        <div className="rounded-xl bg-ink-900/70 p-4 ring-1 ring-line/10">
          <pre className="max-h-40 overflow-auto whitespace-pre-wrap font-mono text-xs leading-relaxed text-ink-300">{selected.description}</pre>
          <p className="mt-3 text-xs text-ink-400">The benchmark's official hidden tests are used only to judge the final code.</p>
        </div>
      )}
    </div>
  )
}

// ------------------------------------------------------------------ live progress

function activeAgent(state: RunState): string | undefined {
  if (state.live) return state.live.agent
  for (const v of state.variants) {
    const step = state.byVariant[v]?.steps.findLast(s => s.status === 'running')
    if (step) return step.agent
  }
  for (const a of ['requirement', 'developer', 'tester_base'] as const) if (state.shared[a].status === 'running') return a
  return undefined
}

function Progress({ state, isDemo }: { state: RunState; isDemo: boolean }) {
  const running = state.status === 'running'
  const now = useNow(running, 250)
  const elapsed = state.startedAt ? ((state.finishedAt ?? now) - state.startedAt) / 1000 : 0
  const pct = state.status === 'done' ? 100 : Math.max(3, Math.round((state.stepsDone / state.stepsTotal) * 100))
  const agent = activeAgent(state)
  return (
    <Card className="overflow-hidden">
      <div className="h-1.5 bg-ink-800">
        <motion.div className="h-full bg-brand-500" animate={{ width: `${pct}%` }} transition={{ type: 'spring', stiffness: 80, damping: 20 }} />
      </div>
      <div className="flex flex-wrap items-center gap-3 p-4">
        {running && <><Spinner /><span className="text-sm font-medium text-ink-100">{state.message ?? (agent ? DOING[agent] : 'Working')}…</span></>}
        {state.status === 'done' && <span className="flex items-center gap-2 text-sm font-semibold text-ink-100"><CheckCircle2 className="h-4 w-4" /> Pipeline finished</span>}
        {state.status === 'error' && <span className="flex items-center gap-2 text-sm font-semibold text-ink-100"><XCircle className="h-4 w-4" /> {state.error}</span>}
        {isDemo && <Badge tone="warn">Demo</Badge>}
        <span className="ml-auto flex items-center gap-3 font-mono text-sm tabular-nums text-ink-300">
          {state.llm && <span className="hidden font-sans text-xs sm:inline"><EngineBadge name={state.llm} /></span>}
          {state.tokens > 0 && <span className="text-xs text-ink-400" title="Tokens generated by the LLM in this run">{state.tokens.toLocaleString()} tokens</span>}
          <span>{pct}%</span><span className="flex items-center gap-1"><Clock className="h-4 w-4 text-ink-400" />{fmtSeconds(elapsed)}</span>
        </span>
      </div>
    </Card>
  )
}

type CardStatus = 'pending' | 'running' | 'done' | 'failed' | 'skipped'

function agentStatuses(state: RunState, mv?: Variant): Record<string, { status: CardStatus; note?: string }> {
  const vs = mv ? state.byVariant[mv] : undefined
  const steps = vs?.steps ?? []
  const has = (agent: string, st?: 'running' | 'done') => steps.some(s => s.agent.startsWith(agent) && (!st || s.status === st))
  const variantDone = vs?.status === 'done'
  const sh = (k: 'requirement' | 'developer' | 'tester_base'): CardStatus =>
    state.shared[k].status === 'running' ? 'running' : state.shared[k].status === 'done' ? 'done' : 'pending'
  const tests = steps.filter(s => s.agent === 'tester' && s.status === 'done').at(-1)?.results ?? []
  const passed = tests.filter(t => t.ok).length
  const fixes = steps.filter(s => s.agent === 'fixer' && s.status === 'done').length
  const rejected = steps.filter(s => s.agent === 'fixer' && s.rejected).length
  const dropped = state.shared.tester_base.dropped ?? 0
  const out: Record<string, { status: CardStatus; note?: string }> = {
    requirement: { status: sh('requirement') },
    developer: { status: sh('developer') },
    bug_predictor: mv === 'baseline' ? { status: 'skipped', note: 'Not used' }
      : { status: has('bug_predictor', 'running') ? 'running' : has('bug_predictor', 'done') ? 'done' : 'pending' },
    tester: {
      status: sh('tester_base') === 'running' || has('tester', 'running') ? 'running' : variantDone ? 'done' : (sh('tester_base') === 'done' || has('tester')) ? 'running' : 'pending',
      note: tests.length ? `${passed}/${tests.length} passed${dropped ? ` · ${dropped} wrong test${dropped > 1 ? 's' : ''} removed` : ''}` : undefined,
    },
    fixer: {
      status: has('fixer', 'running') ? 'running' : variantDone ? (fixes ? 'done' : 'skipped') : fixes ? 'done' : 'pending',
      note: fixes ? `${fixes} attempt${fixes > 1 ? 's' : ''}${rejected ? ` · ${rejected} rejected (no improvement)` : ''}` : variantDone ? 'Not needed' : undefined,
    },
  }
  if (state.status === 'error') for (const k of Object.keys(out)) if (out[k].status === 'running') out[k] = { status: 'failed', note: 'Stopped' }
  return out
}

const STATUS_UI: Record<CardStatus, { label: string; icon: typeof CheckCircle2 }> = {
  pending: { label: 'Pending', icon: CircleDashed },
  running: { label: 'Running', icon: Loader2 },
  done: { label: 'Completed', icon: CheckCircle2 },
  failed: { label: 'Failed', icon: XCircle },
  skipped: { label: 'Skipped', icon: MinusCircle },
}
const SHORT: Record<string, string> = {
  requirement: 'Turns your task into a precise spec', developer: 'Writes the code', bug_predictor: 'Checks the code for likely bugs',
  tester: 'Writes and runs tests', fixer: 'Repairs failing code',
}

function Stepper({ state, mv }: { state: RunState; mv?: Variant }) {
  const st = agentStatuses(state, mv)
  return (
    <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-5">
      {AGENTS.map((a, i) => {
        const s = st[a.key] ?? { status: 'pending' }
        const ui = STATUS_UI[s.status]
        const Icon = a.icon
        const active = s.status === 'running'
        return (
          <motion.div key={a.key} layout
            className={clsx('relative rounded-2xl p-4 ring-1 transition-all', active ? 'glass ring-2 ring-brand-500 shadow-lg' : 'glass ring-line/5',
              (s.status === 'pending' || s.status === 'skipped') && 'opacity-55')}>
            {active && <motion.span layoutId="activeGlow" className="absolute inset-0 -z-10 rounded-2xl bg-brand-500/10" />}
            <div className="flex items-center gap-3">
              <div className={clsx('grid h-10 w-10 shrink-0 place-items-center rounded-xl', s.status === 'done' || active ? 'bg-brand-500 text-on-brand' : 'bg-line/[0.06] text-ink-300')}>
                <Icon className="h-5 w-5" />
              </div>
              <div className="min-w-0">
                <p className="text-[10px] font-semibold uppercase tracking-widest text-ink-400">Step {i + 1}</p>
                <p className="truncate font-semibold text-ink-100">{a.name}</p>
              </div>
            </div>
            <p className="mt-2 text-xs text-ink-400">{SHORT[a.key]}</p>
            <p className="mt-2 flex items-center gap-1.5 text-xs font-semibold text-ink-100">
              <ui.icon className={clsx('h-3.5 w-3.5', active && 'animate-spin')} /> {ui.label}{s.note ? ` · ${s.note}` : ''}
            </p>
          </motion.div>
        )
      })}
    </div>
  )
}

// ------------------------------------------------------------------ per-agent outputs

function Section({ icon, title, status, children }: { icon: ReactNode; title: string; status?: ReactNode; children: ReactNode }) {
  return (
    <Card className="p-5">
      <div className="mb-3 flex flex-wrap items-center gap-2">
        <span className="grid h-8 w-8 place-items-center rounded-lg bg-line/[0.06] text-ink-100">{icon}</span>
        <h3 className="font-semibold text-ink-100">{title}</h3>
        <div className="ml-auto">{status}</div>
      </div>
      {children}
    </Card>
  )
}

/** The model sometimes puts raw line breaks inside JSON strings; the backend then keeps the text as-is. Parse it leniently. */
function lenientSpec(spec: Record<string, unknown>): Record<string, unknown> {
  if (spec.inputs || spec.output || typeof spec.description !== 'string') return spec
  const m = spec.description.match(/\{[\s\S]*\}/)
  if (!m) return spec
  try { return { ...JSON.parse(m[0].replace(/\r?\n/g, ' ')), function_name: spec.function_name } } catch { return spec }
}

function SpecView({ spec: raw }: { spec: Record<string, unknown> }) {
  const spec = lenientSpec(raw)
  const str = (v: unknown) => (typeof v === 'string' ? v : v == null ? '' : JSON.stringify(v))
  const inputs = Array.isArray(spec.inputs) ? spec.inputs as Record<string, unknown>[] : []
  const output = (spec.output && typeof spec.output === 'object' ? spec.output : null) as Record<string, unknown> | null
  const behaviour = Array.isArray(spec.behaviour) ? (spec.behaviour as unknown[]).map(str) : spec.behaviour ? [str(spec.behaviour)] : []
  const edges = Array.isArray(spec.edge_cases) ? (spec.edge_cases as unknown[]).map(str) : []
  if (!inputs.length && !output && !behaviour.length && !edges.length) {
    return <pre className="max-h-48 overflow-auto whitespace-pre-wrap font-mono text-xs text-ink-300">{str(spec.description ?? spec)}</pre>
  }
  return (
    <div className="grid gap-4 text-sm md:grid-cols-2">
      <div className="space-y-2">
        {spec.signature != null && <p><span className="text-ink-400">Signature: </span><code className="font-mono text-ink-100">{str(spec.signature)}</code></p>}
        {inputs.length > 0 && (
          <div><p className="text-ink-400">Inputs</p>
            <ul className="mt-1 space-y-0.5">{inputs.map((x, i) => <li key={i} className="text-ink-200"><code className="font-mono text-ink-100">{str(x.name)}</code>{x.type ? <span className="text-ink-400"> ({str(x.type)})</span> : null}{x.description ? `: ${str(x.description)}` : ''}</li>)}</ul>
          </div>
        )}
        {output && <p><span className="text-ink-400">Returns: </span><span className="text-ink-200">{str(output.type)}{output.description ? `: ${str(output.description)}` : ''}</span></p>}
      </div>
      <div className="space-y-2">
        {behaviour.length > 0 && <div><p className="text-ink-400">Expected behaviour</p><p className="mt-1 text-ink-200">{behaviour.join(' ')}</p></div>}
        {edges.length > 0 && <div><p className="text-ink-400">Edge cases to handle</p><ul className="mt-1 list-disc space-y-0.5 pl-5 text-ink-200">{edges.map((e, i) => <li key={i}>{e}</li>)}</ul></div>}
      </div>
    </div>
  )
}

function TestList({ results, max = 12, yours = [] }: { results: TestOutcome[]; max?: number; yours?: string[] }) {
  return (
    <ul className="space-y-1">
      {results.slice(0, max).map((r, i) => (
        <li key={i} className="flex items-start gap-2 rounded-lg bg-ink-900/60 px-3 py-1.5 font-mono text-xs">
          {r.ok ? <CheckCircle2 className="mt-0.5 h-3.5 w-3.5 shrink-0 text-ink-400" /> : <XCircle className="mt-0.5 h-3.5 w-3.5 shrink-0 text-ink-100" />}
          <span className="min-w-0 flex-1">
            <span className={clsx('break-all', r.ok ? 'text-ink-300' : 'text-ink-100')}>{r.test}</span>
            {r.targeted && <span className="ml-2 rounded bg-line/10 px-1.5 py-px font-sans text-[10px] text-ink-300">edge-case test</span>}
            {yours.length > 0 && <span className="ml-2 rounded bg-line/10 px-1.5 py-px font-sans text-[10px] text-ink-300">{yours.includes(r.test.trim()) ? 'your test' : 'written by the agents'}</span>}
            {!r.ok && r.error && <span className="block break-all font-sans text-[11px] text-ink-400">→ {r.error}</span>}
          </span>
        </li>
      ))}
      {results.length > max && <li className="px-3 text-xs text-ink-400">+ {results.length - max} more</li>}
    </ul>
  )
}

function testCounts(step?: Step) {
  const r = step?.results ?? []
  const passed = r.filter(x => x.ok).length
  return { total: r.length, passed, failed: r.length - passed, loadError: step?.loadError }
}

function AgentOutputs({ state, mv, draftFindings }: { state: RunState; mv?: Variant; draftFindings: Finding[] | null }) {
  const vs = mv ? state.byVariant[mv] : undefined
  const steps = vs?.steps ?? []
  const spec = state.shared.requirement.value
  const draft = state.shared.developer.value
  const bug = steps.find(s => s.agent === 'bug_predictor' && s.status === 'done' && !s.iter)
  const testSteps = steps.filter(s => s.agent === 'tester' && s.status === 'done')
  const firstTest = testSteps[0]
  const lastTest = testSteps.at(-1)
  const fixSteps = steps.filter(s => s.agent === 'fixer' && s.status === 'done')
  const targeted = steps.find(s => s.agent === 'tester_targeted' && s.status === 'done')?.tests ?? []
  const draftBugs = (draftFindings ?? []).filter(isBug)

  return (
    <div className="space-y-4">
      {spec && (
        <Section icon={<FileText className="h-4 w-4" />} title="Requirements" status={<Done />}><SpecView spec={spec} /></Section>
      )}
      {draft !== undefined && (
        <Section icon={<Code2 className="h-4 w-4" />} title="Developer: first draft" status={<Done />}>
          {draft.trim()
            ? <CodeView code={draft} height={Math.min(380, 40 + draft.split('\n').length * 19)} marks={marksFor(draftBugs)} />
            : <p className="text-sm text-ink-300">The Developer returned no code. Try a shorter task description.</p>}
        </Section>
      )}
      {bug?.risk && draft && (
        <Section icon={<Bug className="h-4 w-4" />} title="Bug Predictor"
          status={<span className="text-sm font-semibold text-ink-100">{bug.risk.flagged || draftBugs.length ? 'Potential bugs detected' : 'No potential bugs detected'}</span>}>
          <div className="space-y-3 text-sm text-ink-300">
            {draftBugs.length > 0
              ? <ul className="space-y-3">{draftBugs.map((f, i) => <FindingCard key={i} f={f} index={i} />)}</ul>
              : <p>No exact problems found in the draft{bug.risk.flagged ? ', but the bug-prediction model rates it as high risk.' : '.'}</p>}
            {bug.risk.flagged && (
              <p>Because the risk is high, the Tester adds {targeted.length || 'extra'} edge-case tests and runs them first.</p>
            )}
            <Disclosure title="Advanced Model Details">
              <KV k="Bug probability (whole draft)" v={<span className="font-mono">{(bug.risk.score * 100).toFixed(1)}%</span>} />
              <KV k="Pipeline threshold" v={<span className="font-mono">{bug.risk.threshold.toFixed(3)}</span>} />
              <KV k="Risk level" v={riskLevel(bug.risk.score)} />
              <KV k="Model" v={MODEL_FACTS.name} />
              <KV k="MBPP held-out benchmark" v={MODEL_FACTS.mbpp} />
              {bug.risk.hot_lines.length > 0 && (
                <div className="mt-3">
                  <p className="text-xs text-ink-400">Lines the model paid most attention to (what influenced the score, not confirmed bug locations):</p>
                  <ul className="mt-1 space-y-1">{bug.risk.hot_lines.map(h => <li key={h.line_start} className="truncate rounded bg-blood-night px-2 py-1 font-mono text-xs text-vanilla">L{h.line_start} {h.text}</li>)}</ul>
                </div>
              )}
            </Disclosure>
          </div>
        </Section>
      )}
      {firstTest && (
        <Section icon={<FlaskConical className="h-4 w-4" />} title="Tester: first run" status={<TestBadge {...testCounts(firstTest)} />}>
          {firstTest.loadError && <p className="mb-2 text-sm text-ink-100">The code failed to load: {firstTest.loadError}</p>}
          <TestList results={firstTest.results ?? []} />
        </Section>
      )}
      {fixSteps.map((f, i) => {
        const before = i === 0 ? draft ?? '' : fixSteps[i - 1].code ?? ''
        const failing = (testSteps[i]?.results ?? []).filter(r => !r.ok)
        const after = testSteps[i + 1]
        const hunks = diffHunks(before, f.code ?? '')
        return (
          <Section key={i} icon={<Wrench className="h-4 w-4" />} title={`Fixer: correction attempt ${i + 1}`}
            status={after ? <TestBadge {...testCounts(after)} /> : <span className="text-xs text-ink-400">testing…</span>}>
            <div className="grid gap-4 lg:grid-cols-2">
              <div className="space-y-3 text-sm">
                <div>
                  <p className="font-semibold text-ink-100">Issue identified</p>
                  <ul className="mt-1 space-y-1 text-ink-300">
                    {testSteps[i]?.loadError && <li>The code failed to load: {testSteps[i]?.loadError}</li>}
                    {failing.slice(0, 4).map((r, k) => <li key={k} className="break-all font-mono text-xs">✗ {r.test}<span className="block font-sans text-ink-400">→ {r.error || 'failed'}</span></li>)}
                  </ul>
                </div>
                <div>
                  <p className="font-semibold text-ink-100">Correction applied</p>
                  {hunks.length === 0 ? <p className="text-ink-400">No change to the code.</p> : (
                    <ul className="mt-1 space-y-2">
                      {hunks.slice(0, 4).map((h, k) => (
                        <li key={k} className="overflow-hidden rounded-lg font-mono text-xs">
                          <p className="bg-line/[0.06] px-2 py-0.5 font-sans text-[11px] text-ink-400">around line {h.oldStart}</p>
                          {h.before.map((l, j) => <p key={`b${j}`} className="whitespace-pre bg-blood-night px-2 text-vanilla/50 line-through">- {l}</p>)}
                          {h.after.map((l, j) => <p key={`a${j}`} className="whitespace-pre bg-blood-night px-2 text-vanilla">+ {l}</p>)}
                        </li>
                      ))}
                    </ul>
                  )}
                </div>
                {after && <p className="font-semibold text-ink-100">{testCounts(after).failed || after.loadError ? `Still failing: ${testCounts(after).failed} test(s)` : 'All tests pass after this fix'}</p>}
              </div>
              {f.code && <CodeView code={f.code} height={Math.min(340, 40 + f.code.split('\n').length * 19)} />}
            </div>
          </Section>
        )
      })}
      {lastTest && lastTest !== firstTest && vs?.status === 'done' && (
        <Section icon={<FlaskConical className="h-4 w-4" />} title="Tester: final run" status={<TestBadge {...testCounts(lastTest)} />}>
          <TestList results={lastTest.results ?? []} />
        </Section>
      )}
    </div>
  )
}

function Done() {
  return <span className="flex items-center gap-1 text-xs font-semibold text-ink-300"><CheckCircle2 className="h-3.5 w-3.5" /> Completed</span>
}

function TestBadge({ total, passed, failed, loadError }: { total: number; passed: number; failed: number; loadError?: string }) {
  if (loadError) return <span className="sev sev-critical">Code failed to load</span>
  if (!total) return <span className="text-xs text-ink-400">No tests</span>
  return <span className={clsx('sev', failed ? 'sev-high' : 'sev-low')}>{passed}/{total} passed{failed ? ` · ${failed} failed` : ''}</span>
}

// ------------------------------------------------------------------ final results

function FinalResults({ state, mv, draftFindings, finalScan }: {
  state: RunState; mv: Variant; draftFindings: Finding[] | null; finalScan: ScanResult | null
}) {
  const vs = state.byVariant[mv]
  if (!vs) return null
  const steps = vs.steps
  const finalCode = vs.result?.final_code ?? vs.currentCode ?? state.shared.developer.value ?? ''
  const draft = state.shared.developer.value ?? ''
  const testSteps = steps.filter(s => s.agent === 'tester' && s.status === 'done')
  const last = testCounts(testSteps.at(-1))
  const lastResults = testSteps.at(-1)?.results ?? []
  const allPassed = last.total > 0 && last.failed === 0 && !last.loadError
  const fixes = steps.filter(s => s.agent === 'fixer' && s.status === 'done').length
  const draftBugs = (draftFindings ?? []).filter(isBug)
  const finalBugs = (finalScan?.findings ?? []).filter(isBug)
  const bug = steps.find(s => s.agent === 'bug_predictor' && s.status === 'done' && !s.iter)
  const hidden = vs.result?.final_hidden_pass
  const hunks = diffHunks(draft, finalCode)
  const failing = lastResults.filter(r => !r.ok)
  const firstFails = (testSteps[0]?.results ?? []).filter(r => !r.ok)
  const fixedCount = firstFails.length - failing.length

  // Bug detection status, from real results only: exact checks on the final code + the final test run.
  const status = finalScan == null ? 'checking'
    : finalBugs.length || failing.length || last.loadError ? 'issues' : 'clean'

  return (
    <section className="space-y-4">
      <h2 className="flex items-center gap-2 pt-4 text-2xl font-bold text-ink-100"><ListTree className="h-6 w-6" /> Final results</h2>

      {/* overall verdict: never shows success while tests fail */}
      <Card className={clsx('flex flex-wrap items-center gap-4 p-5 ring-2', allPassed ? 'ring-brand-500/40' : 'ring-line/40')}>
        {allPassed ? <CheckCircle2 className="h-8 w-8 shrink-0 text-ink-100" /> : <AlertTriangle className="h-8 w-8 shrink-0 text-ink-100" />}
        <div>
          <p className="text-lg font-bold text-ink-100">
            {last.loadError ? 'The final code does not load' : !last.total ? 'No tests were available to check the code'
              : allPassed ? `All ${last.total} tests passed` : `Tests failing: ${last.failed} of ${last.total}`}
          </p>
          <p className="text-sm text-ink-400">
            {fixes ? `${fixes} correction attempt${fixes > 1 ? 's' : ''}` : 'No corrections were needed'}
            {hidden != null && ` · official benchmark tests: ${hidden ? 'passed' : 'failed'}`}
          </p>
        </div>
      </Card>

      <div className="grid gap-4 lg:grid-cols-[1.3fr_1fr]">
        {/* generated code, problem lines highlighted when the location is exact */}
        <Card className="p-5">
          <div className="mb-3 flex items-center gap-2">
            <h3 className="font-semibold text-ink-100">Generated code</h3>
            <div className="ml-auto"><CopyButton text={finalCode} /></div>
          </div>
          <CodeView code={finalCode} marks={marksFor(finalBugs)} height={Math.min(460, 40 + finalCode.split('\n').length * 19)} />
          {finalBugs.length > 0 && <p className="mt-2 text-xs text-ink-400">Highlighted lines have issues found by exact checks.</p>}
        </Card>

        <div className="space-y-4">
          {/* bug detection status */}
          <Card className="p-5">
            <h3 className="font-semibold text-ink-100">Bug detection status</h3>
            {status === 'checking' ? (
              <p className="mt-2 flex items-center gap-2 text-sm text-ink-300"><Spinner /> Checking the final code…</p>
            ) : status === 'clean' ? (
              <>
                <p className="mt-2 flex items-center gap-2 font-semibold text-ink-100"><CheckCircle2 className="h-4 w-4" /> No potential bugs were identified</p>
                <p className="mt-1 text-sm text-ink-400">The checks and tests found nothing wrong. This is not a guarantee that the code is bug-free.</p>
              </>
            ) : (
              <>
                <p className="mt-2 flex items-center gap-2 font-semibold text-ink-100"><Bug className="h-4 w-4" /> Potential bugs detected</p>
                <ul className="mt-1 space-y-0.5 text-sm text-ink-300">
                  {finalBugs.length > 0 && <li>{finalBugs.length} issue{finalBugs.length > 1 ? 's' : ''} found in the code</li>}
                  {failing.length > 0 && <li>{failing.length} test{failing.length > 1 ? 's' : ''} still failing</li>}
                  {last.loadError && <li>The code fails to load</li>}
                </ul>
              </>
            )}
            {(draftBugs.length > 0 || firstFails.length > 0 || bug?.risk?.flagged) && (
              <p className="mt-3 border-t border-line/10 pt-3 text-sm text-ink-400">
                First draft: {[
                  bug?.risk?.flagged && 'flagged as high risk',
                  draftBugs.length > 0 && `${draftBugs.length} issue${draftBugs.length > 1 ? 's' : ''} found`,
                  firstFails.length > 0 && `${firstFails.length} test${firstFails.length > 1 ? 's' : ''} failed`,
                ].filter(Boolean).join(' · ')}{fixes > 0 && fixedCount > 0 ? ` → ${fixedCount} fixed by the Fixer` : ''}
              </p>
            )}
          </Card>

          {/* test execution summary */}
          <Card className="p-5">
            <h3 className="font-semibold text-ink-100">Test execution summary</h3>
            <div className="mt-3 grid grid-cols-3 gap-2 text-center">
              {[['Executed', last.total], ['Passed', last.passed], ['Failed', last.failed]].map(([k, v]) => (
                <div key={k as string} className="rounded-xl bg-line/[0.04] p-3 ring-1 ring-line/10">
                  <p className="text-2xl font-bold text-ink-100">{v as number}</p><p className="text-xs text-ink-400">{k as string}</p>
                </div>
              ))}
            </div>
            {failing.length > 0 && (
              <div className="mt-3">
                <TestList results={failing} max={5} yours={(state.publicTests ?? []).map(t => t.trim())} />
                {(state.publicTests ?? []).length > 0 && failing.every(r => !(state.publicTests ?? []).map(t => t.trim()).includes(r.test.trim())) && (
                  <p className="mt-2 text-xs text-ink-400">All of your own tests pass; only tests written by the agents fail. Check whether those tests are correct.</p>
                )}
              </div>
            )}
          </Card>
        </div>
      </div>

      {/* locations and explanations: exact locations only */}
      {finalBugs.length > 0 && (
        <Card className="p-5">
          <h3 className="font-semibold text-ink-100">Detected bug locations</h3>
          <ul className="mt-3 space-y-3">{finalBugs.map((f, i) => <FindingCard key={i} f={f} index={i} />)}</ul>
        </Card>
      )}
      {finalBugs.length === 0 && failing.length > 0 && (
        <Card className="p-5">
          <h3 className="font-semibold text-ink-100">Bug explanation</h3>
          <p className="mt-1 text-sm text-ink-300">
            The failing tests above show inputs where the code gives a wrong result. Test failures don't point to an exact line,
            so no line is highlighted. Paste the code into the <b>Bug Scanner</b> for a line-by-line review.
          </p>
        </Card>
      )}

      {/* suggested fix: only the final recommended correction */}
      <Card className="p-5">
        <h3 className="font-semibold text-ink-100">Suggested fix</h3>
        {hunks.length > 0 ? (
          <>
            <p className="mt-1 text-sm text-ink-400">The final correction, compared with the first draft:</p>
            <ul className="mt-2 space-y-2">
              {hunks.slice(0, 6).map((h, k) => (
                <li key={k} className="overflow-hidden rounded-lg font-mono text-xs">
                  <p className="bg-line/[0.06] px-2 py-0.5 font-sans text-[11px] text-ink-400">around line {h.oldStart}</p>
                  {h.before.map((l, j) => <p key={`b${j}`} className="whitespace-pre bg-blood-night px-2 text-vanilla/50 line-through">- {l}</p>)}
                  {h.after.map((l, j) => <p key={`a${j}`} className="whitespace-pre bg-blood-night px-2 text-vanilla">+ {l}</p>)}
                </li>
              ))}
            </ul>
          </>
        ) : <p className="mt-1 text-sm text-ink-300">{fixes > 0
          ? `The Fixer made ${fixes} attempt${fixes > 1 ? 's' : ''} but kept the code unchanged, so the first draft is the final code.`
          : 'No correction was needed; the first draft is the final code.'}</p>}
        {finalBugs.length > 0 && (
          <ul className="mt-3 list-disc space-y-1 pl-5 text-sm text-ink-300">
            {finalBugs.map((f, i) => <li key={i}><span className="font-mono text-ink-100">L{f.line}</span>: {f.suggestion}</li>)}
          </ul>
        )}
        {!allPassed && last.total > 0 && (
          <p className="mt-3 text-sm text-ink-300">The agents could not make every test pass. Generated tests can themselves be wrong, so compare the failing tests with your task before changing the code.</p>
        )}
      </Card>

      <Disclosure title="Advanced Model Details">
        <div className="grid gap-x-8 md:grid-cols-2">
          <div>
            {bug?.risk && <KV k="Bug probability, first draft (whole snippet)" v={<span className="font-mono">{(bug.risk.score * 100).toFixed(1)}%</span>} />}
            {finalScan && <KV k="Bug probability, final code (whole snippet)" v={<span className="font-mono">{(finalScan.score * 100).toFixed(1)}%</span>} />}
            {bug?.risk && <KV k="Pipeline threshold (extra tests above this)" v={<span className="font-mono">{bug.risk.threshold.toFixed(3)}</span>} />}
            {bug?.risk && <KV k="First-draft risk level" v={riskLevel(bug.risk.score)} />}
          </div>
          <div>
            <KV k="Model" v={MODEL_FACTS.name} />
            <KV k="Version" v={MODEL_FACTS.version} />
            <KV k="MBPP held-out benchmark" v={MODEL_FACTS.mbpp} />
            <KV k="HumanEval benchmark" v={MODEL_FACTS.humaneval} />
          </div>
        </div>
        <p className="mt-3 text-xs leading-relaxed text-ink-400">
          The bug probability scores the whole code snippet; it is not the chance that any particular line is wrong. Highlighted
          lines come only from exact checks (parser, syntax tree, running the code).
        </p>
      </Disclosure>
    </section>
  )
}

function Logs({ state, mv }: { state: RunState; mv?: Variant }) {
  const rows: { who: string; what: string; secs?: number }[] = []
  ;(['requirement', 'developer', 'tester_base'] as const).forEach(k => {
    const s = state.shared[k]
    if (s.status !== 'idle') rows.push({ who: agentLabel[k], what: s.status, secs: s.seconds })
  })
  state.variants.forEach(v => (state.byVariant[v]?.steps ?? []).forEach(s => rows.push({
    who: `${agentLabel[s.agent]}${s.iter ? ` · iteration ${s.iter}` : ''}`, what: `${v} · ${s.status}`, secs: s.seconds,
  })))
  const r = mv ? state.byVariant[mv]?.result : undefined
  return (
    <div className="space-y-3">
      <Disclosure title="Execution Logs" icon={<ListTree className="h-4 w-4 text-ink-400" />}>
        <ul className="max-h-72 space-y-0.5 overflow-auto font-mono text-xs">
          {rows.map((x, i) => (
            <li key={i} className="flex gap-3 border-b border-line/5 py-1 text-ink-300">
              <span className="w-6 text-ink-400">{i + 1}</span><span className="flex-1">{x.who}</span><span className="text-ink-400">{x.what}</span>
              <span className="w-14 text-right tabular-nums">{x.secs != null ? `${x.secs}s` : ''}</span>
            </li>
          ))}
        </ul>
        {state.warnings.map((w, i) => <p key={i} className="mt-2 text-xs text-ink-100">⚠ {w}</p>)}
        {r && (
          <div className="mt-4 grid gap-x-8 md:grid-cols-2">
            <KV k="Fix iterations" v={r.fix_iterations} />
            <KV k="Edge-case tests added" v={r.n_targeted_tests} />
            <KV k="Test executions" v={r.test_executions} />
            <KV k="LLM calls (this variant)" v={r.variant_llm.llm_calls} />
            <KV k="Generated tokens" v={r.variant_llm.completion_tokens} />
            <KV k="LLM time" v={`${r.variant_llm.llm_seconds}s`} />
            {state.llm && <KV k="Language model" v={state.llm} />}
          </div>
        )}
      </Disclosure>
      {state.variants.length > 1 && state.status === 'done' && (
        <Disclosure title="Research comparison" icon={<Scale className="h-4 w-4 text-ink-400" />}>
          <table className="w-full text-sm">
            <thead><tr className="text-left text-xs uppercase tracking-wider text-ink-400"><th className="pb-2">Variant</th><th>Fixes</th><th>Edge-case tests</th><th>Tests run</th><th>Benchmark tests</th></tr></thead>
            <tbody className="divide-y divide-line/5 text-ink-200">
              {state.variants.map(v => {
                const x = state.byVariant[v]?.result
                return <tr key={v}><td className="py-1.5 font-semibold">{v === 'proposed' ? 'With Bug Predictor' : v === 'baseline' ? 'Without Bug Predictor' : 'Random flags'}</td>
                  <td>{x?.fix_iterations ?? '–'}</td><td>{x?.n_targeted_tests ?? '–'}</td><td>{x?.test_executions ?? '–'}</td>
                  <td>{x?.final_hidden_pass == null ? 'n/a' : x.final_hidden_pass ? 'passed' : 'failed'}</td></tr>
              })}
            </tbody>
          </table>
        </Disclosure>
      )}
    </div>
  )
}
