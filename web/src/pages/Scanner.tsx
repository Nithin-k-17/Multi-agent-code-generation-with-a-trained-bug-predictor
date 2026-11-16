import clsx from 'clsx'
import { AnimatePresence, motion } from 'framer-motion'
import { AlertTriangle, Bot, CheckCircle2, Cpu, Eraser, ListChecks, ScanSearch, Sparkles, Wrench } from 'lucide-react'
import { useCallback, useRef, useState } from 'react'
import {
  CopyButton, Disclosure, FindingCard, isBug, KV, marksFor, MODEL_FACTS, riskLevel, SeveritySummary,
} from '../components/BugReport'
import CodeView from '../components/CodeView'
import Model3D from '../components/Model3D'
import { useHealth } from '../components/Navbar'
import { Button, Card, Spinner } from '../components/ui'
import { api, type Finding, type ReviewResult, type ScanResult } from '../lib/api'
import { fmtSeconds, useNow, useRunShortcut } from '../lib/hooks'

type Phase = 'idle' | 'scanning' | 'reviewing' | 'done'

const PLACEHOLDER = `Paste your Python code here…

It is analysed automatically as soon as you paste.
You can also type, then press "Find bugs" (Ctrl + Enter).`

export default function Scanner() {
  const { offline } = useHealth(10000)
  const [code, setCode] = useState('')
  const [analyzed, setAnalyzed] = useState<string | null>(null)
  const [scan, setScan] = useState<ScanResult | null>(null)
  const [review, setReview] = useState<ReviewResult | null>(null)
  const [phase, setPhase] = useState<Phase>('idle')
  const [error, setError] = useState<string | null>(null)
  const [reviewNote, setReviewNote] = useState<string | null>(null)
  const [reviewStarted, setReviewStarted] = useState(0)
  const reqId = useRef(0)
  const ctrl = useRef<AbortController | null>(null)
  const resultsRef = useRef<HTMLDivElement>(null)
  const now = useNow(phase === 'reviewing', 250)

  const analyze = useCallback(async (src: string) => {
    if (!src.trim()) return
    const id = ++reqId.current
    ctrl.current?.abort()
    const c = new AbortController()
    ctrl.current = c
    setError(null); setReviewNote(null); setScan(null); setReview(null); setAnalyzed(src)
    setPhase('scanning')
    try {
      const s = await api.scan(src)
      if (id !== reqId.current) return
      setScan(s)
      setPhase('reviewing'); setReviewStarted(Date.now())
      setTimeout(() => resultsRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' }), 60)
      try {
        const r = await api.review(src, c.signal)
        if (id === reqId.current) setReview(r)
      } catch (e) {
        if (id === reqId.current && (e as Error).name !== 'AbortError') setReviewNote((e as Error).message)
      }
    } catch (e) {
      if (id === reqId.current) setError((e as Error).message)
    } finally {
      if (id === reqId.current) setPhase('done')
    }
  }, [])

  useRunShortcut(() => analyze(code), !offline && !!code.trim())

  const clear = () => {
    reqId.current++; ctrl.current?.abort()
    setCode(''); setScan(null); setReview(null); setAnalyzed(null); setPhase('idle'); setError(null); setReviewNote(null)
  }

  const findings: Finding[] = review?.findings ?? scan?.findings ?? []
  const bugs = findings.filter(isBug)
  const warnings = findings.filter(f => !isBug(f))
  const stale = analyzed !== null && analyzed !== code
  const busy = phase === 'scanning' || phase === 'reviewing'
  const level = scan ? riskLevel(scan.score) : null

  return (
    <div className="mx-auto max-w-7xl px-4 py-8 sm:px-6">
      {/* header with the code-screen model blended in */}
      <div className="relative grid items-center gap-4 md:grid-cols-[1.2fr_1fr]">
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.2em] text-brand-400">Bug Scanner</p>
          <h1 className="mt-2 text-3xl font-bold tracking-tight text-ink-100 sm:text-5xl">Paste code. <span className="text-gradient">See the bugs.</span></h1>
          <p className="mt-3 max-w-xl text-ink-300">
            You get the number of bugs, where they are, what kind they are and how serious they are, plus a corrected version you can copy.
          </p>
        </div>
        <Model3D scene="hologram" className="h-60 sm:h-72 lg:h-80" zoom={1.05} />
      </div>

      {offline && (
        <Card className="mt-6 flex items-start gap-3 p-4">
          <AlertTriangle className="mt-0.5 h-5 w-5 shrink-0 text-amber-300" />
          <p className="text-sm text-ink-300"><b className="text-ink-100">The analysis server is not running.</b> Start it from the <code className="font-mono">ZIPFORLAP</code> folder with <code className="font-mono">python -m uvicorn server.app:app --port 8000</code>.</p>
        </Card>
      )}

      {/* editor + the code-screen model beside it */}
      <div className="mt-6 grid items-center gap-6 lg:grid-cols-[1fr_320px]">
      <Card className="p-4">
        <CodeView code={code} onChange={setCode} onPaste={v => { if (!offline) setTimeout(() => analyze(v), 50) }} onSubmit={v => { if (!offline) analyze(v) }}
          marks={!stale ? marksFor(findings) : []} height={340} placeholder={PLACEHOLDER} />
        <div className="mt-3 flex flex-wrap items-center gap-3">
          <p className="text-xs text-ink-400">{code ? `${code.split('\n').length} lines` : 'Python 3'}{stale && !busy ? ' · code changed since the last scan' : ''}</p>
          <div className="ml-auto flex gap-2">
            {code && <Button variant="ghost" onClick={clear}><Eraser className="h-4 w-4" /> Clear</Button>}
            <Button onClick={() => analyze(code)} disabled={offline || !code.trim() || busy}>
              {busy ? <Spinner /> : <ScanSearch className="h-4 w-4" />} {stale ? 'Scan again' : 'Find bugs'}
            </Button>
          </div>
        </div>
      </Card>
      <Model3D scene="screen" className="h-56 lg:h-[380px]" zoom={1.05} />
      </div>

      {error && <Card className="mt-6 p-4 text-sm text-ink-100"><AlertTriangle className="mr-2 inline h-4 w-4" />Scan failed: {error}</Card>}

      <div ref={resultsRef} className="scroll-mt-20" />
      <AnimatePresence>
        {(scan || phase === 'scanning') && (
          <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} className="mt-8 space-y-6">
            {/* 1. summary */}
            <Card className="p-6">
              {phase === 'scanning' || !scan ? (
                <p className="flex items-center gap-2 text-ink-300"><Spinner /> Checking your code…</p>
              ) : (
                <div className="flex flex-wrap items-center gap-6">
                  <div>
                    <p className="text-xs font-medium uppercase tracking-wider text-ink-400">Number of bugs</p>
                    <p className="text-5xl font-extrabold tracking-tight text-ink-100">{bugs.length}</p>
                  </div>
                  <div className="min-w-0 flex-1">
                    {bugs.length > 0 ? (
                      <>
                        <p className="text-lg font-semibold text-ink-100">{bugs.length === 1 ? '1 bug found' : `${bugs.length} bugs found`}</p>
                        <div className="mt-1.5"><SeveritySummary findings={bugs} /></div>
                      </>
                    ) : phase === 'reviewing' ? (
                      <p className="text-lg font-semibold text-ink-100">No bugs from the exact checks so far</p>
                    ) : (
                      <>
                        <p className="flex items-center gap-2 text-lg font-semibold text-ink-100"><CheckCircle2 className="h-5 w-5" /> No potential bugs were identified</p>
                        <p className="mt-1 text-sm text-ink-400">This does not guarantee the code is bug-free; test it with edge cases too.
                          {level !== 'Low' && ' The bug-prediction model still rates this code as higher risk than usual.'}</p>
                      </>
                    )}
                    {warnings.length > 0 && <p className="mt-2 text-sm text-ink-400">+ {warnings.length} code-quality {warnings.length === 1 ? 'warning' : 'warnings'}</p>}
                  </div>
                  <div className="rounded-xl bg-line/[0.04] px-4 py-3 text-sm ring-1 ring-line/10">
                    <p className="text-xs text-ink-400">Bug-prediction model</p>
                    <p className="font-semibold text-ink-100">{level} risk</p>
                  </div>
                </div>
              )}
              {phase === 'reviewing' && (
                <p className="mt-4 flex items-center gap-2 border-t border-line/10 pt-4 text-sm text-ink-300">
                  <Bot className="h-4 w-4" /> AI reviewer is checking the logic and writing a fix… <Spinner />
                  <span className="ml-auto font-mono text-xs tabular-nums text-ink-400">{fmtSeconds((now - reviewStarted) / 1000)}</span>
                </p>
              )}
              {reviewNote && <p className="mt-4 border-t border-line/10 pt-4 text-sm text-ink-400"><AlertTriangle className="mr-1 inline h-4 w-4" /> AI review unavailable: {reviewNote}. Showing exact checks only.</p>}
            </Card>

            {/* 2. bug detection */}
            {findings.length > 0 && (
              <section>
                <h2 className="mb-3 flex items-center gap-2 text-lg font-bold text-ink-100"><ListChecks className="h-5 w-5" /> Bug detection</h2>
                <ul className="space-y-3">
                  {findings.map((f, i) => <FindingCard key={`${f.line}-${f.title}-${i}`} f={f} index={i} />)}
                </ul>
              </section>
            )}

            {/* 3. suggested fix */}
            {review?.fixed_code && (
              <section>
                <div className="mb-3 flex flex-wrap items-center gap-3">
                  <h2 className="flex items-center gap-2 text-lg font-bold text-ink-100"><Wrench className="h-5 w-5" /> Suggested fix</h2>
                  <div className="ml-auto"><CopyButton text={review.fixed_code} label="Copy fixed code" /></div>
                </div>
                <Card className="p-4">
                  <CodeView code={review.fixed_code} height={Math.min(460, 40 + review.fixed_code.split('\n').length * 19)} />
                  <div className="mt-4 space-y-2 text-sm text-ink-300">
                    <p className="font-semibold text-ink-100">What was corrected</p>
                    <ul className="list-disc space-y-1 pl-5">
                      {findings.filter(f => f.source === 'ai' || f.source === 'static' || f.source === 'execution').map((f, i) => (
                        <li key={i}><span className="font-mono text-ink-100">{f.end_line > f.line ? `L${f.line}–${f.end_line}` : `L${f.line}`}</span>: {f.suggestion.split('\n')[0]}</li>
                      ))}
                    </ul>
                    {review.fix_check && (
                      <p className="flex items-center gap-2 pt-1 text-ink-400">
                        <Sparkles className="h-4 w-4" />
                        {review.fix_check.remaining.filter(isBug).length === 0
                          ? 'Re-checked: the exact checks find no issues in the fixed code.'
                          : `Re-checked: ${review.fix_check.remaining.filter(isBug).length} issue(s) remain in the fixed code; review it before using.`}
                      </p>
                    )}
                    <p className="text-xs text-ink-400">AI-written fix; please review and test it before using it.</p>
                  </div>
                </Card>
              </section>
            )}

            {/* 4. advanced */}
            {scan && (
              <Disclosure title="Advanced Model Details" icon={<Cpu className="h-4 w-4 text-ink-400" />}>
                <div className="grid gap-6 md:grid-cols-2">
                  <div>
                    <KV k="Bug probability (whole snippet)" v={<span className="font-mono">{(scan.score * 100).toFixed(1)}%</span>} />
                    <KV k="Classification threshold" v={<span className="font-mono">{scan.threshold.toFixed(2)}</span>} />
                    <KV k="Prediction" v={scan.score >= scan.threshold ? 'Buggy' : 'Clean'} />
                    {review?.fix_check && <KV k="Bug probability after fix" v={<span className="font-mono">{(review.fix_check.score * 100).toFixed(1)}%</span>} />}
                    <KV k="Model" v={MODEL_FACTS.name} />
                    <KV k="Version" v={MODEL_FACTS.version} />
                    <KV k="MBPP held-out benchmark" v={MODEL_FACTS.mbpp} />
                    <KV k="HumanEval benchmark" v={MODEL_FACTS.humaneval} />
                    {review && <KV k="AI reviewer" v={`${review.reviewer || 'Qwen2.5-Coder-1.5B'} · ${review.seconds}s`} />}
                  </div>
                  <div className="text-sm leading-relaxed text-ink-300">
                    <p className="font-semibold text-ink-100">How to read this</p>
                    <p className="mt-1">The bug probability is a score for the <b>whole code snippet</b>. It is not the probability that any particular line is wrong.</p>
                    <p className="mt-2">Bug locations come from exact checks (parser, syntax tree, running the file) and from the lines the AI fix had to change. Each finding says which source it came from.</p>
                    {scan.hot_lines.length > 0 && (
                      <>
                        <p className="mt-3 font-semibold text-ink-100">Lines the model paid most attention to</p>
                        <p className="text-xs text-ink-400">Gradient saliency; shows what influenced the score, not a confirmed bug location.</p>
                        <ul className="mt-1.5 space-y-1">
                          {scan.hot_lines.slice(0, 3).map(h => (
                            <li key={h.line_start} className={clsx('truncate rounded-md bg-blood-night px-2 py-1 font-mono text-xs text-vanilla')}>L{h.line_start} {h.text}</li>
                          ))}
                        </ul>
                      </>
                    )}
                  </div>
                </div>
              </Disclosure>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
