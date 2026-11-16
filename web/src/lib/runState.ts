import type { AgentName, PipelineEvent, Risk, TestOutcome, Variant, VariantResult } from './api'

export type Status = 'idle' | 'running' | 'done' | 'error'

export interface Step {
  key: string
  agent: AgentName
  status: 'running' | 'done'
  iter?: number
  code?: string
  tests?: string[]
  risk?: Risk
  results?: TestOutcome[]
  loadError?: string
  seconds?: number
  startedAt?: number
  /** the Fixer's change was rejected because it did not pass more trusted tests */
  rejected?: boolean
}

export interface VariantState { status: Status; steps: Step[]; result?: VariantResult; currentCode?: string; risk?: Risk }

interface SharedStep<T> { status: Status; seconds?: number; startedAt?: number; value?: T; dropped?: number }

/** What the LLM is writing right now (streamed token by token). */
export interface LiveOutput {
  agent: AgentName; variant?: Variant; iter?: number; text: string; startedAt: number
  /** streamed chunks (≈ tokens) and when the first one arrived, for the live tokens/second meter */
  chunks?: number; firstTokenAt?: number
}

export interface RunState {
  status: Status
  message?: string
  error?: string
  taskId?: string
  llm?: string
  hasHidden?: boolean
  publicTests?: string[]
  autoTests?: boolean
  warnings: string[]
  variants: Variant[]
  shared: {
    requirement: SharedStep<Record<string, unknown>>
    developer: SharedStep<string>
    tester_base: SharedStep<string[]>
  }
  initialHiddenPass?: boolean | null
  byVariant: Partial<Record<Variant, VariantState>>
  live?: LiveOutput
  startedAt?: number
  finishedAt?: number
  stepsDone: number
  stepsTotal: number
  /** total streamed chunks (≈ tokens) generated in this run */
  tokens: number
}

export const initialRun = (): RunState => ({
  status: 'idle',
  variants: [],
  warnings: [],
  shared: { requirement: { status: 'idle' }, developer: { status: 'idle' }, tester_base: { status: 'idle' } },
  byVariant: {},
  stepsDone: 0,
  stepsTotal: 1,
  tokens: 0,
})

/** Rough number of agent steps for the progress bar: 3 shared + per variant (bug, targeted, tests, fixes). */
const estimateSteps = (variants: Variant[], maxFix: number) =>
  3 + variants.reduce((n, v) => n + (v === 'baseline' ? 0 : 2) + 1 + Math.ceil(maxFix * 0.6) * 2, 0)

export function reduce(state: RunState, ev: PipelineEvent, maxFix = 3): RunState {
  const now = Date.now()
  const s: RunState = { ...state, shared: { ...state.shared }, byVariant: { ...state.byVariant } }
  switch (ev.type) {
    case 'status':
      return { ...s, status: 'running', message: ev.message, startedAt: s.startedAt ?? now }
    case 'start':
      for (const v of ev.variants) s.byVariant[v] = { status: 'idle', steps: [] }
      return {
        ...s, status: 'running', taskId: ev.task_id, llm: ev.llm, hasHidden: ev.has_hidden, variants: ev.variants,
        publicTests: ev.public_tests, autoTests: ev.auto_tests,
        message: undefined, startedAt: s.startedAt ?? now, stepsTotal: estimateSteps(ev.variants, maxFix),
      }
    case 'warning':
      return { ...s, warnings: [...s.warnings, ev.message] }
    case 'shared_done':
      return { ...s, initialHiddenPass: ev.initial_hidden_pass }
    case 'variant_start':
      s.byVariant[ev.variant] = { ...(s.byVariant[ev.variant] ?? { steps: [] }), status: 'running', currentCode: s.shared.developer.value }
      return s
    case 'variant_done':
      s.byVariant[ev.variant] = { ...s.byVariant[ev.variant]!, status: 'done', result: ev.result, currentCode: ev.result.final_code }
      return s
    case 'done':
      return { ...s, status: 'done', live: undefined, finishedAt: now, stepsDone: s.stepsTotal }
    case 'error':
      return { ...s, status: 'error', error: ev.message, live: undefined, finishedAt: now }
    case 'token': {
      const same = s.live && s.live.agent === ev.agent && s.live.variant === ev.variant && s.live.iter === ev.iter
      const n = ev.n ?? 1  // tokens in this delta (fast decoding streams several at once)
      const live: LiveOutput = same
        ? { ...s.live!, text: s.live!.text + ev.text, chunks: (s.live!.chunks ?? 0) + n, firstTokenAt: s.live!.firstTokenAt ?? now }
        : { agent: ev.agent, variant: ev.variant, iter: ev.iter, text: ev.text, startedAt: now, chunks: n, firstTokenAt: now }
      return { ...s, live, tokens: (s.tokens ?? 0) + n }
    }
    case 'agent': {
      if (ev.status === 'done') s.stepsDone = Math.min(s.stepsDone + 1, s.stepsTotal - 1)
      if (ev.status === 'running') s.live = { agent: ev.agent, variant: ev.variant, iter: ev.iter, text: '', startedAt: now }
      else if (s.live?.agent === ev.agent && s.live.variant === ev.variant) s.live = undefined

      if (!ev.variant) {
        const a = ev.agent as 'requirement' | 'developer' | 'tester_base'
        const value = a === 'requirement' ? ev.spec : a === 'developer' ? ev.code : ev.tests
        s.shared[a] = {
          ...s.shared[a], status: ev.status, ...(ev.status === 'running' && { startedAt: now }),
          ...(ev.seconds !== undefined && { seconds: ev.seconds }), ...(value !== undefined && { value }),
          ...(ev.dropped !== undefined && { dropped: ev.dropped }),
        } as never
        return s
      }
      const vs: VariantState = { ...(s.byVariant[ev.variant] ?? { status: 'running', steps: [] }) }
      const steps = [...vs.steps]
      const key = `${ev.agent}:${ev.iter ?? 0}`
      const i = steps.findIndex(st => st.key === key && st.status === 'running')
      const step: Step = {
        ...(i >= 0 ? steps[i] : { key, agent: ev.agent, startedAt: now }), status: ev.status, iter: ev.iter,
        ...(ev.code !== undefined && { code: ev.code }), ...(ev.tests && { tests: ev.tests }),
        ...(ev.risk && { risk: ev.risk }), ...(ev.results && { results: ev.results }),
        ...(ev.load_error && { loadError: ev.load_error }), ...(ev.seconds !== undefined && { seconds: ev.seconds }),
        ...(ev.rejected && { rejected: true }),
      }
      if (i >= 0) steps[i] = step
      else steps.push(step)
      vs.steps = steps
      if (ev.code !== undefined && ev.status === 'done') vs.currentCode = ev.code
      if (ev.risk) vs.risk = ev.risk
      s.byVariant[ev.variant] = vs
      return s
    }
  }
}
