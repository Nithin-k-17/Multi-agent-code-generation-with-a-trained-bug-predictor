// Typed client for the FastAPI backend (server/app.py).

export type Variant = 'baseline' | 'proposed' | 'random'
export type AgentName = 'requirement' | 'developer' | 'tester_base' | 'bug_predictor' | 'tester_targeted' | 'tester' | 'fixer'

export interface FuncScore { name: string; line_start: number; line_end: number; score: number }
export interface HotLine { line_start: number; line_end: number; saliency: number; text: string }
export interface Risk {
  mode: 'model' | 'random'
  score: number
  threshold: number
  flagged: boolean
  functions: FuncScore[]
  risky_functions: FuncScore[]
  hot_lines: HotLine[]
  line_saliency?: Record<string, number>
}
export type BugType = 'Syntax error' | 'Runtime error' | 'Logic error' | 'Edge case' | 'Performance' | 'Code quality'
export type Severity = 'Critical' | 'High' | 'Medium' | 'Low'
/** One detected bug. `source`: static = exact check, execution = sandbox run, ai = location of a change the AI fix made. */
export interface Finding {
  line: number; end_line: number; type: BugType; severity: Severity; title: string
  explanation: string; suggestion: string; source: 'static' | 'execution' | 'ai'; before?: string; after?: string
}
export interface ScanResult extends Risk { findings: Finding[] }
export interface ReviewResult {
  findings: Finding[]; fixed_code: string; says_clean: boolean; seconds: number; reviewer: string
  fix_check: { remaining: Finding[]; score: number; threshold: number } | null
}

export interface TestOutcome { test: string; ok: boolean; error: string; targeted?: boolean }

export interface VariantResult {
  task_id: string
  variant: Variant
  initial_hidden_pass: boolean | null
  final_hidden_pass: boolean | null
  internal_tests_passed: boolean
  fix_iterations: number
  n_targeted_tests: number
  test_executions: number
  flagged_initial: boolean | null
  risk_score_initial: number | null
  final_code: string
  variant_llm: { llm_calls: number; completion_tokens: number; llm_seconds: number; cache_hits?: number }
  variant_seconds: number
}

export type PipelineEvent =
  | { type: 'status'; message: string }
  | { type: 'start'; task_id: string; variants: Variant[]; llm: string; has_hidden: boolean; public_tests?: string[]; auto_tests?: boolean }
  | { type: 'warning'; message: string }
  | { type: 'shared_done'; initial_hidden_pass: boolean | null }
  | { type: 'variant_start'; variant: Variant }
  | { type: 'variant_done'; variant: Variant; result: VariantResult }
  | { type: 'done'; seconds?: number }
  | { type: 'token'; text: string; agent: AgentName; variant?: Variant; iter?: number; n?: number }
  | { type: 'error'; message: string; trace?: string }
  | {
      type: 'agent'; agent: AgentName; status: 'running' | 'done'; variant?: Variant; iter?: number
      spec?: Record<string, unknown>; code?: string; tests?: string[]; risk?: Risk
      results?: TestOutcome[]; load_error?: string; seconds?: number; dropped?: number; rejected?: boolean
    }

export interface Health {
  ok: boolean; predictor_loaded: boolean; llm_loaded: boolean; cuda: boolean; busy: boolean
  threshold: number | null; warmup: 'idle' | 'loading_predictor' | 'loading_llm' | 'ready' | 'error'; llm_error: string | null
  llm: { backend: string; model: string | null; base_url: string | null; quantize: string | null }
}

export interface BenchmarkProblem {
  task_id: string; description: string; entry_point: string; starter_code: string
  public_tests: string[]; has_hidden: boolean
}

export interface RunRequest {
  description?: string; entry_point?: string; starter_code?: string; public_tests?: string[]
  benchmark_split?: string; task_id?: string
  variants: Variant[]; max_fix_iters: number; n_base_tests: number; n_targeted_tests: number; n_candidates?: number
  /** Code generator: "default" (Qwen2.5-Coder 1.5B) or an Ollama model id such as "qwen2.5-coder:7b". */
  model?: string
}

/** One entry of the model picker (GET /api/models). */
export interface ModelOption { id: string; label: string; detail: string; params: string; size_gb: number; default: boolean }

export interface VariantSummary { [metric: string]: number }
export interface Comparison {
  n_paired: number; 'pass@1_diff': number; 'pass@1_diff_ci95': [number, number]
  solved_only_by_second: number; solved_only_by_first: number; mcnemar_p: number
  fix_iter_diff: number; fix_iter_diff_ci95: [number, number]
}
export interface RunSummary {
  config: Record<string, unknown>
  variants: Record<string, VariantSummary>
  comparisons: Record<string, Comparison>
  bug_agent: Record<string, number>
  shared_stage?: { n: number; avg_llm_calls: number; avg_seconds: number; any_candidate_pass?: number }
  test_quality?: { raw_tests: number; raw_test_accuracy: number; kept_tests: number; kept_test_accuracy: number; tests_kept: number }
  policies?: {
    n: number; codebert_flag_rate: number; threshold: number; oracle_any_candidate: number; ties_with_distinct_code: number
    gating: Record<string, { 'pass@1': number; dev_calls_per_problem: number }>
    tie_break: Record<string, number>
  }
}
export interface RunListItem { name: string; summary: RunSummary | null; modified: number }

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = res.statusText
    try { detail = (await res.json()).detail ?? detail } catch { /* not json */ }
    throw new Error(typeof detail === 'string' ? detail : JSON.stringify(detail))
  }
  return res.json() as Promise<T>
}

export const api = {
  health: () => fetch('/api/health').then(r => json<Health>(r)),
  model: () => fetch('/api/model').then(r => json<Record<string, any>>(r)),
  scan: (code: string) =>
    fetch('/api/scan', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ code }) })
      .then(r => json<ScanResult>(r)),
  review: (code: string, signal?: AbortSignal) =>
    fetch('/api/review', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ code }), signal })
      .then(r => json<ReviewResult>(r)),
  benchmarks: (split: string, limit = 60) => fetch(`/api/benchmarks/${split}?limit=${limit}`).then(r => json<BenchmarkProblem[]>(r)),
  runs: () => fetch('/api/runs').then(r => json<RunListItem[]>(r)),
  run: (name: string) => fetch(`/api/runs/${encodeURIComponent(name)}`).then(r => json<RunDetail>(r)),
  models: () => fetch('/api/models').then(r => json<{ models: ModelOption[]; active: string }>(r)),
}

export interface RunRow {
  task_id: string; variant: Variant; initial_hidden_pass: boolean | null; final_hidden_pass: boolean | null
  fix_iterations: number; n_targeted_tests: number; test_executions: number
  flagged_initial: boolean | null; risk_score_initial: number | null; variant_seconds: number
}
export interface RunDetail { name: string; summary: RunSummary | null; rows: RunRow[] }

/** POSTs a run and invokes `onEvent` for every Server-Sent Event until the stream ends. */
export async function streamRun(req: RunRequest, onEvent: (e: PipelineEvent) => void, signal?: AbortSignal) {
  const res = await fetch('/api/run', {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(req), signal,
  })
  if (!res.ok || !res.body) await json(res)
  const reader = res.body!.getReader()
  const decoder = new TextDecoder()
  let buf = ''
  for (;;) {
    const { value, done } = await reader.read()
    if (done) break
    buf += decoder.decode(value, { stream: true })
    let idx
    while ((idx = buf.indexOf('\n\n')) >= 0) {
      const chunk = buf.slice(0, idx)
      buf = buf.slice(idx + 2)
      const line = chunk.split('\n').find(l => l.startsWith('data: '))
      if (line) onEvent(JSON.parse(line.slice(6)))
    }
  }
}
