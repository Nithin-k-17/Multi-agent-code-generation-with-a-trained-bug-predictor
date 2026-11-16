// A scripted replay used when the backend is offline (e.g. presenting without a GPU).
// It illustrates the mechanism; it is NOT an experimental result and the UI labels it as such.
import type { PipelineEvent } from './api'

const task = 'Write a function to find the second largest distinct number in a list.'
const draft = `def second_largest(nums):
    nums = sorted(nums)
    if len(nums) < 2:
        return None
    return nums[-2]`
const fixed = `def second_largest(nums):
    distinct = sorted(set(nums))
    if len(distinct) < 2:
        return None
    return distinct[-2]`

const publicTest = 'assert second_largest([1, 2, 3, 4]) == 3'
const base = [
  'assert second_largest([10, 20, 30]) == 20',
  'assert second_largest([7]) is None',
  'assert second_largest([-1, -5, -3]) == -3',
  'assert second_largest([0, 100]) == 0',
]
const targeted = [
  'assert second_largest([5, 5, 4]) == 4',
  'assert second_largest([9, 9, 9]) is None',
  'assert second_largest([3, 3, 2, 2, 1]) == 2',
]

const risk = {
  mode: 'model' as const, score: 0.861, threshold: 0.795, flagged: true,
  functions: [{ name: 'second_largest', line_start: 1, line_end: 5, score: 0.861 }],
  risky_functions: [{ name: 'second_largest', line_start: 1, line_end: 5, score: 0.861 }],
  hot_lines: [
    { line_start: 5, line_end: 5, saliency: 0.212, text: 'return nums[-2]' },
    { line_start: 2, line_end: 2, saliency: 0.094, text: 'nums = sorted(nums)' },
  ],
}

const ok = (t: string, targetedFlag = false) => ({ test: t, ok: true, error: '', targeted: targetedFlag })

export const demoTask = { description: task, entry_point: 'second_largest', public_tests: [publicTest] }

export const demoEvents: [number, PipelineEvent][] = [
  [200, { type: 'start', task_id: 'demo', variants: ['baseline', 'proposed'], llm: 'demo replay', has_hidden: true }],
  [100, { type: 'agent', agent: 'requirement', status: 'running' }],
  [1300, { type: 'agent', agent: 'requirement', status: 'done', spec: {
    function_name: 'second_largest', signature: 'second_largest(nums: list[int]) -> int | None',
    output: { type: 'int | None', description: 'second largest DISTINCT value, None if fewer than two' },
    edge_cases: ['duplicates of the maximum', 'single element', 'all equal', 'negative numbers'],
  } }],
  [100, { type: 'agent', agent: 'developer', status: 'running' }],
  [1500, { type: 'agent', agent: 'developer', status: 'done', code: draft }],
  [100, { type: 'agent', agent: 'tester_base', status: 'running' }],
  [1100, { type: 'agent', agent: 'tester_base', status: 'done', tests: base }],
  [200, { type: 'shared_done', initial_hidden_pass: false }],

  [200, { type: 'variant_start', variant: 'baseline' }],
  [100, { type: 'agent', variant: 'baseline', agent: 'tester', status: 'running', iter: 0 }],
  [700, { type: 'agent', variant: 'baseline', agent: 'tester', status: 'done', iter: 0, results: [ok(publicTest), ...base.map(t => ok(t))] }],
  [200, { type: 'variant_done', variant: 'baseline', result: {
    task_id: 'demo', variant: 'baseline', initial_hidden_pass: false, final_hidden_pass: false, internal_tests_passed: true,
    fix_iterations: 0, n_targeted_tests: 0, test_executions: 5, flagged_initial: null, risk_score_initial: null,
    final_code: draft, variant_llm: { llm_calls: 0, completion_tokens: 0, llm_seconds: 0 }, variant_seconds: 0.4,
  } }],

  [300, { type: 'variant_start', variant: 'proposed' }],
  [100, { type: 'agent', variant: 'proposed', agent: 'bug_predictor', status: 'running' }],
  [900, { type: 'agent', variant: 'proposed', agent: 'bug_predictor', status: 'done', risk, code: draft }],
  [100, { type: 'agent', variant: 'proposed', agent: 'tester_targeted', status: 'running' }],
  [1200, { type: 'agent', variant: 'proposed', agent: 'tester_targeted', status: 'done', tests: targeted }],
  [100, { type: 'agent', variant: 'proposed', agent: 'tester', status: 'running', iter: 0 }],
  [800, { type: 'agent', variant: 'proposed', agent: 'tester', status: 'done', iter: 0, results: [
    { test: targeted[0], ok: false, error: 'AssertionError', targeted: true },
    { test: targeted[1], ok: false, error: 'AssertionError', targeted: true },
    { test: targeted[2], ok: false, error: 'AssertionError', targeted: true },
    ok(publicTest), ...base.map(t => ok(t)),
  ] }],
  [100, { type: 'agent', variant: 'proposed', agent: 'fixer', status: 'running', iter: 1 }],
  [1500, { type: 'agent', variant: 'proposed', agent: 'fixer', status: 'done', iter: 1, code: fixed }],
  [600, { type: 'agent', variant: 'proposed', agent: 'bug_predictor', status: 'done', iter: 1, code: fixed, risk: {
    ...risk, score: 0.412, flagged: false, risky_functions: [], hot_lines: [],
    functions: [{ name: 'second_largest', line_start: 1, line_end: 5, score: 0.412 }],
  } }],
  [100, { type: 'agent', variant: 'proposed', agent: 'tester', status: 'running', iter: 1 }],
  [800, { type: 'agent', variant: 'proposed', agent: 'tester', status: 'done', iter: 1, results: [
    ...targeted.map(t => ok(t, true)), ok(publicTest), ...base.map(t => ok(t)),
  ] }],
  [200, { type: 'variant_done', variant: 'proposed', result: {
    task_id: 'demo', variant: 'proposed', initial_hidden_pass: false, final_hidden_pass: true, internal_tests_passed: true,
    fix_iterations: 1, n_targeted_tests: 3, test_executions: 16, flagged_initial: true, risk_score_initial: 0.861,
    final_code: fixed, variant_llm: { llm_calls: 2, completion_tokens: 210, llm_seconds: 2.7 }, variant_seconds: 4.9,
  } }],
  [100, { type: 'done' }],
]

/** Text an agent "writes" before its done event, so the demo streams like a real run. */
function streamedText(ev: PipelineEvent): string | null {
  if (ev.type !== 'agent' || ev.status !== 'done') return null
  if (ev.agent === 'requirement' && ev.spec) return '```json\n' + JSON.stringify(ev.spec, null, 2) + '\n```'
  if ((ev.agent === 'developer' || ev.agent === 'fixer') && ev.code) return '```python\n' + ev.code + '\n```'
  if ((ev.agent === 'tester_base' || ev.agent === 'tester_targeted') && ev.tests) return '```python\n' + ev.tests.join('\n') + '\n```'
  return null
}

/** Plays the scripted events with their delays; returns a cancel function. */
export function playDemo(onEvent: (e: PipelineEvent) => void, speed = 1) {
  let cancelled = false
  const sleep = (ms: number) => new Promise(r => setTimeout(r, ms / speed))
  ;(async () => {
    for (const [delay, ev] of demoEvents) {
      const text = streamedText(ev)
      if (text && ev.type === 'agent') {
        const chunks = text.match(/[\s\S]{1,4}/g) ?? []
        for (const c of chunks) {
          await sleep(delay / chunks.length)
          if (cancelled) return
          onEvent({ type: 'token', text: c, agent: ev.agent, variant: ev.variant, iter: ev.iter })
        }
      } else {
        await sleep(delay)
      }
      if (cancelled) return
      onEvent(text && ev.type === 'agent' ? { ...ev, seconds: +(delay / 1000).toFixed(1) } : ev)
    }
  })()
  return () => { cancelled = true }
}
