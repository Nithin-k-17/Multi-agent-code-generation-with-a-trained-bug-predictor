# Pipeline results: `mbpp_test60_v2`

Split **mbpp_test**, LLM **Qwen/Qwen2.5-Coder-1.5B-Instruct (GGUF q4_k_m, llama.cpp) +lookup**, max fix iterations 3, base tests 5, targeted tests 5, bug config {'threshold': 0.7951, 'random_flag_rate': 0.4006}.

| metric | baseline | proposed | random |
|---|---|---|---|
| n | 60 | 60 | 60 |
| pass@1 | 53.3% | 53.3% | 53.3% |
| initial_pass@1 | 51.7% | 51.7% | 51.7% |
| selected_pass@1 | 53.3% | 53.3% | 53.3% |
| internal_tests_passed | 60.0% | 60.0% | 60.0% |
| avg_fix_iterations | 0.400 | 0.400 | 0.400 |
| avg_fixes_rejected | 0.400 | 0.400 | 0.400 |
| avg_fix_iterations_when_needed | 1.000 | 1.000 | 1.000 |
| n_needed_fix | 24 | 24 | 24 |
| repaired_initially_failing | 3.4% | 3.4% | 3.4% |
| broke_initially_passing | 0.0% | 0.0% | 0.0% |
| avg_iters_to_hidden_pass | 0.000 | 0.000 | 0.000 |
| avg_test_executions | 2.817 | 2.950 | 3.200 |
| avg_targeted_tests | 0.000 | 0.133 | 0.383 |
| flag_rate | - | 10.0% | 51.7% |
| avg_llm_calls | 0.400 | 0.500 | 0.917 |
| avg_completion_tokens | 27.200 | 24.350 | 109.317 |
| avg_seconds | 3.478 | 2.529 | 9.816 |
| avg_bug_agent_seconds | 0.000 | 0.139 | 0.000 |

## Paired comparisons

**proposed_vs_baseline** (n=60): pass@1 diff 0.0% 95% CI [0.0%, 0.0%]; solved only by second/first: 0/0, McNemar p=1.0000; fix-iteration diff 0.000 95% CI [0.000, 0.000].

**random_vs_baseline** (n=60): pass@1 diff 0.0% 95% CI [0.0%, 0.0%]; solved only by second/first: 0/0, McNemar p=1.0000; fix-iteration diff 0.000 95% CI [0.000, 0.000].

**proposed_vs_random** (n=60): pass@1 diff 0.0% 95% CI [0.0%, 0.0%]; solved only by second/first: 0/0, McNemar p=1.0000; fix-iteration diff 0.000 95% CI [0.000, 0.000].

## Agent-written test accuracy (vs. official reference solutions, analysis only)

- raw_tests: 255
- raw_test_accuracy: 40.0%
- kept_tests: 109
- kept_test_accuracy: 70.6%
- tests_kept: 42.7%

- candidates per problem: any candidate passes hidden tests in 53.3% of problems

## Bug-prediction agent on first drafts

- n: 60
- roc_auc: 0.670
- bug_rate_flagged: 50.0%
- bug_rate_unflagged: 48.1%
- recall: 10.3%
