# Pipeline results: `dev_val30_v2`

Split **mbpp_val**, LLM **Qwen/Qwen2.5-Coder-1.5B-Instruct (GGUF q4_k_m, llama.cpp) +lookup**, max fix iterations 3, base tests 5, targeted tests 5, bug config {'threshold': 0.7951, 'random_flag_rate': 0.4006}.

| metric | baseline | proposed | random |
|---|---|---|---|
| n | 30 | 30 | 30 |
| pass@1 | 43.3% | 40.0% | 43.3% |
| initial_pass@1 | 40.0% | 40.0% | 40.0% |
| selected_pass@1 | 43.3% | 40.0% | 43.3% |
| internal_tests_passed | 50.0% | 50.0% | 50.0% |
| avg_fix_iterations | 0.500 | 0.500 | 0.500 |
| avg_fixes_rejected | 0.500 | 0.500 | 0.500 |
| avg_fix_iterations_when_needed | 1.000 | 1.000 | 1.000 |
| n_needed_fix | 15 | 15 | 15 |
| repaired_initially_failing | 5.6% | 5.6% | 5.6% |
| broke_initially_passing | 0.0% | 8.3% | 0.0% |
| avg_iters_to_hidden_pass | 0.000 | 0.000 | 0.000 |
| avg_test_executions | 3.433 | 3.500 | 4.000 |
| avg_targeted_tests | 0.000 | 0.067 | 0.567 |
| flag_rate | - | 10.0% | 36.7% |
| avg_llm_calls | 0.500 | 0.600 | 0.867 |
| avg_completion_tokens | 29.033 | 46.200 | 104.767 |
| avg_seconds | 4.006 | 3.993 | 9.204 |
| avg_bug_agent_seconds | 0.000 | 0.282 | 0.000 |

## Paired comparisons

**proposed_vs_baseline** (n=30): pass@1 diff -3.3% 95% CI [-10.0%, 0.0%]; solved only by second/first: 0/1, McNemar p=1.0000; fix-iteration diff 0.000 95% CI [0.000, 0.000].

**random_vs_baseline** (n=30): pass@1 diff 0.0% 95% CI [0.0%, 0.0%]; solved only by second/first: 0/0, McNemar p=1.0000; fix-iteration diff 0.000 95% CI [0.000, 0.000].

**proposed_vs_random** (n=30): pass@1 diff -3.3% 95% CI [-10.0%, 0.0%]; solved only by second/first: 0/1, McNemar p=1.0000; fix-iteration diff 0.000 95% CI [0.000, 0.000].

## Agent-written test accuracy (vs. official reference solutions, analysis only)

- raw_tests: 129
- raw_test_accuracy: 45.7%
- kept_tests: 73
- kept_test_accuracy: 64.4%
- tests_kept: 56.6%

- candidates per problem: any candidate passes hidden tests in 43.3% of problems

## Bug-prediction agent on first drafts

- n: 30
- roc_auc: 0.505
- bug_rate_flagged: 33.3%
- bug_rate_unflagged: 63.0%
- recall: 5.6%
