# Pipeline results: `mbpp_test_ollama_30`

Split **mbpp_test**, LLM **qwen2.5-coder:1.5b**, max fix iterations 3, base tests 5, targeted tests 5, bug config {'threshold': 0.7951, 'random_flag_rate': 0.4006}.

| metric | baseline | proposed | random |
|---|---|---|---|
| n | 30 | 30 | 30 |
| pass@1 | 50.0% | 50.0% | 50.0% |
| initial_pass@1 | 50.0% | 50.0% | 50.0% |
| selected_pass@1 | - | - | - |
| internal_tests_passed | 23.3% | 23.3% | 20.0% |
| avg_fix_iterations | 2.300 | 2.300 | 2.400 |
| avg_fixes_rejected | - | - | - |
| avg_fix_iterations_when_needed | 3.000 | 3.000 | 3.000 |
| n_needed_fix | 23 | 23 | 24 |
| repaired_initially_failing | 0.0% | 0.0% | 0.0% |
| broke_initially_passing | 0.0% | 0.0% | 0.0% |
| avg_iters_to_hidden_pass | - | - | - |
| avg_test_executions | 16.533 | 18.000 | 21.033 |
| avg_targeted_tests | 0.000 | 0.367 | 1.300 |
| flag_rate | - | 20.0% | 50.0% |
| avg_llm_calls | 2.300 | 2.500 | 2.900 |
| avg_completion_tokens | 67.467 | 59.400 | 146.333 |
| avg_seconds | 17.958 | 11.017 | 20.180 |
| avg_bug_agent_seconds | 0.000 | 0.890 | 0.003 |

## Paired comparisons

**proposed_vs_baseline** (n=30): pass@1 diff 0.0% 95% CI [0.0%, 0.0%]; solved only by second/first: 0/0, McNemar p=1.0000; fix-iteration diff 0.000 95% CI [0.000, 0.000].

**random_vs_baseline** (n=30): pass@1 diff 0.0% 95% CI [0.0%, 0.0%]; solved only by second/first: 0/0, McNemar p=1.0000; fix-iteration diff 0.100 95% CI [0.000, 0.300].

**proposed_vs_random** (n=30): pass@1 diff 0.0% 95% CI [0.0%, 0.0%]; solved only by second/first: 0/0, McNemar p=1.0000; fix-iteration diff -0.100 95% CI [-0.300, 0.000].

## Agent-written test accuracy (vs. official reference solutions, analysis only)

- raw_tests: 115
- raw_test_accuracy: 47.8%
- kept_tests: 115
- kept_test_accuracy: 47.8%
- tests_kept: 100.0%

- candidates per problem: any candidate passes hidden tests in 50.0% of problems

## Bug-prediction agent on first drafts

- n: 30
- roc_auc: 0.667
- bug_rate_flagged: 33.3%
- bug_rate_unflagged: 54.2%
- recall: 13.3%
