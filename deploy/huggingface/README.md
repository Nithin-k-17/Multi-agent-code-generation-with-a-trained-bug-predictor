---
title: BugAware Agents
emoji: 🐞
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 5.50.0
python_version: "3.11"
app_file: app.py
pinned: false
short_description: Multi-agent code generation with a trained bug predictor
---

# BugAware Agents

Find, explain and fix bugs in Python code, or let five AI agents write and test code for you.

- **Bug Scanner:** paste Python code and get a bug-risk score per function from a fine-tuned CodeBERT
  model, highlighted risky lines, exact findings with line numbers, and a suggested fix.
- **Agent Pipeline:** describe a task and watch five agents (Requirement → Developer → Bug Predictor →
  Tester → Fixer) write, test and repair the code live.

Runs on a free CPU: the first visit after a quiet period wakes the Space up (about a minute), and one
agent run can be active at a time. Models: Qwen2.5-Coder-1.5B (4-bit, llama.cpp) for code generation and
a fine-tuned `microsoft/codebert-base` bug predictor.
