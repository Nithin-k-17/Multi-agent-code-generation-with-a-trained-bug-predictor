import { Bug, ClipboardList, Code2, FlaskConical, Wrench, type LucideIcon } from 'lucide-react'
import type { AgentName } from './api'

export interface AgentMeta { key: string; step: number; name: string; icon: LucideIcon; blurb: string; accent: string; isNew?: boolean }

export const AGENTS: AgentMeta[] = [
  { key: 'requirement', step: 1, name: 'Requirement', icon: ClipboardList, accent: '#7FA9CB',
    blurb: 'Turns the plain-English task into a structured spec: inputs, outputs, behaviour, edge cases.' },
  { key: 'developer', step: 2, name: 'Developer', icon: Code2, accent: '#5B8DB8',
    blurb: 'Writes the first implementation from the spec.' },
  { key: 'bug_predictor', step: 3, name: 'Bug Predictor', icon: Bug, accent: '#4A7FA7', isNew: true,
    blurb: 'Checks each new draft for likely bugs before any test runs, so testing starts where problems are most likely.' },
  { key: 'tester', step: 4, name: 'Tester', icon: FlaskConical, accent: '#9CC0DD',
    blurb: 'Writes and runs tests in a sandbox, adding extra edge-case tests when bugs look likely.' },
  { key: 'fixer', step: 5, name: 'Fixer', icon: Wrench, accent: '#3A6A91',
    blurb: 'Repairs failing code, guided by the failures and the flagged lines.' },
]

export const agentLabel: Record<AgentName, string> = {
  requirement: 'Requirement Agent',
  developer: 'Developer Agent',
  tester_base: 'Tester · writing tests',
  bug_predictor: 'Bug-Prediction Agent',
  tester_targeted: 'Tester · edge-case tests',
  tester: 'Tester · running tests',
  fixer: 'Fixer Agent',
}

export const metaFor = (agent: AgentName): AgentMeta =>
  AGENTS.find(a => agent.startsWith(a.key)) ?? AGENTS[3]
