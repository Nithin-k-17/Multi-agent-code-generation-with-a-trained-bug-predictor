import clsx from 'clsx'
import { Cpu, Menu, Moon, Sun, X } from 'lucide-react'
import { AnimatePresence, motion } from 'framer-motion'
import { useEffect, useState } from 'react'
import { Link, NavLink, useLocation } from 'react-router-dom'
import { api, type Health } from '../lib/api'
import { useTheme } from '../lib/theme'
import { engineLabel } from './ui'

export function ThemeToggle() {
  const { theme, toggle } = useTheme()
  const dark = theme === 'dark'
  return (
    <button onClick={toggle} aria-label={dark ? 'Switch to light theme' : 'Switch to dark theme'} title={dark ? 'Light theme' : 'Dark theme'}
      className="relative inline-flex h-9 w-16 shrink-0 items-center rounded-full bg-ink-800 p-1 ring-1 ring-inset ring-line/15 transition-colors hover:ring-line/30">
      <span className={clsx('grid h-7 w-7 place-items-center rounded-full bg-brand-500 text-on-brand shadow-md transition-transform duration-300', dark && 'translate-x-7')}>
        {dark ? <Moon className="h-4 w-4" /> : <Sun className="h-4 w-4" />}
      </span>
    </button>
  )
}

const links = [
  { to: '/scanner', label: 'Bug Scanner' },
  { to: '/playground', label: 'Agent Pipeline' },
]

export function useHealth(pollMs = 8000) {
  const [health, setHealth] = useState<Health | null>(null)
  const [offline, setOffline] = useState(false)
  useEffect(() => {
    let alive = true
    const tick = () => api.health().then(h => { if (alive) { setHealth(h); setOffline(false) } })
      .catch(() => { if (alive) setOffline(true) })
    tick()
    const id = setInterval(tick, pollMs)
    return () => { alive = false; clearInterval(id) }
  }, [pollMs])
  return { health, offline }
}

/** Thin bar under the navbar that fills as the page scrolls. */
function ScrollProgress() {
  const [p, setP] = useState(0)
  useEffect(() => {
    const on = () => {
      const max = document.documentElement.scrollHeight - window.innerHeight
      setP(max > 0 ? Math.min(1, window.scrollY / max) : 0)
    }
    on()
    window.addEventListener('scroll', on, { passive: true })
    window.addEventListener('resize', on)
    return () => { window.removeEventListener('scroll', on); window.removeEventListener('resize', on) }
  }, [])
  return <div className="absolute inset-x-6 -bottom-px h-[2px] origin-left rounded-full bg-gradient-to-r from-brand-500 via-brand-300 to-cyan-400" style={{ transform: `scaleX(${p})` }} />
}

export default function Navbar() {
  const [open, setOpen] = useState(false)
  const [scrolled, setScrolled] = useState(false)
  const { health, offline } = useHealth()
  const { pathname } = useLocation()
  useEffect(() => {
    const on = () => setScrolled(window.scrollY > 8)
    on()
    window.addEventListener('scroll', on, { passive: true })
    return () => window.removeEventListener('scroll', on)
  }, [])
  const warming = health && (health.warmup === 'loading_predictor' || health.warmup === 'loading_llm')
  const dot = offline ? 'bg-rose-500' : health?.busy ? 'bg-amber-400 animate-pulse' : warming ? 'bg-cyan-400 animate-pulse' : health ? 'bg-emerald-400' : 'bg-ink-400'
  const label = offline ? 'Server offline' : health?.busy ? 'Agents working' : warming ? 'Loading models…'
    : health?.warmup === 'ready' ? 'Ready' : health ? 'Online' : 'Connecting…'

  return (
    <header className="sticky top-0 z-40 px-3 pt-3 sm:px-6">
      <nav className={clsx('relative mx-auto flex h-14 max-w-7xl items-center justify-between rounded-2xl px-3 transition-all duration-300 sm:px-4',
        scrolled ? 'glass shadow-xl' : 'bg-transparent')}>
        <Link to="/" className="group flex items-center gap-2.5" onClick={() => setOpen(false)}>
          <img src="/logo.svg" alt="" className="h-8 w-8 transition-transform duration-500 group-hover:rotate-[360deg]" />
          <span className="text-lg font-bold tracking-tight text-ink-100">BugAware<span className="text-gradient"> Agents</span></span>
        </Link>
        <div className="hidden items-center gap-1 rounded-xl bg-line/[0.04] p-1 ring-1 ring-line/10 md:flex">
          {links.map(l => {
            const active = pathname.startsWith(l.to)
            return (
              <NavLink key={l.to} to={l.to} className={clsx('relative rounded-lg px-3.5 py-1.5 text-sm font-medium transition-colors', active ? 'text-on-brand' : 'text-ink-300 hover:text-ink-100')}>
                {active && <motion.span layoutId="navActive" className="absolute inset-0 rounded-lg bg-brand-500 shadow-md" transition={{ type: 'spring', stiffness: 420, damping: 34 }} />}
                <span className="relative">{l.label}</span>
              </NavLink>
            )
          })}
        </div>
        <div className="hidden items-center gap-3 md:flex">
          <span className="flex items-center gap-2 rounded-full bg-line/5 px-3 py-1 text-xs text-ink-300 ring-1 ring-line/10" title={health?.llm.model ?? ''}>
            <span className="relative flex h-2 w-2"><span className={clsx('absolute inline-flex h-full w-full animate-ping rounded-full opacity-60', dot)} /><span className={clsx('relative h-2 w-2 rounded-full', dot)} /></span> {label}
            {health?.llm && !offline && <span className="border-l border-line/15 pl-2 text-ink-400">{engineLabel(health.llm.model, health.llm.backend)}</span>}
          </span>
          <ThemeToggle />
        </div>
        <div className="flex items-center gap-2 md:hidden">
          <ThemeToggle />
          <button className="rounded-lg p-2 text-ink-300" onClick={() => setOpen(o => !o)} aria-label="Menu" aria-expanded={open}>
            {open ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
          </button>
        </div>
        {scrolled && <ScrollProgress />}
      </nav>
      <AnimatePresence>
        {open && (
          <motion.div initial={{ opacity: 0, y: -8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -8 }}
            className="glass mx-auto mt-2 max-w-7xl rounded-2xl p-2 md:hidden">
            {links.map(l => (
              <NavLink key={l.to} to={l.to} onClick={() => setOpen(false)}
                className={({ isActive }) => clsx('block rounded-xl px-3 py-2.5 text-sm font-medium', isActive ? 'bg-brand-500 text-on-brand' : 'text-ink-300')}>
                {l.label}
              </NavLink>
            ))}
            <p className="mt-1 flex items-center gap-2 px-3 py-1 text-xs text-ink-400"><span className={clsx('h-2 w-2 rounded-full', dot)} /> {label}</p>
          </motion.div>
        )}
      </AnimatePresence>
    </header>
  )
}

/** A seamless sine wave, two screen-widths wide, so it can scroll forever by sliding half its width. */
function wavePath(amp: number, periods: number, phase: number) {
  const W = 2880, H = 80, mid = 34
  let d = `M0 ${H} L0 ${mid}`
  for (let x = 0; x <= W; x += 24) d += ` L${x} ${(mid + amp * Math.sin((x / W) * Math.PI * 2 * periods + phase)).toFixed(1)}`
  return d + ` L${W} ${H} Z`
}
const WAVES = [
  { d: wavePath(12, 4, 0), dur: '22s', op: 0.35 },
  { d: wavePath(16, 6, 1.4), dur: '15s', op: 0.55 },
  { d: wavePath(9, 8, 2.6), dur: '9s', op: 1 },
]

export function Footer() {
  return (
    <footer className="relative mt-40 bg-blood-deep text-vanilla/80 dark:bg-ink-900">
      {/* three wave layers drifting at different speeds */}
      <div aria-hidden className="absolute -top-[79px] left-0 h-20 w-full overflow-hidden text-blood-deep dark:text-ink-900">
        {WAVES.map((w, i) => (
          <svg key={i} className="wave-drift absolute bottom-0 left-0 h-full w-[200%]" style={{ animationDuration: w.dur, opacity: w.op }} viewBox="0 0 2880 80" preserveAspectRatio="none">
            <path fill="currentColor" d={w.d} />
          </svg>
        ))}
      </div>
      <div className="mx-auto grid max-w-7xl gap-8 px-4 py-12 sm:grid-cols-[1.4fr_1fr_1fr] sm:px-6">
        <div>
          <div className="flex items-center gap-2.5">
            <img src="/logo.svg" alt="" className="h-8 w-8" />
            <span className="text-lg font-bold text-vanilla">BugAware Agents</span>
          </div>
          <p className="mt-3 max-w-sm text-sm leading-relaxed">Find, explain and fix bugs in Python code, or let five AI agents write and test it for you.</p>
        </div>
        <div>
          <p className="text-xs font-semibold uppercase tracking-widest text-vanilla/50">Tools</p>
          <ul className="mt-3 space-y-2 text-sm">
            <li><Link to="/scanner" className="transition hover:text-vanilla">Bug Scanner</Link></li>
            <li><Link to="/playground" className="transition hover:text-vanilla">Agent Pipeline</Link></li>
          </ul>
        </div>
        <div>
          <p className="text-xs font-semibold uppercase tracking-widest text-vanilla/50">Privacy</p>
          <p className="mt-3 flex items-start gap-2 text-sm"><Cpu className="mt-0.5 h-4 w-4 shrink-0" /> Everything runs locally on your machine. Your code never leaves it.</p>
        </div>
      </div>
    </footer>
  )
}
