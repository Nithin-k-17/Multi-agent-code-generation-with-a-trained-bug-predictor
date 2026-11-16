import clsx from 'clsx'
import { AnimatePresence, motion } from 'framer-motion'
import { AlertTriangle, CheckCircle2, Info, X } from 'lucide-react'
import { createContext, useCallback, useContext, useState, type ReactNode } from 'react'

type Tone = 'good' | 'bad' | 'info'
interface ToastItem { id: number; title: string; body?: string; tone: Tone }

const ToastCtx = createContext<(t: Omit<ToastItem, 'id'>) => void>(() => {})

export const useToast = () => useContext(ToastCtx)

export function ToastProvider({ children }: { children: ReactNode }) {
  const [items, setItems] = useState<ToastItem[]>([])
  const dismiss = (id: number) => setItems(xs => xs.filter(x => x.id !== id))
  const push = useCallback((t: Omit<ToastItem, 'id'>) => {
    const id = Date.now() + Math.random()
    setItems(xs => [...xs.slice(-2), { ...t, id }])
    setTimeout(() => dismiss(id), 5000)
  }, [])

  const icons = { good: CheckCircle2, bad: AlertTriangle, info: Info }
  return (
    <ToastCtx.Provider value={push}>
      {children}
      <div className="pointer-events-none fixed bottom-5 right-5 z-50 flex w-[min(380px,calc(100vw-2.5rem))] flex-col gap-2" aria-live="polite">
        <AnimatePresence initial={false}>
          {items.map(t => {
            const Icon = icons[t.tone]
            return (
              <motion.div key={t.id} layout initial={{ opacity: 0, y: 16, scale: 0.97 }} animate={{ opacity: 1, y: 0, scale: 1 }}
                exit={{ opacity: 0, x: 40 }} transition={{ type: 'spring', stiffness: 420, damping: 32 }}
                className="glass pointer-events-auto flex items-start gap-3 rounded-xl p-4 shadow-2xl shadow-blood/20">
                <Icon className={clsx('mt-0.5 h-5 w-5 shrink-0', t.tone === 'good' ? 'text-emerald-400' : t.tone === 'bad' ? 'text-rose-400' : 'text-cyan-300')} />
                <div className="min-w-0 flex-1">
                  <p className="text-sm font-semibold text-ink-100">{t.title}</p>
                  {t.body && <p className="mt-0.5 text-xs text-ink-300">{t.body}</p>}
                </div>
                <button onClick={() => dismiss(t.id)} className="text-ink-400 hover:text-ink-100" aria-label="Dismiss"><X className="h-4 w-4" /></button>
              </motion.div>
            )
          })}
        </AnimatePresence>
      </div>
    </ToastCtx.Provider>
  )
}
