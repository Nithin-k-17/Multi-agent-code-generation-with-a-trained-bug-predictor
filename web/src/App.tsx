import { AnimatePresence, MotionConfig, motion } from 'framer-motion'
import { lazy, Suspense, useEffect } from 'react'
import { Route, Routes, useLocation } from 'react-router-dom'
import Navbar, { Footer } from './components/Navbar'
import { ToastProvider } from './components/Toast'
import { Spinner } from './components/ui'
import Home from './pages/Home'
import Backdrop from './components/Backdrop'

// Heavier pages (charts, editor) load on demand so the landing page stays fast.
const Playground = lazy(() => import('./pages/Playground'))
const Scanner = lazy(() => import('./pages/Scanner'))

// Warm the lazy chunks after first paint so navigating feels instant.
const preload = () => [import('./pages/Playground'), import('./pages/Scanner')]

function ScrollToTop() {
  const { pathname } = useLocation()
  useEffect(() => { window.scrollTo({ top: 0 }) }, [pathname])
  return null
}

export default function App() {
  const location = useLocation()
  useEffect(() => {
    const id = setTimeout(preload, 1200)
    return () => clearTimeout(id)
  }, [])

  // Spotlight: cards light up under the pointer (see .glass in index.css)
  useEffect(() => {
    let last: HTMLElement | null = null
    const on = (e: PointerEvent) => {
      const el = (e.target as HTMLElement | null)?.closest?.('.glass') as HTMLElement | null
      if (last && last !== el) { last.style.removeProperty('--mx'); last.style.removeProperty('--my') }
      if (el) {
        const r = el.getBoundingClientRect()
        el.style.setProperty('--mx', `${e.clientX - r.left}px`)
        el.style.setProperty('--my', `${e.clientY - r.top}px`)
      }
      last = el
    }
    window.addEventListener('pointermove', on, { passive: true })
    return () => window.removeEventListener('pointermove', on)
  }, [])

  return (
    <MotionConfig reducedMotion="user">
      <ToastProvider>
        <div className="relative isolate flex min-h-screen flex-col">
          <ScrollToTop />
          <Backdrop />
          <Navbar />
          <main className="flex-1">
            <Suspense fallback={<div className="grid min-h-[50vh] place-items-center text-ink-400"><Spinner className="h-6 w-6" /></div>}>
              <AnimatePresence mode="wait" initial={false}>
                <motion.div key={location.pathname} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -4 }}
                  transition={{ duration: 0.18, ease: 'easeOut' }}>
                  <Routes location={location}>
                    <Route path="/" element={<Home />} />
                    <Route path="/playground" element={<Playground />} />
                    <Route path="/scanner" element={<Scanner />} />
                    <Route path="*" element={<Home />} />
                  </Routes>
                </motion.div>
              </AnimatePresence>
            </Suspense>
          </main>
          <Footer />
        </div>
      </ToastProvider>
    </MotionConfig>
  )
}
