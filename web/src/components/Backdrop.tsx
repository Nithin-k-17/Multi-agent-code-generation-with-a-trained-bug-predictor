import { useEffect, useRef } from 'react'
import { useTheme } from '../lib/theme'

/**
 * Living page background: a slow "neural network" constellation drifting over a soft mesh gradient.
 * Nodes link up when they come close, and links reach toward the pointer.
 *   light: built from all five blues (#0A1931 #1A3D63 #4A7FA7 #B3CFE5 #F6FAFD)
 *   dark : only the black shades (#060606 #0C0C0C #0B1215 #101720 #171717 #0D1717 #020D19 #011222)
 *          with faint white links
 */
const LOOKS = {
  light: {
    base: '#F6FAFD',
    blobs: ['rgba(179,207,229,.75)', 'rgba(74,127,167,.30)', 'rgba(26,61,99,.16)', 'rgba(179,207,229,.6)'],
    nodes: ['#1A3D63', '#4A7FA7', '#0A1931', '#4A7FA7'],
    link: '26,61,99', linkAlpha: 0.20, nodeAlpha: 0.55,
  },
  dark: {
    base: '#060606',
    blobs: ['rgba(1,18,34,.9)', 'rgba(13,23,23,.85)', 'rgba(16,23,32,.9)', 'rgba(2,13,25,.85)'],
    nodes: ['#FFFFFF', '#D4D4D4', '#9A9A9A', '#FFFFFF'],
    link: '255,255,255', linkAlpha: 0.09, nodeAlpha: 0.35,
  },
}

export default function Backdrop() {
  const ref = useRef<HTMLCanvasElement>(null)
  const { theme } = useTheme()

  useEffect(() => {
    const canvas = ref.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    const look = LOOKS[theme]
    const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches
    let w = 0, h = 0, dpr = 1
    type Node = { x: number; y: number; vx: number; vy: number; r: number; c: string }
    let nodes: Node[] = []
    const pointer = { x: -9999, y: -9999 }

    const resize = () => {
      dpr = Math.min(window.devicePixelRatio || 1, 2)
      w = window.innerWidth; h = window.innerHeight
      canvas.width = w * dpr; canvas.height = h * dpr
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
      const count = Math.round(Math.min(90, Math.max(28, (w * h) / 22000)))
      nodes = Array.from({ length: count }, (_, i) => ({
        x: Math.random() * w, y: Math.random() * h,
        vx: (Math.random() - 0.5) * 0.25, vy: (Math.random() - 0.5) * 0.25,
        r: 1.2 + Math.random() * 2.2, c: look.nodes[i % look.nodes.length],
      }))
    }
    resize()
    window.addEventListener('resize', resize)
    const onMove = (e: PointerEvent) => { pointer.x = e.clientX; pointer.y = e.clientY }
    const onLeave = () => { pointer.x = -9999; pointer.y = -9999 }
    window.addEventListener('pointermove', onMove, { passive: true })
    document.addEventListener('pointerleave', onLeave)

    const LINK = 150
    let raf = 0
    const draw = (t: number) => {
      ctx.fillStyle = look.base
      ctx.fillRect(0, 0, w, h)
      // drifting mesh-gradient blobs
      look.blobs.forEach((c, i) => {
        const a = t / 9000 + i * 1.7
        const cx = w * (0.5 + 0.42 * Math.cos(a * (i % 2 ? 0.8 : 1.1))), cy = h * (0.5 + 0.38 * Math.sin(a * 0.9 + i))
        const rad = Math.max(w, h) * (0.32 + 0.06 * i)
        const g = ctx.createRadialGradient(cx, cy, 0, cx, cy, rad)
        g.addColorStop(0, c); g.addColorStop(1, 'rgba(0,0,0,0)')
        ctx.fillStyle = g
        ctx.fillRect(0, 0, w, h)
      })
      // constellation
      if (!reduced) {
        for (const n of nodes) {
          n.x += n.vx; n.y += n.vy
          if (n.x < -20) n.x = w + 20
          if (n.x > w + 20) n.x = -20
          if (n.y < -20) n.y = h + 20
          if (n.y > h + 20) n.y = -20
        }
      }
      ctx.lineWidth = 1
      for (let i = 0; i < nodes.length; i++) {
        const a = nodes[i]
        for (let j = i + 1; j < nodes.length; j++) {
          const b = nodes[j]
          const d = Math.hypot(a.x - b.x, a.y - b.y)
          if (d < LINK) {
            ctx.strokeStyle = `rgba(${look.link},${look.linkAlpha * (1 - d / LINK)})`
            ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke()
          }
        }
        const pd = Math.hypot(a.x - pointer.x, a.y - pointer.y)
        if (pd < LINK * 1.4) {
          ctx.strokeStyle = `rgba(${look.link},${look.linkAlpha * 2 * (1 - pd / (LINK * 1.4))})`
          ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(pointer.x, pointer.y); ctx.stroke()
        }
      }
      ctx.globalAlpha = look.nodeAlpha
      for (const n of nodes) {
        ctx.fillStyle = n.c
        ctx.beginPath(); ctx.arc(n.x, n.y, n.r, 0, Math.PI * 2); ctx.fill()
      }
      ctx.globalAlpha = 1
      if (!reduced) raf = requestAnimationFrame(draw)
    }
    raf = requestAnimationFrame(draw)
    const onVis = () => { cancelAnimationFrame(raf); if (!document.hidden && !reduced) raf = requestAnimationFrame(draw) }
    document.addEventListener('visibilitychange', onVis)
    return () => {
      cancelAnimationFrame(raf)
      window.removeEventListener('resize', resize)
      window.removeEventListener('pointermove', onMove)
      document.removeEventListener('pointerleave', onLeave)
      document.removeEventListener('visibilitychange', onVis)
    }
  }, [theme])

  return <canvas ref={ref} aria-hidden className="pointer-events-none fixed inset-0 -z-20 h-full w-full" />
}
