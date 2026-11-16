import clsx from 'clsx'
import { lazy, Suspense } from 'react'
import type { SceneName } from './Scene3D'

// three.js is large, so the 3D models load right after first paint.
const Scene3D = lazy(() => import('./Scene3D'))

/**
 * An interactive 3D model that melts into the page: no box, soft faded edges, a faint glow behind it.
 * Drag to rotate, click to spin, pinch or use the buttons to zoom.
 */
export default function Model3D({ scene, className, zoom, interactive = true }: {
  scene: SceneName; className?: string; zoom?: number; interactive?: boolean
}) {
  return (
    <div className={clsx('relative', className)}>
      <div aria-hidden className="pointer-events-none absolute inset-[18%] animate-pulse-soft rounded-full bg-cyan-500/40 blur-3xl dark:bg-white/[0.05]" />
      <Suspense fallback={<div className="grid h-full place-items-center"><div className="h-10 w-10 animate-spin rounded-full border-2 border-line/15 border-t-brand-500" /></div>}>
        <Scene3D scene={scene} zoom={zoom} interactive={interactive} className="h-full w-full" />
      </Suspense>
    </div>
  )
}
