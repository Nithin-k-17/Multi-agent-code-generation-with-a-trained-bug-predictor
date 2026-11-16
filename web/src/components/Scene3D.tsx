import clsx from 'clsx'
import { Move3d, RotateCcw, ZoomIn, ZoomOut } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import * as THREE from 'three'
import { usePalette, useTheme, type Palette, type Theme } from '../lib/theme'

/*
  Three 3D models built only from the three palette colours, each shown in its own framed canvas:
    A. a round-headed robot typing on a laptop (glowing sage eyes, code on the screen)
    B. a large tilted code screen that types itself out
    C. a holographic wireframe laptop saying "Hello World!" with headphones and a mouse
*/

// Model colours. Set from the active theme's palette right before a scene is built (see applyPalette).
let VANILLA = '#F6FAFD'   // light highlights, text on screens
let SAGE = '#4A7FA7'      // steel blue: metal, outlines, glows
let SAGE_LIGHT = '#B3CFE5'
let BLOOD = '#1A3D63'     // main body colour
let BLOOD_NIGHT = '#0A1931' // screen background
let BLOOD_TINT = '#7FA9CB'  // code keywords on screens
let JOINT = '#0A1931'

function applyPalette(p: Palette, theme: Theme) {
  const dark = theme === 'dark'
  // light: the five blues · dark: black / charcoal bodies with white and grey highlights only
  VANILLA = dark ? '#FAFAFA' : '#F6FAFD'
  SAGE = dark ? '#9A9A9A' : '#4A7FA7'
  SAGE_LIGHT = dark ? '#E5E5E5' : '#B3CFE5'
  BLOOD = dark ? '#3A3A3A' : '#1A3D63'
  BLOOD_NIGHT = dark ? '#060606' : p.panel
  BLOOD_TINT = dark ? '#FFFFFF' : '#7FA9CB'
  JOINT = dark ? '#0C0C0C' : '#0A1931'
  MATTE_BODY = dark ? '#161616' : '#0A1931'
  MATTE_JOINT = dark ? '#0C0C0C' : '#06101F'
  MATTE_METAL = dark ? '#5A5A5A' : '#4A7FA7'
  MATTE_FLOOR = dark ? '#0B1215' : '#D6E6F2'
  GLOW = dark ? '#FFFFFF' : '#B3CFE5'
}
let MATTE_BODY = '#0A1931', MATTE_JOINT = '#06101F', MATTE_METAL = '#4A7FA7', MATTE_FLOOR = '#D6E6F2', GLOW = '#B3CFE5'

const CODE = [
  'def second_largest(nums):',
  '    nums = sorted(set(nums))',
  '    if len(nums) < 2:',
  '        return None',
  '    return nums[-2]',
  '',
  '# Step 3: bug-prediction agent',
  'score = bug_agent.score(code)',
  'if score >= tau:',
  '    tests += targeted(hot_lines)',
  '',
  'for it in range(max_fix_iters):',
  '    result = run_tests(code, tests)',
  '    if result.passed:',
  '        break',
  '    code = fixer.repair(code, result)',
  'print("all tests passed")',
]
const KEYWORDS = new Set(['def', 'return', 'if', 'for', 'in', 'None', 'break', 'import', 'from', 'print', 'range', 'len', 'sorted', 'set'])

/** Draws syntax-coloured code onto a 2D canvas. `chars` limits how much is revealed (typing effect). */
function drawCode(ctx: CanvasRenderingContext2D, w: number, h: number, fontPx: number, chars = Infinity, numbers = true, offsetLine = 0) {
  ctx.fillStyle = BLOOD_NIGHT
  ctx.fillRect(0, 0, w, h)
  ctx.font = `500 ${fontPx}px "JetBrains Mono", ui-monospace, monospace`
  ctx.textBaseline = 'top'
  const lh = fontPx * 1.55
  const left = numbers ? fontPx * 3.2 : fontPx
  let budget = chars
  const total = Math.ceil(h / lh) + 1
  for (let i = 0; i < total; i++) {
    const line = CODE[(i + offsetLine) % CODE.length]
    const y = fontPx * 0.8 + i * lh
    if (numbers) {
      ctx.fillStyle = 'rgba(179,207,229,.55)'
      ctx.fillRect(fontPx * 0.6, y + fontPx * 0.25, fontPx * 1.3, fontPx * 0.55)
    }
    if (budget <= 0) continue
    let x = left
    const shown = line.slice(0, Math.max(0, budget))
    budget -= line.length + 1
    for (const tok of shown.split(/(\s+|[()[\],:.=<>+#"-])/)) {
      if (!tok) continue
      const comment = shown.trimStart().startsWith('#')
      ctx.fillStyle = comment ? SAGE : KEYWORDS.has(tok) ? BLOOD_TINT : /^["\d]/.test(tok) ? SAGE_LIGHT : VANILLA
      ctx.fillText(tok, x, y)
      x += ctx.measureText(tok).width
    }
    if (budget <= 0 && budget > -line.length - 1) { // caret
      ctx.fillStyle = SAGE
      ctx.fillRect(x + 2, y, fontPx * 0.55, fontPx * 1.05)
    }
  }
}

function canvasTexture(w: number, h: number) {
  const c = document.createElement('canvas')
  c.width = w; c.height = h
  const ctx = c.getContext('2d')!
  const tex = new THREE.CanvasTexture(c)
  tex.colorSpace = THREE.SRGBColorSpace
  tex.anisotropy = 4
  return { c, ctx, tex }
}

/** Soft radial blob used as a fake contact shadow / glow. */
function radialTexture(color: string, alpha: number) {
  const { ctx, tex } = canvasTexture(128, 128)
  const g = ctx.createRadialGradient(64, 64, 0, 64, 64, 64)
  const c = new THREE.Color(color)
  const rgb = `${Math.round(c.r * 255)},${Math.round(c.g * 255)},${Math.round(c.b * 255)}`
  g.addColorStop(0, `rgba(${rgb},${alpha})`)
  g.addColorStop(1, `rgba(${rgb},0)`)
  ctx.fillStyle = g
  ctx.fillRect(0, 0, 128, 128)
  tex.needsUpdate = true
  return tex
}

/** Capsule mesh stretched between two points (used for arms and legs). */
function limb(mat: THREE.Material, r: number) {
  const m = new THREE.Mesh(new THREE.CapsuleGeometry(r, 1, 6, 14), mat)
  const set = (a: THREE.Vector3, b: THREE.Vector3) => {
    const d = b.clone().sub(a)
    const len = d.length()
    m.position.copy(a).addScaledVector(d, 0.5)
    m.scale.set(1, Math.max(0.05, len / (1 + 2 * r)), 1)
    m.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), d.normalize())
  }
  return { mesh: m, set }
}

function edges(geo: THREE.BufferGeometry, color: string, opacity = 1) {
  return new THREE.LineSegments(new THREE.EdgesGeometry(geo), new THREE.LineBasicMaterial({ color, transparent: opacity < 1, opacity }))
}

// ------------------------------------------------------------------ A. robot coder
function buildRobot(matte = false) {
  const g = new THREE.Group()
  // gloss: clear-coated shell · matte: soft, light-absorbing finish with glowing eyes and screen
  const shell = matte
    ? new THREE.MeshStandardMaterial({ color: MATTE_BODY, roughness: 0.82, metalness: 0.08 })
    : new THREE.MeshPhysicalMaterial({ color: BLOOD, roughness: 0.32, metalness: 0.15, clearcoat: 1, clearcoatRoughness: 0.25 })
  const joint = new THREE.MeshStandardMaterial({ color: matte ? MATTE_JOINT : JOINT, roughness: matte ? 0.9 : 0.5 })
  const metal = new THREE.MeshStandardMaterial({ color: matte ? MATTE_METAL : SAGE, roughness: matte ? 0.45 : 0.35, metalness: 0.55 })
  const glow = new THREE.MeshBasicMaterial({ color: matte ? GLOW : VANILLA, toneMapped: false })

  const head = new THREE.Mesh(new THREE.SphereGeometry(0.95, 48, 48), shell)
  head.position.set(0, 2.05, 0)
  g.add(head)
  const ear = new THREE.Mesh(new THREE.CylinderGeometry(0.24, 0.24, 0.08, 32), joint)
  ear.position.set(-0.45, 0.05, 0.83)
  ear.lookAt(new THREE.Vector3(-0.9, 0.1, 1.7))
  ear.rotateX(Math.PI / 2)
  head.add(ear)
  // glowing eyes on the side of the face that looks at the laptop (+x), turned slightly to the camera
  for (const d of [new THREE.Vector3(0.42, -0.05, 0.9), new THREE.Vector3(0.74, -0.05, 0.67)]) {
    const n = d.normalize()
    const eye = new THREE.Mesh(new THREE.SphereGeometry(0.13, 20, 20), glow)
    eye.position.copy(n).multiplyScalar(0.88)
    eye.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), n)
    eye.scale.set(0.75, 1.3, 0.8)
    head.add(eye)
    const halo = new THREE.Mesh(new THREE.PlaneGeometry(0.5, 0.5), new THREE.MeshBasicMaterial({ map: radialTexture(SAGE_LIGHT, 0.7), transparent: true, depthWrite: false }))
    halo.position.copy(n).multiplyScalar(0.97)
    halo.quaternion.copy(eye.quaternion)
    head.add(halo)
  }
  const neck = new THREE.Mesh(new THREE.CylinderGeometry(0.16, 0.2, 0.3, 20), joint)
  neck.position.set(0, 1.12, 0)
  g.add(neck)
  const body = new THREE.Mesh(new THREE.CapsuleGeometry(0.52, 0.45, 8, 24), shell)
  body.position.set(-0.05, 0.62, 0)
  body.rotation.z = -0.18
  g.add(body)

  // laptop on the robot's lap, screen facing the robot (−x)
  const laptop = new THREE.Group()
  laptop.position.set(1.55, 0.47, 0)
  const base = new THREE.Mesh(new THREE.BoxGeometry(1.25, 0.05, 1.0), metal)
  laptop.add(base)
  const hinge = new THREE.Group()
  hinge.position.set(0.62, 0.02, 0)
  hinge.rotation.z = 0.22
  laptop.add(hinge)
  const lid = new THREE.Mesh(new THREE.BoxGeometry(0.04, 0.9, 1.0), metal)
  lid.position.set(0, 0.45, 0)
  hinge.add(lid)
  const scr = canvasTexture(512, 460)
  scr.tex.wrapT = THREE.RepeatWrapping
  drawCode(scr.ctx, 512, 460, 22, Infinity, false)
  scr.tex.needsUpdate = true
  const screen = new THREE.Mesh(new THREE.PlaneGeometry(0.92, 0.82), new THREE.MeshBasicMaterial({ map: scr.tex }))
  screen.position.set(-0.025, 0.45, 0)
  screen.rotation.y = -Math.PI / 2
  hinge.add(screen)
  g.add(laptop)
  const screenGlow = new THREE.PointLight(matte ? GLOW : SAGE, matte ? 9 : 6, matte ? 4 : 3.2, 1.6)
  screenGlow.position.set(1.4, 1.0, 0)
  g.add(screenGlow)
  if (matte) {
    // eye light spilling onto the face, and a soft floor the robot sits on
    const eyeLight = new THREE.PointLight(GLOW, 2.2, 2.2, 2)
    eyeLight.position.set(0.9, 2.0, 0.9)
    g.add(eyeLight)
    const floor = new THREE.Mesh(new THREE.CircleGeometry(2.6, 64), new THREE.MeshStandardMaterial({ color: MATTE_FLOOR, roughness: 0.95 }))
    floor.rotation.x = -Math.PI / 2
    floor.position.set(0.6, -0.31, 0)
    floor.userData.noBounds = true
    g.add(floor)
  }

  const arms = [-1, 1].map(s => ({ s, l: limb(shell, 0.12), hand: new THREE.Mesh(new THREE.SphereGeometry(0.13, 20, 20), joint) }))
  arms.forEach(a => { g.add(a.l.mesh, a.hand) })
  for (const s of [-1, 1]) {
    const thigh = limb(shell, 0.15), shin = limb(shell, 0.14)
    const hip = new THREE.Vector3(0.05, 0.22, 0.27 * s), knee = new THREE.Vector3(1.0, 0.27, 0.33 * s), foot = new THREE.Vector3(1.1, -0.16, 0.34 * s)
    thigh.set(hip, knee); shin.set(knee, foot)
    g.add(thigh.mesh, shin.mesh)
  }

  const shadow = new THREE.Mesh(new THREE.PlaneGeometry(4.2, 2.6), new THREE.MeshBasicMaterial({ map: radialTexture(BLOOD, 0.35), transparent: true, depthWrite: false }))
  shadow.rotation.x = -Math.PI / 2
  shadow.position.set(0.7, -0.3, 0)
  g.add(shadow)

  let scroll = 0
  const update = (t: number) => {
    for (const a of arms) {
      const tap = Math.max(0, Math.sin(t * 13 + (a.s > 0 ? 0 : Math.PI)))
      const shoulder = new THREE.Vector3(0.12, 1.0, 0.46 * a.s)
      const hand = new THREE.Vector3(1.3 + 0.04 * Math.sin(t * 2 + a.s), 0.6 + 0.05 * tap, 0.24 * a.s)
      a.hand.position.copy(hand)
      a.l.set(shoulder, hand)
    }
    head.rotation.z = 0.05 * Math.sin(t * 0.9)
    head.rotation.y = 0.08 * Math.sin(t * 0.6)
    scroll = (scroll + 0.0016) % 1
    scr.tex.offset.y = -scroll
    screenGlow.intensity = 5 + Math.sin(t * 7) * 0.6
  }
  return { group: g, update }
}

// ------------------------------------------------------------------ B. tilted code screen
function buildCodeScreen() {
  const g = new THREE.Group()
  const W = 1280, H = 760
  const scr = canvasTexture(W, H)
  const panel = new THREE.Mesh(new THREE.PlaneGeometry(7.2, 4.27), new THREE.MeshBasicMaterial({ map: scr.tex, side: THREE.DoubleSide }))
  g.add(panel)
  const frame = new THREE.Mesh(new THREE.BoxGeometry(7.45, 4.5, 0.12), new THREE.MeshPhysicalMaterial({ color: BLOOD, roughness: 0.3, clearcoat: 1 }))
  frame.position.z = -0.08
  g.add(frame)
  g.add(edges(new THREE.PlaneGeometry(7.25, 4.32), SAGE, 0.8))
  const halo = new THREE.Mesh(new THREE.PlaneGeometry(11, 7.5), new THREE.MeshBasicMaterial({ map: radialTexture(SAGE, 0.45), transparent: true, depthWrite: false }))
  halo.position.z = -0.4
  g.add(halo)
  g.rotation.set(-0.12, 0.55, 0.06)

  const totalChars = CODE.join('\n').length
  let chars = 0, hold = 0, last = 0, page = 0
  const update = (t: number) => {
    if (t - last < 0.05) return
    last = t
    if (chars < totalChars) chars += 3
    else if ((hold += 1) > 60) { chars = 0; hold = 0; page = (page + 6) % CODE.length }
    drawCode(scr.ctx, W, H, 34, chars, true, page)
    // soft scanlines in sage
    scr.ctx.fillStyle = 'rgba(179,207,229,.05)'
    for (let y = (t * 40) % 6; y < H; y += 6) scr.ctx.fillRect(0, y, W, 1)
    scr.tex.needsUpdate = true
  }
  return { group: g, update }
}

// ------------------------------------------------------------------ C. holographic laptop
function buildHologram() {
  const g = new THREE.Group()
  const fill = new THREE.MeshBasicMaterial({ color: SAGE, transparent: true, opacity: 0.12, depthWrite: false, side: THREE.DoubleSide })

  const baseGeo = new THREE.BoxGeometry(3.4, 0.12, 2.2)
  const base = new THREE.Mesh(baseGeo, fill)
  base.add(edges(baseGeo, SAGE))
  g.add(base)
  // keyboard grid
  const keys = new THREE.Group()
  for (let r = 0; r < 4; r++) for (let c = 0; c < 12; c++) {
    const k = edges(new THREE.BoxGeometry(0.2, 0.02, 0.2), SAGE_LIGHT, 0.6)
    k.position.set(-1.32 + c * 0.24, 0.07, -0.75 + r * 0.25)
    keys.add(k)
  }
  const pad = edges(new THREE.BoxGeometry(0.9, 0.01, 0.45), SAGE_LIGHT, 0.6)
  pad.position.set(0, 0.07, 0.72)
  keys.add(pad)
  g.add(keys)

  const hinge = new THREE.Group()
  hinge.position.set(0, 0.06, -1.1)
  hinge.rotation.x = -0.18
  g.add(hinge)
  const lidGeo = new THREE.BoxGeometry(3.4, 2.15, 0.06)
  const lid = new THREE.Mesh(lidGeo, fill)
  lid.position.y = 1.08
  lid.add(edges(lidGeo, SAGE))
  hinge.add(lid)

  const hw = canvasTexture(768, 480)
  const screen = new THREE.Mesh(new THREE.PlaneGeometry(3.15, 1.97), new THREE.MeshBasicMaterial({ map: hw.tex, transparent: true, depthWrite: false }))
  screen.position.set(0, 1.08, 0.04)
  hinge.add(screen)

  // floating code fragments around the screen
  const frags: THREE.Mesh[] = []
  for (const [x, y, w] of [[-2.25, 1.6, 0.9], [2.3, 2.0, 0.8], [2.2, 0.9, 0.7], [-2.2, 0.7, 0.6]] as const) {
    const t = canvasTexture(256, 128)
    t.ctx.clearRect(0, 0, 256, 128)
    for (let i = 0; i < 6; i++) {
      t.ctx.fillStyle = [SAGE, VANILLA, BLOOD_TINT][i % 3]
      t.ctx.fillRect(8 + (i % 2) * 18, 10 + i * 19, 60 + ((i * 53) % 150), 8)
    }
    t.tex.needsUpdate = true
    const m = new THREE.Mesh(new THREE.PlaneGeometry(w, w / 2), new THREE.MeshBasicMaterial({ map: t.tex, transparent: true, depthWrite: false }))
    m.position.set(x, y, 0.2)
    hinge.add(m)
    frags.push(m)
  }

  // headphones
  const solid = new THREE.MeshPhysicalMaterial({ color: BLOOD, roughness: 0.35, clearcoat: 1 })
  const phones = new THREE.Group()
  const band = new THREE.Mesh(new THREE.TorusGeometry(0.62, 0.07, 16, 48, Math.PI), solid)
  phones.add(band)
  for (const s of [-1, 1]) {
    const cup = new THREE.Mesh(new THREE.CylinderGeometry(0.34, 0.34, 0.24, 32), solid)
    cup.rotation.z = Math.PI / 2
    cup.position.set(0.62 * s, -0.05, 0)
    const ring = new THREE.Mesh(new THREE.TorusGeometry(0.22, 0.025, 8, 32), new THREE.MeshBasicMaterial({ color: SAGE_LIGHT }))
    ring.rotation.y = Math.PI / 2
    ring.position.x = 0.13 * s
    cup.add(ring)
    phones.add(cup)
  }
  phones.position.set(-2.55, 0.62, 0.1)
  phones.rotation.set(0.15, 0.9, -0.35)
  g.add(phones)

  // mouse
  const mouse = new THREE.Mesh(new THREE.SphereGeometry(0.5, 32, 24), solid)
  mouse.scale.set(0.55, 0.3, 0.85)
  mouse.position.set(2.45, 0.12, 0.45)
  mouse.add(edges(new THREE.SphereGeometry(0.505, 10, 6), SAGE, 0.5))
  g.add(mouse)

  const shadow = new THREE.Mesh(new THREE.PlaneGeometry(7, 3.4), new THREE.MeshBasicMaterial({ map: radialTexture(BLOOD, 0.22), transparent: true, depthWrite: false }))
  shadow.rotation.x = -Math.PI / 2
  shadow.position.y = -0.1
  g.add(shadow)
  g.rotation.set(0.32, -0.35, 0)

  let last = -1
  const drawHello = (t: number) => {
    const { ctx } = hw
    ctx.clearRect(0, 0, 768, 480)
    ctx.fillStyle = 'rgba(10,25,49,.6)'
    ctx.fillRect(0, 0, 768, 480)
    ctx.fillStyle = 'rgba(179,207,229,.10)'
    for (let y = 0; y < 480; y += 5) ctx.fillRect(0, y, 768, 1)
    // glitch bars
    for (let i = 0; i < 5; i++) {
      const y = (Math.sin(t * 3.1 + i * 7.3) * 0.5 + 0.5) * 460
      ctx.fillStyle = i % 2 ? 'rgba(179,207,229,.28)' : 'rgba(246,250,253,.12)'
      ctx.fillRect(((i * 97 + t * 60) % 300), y, 120 + i * 40, 4 + (i % 3) * 3)
    }
    const jitter = Math.random() < 0.12 ? (Math.random() - 0.5) * 14 : 0
    ctx.font = 'bold 84px "JetBrains Mono", ui-monospace, monospace'
    ctx.textBaseline = 'middle'
    const draw = (txt: string, x: number, y: number) => {
      ctx.fillStyle = BLOOD_TINT; ctx.fillText(txt, x - 4 + jitter, y + 2)
      ctx.fillStyle = SAGE; ctx.fillText(txt, x + 4 - jitter, y - 2)
      ctx.fillStyle = VANILLA; ctx.fillText(txt, x, y)
    }
    draw('Hello', 200, 190)
    draw('World!', 300, 300)
    // corner brackets
    ctx.strokeStyle = SAGE; ctx.lineWidth = 6
    const br = (x: number, y: number, dx: number, dy: number) => { ctx.beginPath(); ctx.moveTo(x + dx * 30, y); ctx.lineTo(x, y); ctx.lineTo(x, y + dy * 30); ctx.stroke() }
    br(170, 130, 1, 1); br(500, 130, -1, 1); br(270, 360, 1, -1); br(640, 360, -1, -1)
    hw.tex.needsUpdate = true
  }

  const update = (t: number) => {
    if (t - last > 0.08) { last = t; drawHello(t) }
    frags.forEach((f, i) => { f.position.y += Math.sin(t * 1.3 + i) * 0.0015 })
    phones.rotation.y = 0.9 + Math.sin(t * 0.7) * 0.08
  }
  return { group: g, update }
}

export type SceneName = 'robot' | 'screen' | 'hologram' | 'coder'

const BUILDERS: Record<SceneName, () => { group: THREE.Object3D; update: (t: number) => void }> = {
  robot: () => buildRobot(false), coder: () => buildRobot(true), screen: buildCodeScreen, hologram: buildHologram,
}

/**
 * One interactive 3D model in its own canvas, framed to fill its box.
 *   drag            rotate (with inertia); vertical page scrolling still works on touch screens
 *   click / tap     spin and hop
 *   pinch, + / -    zoom          double-click or reset button: back to the start view
 * When left alone it slowly turns by itself. Renders only while on screen.
 */
export default function Scene3D({ scene, className, zoom = 1, interactive = true }: {
  scene: SceneName; className?: string; zoom?: number; interactive?: boolean
}) {
  const host = useRef<HTMLDivElement>(null)
  const controls = useRef<{ zoomBy: (f: number) => void; reset: () => void } | null>(null)
  const [hint, setHint] = useState(true)
  const { theme } = useTheme()
  const palette = usePalette()

  useEffect(() => {
    const el = host.current
    if (!el) return
    applyPalette(palette, theme)
    let renderer: THREE.WebGLRenderer
    try {
      renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: 'low-power' })
    } catch {
      return // no WebGL: the box simply stays empty
    }
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    renderer.outputColorSpace = THREE.SRGBColorSpace
    renderer.toneMapping = THREE.ACESFilmicToneMapping
    renderer.toneMappingExposure = 1.1
    Object.assign(renderer.domElement.style, { width: '100%', height: '100%', display: 'block', touchAction: 'pan-y' })
    el.appendChild(renderer.domElement)

    const world = new THREE.Scene()
    const camera = new THREE.PerspectiveCamera(32, 1, 0.1, 100)
    const matte = scene === 'coder'
    world.add(new THREE.HemisphereLight(palette.light, SAGE, matte ? (theme === 'dark' ? 0.55 : 1.1) : theme === 'dark' ? 1.3 : 1.7))
    const key = new THREE.DirectionalLight(VANILLA, matte ? 1.4 : 2.6)
    key.position.set(-4, 6, 6)
    world.add(key)
    const rim = new THREE.DirectionalLight(SAGE, matte ? 3 : 2.4)
    rim.position.set(5, 2, -4)
    world.add(rim)

    const part = BUILDERS[scene]()
    if (scene === 'robot') part.group.rotation.y = 0.35
    if (scene === 'coder') {
      part.group.rotation.y = -0.55 // face and glowing eyes turned toward the viewer
      const back = new THREE.DirectionalLight(theme === 'dark' ? '#FFFFFF' : '#B3CFE5', theme === 'dark' ? 1.6 : 1.2)
      back.position.set(-3, 4, -5) // rim light so the matte silhouette reads against the background
      world.add(back)
    }
    const pivot = new THREE.Group()
    pivot.add(part.group)
    world.add(pivot)

    // Centre the model and pull the camera back just far enough to fit it.
    part.update(0)
    // Bounds of the solid parts only: soft shadows, glows, halos and the floor would pull the centre off the model.
    part.group.updateMatrixWorld(true)
    const box = new THREE.Box3()
    part.group.traverse(o => {
      const m = o as THREE.Mesh
      if (!m.isMesh || m.userData.noBounds) return
      const mat = m.material as THREE.Material
      if (mat.transparent && !mat.depthWrite) return
      m.geometry.computeBoundingBox()
      box.union(m.geometry.boundingBox!.clone().applyMatrix4(m.matrixWorld))
    })
    if (box.isEmpty()) box.setFromObject(part.group)
    const center = box.getCenter(new THREE.Vector3())
    part.group.position.sub(center)
    const size = box.getSize(new THREE.Vector3())

    let userZoom = 1, zoomNow = 1
    let baseDist = 10
    const layout = () => {
      const w = Math.max(1, el.clientWidth), h = Math.max(1, el.clientHeight)
      renderer.setSize(w, h, false)
      camera.aspect = w / h
      const vFov = THREE.MathUtils.degToRad(camera.fov)
      const fitH = (size.y / 2) / Math.tan(vFov / 2)
      const fitW = (size.x / 2) / (Math.tan(vFov / 2) * camera.aspect)
      // Fit for the resting view, but never closer than what keeps the model in frame at any rotation.
      const r = Math.hypot(size.x, size.z) / 2
      const anyAngle = Math.max(r / Math.tan(vFov / 2) / camera.aspect, (size.y / 2) / Math.tan(vFov / 2)) + r
      baseDist = Math.max((Math.max(fitH, fitW) + size.z / 2) * 1.08, anyAngle * 0.92) / zoom
      camera.updateProjectionMatrix()
    }
    layout()
    const ro = new ResizeObserver(layout)
    ro.observe(el)

    // ---------------- interaction state
    let yaw = 0, pitch = 0, vel = 0, spin = 0, hop = 0
    let lastTouch = -10
    const pointers = new Map<number, { x: number; y: number }>()
    let downAt: { x: number; y: number; t: number } | null = null
    let pinchDist = 0
    const canvas = renderer.domElement
    const clock = new THREE.Clock()

    const onDown = (e: PointerEvent) => {
      if (!interactive) return
      canvas.setPointerCapture(e.pointerId)
      pointers.set(e.pointerId, { x: e.clientX, y: e.clientY })
      downAt = { x: e.clientX, y: e.clientY, t: performance.now() }
      if (pointers.size === 2) {
        const [p1, p2] = [...pointers.values()]
        pinchDist = Math.hypot(p1.x - p2.x, p1.y - p2.y)
      }
      vel = 0
      lastTouch = clock.getElapsedTime()
      setHint(false)
      canvas.style.cursor = 'grabbing'
    }
    const onMove = (e: PointerEvent) => {
      const prev = pointers.get(e.pointerId)
      if (!prev) return
      const dx = e.clientX - prev.x, dy = e.clientY - prev.y
      pointers.set(e.pointerId, { x: e.clientX, y: e.clientY })
      if (pointers.size === 2) {
        const [p1, p2] = [...pointers.values()]
        const d = Math.hypot(p1.x - p2.x, p1.y - p2.y)
        if (pinchDist) userZoom = THREE.MathUtils.clamp(userZoom * (d / pinchDist), 0.6, 2.4)
        pinchDist = d
        return
      }
      yaw += dx * 0.011
      pitch = THREE.MathUtils.clamp(pitch + dy * 0.006, -0.55, 0.55)
      vel = dx * 0.011
      lastTouch = clock.getElapsedTime()
    }
    const onUp = (e: PointerEvent) => {
      pointers.delete(e.pointerId)
      if (pointers.size < 2) pinchDist = 0
      canvas.style.cursor = 'grab'
      if (downAt && Math.hypot(e.clientX - downAt.x, e.clientY - downAt.y) < 5 && performance.now() - downAt.t < 350) {
        spin = 1; hop = 1 // click / tap: spin and hop
      }
      downAt = null
      lastTouch = clock.getElapsedTime()
    }
    const reset = () => { yaw = 0; pitch = 0; vel = 0; userZoom = 1; lastTouch = clock.getElapsedTime() }
    controls.current = { zoomBy: f => { userZoom = THREE.MathUtils.clamp(userZoom * f, 0.6, 2.4); setHint(false) }, reset }
    if (interactive) {
      canvas.style.cursor = 'grab'
      canvas.addEventListener('pointerdown', onDown)
      canvas.addEventListener('pointermove', onMove)
      canvas.addEventListener('pointerup', onUp)
      canvas.addEventListener('pointercancel', onUp)
      canvas.addEventListener('dblclick', reset)
    }

    const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches
    let raf = 0, visible = true
    const frame = () => {
      const t = clock.getElapsedTime()
      if (!reduced) part.update(t)
      const idle = t - lastTouch > 2.5 && pointers.size === 0
      if (pointers.size === 0) { yaw += vel; vel *= 0.94 }
      if (idle && !reduced) { yaw += 0.0035; pitch += (0 - pitch) * 0.02 }
      let extra = 0
      if (spin > 0) { spin = Math.max(0, spin - 0.016); extra = (1 - (1 - (1 - spin)) ** 3) * Math.PI * 2; if (spin === 0) yaw += Math.PI * 2 }
      pivot.rotation.y = yaw + (spin > 0 ? extra : 0)
      pivot.rotation.x = pitch
      if (hop > 0) hop = Math.max(0, hop - 0.025)
      pivot.position.y = Math.sin(t * 1.1) * 0.06 + Math.sin(hop * Math.PI) * size.y * 0.12
      zoomNow += (userZoom - zoomNow) * 0.12
      camera.position.set(0, size.y * 0.06, baseDist / zoomNow)
      camera.lookAt(0, 0, 0)
      renderer.render(world, camera)
      raf = requestAnimationFrame(frame)
    }
    const start = () => { if (!raf && visible && !document.hidden) raf = requestAnimationFrame(frame) }
    const stop = () => { cancelAnimationFrame(raf); raf = 0 }
    const io = new IntersectionObserver(([e]) => { visible = e.isIntersecting; if (visible) start(); else stop() })
    io.observe(el)
    const onVis = () => (document.hidden ? stop() : start())
    document.addEventListener('visibilitychange', onVis)
    camera.position.set(0, size.y * 0.06, baseDist)
    camera.lookAt(0, 0, 0)
    renderer.render(world, camera)
    start()

    return () => {
      stop()
      io.disconnect()
      ro.disconnect()
      controls.current = null
      canvas.removeEventListener('pointerdown', onDown)
      canvas.removeEventListener('pointermove', onMove)
      canvas.removeEventListener('pointerup', onUp)
      canvas.removeEventListener('pointercancel', onUp)
      canvas.removeEventListener('dblclick', reset)
      document.removeEventListener('visibilitychange', onVis)
      world.traverse(o => {
        const m = o as THREE.Mesh
        m.geometry?.dispose()
        const mats = Array.isArray(m.material) ? m.material : m.material ? [m.material] : []
        mats.forEach(mt => { (mt as THREE.MeshBasicMaterial).map?.dispose(); mt.dispose() })
      })
      renderer.dispose()
      renderer.domElement.remove()
    }
  }, [scene, zoom, theme, palette, interactive])

  const btn = 'grid h-8 w-8 place-items-center rounded-full bg-ink-900/80 text-ink-100 ring-1 ring-line/15 backdrop-blur transition hover:scale-110 hover:bg-ink-800'
  return (
    <div className={clsx('group/scene relative', className)}>
      <div ref={host} className="blend-3d absolute inset-0" aria-label={`Interactive 3D model: drag to rotate, click to spin`} role="img" />
      {interactive && (
        <>
          <div className={clsx('pointer-events-none absolute bottom-2 left-1/2 -translate-x-1/2 whitespace-nowrap rounded-full bg-ink-900/75 px-3 py-1 text-[11px] font-medium text-ink-300 ring-1 ring-line/10 backdrop-blur transition-opacity duration-500',
            hint ? 'opacity-100' : 'opacity-0')}>
            <Move3d className="mr-1 inline h-3.5 w-3.5" /> Drag to rotate · click to spin
          </div>
          <div className="absolute right-2 top-2 flex gap-1.5 opacity-0 md:bottom-2 md:top-auto transition-opacity duration-300 group-hover/scene:opacity-100 focus-within:opacity-100 max-md:opacity-100">
            <button type="button" className={btn} onClick={() => controls.current?.zoomBy(1.2)} aria-label="Zoom in"><ZoomIn className="h-4 w-4" /></button>
            <button type="button" className={btn} onClick={() => controls.current?.zoomBy(1 / 1.2)} aria-label="Zoom out"><ZoomOut className="h-4 w-4" /></button>
            <button type="button" className={btn} onClick={() => controls.current?.reset()} aria-label="Reset view"><RotateCcw className="h-4 w-4" /></button>
          </div>
        </>
      )}
    </div>
  )
}
