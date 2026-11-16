import { createContext, useCallback, useContext, useLayoutEffect, useState, type ReactNode } from 'react'

export type Theme = 'light' | 'dark'

/** Colours for things CSS classes can't reach: charts, the code editor, SVG gauges and the 3D models. */
export interface Palette {
  bg: string; surface: string; text: string; muted: string; grid: string
  primary: string; primarySoft: string; accent: string; accentLight: string
  /** code panels (always dark) and the text on them */
  panel: string; panelText: string; keyword: string; string: string; number: string; comment: string; lineNo: string
  good: string; warn: string; bad: string; track: string
  /** 3D model materials */
  shell: string; joint: string; metal: string; glow: string; light: string
}

export const PALETTES: Record<Theme, Palette> = {
  light: {
    bg: '#F6FAFD', surface: '#FFFFFF', text: '#0A1931', muted: '#4A7192', grid: 'rgba(10,25,49,.08)',
    primary: '#1A3D63', primarySoft: '#4A7FA7', accent: '#B3CFE5', accentLight: '#D6E6F2',
    panel: '#0A1931', panelText: '#F6FAFD', keyword: '#B3CFE5', string: '#9CC0DD', number: '#D6E6F2', comment: '#7FA9CB', lineNo: '#4A7FA7',
    good: '#4A7FA7', warn: '#2F6390', bad: '#0A1931', track: '#D6E6F2',
    shell: '#1A3D63', joint: '#0A1931', metal: '#B3CFE5', glow: '#F6FAFD', light: '#F6FAFD',
  },
  dark: {
    bg: '#060606', surface: '#0C0C0C', text: '#FAFAFA', muted: '#9A9A9A', grid: 'rgba(255,255,255,.08)',
    primary: '#F2F2F2', primarySoft: '#BDBDBD', accent: '#171717', accentLight: '#2A2A2A',
    panel: '#0B1215', panelText: '#FAFAFA', keyword: '#FFFFFF', string: '#BDBDBD', number: '#D4D4D4', comment: '#6B6B6B', lineNo: '#3A3A3A',
    good: '#6B6B6B', warn: '#BDBDBD', bad: '#FAFAFA', track: '#171717',
    shell: '#2A2A2A', joint: '#0C0C0C', metal: '#9A9A9A', glow: '#FFFFFF', light: '#E5E5E5',
  },
}

const read = (): Theme => (document.documentElement.dataset.theme === 'dark' ? 'dark' : 'light')

const ThemeCtx = createContext<{ theme: Theme; toggle: () => void }>({ theme: 'light', toggle: () => {} })

export function ThemeProvider({ children }: { children: ReactNode }) {
  const [theme, setTheme] = useState<Theme>(read)
  useLayoutEffect(() => {
    document.documentElement.dataset.theme = theme
    try { localStorage.setItem('bugaware.theme', theme) } catch { /* storage blocked */ }
  }, [theme])
  const toggle = useCallback(() => setTheme(t => (t === 'dark' ? 'light' : 'dark')), [])
  return <ThemeCtx.Provider value={{ theme, toggle }}>{children}</ThemeCtx.Provider>
}

export const useTheme = () => useContext(ThemeCtx)
export const usePalette = () => PALETTES[useTheme().theme]
