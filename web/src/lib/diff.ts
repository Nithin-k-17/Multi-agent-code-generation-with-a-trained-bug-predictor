/** A changed region between two versions of the code (1-based line numbers in the old version). */
export interface Hunk { oldStart: number; oldEnd: number; before: string[]; after: string[] }

const norm = (l: string) => l.replace(/#.*$/, '').trim().replace(/\s+/g, ' ')

/** Line diff (LCS). Whitespace- and comment-only changes are ignored. */
export function diffHunks(oldCode: string, newCode: string): Hunk[] {
  const a = oldCode.split('\n'), b = newCode.split('\n')
  const n = a.length, m = b.length
  const dp: number[][] = Array.from({ length: n + 1 }, () => new Array(m + 1).fill(0))
  for (let i = n - 1; i >= 0; i--)
    for (let j = m - 1; j >= 0; j--)
      dp[i][j] = norm(a[i]) === norm(b[j]) ? dp[i + 1][j + 1] + 1 : Math.max(dp[i + 1][j], dp[i][j + 1])
  const hunks: Hunk[] = []
  let i = 0, j = 0, cur: Hunk | null = null
  const flush = () => {
    if (cur && (cur.before.some(l => norm(l)) || cur.after.some(l => norm(l)))) hunks.push(cur)
    cur = null
  }
  while (i < n || j < m) {
    if (i < n && j < m && norm(a[i]) === norm(b[j])) { flush(); i++; j++; continue }
    if (!cur) cur = { oldStart: Math.max(1, i + 1), oldEnd: Math.max(1, i), before: [], after: [] }
    if (j < m && (i >= n || dp[i][j + 1] >= dp[i + 1][j])) { cur.after.push(b[j]); j++ }
    else { cur.before.push(a[i]); cur.oldEnd = i + 1; i++ }
  }
  flush()
  return hunks.map(h => ({ ...h, oldEnd: Math.max(h.oldStart, h.oldEnd) }))
}
