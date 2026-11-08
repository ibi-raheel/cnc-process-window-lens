// Typed views over public/data/results.json. Nothing here computes a new result:
// it only selects, formats and re-arranges numbers the pipeline exported.

export type Rate = { rate: number; lo?: number; hi?: number; k: number; n: number }

export type DetKey = 'rms' | 't2spe' | 'cycle' | 'ridge' | 'knn'

export const DETS: { key: DetKey; name: string; short: string; ai?: boolean; note?: string }[] = [
  { key: 'rms', name: 'RMS chart', short: 'RMS' },
  { key: 't2spe', name: 'T²+SPE chart', short: 'T²+SPE' },
  { key: 'cycle', name: 'Cycle-time chart', short: 'Cycle time', note: 'reads how the files were cut' },
  { key: 'ridge', name: 'MiniRocket, supervised', short: 'AI supervised', ai: true },
  { key: 'knn', name: 'MiniRocket fingerprint', short: 'AI fingerprint', ai: true },
]
export const detName = (k: string) => DETS.find((d) => d.key === k)?.name ?? k

export type Alarm = {
  row: number
  file: string
  machine: string
  op: string
  timeframe: string
  label: number
  part: number
  later: boolean
  duration_ratio: number
  scores: Record<string, number>
  alarms: string[]
  drift_share: number | null
  drift_label: string | null
  band_z_top: { band: string; z: number }[]
  band_z: number[]
  windows: {
    t_s: number[]
    dist: number[]
    worst: { t_s: number; cycle_frac: number; dist: number; matched_file: string; matched_t_s: number }[]
    band_diff_db_top: { band: string; db: number }[]
  }
}

type DetRequal = {
  far: Rate
  det: Rate
  auc: number
  far_cluster?: { rate: number; ci: [number, number]; n_groups: number; group_min: number; group_median: number; group_max: number }
}
export type RequalRow = {
  N: number
  n_groups: number
  rule: string
  guarantee: number | null
  n_good_eval: number
  n_bad_eval: number
  detectors: Record<string, DetRequal>
}

type DriftStatic = { far_later: Rate; det_later: Rate; far_by_tf: Record<string, Rate> }
type Matched = Record<string, { far: number; det: number; k_det: number; n_bad: number; n_good: number; det_ci: [number, number] }>

export type Arm = {
  seeds: Record<string, { 'drift_alpha_0.05': Record<string, DriftStatic> }>
  matched_far: Record<string, Matched>
  requal: { first: Record<string, RequalRow>; random: Record<string, any> }
  cost_scenario: { triage_minutes_assumed: number; rows: { N: number; detector: string; far: number; alarms_per_1000_good: number; engineer_hours_per_1000_good: number }[] }
  lomo: Record<string, { n_good: number; n_bad: number; rows: Record<string, Record<string, any>> }>
  drift_subspace: {
    alarm_labels_at_alpha_0_05?: never
    'alarm_labels_at_alpha_0.05': Record<string, { false_alarms: number; false_alarms_labelled_drift: number; true_alarms: number; true_alarms_labelled_out_of_window: number }>
    later_good_share_median: number
    later_bad_share_median: number
    defect_direction_share: Record<string, number>
  }
  fingerprints: { cell: string; lo: number; hi: number; axes: string; n_bad: number; n_match_top3: number }[]
  seed_summary: Record<string, { auc_by_seed: number[] }>
}

export type Cell = { machine: string; op: string; n: number; n_bad?: number; skipped: boolean }

export type Results = {
  dataset: { n_recordings: number; n_bad: number; timeframes: string[] }
  cells: Cell[]
  example: { machine: string; op: string; recordings: { label: number; tf: string; duration_s: number; env: [number, number][][] }[] }
  root_cause: { psd: { machine: string; op: string; axis: string; train_timeframes: string[]; later_timeframe: string; freq: number[]; log10_psd_train: number[]; log10_psd_later: number[]; mean_log_diff_10_100: number; mean_log_diff_100_1000: number } }
  ramp: {
    copy: { far: Rate; matrix: Record<string, Record<string, Rate>> }
    own_qualification: { far: Rate }
    variants: { recentre: { N: number; far: Rate; far_random_subset_mean: number; far_random_subset_ci: [number, number] }[]; recentre_rescale: { N: number; far: Rate; far_random_subset_mean: number; far_random_subset_ci: [number, number] }[] }
  }
  ai: {
    data: { n_later_good: number; n_later_bad: number; cells_qualified_beyond_2019: Record<string, string[]> }
    regression: Record<string, any>
    decision: { by_seed: Record<string, Record<string, any[]>>; passed_heads: string[] }
    alarms: Alarm[]
    parts: { cells: string[]; cell: number[]; tf: number[]; label: number[]; part: number[]; u: Record<string, number[]> }
    arms: { crop: Arm; align: Arm; full: Arm }
  }
}

export const TF_LABEL = (t: string) => t.replace('_', ' ')
export const pct = (v: number, d = 1) => `${(v * 100).toFixed(d)}%`
/** Percent from exact counts, as METHOD.md does. */
export const pctK = (k: number, n: number, d = 1) => `${((100 * k) / n).toFixed(d)}%`

// Attribution bands: features.py BAND_EDGES = geomspace(10, 1000, 9); windows.py splits each band above
// 100 Hz into 4 log sub-bands -> 20 bands per axis, ordered x, y, z (drift.band_names).
const BE = Array.from({ length: 9 }, (_, i) => 10 * Math.pow(100, i / 8))
export const ATTR_EDGES: number[] = (() => {
  const e = BE.slice(0, 4)
  for (let i = 4; i < 8; i++) for (let j = 0; j < 4; j++) e.push(BE[i] * Math.pow(BE[i + 1] / BE[i], j / 4))
  e.push(1000)
  return e
})()
export const bandLabel = (j: number) => `${ATTR_EDGES[j].toFixed(0)}–${ATTR_EDGES[j + 1].toFixed(0)} Hz`
export const prettyBand = (b: string) => b.replace(/(\d)-(\d)/, '$1–$2').replace(/^([xyz]) /, '$1-axis ')

/** "M01/OP05/good/M01_Aug_2019_OP05_007.h5" -> "good part 007, Aug 2019" */
export function fileLabel(f: string) {
  const m = f.match(/\/(good|bad)\/M\d\d_([A-Za-z]{3})_(\d{4})_OP\d\d_(\d+)\.h5$/)
  if (!m) return f
  return `${m[1]} part ${m[4]}, ${m[2]} ${m[3]}`
}

/** Reference (qualification) timeframe index per cell: Feb + Aug 2019 unless the cell qualified later. */
export function lastRefIndex(r: Results, cell: string) {
  const q = r.ai.data.cells_qualified_beyond_2019[cell]
  if (!q) return 1
  return Math.max(...q.map((t) => r.dataset.timeframes.indexOf(t)))
}

export type Part = { i: number; tf: number; part: number; label: number; u: number; later: boolean }
export function partsOf(r: Results, cell: string, det: DetKey): Part[] {
  const P = r.ai.parts
  const ci = P.cells.indexOf(cell)
  const ref = lastRefIndex(r, cell)
  const out: Part[] = []
  for (let i = 0; i < P.cell.length; i++) {
    if (P.cell[i] !== ci) continue
    out.push({ i, tf: P.tf[i], part: P.part[i], label: P.label[i], u: P.u[det][i], later: P.tf[i] > ref })
  }
  // production order: timeframe, then good parts by index, bad parts after the goods of their timeframe
  return out.sort((a, b) => a.tf - b.tf || a.label - b.label || a.part - b.part)
}

/** Beta(a, b) CDF for integer a in {1, 2} (all the cases the re-qualification curve needs). */
function betaCdf(x: number, a: number, b: number) {
  if (a === 1) return 1 - Math.pow(1 - x, b)
  // a = 2: second order statistic of b + 1 uniforms
  const n = b + 1
  return 1 - Math.pow(1 - x, n) - n * x * Math.pow(1 - x, n - 1)
}
function betaQ(q: number, a: number, b: number) {
  let lo = 0, hi = 1
  for (let i = 0; i < 60; i++) {
    const m = (lo + hi) / 2
    if (betaCdf(m, a, b) < q) lo = m
    else hi = m
  }
  return (lo + hi) / 2
}

/** METHOD.md section 13: nominal false-alarm rate and the spread of one calibration's rate. */
export function guarantee(N: number, alpha = 0.05) {
  if (N === 0) return null
  const k = Math.ceil(Math.round((N + 1) * (1 - alpha) * 1e9) / 1e9)
  if (k > N) {
    // max-of-N fallback: P = 1/(N+1); one calibration's rate ~ Beta(1, N)
    return { rule: 'max-of-N', k: N, nominal: 1 / (N + 1), a: 1, b: N, q05: betaQ(0.05, 1, N), q95: betaQ(0.95, 1, N) }
  }
  const a = N + 1 - k
  return { rule: 'split conformal', k, nominal: 1 - k / (N + 1), a, b: k, q05: betaQ(0.05, a, k), q95: betaQ(0.95, a, k) }
}
