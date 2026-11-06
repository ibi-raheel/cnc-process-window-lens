import React, { useEffect, useState } from 'react'
import { createRoot } from 'react-dom/client'

type Score = { roc_auc: number; pr_auc: number; far: number; det: number; n_good: number; n_bad: number }
type Cell = { machine: string; op: string; skipped: boolean; far_static?: number | null; far_rolling?: number | null; n_good_post?: number; n_bad?: number }
type Results = {
  dataset: { n_recordings: number; n_bad: number; timeframes: string[]; alpha: number }
  scorecard: Record<string, Score>
  transfer: Record<string, Record<string, { far: number; det: number; n_good: number; n_bad: number }>>
  far_curve: Record<string, ({ rate: number; n: number } | null)[]>
  loo_far: Record<string, { rate: number; n: number }>
  cells: Cell[]
}

const pct = (v: number | null | undefined) => (v == null ? '–' : `${(v * 100).toFixed(1)}%`)
const LABEL: Record<string, string> = { rms: 'RMS Shewhart chart', pca: 'PCA T² + SPE', iso: 'Isolation Forest' }

function App() {
  const [r, setR] = useState<Results | null>(null)
  useEffect(() => { fetch('/data/results.json').then((x) => x.json()).then(setR) }, [])
  if (!r) return <p>Loading results…</p>
  const ms = Object.keys(r.transfer)
  return (
    <main style={{ fontFamily: 'system-ui, sans-serif', maxWidth: 860, margin: '2rem auto', padding: '0 1rem', lineHeight: 1.5 }}>
      <h1>CNC process-window lens</h1>
      <p>{r.dataset.n_recordings} real production vibration recordings ({r.dataset.n_bad} labelled bad), three CNC mills, 15 operations, 2019 to 2021. Interim results page; the full guided view comes later.</p>
      <h2>1. Static baseline: false-alarm rate on good parts by timeframe</h2>
      <table><thead><tr><th>Timeframe</th><th>Static</th><th>Rolling re-baseline</th><th>n good</th></tr></thead>
        <tbody>{r.dataset.timeframes.map((t, i) => (
          <tr key={t}><td>{t.replace('_', ' ')}</td><td>{r.far_curve.static[i] ? pct(r.far_curve.static[i]!.rate) : 'training'}</td>
            <td>{r.far_curve.rolling[i] ? pct(r.far_curve.rolling[i]!.rate) : '–'}</td><td>{r.far_curve.static[i]?.n ?? ''}</td></tr>))}</tbody></table>
      <h2>2. Baseline copied between machines (false-alarm rate on good parts)</h2>
      <table><thead><tr><th>Fit on \ applied to</th>{ms.map((m) => <th key={m}>{m}</th>)}</tr></thead>
        <tbody>{ms.map((a) => (<tr key={a}><th>{a}</th>{ms.map((b) => <td key={b}>{pct(r.transfer[a][b].far)}</td>)}</tr>))}</tbody></table>
      <h2>3. Detector scorecard (held-out timeframes)</h2>
      <table><thead><tr><th>Detector</th><th>ROC-AUC</th><th>PR-AUC</th><th>False alarms</th><th>Detection</th></tr></thead>
        <tbody>{Object.entries(r.scorecard).map(([k, v]) => (
          <tr key={k}><td>{LABEL[k] ?? k}</td><td>{v.roc_auc.toFixed(3)}</td><td>{v.pr_auc.toFixed(3)}</td><td>{pct(v.far)}</td><td>{pct(v.det)}</td></tr>))}</tbody></table>
      <details><summary>Show details: per-cell false-alarm rate</summary>
        <table><thead><tr><th>Machine</th><th>Op</th><th>Static</th><th>Rolling</th><th>n good</th><th>n bad</th></tr></thead>
          <tbody>{r.cells.filter((c) => !c.skipped).map((c) => (
            <tr key={c.machine + c.op}><td>{c.machine}</td><td>{c.op}</td><td>{pct(c.far_static)}</td><td>{pct(c.far_rolling)}</td><td>{c.n_good_post}</td><td>{c.n_bad}</td></tr>))}</tbody></table>
      </details>
      <p style={{ fontSize: '0.85rem' }}>Data: Bosch CNC Machining dataset (Tnani, Feil, Diepold, Procedia CIRP 107, 2022), CC-BY-4.0.</p>
    </main>)
}
createRoot(document.getElementById('root')!).render(<React.StrictMode><App /></React.StrictMode>)
