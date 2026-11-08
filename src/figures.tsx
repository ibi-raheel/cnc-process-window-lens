import React, { useMemo } from 'react'
import { ATTR_EDGES, Alarm, DETS, DetKey, Results, TF_LABEL, bandLabel, fileLabel, lastRefIndex, partsOf } from './data'
import { useMedia, useWidth } from './ui'

type ShowTip = (e: React.MouseEvent | React.FocusEvent, c: React.ReactNode) => void

/* ---------- hero: one real recording as an instrument trace ---------- */
export function Trace({ rec, title }: { rec: Results['example']['recordings'][number]; title: string }) {
  const [ref, w] = useWidth<HTMLDivElement>(1200)
  const H = 40, gap = 6, top = 2, left = 22, bottom = 22
  const h = top + 3 * H + 2 * gap + bottom
  const n = rec.env[0].length
  const X = (i: number) => left + (i / (n - 1)) * (w - left - 2)
  const paths = rec.env.map((env, a) => {
    const m = Math.max(...env.map(([lo, hi]) => Math.max(Math.abs(lo), Math.abs(hi))))
    const y0 = top + a * (H + gap) + H / 2
    const Y = (v: number) => y0 - (v / m) * (H / 2)
    let d = `M${X(0)},${Y(env[0][1])}`
    for (let i = 1; i < n; i++) d += `L${X(i).toFixed(1)},${Y(env[i][1]).toFixed(1)}`
    for (let i = n - 1; i >= 0; i--) d += `L${X(i).toFixed(1)},${Y(env[i][0]).toFixed(1)}`
    return { d: d + 'Z', y0 }
  })
  const ticks: number[] = []
  for (let t = 0; t <= rec.duration_s; t += 5) ticks.push(t)
  return (
    <div className="trace" ref={ref}>
      <div className="trace-head">
        <span>{title}</span>
        <span>min/max envelope, 600 bins · each axis on its own scale</span>
      </div>
      <svg width={w} height={h} viewBox={`0 0 ${w} ${h}`} role="img" aria-label={`Vibration envelope of one recording, ${rec.duration_s.toFixed(1)} seconds, three axes`}>
        {paths.map((p, a) => (
          <g key={a}>
            <line x1={left} x2={w} y1={p.y0} y2={p.y0} stroke="var(--hair)" />
            <text className="ax" x={0} y={p.y0 + 4}>{'xyz'[a]}</text>
            <path d={p.d} fill="var(--ink)" />
          </g>
        ))}
        <g className="ax">
          <line className="base" x1={left} x2={w} y1={h - bottom + 2} y2={h - bottom + 2} />
          {ticks.map((t) => {
            const x = left + (t / rec.duration_s) * (w - left - 2)
            return (
              <g key={t}>
                <line x1={x} x2={x} y1={h - bottom + 2} y2={h - bottom + 6} className="base" />
                <text x={x} y={h - 4} textAnchor={t === 0 ? 'start' : 'middle'}>{t === 0 ? '0 s' : `${t}`}</text>
              </g>
            )
          })}
        </g>
      </svg>
    </div>
  )
}

/* ---------- step 1: operation matrix ---------- */
const OPS = Array.from({ length: 15 }, (_, i) => `OP${String(i).padStart(2, '0')}`)
const MACH = ['M01', 'M02', 'M03']

export function OpMatrix({ r, cell, onPick, tip }: { r: Results; cell: string; onPick: (c: string) => void; tip: ShowTip }) {
  const narrow = useMedia('(max-width: 760px)')
  const byKey = useMemo(() => new Map(r.cells.map((c) => [`${c.machine}-${c.op}`, c])), [r])
  const modelled = new Set(r.ai.parts.cells)
  const box = (m: string, op: string) => {
    const key = `${m}-${op}`
    const c = byKey.get(key)
    if (!c) return <div key={key} className="opc none" aria-label={`${m} ${op}: no such operation`} />
    const ok = modelled.has(key)
    const nb = c.n_bad ?? 0
    const desc = ok ? `${m} ${op}: ${c.n} recordings, ${nb} labelled bad` : `${m} ${op}: ${c.n} good recordings, too few to model`
    return (
      <button
        key={key}
        type="button"
        className={`opc${ok ? '' : ' skip'}`}
        aria-pressed={cell === key}
        aria-label={desc}
        disabled={!ok}
        onClick={() => ok && onPick(key)}
        onMouseMove={(e) => tip(e, <><b>{m} · {op}</b><br />{ok ? <>{c.n} recordings · {nb} bad</> : <>{c.n} good recordings, too few to model; skipped</>}</>)}
        onFocus={(e) => tip(e, desc)}
      >
        <span className="n">{c.n}</span>
        {ok ? <span className="bads">{Array.from({ length: nb }, (_, i) => <i key={i} />)}</span> : <span className="tag">skipped</span>}
      </button>
    )
  }
  if (narrow) {
    return (
      <div className="opm" style={{ gridTemplateColumns: '48px repeat(3, minmax(0, 1fr))' }}>
        <div />
        {MACH.map((m) => <div key={m} className="h" style={{ fontSize: 13 }}>{m}</div>)}
        {OPS.map((op) => (
          <React.Fragment key={op}>
            <div className="rh" style={{ fontSize: 13 }}>{op}</div>
            {MACH.map((m) => box(m, op))}
          </React.Fragment>
        ))}
      </div>
    )
  }
  return (
    <div className="opm" style={{ gridTemplateColumns: '52px repeat(15, minmax(0, 1fr))' }}>
      <div />
      {OPS.map((op) => <div key={op} className="h">{op}</div>)}
      {MACH.map((m) => (
        <React.Fragment key={m}>
          <div className="rh">{m}</div>
          {OPS.map((op) => box(m, op))}
        </React.Fragment>
      ))}
    </div>
  )
}

/* ---------- step 2: control chart, limit-normalised score u per part ---------- */
export function ControlChart({ r, cell, det, upTo, tip }: { r: Results; cell: string; det: DetKey; upTo: number; tip: ShowTip }) {
  const [ref, w] = useWidth<HTMLDivElement>(1000)
  const parts = useMemo(() => partsOf(r, cell, det), [r, cell, det])
  const ref_tf = lastRefIndex(r, cell)
  const TF = r.dataset.timeframes
  const mL = 40, mR = 8, mT = 28, mB = 40
  const h = w < 560 ? 280 : 340
  const iw = w - mL - mR, ih = h - mT - mB
  const gap = w < 560 ? 8 : 14
  const counts = TF.map((_, t) => parts.filter((p) => p.tf === t).length)
  // every timeframe block gets at least minW px so its label fits; the rest is shared by part count
  const minW = w < 560 ? 38 : 24
  const avail = iw - gap * (TF.length - 1)
  let fixed = new Array(TF.length).fill(false)
  let unit = avail / counts.reduce((a, b) => a + Math.max(b, 1), 0)
  for (let pass = 0; pass < 4; pass++) {
    fixed = counts.map((c) => Math.max(c, 1) * unit < minW)
    const free = counts.reduce((a, c, i) => a + (fixed[i] ? 0 : Math.max(c, 1)), 0)
    const used = fixed.filter(Boolean).length * minW
    unit = free > 0 ? (avail - used) / free : unit
  }
  const slots = counts.map((c, i) => (fixed[i] ? minW / unit : Math.max(c, 1)))
  const x0: number[] = []
  let acc = mL
  slots.forEach((s) => { x0.push(acc); acc += s * unit + gap })
  const YMIN = -1, YMAX = 3
  const Y = (u: number) => mT + ((YMAX - Math.min(YMAX, Math.max(YMIN, u))) / (YMAX - YMIN)) * ih
  const idx = new Array(TF.length).fill(0)
  const pts = parts.map((p) => {
    const k = idx[p.tf]++
    return { ...p, x: x0[p.tf] + (k + 0.5) * ((slots[p.tf] * unit) / Math.max(1, counts[p.tf])) }
  })
  const qualEnd = x0[ref_tf] + slots[ref_tf] * unit
  const cursorX = x0[upTo] + slots[upTo] * unit + gap / 2
  const name = DETS.find((d) => d.key === det)!.name
  return (
    <div ref={ref}>
      <svg width={w} height={h} viewBox={`0 0 ${w} ${h}`} role="img" aria-label={`${name}, ${cell}: limit-normalised score per part over six timeframes; the limit is u = 1`}>
        <defs>
          <pattern id="hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
            <line x1="0" y1="0" x2="0" y2="6" stroke="var(--hair)" strokeWidth="1" />
          </pattern>
        </defs>
        {/* tolerance zone: below the limit */}
        <rect x={mL} y={Y(1)} width={iw} height={Y(YMIN) - Y(1)} fill="var(--tol)" />
        {/* qualification bracket */}
        <g>
          <path d={`M${mL},${mT - 8} V${mT - 14} H${qualEnd} V${mT - 8}`} fill="none" stroke="var(--ink)" />
          <text className="lbl-d" x={mL + 4} y={mT - 18}>Qualification run · limits set here</text>
        </g>
        {/* grid */}
        <g className="ax">
          {[-1, 0, 2, 3].map((v) => (
            <g key={v}>
              <line x1={mL} x2={w - mR} y1={Y(v)} y2={Y(v)} />
              <text x={mL - 8} y={Y(v) + 4} textAnchor="end">{v}</text>
            </g>
          ))}
          <text x={mL - 8} y={Y(1) + 4} textAnchor="end" style={{ fill: 'var(--ink)' }}>1</text>
        </g>
        <line x1={mL} x2={w - mR} y1={Y(1)} y2={Y(1)} stroke="var(--ink)" strokeWidth={1.5} />
        <text className="lbl" x={mL + 6} y={Y(1) - 7} style={{ paintOrder: 'stroke', stroke: 'var(--paper)', strokeWidth: 4 }}>limit u = 1 · 5% design</text>
        {/* timeframe blocks */}
        {TF.map((t, i) => (
          <g key={t} className="ax">
            <line className="base" x1={x0[i]} x2={x0[i] + slots[i] * unit} y1={h - mB + 6} y2={h - mB + 6} />
            <text x={x0[i] + (slots[i] * unit) / 2} y={h - mB + 22} textAnchor="middle" style={{ fill: i <= upTo ? 'var(--ink-2)' : 'var(--hair)' }}>
              {w < 560 ? `${t[0]} '${t.slice(-2)}` : TF_LABEL(t)}
            </text>
            {counts[i] === 0 && <text x={x0[i] + (slots[i] * unit) / 2} y={Y(0) + 4} textAnchor="middle">–</text>}
          </g>
        ))}
        {/* points */}
        {pts.map((p) => {
          const faded = p.tf > upTo
          const above = p.u > 1
          const clipped = p.u > YMAX
          const y = Y(p.u)
          const fa = above && p.label === 0
          const col = fa ? 'var(--signal)' : 'var(--ink)'
          const s = Math.min(4.2, Math.max(2.4, unit * 0.32))
          const label = (
            <>
              <b>{TF_LABEL(TF[p.tf])} · {p.label ? 'bad' : 'good'} part {String(p.part).padStart(3, '0')}</b>
              <br />u = {p.u.toFixed(2)} {above ? (p.label ? '· alarm, true' : '· alarm on a good part') : '· inside limit'}
              {p.tf <= ref_tf && <><br />qualification part (scored leave-one-out)</>}
            </>
          )
          return (
            <g key={p.i} opacity={faded ? 0.1 : 1} style={{ transition: 'opacity 200ms ease' }} onMouseMove={(e) => tip(e, label)}>
              <rect x={p.x - Math.max(unit, 8) / 2} y={mT} width={Math.max(unit, 8)} height={ih} fill="transparent" />
              {p.label ? (
                <path d={`M${p.x - s},${y - s}L${p.x + s},${y + s}M${p.x - s},${y + s}L${p.x + s},${y - s}`} stroke="var(--ink)" strokeWidth={1.8} />
              ) : clipped ? (
                <path d={`M${p.x},${y - s - 1}L${p.x + s + 0.5},${y + s}L${p.x - s - 0.5},${y + s}Z`} fill={col} />
              ) : (
                <circle cx={p.x} cy={y} r={s} fill={fa ? col : 'var(--paper)'} stroke={col} strokeWidth={1.3} />
              )}
            </g>
          )
        })}
        {/* scrub cursor */}
        {upTo < TF.length - 1 && (
          <g>
            <line x1={cursorX} x2={cursorX} y1={mT - 4} y2={h - mB + 6} stroke="var(--ink)" strokeDasharray="3 3" />
          </g>
        )}
      </svg>
    </div>
  )
}

/* ---------- step 2 details: Welch PSD, qualification vs later ---------- */
export function PsdPrint({ psd }: { psd: Results['root_cause']['psd'] }) {
  const [ref, w] = useWidth<HTMLDivElement>(700)
  const h = 220, mL = 40, mR = 10, mT = 12, mB = 30
  const lx = (f: number) => mL + ((Math.log10(f) - 1) / 2) * (w - mL - mR)
  const sel = psd.freq.map((f, i) => [f, i] as const).filter(([f]) => f >= 10)
  const all = sel.flatMap(([, i]) => [psd.log10_psd_train[i], psd.log10_psd_later[i]])
  const lo = Math.floor(Math.min(...all)), hi = Math.ceil(Math.max(...all))
  const Y = (v: number) => mT + ((hi - v) / (hi - lo)) * (h - mT - mB)
  const line = (arr: number[]) => sel.map(([f, i], k) => `${k ? 'L' : 'M'}${lx(f).toFixed(1)},${Y(arr[i]).toFixed(1)}`).join('')
  const yt: number[] = []
  for (let v = lo; v <= hi; v++) yt.push(v)
  return (
    <div ref={ref}>
      <div className="legend">
        <span><i className="sw" style={{ background: 'var(--ink)', height: 2 }} />Qualification ({psd.train_timeframes.map(TF_LABEL).join(', ')}, n = {(psd as any).n_train})</span>
        <span><i className="sw" style={{ background: 'var(--signal)', height: 2 }} />{TF_LABEL(psd.later_timeframe)} (n = {(psd as any).n_later})</span>
      </div>
      <svg width={w} height={h} viewBox={`0 0 ${w} ${h}`} role="img" aria-label="Mean Welch power spectral density, qualification against a later timeframe">
        <g className="ax">
          {yt.map((v) => (
            <g key={v}>
              <line x1={mL} x2={w - mR} y1={Y(v)} y2={Y(v)} />
              <text x={mL - 6} y={Y(v) + 4} textAnchor="end">{v}</text>
            </g>
          ))}
          {[10, 20, 50, 100, 200, 500, 1000].map((f) => (
            <g key={f}>
              <line x1={lx(f)} x2={lx(f)} y1={mT} y2={h - mB} />
              <text x={lx(f)} y={h - 10} textAnchor="middle">{f >= 1000 ? '1 kHz' : f}</text>
            </g>
          ))}
          <line className="base" x1={mL} x2={w - mR} y1={h - mB} y2={h - mB} />
        </g>
        <path d={line(psd.log10_psd_train)} fill="none" stroke="var(--ink)" strokeWidth={1.4} />
        <path d={line(psd.log10_psd_later)} fill="none" stroke="var(--signal)" strokeWidth={1.4} />
        <text className="ax" x={mL} y={mT - 2}>log10 PSD</text>
      </svg>
    </div>
  )
}

/* ---------- step 4: windows matched to known-good ---------- */
export function WindowStrip({ a, tip }: { a: Alarm; tip: ShowTip }) {
  const [ref, w] = useWidth<HTMLDivElement>(800)
  const W = a.windows
  const L = 0.512
  const t0 = W.t_s[0], t1 = W.t_s[W.t_s.length - 1] + L
  const h = 190, mL = 40, mR = 8, mT = 26, mB = 34
  const X = (t: number) => mL + ((t - t0) / (t1 - t0)) * (w - mL - mR)
  const dmax = Math.max(...W.dist) * 1.12
  const Y = (d: number) => mT + (1 - d / dmax) * (h - mT - mB)
  const worstIdx = W.worst.map((ws) => W.t_s.findIndex((t) => Math.abs(t - ws.t_s) < 0.006))
  const ystep = dmax > 6 ? 2 : dmax > 2.5 ? 1 : 0.5
  const yt: number[] = []
  for (let v = 0; v <= dmax; v += ystep) yt.push(+v.toFixed(1))
  const dt = Math.min(L, ...W.t_s.slice(1).map((t, i) => t - W.t_s[i]))
  const span = t1 - t0
  const stepT = span > 30 ? 10 : span > 12 ? 5 : 2
  const xt: number[] = []
  for (let t = Math.ceil(t0 / stepT) * stepT; t <= t1; t += stepT) xt.push(t)
  return (
    <div ref={ref}>
      <svg width={w} height={h} viewBox={`0 0 ${w} ${h}`} role="img" aria-label="Distance from each half-second window to its nearest known-good window">
        <g className="ax">
          {yt.map((v) => (
            <g key={v}>
              <line x1={mL} x2={w - mR} y1={Y(v)} y2={Y(v)} />
              <text x={mL - 6} y={Y(v) + 4} textAnchor="end">{v}</text>
            </g>
          ))}
          <line className="base" x1={mL} x2={w - mR} y1={Y(0)} y2={Y(0)} />
          {xt.map((t) => (
            <g key={t}>
              <line className="base" x1={X(t)} x2={X(t)} y1={Y(0)} y2={Y(0) + 4} />
              <text x={X(t)} y={h - 14} textAnchor="middle">{t} s</text>
            </g>
          ))}
        </g>
        <text className="lbl-d" x={mL} y={12}>Distance to nearest known-good window</text>
        {W.t_s.map((t, i) => {
          const wi = worstIdx.indexOf(i)
          const bw = Math.max(2, X(t + dt) - X(t) - 2)
          return (
            <g key={i} onMouseMove={(e) => tip(e, <><b>Window at {t.toFixed(2)} s</b><br />distance {W.dist[i].toFixed(2)}{wi >= 0 && <><br />matched {fileLabel(W.worst[wi].matched_file)} at {W.worst[wi].matched_t_s.toFixed(1)} s</>}</>)}>
              <rect x={X(t)} y={mT} width={bw} height={Y(0) - mT} fill="transparent" />
              <rect x={X(t)} y={Y(W.dist[i])} width={bw} height={Y(0) - Y(W.dist[i])} fill={wi >= 0 ? 'var(--signal)' : '#b4b9c0'} />
              {wi >= 0 && (
                <g>
                  <rect x={X(t) + bw / 2 - 8} y={Y(W.dist[i]) - 20} width={16} height={15} fill="var(--paper)" stroke="var(--signal)" />
                  <text x={X(t) + bw / 2} y={Y(W.dist[i]) - 9} textAnchor="middle" className="lbl sigt" style={{ fontSize: 11 }}>{wi + 1}</text>
                </g>
              )}
            </g>
          )
        })}
        <text className="ax" x={w - mR} y={h - 1} textAnchor="end">time in recording</text>
      </svg>
    </div>
  )
}

/* ---------- CMM-style rows: each chart's reading of one part ---------- */
const TOL_MIN = -1, TOL_MAX = 3
const tolPos = (u: number) => (100 * (Math.min(TOL_MAX, Math.max(TOL_MIN, u)) - TOL_MIN)) / (TOL_MAX - TOL_MIN)
export function CmmRows({ a }: { a: Alarm }) {
  return (
    <div className="tbl-wrap">
      <table className="cmm">
        <thead>
          <tr>
            <th>Chart</th>
            <th className="r lim-col">Limit</th>
            <th className="r">Actual u</th>
            <th style={{ width: '34%' }}>Tolerance</th>
            <th className="r">Result</th>
          </tr>
        </thead>
        <tbody>
          {DETS.map((d) => {
            const u = a.scores[d.key]
            const al = u > 1
            return (
              <tr key={d.key}>
                <td className="nm">{d.name}{d.note && <small>{d.note}</small>}</td>
                <td className="r lim-col">1.00</td>
                <td className={`r${al ? ' sig' : ''}`}>{u.toFixed(2)}</td>
                <td>
                  <div className="tolbar" aria-hidden="true">
                    <div className="zone" style={{ width: `${tolPos(1)}%` }} />
                    <div className={`mk${al ? ' al' : ''}`} style={{ left: `${tolPos(u)}%` }} />
                    {u > TOL_MAX && <div className="over" />}
                  </div>
                </td>
                <td className="r"><span className={`verdict${al ? ' al' : ''}`}>{al ? 'Alarm' : 'Pass'}</span></td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

/* ---------- step 5: alarms by drift share ---------- */
export function AlarmStrip({ alarms, sel, onSel, tip }: { alarms: Alarm[]; sel: number; onSel: (row: number) => void; tip: ShowTip }) {
  const [ref, w] = useWidth<HTMLDivElement>(800)
  const rows = [
    { key: 0, title: w < 560 ? 'Good parts flagged' : 'Good parts flagged (the 40 strongest false alarms)', items: alarms.filter((a) => a.label === 0) },
    { key: 1, title: w < 560 ? 'Bad parts, later' : 'Bad parts made after qualification', items: alarms.filter((a) => a.label === 1) },
  ]
  const nar = w < 560
  const mL = 8, mR = 8, rowH = 64, mT = 34
  const h = mT + rows.length * (rowH + 26) + 26
  const X = (s: number) => mL + s * (w - mL - mR)
  const jitter = (row: number) => ((((row * 2654435761) >>> 0) % 1000) / 1000 - 0.5) * (rowH - 18)
  return (
    <div ref={ref}>
      <svg width={w} height={h} viewBox={`0 0 ${w} ${h}`} role="img" aria-label="Alarms placed by drift share; the cut is 0.5">
        <rect x={X(0.5)} y={mT - 6} width={X(1) - X(0.5)} height={h - mT - 20} fill="var(--tol)" />
        <text className="lbl-d" x={X(0.5) - 8} y={14} textAnchor="end">{nar ? '← Investigate' : '← Out of window: investigate'}</text>
        <text className="lbl-d" x={X(0.5) + 8} y={14}>{nar ? 'Re-baseline →' : 'Nuisance drift: re-baseline →'}</text>
        {rows.map((rw, ri) => {
          const y0 = mT + ri * (rowH + 26)
          return (
            <g key={rw.key}>
              <text className="lbl" x={mL} y={y0 + 10} style={{ fill: 'var(--ink-2)', paintOrder: 'stroke', stroke: 'var(--paper)', strokeWidth: 4 }}>{rw.title} · {rw.items.length}</text>
              <line x1={mL} x2={w - mR} y1={y0 + rowH + 6} y2={y0 + rowH + 6} stroke="var(--hair)" />
              {rw.items.map((a) => {
                const cx = X(a.drift_share ?? 0)
                const cy = y0 + 18 + (rowH - 12) / 2 + jitter(a.row) * 0.85
                const on = a.row === sel
                const lbl = <><b>{a.machine} · {a.op} · {TF_LABEL(a.timeframe)} · {a.label ? 'bad' : 'good'} part {String(a.part).padStart(3, '0')}</b><br />drift share {a.drift_share?.toFixed(2)} → {a.drift_label}</>
                return (
                  <g key={a.row} style={{ cursor: 'pointer' }} onClick={() => onSel(a.row)} onMouseMove={(e) => tip(e, lbl)}>
                    <circle cx={cx} cy={cy} r={11} fill="transparent" />
                    {a.label ? (
                      <path d={`M${cx - 4},${cy - 4}L${cx + 4},${cy + 4}M${cx - 4},${cy + 4}L${cx + 4},${cy - 4}`} stroke={on ? 'var(--signal)' : 'var(--ink)'} strokeWidth={1.8} />
                    ) : (
                      <circle cx={cx} cy={cy} r={3.8} fill="var(--paper)" stroke={on ? 'var(--signal)' : 'var(--ink)'} strokeWidth={1.3} />
                    )}
                    {on && <circle cx={cx} cy={cy} r={9} fill="none" stroke="var(--signal)" strokeWidth={1.8} />}
                  </g>
                )
              })}
            </g>
          )
        })}
        <g className="ax">
          {[0, 0.25, 0.5, 0.75, 1].map((s) => (
            <g key={s}>
              <line className="base" x1={X(s)} x2={X(s)} y1={h - 22} y2={h - 18} />
              <text x={X(s)} y={h - 4} textAnchor={s === 0 ? 'start' : s === 1 ? 'end' : 'middle'}>{s === 0.5 ? 'cut 0.5' : s}</text>
            </g>
          ))}
          <line className="base" x1={mL} x2={w - mR} y1={h - 22} y2={h - 22} />
        </g>
        <line x1={X(0.5)} x2={X(0.5)} y1={mT - 6} y2={h - 22} stroke="var(--ink)" strokeWidth={1.5} />
        <path d={`M${X(0.05)},${h - 22}l-4,-7h8z`} fill="var(--ink-3)" />
        <text className="ax" x={X(0.05) + 7} y={h - 25}>{nar ? 'random 0.05' : 'random direction 0.05'}</text>
      </svg>
    </div>
  )
}

/* ---------- step 5: band attribution as a spectrum printout ---------- */
export function Spectrum({ a }: { a: Alarm }) {
  const [ref, w] = useWidth<HTMLDivElement>(600)
  const nb = 20
  const maxAbs = Math.max(8, ...a.band_z.map((v) => Math.abs(v)))
  const lim = Math.ceil(maxAbs / 4) * 4
  const top = a.band_z.reduce((bi, v, i, arr) => (Math.abs(v) > Math.abs(arr[bi]) ? i : bi), 0)
  const mL = 30, mR = 6, rowH = 74, gap = 14, mT = 6, mB = 26
  const h = mT + 3 * rowH + 2 * gap + mB
  const lx = (f: number) => mL + ((Math.log10(f) - 1) / 2) * (w - mL - mR)
  return (
    <div ref={ref}>
      <svg width={w} height={h} viewBox={`0 0 ${w} ${h}`} role="img" aria-label="Band power z-score per axis and band, against the operation's qualification parts">
        {[0, 1, 2].map((ax) => {
          const y0 = mT + ax * (rowH + gap)
          const Y = (z: number) => y0 + rowH / 2 - (Math.max(-lim, Math.min(lim, z)) / lim) * (rowH / 2)
          let d = ''
          for (let j = 0; j < nb; j++) {
            const z = a.band_z[ax * nb + j]
            const xa = lx(ATTR_EDGES[j]), xb = lx(ATTR_EDGES[j + 1])
            d += `${j ? 'L' : 'M'}${xa.toFixed(1)},${Y(z).toFixed(1)}H${xb.toFixed(1)}`
          }
          const tj = Math.floor(top / nb) === ax ? top % nb : -1
          return (
            <g key={ax}>
              <rect x={mL} y={Y(3)} width={w - mL - mR} height={Y(-3) - Y(3)} fill="var(--tol)" />
              <line x1={mL} x2={w - mR} y1={Y(0)} y2={Y(0)} stroke="var(--hair)" />
              <text className="ax" x={mL - 6} y={Y(lim) + 9} textAnchor="end">+{lim}</text>
              <text className="ax" x={mL - 6} y={Y(0) + 4} textAnchor="end">0</text>
              <text className="ax" x={mL - 6} y={Y(-lim)} textAnchor="end">−{lim}</text>
              <text className="lbl" x={mL + 6} y={y0 + 12}>{'xyz'[ax]}-axis</text>
              {tj >= 0 && (
                <rect x={lx(ATTR_EDGES[tj])} y={Math.min(Y(0), Y(a.band_z[top]))} width={lx(ATTR_EDGES[tj + 1]) - lx(ATTR_EDGES[tj])} height={Math.abs(Y(a.band_z[top]) - Y(0))} fill="var(--signal)" />
              )}
              <path d={d} fill="none" stroke="var(--ink)" strokeWidth={1.4} />
              <line x1={mL} x2={mL} y1={y0} y2={y0 + rowH} stroke="var(--ink)" />
            </g>
          )
        })}
        <g className="ax">
          <line className="base" x1={mL} x2={w - mR} y1={h - mB + 4} y2={h - mB + 4} />
          {[10, 20, 50, 100, 200, 500, 1000].map((f) => (
            <g key={f}>
              <line className="base" x1={lx(f)} x2={lx(f)} y1={h - mB + 4} y2={h - mB + 8} />
              {(w >= 440 || (f !== 20 && f !== 500)) && (
                <text x={lx(f)} y={h - 6} textAnchor={f === 1000 ? 'end' : f === 10 ? 'start' : 'middle'}>{f === 1000 ? '1 kHz' : f === 10 ? '10 Hz' : f}</text>
              )}
            </g>
          ))}
        </g>
      </svg>
      <div className="legend" style={{ marginTop: 6 }}>
        <span><i className="sw" style={{ background: 'var(--tol)', border: '1px solid var(--hair)' }} />±3 z: normal band spread</span>
        <span><i className="sw" style={{ background: 'var(--signal)' }} />top band: {'xyz'[Math.floor(top / nb)]}-axis {bandLabel(top % nb)}, z = {a.band_z[top] > 0 ? '+' : ''}{a.band_z[top].toFixed(1)}</span>
      </div>
    </div>
  )
}
