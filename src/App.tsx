import React, { useEffect, useMemo, useRef, useState } from 'react'
import {
  Alarm, DETS, DetKey, Results, TF_LABEL, detName, fileLabel, guarantee, lastRefIndex, partsOf, pct, pctK, prettyBand,
} from './data'
import { AlarmStrip, CmmRows, ControlChart, OpMatrix, PsdPrint, Spectrum, Trace, WindowStrip } from './figures'
import { LOG_TICKS, More, Plate, Seg, Tally, logPos, tickLabel, useTip } from './ui'

const STEPS = [
  { id: 'data', short: 'Three mills' },
  { id: 'stale', short: 'Limits go stale' },
  { id: 'copy', short: 'Copying fails' },
  { id: 'fingerprint', short: 'AI fingerprint' },
  { id: 'verdict', short: 'Drift or defect' },
  { id: 'requal', short: 'Re-qualify' },
  { id: 'cost', short: 'What it saves' },
  { id: 'proof', short: 'How we know' },
]
const DEFAULT_CELL = 'M01-OP05'
const REPO = 'https://github.com/ibi-raheel/cnc-process-window-lens/blob/main/METHOD.md'

export default function App() {
  const [r, setR] = useState<Results | null>(null)
  const [err, setErr] = useState(false)
  useEffect(() => {
    fetch(`${import.meta.env.BASE_URL}data/results.json`)
      .then((x) => (x.ok ? x.json() : Promise.reject()))
      .then(setR)
      .catch(() => setErr(true))
  }, [])
  if (err) return <p style={{ padding: 24 }}>The results file did not load. Reload the page, or run the pipeline to regenerate public/data/results.json.</p>
  if (!r) return <p className="mono" style={{ padding: 24, color: 'var(--ink-3)' }}>Loading 1,702 recordings of results…</p>
  return <Story r={r} />
}

function useActiveStep() {
  const [active, setActive] = useState(0)
  const [progress, setProgress] = useState(0)
  useEffect(() => {
    const els = STEPS.map((s) => document.getElementById(s.id)).filter(Boolean) as HTMLElement[]
    const io = new IntersectionObserver(
      (entries) => {
        entries.forEach((e) => {
          if (e.isIntersecting) setActive(els.indexOf(e.target as HTMLElement))
        })
      },
      { rootMargin: '-45% 0px -50% 0px' },
    )
    els.forEach((el) => io.observe(el))
    const onScroll = () => {
      const first = els[0], last = els[els.length - 1]
      if (!first || !last) return
      const start = first.offsetTop - window.innerHeight * 0.5
      const end = last.offsetTop + last.offsetHeight - window.innerHeight
      setProgress(Math.min(1, Math.max(0, (window.scrollY - start) / Math.max(1, end - start))))
    }
    onScroll()
    window.addEventListener('scroll', onScroll, { passive: true })
    return () => { io.disconnect(); window.removeEventListener('scroll', onScroll) }
  }, [])
  return { active, progress }
}

function Story({ r }: { r: Results }) {
  const [cell, setCell] = useState(DEFAULT_CELL)
  const { active, progress } = useActiveStep()
  const good = r.example.recordings.find((x) => x.label === 0)!
  return (
    <>
      <header className="titleblock">
        <div className="tb-grid">
          <div className="tb-name">
            <h1>CNC Process-Window Lens</h1>
            <p>
              How fast does a machining operation's vibration baseline go stale, and how many good parts does it take to trust it again?{' '}
              <strong>Re-qualify on about 20 parts. Don't copy.</strong>
            </p>
          </div>
          <dl className="tb-fields">
            <div className="field"><dt>Data</dt><dd>{r.dataset.n_recordings.toLocaleString()} recordings, {r.dataset.n_bad} bad</dd></div>
            <div className="field"><dt>Machines</dt><dd>3 CNC mills, <span style={{ whiteSpace: 'nowrap' }}>M01–M03</span></dd></div>
            <div className="field"><dt>Operations</dt><dd>OP00–OP14 per machine</dd></div>
            <div className="field"><dt>Period</dt><dd>Feb 2019 – Aug 2021</dd></div>
            <div className="field"><dt>Sensor</dt><dd>3-axis accel., 2&nbsp;kHz</dd></div>
            <div className="field"><dt>Limits</dt><dd>split-conformal, 5%</dd></div>
          </dl>
        </div>
        <Trace rec={good} title={`One recording · ${r.example.machine} · ${r.example.op} · ${TF_LABEL(good.tf)} · good part · ${good.duration_s.toFixed(1)} s`} />
      </header>

      <div className="mbar" aria-hidden="true">
        <div className="mbar-row">
          <b>{STEPS[active].short}</b>
          <span>{active + 1} / {STEPS.length}</span>
        </div>
        <div className="mbar-track"><i style={{ transform: `scaleX(${Math.max(0.02, progress)})` }} /></div>
      </div>

      <div className="frame">
        <nav className="rail" aria-label="Steps">
          <div className="rail-inner">
            <ol>
              <span className="rail-fill" style={{ height: 'calc(100% - 32px)', transform: `scaleY(${active / (STEPS.length - 1)})` }} />
              {STEPS.map((s, i) => (
                <li key={s.id}>
                  <a href={`#${s.id}`} className={i === active ? 'on' : i < active ? 'done' : ''} aria-current={i === active ? 'step' : undefined}>
                    <span className="box">{i + 1}</span>
                    {s.short}
                  </a>
                </li>
              ))}
            </ol>
            <div className="rail-foot">
              <b>Following</b>
              {cell.replace('-', ' · ')}
            </div>
          </div>
        </nav>
        <main>
          <StepData r={r} cell={cell} setCell={setCell} />
          <StepStale r={r} cell={cell} setCell={setCell} />
          <StepCopy r={r} />
          <StepFingerprint r={r} cell={cell} setCell={setCell} />
          <StepVerdict r={r} />
          <StepRequal r={r} />
          <StepCost r={r} />
          <StepProof r={r} />
        </main>
      </div>

      <footer className="foot">
        <p>
          Data: Bosch CNC Machining dataset (Tnani, Feil, Diepold, <i>Procedia CIRP</i> 107, 2022), CC BY 4.0. Every number on this page is
          measured by the pipeline and precomputed; no simulated data.
        </p>
        <p className="end">More work at <a href="https://ibiraheel.com">ibiraheel.com</a></p>
      </footer>
    </>
  )
}

function StepHead({ n, title, children }: { n: number; title: string; children: React.ReactNode }) {
  return (
    <div className="step-head">
      <div className="step-no" aria-hidden="true">{n}</div>
      <div>
        <h2><span className="sr">Step {n}. </span>{title}</h2>
        {children}
      </div>
    </div>
  )
}

function Picker({ r, cell, setCell }: { r: Results; cell: string; setCell: (c: string) => void }) {
  const [m, op] = cell.split('-')
  const cells = r.ai.parts.cells
  const ops = cells.filter((c) => c.startsWith(m)).map((c) => c.split('-')[1])
  return (
    <>
      <label className="ctl">
        <span>Machine</span>
        <Seg label="Machine" value={m} options={['M01', 'M02', 'M03'].map((v) => ({ v, label: v }))}
          onChange={(v) => setCell(cells.includes(`${v}-${op}`) ? `${v}-${op}` : cells.find((c) => c.startsWith(v))!)} />
      </label>
      <label className="ctl">
        <span>Operation</span>
        <select className="sel" value={op} onChange={(e) => setCell(`${m}-${e.target.value}`)}>
          {ops.map((o) => <option key={o} value={o}>{o}</option>)}
        </select>
      </label>
    </>
  )
}

/* ============ 1 ============ */
function StepData({ r, cell, setCell }: { r: Results; cell: string; setCell: (c: string) => void }) {
  const t = useTip()
  const c = r.cells.find((x) => `${x.machine}-${x.op}` === cell)!
  return (
    <section className="step" id="data">
      <StepHead n={1} title="Three mills, 15 operations, 2.5 years of real vibration">
        <p>Each recording is one part going through one operation on one machine, labelled good or bad by the dataset authors. Pick an operation to follow it through the story.</p>
      </StepHead>
      <Plate
        head={[['Fig. 1', 'Operation matrix'], ['Cell value', 'recordings'], ['Marks', 'bad parts'], ['Following', `${c.machine} · ${c.op}`]]}
        bodyRef={t.host}
        foot={<><span><b>{r.dataset.n_recordings.toLocaleString()}</b> recordings</span><span><b>{r.dataset.n_bad}</b> labelled bad (4.1%)</span><span><b>6</b> timeframes, Feb 2019 – Aug 2021</span><span><b>43</b> operations modelled</span></>}
      >
        <OpMatrix r={r} cell={cell} onPick={setCell} tip={t.show} />
        <div onMouseLeave={t.hide} />
        {t.node}
      </Plate>
      <More>
        <p>Tri-axial accelerometers inside three brownfield CNC milling machines record each operation at 2 kHz. M03 has operations OP00–OP13, and M03 OP03 has only 13 good recordings, too few to set limits, so it is skipped. That leaves 43 machine × operation cells.</p>
        <p>A bad label means the dataset authors marked the process as anomalous; it is not a measured dimensional defect. There are only 70 bad recordings, 0–4 per cell, so detection is always pooled across cells with cluster-aware intervals.</p>
        <p>Licence: data CC BY 4.0. Tnani, Feil, Diepold, <i>Smart Data Collection System for Brownfield CNC Milling Machines</i>, Procedia CIRP 107 (2022) 131–136.</p>
      </More>
    </section>
  )
}

/* ============ 2 ============ */
function StepStale({ r, cell, setCell }: { r: Results; cell: string; setCell: (c: string) => void }) {
  const t = useTip()
  const [det, setDet] = useState<DetKey>('t2spe')
  const TF = r.dataset.timeframes
  const ref = lastRefIndex(r, cell)
  const [upTo, setUpTo] = useState(5)
  const lo = ref
  const cur = Math.max(upTo, lo)
  const parts = useMemo(() => partsOf(r, cell, det), [r, cell, det])
  const laterGood = parts.filter((p) => p.later && p.label === 0 && p.tf <= cur)
  const flagged = laterGood.filter((p) => p.u > 1).length
  const drift = r.ai.arms.align.seeds['0']['drift_alpha_0.05']
  const pooled = cur > 1 ? drift[det].far_by_tf[TF[cur]] : null
  const range = DETS.map((d) => drift[d.key].far_later.rate)
  return (
    <section className="step" id="stale">
      <StepHead n={2} title="A qualified baseline goes stale">
        <p>Limits set on the 2019 qualification run hold at 4.4–4.7% false alarms inside that run, as designed. On the 762 good parts made afterwards they flagged <strong>{pct(Math.min(...range), 0)}–{pct(Math.max(...range), 0)}</strong>, depending on the chart.</p>
        <p>Drag through time to watch it happen.</p>
      </StepHead>
      <div className="controls">
        <Picker r={r} cell={cell} setCell={setCell} />
        <label className="ctl">
          <span>Chart</span>
          <Seg label="Chart" value={det} onChange={setDet} options={DETS.map((d) => ({ v: d.key, label: d.short }))} />
        </label>
      </div>
      <Plate
        head={[['Fig. 2', 'Control chart'], ['Chart', detName(det)], ['Operation', cell.replace('-', ' · ')], ['Score', 'u = score ÷ limit']]}
        bodyRef={t.host}
        foot={
          cur <= lo ? (
            <span>Inside qualification the limit is calibrated to flag 5% of good parts.</span>
          ) : (
            <>
              <span>This operation, after qualification through {TF_LABEL(TF[cur])}: <b className={flagged ? 'sig' : ''}>{flagged} of {laterGood.length}</b> good parts flagged</span>
              {pooled && <span>All 43 operations, {TF_LABEL(TF[cur])}: <b>{pctK(pooled.k, pooled.n)}</b> ({pooled.k}/{pooled.n})</span>}
            </>
          )
        }
      >
        <div className="legend">
          <span><svg width="10" height="10"><circle cx="5" cy="5" r="3.6" fill="var(--paper)" stroke="var(--ink)" strokeWidth="1.3" /></svg>good part</span>
          <span><svg width="10" height="10"><circle cx="5" cy="5" r="4" fill="var(--signal)" /></svg>false alarm on a good part</span>
          <span><svg width="10" height="10"><path d="M1.5 1.5l7 7M1.5 8.5l7-7" stroke="var(--ink)" strokeWidth="1.8" /></svg>bad part</span>
          <span><svg width="10" height="10"><path d="M5 1l4 8H1z" fill="var(--ink-2)" /></svg>above 3, clipped</span>
        </div>
        <div onMouseLeave={t.hide}>
          <ControlChart r={r} cell={cell} det={det} upTo={cur} tip={t.show} />
        </div>
        <div className="scrub">
          <label htmlFor="scrub">Production through</label>
          <input id="scrub" type="range" min={lo} max={5} step={1} value={cur} onChange={(e) => setUpTo(+e.target.value)}
            aria-valuetext={`through ${TF_LABEL(TF[cur])}`} />
          <output htmlFor="scrub">{TF_LABEL(TF[cur])}</output>
        </div>
        {t.node}
      </Plate>
      <More>
        <StaleDetails r={r} />
      </More>
    </section>
  )
}

function StaleDetails({ r }: { r: Results }) {
  const TF = r.dataset.timeframes
  const drift = r.ai.arms.align.seeds['0']['drift_alpha_0.05']
  const crop = r.ai.arms.crop.seeds['0']['drift_alpha_0.05']
  const cal = (r.ai as any).calibration
  return (
    <>
      <p>Each chart's limit is a per-machine split-conformal limit on the qualification parts' out-of-sample scores, designed to flag 5% of good parts. Scores are shown as u = (score − machine median) ÷ (limit − median), so u &gt; 1 is an alarm on every chart. Windows are program-aligned (see step 4).</p>
      <h4>False alarms on later good parts, all operations</h4>
      <div className="tbl-wrap">
        <table className="rep">
          <thead><tr><th>Chart</th>{TF.slice(2).map((t) => <th key={t} className="r">{TF_LABEL(t)}</th>)}<th className="r">All later</th><th className="r">Prefix arm</th></tr></thead>
          <tbody>
            {DETS.map((d) => (
              <tr key={d.key}>
                <td>{d.name}</td>
                {TF.slice(2).map((t) => { const x = drift[d.key].far_by_tf[t]; return <td key={t} className="r">{pctK(x.k, x.n)}</td> })}
                <td className="r">{pctK(drift[d.key].far_later.k, drift[d.key].far_later.n)} [{pct(drift[d.key].far_later.lo!)}, {pct(drift[d.key].far_later.hi!)}]</td>
                <td className="r">{pctK(crop[d.key].far_later.k, crop[d.key].far_later.n)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p style={{ marginTop: 12 }}>The textbook 3σ Shewhart limit is miscalibrated even inside qualification: {pct(cal.rms.loo_far_3sigma.rate)} leave-one-out false alarms on 831 parts against 0.27% nominal, eight times the design rate.</p>
      <h4>Why it drifts</h4>
      <p>More of the drift is in spectral shape than in overall level: a shape-only monitor flags 41.9% of later good parts, a gain-only monitor 24.3%. Some shifts hit a whole machine at once; on M03 in Feb 2020, x-axis 100–178 Hz power rose a median 5.2σ in 13 of 13 operations. That points to one re-baseline per machine, not 13 investigations. The printout below is the most-shifted operation, {r.root_cause.psd.machine} {r.root_cause.psd.op} ({r.root_cause.psd.axis}-axis): 10–100 Hz power is {Math.abs(r.root_cause.psd.mean_log_diff_10_100).toFixed(2)} decades lower in {TF_LABEL(r.root_cause.psd.later_timeframe)}.</p>
      <PsdPrint psd={r.root_cause.psd} />
    </>
  )
}

/* ============ 3 ============ */
function StepCopy({ r }: { r: Results }) {
  const rp = r.ramp
  const rc10 = rp.variants.recentre.find((v) => v.N === 10)!
  const rs10 = rp.variants.recentre_rescale.find((v) => v.N === 10)!
  const rows = [
    { name: 'Copied unchanged', sub: 'lead machine’s monitor as is', v: rp.copy.far.rate, lo: rp.copy.far.lo!, hi: rp.copy.far.hi!, sig: true },
    { name: 'Re-centred on 10 parts', sub: 'copied loadings, new mean', v: rc10.far.rate, lo: rc10.far.lo!, hi: rc10.far.hi! },
    { name: 'Re-centred + re-scaled', sub: 'first 10 parts in order', v: rs10.far.rate, lo: rs10.far.lo!, hi: rs10.far.hi! },
    { name: 'Re-centred + re-scaled', sub: '10 random qualification parts', v: rs10.far_random_subset_mean, lo: rs10.far_random_subset_ci[0], hi: rs10.far_random_subset_ci[1] },
    { name: 'Own qualification', sub: '16–25 parts on the new machine', v: rp.own_qualification.far.rate, lo: rp.own_qualification.far.lo!, hi: rp.own_qualification.far.hi! },
  ]
  const lomo = r.ai.arms.align.lomo
  const nm = (m: string) => lomo[m].rows['N=20/first']
  return (
    <section className="step" id="copy">
      <StepHead n={3} title="Copying a baseline to a new machine fails">
        <p>Copied unchanged to a sister machine, a qualified T² monitor flagged <strong>{pct(rp.copy.far.rate)}</strong> of good parts against a 1% design rate. Re-centring on the new machine's parts is not enough; each machine also has its own spread.</p>
      </StepHead>
      <Plate
        head={[['Fig. 3', 'Baseline transfer'], ['Monitor', 'PCA T², calibrated'], ['Design rate', '1%'], ['Drift', 'none: same periods']]}
        foot={<><span>Measured on {rp.copy.far.n.toLocaleString()} good parts across 6 machine pairs.</span><span>Bars: 95% interval</span></>}
      >
        <div className="rq" style={{ gridTemplateColumns: 'minmax(120px, 210px) minmax(0, 1fr) 70px' }}>
          <div className="rq-sp" />
          <div className="rq-axis log">{LOG_TICKS.map((tk) => <span key={tk} style={{ left: `${logPos(tk)}%` }}>{tickLabel(tk)}</span>)}</div>
          <div className="rq-sp" />
          {rows.map((x, i) => (
            <div className="rq-row" key={i}>
              <div className="rq-name">{x.name}<small>{x.sub}</small></div>
              <div className="rq-track">
                {LOG_TICKS.map((tk) => <div key={tk} className="rq-grid" style={{ left: `${logPos(tk)}%` }} />)}
                <div className="rq-band" style={{ transform: `scaleX(${logPos(0.01) / 100})`, background: 'var(--tol)' }} />
                <div className="rq-nom" style={{ transform: `translateX(${logPos(0.01)}%)` }} />
                <div className="rq-ci" style={{ transform: `translateX(${logPos(x.lo)}%) scaleX(${(logPos(x.hi) - logPos(x.lo)) / 100})` }} />
                <div className={`rq-pt${x.sig ? ' ai' : ''}`} style={{ transform: `translateX(${logPos(x.v)}%)` }}><i /></div>
              </div>
              <div className="rq-det" style={{ alignItems: 'flex-end', paddingLeft: 8 }}>
                <span className={x.sig ? 'sig' : ''}>{pct(x.v)}</span>
              </div>
            </div>
          ))}
        </div>
        <div className="legend" style={{ marginTop: 12, marginBottom: 0 }}>
          <span><i className="sw" style={{ background: 'var(--tol)', border: '1px solid var(--hair)' }} />within the 1% design rate</span>
          <span><svg width="14" height="12"><path d="M7 0v12" stroke="var(--ink)" strokeWidth="1.5" strokeDasharray="3 2" /></svg>design rate</span>
        </div>
      </Plate>
      <More>
        <p>Machine A's calibrated T² monitor is scored on machine B's qualification-period good parts, so no drift is mixed in. The first 10 parts in recording order are a few Feb 2019 parts plus the first Aug 2019 ones, so they under-represent the period: 7.6% is the more realistic figure for a ramp, 3.0% the best case.</p>
        <h4>Copied unchanged, by machine pair</h4>
        <div className="tbl-wrap">
          <table className="rep">
            <thead><tr><th>Lead machine</th><th>New machine</th><th className="r">False alarms</th><th className="r">95% interval</th></tr></thead>
            <tbody>
              {Object.entries(rp.copy.matrix).flatMap(([a, row]) => Object.entries(row).map(([b, v]) => (
                <tr key={a + b}><td>{a}</td><td>{b}</td><td className="r">{pctK(v.k, v.n)}</td><td className="r">{pct(v.lo!)}–{pct(v.hi!)}</td></tr>
              )))}
            </tbody>
          </table>
        </div>
        <h4>The same test with the AI charts (program-aligned windows)</h4>
        <p>Charts copied from the other two machines and re-qualified on the first 20 parts of the new one. Bad parts caught:</p>
        <div className="tbl-wrap">
          <table className="rep">
            <thead><tr><th>New machine</th>{(['rms', 't2spe', 'ridge', 'knn'] as DetKey[]).map((k) => <th key={k} className="r">{detName(k)}</th>)}</tr></thead>
            <tbody>
              {['M01', 'M02'].map((m) => (
                <tr key={m}><td>{m}</td>{(['rms', 't2spe', 'ridge', 'knn'] as DetKey[]).map((k) => <td key={k} className="r">{nm(m)[k].det_ci.k}/{nm(m)[k].det_ci.n}</td>)}</tr>
              ))}
            </tbody>
          </table>
        </div>
        <p style={{ marginTop: 10 }}>The label-free fingerprint caught 13 of 24 on M02 against 0 for the re-centred RMS copy (exact McNemar p = 0.0002), but only 6 of 27 on M01 (p = 0.13). M02's bad parts come from just 2 timeframes, so treat it as unproven. The supervised model needs labelled bad history from other machines, and the bad files share the same trimming. M03 has 3 bad parts and supports no conclusion.</p>
      </More>
    </section>
  )
}

/* ============ 4 ============ */
function StepFingerprint({ r, cell, setCell }: { r: Results; cell: string; setCell: (c: string) => void }) {
  const t = useTip()
  const TFS = r.dataset.timeframes
  const bads = r.ai.alarms
    .filter((a) => a.label === 1 && `${a.machine}-${a.op}` === cell)
    .sort((x, y) => TFS.indexOf(x.timeframe) - TFS.indexOf(y.timeframe) || x.part - y.part)
  const [row, setRow] = useState<number | null>(null)
  const a = bads.find((x) => x.row === row) ?? bads[0]
  const mf = (arm: 'align' | 'crop') => r.ai.arms[arm].matched_far['0.05']
  const W = a?.windows
  return (
    <section className="step" id="fingerprint">
      <StepHead n={4} title="The AI fingerprint">
        <p>A MiniRocket model cuts each cycle into 24 half-second windows and matches every window to the nearest one from known-good parts. It never sees a bad part, and it points at the moment in the cycle that moved.</p>
        <p>It can flag parts the RMS chart passes. But across all 16 bad parts made after qualification, at equal false alarms, it does no better than the classical charts.</p>
      </StepHead>
      <div className="controls">
        <Picker r={r} cell={cell} setCell={setCell} />
        {bads.length > 0 && (
          <div className="ctl">
            <span>Bad part</span>
            <div className="chips">
              {bads.map((b) => (
                <button key={b.row} type="button" className="chip" aria-pressed={b.row === a.row} onClick={() => setRow(b.row)}>
                  {TF_LABEL(b.timeframe).replace(' 20', " '")} #{String(b.part).padStart(3, '0')}
                </button>
              ))}
            </div>
          </div>
        )}
      </div>
      {!a ? (
        <Plate head={[['Fig. 4', 'Window match'], ['Operation', cell.replace('-', ' · ')]]}>
          <p style={{ margin: 0 }}>No bad parts on this operation. <button type="button" className="chip" aria-pressed="false" onClick={() => setCell(DEFAULT_CELL)}>Show M01 · OP05</button></p>
        </Plate>
      ) : (
        <Plate
          flush
          head={[['Fig. 4', 'Inspection of one part'], ['Part', `${a.machine} · ${a.op} · ${TF_LABEL(a.timeframe)} · bad #${String(a.part).padStart(3, '0')}`], ['Windows', '24 × 0.51 s, program-aligned']]}
        >
          <div className="plate-body" ref={t.host} onMouseLeave={t.hide}>
            <WindowStrip a={a} tip={t.show} />
            <div className="legend" style={{ marginTop: 10, marginBottom: 0, display: 'grid', gap: 4 }}>
              {W.worst.map((ws, i) => (
                <span key={i}>
                  <b className="sig" style={{ minWidth: 14 }}>{i + 1}</b>
                  {ws.t_s.toFixed(1)} s ({Math.round(ws.cycle_frac * 100)}% into the cycle) matched {fileLabel(ws.matched_file)} at {ws.matched_t_s.toFixed(1)} s
                </span>
              ))}
              <span style={{ display: 'block' }}>Band that moved most against those good windows: <b>{prettyBand(W.band_diff_db_top[0].band)}, {W.band_diff_db_top[0].db > 0 ? '+' : ''}{W.band_diff_db_top[0].db.toFixed(1)} dB</b></span>
            </div>
            {t.node}
          </div>
          <div className="split">
            <div>
              <p className="subhead">What each chart said about this part</p>
              <CmmRows a={a} />
            </div>
            <div>
              <p className="subhead">All 16 later bad parts, every chart at 5% false alarms</p>
              <EqualFar label="Program-aligned windows" m={mf('align')} />
              <EqualFar label="Start-anchored prefix" m={mf('crop')} muted />
              <div className="stamp" style={{ marginTop: 14 }}>
                <b>Gate: not passed</b>
                <span>Neither AI model beat the best classical chart in all 3 seeds on both window sets.</span>
              </div>
            </div>
          </div>
        </Plate>
      )}
      <More>
        <FingerprintDetails r={r} />
      </More>
    </section>
  )
}

function EqualFar({ label, m, muted }: { label: string; m: Record<string, { k_det: number; n_bad: number }>; muted?: boolean }) {
  return (
    <div style={{ marginBottom: 10, opacity: muted ? 0.78 : 1 }}>
      <div className="mono" style={{ fontSize: 12, color: 'var(--ink-3)', margin: '0 0 4px' }}>{label}</div>
      <table className="cmm">
        <tbody>
          {DETS.map((d) => (
            <tr key={d.key}>
              <td style={{ padding: '4px 8px 4px 0', fontFamily: 'var(--display)', fontSize: 14 }}>{d.short}{d.key === 'cycle' && '*'}</td>
              <td style={{ padding: '4px 0' }}><Tally k={m[d.key].k_det} n={m[d.key].n_bad} sig={d.key === 'knn'} /></td>
              <td className="r" style={{ padding: '4px 0 4px 8px', whiteSpace: 'nowrap' }}>{m[d.key].k_det} / {m[d.key].n_bad}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function FingerprintDetails({ r }: { r: Results }) {
  const reg = r.ai.regression
  const seeds = r.ai.decision.by_seed
  const fp = r.ai.arms.align.fingerprints
  return (
    <>
      <h4>The shortcut that had to be removed first</h4>
      <p>The bad files in this dataset look like trimmed excerpts. Every one is shorter than a normal cycle (median {reg.bad_duration_ratio_quantiles['0.5']}× the qualification median), none of the 70 has a length that is a whole number of the logger's 1,024-sample blocks (against {pct(reg.good_length_multiple_of_1024_share, 0)} of good files), and {reg.bad_starting_over_1s_late} of 70 start more than 1 s late in the program (median {reg.start_offset_s_quantiles.bad['0.5']} s). Length alone ranks bad parts at AUC {reg.duration_only_auc}, start offset alone at {reg.start_offset_auc}.</p>
      <p>* So the cycle-time chart's lead most likely measures how the files were cut, not the machine. The AI models were therefore scored on a common start-anchored prefix (hides length), then on program-aligned windows found by matching each recording's vibration envelope to its operation's template, without labels (hides length and start). On the prefix the fingerprint looked like 6 catches against 1 for RMS; aligned, it is 4 against 3.</p>
      <h4>Gate, decided before the first run</h4>
      <p>A model passes only if, in all 3 seeds and on both window sets, (A) its like-for-like AUC beats the best classical chart with a paired cluster-bootstrap interval above 0, and (B) at 5% false alarms on later parts it catches clearly more bad parts than the RMS chart.</p>
      <div className="tbl-wrap">
        <table className="rep">
          <thead><tr><th>Model</th><th>Windows</th><th>Seed</th><th className="r">AUC</th><th className="r">Δ vs best classical [95%]</th><th>A</th><th className="r">Caught at 5% (RMS)</th><th>B</th></tr></thead>
          <tbody>
            {(['ridge', 'knn'] as const).flatMap((h) => (['crop', 'align'] as const).flatMap((arm) => seeds[arm][h].map((s: any, i: number) => (
              <tr key={h + arm + i}>
                <td>{h === 'ridge' ? 'Supervised' : 'Fingerprint'}</td><td>{arm === 'crop' ? 'prefix' : 'aligned'}</td><td>{i}</td>
                <td className="r">{s.auc.toFixed(3)}</td>
                <td className="r">{s.A_delta_auc.est >= 0 ? '+' : ''}{s.A_delta_auc.est.toFixed(3)} [{s.A_delta_auc.ci[0].toFixed(3)}, {s.A_delta_auc.ci[1].toFixed(3)}]</td>
                <td>{s.A_pass ? 'yes' : 'no'}</td>
                <td className="r">{Math.round(s.det_at_far * 16)} of 16 ({Math.round(s.rms_det_at_far * 16)})</td>
                <td>{s.B_pass ? 'yes' : 'no'}</td>
              </tr>
            ))))}
          </tbody>
        </table>
      </div>
      <p style={{ marginTop: 10 }}>Resampling whole machine × timeframe episodes instead of strata or parts, no model passes either condition in any arm or seed. The 16 later bad parts come from only 6 episodes.</p>
      <h4>Fingerprints from an earlier prototype, re-checked</h4>
      <p>{fp.map((f) => `${f.cell.replace('-', ' ')} at ${f.lo}–${f.hi} Hz: ${f.n_match_top3} of ${f.n_bad}`).join('; ')}. Only M01 OP05 reproduces: all 4 bad parts peak at 178–205 Hz on all three axes, with +16 to +26 dB against their matched good windows, while the RMS chart reads normal.</p>
    </>
  )
}

/* ============ 5 ============ */
function StepVerdict({ r }: { r: Results }) {
  const t = useTip()
  const later = useMemo(() => r.ai.alarms.filter((a) => a.later && a.drift_share != null), [r])
  const def = later.find((a) => a.machine === 'M01' && a.op === 'OP05' && a.label === 1) ?? later[0]
  const [sel, setSel] = useState(def.row)
  const a = later.find((x) => x.row === sel)!
  const lab = r.ai.arms.align.drift_subspace['alarm_labels_at_alpha_0.05'].knn
  const isDrift = (a.drift_share ?? 0) >= 0.5
  return (
    <section className="step" id="verdict">
      <StepHead n={5} title="Drift or defect?">
        <p>Each alarm gets a verdict: does its band pattern point along the directions benign drift took in earlier periods (re-baseline), or outside them (investigate)? Click an alarm to see the band that fired.</p>
        <p>The honest result: drift and defects overlap, so the verdict is shown here but not recommended for triage.</p>
      </StepHead>
      <div className="controls">
        <label className="ctl">
          <span>Alarm</span>
          <select className="sel" value={sel} onChange={(e) => setSel(+e.target.value)} style={{ maxWidth: '100%' }}>
            {later.map((x) => (
              <option key={x.row} value={x.row}>{x.label ? 'bad' : 'good'} · {x.machine} {x.op} · {TF_LABEL(x.timeframe)} #{String(x.part).padStart(3, '0')}</option>
            ))}
          </select>
        </label>
      </div>
      <Plate
        flush
        head={[['Fig. 5', 'Alarm triage'], ['Chart', 'MiniRocket fingerprint'], ['Alarms', `${later.length} after qualification`], ['Cut', 'drift share 0.5']]}
      >
        <div className="plate-body" ref={t.host} onMouseLeave={t.hide}>
          <AlarmStrip alarms={later} sel={sel} onSel={setSel} tip={t.show} />
          {t.node}
        </div>
        <div className="split">
          <div>
            <p className="subhead">Band attribution · {a.machine} · {a.op} · {TF_LABEL(a.timeframe)} · {a.label ? 'bad' : 'good'} part {String(a.part).padStart(3, '0')}</p>
            <Spectrum a={a} />
          </div>
          <div>
            <p className="subhead">Verdict</p>
            <div className={`stamp${isDrift ? '' : ' sig'}`}>
              <b>{isDrift ? 'Nuisance drift' : 'Out of window'}</b>
              <span>{isDrift ? 're-baseline' : 'investigate'} · drift share {a.drift_share!.toFixed(2)}</span>
            </div>
            <table className="cmm" style={{ marginTop: 14 }}>
              <tbody>
                <tr><td className="muted">Dataset label</td><td className="r">{a.label ? 'bad part' : 'good part'}</td></tr>
                <tr><td className="muted">Verdict was</td><td className="r">{(a.label === 1) !== isDrift ? 'right' : 'wrong'}</td></tr>
                <tr><td className="muted">Top band</td><td className="r">{prettyBand(a.band_z_top[0].band)}</td></tr>
                <tr><td className="muted">Charts that alarmed</td><td className="r" style={{ whiteSpace: 'normal' }}>{a.alarms.filter((k) => DETS.some((d) => d.key === k)).map((k) => DETS.find((d) => d.key === k)!.short).join(', ')}</td></tr>
              </tbody>
            </table>
            <p className="mono" style={{ fontSize: 12.5, color: 'var(--ink-2)', margin: '14px 0 0' }}>
              Across all {lab.false_alarms} false alarms of this chart, only {lab.false_alarms_labelled_drift} ({pctK(lab.false_alarms_labelled_drift, lab.false_alarms, 0)}) read as drift, and {lab.true_alarms - lab.true_alarms_labelled_out_of_window} of {lab.true_alarms} real defects did too.
            </p>
          </div>
        </div>
      </Plate>
      <More>
        <VerdictDetails r={r} />
      </More>
    </section>
  )
}

function VerdictDetails({ r }: { r: Results }) {
  const ds = r.ai.arms.align.drift_subspace
  return (
    <>
      <p>For each later period, the drift subspace is the top 3 directions of benign shifts seen <i>before</i> that period only: each operation's mean good-part band vector in earlier periods, plus machine-to-machine offsets. An alarm's drift share is the fraction of its 60-band z-vector (3 axes × 20 bands) that lies in that subspace; a random direction gives 0.05. The 0.5 cut was declared in advance.</p>
      <p>It fails because defects point the same way: the mean defect direction puts {Object.entries(ds.defect_direction_share).map(([k, v]) => `${pct(v, 0)} (${TF_LABEL(k)})`).join(', ')} of its energy in the drift subspace. Later good parts have a median share of {ds.later_good_share_median.toFixed(2)}, later bad parts {ds.later_bad_share_median.toFixed(2)}: the opposite of what a triage label needs. The machine-level sign test (step 2) stays the re-baseline trigger.</p>
      <p>The strip shows the 40 strongest false alarms of the fingerprint chart on later good parts and all 16 bad parts made after qualification. Band z-scores are against the operation's qualification parts with a shrunk standard deviation.</p>
    </>
  )
}

/* ============ 6 ============ */
const NS = [0, 5, 10, 19, 20, 50]
function StepRequal({ r }: { r: Results }) {
  const [ni, setNi] = useState(0)
  const N = NS[ni]
  const row = r.ai.arms.align.requal.first[`N=${N}`]
  const g = guarantee(N)
  const nominal = row.guarantee ?? 0.05
  const p05 = g ? logPos(g.q05) : 0, p95 = g ? logPos(g.q95) : 0
  return (
    <section className="step" id="requal">
      <StepHead n={6} title="Re-qualify, don't copy">
        <p>After a change, take the first N confirmed-good parts and recalibrate each chart's limit. <strong>Drag N:</strong> by about 20 parts every chart is back near the 4.76% the conformal rule promises on average.</p>
      </StepHead>
      <div className="controls">
        <div className="ctl" style={{ width: 'min(560px, 100%)' }}>
          <span>Confirmed-good parts after the change: N = {N}</span>
          <div className="range">
            <input type="range" min={0} max={NS.length - 1} step={1} value={ni} onChange={(e) => setNi(+e.target.value)} aria-label="Number of re-qualification parts" aria-valuetext={`N = ${N}`} />
            <div className="ticks">
              {NS.map((n, i) => <button key={n} type="button" aria-current={i === ni} onClick={() => setNi(i)}>{n}</button>)}
            </div>
          </div>
        </div>
      </div>
      <Plate
        head={[['Fig. 6', 'Re-qualification'], ['N', String(N)], ['Limit rule', N === 0 ? 'static 2019 limit' : g!.rule], ['Nominal', N === 0 ? '5% design' : pct(nominal, 2)]]}
        foot={
          <>
            {N === 0 && <span>Static 2019 limits on the later parts, designed for 5%.</span>}
            {g && g.rule === 'max-of-N' && <span>Below 19 parts no split-conformal 5% limit exists; the max-of-N limit gives 1/(N+1) = <b>{pct(g.nominal)}</b>.</span>}
            {g && g.rule === 'split conformal' && <span>k = ⌈(N+1) × 0.95⌉ = {g.k} → 1 − k/(N+1) = <b>{pct(g.nominal, 2)}</b>, on average over re-qualifications.</span>}
            {g && <span>One re-qualification's own rate: 5–95% range <b>{pct(g.q05)}–{pct(g.q95)}</b>.</span>}
            <span>{row.n_good_eval} good and {row.n_bad_eval} bad parts left to score, {row.n_groups} machine × periods.</span>
            {N === 50 && <span className="sig">At N = 50 only 3 bad parts remain: detection here means little.</span>}
          </>
        }
      >
        <div className="rq">
          <div className="rq-sp" />
          <div className="rq-axis log">{LOG_TICKS.map((tk) => <span key={tk} style={{ left: `${logPos(tk)}%` }}>{tickLabel(tk)}</span>)}</div>
          <div className="rq-det" style={{ font: "600 10.5px/1 var(--display)", letterSpacing: '0.09em', textTransform: 'uppercase', color: 'var(--ink-3)', minHeight: 0, justifyContent: 'flex-end', paddingBottom: 6, paddingLeft: 18 }}>Bad caught</div>
          {DETS.map((d) => {
            const x = row.detectors[d.key]
            const far = x.far.k / x.far.n
            const ci = x.far_cluster?.ci ?? [x.far.lo!, x.far.hi!]
            return (
              <div className="rq-row" key={d.key}>
                <div className="rq-name">{d.short}{d.key === 'cycle' && '*'}<small>{d.key === 'knn' ? 'label-free' : d.key === 'ridge' ? 'needs bad labels' : d.key === 'cycle' ? 'file length' : 'classical'}</small></div>
                <div className="rq-track">
                  {LOG_TICKS.map((tk) => <div key={tk} className="rq-grid" style={{ left: `${logPos(tk)}%` }} />)}
                  <div className="rq-band" style={{ transform: `translateX(${p05}%) scaleX(${(p95 - p05) / 100})` }} />
                  <div className="rq-nom" style={{ transform: `translateX(${logPos(nominal)}%)` }} />
                  <div className="rq-ci" style={{ transform: `translateX(${logPos(ci[0])}%) scaleX(${(logPos(ci[1]) - logPos(ci[0])) / 100})` }} />
                  <div className={`rq-pt${d.key === 'knn' ? ' ai' : ''}`} style={{ transform: `translateX(${logPos(far)}%)` }}><i /></div>
                  <div className={`rq-val${logPos(far) > 78 ? ' flip' : ''}`} style={{ transform: `translateX(${logPos(far)}%)` }}><span>{pctK(x.far.k, x.far.n)}</span></div>
                </div>
                <div className="rq-det">
                  <span>{x.det.k}/{x.det.n}</span>
                  <Tally k={x.det.k} n={x.det.n} sig={d.key === 'knn'} />
                </div>
              </div>
            )
          })}
        </div>
        <div className="legend" style={{ marginTop: 12, marginBottom: 0 }}>
          <span><svg width="14" height="12"><path d="M7 0v12" stroke="var(--ink)" strokeWidth="1.5" strokeDasharray="3 2" /></svg>nominal rate</span>
          <span><i className="sw" style={{ background: 'var(--tol)', border: '1px solid var(--hair)' }} />where one re-qualification lands, 5–95%</span>
          <span><svg width="22" height="10"><path d="M0 5h22" stroke="var(--ink-3)" /></svg>95% interval over machine × periods</span>
        </div>
      </Plate>
      <More>
        <RequalDetails r={r} />
      </More>
    </section>
  )
}

function RequalDetails({ r }: { r: Results }) {
  const crop = r.ai.arms.crop.requal.first
  const al = r.ai.arms.align.requal.first
  const rnd = r.ai.arms.align.requal.random['N=20']
  return (
    <>
      <p>Location charts are re-centred hierarchically (a machine-wide shift plus a shrunk per-operation correction); the fingerprint appends the new parts to its memory bank with no retraining. The limit is then set on the N parts' leave-one-out scores: split conformal from N = 19, max-of-N below. Because those scores are leave-one-out, 4.76% is nominal; a simulation puts the procedure at about 4.9%.</p>
      <p>The guarantee is an average. One re-qualification's rate here ranged from 0% to 29% per machine × period, which is what the Beta(1, 20) law predicts plus some non-exchangeability: pool alarms over several re-qualifications before deciding a limit is wrong.</p>
      <h4>False alarms / bad parts caught, both window sets</h4>
      <div className="tbl-wrap">
        <table className="rep">
          <thead><tr><th>N</th><th>Windows</th>{DETS.map((d) => <th key={d.key} className="r">{d.short}</th>)}</tr></thead>
          <tbody>
            {NS.flatMap((n) => [['prefix', crop], ['aligned', al]].map(([nm, src]: any) => (
              <tr key={n + nm}>
                <td>{n}</td><td>{nm}</td>
                {DETS.map((d) => { const x = src[`N=${n}`].detectors[d.key]; return <td key={d.key} className="r">{pctK(x.far.k, x.far.n)} / {x.det.k}</td> })}
              </tr>
            )))}
          </tbody>
        </table>
      </div>
      <p style={{ marginTop: 10 }}>With 20 random good parts instead of the first 20 (50 draws, aligned), false alarms run {DETS.map((d) => pct(rnd.detectors[d.key].far.rate)).join(', ')} for {DETS.map((d) => d.short).join(', ')}. No AI model beats the classical charts at any N: at N = 20 on aligned windows the fingerprint catches 11 of 16 against 10 for RMS (exact McNemar p = 1.0).</p>
    </>
  )
}

/* ============ 7 ============ */
function StepCost({ r }: { r: Results }) {
  const [min, setMin] = useState(15)
  const rows = r.ai.arms.align.cost_scenario.rows
  const get = (N: number, k: string) => rows.find((x) => x.N === N && x.detector === k)!.far
  const data = DETS.map((d) => ({ d, stale: get(0, d.key) * 1000 * min / 60, req: get(20, d.key) * 1000 * min / 60 }))
  const rawMax = Math.max(...data.map((x) => x.stale))
  const tstep = [1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000, 2000].find((s) => rawMax / s <= 5)!
  const max = Math.ceil(rawMax / tstep) * tstep
  const ticks = Array.from({ length: Math.round(max / tstep) + 1 }, (_, i) => i * tstep)
  const saved = data.map((x) => x.stale - x.req)
  const f = (h: number) => (h >= 100 ? h.toFixed(0) : h.toFixed(1))
  const clamp = (v: number) => Math.min(120, Math.max(1, Math.round(v)))
  return (
    <section className="step" id="cost">
      <StepHead n={7} title="What it saves">
        <p>Every false alarm costs someone triage time. Set the minutes per alarm; the false-alarm rates behind it are measured.</p>
        <p>At {min} min, re-qualifying on 20 parts saves <strong>{f(Math.min(...saved))}–{f(Math.max(...saved))} engineer-hours per 1,000 good parts</strong> after a drift event, depending on the chart.</p>
      </StepHead>
      <div className="controls">
        <div className="ctl">
          <span>Triage minutes per alarm</span>
          <div className="cost-in">
            <button type="button" aria-label="Fewer minutes" onClick={() => setMin((m) => clamp(m - 5))}>−</button>
            <input type="number" inputMode="numeric" min={1} max={120} value={min} aria-label="Triage minutes per alarm" onChange={(e) => setMin(clamp(+e.target.value || 1))} />
            <button type="button" aria-label="More minutes" onClick={() => setMin((m) => clamp(m + 5))}>+</button>
          </div>
        </div>
      </div>
      <Plate
        head={[['Fig. 7', 'Triage load'], ['Basis', '1,000 good parts'], ['Assumed', `${min} min / alarm`], ['Windows', 'program-aligned']]}
        foot={<><span>hours = false-alarm rate × 1,000 × minutes ÷ 60</span><span>Stale: 2019 limits · Re-qualified: first 20 good parts</span></>}
      >
        <div className="rq" style={{ gridTemplateColumns: 'minmax(96px, 172px) minmax(0, 1fr) minmax(88px, 150px)' }}>
          {data.map(({ d, stale, req }) => (
            <div className="rq-row" key={d.key}>
              <div className="rq-name">{d.short}{d.key === 'cycle' && '*'}</div>
              <div className="rq-track">
                {ticks.slice(1, -1).map((v) => <div key={v} className="rq-grid" style={{ left: `${(100 * v) / max}%` }} />)}
                <div style={{ position: 'absolute', top: '50%', left: `${(100 * req) / max}%`, width: `${(100 * (stale - req)) / max}%`, height: 1, background: 'var(--ink-3)' }} />
                <div className="rq-pt" style={{ transform: `translateX(${(100 * stale) / max}%)` }}><i style={{ background: 'var(--paper)', boxShadow: '0 0 0 1.5px var(--ink)' }} /></div>
                <div className={`rq-pt${d.key === 'knn' ? ' ai' : ''}`} style={{ transform: `translateX(${(100 * req) / max}%)` }}><i /></div>
              </div>
              <div className="rq-det mono" style={{ fontSize: 13 }}>
                <span><span className="muted">{f(stale)} h →</span> <b style={{ fontWeight: 500 }}>{f(req)} h</b></span>
              </div>
            </div>
          ))}
          <div className="rq-sp" />
          <div className="rq-axis" style={{ borderBottom: 0, borderTop: '1px solid var(--ink)', height: 24 }}>
            {ticks.map((v, i) => <span key={v} style={{ left: `${(100 * v) / max}%`, bottom: 'auto', top: 6, transform: i === 0 ? 'none' : i === ticks.length - 1 ? 'translateX(-100%)' : 'translateX(-50%)' }}>{i === ticks.length - 1 ? `${v} h` : v}</span>)}
          </div>
          <div className="rq-sp" />
        </div>
        <div className="legend" style={{ marginTop: 10, marginBottom: 0 }}>
          <span><svg width="12" height="12"><rect x="1.5" y="1.5" width="9" height="9" fill="var(--paper)" stroke="var(--ink)" strokeWidth="1.5" /></svg>stale 2019 limits</span>
          <span><svg width="12" height="12"><rect x="1" y="1" width="10" height="10" fill="var(--ink)" /></svg>re-qualified on 20 parts</span>
        </div>
      </Plate>
      <More>
        <p>This is a scenario on measured rates. The minutes per alarm are an assumption; the false-alarm rates are measured on 762 later good parts (stale) and 542 (re-qualified, the rest of each period after the first 20).</p>
        <p>On the start-anchored prefix, the same calculation at 15 minutes gives 41–200 h with stale limits and 3.7–13.4 h re-qualified; on program-aligned windows 41–233 h and 5.1–16.6 h. * The cycle-time chart's false alarms on good parts are real, but its detection numbers measure file cutting.</p>
      </More>
    </section>
  )
}

/* ============ 8 ============ */
const CHECKS: [string, string][] = [
  ['Conformal limit by hand: 20 scores at α = 0.05 give k = ⌈21 × 0.95⌉ = 20 and false alarms 1/21', 'exact'],
  ['Conformal false-alarm rate by simulation, 40,000 trials, Gaussian and heavy-tailed', 'within 3 SE'],
  ['One 20-part calibration’s rate by simulation: Beta(1, 20), mean 4.76%, 95th percentile 13.9%', 'matches'],
  ['Real data: conformal limits inside the qualification period, every chart, arm and seed', '4.4–4.8% vs 5%'],
  ['T² control limit by hand, p = 3, n = 20, α = 0.01', '18.25 = 18.25'],
  ['Every embedding fold fitted without its target machine, bad parts or post-2019 parts (27 folds)', 'none leaked'],
  ['Envelope alignment recovers known start offsets of 0, 0.5, 3 and 11 s', 'exact'],
  ['Reimplemented RMS and T²+SPE charts against the original like-for-like AUCs', '0.884 / 0.960'],
]
const LIMITS = [
  'Only 16 bad parts come after qualification, from 6 production episodes: one part is 6 points of detection.',
  'A bad label is an anomalous process, not a measured dimensional defect.',
  'No spindle speed or maintenance log: band shifts are located, not tied to a mechanism.',
  'The conformal rate is an average over re-qualifications, not a promise for any single one.',
  'The supervised model’s near-perfect ranking may still rest on residual file-cutting cues.',
]
function StepProof({ r }: { r: Results }) {
  void r
  const check = <svg viewBox="0 0 14 14" aria-hidden="true"><path d="M2 7.5l3.2 3L12 3.5" fill="none" stroke="var(--ink)" strokeWidth="1.8" /></svg>
  return (
    <section className="step" id="proof">
      <StepHead n={8} title="How we know">
        <p>Limits are checked by hand and by simulation, every AI score is out of sample, and two shortcuts hiding in the data were found and removed before any claim was made.</p>
      </StepHead>
      <Plate flush head={[['Fig. 8', 'Validation record'], ['Tests', '72 automated (pytest)'], ['Method', 'METHOD.md']]}>
        <div className="split" style={{ borderTop: 0 }}>
          <div>
            <p className="subhead">Checks</p>
            <ul className="checks">
              {CHECKS.map(([c, res]) => <li key={c}>{check}<span>{c}</span><span className="res">{res}</span></li>)}
            </ul>
          </div>
          <div>
            <p className="subhead">What this does not show</p>
            <ul className="lim">{LIMITS.map((l) => <li key={l}>{l}</li>)}</ul>
            <p className="subhead" style={{ marginTop: 18 }}>One finding from the checks</p>
            <p style={{ margin: 0, fontSize: 15.5, color: 'var(--ink-2)' }}>The textbook 3σ limit flagged 2.2% of qualification parts leave-one-out, eight times its 0.27% nominal rate. Limits here are calibrated from data instead.</p>
          </div>
        </div>
      </Plate>
      <More>
        <p>Pipeline: detrend and Welch spectra per axis; PCA T² (F-distribution limit) and SPE (Jackson–Mudholkar); MiniRocket embeddings (9,996 features) cross-fitted leave-one-machine-out for 3 seeds; a supervised dual ridge and a label-free nearest-window memory bank; per-machine split-conformal limits in exact integer arithmetic; cluster bootstraps over machine × timeframe episodes; exact McNemar tests.</p>
        <p>The gate, the comparisons and the seeds were fixed in code before the first full run. Two later reviews tightened it: a leakage review found the start-offset shortcut and added the program-aligned windows, and a statistics review re-ran the gate over whole production episodes. Neither changed the decision. Equations, every table and the reproduce commands are in <a href={REPO}>METHOD.md</a>.</p>
      </More>
    </section>
  )
}
