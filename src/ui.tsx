import React, { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react'

export function useWidth<T extends HTMLElement>(fallback = 800) {
  const ref = useRef<T>(null)
  const [w, setW] = useState(fallback)
  useLayoutEffect(() => {
    const el = ref.current
    if (!el) return
    const set = () => setW(Math.max(240, Math.floor(el.getBoundingClientRect().width)))
    set()
    const ro = new ResizeObserver(set)
    ro.observe(el)
    return () => ro.disconnect()
  }, [])
  return [ref, w] as const
}

export function useMedia(q: string) {
  const [m, setM] = useState(() => (typeof window !== 'undefined' ? window.matchMedia(q).matches : false))
  useEffect(() => {
    const mq = window.matchMedia(q)
    const on = () => setM(mq.matches)
    on()
    mq.addEventListener('change', on)
    return () => mq.removeEventListener('change', on)
  }, [q])
  return m
}

/* ---------- tooltip: positioned inside the nearest .plate-body ---------- */
export type TipState = { x: number; y: number; content: React.ReactNode } | null
export function useTip() {
  const host = useRef<HTMLDivElement>(null)
  const [tip, setTip] = useState<TipState>(null)
  const show = useCallback((e: React.MouseEvent | React.FocusEvent, content: React.ReactNode) => {
    const h = host.current
    if (!h) return
    const hr = h.getBoundingClientRect()
    let cx: number, cy: number
    if ('clientX' in e && (e as React.MouseEvent).clientX) {
      cx = (e as React.MouseEvent).clientX
      cy = (e as React.MouseEvent).clientY
    } else {
      const r = (e.target as Element).getBoundingClientRect()
      cx = r.left + r.width / 2
      cy = r.top
    }
    setTip({ x: cx - hr.left, y: cy - hr.top, content })
  }, [])
  const hide = useCallback(() => setTip(null), [])
  const node = tip ? <Tip host={host} tip={tip} /> : null
  return { host, show, hide, node }
}
function Tip({ host, tip }: { host: React.RefObject<HTMLDivElement>; tip: NonNullable<TipState> }) {
  const ref = useRef<HTMLDivElement>(null)
  const [pos, setPos] = useState({ left: tip.x + 12, top: tip.y + 12 })
  useLayoutEffect(() => {
    const el = ref.current, h = host.current
    if (!el || !h) return
    const W = h.clientWidth, tw = el.offsetWidth, th = el.offsetHeight
    let left = tip.x + 14
    if (left + tw > W - 4) left = tip.x - tw - 14
    if (left < 4) left = Math.max(4, Math.min(W - tw - 4, tip.x - tw / 2))
    let top = tip.y - th - 12
    if (top < 0) top = tip.y + 16
    setPos({ left, top })
  }, [tip, host])
  return (
    <div className="tip" ref={ref} style={pos} role="status">
      {tip.content}
    </div>
  )
}

/* ---------- report plate ---------- */
export function Plate({ head, foot, children, bodyRef, flush }: {
  head: [string, React.ReactNode][]
  foot?: React.ReactNode
  children: React.ReactNode
  bodyRef?: React.Ref<HTMLDivElement>
  flush?: boolean
}) {
  return (
    <figure className="plate">
      <div className="plate-head">
        {head.map(([k, v]) => (
          <div key={k}>
            <div className="k">{k}</div>
            <div className="v">{v}</div>
          </div>
        ))}
      </div>
      {flush ? children : <div className="plate-body" ref={bodyRef}>{children}</div>}
      {foot && <figcaption className="plate-foot">{foot}</figcaption>}
    </figure>
  )
}

export function More({ children, label = 'Show details' }: { children: React.ReactNode; label?: string }) {
  return (
    <details className="more">
      <summary>
        <span className="pm" aria-hidden="true" />
        {label}
      </summary>
      <div className="more-body">{children}</div>
    </details>
  )
}

export function Seg<T extends string | number>({ value, options, onChange, label }: {
  value: T
  options: { v: T; label: string }[]
  onChange: (v: T) => void
  label: string
}) {
  return (
    <div className="seg" role="group" aria-label={label}>
      {options.map((o) => (
        <button key={String(o.v)} type="button" aria-pressed={o.v === value} onClick={() => onChange(o.v)}>
          {o.label}
        </button>
      ))}
    </div>
  )
}

export function Tally({ k, n, sig }: { k: number; n: number; sig?: boolean }) {
  return (
    <span className={`tally${sig ? ' sigon' : ''}`} aria-label={`${k} of ${n}`}>
      {Array.from({ length: n }, (_, i) => <i key={i} className={i < k ? 'on' : ''} />)}
    </span>
  )
}

/* log scale for rates, shared by the copy ladder and the re-qualification rows */
export const LOG_MIN = 0.002
export const logPos = (v: number) => {
  const c = Math.min(1, Math.max(LOG_MIN, v))
  return (100 * (Math.log10(c) - Math.log10(LOG_MIN))) / (0 - Math.log10(LOG_MIN))
}
export const LOG_TICKS = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1]
export const tickLabel = (t: number) => `${+(t * 100).toFixed(1)}%`
