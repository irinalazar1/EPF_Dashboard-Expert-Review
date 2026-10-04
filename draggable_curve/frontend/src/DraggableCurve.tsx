import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ComponentProps, Streamlit } from "streamlit-component-lib";

interface Args {
  values: number[];
  labels: string[];
  radius: number;
  height: number;
  forecast?: number[];
  band_lower?: number[];
  band_upper?: number[];
  flagged?: boolean[];
}

const PAD_LEFT = 48;
const PAD_RIGHT = 16;
const PAD_TOP = 26; // room for the "EUR / MWh" unit label above the topmost gridline
const PAD_BOTTOM = 32;

/**
 * Linear falloff weight for how much a point at `distance` slots away from
 * the dragged point should move. 1.0 at distance 0, straight line down to
 * 0.0 at distance == radius, untouched beyond that.
 */
function falloffWeight(distance: number, radius: number): number {
  if (radius <= 0) return distance === 0 ? 1 : 0;
  const w = 1 - distance / radius;
  return w > 0 ? w : 0;
}

const DraggableCurve: React.FC<ComponentProps> = (props) => {
  const args = props.args as Args;
  const { labels, radius, height, forecast, flagged } = args;
  const bandLower = args.band_lower;
  const bandUpper = args.band_upper;

  // `values` is the live, possibly-mid-drag array driving the chart.
  // `baseValuesRef` is a frozen snapshot taken at the moment a drag starts,
  // so every mousemove computes its delta from a fixed origin rather than
  // compounding small movements into drift.
  const [values, setValues] = useState<number[]>(args.values);
  const baseValuesRef = useRef<number[]>(args.values);
  const dragIndexRef = useRef<number | null>(null);
  const svgRef = useRef<SVGSVGElement | null>(null);

  // Which point to show a value tooltip for. Drag takes priority over
  // hover (set below via `activeIndex`) since during a drag the pointer
  // may not be exactly over the point being moved.
  const [hoverIndex, setHoverIndex] = useState<number | null>(null);

  const width = Math.max(props.width || 700, 300);
  const n = values.length;

  // Re-sync from Python if the incoming values genuinely changed (e.g. the
  // user picked a different forecast date) -- but not on every re-render,
  // or an in-progress drag would keep getting reset out from under itself.
  const argsValuesKey = args.values.join(",");
  const lastSyncedKeyRef = useRef<string>(argsValuesKey);
  useEffect(() => {
    if (argsValuesKey !== lastSyncedKeyRef.current) {
      lastSyncedKeyRef.current = argsValuesKey;
      setValues(args.values);
      baseValuesRef.current = args.values;
    }
  }, [argsValuesKey, args.values]);

  useEffect(() => {
    Streamlit.setComponentReady();
  }, []);

  useEffect(() => {
    Streamlit.setFrameHeight(height + PAD_TOP + PAD_BOTTOM + 52);
  }, [height]);

  // Axis range is frozen for the duration of an active drag, recomputed
  // only while idle. Without this, dragging one point to an extreme
  // rescales the whole y-axis live (since it's fit to current min/max on
  // every render) -- every OTHER point's pixel position then shifts too,
  // even though its actual value barely changed, which looks exactly like
  // "the other values moved the opposite way" even though they didn't.
  const axisRangeRef = useRef<[number, number]>([0, 1]);
  const isDragging = dragIndexRef.current !== null;

  const [yMin, yMax] = useMemo(() => {
    if (isDragging) {
      return axisRangeRef.current;
    }
    const allValues = [...values];
    if (bandLower) allValues.push(...bandLower);
    if (bandUpper) allValues.push(...bandUpper);
    if (forecast) allValues.push(...forecast);
    const lo = Math.min(...allValues);
    const hi = Math.max(...allValues);
    const pad = (hi - lo) * 0.15 || 1;
    const range: [number, number] = [lo - pad, hi + pad];
    axisRangeRef.current = range;
    return range;
  }, [values, bandLower, bandUpper, forecast, isDragging]);

  const plotWidth = width - PAD_LEFT - PAD_RIGHT;
  const plotHeight = height - PAD_TOP - PAD_BOTTOM;

  const xForIndex = useCallback(
    (i: number) => PAD_LEFT + (n <= 1 ? 0 : (i / (n - 1)) * plotWidth),
    [n, plotWidth]
  );
  const yForValue = useCallback(
    (v: number) => PAD_TOP + (1 - (v - yMin) / (yMax - yMin || 1)) * plotHeight,
    [yMin, yMax, plotHeight]
  );
  const valueForY = useCallback(
    (y: number) => yMin + (1 - (y - PAD_TOP) / plotHeight) * (yMax - yMin),
    [yMin, yMax, plotHeight]
  );
  const indexForX = useCallback(
    (x: number) => {
      const raw = ((x - PAD_LEFT) / plotWidth) * (n - 1);
      return Math.min(n - 1, Math.max(0, Math.round(raw)));
    },
    [n, plotWidth]
  );

  const applyDrag = useCallback(
    (dragIndex: number, mouseY: number) => {
      const base = baseValuesRef.current;
      const newValueAtDrag = valueForY(mouseY);
      const delta = newValueAtDrag - base[dragIndex];

      const next = base.slice();
      for (let j = Math.max(0, dragIndex - radius); j <= Math.min(n - 1, dragIndex + radius); j++) {
        const weight = falloffWeight(Math.abs(j - dragIndex), radius);
        next[j] = base[j] + delta * weight;
      }
      setValues(next);
    },
    [n, radius, valueForY]
  );

  const getSvgPoint = (clientX: number, clientY: number) => {
    const rect = svgRef.current?.getBoundingClientRect();
    if (!rect) return { x: 0, y: 0 };
    return { x: clientX - rect.left, y: clientY - rect.top };
  };

  const handlePointerDown = (clientX: number, clientY: number) => {
    const { x, y } = getSvgPoint(clientX, clientY);
    const idx = indexForX(x);
    dragIndexRef.current = idx;
    baseValuesRef.current = values.slice(); // freeze the starting shape for this drag
    applyDrag(idx, y);
  };

  const handlePointerMove = (clientX: number, clientY: number) => {
    if (dragIndexRef.current === null) return;
    const { y } = getSvgPoint(clientX, clientY);
    applyDrag(dragIndexRef.current, y);
  };

  const handlePointerUp = () => {
    if (dragIndexRef.current === null) return;
    dragIndexRef.current = null;
    lastSyncedKeyRef.current = values.join(","); // this IS the new "incoming" state now
    Streamlit.setComponentValue(values);
    setValues(values.slice()); // force a re-render now so the y-axis re-fits immediately, without waiting on the Python round-trip
  };

  // Refs always hold this render's freshest handler closures (correctly
  // capturing the current `values`, `radius`, etc.), so the window
  // listeners below can attach exactly ONCE for the component's lifetime
  // instead of being torn down and rebuilt on every mousemove during a
  // drag -- which is what happened when this effect depended on `values`.
  const handlePointerMoveRef = useRef(handlePointerMove);
  const handlePointerUpRef = useRef(handlePointerUp);
  handlePointerMoveRef.current = handlePointerMove;
  handlePointerUpRef.current = handlePointerUp;

  useEffect(() => {
    const onMouseMove = (e: MouseEvent) => handlePointerMoveRef.current(e.clientX, e.clientY);
    const onMouseUp = () => handlePointerUpRef.current();
    const onTouchMove = (e: TouchEvent) => {
      if (e.touches.length > 0) {
        handlePointerMoveRef.current(e.touches[0].clientX, e.touches[0].clientY);
        e.preventDefault();
      }
    };
    const onTouchEnd = () => handlePointerUpRef.current();

    window.addEventListener("mousemove", onMouseMove);
    window.addEventListener("mouseup", onMouseUp);
    window.addEventListener("touchmove", onTouchMove, { passive: false });
    window.addEventListener("touchend", onTouchEnd);
    return () => {
      window.removeEventListener("mousemove", onMouseMove);
      window.removeEventListener("mouseup", onMouseUp);
      window.removeEventListener("touchmove", onTouchMove);
      window.removeEventListener("touchend", onTouchEnd);
    };
  }, []); // mount once -- handlers are always invoked via the refs above, so they never go stale

  // Hover tracking is separate from drag tracking: this only updates
  // `hoverIndex` (real React state, so it re-renders) while idle, letting
  // someone scan values before touching anything. During an active drag,
  // `dragIndexRef` already identifies the relevant point every render (via
  // the setValues() call in applyDrag), so hover state is simply ignored
  // in favor of it -- see `activeIndex` below.
  const handleSvgMouseMove = (clientX: number) => {
    if (dragIndexRef.current !== null) return;
    const { x } = getSvgPoint(clientX, 0);
    setHoverIndex(indexForX(x));
  };
  const handleSvgMouseLeave = () => {
    if (dragIndexRef.current === null) setHoverIndex(null);
  };

  const activeIndex = dragIndexRef.current !== null ? dragIndexRef.current : hoverIndex;

  const linePath = useMemo(() => {
    return values.map((v, i) => `${i === 0 ? "M" : "L"}${xForIndex(i)},${yForValue(v)}`).join(" ");
  }, [values, xForIndex, yForValue]);

  const bandPath = useMemo(() => {
  if (!bandLower || !bandUpper || bandLower.length !== n || bandUpper.length !== n) return null;
  let d = "";
  for (let i = 0; i < n; i++) d += `${i === 0 ? "M" : "L"}${xForIndex(i)},${yForValue(bandUpper[i])} `;
  for (let i = n - 1; i >= 0; i--) d += `L${xForIndex(i)},${yForValue(bandLower[i])} `;
  return d + "Z";
  }, [bandLower, bandUpper, n, xForIndex, yForValue]);

  const forecastPath = useMemo(() => {
    if (!forecast || forecast.length !== n) return null;
    return forecast.map((v, i) => `${i === 0 ? "M" : "L"}${xForIndex(i)},${yForValue(v)}`).join(" ");
  }, [forecast, n, xForIndex, yForValue]);

  // Hour-aligned x-axis labels, not an arbitrary "every n/8th point" step:
  // labels only ever land on an exact hour (":00"), and the hour STEP
  // (every 1h/2h/3h/4h) adapts to available width so labels never overlap
  // -- shows all 24 hours when there's room, thins out gracefully on a
  // narrow/mobile chart instead of a fixed density regardless of size.
  const hourLabelStep = useMemo(() => {
    const pxIfEveryHour = plotWidth / 24;
    if (pxIfEveryHour >= 24) return 1;
    if (pxIfEveryHour >= 12) return 2;
    if (pxIfEveryHour >= 8) return 3;
    return 4;
  }, [plotWidth]);

  const isHourLabel = useCallback(
    (i: number) => {
      const parts = labels[i]?.split(":");
      if (!parts || parts.length < 2) return false;
      const hh = Number(parts[0]);
      const mm = Number(parts[1]);
      return mm === 0 && hh % hourLabelStep === 0;
    },
    [labels, hourLabelStep]
  );

  // Tooltip box placement: anchored near the active point, but flipped to
  // stay inside the chart when that point is close to an edge, rather than
  // running text off the side (near x=0) or over the axis (near the top).
  const tooltip = useMemo(() => {
    if (activeIndex === null) return null;
    const v = values[activeIndex];
    const x = xForIndex(activeIndex);
    const y = yForValue(v);
    const boxWidth = 108;
    const boxHeight = 20;
    const anchorLeft = x + boxWidth + 10 > width - PAD_RIGHT;
    const boxX = anchorLeft ? x - boxWidth - 10 : x + 10;
    const wantsAbove = y - boxHeight - 8 >= PAD_TOP;
    const boxY = wantsAbove ? y - boxHeight - 8 : y + 8;

    const original = forecast && forecast.length === n ? forecast[activeIndex] : undefined;
    const showOriginal = original !== undefined && Math.abs(original - v) > 0.05;
    const text = showOriginal
      ? `${labels[activeIndex]} — ${v.toFixed(1)} (was ${original.toFixed(1)})`
      : `${labels[activeIndex]} — ${v.toFixed(1)}`;

    return { boxX, boxY, boxWidth, boxHeight, text };
  }, [activeIndex, values, xForIndex, yForValue, width, forecast, n, labels]);

  return (
    <div>
      <svg
        ref={svgRef}
        width={width}
        height={height}
        style={{ touchAction: "none", cursor: "ns-resize", display: "block" }}
        onMouseDown={(e) => handlePointerDown(e.clientX, e.clientY)}
        onMouseMove={(e) => handleSvgMouseMove(e.clientX)}
        onMouseLeave={handleSvgMouseLeave}
        onTouchStart={(e) => {
          if (e.touches.length > 0) handlePointerDown(e.touches[0].clientX, e.touches[0].clientY);
        }}
      >
        {/* Invisible full-area hit target. SVG only registers pointer events
           on "painted" pixels by default (visiblePainted) -- without this,
           only the 2.5px line stroke and 3px circles themselves are
           clickable, which in practice is nearly impossible to hit. This
           rect makes the entire plot area draggable, not just those slivers. */}
        <rect
          x={PAD_LEFT}
          y={PAD_TOP}
          width={plotWidth}
          height={plotHeight}
          fill="transparent"
          style={{ pointerEvents: "all" }}
        />

        {/* y-axis gridlines + labels */}
        <text x={PAD_LEFT + 4} y={14} fontSize={10} textAnchor="start" fill="var(--text-color, #888)" opacity={0.7}>
          EUR / MWh
        </text>
        {[0, 0.25, 0.5, 0.75, 1].map((t) => {
          const v = yMin + t * (yMax - yMin);
          const y = yForValue(v);
          return (
            <g key={t}>
              <line x1={PAD_LEFT} y1={y} x2={width - PAD_RIGHT} y2={y} stroke="var(--gray-70, #444)" strokeWidth={0.5} opacity={0.35} />
              <text x={PAD_LEFT - 8} y={y} fontSize={11} textAnchor="end" dominantBaseline="middle" fill="var(--text-color, #888)">
                {v.toFixed(0)}
              </text>
            </g>
          );
        })}

        {/* x-axis labels: one per exact hour, thinned adaptively -- see hourLabelStep */}
        {values.map((_, i) =>
          isHourLabel(i) ? (
            <text key={i} x={xForIndex(i)} y={height - PAD_BOTTOM + 16} fontSize={10} textAnchor="middle" fill="var(--text-color, #888)">
              {labels[i]}
            </text>
          ) : null
        )}

        {bandPath && (
          <path d={bandPath} fill="rgba(100,100,255,0.35)" stroke="none" style={{ pointerEvents: "none" }} />
        )}
        {forecastPath && (
          <path d={forecastPath} fill="none" stroke="var(--text-color, #888)" strokeWidth={1.5}
                strokeDasharray="5 4" opacity={0.6} style={{ pointerEvents: "none" }} />
        )}

        {/* the curve itself */}
        <path d={linePath} fill="none" stroke="#6366f1" strokeWidth={2.5} />

        {/* draggable point handles -- enlarged on hover too, not just while dragging */}
        {values.map((v, i) => (
          <circle
            key={i}
            cx={xForIndex(i)}
            cy={yForValue(v)}
            r={activeIndex === i ? 6 : 4}
            fill={flagged && flagged[i] ? "#f59e0b" : "#6366f1"}
            stroke="white"
            strokeWidth={1}
            style={{ cursor: "ns-resize" }}
          />
        ))}

        {/* Value/time tooltip for whichever point is hovered or being
           dragged -- this is the actual fix for "how do experts know what
           they're adjusting": without it, only 5 sparse y-axis gridline
           values were visible, with no way to read an individual point. */}
        {tooltip && (
          <g style={{ pointerEvents: "none" }}>
            <rect
              x={tooltip.boxX}
              y={tooltip.boxY}
              width={tooltip.boxWidth}
              height={tooltip.boxHeight}
              rx={4}
              fill="var(--background-color, #1c222b)"
              stroke="#6366f1"
              strokeWidth={1}
              opacity={0.95}
            />
            <text
              x={tooltip.boxX + tooltip.boxWidth / 2}
              y={tooltip.boxY + tooltip.boxHeight / 2}
              fontSize={11}
              textAnchor="middle"
              dominantBaseline="middle"
              fill="var(--text-color, #eee)"
            >
              {tooltip.text}
            </text>
          </g>
        )}
      </svg>
      <div style={{ fontSize: 12, color: "var(--text-color, #888)", marginTop: 4 }}>
        Drag any point to adjust it — nearby points within {radius} slots shift too. Hover a point to see its exact time and value.
      </div>
      {/* Real legend with color swatches, matching what Plotly's native
        legend showed before this chart became the draggable_curve widget --
        only lists entries for whatever was actually supplied as args. */}
      <div style={{ display: "flex", flexWrap: "wrap", gap: 14, fontSize: 12, color: "var(--text-color, #888)", marginTop: 6 }}>
        <span style={{ display: "flex", alignItems: "center", gap: 5 }}>
          <span style={{ width: 14, height: 3, background: "#6366f1", display: "inline-block", borderRadius: 1 }} />
          Adjusted
        </span>
        {forecastPath && (
          <span style={{ display: "flex", alignItems: "center", gap: 5 }}>
            <span style={{ width: 14, height: 0, borderTop: "2px dashed var(--text-color, #888)", display: "inline-block", opacity: 0.6 }} />
            DNN Forecast
          </span>
        )}
        {bandPath && (
          <span style={{ display: "flex", alignItems: "center", gap: 5 }}>
            <span style={{ width: 14, height: 10, background: "rgba(100,100,255,0.35)", display: "inline-block", borderRadius: 2 }} />
             Likely price range (80%)
          </span>
        )}
        {flagged && (
          <span style={{ display: "flex", alignItems: "center", gap: 5 }}>
            <span style={{ width: 9, height: 9, background: "#f59e0b", borderRadius: "50%", display: "inline-block" }} />
            Flagged (5th/95th pct)
          </span>
        )}
      </div>
    </div>
  );
};

export default DraggableCurve;