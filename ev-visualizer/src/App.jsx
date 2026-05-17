import { useState, useEffect, useRef, useCallback, useMemo } from "react";

// ─── Google Fonts ─────────────────────────────────────────────────────────────
const fontLink = document.createElement("link");
fontLink.rel = "stylesheet";
fontLink.href = "https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Space+Mono:wght@400;700&display=swap";
document.head.appendChild(fontLink);

// ─── CSS Injection ────────────────────────────────────────────────────────────
const css = `
  :root {
    --bg:         #06090f;
    --surface:    #0c1220;
    --surface2:   #111928;
    --border:     rgba(255,255,255,0.06);
    --border2:    rgba(255,255,255,0.12);
    --amber:      #f59e0b;
    --amber-dim:  rgba(245,158,11,0.15);
    --teal:       #14b8a6;
    --teal-dim:   rgba(20,184,166,0.12);
    --rose:       #f43f5e;
    --rose-dim:   rgba(244,63,94,0.12);
    --lime:       #84cc16;
    --lime-dim:   rgba(132,204,22,0.12);
    --sky:        #38bdf8;
    --sky-dim:    rgba(56,189,248,0.1);
    --violet:     #8b5cf6;
    --violet-dim: rgba(139,92,246,0.12);
    --muted:      #4b5563;
    --text:       #f0f4f8;
    --text-dim:   #b8c5d4;
    --text-faint: #6b7f96;
    --font-display: 'Syne', sans-serif;
    --font-mono:    'Space Mono', monospace;
  }
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--font-mono);
    overflow: hidden;
    height: 100vh;
    width: 100vw;
    font-size: 13px;
  }
  ::-webkit-scrollbar { width: 4px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: var(--muted); border-radius: 4px; }

  @keyframes shimmer {
    0%   { background-position: -200% 0; }
    100% { background-position:  200% 0; }
  }
  @keyframes pulse-ring {
    0%   { transform: scale(1);   opacity: 0.8; }
    50%  { transform: scale(1.4); opacity: 0; }
    100% { transform: scale(1);   opacity: 0; }
  }
  @keyframes fadeIn {
    from { opacity: 0; transform: translateY(6px); }
    to   { opacity: 1; transform: translateY(0); }
  }
  @keyframes blink {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.2; }
  }
  @keyframes rotateGlow {
    0%   { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
  }
  .fade-in { animation: fadeIn 0.35s ease both; }

  .stat-card {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 11px 14px;
    transition: border-color 0.2s;
    flex: 1;
  }
  .stat-card:hover { border-color: var(--border2); }

  .panel {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 10px;
  }

  input[type=range] {
    -webkit-appearance: none;
    height: 3px;
    border-radius: 99px;
    background: var(--surface2);
    outline: none;
    cursor: pointer;
    flex: 1;
  }
  input[type=range]::-webkit-slider-thumb {
    -webkit-appearance: none;
    width: 14px; height: 14px;
    border-radius: 50%;
    background: var(--amber);
    cursor: pointer;
    box-shadow: 0 0 8px var(--amber);
  }
  input[type=number], select {
    background: var(--surface2);
    border: 1px solid var(--border2);
    border-radius: 8px;
    color: var(--text);
    font-family: var(--font-mono);
    font-size: 12px;
    padding: 8px 12px;
    outline: none;
    width: 100%;
    transition: border-color 0.2s;
  }
  input[type=number]:focus, select:focus { border-color: var(--amber); }
  select option { background: var(--surface2); }

  .btn-primary {
    display: flex; align-items: center; justify-content: center; gap: 8px;
    width: 100%; padding: 12px;
    border-radius: 10px; border: none;
    background: linear-gradient(135deg, var(--amber) 0%, #ef4444 100%);
    color: #000;
    font-family: var(--font-display);
    font-size: 13px; font-weight: 800;
    letter-spacing: 0.06em;
    cursor: pointer;
    transition: opacity 0.2s, transform 0.15s;
    text-transform: uppercase;
  }
  .btn-primary:hover:not(:disabled) { opacity: 0.88; transform: translateY(-1px); }
  .btn-primary:disabled { opacity: 0.38; cursor: not-allowed; transform: none; }

  .ctrl-btn {
    display: flex; align-items: center; justify-content: center;
    padding: 6px 11px;
    border-radius: 7px;
    border: 1px solid var(--border2);
    background: var(--surface2);
    color: var(--text-dim);
    font-family: var(--font-mono);
    font-size: 13px;
    cursor: pointer;
    transition: all 0.15s;
    white-space: nowrap;
  }
  .ctrl-btn:hover:not(:disabled) { border-color: var(--amber); color: var(--amber); }
  .ctrl-btn:disabled { opacity: 0.3; cursor: not-allowed; }
  .ctrl-btn.active { background: var(--amber-dim); border-color: var(--amber); color: var(--amber); }

  .algo-btn {
    flex: 1; padding: 9px 0;
    border-radius: 8px; border: 1px solid var(--border2);
    background: transparent;
    color: var(--text-faint);
    font-family: var(--font-mono);
    font-size: 12px;
    cursor: pointer;
    transition: all 0.2s;
    font-weight: 700;
  }
  .algo-btn.active-greedy {
    border-color: var(--rose); background: var(--rose-dim); color: var(--rose);
  }
  .algo-btn.active-ai {
    border-color: var(--teal); background: var(--teal-dim); color: var(--teal);
  }

  .speed-btn {
    padding: 5px 10px;
    border-radius: 6px;
    border: 1px solid var(--border);
    background: transparent;
    color: var(--text-faint);
    font-family: var(--font-mono);
    font-size: 11px;
    cursor: pointer;
    transition: all 0.15s;
  }
  .speed-btn.active {
    border-color: var(--amber); background: var(--amber-dim);
    color: var(--amber); font-weight: 700;
  }
  .tag {
    display: inline-flex; align-items: center;
    padding: 2px 8px; border-radius: 99px;
    font-size: 10px; font-weight: 700;
    letter-spacing: 0.06em; text-transform: uppercase;
  }

  /* ── Extra animations ─────────────────────────────────────────── */
  @keyframes fadeSlideUp {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
  }
  @keyframes popIn {
    0%   { transform: scale(0.9); opacity: 0; }
    65%  { transform: scale(1.04); }
    100% { transform: scale(1);   opacity: 1; }
  }
  section { animation: fadeSlideUp 0.28s ease both; }

  /* ── Aside scrollbar ──────────────────────────────────────────── */
  aside::-webkit-scrollbar { width: 3px; }

  /* ── Canvas cursor override when panning ─────────────────────── */
  canvas { image-rendering: auto; }

  /* ── Pill button group ────────────────────────────────────────── */
  .pill-group {
    display: flex;
    border-radius: 8px;
    overflow: hidden;
    border: 1px solid var(--border2);
  }
  .pill-group button {
    flex: 1; padding: 7px 0;
    border: none; background: transparent;
    color: var(--text-faint);
    font-family: var(--font-mono); font-size: 12px;
    cursor: pointer;
    border-right: 1px solid var(--border2);
    transition: background 0.15s, color 0.15s;
  }
  .pill-group button:last-child { border-right: none; }
  .pill-group button.active {
    background: var(--amber-dim); color: var(--amber); font-weight: 700;
  }
  .pill-group button:hover:not(.active) {
    background: var(--surface2); color: var(--text);
  }

  /* ── Right panel section dividers ─────────────────────────────── */
  aside > section {
    padding-bottom: 13px;
    border-bottom: 1px solid var(--border);
  }
  aside > section:last-of-type { border-bottom: none; padding-bottom: 0; }

  /* ── Stat card glow on hover ───────────────────────────────────── */
  .stat-card:hover {
    box-shadow: 0 0 0 1px rgba(245,158,11,0.18);
  }

  /* ── Footer kbd ────────────────────────────────────────────────── */
  footer kbd {
    padding: 1px 5px; border-radius: 4px;
    border: 1px solid var(--border2);
    background: var(--surface2);
    font-family: var(--font-mono); font-size: 9px;
    color: var(--amber);
  }

  /* ── Panel hover border ────────────────────────────────────────── */
  .panel:hover { border-color: rgba(255,255,255,0.1); }

  /* ── Demand bar hover tooltip ──────────────────────────────────── */
  .demand-bar { cursor: default; }
  .demand-bar:hover { opacity: 1 !important; }

  /* ── SoC bar color transitions ─────────────────────────────────── */
  .soc-bar { transition: width 0.4s ease, background 0.4s ease; }

  /* ── Loading spinner ring ──────────────────────────────────────── */
  .spin-ring { animation: rotateGlow 1s linear infinite; }
  .spin-ring-slow { animation: rotateGlow 1.6s linear infinite reverse; }
`;
const styleEl = document.createElement("style");
styleEl.textContent = css;
document.head.appendChild(styleEl);

// ─── Constants ────────────────────────────────────────────────────────────────
const API_URL = "http://127.0.0.1:8000";
const CITY_KM = 20;
const FRAME_INTERVAL_MIN = 15;

const PROFILE_NAMES = [
  "Normal","Commuter","Saturday","Sunday","Low",
  "High Stress","Flattened","Bimodal","Event","Early Spike",
];

const PROFILE_DESCRIPTIONS = [
  "Τυπική καθημερινή με μέτρια ζήτηση και πρωινή αιχμή.",
  "Έντονη πρωινή/απογευματινή αιχμή λόγω επαγγελματιών.",
  "Σαββατιάτικο προφίλ: απογευματινή & βραδινή αιχμή.",
  "Κυριακή: χαμηλή ζήτηση, ομοιόμορφη κατανομή 24ώρου.",
  "Ελάχιστη κίνηση — ιδανικό για αξιολόγηση αποδοτικότητας.",
  "Ακραία φόρτωση: πολλαπλές αιχμές, stress-test σεναρίου.",
  "Επίπεδη κατανομή χωρίς έντονες αιχμές όλη την ημέρα.",
  "Διπολικό: έντονη πρωί & βράδυ, ησυχία το μεσημέρι.",
  "Ειδικό γεγονός: απότομη τοπική αιχμή στη μέση της ημέρας.",
  "Πρώιμη έκρηξη ζήτησης τα πρώτα λεπτά της ημέρας.",
];

const STATE_META = {
  0: { label: "Idle",            color: "#64748b", glow: false },
  1: { label: "With Customer",   color: "#38bdf8", glow: true  },
  2: { label: "Rebalancing",     color: "#a78bfa", glow: false },
  3: { label: "Waiting Charger", color: "#fb923c", glow: true  },
  4: { label: "Charging",        color: "#84cc16", glow: true  },
  5: { label: "Stranded",        color: "#f43f5e", glow: true  },
};

const SPEED_OPTIONS = [
  { label: "0.25×", ms: 800  },
  { label: "0.5×",  ms: 400  },
  { label: "1×",    ms: 200  },
  { label: "2×",    ms: 120  },
  { label: "4×",    ms: 60   },
];

// ─── Helpers ──────────────────────────────────────────────────────────────────
const toCanvas = (km, S, offset, scale) => (km / CITY_KM) * S * scale - offset;
const fmtMin   = (m) => `${String(Math.floor(m/60)%24).padStart(2,"0")}:${String(m%60).padStart(2,"0")}`;
const fmtEur   = (n) => new Intl.NumberFormat("el-GR",{maximumFractionDigits:0}).format(n);
const randSeed = () => Math.floor(Math.random() * 99999) + 1;

// ─── Sparkline ────────────────────────────────────────────────────────────────
function Sparkline({ data, color, height = 52, width = 220, cursorIdx }) {
  if (!data || data.length < 2) return <div style={{ height, width }} />;
  const max = Math.max(...data, 1);
  const min = Math.min(...data);
  const range = max - min || 1;
  const pts = data.map((v, i) => [
    (i / (data.length - 1)) * width,
    height - ((v - min) / range) * (height - 6) - 3,
  ]);
  const polyline = pts.map(p => p.join(",")).join(" ");
  const area = [`0,${height}`, ...pts.map(p => p.join(",")), `${width},${height}`].join(" ");
  const gid = `sg${color.replace(/[^a-z0-9]/gi,"")}`;
  const cx = cursorIdx != null ? pts[Math.min(cursorIdx, pts.length - 1)] : null;
  return (
    <svg width={width} height={height} style={{ display:"block", overflow:"visible" }}>
      <defs>
        <linearGradient id={gid} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%"   stopColor={color} stopOpacity="0.35" />
          <stop offset="100%" stopColor={color} stopOpacity="0"    />
        </linearGradient>
      </defs>
      <polygon points={area} fill={`url(#${gid})`} />
      <polyline points={polyline} fill="none" stroke={color} strokeWidth="1.5" strokeLinejoin="round" />
      {cx && <>
        <line x1={cx[0]} y1={0} x2={cx[0]} y2={height} stroke={color} strokeWidth="1" strokeDasharray="3,3" opacity="0.4" />
        <circle cx={cx[0]} cy={cx[1]} r="3.5" fill={color} stroke="var(--bg)" strokeWidth="1.5" />
      </>}
    </svg>
  );
}

// ─── Section Header ───────────────────────────────────────────────────────────
function SectionHeader({ children, accent = "var(--amber)" }) {
  return (
    <div style={{ fontSize:11, fontWeight:700, textTransform:"uppercase", letterSpacing:"0.14em",
      color:"var(--text-faint)", display:"flex", alignItems:"center", gap:8, marginBottom:10 }}>
      <div style={{ width:14, height:2, background:accent, borderRadius:99, flexShrink:0 }} />
      {children}
    </div>
  );
}

// ─── StatCard ─────────────────────────────────────────────────────────────────
function StatCard({ label, value, unit, color, small }) {
  return (
    <div className="stat-card">
      <div style={{ fontSize:11, color:"var(--text-faint)", textTransform:"uppercase", letterSpacing:"0.1em", marginBottom:5 }}>{label}</div>
      <div style={{ fontSize: small ? 18 : 24, fontWeight:700, color: color||"var(--text)", fontFamily:"var(--font-display)", lineHeight:1.1 }}>
        {value}
        {unit && <span style={{ fontSize:13, fontWeight:400, color:"var(--text-dim)", marginLeft:3 }}>{unit}</span>}
      </div>
    </div>
  );
}

// ─── Loading Overlay ──────────────────────────────────────────────────────────
function LoadingOverlay({ progress }) {
  return (
    <div style={{ position:"absolute", inset:0, background:"rgba(6,9,15,0.92)",
      display:"flex", alignItems:"center", justifyContent:"center",
      flexDirection:"column", gap:20, zIndex:50, borderRadius:12, backdropFilter:"blur(4px)" }}>
      <div style={{ position:"relative", width:72, height:72 }}>
        <div style={{ position:"absolute", inset:0, border:"2px solid var(--border2)", borderRadius:"50%" }} />
        <div style={{ position:"absolute", inset:0, border:"2px solid transparent",
          borderTopColor:"var(--amber)", borderRadius:"50%",
          animation:"rotateGlow 0.9s linear infinite" }} />
        <div style={{ position:"absolute", inset:8, border:"1px solid transparent",
          borderTopColor:"var(--rose)", borderRadius:"50%",
          animation:"rotateGlow 1.5s linear infinite reverse" }} />
        <div style={{ position:"absolute", inset:0, display:"flex", alignItems:"center",
          justifyContent:"center", fontSize:12, fontWeight:700, color:"var(--amber)" }}>
          {progress}%
        </div>
      </div>
      <div style={{ width:260 }}>
        <div style={{ height:3, background:"var(--surface2)", borderRadius:99, overflow:"hidden" }}>
          <div style={{ height:"100%", width:`${progress}%`,
            background:"linear-gradient(90deg, var(--amber), var(--rose))", borderRadius:99,
            transition:"width 0.3s ease", boxShadow:"0 0 10px var(--amber)" }} />
        </div>
        <div style={{ display:"flex", justifyContent:"space-between", marginTop:6, fontSize:10, color:"var(--text-faint)" }}>
          <span>Εκτέλεση simulation…</span>
          <span style={{ color:"var(--amber)" }}>{progress}%</span>
        </div>
      </div>
      <div style={{ fontSize:11, color:"var(--text-faint)", textAlign:"center", lineHeight:1.8 }}>
        Υπολογισμός 1.440 λεπτών ημέρας<br />
        <span style={{ color:"var(--amber)", animation:"blink 1.2s ease infinite" }}>●</span>
        {" "}Επεξεργασία δεδομένων στόλου…
      </div>
    </div>
  );
}

// ─── Clock ────────────────────────────────────────────────────────────────────
function SimClock({ minute }) {
  const h = Math.floor(minute / 60) % 24;
  const m = minute % 60;
  const isNight = h < 6 || h >= 22;
  const isPeak  = (h>=7&&h<=9)||(h>=16&&h<=19);
  const period  = isNight ? { label:"Νύχτα", color:"#a78bfa" }
                : isPeak  ? { label:"Αιχμή",  color:"#f59e0b" }
                :            { label:"Ημέρα",  color:"#38bdf8" };
  return (
    <div style={{ display:"flex", alignItems:"center", gap:10, flexShrink:0 }}>
      <div style={{ fontFamily:"var(--font-display)", fontSize:26, fontWeight:800,
        color:period.color, letterSpacing:"-0.02em",
        textShadow:`0 0 20px ${period.color}55`, minWidth:78 }}>
        {String(h).padStart(2,"0")}:{String(m).padStart(2,"0")}
      </div>
      <div>
        <div style={{ fontSize:9, color:"var(--text-faint)", textTransform:"uppercase", letterSpacing:"0.1em" }}>Ώρα</div>
        <div className="tag" style={{ background:`${period.color}20`, color:period.color, marginTop:2 }}>
          {period.label}
        </div>
      </div>
    </div>
  );
}

// ─── State Distribution Bar ───────────────────────────────────────────────────
function StateDistBar({ counts, total }) {
  if (!total) return null;
  return (
    <div>
      <div style={{ display:"flex", height:7, borderRadius:99, overflow:"hidden", gap:1 }}>
        {Object.entries(STATE_META).map(([k,v]) => {
          const n = counts[k]||0;
          const pct = (n/total)*100;
          if (pct < 0.4) return null;
          return <div key={k} style={{ width:`${pct}%`, background:v.color, transition:"width 0.4s" }} title={`${v.label}: ${n}`} />;
        })}
      </div>
      <div style={{ display:"flex", flexWrap:"wrap", gap:"5px 10px", marginTop:8 }}>
        {Object.entries(STATE_META).map(([k,v]) => {
          const n = counts[k]||0;
          if (!n) return null;
          return (
            <div key={k} style={{ display:"flex", alignItems:"center", gap:4, fontSize:10 }}>
              <div style={{ width:6, height:6, borderRadius:"50%", background:v.color, flexShrink:0 }} />
              <span style={{ color:"var(--text-faint)" }}>{v.label}</span>
              <span style={{ color:v.color, fontWeight:700 }}>{n}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ─── Station List ─────────────────────────────────────────────────────────────
function StationList({ stations, queues }) {
  if (!stations.length) return null;
  const sorted = [...stations].sort((a,b) => (queues[b.id]||0)-(queues[a.id]||0));
  return (
    <div style={{ display:"flex", flexDirection:"column", gap:3 }}>
      {sorted.slice(0,10).map(st => {
        const q = queues[st.id]||0;
        const isHub = st.type==="SUPER-HUB";
        const color = isHub ? "var(--amber)" : "var(--violet)";
        const pct   = Math.min((q/20)*100, 100);
        return (
          <div key={st.id} style={{ display:"flex", alignItems:"center", gap:7, padding:"4px 7px",
            borderRadius:6, background:q>5?"rgba(244,63,94,0.05)":"transparent",
            border:`1px solid ${q>5?"rgba(244,63,94,0.14)":"transparent"}` }}>
            <div style={{ width:18, textAlign:"right", fontSize:9, color:"var(--text-faint)" }}>#{st.id}</div>
            <div style={{ width:6, height:6, borderRadius:"50%", background:color, flexShrink:0 }} />
            <div style={{ flex:1 }}>
              <div style={{ height:3, background:"var(--border)", borderRadius:99 }}>
                <div style={{ height:"100%", width:`${pct}%`,
                  background:q>5?"var(--rose)":color, borderRadius:99, transition:"width 0.3s" }} />
              </div>
            </div>
            <div style={{ minWidth:22, textAlign:"right", fontSize:11, fontWeight:700,
              color:q>5?"var(--rose)":q>0?color:"var(--text-faint)" }}>{q}</div>
            <div className="tag" style={{ background:isHub?"var(--amber-dim)":"var(--violet-dim)",
              color:isHub?"var(--amber)":"var(--violet)", fontSize:8 }}>{isHub?"HUB":"STD"}</div>
          </div>
        );
      })}
    </div>
  );
}

// ─── Canvas Draw ──────────────────────────────────────────────────────────────
function drawCity(ctx, S, scale, offX, offY, frame, stations, heatmap) {
  ctx.clearRect(0, 0, S, S);

  // BG
  ctx.fillStyle = "#06090f";
  ctx.fillRect(0, 0, S, S);

  // Center glow
  const cx0 = toCanvas(10, S, offX, scale);
  const cy0 = toCanvas(10, S, offY, scale);
  const grd = ctx.createRadialGradient(cx0, cy0, 0, cx0, cy0, S * 0.5 * scale);
  grd.addColorStop(0, "rgba(20,184,166,0.05)");
  grd.addColorStop(1, "transparent");
  ctx.fillStyle = grd;
  ctx.fillRect(0, 0, S, S);

  // Grid
  ctx.strokeStyle = "rgba(255,255,255,0.04)";
  ctx.lineWidth = 0.5;
  for (let g = 0; g <= CITY_KM; g += 2) {
    const px = toCanvas(g, S, offX, scale);
    const py = toCanvas(g, S, offY, scale);
    ctx.beginPath(); ctx.moveTo(px,0); ctx.lineTo(px,S); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(0,py); ctx.lineTo(S,py); ctx.stroke();
  }

  if (!frame) return;

  // Heatmap
  if (heatmap) {
    const cells = 20;
    const km = CITY_KM / cells;
    const dens = Array.from({length:cells}, () => new Array(cells).fill(0));
    frame.taxis.forEach(ev => {
      if (ev.s === 5) return;
      const xi = Math.min(Math.floor(ev.x / km), cells-1);
      const yi = Math.min(Math.floor(ev.y / km), cells-1);
      if (xi>=0 && yi>=0) dens[yi][xi]++;
    });
    const maxD = Math.max(...dens.flat(), 1);
    const cpx = km * (S / CITY_KM) * scale;
    for (let row=0; row<cells; row++) {
      for (let col=0; col<cells; col++) {
        const d = dens[row][col];
        if (!d) continue;
        const alpha = (d/maxD)*0.20;
        ctx.fillStyle = `rgba(56,189,248,${alpha})`;
        ctx.fillRect(toCanvas(col*km,S,offX,scale), toCanvas(row*km,S,offY,scale), cpx+1, cpx+1);
      }
    }
  }

  // Stations
  stations.forEach(st => {
    const x = toCanvas(st.x, S, offX, scale);
    const y = toCanvas(st.y, S, offY, scale);
    if (x<-20||x>S+20||y<-20||y>S+20) return;
    const q   = (frame.queues&&frame.queues[st.id])||0;
    const hub = st.type==="SUPER-HUB";
    const r   = hub ? 13 : 8;
    const col = hub ? "#f59e0b" : "#8b5cf6";

    // halo
    const hg = ctx.createRadialGradient(x,y,r,x,y,r*3);
    hg.addColorStop(0, hub?"rgba(245,158,11,0.14)":"rgba(139,92,246,0.1)");
    hg.addColorStop(1,"transparent");
    ctx.fillStyle = hg;
    ctx.beginPath(); ctx.arc(x,y,r*3,0,Math.PI*2); ctx.fill();

    ctx.beginPath(); ctx.arc(x,y,r,0,Math.PI*2);
    ctx.fillStyle = hub?"rgba(245,158,11,0.18)":"rgba(139,92,246,0.15)"; ctx.fill();
    ctx.strokeStyle = col; ctx.lineWidth = hub?2:1.5; ctx.stroke();

    if (scale > 0.8) {
      ctx.fillStyle = col;
      ctx.font = `${hub?10:7}px sans-serif`;
      ctx.textAlign = "center"; ctx.textBaseline = "middle";
      ctx.fillText("⚡", x, y);
    }

    if (q > 0) {
      const br=7, bx=x+r-2, by=y-r+2;
      ctx.beginPath(); ctx.arc(bx,by,br,0,Math.PI*2);
      ctx.fillStyle = q>5?"#f43f5e":"#fb923c"; ctx.fill();
      ctx.fillStyle = "#fff";
      ctx.font = "bold 8px monospace";
      ctx.textAlign="center"; ctx.textBaseline="middle";
      ctx.fillText(q>99?"99":String(q), bx, by);
    }
  });

  // Taxis
  frame.taxis.forEach(ev => {
    const x = toCanvas(ev.x, S, offX, scale);
    const y = toCanvas(ev.y, S, offY, scale);
    if (x<-4||x>S+4||y<-4||y>S+4) return;
    const m = STATE_META[ev.s]||STATE_META[0];
    const r = scale > 1.5 ? 3.2 : 2.0;
    if (m.glow && scale > 0.9) {
      ctx.beginPath(); ctx.arc(x,y,r*2.8,0,Math.PI*2);
      ctx.fillStyle = `${m.color}18`; ctx.fill();
    }
    ctx.beginPath(); ctx.arc(x,y,r,0,Math.PI*2);
    ctx.fillStyle = m.color; ctx.fill();
  });
}

// ─── Day-Phase Timeline ───────────────────────────────────────────────────────
function DayPhaseBar({ currentMinute }) {
  const phases = [
    { label:"Νύχτα",    start:0,    end:360,  color:"#a78bfa" },
    { label:"Πρωί",     start:360,  end:540,  color:"#fb923c" },
    { label:"Αιχμή↑",  start:540,  end:600,  color:"#f59e0b" },
    { label:"Πρωινό",  start:600,  end:900,  color:"#38bdf8" },
    { label:"Μεσ/ρι",  start:900,  end:960,  color:"#14b8a6" },
    { label:"Απόγ/μα", start:960,  end:1080, color:"#38bdf8" },
    { label:"Αιχμή↓",  start:1080, end:1200, color:"#f59e0b" },
    { label:"Βράδυ",   start:1200, end:1320, color:"#fb923c" },
    { label:"Νύχτα",   start:1320, end:1440, color:"#a78bfa" },
  ];
  const total = 1440;
  return (
    <div>
      <div style={{ display:"flex", height:18, borderRadius:99, overflow:"hidden", gap:1 }}>
        {phases.map((ph,i) => {
          const w = ((ph.end - ph.start) / total) * 100;
          const active = currentMinute >= ph.start && currentMinute < ph.end;
          return (
            <div key={i} style={{ width:`${w}%`, background:ph.color,
              opacity: active ? 1 : 0.28,
              display:"flex", alignItems:"center", justifyContent:"center",
              fontSize:9, fontWeight:700, color:"#000",
              transition:"opacity 0.4s",
              overflow:"hidden", whiteSpace:"nowrap",
            }} title={`${ph.label}: ${ph.start/60|0}:00-${ph.end/60|0}:00`}>
              {w > 6 ? ph.label : ""}
            </div>
          );
        })}
      </div>
      <div style={{ display:"flex", justifyContent:"space-between", marginTop:4, fontSize:8, color:"var(--text-faint)" }}>
        {[0,4,8,12,16,20,24].map(h=><span key={h}>{String(h).padStart(2,"0")}:00</span>)}
      </div>
    </div>
  );
}

// ─── Demand Heatmap Hourly ────────────────────────────────────────────────────
function DemandHeatRow({ queueTS, currentMin }) {
  if (!queueTS || queueTS.length < 60) return null;
  const hourlyPeak = Array.from({length:24}, (_,h) => {
    const slice = queueTS.slice(h*60, (h+1)*60);
    return slice.length ? Math.max(...slice) : 0;
  });
  const maxV = Math.max(...hourlyPeak, 1);
  const curH = Math.floor(currentMin / 60) % 24;
  return (
    <div>
      <div style={{ display:"flex", gap:2, height:28 }}>
        {hourlyPeak.map((v,h) => {
          const pct = v / maxV;
          const active = h === curH;
          const col = pct > 0.75 ? "#f43f5e" : pct > 0.45 ? "#f59e0b" : pct > 0.2 ? "#38bdf8" : "#334155";
          return (
            <div key={h} style={{ flex:1, display:"flex", flexDirection:"column",
              alignItems:"center", justifyContent:"flex-end", gap:2 }}
              title={`${String(h).padStart(2,"0")}:00 — max queue: ${v}`}>
              <div style={{ width:"100%", background:col, opacity: active?1:0.55,
                height:`${Math.max(pct*22,2)}px`, borderRadius:"2px 2px 0 0",
                boxShadow: active?`0 0 6px ${col}`:"none",
                transition:"height 0.3s, opacity 0.3s" }} />
            </div>
          );
        })}
      </div>
      <div style={{ display:"flex", justifyContent:"space-between", marginTop:2, fontSize:10, color:"var(--text-faint)" }}>
        <span>00</span><span>06</span><span>12</span><span>18</span><span>24</span>
      </div>
    </div>
  );
}

// ─── Fleet Health Gauges ──────────────────────────────────────────────────────
function FleetHealthGauges({ stateCounts, totalFleet, avgSocNow }) {
  if (!totalFleet) return null;
  const charging = (stateCounts[4]||0);
  const withCust = (stateCounts[1]||0);
  const stranded = (stateCounts[5]||0);
  const waiting  = (stateCounts[3]||0);
  const idle     = (stateCounts[0]||0);
  const utilisation = ((withCust + charging) / totalFleet) * 100;
  const efficiency  = stranded === 0 ? 100 : Math.max(0, 100 - (stranded/totalFleet)*100);

  const GaugePill = ({ label, value, max, color, unit="%", warning }) => {
    const pct = Math.min((value/max)*100,100);
    return (
      <div style={{ display:"flex", flexDirection:"column", gap:4 }}>
        <div style={{ display:"flex", justifyContent:"space-between", fontSize:12 }}>
          <span style={{ color:"var(--text-faint)" }}>{label}</span>
          <span style={{ color: warning&&value>0?"var(--rose)":color, fontWeight:700 }}>
            {typeof value==="number"?value.toFixed(1):value}{unit}
          </span>
        </div>
        <div style={{ height:5, background:"var(--border)", borderRadius:99, overflow:"hidden" }}>
          <div style={{ height:"100%", width:`${pct}%`,
            background: warning&&value>0?"var(--rose)":color,
            borderRadius:99, transition:"width 0.4s",
            boxShadow:`0 0 6px ${warning&&value>0?"var(--rose)":color}55` }} />
        </div>
      </div>
    );
  };

  return (
    <div style={{ display:"flex", flexDirection:"column", gap:8 }}>
      <GaugePill label="Utilisation"  value={utilisation}                    max={100} color="var(--teal)" />
      <GaugePill label="Fleet Efficiency" value={efficiency}                 max={100} color="var(--lime)" />
      <GaugePill label="Avg SoC"      value={avgSocNow!=null?avgSocNow*100:0} max={100} color="var(--sky)" />
      <GaugePill label="Stranded"     value={stranded}                       max={Math.max(totalFleet*0.05,1)}
        color="var(--rose)" unit=" ταξί" warning />
      <GaugePill label="Waiting Charger" value={waiting}                     max={Math.max(totalFleet*0.1,1)}
        color="var(--amber)" unit=" ταξί" />
    </div>
  );
}

// ─── Algorithm Comparison Card ────────────────────────────────────────────────
function AlgoInfoCard({ algorithm }) {
  const info = algorithm === "ai"
    ? {
        name: "PPO AI Agent",
        color: "var(--teal)",
        bg: "var(--teal-dim)",
        icon: "🤖",
        details: [
          { k: "Αρχιτεκτονική", v: "MLP 256×256" },
          { k: "Εκπαίδευση",    v: "3.000.000 steps" },
          { k: "Observations",  v: "48 features" },
          { k: "Actions",       v: "18 (16 σταθμοί + idle + rebal)" },
          { k: "Γ (discount)",  v: "γ = 0.995" },
          { k: "Framework",     v: "Stable-Baselines3" },
        ],
      }
    : {
        name: "Greedy Heuristic",
        color: "var(--rose)",
        bg: "var(--rose-dim)",
        icon: "🔴",
        details: [
          { k: "Τύπος",          v: "Rule-based" },
          { k: "Εκπαίδευση",     v: "Καμία" },
          { k: "Φόρτιση",        v: "SoC ≤ 25% → κοντινός σταθμός" },
          { k: "Rebalancing",    v: "Εάν απόσταση > 5km από κέντρο" },
          { k: "Score σταθμού",  v: "dist + queue × 2.0" },
          { k: "Πολυπλοκότητα",  v: "O(n·stations)" },
        ],
      };
  return (
    <div style={{ padding:"10px 12px", borderRadius:10,
      background: info.bg, border:`1px solid ${info.color}40`,
      fontSize:12 }}>
      <div style={{ display:"flex", alignItems:"center", gap:7, marginBottom:8 }}>
        <span>{info.icon}</span>
        <span style={{ fontFamily:"var(--font-display)", fontWeight:700, color:info.color, fontSize:14 }}>
          {info.name}
        </span>
      </div>
      <div style={{ display:"flex", flexDirection:"column", gap:4 }}>
        {info.details.map(d=>(
          <div key={d.k} style={{ display:"flex", justifyContent:"space-between", gap:8 }}>
            <span style={{ color:"var(--text-faint)" }}>{d.k}</span>
            <span style={{ color:"var(--text)", fontWeight:700, textAlign:"right" }}>{d.v}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

// ─── Main App ─────────────────────────────────────────────────────────────────
export default function App() {
  const canvasRef   = useRef(null);
  const animRef     = useRef(null);
  const progressRef = useRef(null);
  const abortRef    = useRef(null);

  const [algorithm,  setAlgorithm]  = useState("baseline");
  const [profile,    setProfile]    = useState(0);
  const [seed,       setSeed]       = useState("");

  const [loading,    setLoading]    = useState(false);
  const [loadPct,    setLoadPct]    = useState(0);
  const [error,      setError]      = useState(null);

  const [frames,     setFrames]     = useState([]);
  const [stations,   setStations]   = useState([]);
  const [stats,      setStats]      = useState(null);
  const [queueTS,    setQueueTS]    = useState([]);
  const [socTS,      setSocTS]      = useState([]);

  const [frameIdx,   setFrameIdx]   = useState(0);
  const [playing,    setPlaying]    = useState(false);
  const [speedMs,    setSpeedMs]    = useState(200);

  const [canvasSize, setCanvasSize] = useState(480);
  const [zoom,       setZoom]       = useState(1.0);
  const [panX,       setPanX]       = useState(0);
  const [panY,       setPanY]       = useState(0);
  const [isPanning,  setIsPanning]  = useState(false);
  const [showHeatmap,setShowHeatmap]= useState(true);
  const panStart = useRef(null);
  const panBase  = useRef({x:0,y:0});

  // ── Responsive ───────────────────────────────────────────────────────────
  useEffect(() => {
    const calc = () => {
      const available = Math.min(
        window.innerWidth  - 340 - 360 - 48,
        window.innerHeight - 50  - 160 - 48,
      );
      setCanvasSize(Math.max(300, Math.min(available, 700)));
    };
    calc();
    window.addEventListener("resize", calc);
    return () => window.removeEventListener("resize", calc);
  }, []);

  // ── Fake progress ─────────────────────────────────────────────────────────
  const startProgress = useCallback(() => {
    setLoadPct(0);
    let p = 0;
    const tick = () => {
      p += Math.random() * 4.5 + 0.8;
      if (p >= 92) { setLoadPct(92); return; }
      setLoadPct(Math.floor(p));
      progressRef.current = setTimeout(tick, 160);
    };
    progressRef.current = setTimeout(tick, 160);
  }, []);
  const stopProgress = useCallback(() => {
    clearTimeout(progressRef.current);
    setLoadPct(100);
  }, []);

  // ── Playback ──────────────────────────────────────────────────────────────
  useEffect(() => {
    if (!playing || !frames.length) { clearInterval(animRef.current); return; }
    animRef.current = setInterval(() => {
      setFrameIdx(i => {
        if (i >= frames.length - 1) { setPlaying(false); return i; }
        return i + 1;
      });
    }, speedMs);
    return () => clearInterval(animRef.current);
  }, [playing, frames, speedMs]);

  // ── Draw ──────────────────────────────────────────────────────────────────
  useEffect(() => {
    const c = canvasRef.current;
    if (!c) return;
    drawCity(c.getContext("2d"), canvasSize, zoom, panX, panY,
      frames[frameIdx], stations, showHeatmap);
  }, [frameIdx, frames, stations, canvasSize, zoom, panX, panY, showHeatmap]);

  // ── Wheel zoom ────────────────────────────────────────────────────────────
  const onWheel = useCallback((e) => {
    e.preventDefault();
    setZoom(z => Math.max(0.5, Math.min(6, z + (e.deltaY < 0 ? 0.13 : -0.13))));
  }, []);
  useEffect(() => {
    const c = canvasRef.current;
    if (!c) return;
    c.addEventListener("wheel", onWheel, { passive: false });
    return () => c.removeEventListener("wheel", onWheel);
  }, [onWheel]);

  // ── Pan ───────────────────────────────────────────────────────────────────
  const onMD = useCallback((e) => {
    setIsPanning(true);
    panStart.current = { x:e.clientX, y:e.clientY };
    panBase.current  = { x:panX, y:panY };
  }, [panX, panY]);
  const onMM = useCallback((e) => {
    if (!isPanning||!panStart.current) return;
    setPanX(panBase.current.x - (e.clientX - panStart.current.x));
    setPanY(panBase.current.y - (e.clientY - panStart.current.y));
  }, [isPanning]);
  const onMU = useCallback(() => { setIsPanning(false); panStart.current=null; }, []);
  const resetView = useCallback(() => { setZoom(1); setPanX(0); setPanY(0); }, []);

  // ── Keyboard shortcuts ────────────────────────────────────────────────────
  useEffect(() => {
    const onKey = (e) => {
      if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
      if (e.code === "Space") {
        e.preventDefault();
        setPlaying(p => frames.length > 0 ? !p : false);
      } else if (e.code === "ArrowRight") {
        setPlaying(false);
        setFrameIdx(i => Math.min(i + 1, frames.length - 1));
      } else if (e.code === "ArrowLeft") {
        setPlaying(false);
        setFrameIdx(i => Math.max(i - 1, 0));
      } else if (e.code === "Home") {
        setPlaying(false); setFrameIdx(0);
      } else if (e.code === "End") {
        setPlaying(false); setFrameIdx(frames.length - 1);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [frames]);

  // ── Stop simulation ───────────────────────────────────────────────────────
  const stopSimulation = useCallback(() => {
    if (abortRef.current) abortRef.current.abort();
    clearTimeout(progressRef.current);
    setLoading(false);
    setLoadPct(0);
    setError("Το simulation διακόπηκε από τον χρήστη.");
  }, []);

  // ── Run ───────────────────────────────────────────────────────────────────
  const run = useCallback(async () => {
    // Create a new AbortController for this request
    const controller = new AbortController();
    abortRef.current = controller;

    setLoading(true); setError(null); setPlaying(false);
    setFrameIdx(0); setFrames([]); setStats(null);
    setQueueTS([]); setSocTS([]);
    startProgress();
    const usedSeed = seed===""?randSeed():Number(seed);
    try {
      const res = await fetch(`${API_URL}/simulate`, {
        method:"POST", headers:{"Content-Type":"application/json"},
        body: JSON.stringify({algorithm, profile, seed:usedSeed}),
        signal: controller.signal,
      });
      if (!res.ok) { const e=await res.json(); throw new Error(e.detail||`Error ${res.status}`); }
      const data = await res.json();
      stopProgress();
      await new Promise(r => setTimeout(r, 420));
      setFrames(data.frames||[]);
      setStations(data.stations||[]);
      setStats(data.stats||null);
      setQueueTS(data.queues_over_time||[]);
      setSocTS(data.avg_soc_over_time||[]);
      setFrameIdx(0);
    } catch(e) {
      stopProgress();
      if (e.name === "AbortError") {
        setError("Το simulation διακόπηκε από τον χρήστη.");
      } else {
        setError(e.message);
      }
    } finally {
      setLoading(false);
      abortRef.current = null;
    }
  }, [algorithm, profile, seed, startProgress, stopProgress]);

  // ── Derived ───────────────────────────────────────────────────────────────
  const curMin  = frameIdx * FRAME_INTERVAL_MIN;
  const tsIdx   = Math.min(curMin, queueTS.length - 1);
  const curFrame = frames[frameIdx];

  const stateCounts = useMemo(() => {
    if (!curFrame) return {};
    return curFrame.taxis.reduce((a,ev) => { a[ev.s]=(a[ev.s]||0)+1; return a; }, {});
  }, [curFrame]);

  const currentQueues = useMemo(() => {
    if (!curFrame||!curFrame.queues) return {};
    return curFrame.queues.reduce((a,q,i) => { a[i]=q; return a; }, {});
  }, [curFrame]);

  const totalFleet  = frames[0]?.taxis?.length || 0;
  const avgSocNow   = useMemo(() => {
    if (!curFrame) return null;
    const v = curFrame.taxis.map(t=>t.soc||0);
    return v.length ? v.reduce((a,b)=>a+b,0)/v.length : null;
  }, [curFrame]);
  const totalQNow = curFrame?.queues?.reduce((a,b)=>a+b,0)??null;

  // ─────────────────────────────────────────────────────────────────────────
  return (
    <div style={{ width:"100vw", height:"100vh", display:"flex", flexDirection:"column",
      background:"var(--bg)", overflow:"hidden" }}>

      {/* ══ HEADER ══════════════════════════════════════════════════════════ */}
      <header style={{ height:50, display:"flex", alignItems:"center", padding:"0 20px",
        borderBottom:"1px solid var(--border)", background:"var(--surface)",
        flexShrink:0, gap:16, zIndex:10 }}>
        <div style={{ display:"flex", alignItems:"center", gap:10 }}>
          <div style={{ width:30,height:30,borderRadius:8,
            background:"linear-gradient(135deg,var(--amber),var(--rose))",
            display:"flex",alignItems:"center",justifyContent:"center",fontSize:16,flexShrink:0 }}>⚡</div>
          <div>
            <div style={{ fontFamily:"var(--font-display)",fontSize:14,fontWeight:800,letterSpacing:"-0.01em" }}>
              EV Fleet Simulator
            </div>
          </div>
        </div>
        <div style={{ flex:1 }} />
        <div className="tag" style={{ background:algorithm==="ai"?"var(--teal-dim)":"var(--rose-dim)",
          color:algorithm==="ai"?"var(--teal)":"var(--rose)", fontSize:11 }}>
          {algorithm==="ai"?"🤖 PPO AI Agent":"🔴 Greedy Baseline"}
        </div>
        <div className="tag" style={{ background:"var(--amber-dim)",color:"var(--amber)",fontSize:11 }}>
          {PROFILE_NAMES[profile]}
        </div>
        {frames.length>0 && (
          <div style={{ fontSize:11,color:"var(--text-dim)",fontFamily:"var(--font-mono)" }}>
            Frame <span style={{ color:"var(--amber)" }}>{frameIdx+1}</span>/{frames.length}
            {" · "}
            <span style={{ color:"var(--text-faint)" }}>{((frameIdx/Math.max(frames.length-1,1))*100).toFixed(0)}%</span>
          </div>
        )}
      </header>

      {/* ══ BODY ════════════════════════════════════════════════════════════ */}
      <div style={{ flex:1, display:"flex", overflow:"hidden" }}>

        {/* ── LEFT PANEL ─────────────────────────────────────────────────── */}
        <aside style={{ width:340,flexShrink:0,background:"var(--surface)",
          borderRight:"1px solid var(--border)",padding:"18px 18px",
          display:"flex",flexDirection:"column",gap:16,overflowY:"auto" }}>

          <section>
            <SectionHeader accent="var(--rose)">Αλγόριθμος</SectionHeader>
            <div style={{ display:"flex",gap:8 }}>
              <button className={`algo-btn${algorithm==="baseline"?" active-greedy":""}`}
                onClick={()=>setAlgorithm("baseline")}>🔴 Greedy</button>
              <button className={`algo-btn${algorithm==="ai"?" active-ai":""}`}
                onClick={()=>setAlgorithm("ai")}>🤖 PPO AI</button>
            </div>
            <div style={{ marginTop:8,fontSize:12,color:"var(--text-dim)",lineHeight:1.75,
              padding:"8px 12px",background:"var(--surface2)",borderRadius:7,
              borderLeft:`2px solid ${algorithm==="ai"?"var(--teal)":"var(--rose)"}` }}>
              {algorithm==="baseline"
                ? "Ευρετικός αλγόριθμος: επιλέγει κοντινό σταθμό βάσει απόστασης + ουράς. Απλός, χωρίς εκπαίδευση."
                : "PPO agent: εκπαιδεύτηκε για 3M βήματα. Λαμβάνει αποφάσεις από 48 χαρακτηριστικά περιβάλλοντος." }
            </div>
          </section>

          <section>
            <SectionHeader accent="var(--sky)">Προφίλ Ζήτησης</SectionHeader>
            <select value={profile} onChange={e=>setProfile(Number(e.target.value))}>
              {PROFILE_NAMES.map((n,i)=><option key={i} value={i}>{i}. {n}</option>)}
            </select>
            <div style={{ marginTop:8,fontSize:12,color:"var(--text-dim)",lineHeight:1.75,
              padding:"8px 12px",background:"var(--surface2)",borderRadius:7,
              borderLeft:"2px solid var(--sky)" }}>
              {PROFILE_DESCRIPTIONS[profile]}
            </div>
          </section>

          <section>
            <SectionHeader accent="var(--violet)">Seed</SectionHeader>
            <div style={{ display:"flex",gap:8,alignItems:"center" }}>
              <input type="number" value={seed} placeholder="π.χ. 42 (αυτόματο αν κενό)"
                onChange={e=>setSeed(e.target.value)} style={{ flex:1 }} />
              <button className="ctrl-btn" onClick={()=>setSeed(String(randSeed()))}
                style={{ fontSize:11,padding:"7px 9px" }}>🎲 Random</button>
            </div>
            <div style={{ fontSize:11,color:"var(--text-faint)",marginTop:5,paddingLeft:2 }}>
              Κενό = νέο τυχαίο seed κάθε φορά.
            </div>
          </section>

          <div style={{ display:"flex", gap:8 }}>
            <button className="btn-primary" onClick={run} disabled={loading}
              style={{ flex:1 }}>
              {loading
                ? <><div style={{ width:14,height:14,border:"2px solid #000",
                    borderTopColor:"transparent",borderRadius:"50%",
                    animation:"rotateGlow 0.8s linear infinite" }} />Εκτέλεση…</>
                : "▶  Εκτέλεση Simulation"}
            </button>
            {loading && (
              <button
                onClick={stopSimulation}
                style={{
                  padding:"12px 16px", borderRadius:10, border:"2px solid var(--rose)",
                  background:"var(--rose-dim)", color:"var(--rose)",
                  fontFamily:"var(--font-mono)", fontSize:13, fontWeight:700,
                  cursor:"pointer", flexShrink:0, transition:"all 0.15s",
                  display:"flex", alignItems:"center", gap:6,
                }}
                onMouseOver={e => { e.currentTarget.style.background="var(--rose)"; e.currentTarget.style.color="#000"; }}
                onMouseOut={e  => { e.currentTarget.style.background="var(--rose-dim)"; e.currentTarget.style.color="var(--rose)"; }}
              >
                ■ Stop
              </button>
            )}
          </div>

          {error && (
            <div style={{ background:"var(--rose-dim)",border:"1px solid var(--rose)",
              borderRadius:8,padding:"9px 12px",fontSize:11,color:"#fda4af",lineHeight:1.65 }}>
              ⚠ {error}
            </div>
          )}

          {stats && (
            <section className="fade-in">
              <SectionHeader accent="var(--lime)">Αποτελέσματα Ημέρας</SectionHeader>
              <div style={{ display:"flex",flexDirection:"column",gap:7 }}>
                <div style={{ display:"flex",gap:7 }}>
                  <StatCard label="Κέρδος" value={fmtEur(stats.net_profit)} unit="€" color="var(--lime)" />
                  <StatCard label="Service Rate" value={stats.service_rate} unit="%" color="var(--sky)" />
                </div>
                <div style={{ display:"flex",gap:7 }}>
                  <StatCard label="Αναμονή" value={stats.avg_wait_time} unit="min" color="var(--amber)" />
                  <StatCard label="Stranded" value={stats.stranded_taxis} color="var(--rose)" />
                </div>
                <div style={{ display:"flex",gap:7 }}>
                  <StatCard label="Εξυπηρετήθηκαν" value={stats.customers_served.toLocaleString()} color="var(--teal)" small />
                  <StatCard label="Εγκαταλείφθηκαν" value={stats.customers_abandoned.toLocaleString()} color="var(--rose)" small />
                </div>

                {/* Derived KPIs */}
                <div style={{ marginTop:4,padding:"9px 11px",background:"var(--surface2)",
                  borderRadius:9,border:"1px solid var(--border)",
                  display:"flex",flexDirection:"column",gap:7 }}>
                  <div style={{ fontSize:9,color:"var(--text-faint)",textTransform:"uppercase",
                    letterSpacing:"0.1em",marginBottom:2 }}>Παράγωγοι Δείκτες</div>

                  {[
                    {
                      label: "Κέρδος / ταξί",
                      value: `${fmtEur(stats.net_profit / 750)} €`,
                      color: stats.net_profit > 0 ? "var(--lime)" : "var(--rose)",
                    },
                    {
                      label: "Abandon rate",
                      value: `${(stats.customers_abandoned / Math.max(stats.customers_served + stats.customers_abandoned, 1) * 100).toFixed(1)} %`,
                      color: "var(--rose)",
                    },
                    {
                      label: "Avg revenue / served",
                      value: `${(stats.net_profit / Math.max(stats.customers_served, 1) + 40).toFixed(2)} €`,
                      color: "var(--amber)",
                    },
                  ].map(kpi => (
                    <div key={kpi.label} style={{ display:"flex",justifyContent:"space-between",
                      alignItems:"center",gap:8 }}>
                      <span style={{ fontSize:12,color:"var(--text-dim)" }}>{kpi.label}</span>
                      <span style={{ fontSize:13,fontWeight:700,color:kpi.color }}>{kpi.value}</span>
                    </div>
                  ))}
                </div>
              </div>
            </section>
          )}

          <div style={{ flex:1 }} />

          {/* ── Simulation config summary ── */}
          {stats && (
            <section className="fade-in">
              <SectionHeader accent="var(--sky)">Παράμετροι Simulation</SectionHeader>
              <div style={{ display:"flex",flexDirection:"column",gap:4,fontSize:10 }}>
                {[
                  { k:"Οχήματα",    v:"750 ταξί" },
                  { k:"Σταθμοί",    v:`${stations.length} (${stations.filter(s=>s.type==="SUPER-HUB").length} hub)` },
                  { k:"Αλγόριθμος", v:algorithm==="ai"?"PPO Agent":"Greedy Heuristic" },
                  { k:"Προφίλ",     v:PROFILE_NAMES[profile] },
                  { k:"Seed",       v:seed||"auto" },
                  { k:"Frames",     v:`${frames.length} (κάθε 15 λεπτά)` },
                  { k:"Διάρκεια",   v:"1.440 λεπτά (24ώρο)" },
                ].map(row=>(
                  <div key={row.k} style={{ display:"flex",justifyContent:"space-between",
                    padding:"4px 8px",borderRadius:6,
                    background:"var(--surface2)",gap:8 }}>
                    <span style={{ color:"var(--text-dim)",fontSize:12 }}>{row.k}</span>
                    <span style={{ color:"var(--text)",fontWeight:700,textAlign:"right",fontSize:12 }}>{row.v}</span>
                  </div>
                ))}
              </div>
            </section>
          )}

          {/* ── Keyboard shortcuts ── */}
          <section>
            <SectionHeader>Πλήκτρα</SectionHeader>
            <div style={{ display:"flex",flexDirection:"column",gap:4,fontSize:10 }}>
              {[
                { k:"Space",   v:"Play / Pause" },
                { k:"← →",    v:"Προηγ. / Επόμ. frame" },
                { k:"Home",   v:"Πρώτο frame" },
                { k:"End",    v:"Τελευταίο frame" },
              ].map(row=>(
                <div key={row.k} style={{ display:"flex",justifyContent:"space-between",
                  alignItems:"center",gap:8 }}>
                  <kbd style={{ padding:"2px 7px",borderRadius:5,border:"1px solid var(--border2)",
                    background:"var(--surface2)",color:"var(--amber)",fontSize:10,fontFamily:"var(--font-mono)" }}>
                    {row.k}
                  </kbd>
                  <span style={{ color:"var(--text-faint)",flex:1,textAlign:"right" }}>{row.v}</span>
                </div>
              ))}
            </div>
          </section>

          <section>
            <SectionHeader>State Legend</SectionHeader>
            <div style={{ display:"flex",flexDirection:"column",gap:5 }}>
              {Object.entries(STATE_META).map(([k,v])=>(
                <div key={k} style={{ display:"flex",alignItems:"center",gap:8,fontSize:11 }}>
                  <div style={{ width:8,height:8,borderRadius:"50%",background:v.color,flexShrink:0,
                    boxShadow:v.glow?`0 0 5px ${v.color}`:"none" }} />
                  <span style={{ color:"var(--text-dim)",flex:1,fontSize:12 }}>{v.label}</span>
                  {stateCounts[k]!=null && <span style={{ color:v.color,fontWeight:700,fontSize:13 }}>{stateCounts[k]}</span>}
                </div>
              ))}
            </div>
          </section>
        </aside>

        {/* ── CENTER ─────────────────────────────────────────────────────── */}
        <div style={{ flex:1,display:"flex",flexDirection:"column",
          padding:"14px 10px",gap:10,overflow:"hidden",minWidth:0 }}>

          {/* Playback bar */}
          <div className="panel" style={{ padding:"10px 14px",flexShrink:0 }}>
            <div style={{ display:"flex",alignItems:"center",gap:10,flexWrap:"wrap",rowGap:8 }}>
              {frames.length>0
                ? <SimClock minute={curMin} />
                : <div style={{ fontFamily:"var(--font-display)",fontSize:26,fontWeight:800,color:"var(--text-faint)",minWidth:78 }}>--:--</div>}

              <input type="range" min={0} max={Math.max(frames.length-1,0)} value={frameIdx}
                onChange={e=>{setPlaying(false);setFrameIdx(Number(e.target.value));}}
                disabled={!frames.length} />

              <div style={{ display:"flex",gap:5,alignItems:"center" }}>
                <button className="ctrl-btn" disabled={!frames.length}
                  onClick={()=>{setFrameIdx(0);setPlaying(false);}}>⏮</button>
                <button className="ctrl-btn" disabled={!frames.length}
                  onClick={()=>setFrameIdx(i=>Math.max(i-1,0))}>◀</button>
                <button className={`ctrl-btn${playing?" active":""}`} disabled={!frames.length}
                  onClick={()=>setPlaying(p=>!p)} style={{ minWidth:38 }}>{playing?"⏸":"▶"}</button>
                <button className="ctrl-btn" disabled={!frames.length}
                  onClick={()=>setFrameIdx(i=>Math.min(i+1,frames.length-1))}>▶</button>
                <button className="ctrl-btn" disabled={!frames.length}
                  onClick={()=>{setFrameIdx(frames.length-1);setPlaying(false);}}>⏭</button>
              </div>

              <div style={{ display:"flex",gap:4,alignItems:"center" }}>
                <span style={{ fontSize:10,color:"var(--text-faint)" }}>Ταχύτητα</span>
                {SPEED_OPTIONS.map(s=>(
                  <button key={s.ms} className={`speed-btn${speedMs===s.ms?" active":""}`}
                    onClick={()=>setSpeedMs(s.ms)}>{s.label}</button>
                ))}
              </div>
            </div>
          </div>

          {/* Map */}
          <div style={{ flex:1,display:"flex",flexDirection:"column",gap:7,minHeight:0 }}>

            {/* Map toolbar */}
            <div style={{ display:"flex",gap:7,alignItems:"center",flexShrink:0 }}>
              <span style={{ fontSize:10,color:"var(--text-faint)" }}>Χάρτης</span>
              <button className={`ctrl-btn${showHeatmap?" active":""}`}
                onClick={()=>setShowHeatmap(h=>!h)} style={{ fontSize:11 }}>Heatmap</button>
              <button className="ctrl-btn" onClick={resetView} style={{ fontSize:11 }}>⊙ Reset</button>
              <button className="ctrl-btn" onClick={()=>setZoom(z=>Math.min(z+0.3,6))} style={{ fontSize:13 }}>＋</button>
              <button className="ctrl-btn" onClick={()=>setZoom(z=>Math.max(z-0.3,0.5))} style={{ fontSize:13 }}>−</button>
              <span style={{ fontSize:11,color:"var(--amber)",fontWeight:700 }}>{zoom.toFixed(1)}×</span>
              <div style={{ flex:1 }} />
              <span style={{ fontSize:9,color:"var(--text-faint)" }}>Scroll=zoom · Drag=pan</span>
            </div>

            {/* Canvas wrapper */}
            <div style={{ flex:1,position:"relative",borderRadius:12,overflow:"hidden",
              border:"1px solid var(--border)",background:"var(--bg)",
              cursor:isPanning?"grabbing":"grab",
              display:"flex",alignItems:"center",justifyContent:"center" }}>
              <canvas ref={canvasRef} width={canvasSize} height={canvasSize}
                style={{ display:"block",maxWidth:"100%",maxHeight:"100%" }}
                onMouseDown={onMD} onMouseMove={onMM}
                onMouseUp={onMU} onMouseLeave={onMU} />

              {!frames.length && !loading && (
                <div style={{ position:"absolute",inset:0,display:"flex",flexDirection:"column",
                  alignItems:"center",justifyContent:"center",gap:12,
                  color:"var(--text-faint)",pointerEvents:"none" }}>
                  <div style={{ fontSize:48,opacity:0.25 }}>🗺</div>
                  <div style={{ fontFamily:"var(--font-display)",fontSize:15,fontWeight:700 }}>Δεν υπάρχουν δεδομένα</div>
                  <div style={{ fontSize:11 }}>Πάτα «Εκτέλεση Simulation» για να ξεκινήσει</div>
                </div>
              )}
              {loading && <LoadingOverlay progress={loadPct} />}

              {frames.length>0 && !loading && (
                <>
                  <div style={{ position:"absolute",bottom:10,right:12,fontSize:9,
                    color:"var(--text-faint)",background:"rgba(6,9,15,0.72)",
                    padding:"3px 8px",borderRadius:6,pointerEvents:"none" }}>
                    20km×20km · {totalFleet} ταξί · {stations.length} σταθμοί
                  </div>
                  <div style={{ position:"absolute",top:10,left:12,fontSize:9,
                    color:"var(--text-faint)",background:"rgba(6,9,15,0.72)",
                    padding:"3px 8px",borderRadius:6,pointerEvents:"none",
                    display:"flex",alignItems:"center",gap:8 }}>
                    <span style={{ color:algorithm==="ai"?"var(--teal)":"var(--rose)",fontWeight:700 }}>
                      {algorithm==="ai"?"PPO AI":"Greedy"}
                    </span>
                    <span>·</span>
                    <span>{PROFILE_NAMES[profile]}</span>
                    <span>·</span>
                    <span style={{ color:"var(--amber)" }}>{fmtMin(curMin)}</span>
                  </div>
                </>
              )}
            </div>
          </div>

          {/* ── Bottom center: Day phase + per-hour demand ── */}
          {(frames.length>0 || queueTS.length>0) && (
            <div className="panel" style={{ padding:"12px 16px",flexShrink:0,display:"flex",
              flexDirection:"column",gap:10 }}>
              <div>
                <div style={{ fontSize:9,color:"var(--text-faint)",textTransform:"uppercase",
                  letterSpacing:"0.12em",marginBottom:6,fontWeight:700 }}>Φάσεις Ημέρας</div>
                <DayPhaseBar currentMinute={curMin} />
              </div>
              {queueTS.length>0 && (
                <div>
                  <div style={{ fontSize:9,color:"var(--text-faint)",textTransform:"uppercase",
                    letterSpacing:"0.12em",marginBottom:6,fontWeight:700,display:"flex",
                    alignItems:"center",gap:8 }}>
                    Αιχμές Ουράς / Ώρα
                    <span style={{ color:"var(--rose)",fontSize:8,fontWeight:400,
                      textTransform:"none",letterSpacing:0 }}>● peak &gt;75% max</span>
                    <span style={{ color:"var(--amber)",fontSize:8,fontWeight:400,
                      textTransform:"none",letterSpacing:0 }}>● medium</span>
                    <span style={{ color:"var(--sky)",fontSize:8,fontWeight:400,
                      textTransform:"none",letterSpacing:0 }}>● χαμηλό</span>
                  </div>
                  <DemandHeatRow queueTS={queueTS} currentMin={curMin} />
                </div>
              )}
            </div>
          )}
        </div>
        <aside style={{ width:360,flexShrink:0,background:"var(--surface)",
          borderLeft:"1px solid var(--border)",padding:"18px 18px",
          display:"flex",flexDirection:"column",gap:14,overflowY:"auto" }}>

          <section>
            <SectionHeader accent="var(--amber)">Live Metrics</SectionHeader>
            <div style={{ display:"flex",gap:7 }}>
              <div className="stat-card">
                <div style={{ fontSize:9,color:"var(--text-faint)",textTransform:"uppercase",
                  letterSpacing:"0.1em",marginBottom:5 }}>Μέσο SoC</div>
                {avgSocNow!=null ? <>
                  <div style={{ fontSize:20,fontWeight:700,fontFamily:"var(--font-display)",
                    color:avgSocNow<0.3?"var(--rose)":avgSocNow<0.5?"var(--amber)":"var(--lime)" }}>
                    {(avgSocNow*100).toFixed(1)}
                    <span style={{ fontSize:11,fontWeight:400,color:"var(--text-faint)",marginLeft:2 }}>%</span>
                  </div>
                  <div style={{ height:3,background:"var(--border)",borderRadius:99,marginTop:6,overflow:"hidden" }}>
                    <div style={{ height:"100%",width:`${avgSocNow*100}%`,borderRadius:99,transition:"width 0.3s",
                      background:avgSocNow<0.3?"var(--rose)":avgSocNow<0.5?"var(--amber)":"var(--lime)" }} />
                  </div>
                </> : <div style={{ color:"var(--text-faint)",fontSize:14 }}>—</div>}
              </div>
              <div className="stat-card">
                <div style={{ fontSize:9,color:"var(--text-faint)",textTransform:"uppercase",
                  letterSpacing:"0.1em",marginBottom:5 }}>Ουρά σταθμών</div>
                <div style={{ fontSize:20,fontWeight:700,fontFamily:"var(--font-display)",
                  color:(totalQNow??0)>30?"var(--rose)":"var(--text)" }}>
                  {totalQNow!=null ? totalQNow : <span style={{ color:"var(--text-faint)" }}>—</span>}
                  <span style={{ fontSize:11,fontWeight:400,color:"var(--text-faint)",marginLeft:3 }}>ταξί</span>
                </div>
              </div>
            </div>
          </section>

          {/* Fleet Health Gauges */}
          {totalFleet>0 && (
            <section>
              <SectionHeader accent="var(--teal)">Fleet Health</SectionHeader>
              <div className="panel" style={{ padding:"10px 12px" }}>
                <FleetHealthGauges stateCounts={stateCounts} totalFleet={totalFleet} avgSocNow={avgSocNow} />
              </div>
            </section>
          )}

          {totalFleet>0 && (
            <section>
              <SectionHeader accent="var(--sky)">Κατανομή Στόλου</SectionHeader>
              <StateDistBar counts={stateCounts} total={totalFleet} />
            </section>
          )}

          {stations.length>0 && curFrame && (
            <section>
              <SectionHeader accent="var(--violet)">Ουρές Σταθμών</SectionHeader>
              <StationList stations={stations} queues={currentQueues} />
            </section>
          )}

          {queueTS.length>0 && (
            <section>
              <SectionHeader accent="var(--rose)">Ουρά ανά Ώρα</SectionHeader>
              <div className="panel" style={{ padding:"10px 12px" }}>
                <Sparkline data={queueTS} color="#f43f5e" height={50} width={302} cursorIdx={tsIdx} />
                <div style={{ display:"flex",justifyContent:"space-between",marginTop:4,fontSize:10,color:"var(--text-faint)" }}>
                  <span>00:00</span>
                  <span style={{ color:"var(--rose)",fontWeight:700 }}>
                    {queueTS[tsIdx]!=null ? `τώρα: ${queueTS[tsIdx]}` : "—"}
                  </span>
                  <span>23:45</span>
                </div>
              </div>
            </section>
          )}

          {socTS.length>0 && (
            <section>
              <SectionHeader accent="var(--lime)">Μέσο SoC ανά Ώρα</SectionHeader>
              <div className="panel" style={{ padding:"10px 12px" }}>
                <Sparkline data={socTS.map(v=>v*100)} color="#84cc16" height={50} width={302} cursorIdx={tsIdx} />
                <div style={{ display:"flex",justifyContent:"space-between",marginTop:4,fontSize:10,color:"var(--text-faint)" }}>
                  <span>00:00</span>
                  <span style={{ color:"var(--lime)",fontWeight:700 }}>
                    {socTS[tsIdx]!=null ? `τώρα: ${(socTS[tsIdx]*100).toFixed(1)}%` : "—"}
                  </span>
                  <span>23:45</span>
                </div>
              </div>
            </section>
          )}

          {queueTS.length>0 && (
            <section>
              <SectionHeader accent="var(--sky)">Αιχμές Ζήτησης / Ώρα</SectionHeader>
              <div className="panel" style={{ padding:"10px 12px" }}>
                <DemandHeatRow queueTS={queueTS} currentMin={curMin} />
              </div>
            </section>
          )}

          {frames.length>0 && (
            <section>
              <SectionHeader>Πρόοδος Ημέρας</SectionHeader>
              <DayPhaseBar currentMinute={curMin} />
              <div style={{ marginTop:8,height:4,background:"var(--surface2)",borderRadius:99,overflow:"hidden" }}>
                <div style={{ height:"100%",
                  width:`${(frameIdx/Math.max(frames.length-1,1))*100}%`,
                  background:"linear-gradient(90deg,var(--amber),var(--rose))",
                  borderRadius:99,transition:"width 0.2s",boxShadow:"0 0 8px var(--amber)" }} />
              </div>
              <div style={{ display:"flex",justifyContent:"space-between",marginTop:4,
                fontSize:9,color:"var(--text-faint)" }}>
                <span>00:00</span>
                <span style={{ color:"var(--amber)",fontWeight:700 }}>{fmtMin(curMin)}</span>
                <span>23:45</span>
              </div>
            </section>
          )}

          <section>
            <SectionHeader>Αλγόριθμος — Λεπτομέρειες</SectionHeader>
            <AlgoInfoCard algorithm={algorithm} />
          </section>

          <div style={{ flex:1 }} />

          <div style={{ padding:"10px 12px",borderRadius:10,background:"var(--surface2)",
            border:"1px solid var(--border)",fontSize:12,color:"var(--text-dim)",
            lineHeight:1.9,flexShrink:0 }}>
            <div style={{ color:"var(--amber)",fontWeight:700,marginBottom:5,
              fontFamily:"var(--font-display)",fontSize:13 }}>📌 Οδηγίες Χάρτη</div>
            <div style={{ display:"grid",gridTemplateColumns:"auto 1fr",gap:"2px 10px" }}>
              <span style={{ color:"var(--amber)" }}>Scroll</span><span>Zoom In / Out</span>
              <span style={{ color:"var(--amber)" }}>Drag</span><span>Pan (μετακίνηση)</span>
              <span style={{ color:"var(--amber)" }}>Heatmap</span><span>Πυκνότητα ταξί</span>
              <span style={{ color:"var(--amber)" }}>⊙ Reset</span><span>Επαναφορά view</span>
              <span style={{ color:"var(--amber)" }}>0.25×</span><span>Αργή αναπαραγωγή</span>
            </div>
            <div style={{ marginTop:8,display:"flex",gap:10,flexWrap:"wrap" }}>
              <span><span style={{ color:"var(--amber)" }}>●</span> SUPER-HUB</span>
              <span><span style={{ color:"var(--violet)" }}>●</span> Σταθμός</span>
              <span><span style={{ color:"var(--rose)" }}>●</span> Stranded</span>
              <span><span style={{ color:"var(--sky)" }}>●</span> Με πελάτη</span>
            </div>
          </div>
        </aside>
      </div>

      {/* ══ STATUS BAR ══════════════════════════════════════════════════════ */}
      <footer style={{ height:26,flexShrink:0,
        background:"var(--surface)",borderTop:"1px solid var(--border)",
        display:"flex",alignItems:"center",padding:"0 16px",gap:14,
        fontSize:11,color:"var(--text-faint)",overflow:"hidden" }}>
        <span style={{ color:algorithm==="ai"?"var(--teal)":"var(--rose)",fontWeight:700 }}>
          {algorithm==="ai"?"🤖 PPO AI":"🔴 Greedy"}
        </span>
        <span style={{ color:"var(--border2)" }}>│</span>
        <span>Προφίλ: <strong style={{ color:"var(--amber)" }}>{PROFILE_NAMES[profile]}</strong></span>
        {seed!="" && <><span style={{ color:"var(--border2)" }}>│</span>
          <span>Seed: <strong style={{ color:"var(--text-dim)" }}>{seed}</strong></span></>}
        <div style={{ flex:1 }} />
        {curFrame && Object.entries(STATE_META).map(([k,v]) => {
          const n = stateCounts[k]||0;
          if (!n) return null;
          return (
            <span key={k} style={{ display:"flex",alignItems:"center",gap:3 }}>
              <span style={{ width:5,height:5,borderRadius:"50%",background:v.color,
                display:"inline-block",
                boxShadow:v.glow?`0 0 4px ${v.color}`:"none" }} />
              <span style={{ color:v.color,fontWeight:700 }}>{n}</span>
            </span>
          );
        })}
        <div style={{ flex:1 }} />
        {frames.length>0 && (
          <>
            <span>⏱ <strong style={{ color:"var(--amber)" }}>{fmtMin(curMin)}</strong></span>
            <span style={{ color:"var(--border2)" }}>│</span>
            <span>Frame <strong style={{ color:"var(--text-dim)" }}>{frameIdx+1}/{frames.length}</strong></span>
            <span style={{ color:"var(--border2)" }}>│</span>
          </>
        )}
        <span><strong style={{ color:"var(--text-dim)" }}>{zoom.toFixed(1)}×</strong></span>
        <span style={{ color:"var(--border2)" }}>│</span>
      </footer>
    </div>
  );
}