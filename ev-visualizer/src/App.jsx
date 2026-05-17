import { useState, useEffect, useRef, useCallback, useMemo } from "react";

// ─── Google Fonts ─────────────────────────────────────────────────────────────
const fontLink = document.createElement("link");
fontLink.rel = "stylesheet";
fontLink.href = "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap";
document.head.appendChild(fontLink);

// ─── CSS Injection ────────────────────────────────────────────────────────────
const css = `
  :root {
    --bg:           #0a0a0a;
    --bg-elevated:  #111111;
    --surface:      #171717;
    --surface2:     #1f1f1f;
    --surface3:     #262626;
    --border:       rgba(255,255,255,0.08);
    --border-hover: rgba(255,255,255,0.14);
    --accent:       #00d4ff;
    --accent-dim:   rgba(0,212,255,0.12);
    --emerald:      #10b981;
    --emerald-dim:  rgba(16,185,129,0.12);
    --rose:         #f43f5e;
    --rose-dim:     rgba(244,63,94,0.10);
    --amber:        #f59e0b;
    --amber-dim:    rgba(245,158,11,0.12);
    --violet:       #a78bfa;
    --violet-dim:   rgba(167,139,250,0.12);
    --sky:          #38bdf8;
    --sky-dim:      rgba(56,189,248,0.10);
    --text:         #fafafa;
    --text-secondary: #a1a1aa;
    --text-muted:   #71717a;
    --font-sans:    'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    --font-mono:    'JetBrains Mono', 'SF Mono', 'Monaco', monospace;
    --radius:       10px;
    --radius-sm:    6px;
    --radius-lg:    14px;
    --shadow-sm:    0 1px 2px rgba(0,0,0,0.3);
    --shadow-md:    0 4px 12px rgba(0,0,0,0.4);
    --shadow-lg:    0 8px 24px rgba(0,0,0,0.5);
    --transition:   150ms cubic-bezier(0.4, 0, 0.2, 1);
  }

  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  
  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--font-sans);
    overflow: hidden;
    height: 100vh;
    width: 100vw;
    font-size: 14px;
    line-height: 1.5;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
  }

  ::-webkit-scrollbar { width: 6px; height: 6px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { 
    background: var(--surface3); 
    border-radius: 99px;
    transition: background var(--transition);
  }
  ::-webkit-scrollbar-thumb:hover { background: var(--text-muted); }

  /* Animations */
  @keyframes fadeIn {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
  }
  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
  }
  @keyframes spin {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
  }
  @keyframes shimmer {
    0% { background-position: -200% 0; }
    100% { background-position: 200% 0; }
  }

  .fade-in { animation: fadeIn 0.3s ease-out both; }
  .pulse { animation: pulse 2s ease-in-out infinite; }

  /* Card styles */
  .card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    transition: border-color var(--transition), box-shadow var(--transition);
  }
  .card:hover {
    border-color: var(--border-hover);
  }

  /* Panel styles */
  .panel {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
  }

  /* Form elements */
  input[type="number"], select {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    color: var(--text);
    font-family: var(--font-sans);
    font-size: 13px;
    padding: 10px 12px;
    outline: none;
    width: 100%;
    transition: all var(--transition);
  }
  input[type="number"]:focus, select:focus {
    border-color: var(--accent);
    box-shadow: 0 0 0 3px var(--accent-dim);
  }
  input[type="number"]::placeholder { color: var(--text-muted); }
  select option { background: var(--surface2); color: var(--text); }

  /* Range input */
  input[type="range"] {
    -webkit-appearance: none;
    height: 4px;
    border-radius: 99px;
    background: var(--surface3);
    outline: none;
    cursor: pointer;
    flex: 1;
  }
  input[type="range"]::-webkit-slider-thumb {
    -webkit-appearance: none;
    width: 16px;
    height: 16px;
    border-radius: 50%;
    background: var(--accent);
    cursor: pointer;
    box-shadow: 0 0 10px var(--accent-dim), var(--shadow-sm);
    transition: transform var(--transition);
  }
  input[type="range"]::-webkit-slider-thumb:hover {
    transform: scale(1.1);
  }

  /* Button base */
  .btn {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    font-family: var(--font-sans);
    font-weight: 500;
    font-size: 13px;
    border-radius: var(--radius-sm);
    cursor: pointer;
    transition: all var(--transition);
    border: none;
    outline: none;
  }
  .btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  /* Primary button */
  .btn-primary {
    width: 100%;
    padding: 12px 16px;
    background: var(--text);
    color: var(--bg);
    font-weight: 600;
    border-radius: var(--radius);
    letter-spacing: -0.01em;
  }
  .btn-primary:hover:not(:disabled) {
    opacity: 0.9;
    transform: translateY(-1px);
  }
  .btn-primary:active:not(:disabled) {
    transform: translateY(0);
  }

  /* Control button */
  .ctrl-btn {
    padding: 6px 12px;
    background: var(--surface2);
    color: var(--text-secondary);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    font-size: 13px;
  }
  .ctrl-btn:hover:not(:disabled) {
    background: var(--surface3);
    color: var(--text);
    border-color: var(--border-hover);
  }
  .ctrl-btn.active {
    background: var(--accent-dim);
    color: var(--accent);
    border-color: rgba(0,212,255,0.3);
  }

  /* Algorithm toggle buttons */
  .algo-btn {
    flex: 1;
    padding: 10px 12px;
    background: transparent;
    color: var(--text-muted);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    font-family: var(--font-sans);
    font-size: 13px;
    font-weight: 500;
    cursor: pointer;
    transition: all var(--transition);
  }
  .algo-btn:first-child { border-radius: var(--radius-sm) 0 0 var(--radius-sm); }
  .algo-btn:last-child { border-radius: 0 var(--radius-sm) var(--radius-sm) 0; margin-left: -1px; }
  .algo-btn.active-greedy {
    background: var(--rose-dim);
    color: var(--rose);
    border-color: rgba(244,63,94,0.3);
    z-index: 1;
  }
  .algo-btn.active-ai {
    background: var(--emerald-dim);
    color: var(--emerald);
    border-color: rgba(16,185,129,0.3);
    z-index: 1;
  }

  /* Speed buttons */
  .speed-btn {
    padding: 4px 10px;
    background: transparent;
    color: var(--text-muted);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    font-family: var(--font-mono);
    font-size: 11px;
    cursor: pointer;
    transition: all var(--transition);
  }
  .speed-btn:hover:not(.active) {
    background: var(--surface2);
    color: var(--text-secondary);
  }
  .speed-btn.active {
    background: var(--accent-dim);
    color: var(--accent);
    border-color: rgba(0,212,255,0.3);
    font-weight: 600;
  }

  /* Tag/Badge */
  .tag {
    display: inline-flex;
    align-items: center;
    padding: 3px 8px;
    border-radius: 99px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.02em;
  }

  /* Stat card */
  .stat-card {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 14px 16px;
    transition: all var(--transition);
    flex: 1;
  }
  .stat-card:hover {
    border-color: var(--border-hover);
    background: var(--surface3);
  }

  /* Section animations */
  section { animation: fadeIn 0.25s ease-out both; }

  /* Aside scrollbar */
  aside::-webkit-scrollbar { width: 4px; }

  /* Canvas */
  canvas { image-rendering: auto; }

  /* KBD */
  kbd {
    padding: 3px 8px;
    border-radius: var(--radius-sm);
    border: 1px solid var(--border);
    background: var(--surface2);
    font-family: var(--font-mono);
    font-size: 11px;
    color: var(--text-secondary);
  }
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
  "Typical weekday with moderate demand and morning peak.",
  "Heavy morning/evening peaks from commuters.",
  "Saturday profile: afternoon and evening peak.",
  "Sunday: low demand, uniform distribution throughout the day.",
  "Minimal traffic - ideal for efficiency evaluation.",
  "Extreme load: multiple peaks, stress-test scenario.",
  "Flat distribution without major peaks throughout the day.",
  "Bipolar: strong morning and evening, quiet at noon.",
  "Special event: sudden local peak in the middle of the day.",
  "Early surge of demand in the first minutes of the day.",
];

const STATE_META = {
  0: { label: "Idle",            color: "#71717a", glow: false },
  1: { label: "With Customer",   color: "#38bdf8", glow: true  },
  2: { label: "Rebalancing",     color: "#a78bfa", glow: false },
  3: { label: "Waiting Charger", color: "#f59e0b", glow: true  },
  4: { label: "Charging",        color: "#10b981", glow: true  },
  5: { label: "Stranded",        color: "#f43f5e", glow: true  },
};

const SPEED_OPTIONS = [
  { label: "0.25x", ms: 800  },
  { label: "0.5x",  ms: 400  },
  { label: "1x",    ms: 200  },
  { label: "2x",    ms: 120  },
  { label: "4x",    ms: 60   },
];

// ─── Helpers ──────────────────────────────────────────────────────────────────
const toCanvas = (km, S, offset, scale) => (km / CITY_KM) * S * scale - offset;
const fmtMin   = (m) => `${String(Math.floor(m/60)%24).padStart(2,"0")}:${String(m%60).padStart(2,"0")}`;
const fmtEur   = (n) => new Intl.NumberFormat("en-US",{maximumFractionDigits:0}).format(n);
const randSeed = () => Math.floor(Math.random() * 99999) + 1;

// ─── Icons ────────────────────────────────────────────────────────────────────
const Icons = {
  play: (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
      <path d="M8 5.14v14l11-7-11-7z"/>
    </svg>
  ),
  pause: (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
      <path d="M6 4h4v16H6V4zm8 0h4v16h-4V4z"/>
    </svg>
  ),
  skipBack: (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
      <path d="M6 6h2v12H6V6zm3.5 6l8.5 6V6l-8.5 6z"/>
    </svg>
  ),
  skipForward: (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
      <path d="M16 6h2v12h-2V6zm-2.5 6L5 6v12l8.5-6z"/>
    </svg>
  ),
  chevronLeft: (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M15 18l-6-6 6-6"/>
    </svg>
  ),
  chevronRight: (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M9 18l6-6-6-6"/>
    </svg>
  ),
  zap: (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
      <path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z"/>
    </svg>
  ),
  stop: (
    <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor">
      <rect x="6" y="6" width="12" height="12" rx="1"/>
    </svg>
  ),
  refresh: (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M23 4v6h-6M1 20v-6h6"/>
      <path d="M3.51 9a9 9 0 0114.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0020.49 15"/>
    </svg>
  ),
  target: (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>
    </svg>
  ),
  layers: (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/>
    </svg>
  ),
  plus: (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/>
    </svg>
  ),
  minus: (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <line x1="5" y1="12" x2="19" y2="12"/>
    </svg>
  ),
};

// ─── Sparkline ────────────────────────────────────────────────────────────────
function Sparkline({ data, color, height = 52, width = 220, cursorIdx }) {
  if (!data || data.length < 2) return <div style={{ height, width }} />;
  const max = Math.max(...data, 1);
  const min = Math.min(...data);
  const range = max - min || 1;
  const pts = data.map((v, i) => [
    (i / (data.length - 1)) * width,
    height - ((v - min) / range) * (height - 8) - 4,
  ]);
  const polyline = pts.map(p => p.join(",")).join(" ");
  const area = [`0,${height}`, ...pts.map(p => p.join(",")), `${width},${height}`].join(" ");
  const gid = `sg${color.replace(/[^a-z0-9]/gi,"")}`;
  const cx = cursorIdx != null ? pts[Math.min(cursorIdx, pts.length - 1)] : null;
  return (
    <svg width={width} height={height} style={{ display:"block", overflow:"visible" }}>
      <defs>
        <linearGradient id={gid} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%"   stopColor={color} stopOpacity="0.25" />
          <stop offset="100%" stopColor={color} stopOpacity="0"    />
        </linearGradient>
      </defs>
      <polygon points={area} fill={`url(#${gid})`} />
      <polyline points={polyline} fill="none" stroke={color} strokeWidth="1.5" strokeLinejoin="round" strokeLinecap="round" />
      {cx && <>
        <line x1={cx[0]} y1={0} x2={cx[0]} y2={height} stroke={color} strokeWidth="1" strokeDasharray="4,4" opacity="0.4" />
        <circle cx={cx[0]} cy={cx[1]} r="4" fill={color} stroke="var(--bg)" strokeWidth="2" />
      </>}
    </svg>
  );
}

// ─── Section Header ───────────────────────────────────────────────────────────
function SectionHeader({ children, accent = "var(--accent)" }) {
  return (
    <div style={{ 
      fontSize: 11, 
      fontWeight: 600, 
      textTransform: "uppercase", 
      letterSpacing: "0.08em",
      color: "var(--text-muted)", 
      display: "flex", 
      alignItems: "center", 
      gap: 10, 
      marginBottom: 12 
    }}>
      <span style={{ 
        width: 3, 
        height: 14, 
        background: accent, 
        borderRadius: 99, 
        flexShrink: 0 
      }} />
      {children}
    </div>
  );
}

// ─── StatCard ─────────────────────────────────────────────────────────────────
function StatCard({ label, value, unit, color, small }) {
  return (
    <div className="stat-card">
      <div style={{ 
        fontSize: 11, 
        color: "var(--text-muted)", 
        textTransform: "uppercase", 
        letterSpacing: "0.05em", 
        marginBottom: 6 
      }}>
        {label}
      </div>
      <div style={{ 
        fontSize: small ? 20 : 26, 
        fontWeight: 700, 
        color: color || "var(--text)", 
        fontFamily: "var(--font-sans)", 
        lineHeight: 1,
        letterSpacing: "-0.02em"
      }}>
        {value}
        {unit && (
          <span style={{ 
            fontSize: 12, 
            fontWeight: 500, 
            color: "var(--text-muted)", 
            marginLeft: 4 
          }}>
            {unit}
          </span>
        )}
      </div>
    </div>
  );
}

// ─── Loading Overlay ──────────────────────────────────────────────────────────
function LoadingOverlay({ progress }) {
  return (
    <div style={{ 
      position: "absolute", 
      inset: 0, 
      background: "rgba(10,10,10,0.95)",
      backdropFilter: "blur(8px)",
      display: "flex", 
      alignItems: "center", 
      justifyContent: "center",
      flexDirection: "column", 
      gap: 24, 
      zIndex: 50, 
      borderRadius: "var(--radius)" 
    }}>
      <div style={{ position: "relative", width: 80, height: 80 }}>
        <div style={{ 
          position: "absolute", 
          inset: 0, 
          border: "2px solid var(--border)", 
          borderRadius: "50%" 
        }} />
        <div style={{ 
          position: "absolute", 
          inset: 0, 
          border: "2px solid transparent",
          borderTopColor: "var(--accent)", 
          borderRadius: "50%",
          animation: "spin 1s linear infinite" 
        }} />
        <div style={{ 
          position: "absolute", 
          inset: 10, 
          border: "2px solid transparent",
          borderTopColor: "var(--emerald)", 
          borderRadius: "50%",
          animation: "spin 1.5s linear infinite reverse" 
        }} />
        <div style={{ 
          position: "absolute", 
          inset: 0, 
          display: "flex", 
          alignItems: "center",
          justifyContent: "center", 
          fontSize: 14, 
          fontWeight: 700, 
          fontFamily: "var(--font-mono)",
          color: "var(--accent)" 
        }}>
          {progress}%
        </div>
      </div>
      <div style={{ width: 280 }}>
        <div style={{ 
          height: 4, 
          background: "var(--surface3)", 
          borderRadius: 99, 
          overflow: "hidden" 
        }}>
          <div style={{ 
            height: "100%", 
            width: `${progress}%`,
            background: "linear-gradient(90deg, var(--accent), var(--emerald))", 
            borderRadius: 99,
            transition: "width 0.3s ease" 
          }} />
        </div>
        <div style={{ 
          display: "flex", 
          justifyContent: "space-between", 
          marginTop: 8, 
          fontSize: 12, 
          color: "var(--text-muted)" 
        }}>
          <span>Running simulation...</span>
          <span style={{ color: "var(--accent)", fontFamily: "var(--font-mono)" }}>{progress}%</span>
        </div>
      </div>
    </div>
  );
}

// ─── Clock ────────────────────────────────────────────────────────────────────
function SimClock({ minute }) {
  const h = Math.floor(minute / 60) % 24;
  const m = minute % 60;
  const isNight = h < 6 || h >= 22;
  const isPeak  = (h >= 7 && h <= 9) || (h >= 16 && h <= 19);
  const period  = isNight 
    ? { label: "Night", color: "var(--violet)" }
    : isPeak  
    ? { label: "Peak",  color: "var(--amber)" }
    : { label: "Day",   color: "var(--sky)" };
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 12, flexShrink: 0 }}>
      <div style={{ 
        fontFamily: "var(--font-mono)", 
        fontSize: 28, 
        fontWeight: 700,
        color: period.color, 
        letterSpacing: "-0.02em",
        minWidth: 90 
      }}>
        {String(h).padStart(2,"0")}:{String(m).padStart(2,"0")}
      </div>
      <div>
        <div style={{ 
          fontSize: 10, 
          color: "var(--text-muted)", 
          textTransform: "uppercase", 
          letterSpacing: "0.1em",
          marginBottom: 2
        }}>
          Period
        </div>
        <span className="tag" style={{ 
          background: `${period.color}20`, 
          color: period.color 
        }}>
          {period.label}
        </span>
      </div>
    </div>
  );
}

// ─── State Distribution Bar ───────────────────────────────────────────────────
function StateDistBar({ counts, total }) {
  if (!total) return null;
  return (
    <div>
      <div style={{ 
        display: "flex", 
        height: 8, 
        borderRadius: 99, 
        overflow: "hidden", 
        gap: 2,
        background: "var(--surface2)"
      }}>
        {Object.entries(STATE_META).map(([k,v]) => {
          const n = counts[k] || 0;
          const pct = (n / total) * 100;
          if (pct < 0.5) return null;
          return (
            <div 
              key={k} 
              style={{ 
                width: `${pct}%`, 
                background: v.color, 
                transition: "width 0.4s ease" 
              }} 
              title={`${v.label}: ${n}`} 
            />
          );
        })}
      </div>
      <div style={{ display: "flex", flexWrap: "wrap", gap: "6px 14px", marginTop: 10 }}>
        {Object.entries(STATE_META).map(([k,v]) => {
          const n = counts[k] || 0;
          if (!n) return null;
          return (
            <div key={k} style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 12 }}>
              <div style={{ 
                width: 8, 
                height: 8, 
                borderRadius: "50%", 
                background: v.color, 
                flexShrink: 0,
                boxShadow: v.glow ? `0 0 6px ${v.color}` : "none"
              }} />
              <span style={{ color: "var(--text-secondary)" }}>{v.label}</span>
              <span style={{ color: v.color, fontWeight: 600 }}>{n}</span>
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
  const sorted = [...stations].sort((a,b) => (queues[b.id] || 0) - (queues[a.id] || 0));
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
      {sorted.slice(0, 10).map(st => {
        const q = queues[st.id] || 0;
        const isHub = st.type === "SUPER-HUB";
        const color = isHub ? "var(--amber)" : "var(--violet)";
        const pct = Math.min((q / 20) * 100, 100);
        return (
          <div 
            key={st.id} 
            style={{ 
              display: "flex", 
              alignItems: "center", 
              gap: 10, 
              padding: "8px 10px",
              borderRadius: "var(--radius-sm)", 
              background: q > 5 ? "var(--rose-dim)" : "var(--surface2)",
              border: `1px solid ${q > 5 ? "rgba(244,63,94,0.2)" : "transparent"}`,
              transition: "all var(--transition)"
            }}
          >
            <div style={{ 
              width: 24, 
              textAlign: "right", 
              fontSize: 11, 
              color: "var(--text-muted)",
              fontFamily: "var(--font-mono)"
            }}>
              #{st.id}
            </div>
            <div style={{ 
              width: 8, 
              height: 8, 
              borderRadius: "50%", 
              background: color, 
              flexShrink: 0 
            }} />
            <div style={{ flex: 1 }}>
              <div style={{ height: 4, background: "var(--surface3)", borderRadius: 99 }}>
                <div style={{ 
                  height: "100%", 
                  width: `${pct}%`,
                  background: q > 5 ? "var(--rose)" : color, 
                  borderRadius: 99, 
                  transition: "width 0.3s" 
                }} />
              </div>
            </div>
            <div style={{ 
              minWidth: 28, 
              textAlign: "right", 
              fontSize: 13, 
              fontWeight: 600,
              fontFamily: "var(--font-mono)",
              color: q > 5 ? "var(--rose)" : q > 0 ? color : "var(--text-muted)" 
            }}>
              {q}
            </div>
            <span className="tag" style={{ 
              background: isHub ? "var(--amber-dim)" : "var(--violet-dim)",
              color: isHub ? "var(--amber)" : "var(--violet)", 
              fontSize: 9,
              padding: "2px 6px"
            }}>
              {isHub ? "HUB" : "STD"}
            </span>
          </div>
        );
      })}
    </div>
  );
}

// ─── Canvas Draw ──────────────────────────────────────────────────────────────
function drawCity(ctx, S, scale, offX, offY, frame, stations, heatmap) {
  ctx.clearRect(0, 0, S, S);

  // Background
  ctx.fillStyle = "#0a0a0a";
  ctx.fillRect(0, 0, S, S);

  // Center glow
  const cx0 = toCanvas(10, S, offX, scale);
  const cy0 = toCanvas(10, S, offY, scale);
  const grd = ctx.createRadialGradient(cx0, cy0, 0, cx0, cy0, S * 0.5 * scale);
  grd.addColorStop(0, "rgba(0,212,255,0.03)");
  grd.addColorStop(1, "transparent");
  ctx.fillStyle = grd;
  ctx.fillRect(0, 0, S, S);

  // Grid
  ctx.strokeStyle = "rgba(255,255,255,0.03)";
  ctx.lineWidth = 0.5;
  for (let g = 0; g <= CITY_KM; g += 2) {
    const px = toCanvas(g, S, offX, scale);
    const py = toCanvas(g, S, offY, scale);
    ctx.beginPath(); ctx.moveTo(px, 0); ctx.lineTo(px, S); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(0, py); ctx.lineTo(S, py); ctx.stroke();
  }

  if (!frame) return;

  // Heatmap
  if (heatmap) {
    const cells = 20;
    const km = CITY_KM / cells;
    const dens = Array.from({length: cells}, () => new Array(cells).fill(0));
    frame.taxis.forEach(ev => {
      if (ev.s === 5) return;
      const xi = Math.min(Math.floor(ev.x / km), cells - 1);
      const yi = Math.min(Math.floor(ev.y / km), cells - 1);
      if (xi >= 0 && yi >= 0) dens[yi][xi]++;
    });
    const maxD = Math.max(...dens.flat(), 1);
    const cpx = km * (S / CITY_KM) * scale;
    for (let row = 0; row < cells; row++) {
      for (let col = 0; col < cells; col++) {
        const d = dens[row][col];
        if (!d) continue;
        const alpha = (d / maxD) * 0.18;
        ctx.fillStyle = `rgba(0,212,255,${alpha})`;
        ctx.fillRect(toCanvas(col * km, S, offX, scale), toCanvas(row * km, S, offY, scale), cpx + 1, cpx + 1);
      }
    }
  }

  // Stations
  stations.forEach(st => {
    const x = toCanvas(st.x, S, offX, scale);
    const y = toCanvas(st.y, S, offY, scale);
    if (x < -20 || x > S + 20 || y < -20 || y > S + 20) return;
    const q   = (frame.queues && frame.queues[st.id]) || 0;
    const hub = st.type === "SUPER-HUB";
    const r   = hub ? 14 : 9;
    const col = hub ? "#f59e0b" : "#a78bfa";

    // Halo
    const hg = ctx.createRadialGradient(x, y, r, x, y, r * 3);
    hg.addColorStop(0, hub ? "rgba(245,158,11,0.12)" : "rgba(167,139,250,0.08)");
    hg.addColorStop(1, "transparent");
    ctx.fillStyle = hg;
    ctx.beginPath(); ctx.arc(x, y, r * 3, 0, Math.PI * 2); ctx.fill();

    ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fillStyle = hub ? "rgba(245,158,11,0.15)" : "rgba(167,139,250,0.12)"; 
    ctx.fill();
    ctx.strokeStyle = col; 
    ctx.lineWidth = hub ? 2 : 1.5; 
    ctx.stroke();

    if (scale > 0.8) {
      ctx.fillStyle = col;
      ctx.font = `bold ${hub ? 11 : 8}px sans-serif`;
      ctx.textAlign = "center"; 
      ctx.textBaseline = "middle";
      ctx.fillText("⚡", x, y);
    }

    if (q > 0) {
      const br = 8, bx = x + r - 2, by = y - r + 2;
      ctx.beginPath(); ctx.arc(bx, by, br, 0, Math.PI * 2);
      ctx.fillStyle = q > 5 ? "#f43f5e" : "#fb923c"; 
      ctx.fill();
      ctx.fillStyle = "#fff";
      ctx.font = "bold 9px var(--font-mono), monospace";
      ctx.textAlign = "center"; 
      ctx.textBaseline = "middle";
      ctx.fillText(q > 99 ? "99" : String(q), bx, by);
    }
  });

  // Taxis
  frame.taxis.forEach(ev => {
    const x = toCanvas(ev.x, S, offX, scale);
    const y = toCanvas(ev.y, S, offY, scale);
    if (x < -4 || x > S + 4 || y < -4 || y > S + 4) return;
    const m = STATE_META[ev.s] || STATE_META[0];
    const r = scale > 1.5 ? 3.5 : 2.2;
    if (m.glow && scale > 0.9) {
      ctx.beginPath(); ctx.arc(x, y, r * 2.5, 0, Math.PI * 2);
      ctx.fillStyle = `${m.color}15`; 
      ctx.fill();
    }
    ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fillStyle = m.color; 
    ctx.fill();
  });
}

// ─── Day-Phase Timeline ───────────────────────────────────────────────────────
function DayPhaseBar({ currentMinute }) {
  const phases = [
    { label: "Night",   start: 0,    end: 360,  color: "#a78bfa" },
    { label: "Morning", start: 360,  end: 540,  color: "#fb923c" },
    { label: "Peak",    start: 540,  end: 600,  color: "#f59e0b" },
    { label: "Late AM", start: 600,  end: 900,  color: "#38bdf8" },
    { label: "Noon",    start: 900,  end: 960,  color: "#10b981" },
    { label: "Afternoon", start: 960, end: 1080, color: "#38bdf8" },
    { label: "Peak",    start: 1080, end: 1200, color: "#f59e0b" },
    { label: "Evening", start: 1200, end: 1320, color: "#fb923c" },
    { label: "Night",   start: 1320, end: 1440, color: "#a78bfa" },
  ];
  const total = 1440;
  return (
    <div>
      <div style={{ display: "flex", height: 20, borderRadius: 99, overflow: "hidden", gap: 2 }}>
        {phases.map((ph, i) => {
          const w = ((ph.end - ph.start) / total) * 100;
          const active = currentMinute >= ph.start && currentMinute < ph.end;
          return (
            <div 
              key={i} 
              style={{ 
                width: `${w}%`, 
                background: ph.color,
                opacity: active ? 1 : 0.25,
                display: "flex", 
                alignItems: "center", 
                justifyContent: "center",
                fontSize: 9, 
                fontWeight: 600, 
                color: "#000",
                transition: "opacity 0.4s",
                overflow: "hidden", 
                whiteSpace: "nowrap",
              }} 
              title={`${ph.label}: ${ph.start / 60 | 0}:00-${ph.end / 60 | 0}:00`}
            >
              {w > 7 ? ph.label : ""}
            </div>
          );
        })}
      </div>
      <div style={{ 
        display: "flex", 
        justifyContent: "space-between", 
        marginTop: 6, 
        fontSize: 10, 
        color: "var(--text-muted)",
        fontFamily: "var(--font-mono)"
      }}>
        {[0, 4, 8, 12, 16, 20, 24].map(h => (
          <span key={h}>{String(h).padStart(2, "0")}:00</span>
        ))}
      </div>
    </div>
  );
}

// ─── Demand Heatmap Hourly ────────────────────────────────────────────────────
function DemandHeatRow({ queueTS, currentMin }) {
  if (!queueTS || queueTS.length < 60) return null;
  const hourlyPeak = Array.from({length: 24}, (_, h) => {
    const slice = queueTS.slice(h * 60, (h + 1) * 60);
    return slice.length ? Math.max(...slice) : 0;
  });
  const maxV = Math.max(...hourlyPeak, 1);
  const curH = Math.floor(currentMin / 60) % 24;
  return (
    <div>
      <div style={{ display: "flex", gap: 3, height: 32 }}>
        {hourlyPeak.map((v, h) => {
          const pct = v / maxV;
          const active = h === curH;
          const col = pct > 0.75 ? "#f43f5e" : pct > 0.45 ? "#f59e0b" : pct > 0.2 ? "#38bdf8" : "#3f3f46";
          return (
            <div 
              key={h} 
              style={{ 
                flex: 1, 
                display: "flex", 
                flexDirection: "column",
                alignItems: "center", 
                justifyContent: "flex-end", 
                gap: 2 
              }}
              title={`${String(h).padStart(2, "0")}:00 — max queue: ${v}`}
            >
              <div style={{ 
                width: "100%", 
                background: col, 
                opacity: active ? 1 : 0.5,
                height: `${Math.max(pct * 26, 3)}px`, 
                borderRadius: "2px 2px 0 0",
                boxShadow: active ? `0 0 8px ${col}` : "none",
                transition: "all 0.3s" 
              }} />
            </div>
          );
        })}
      </div>
      <div style={{ 
        display: "flex", 
        justifyContent: "space-between", 
        marginTop: 4, 
        fontSize: 10, 
        color: "var(--text-muted)",
        fontFamily: "var(--font-mono)"
      }}>
        <span>00</span><span>06</span><span>12</span><span>18</span><span>24</span>
      </div>
    </div>
  );
}

// ─── Fleet Health Gauges ──────────────────────────────────────────────────────
function FleetHealthGauges({ stateCounts, totalFleet, avgSocNow }) {
  if (!totalFleet) return null;
  const charging = stateCounts[4] || 0;
  const withCust = stateCounts[1] || 0;
  const stranded = stateCounts[5] || 0;
  const waiting  = stateCounts[3] || 0;
  const utilisation = ((withCust + charging) / totalFleet) * 100;
  const efficiency  = stranded === 0 ? 100 : Math.max(0, 100 - (stranded / totalFleet) * 100);

  const GaugePill = ({ label, value, max, color, unit = "%", warning }) => {
    const pct = Math.min((value / max) * 100, 100);
    return (
      <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
        <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12 }}>
          <span style={{ color: "var(--text-secondary)" }}>{label}</span>
          <span style={{ 
            color: warning && value > 0 ? "var(--rose)" : color, 
            fontWeight: 600,
            fontFamily: "var(--font-mono)"
          }}>
            {typeof value === "number" ? value.toFixed(1) : value}{unit}
          </span>
        </div>
        <div style={{ height: 5, background: "var(--surface3)", borderRadius: 99, overflow: "hidden" }}>
          <div style={{ 
            height: "100%", 
            width: `${pct}%`,
            background: warning && value > 0 ? "var(--rose)" : color,
            borderRadius: 99, 
            transition: "width 0.4s" 
          }} />
        </div>
      </div>
    );
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
      <GaugePill label="Utilisation" value={utilisation} max={100} color="var(--emerald)" />
      <GaugePill label="Fleet Efficiency" value={efficiency} max={100} color="var(--sky)" />
      <GaugePill label="Avg SoC" value={avgSocNow != null ? avgSocNow * 100 : 0} max={100} color="var(--accent)" />
      <GaugePill label="Stranded" value={stranded} max={Math.max(totalFleet * 0.05, 1)}
        color="var(--rose)" unit=" taxis" warning />
      <GaugePill label="Waiting Charger" value={waiting} max={Math.max(totalFleet * 0.1, 1)}
        color="var(--amber)" unit=" taxis" />
    </div>
  );
}

// ─── Algorithm Comparison Card ────────────────────────────────────────────────
function AlgoInfoCard({ algorithm }) {
  const info = algorithm === "ai"
    ? {
        name: "PPO AI Agent",
        color: "var(--emerald)",
        bg: "var(--emerald-dim)",
        details: [
          { k: "Architecture", v: "MLP 256x256" },
          { k: "Training",     v: "3,000,000 steps" },
          { k: "Observations", v: "48 features" },
          { k: "Actions",      v: "18 (16 stations + idle + rebal)" },
          { k: "Gamma",        v: "0.995" },
          { k: "Framework",    v: "Stable-Baselines3" },
        ],
      }
    : {
        name: "Greedy Heuristic",
        color: "var(--rose)",
        bg: "var(--rose-dim)",
        details: [
          { k: "Type",         v: "Rule-based" },
          { k: "Training",     v: "None" },
          { k: "Charging",     v: "SoC <= 25% -> nearest station" },
          { k: "Rebalancing",  v: "If distance > 5km from center" },
          { k: "Station Score", v: "dist + queue x 2.0" },
          { k: "Complexity",   v: "O(n * stations)" },
        ],
      };
  return (
    <div style={{ 
      padding: "14px 16px", 
      borderRadius: "var(--radius)",
      background: info.bg, 
      border: `1px solid ${info.color}30`,
      fontSize: 12 
    }}>
      <div style={{ 
        display: "flex", 
        alignItems: "center", 
        gap: 8, 
        marginBottom: 12 
      }}>
        <div style={{
          width: 8,
          height: 8,
          borderRadius: "50%",
          background: info.color,
          boxShadow: `0 0 8px ${info.color}`
        }} />
        <span style={{ 
          fontWeight: 600, 
          color: info.color, 
          fontSize: 14 
        }}>
          {info.name}
        </span>
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
        {info.details.map(d => (
          <div key={d.k} style={{ display: "flex", justifyContent: "space-between", gap: 12 }}>
            <span style={{ color: "var(--text-muted)" }}>{d.k}</span>
            <span style={{ color: "var(--text)", fontWeight: 500, textAlign: "right" }}>{d.v}</span>
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
  const panBase  = useRef({x: 0, y: 0});

  // ── Responsive ───────────────────────────────────────────────────────────
  useEffect(() => {
    const calc = () => {
      const available = Math.min(
        window.innerWidth  - 320 - 340 - 48,
        window.innerHeight - 56  - 160 - 48,
      );
      setCanvasSize(Math.max(300, Math.min(available, 720)));
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
    panStart.current = { x: e.clientX, y: e.clientY };
    panBase.current  = { x: panX, y: panY };
  }, [panX, panY]);
  const onMM = useCallback((e) => {
    if (!isPanning || !panStart.current) return;
    setPanX(panBase.current.x - (e.clientX - panStart.current.x));
    setPanY(panBase.current.y - (e.clientY - panStart.current.y));
  }, [isPanning]);
  const onMU = useCallback(() => { setIsPanning(false); panStart.current = null; }, []);
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
    setError("Simulation stopped by user.");
  }, []);

  // ── Run ───────────────────────────────────────────────────────────────────
  const run = useCallback(async () => {
    const controller = new AbortController();
    abortRef.current = controller;

    setLoading(true); setError(null); setPlaying(false);
    setFrameIdx(0); setFrames([]); setStats(null);
    setQueueTS([]); setSocTS([]);
    startProgress();
    const usedSeed = seed === "" ? randSeed() : Number(seed);
    try {
      const res = await fetch(`${API_URL}/simulate`, {
        method: "POST", 
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({algorithm, profile, seed: usedSeed}),
        signal: controller.signal,
      });
      if (!res.ok) { const e = await res.json(); throw new Error(e.detail || `Error ${res.status}`); }
      const data = await res.json();
      stopProgress();
      await new Promise(r => setTimeout(r, 420));
      setFrames(data.frames || []);
      setStations(data.stations || []);
      setStats(data.stats || null);
      setQueueTS(data.queues_over_time || []);
      setSocTS(data.avg_soc_over_time || []);
      setFrameIdx(0);
    } catch(e) {
      stopProgress();
      if (e.name === "AbortError") {
        setError("Simulation stopped by user.");
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
    return curFrame.taxis.reduce((a, ev) => { a[ev.s] = (a[ev.s] || 0) + 1; return a; }, {});
  }, [curFrame]);

  const currentQueues = useMemo(() => {
    if (!curFrame || !curFrame.queues) return {};
    return curFrame.queues.reduce((a, q, i) => { a[i] = q; return a; }, {});
  }, [curFrame]);

  const totalFleet  = frames[0]?.taxis?.length || 0;
  const avgSocNow   = useMemo(() => {
    if (!curFrame) return null;
    const v = curFrame.taxis.map(t => t.soc || 0);
    return v.length ? v.reduce((a, b) => a + b, 0) / v.length : null;
  }, [curFrame]);
  const totalQNow = curFrame?.queues?.reduce((a, b) => a + b, 0) ?? null;

  // ─────────────────────────────────────────────────────────────────────────
  return (
    <div style={{ 
      width: "100vw", 
      height: "100vh", 
      display: "flex", 
      flexDirection: "column",
      background: "var(--bg)", 
      overflow: "hidden" 
    }}>

      {/* ══ HEADER ══════════════════════════════════════════════════════════ */}
      <header style={{ 
        height: 56, 
        display: "flex", 
        alignItems: "center", 
        padding: "0 20px",
        borderBottom: "1px solid var(--border)", 
        background: "var(--bg-elevated)",
        flexShrink: 0, 
        gap: 16, 
        zIndex: 10 
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <div style={{ 
            width: 36, 
            height: 36, 
            borderRadius: "var(--radius)",
            background: "linear-gradient(135deg, var(--accent), var(--emerald))",
            display: "flex", 
            alignItems: "center", 
            justifyContent: "center",
            color: "#000",
            flexShrink: 0 
          }}>
            {Icons.zap}
          </div>
          <div>
            <div style={{ 
              fontSize: 16, 
              fontWeight: 700, 
              letterSpacing: "-0.02em",
              color: "var(--text)"
            }}>
              EV Fleet Simulator
            </div>
            <div style={{ 
              fontSize: 11, 
              color: "var(--text-muted)",
              letterSpacing: "0.01em"
            }}>
              Electric Vehicle Scheduling Visualization
            </div>
          </div>
        </div>
        <div style={{ flex: 1 }} />
        <div className="tag" style={{ 
          background: algorithm === "ai" ? "var(--emerald-dim)" : "var(--rose-dim)",
          color: algorithm === "ai" ? "var(--emerald)" : "var(--rose)"
        }}>
          {algorithm === "ai" ? "PPO AI Agent" : "Greedy Baseline"}
        </div>
        <div className="tag" style={{ 
          background: "var(--surface2)", 
          color: "var(--text-secondary)",
          border: "1px solid var(--border)"
        }}>
          {PROFILE_NAMES[profile]}
        </div>
        {frames.length > 0 && (
          <div style={{ 
            fontSize: 12, 
            color: "var(--text-secondary)", 
            fontFamily: "var(--font-mono)" 
          }}>
            Frame <span style={{ color: "var(--accent)", fontWeight: 600 }}>{frameIdx + 1}</span>/{frames.length}
          </div>
        )}
      </header>

      {/* ══ BODY ════════════════════════════════════════════════════════════ */}
      <div style={{ flex: 1, display: "flex", overflow: "hidden" }}>

        {/* ── LEFT PANEL ─────────────────────────────────────────────────── */}
        <aside style={{ 
          width: 320, 
          flexShrink: 0, 
          background: "var(--bg-elevated)",
          borderRight: "1px solid var(--border)", 
          padding: "20px",
          display: "flex", 
          flexDirection: "column", 
          gap: 20, 
          overflowY: "auto" 
        }}>

          <section>
            <SectionHeader accent="var(--rose)">Algorithm</SectionHeader>
            <div style={{ display: "flex" }}>
              <button 
                className={`algo-btn${algorithm === "baseline" ? " active-greedy" : ""}`}
                onClick={() => setAlgorithm("baseline")}
              >
                Greedy
              </button>
              <button 
                className={`algo-btn${algorithm === "ai" ? " active-ai" : ""}`}
                onClick={() => setAlgorithm("ai")}
              >
                PPO AI
              </button>
            </div>
            <div style={{ 
              marginTop: 10, 
              fontSize: 12, 
              color: "var(--text-secondary)", 
              lineHeight: 1.6,
              padding: "10px 14px", 
              background: "var(--surface2)", 
              borderRadius: "var(--radius-sm)",
              borderLeft: `3px solid ${algorithm === "ai" ? "var(--emerald)" : "var(--rose)"}` 
            }}>
              {algorithm === "baseline"
                ? "Heuristic algorithm: selects nearest station based on distance + queue. Simple, no training."
                : "PPO agent: trained for 3M steps. Makes decisions from 48 environment features."}
            </div>
          </section>

          <section>
            <SectionHeader accent="var(--sky)">Demand Profile</SectionHeader>
            <select value={profile} onChange={e => setProfile(Number(e.target.value))}>
              {PROFILE_NAMES.map((n, i) => (
                <option key={i} value={i}>{i}. {n}</option>
              ))}
            </select>
            <div style={{ 
              marginTop: 10, 
              fontSize: 12, 
              color: "var(--text-secondary)", 
              lineHeight: 1.6,
              padding: "10px 14px", 
              background: "var(--surface2)", 
              borderRadius: "var(--radius-sm)",
              borderLeft: "3px solid var(--sky)" 
            }}>
              {PROFILE_DESCRIPTIONS[profile]}
            </div>
          </section>

          <section>
            <SectionHeader accent="var(--violet)">Seed</SectionHeader>
            <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
              <input 
                type="number" 
                value={seed} 
                placeholder="e.g. 42 (auto if empty)"
                onChange={e => setSeed(e.target.value)} 
                style={{ flex: 1 }} 
              />
              <button 
                className="ctrl-btn" 
                onClick={() => setSeed(String(randSeed()))}
                style={{ padding: "10px 12px" }}
              >
                Random
              </button>
            </div>
            <div style={{ 
              fontSize: 11, 
              color: "var(--text-muted)", 
              marginTop: 6 
            }}>
              Empty = new random seed each time.
            </div>
          </section>

          <div style={{ display: "flex", gap: 8 }}>
            <button className="btn btn-primary" onClick={run} disabled={loading} style={{ flex: 1 }}>
              {loading ? (
                <>
                  <div style={{ 
                    width: 14, 
                    height: 14, 
                    border: "2px solid #000",
                    borderTopColor: "transparent", 
                    borderRadius: "50%",
                    animation: "spin 0.8s linear infinite" 
                  }} />
                  Running...
                </>
              ) : (
                <>
                  {Icons.play}
                  Run Simulation
                </>
              )}
            </button>
            {loading && (
              <button
                onClick={stopSimulation}
                className="btn"
                style={{
                  padding: "12px 16px",
                  background: "var(--rose-dim)",
                  color: "var(--rose)",
                  border: "1px solid rgba(244,63,94,0.3)",
                }}
              >
                {Icons.stop}
                Stop
              </button>
            )}
          </div>

          {error && (
            <div style={{ 
              background: "var(--rose-dim)", 
              border: "1px solid rgba(244,63,94,0.3)",
              borderRadius: "var(--radius-sm)", 
              padding: "12px 14px", 
              fontSize: 12, 
              color: "#fda4af", 
              lineHeight: 1.6 
            }}>
              {error}
            </div>
          )}

          {stats && (
            <section className="fade-in">
              <SectionHeader accent="var(--emerald)">Day Results</SectionHeader>
              <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                <div style={{ display: "flex", gap: 8 }}>
                  <StatCard label="Profit" value={fmtEur(stats.net_profit)} unit="EUR" color="var(--emerald)" />
                  <StatCard label="Service Rate" value={stats.service_rate} unit="%" color="var(--sky)" />
                </div>
                <div style={{ display: "flex", gap: 8 }}>
                  <StatCard label="Avg Wait" value={stats.avg_wait_time} unit="min" color="var(--amber)" />
                  <StatCard label="Stranded" value={stats.stranded_taxis} color="var(--rose)" />
                </div>
                <div style={{ display: "flex", gap: 8 }}>
                  <StatCard label="Served" value={stats.customers_served.toLocaleString()} color="var(--accent)" small />
                  <StatCard label="Abandoned" value={stats.customers_abandoned.toLocaleString()} color="var(--rose)" small />
                </div>

                {/* Derived KPIs */}
                <div style={{ 
                  marginTop: 4, 
                  padding: "12px 14px", 
                  background: "var(--surface2)",
                  borderRadius: "var(--radius)", 
                  border: "1px solid var(--border)",
                  display: "flex", 
                  flexDirection: "column", 
                  gap: 8 
                }}>
                  <div style={{ 
                    fontSize: 10, 
                    color: "var(--text-muted)", 
                    textTransform: "uppercase",
                    letterSpacing: "0.08em", 
                    marginBottom: 2 
                  }}>
                    Derived KPIs
                  </div>
                  {[
                    {
                      label: "Profit / taxi",
                      value: `${fmtEur(stats.net_profit / 750)} EUR`,
                      color: stats.net_profit > 0 ? "var(--emerald)" : "var(--rose)",
                    },
                    {
                      label: "Abandon rate",
                      value: `${(stats.customers_abandoned / Math.max(stats.customers_served + stats.customers_abandoned, 1) * 100).toFixed(1)} %`,
                      color: "var(--rose)",
                    },
                    {
                      label: "Avg revenue / served",
                      value: `${(stats.net_profit / Math.max(stats.customers_served, 1) + 40).toFixed(2)} EUR`,
                      color: "var(--amber)",
                    },
                  ].map(kpi => (
                    <div key={kpi.label} style={{ 
                      display: "flex", 
                      justifyContent: "space-between",
                      alignItems: "center", 
                      gap: 8 
                    }}>
                      <span style={{ fontSize: 12, color: "var(--text-secondary)" }}>{kpi.label}</span>
                      <span style={{ fontSize: 13, fontWeight: 600, color: kpi.color }}>{kpi.value}</span>
                    </div>
                  ))}
                </div>
              </div>
            </section>
          )}

          <div style={{ flex: 1 }} />

          {/* ── Keyboard shortcuts ── */}
          <section>
            <SectionHeader>Keyboard Shortcuts</SectionHeader>
            <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
              {[
                { k: "Space", v: "Play / Pause" },
                { k: "Arrow Keys", v: "Prev / Next frame" },
                { k: "Home", v: "First frame" },
                { k: "End", v: "Last frame" },
              ].map(row => (
                <div key={row.k} style={{ 
                  display: "flex", 
                  justifyContent: "space-between",
                  alignItems: "center", 
                  gap: 8 
                }}>
                  <kbd>{row.k}</kbd>
                  <span style={{ color: "var(--text-muted)", flex: 1, textAlign: "right", fontSize: 12 }}>{row.v}</span>
                </div>
              ))}
            </div>
          </section>

          <section>
            <SectionHeader>State Legend</SectionHeader>
            <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
              {Object.entries(STATE_META).map(([k, v]) => (
                <div key={k} style={{ display: "flex", alignItems: "center", gap: 10, fontSize: 12 }}>
                  <div style={{ 
                    width: 10, 
                    height: 10, 
                    borderRadius: "50%", 
                    background: v.color, 
                    flexShrink: 0,
                    boxShadow: v.glow ? `0 0 6px ${v.color}` : "none" 
                  }} />
                  <span style={{ color: "var(--text-secondary)", flex: 1 }}>{v.label}</span>
                  {stateCounts[k] != null && (
                    <span style={{ color: v.color, fontWeight: 600, fontFamily: "var(--font-mono)" }}>
                      {stateCounts[k]}
                    </span>
                  )}
                </div>
              ))}
            </div>
          </section>
        </aside>

        {/* ── CENTER ─────────────────────────────────────────────────────── */}
        <div style={{ 
          flex: 1, 
          display: "flex", 
          flexDirection: "column",
          padding: "16px", 
          gap: 12, 
          overflowY: "auto", 
          overflowX: "hidden",
          minWidth: 0 
        }}>

          {/* Playback bar */}
          <div className="panel" style={{ padding: "12px 16px", flexShrink: 0 }}>
            <div style={{ display: "flex", alignItems: "center", gap: 12, flexWrap: "wrap", rowGap: 10 }}>
              {frames.length > 0
                ? <SimClock minute={curMin} />
                : <div style={{ 
                    fontFamily: "var(--font-mono)", 
                    fontSize: 28, 
                    fontWeight: 700, 
                    color: "var(--text-muted)", 
                    minWidth: 90 
                  }}>--:--</div>
              }

              <input 
                type="range" 
                min={0} 
                max={Math.max(frames.length - 1, 0)} 
                value={frameIdx}
                onChange={e => { setPlaying(false); setFrameIdx(Number(e.target.value)); }}
                disabled={!frames.length} 
              />

              <div style={{ display: "flex", gap: 4, alignItems: "center" }}>
                <button className="ctrl-btn" disabled={!frames.length}
                  onClick={() => { setFrameIdx(0); setPlaying(false); }}>
                  {Icons.skipBack}
                </button>
                <button className="ctrl-btn" disabled={!frames.length}
                  onClick={() => setFrameIdx(i => Math.max(i - 1, 0))}>
                  {Icons.chevronLeft}
                </button>
                <button 
                  className={`ctrl-btn${playing ? " active" : ""}`} 
                  disabled={!frames.length}
                  onClick={() => setPlaying(p => !p)} 
                  style={{ minWidth: 44, padding: "6px 14px" }}
                >
                  {playing ? Icons.pause : Icons.play}
                </button>
                <button className="ctrl-btn" disabled={!frames.length}
                  onClick={() => setFrameIdx(i => Math.min(i + 1, frames.length - 1))}>
                  {Icons.chevronRight}
                </button>
                <button className="ctrl-btn" disabled={!frames.length}
                  onClick={() => { setFrameIdx(frames.length - 1); setPlaying(false); }}>
                  {Icons.skipForward}
                </button>
              </div>

              <div style={{ display: "flex", gap: 4, alignItems: "center" }}>
                <span style={{ fontSize: 11, color: "var(--text-muted)", marginRight: 4 }}>Speed</span>
                {SPEED_OPTIONS.map(s => (
                  <button 
                    key={s.ms} 
                    className={`speed-btn${speedMs === s.ms ? " active" : ""}`}
                    onClick={() => setSpeedMs(s.ms)}
                  >
                    {s.label}
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* Map */}
          <div style={{ flexShrink: 0, display: "flex", flexDirection: "column", gap: 8, minHeight: 700 }}>

            {/* Map toolbar */}
            <div style={{ display: "flex", gap: 8, alignItems: "center", flexShrink: 0 }}>
              <span style={{ fontSize: 11, color: "var(--text-muted)", fontWeight: 500 }}>Map View</span>
              <button 
                className={`ctrl-btn${showHeatmap ? " active" : ""}`}
                onClick={() => setShowHeatmap(h => !h)}
              >
                {Icons.layers}
                <span style={{ marginLeft: 4 }}>Heatmap</span>
              </button>
              <button className="ctrl-btn" onClick={resetView}>
                {Icons.target}
                <span style={{ marginLeft: 4 }}>Reset</span>
              </button>
              <button className="ctrl-btn" onClick={() => setZoom(z => Math.min(z + 0.3, 6))}>
                {Icons.plus}
              </button>
              <button className="ctrl-btn" onClick={() => setZoom(z => Math.max(z - 0.3, 0.5))}>
                {Icons.minus}
              </button>
              <span style={{ 
                fontSize: 12, 
                color: "var(--accent)", 
                fontWeight: 600, 
                fontFamily: "var(--font-mono)" 
              }}>
                {zoom.toFixed(1)}x
              </span>
              <div style={{ flex: 1 }} />
              <span style={{ fontSize: 10, color: "var(--text-muted)" }}>Scroll = zoom | Drag = pan</span>
            </div>

            {/* Canvas wrapper */}
            <div style={{ 
              flex: 1,
              minHeight: 600,
              position: "relative", 
              borderRadius: "var(--radius-lg)", 
              overflow: "hidden",
              border: "1px solid var(--border)", 
              background: "var(--bg)",
              cursor: isPanning ? "grabbing" : "grab",
              display: "flex", 
              alignItems: "center", 
              justifyContent: "center" 
            }}>
              {/* Decorative corner elements */}
              <div style={{ position: "absolute", inset: 0, pointerEvents: "none", overflow: "hidden" }}>
                {/* Top-left taxi icon */}
                <div style={{ 
                  position: "absolute", 
                  top: 16, 
                  left: 16, 
                  opacity: 0.04,
                  transform: "rotate(-15deg)"
                }}>
                  <svg width="80" height="80" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1">
                    <path d="M5 11h14v7H5z" fill="currentColor" stroke="none"/>
                    <rect x="3" y="11" width="18" height="7" rx="2"/>
                    <path d="M6 11V7a1 1 0 0 1 1-1h10a1 1 0 0 1 1 1v4"/>
                    <circle cx="7" cy="18" r="1.5" fill="currentColor"/>
                    <circle cx="17" cy="18" r="1.5" fill="currentColor"/>
                    <path d="M9 6V4a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2"/>
                    <rect x="9" y="2" width="6" height="2" rx="0.5" fill="var(--amber)" opacity="0.6"/>
                  </svg>
                </div>
                {/* Top-right lightning bolt */}
                <div style={{ 
                  position: "absolute", 
                  top: 20, 
                  right: 20, 
                  opacity: 0.04,
                  transform: "rotate(10deg)"
                }}>
                  <svg width="70" height="70" viewBox="0 0 24 24" fill="currentColor">
                    <path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z"/>
                  </svg>
                </div>
                {/* Bottom-left charging station */}
                <div style={{ 
                  position: "absolute", 
                  bottom: 50, 
                  left: 20, 
                  opacity: 0.04,
                  transform: "rotate(-5deg)"
                }}>
                  <svg width="60" height="60" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                    <rect x="4" y="4" width="10" height="16" rx="2"/>
                    <path d="M14 9h2a2 2 0 0 1 2 2v4a2 2 0 0 0 2 2h0"/>
                    <path d="M20 17v2"/>
                    <path d="M7 12h4"/>
                    <path d="M9 10v4"/>
                  </svg>
                </div>
                {/* Bottom-right battery */}
                <div style={{ 
                  position: "absolute", 
                  bottom: 50, 
                  right: 100, 
                  opacity: 0.04,
                  transform: "rotate(8deg)"
                }}>
                  <svg width="65" height="65" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                    <rect x="2" y="7" width="18" height="10" rx="2"/>
                    <path d="M22 11v2"/>
                    <path d="M6 10v4"/>
                    <path d="M10 10v4"/>
                    <path d="M14 10v4"/>
                  </svg>
                </div>
                {/* Center-right car silhouette */}
                <div style={{ 
                  position: "absolute", 
                  top: "50%", 
                  right: 14, 
                  opacity: 0.03,
                  transform: "translateY(-50%) rotate(90deg)"
                }}>
                  <svg width="50" height="50" viewBox="0 0 24 24" fill="currentColor">
                    <path d="M5 17a2 2 0 1 0 4 0 2 2 0 0 0-4 0zm10 0a2 2 0 1 0 4 0 2 2 0 0 0-4 0z"/>
                    <path d="M3 12h18v5a1 1 0 0 1-1 1h-1a3 3 0 0 1-6 0H9a3 3 0 0 1-6 0H4a1 1 0 0 1-1-1v-5z"/>
                    <path d="M5 12V9a1 1 0 0 1 1-1h2l2-3h4l2 3h2a1 1 0 0 1 1 1v3"/>
                  </svg>
                </div>
                {/* Center-left speedometer */}
                <div style={{ 
                  position: "absolute", 
                  top: "40%", 
                  left: 14, 
                  opacity: 0.03
                }}>
                  <svg width="45" height="45" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                    <circle cx="12" cy="12" r="10"/>
                    <path d="M12 6v2"/>
                    <path d="M12 16v2"/>
                    <path d="M6 12h2"/>
                    <path d="M16 12h2"/>
                    <path d="M12 12l3-3"/>
                  </svg>
                </div>
              </div>
              <canvas 
                ref={canvasRef} 
                width={canvasSize} 
                height={canvasSize}
                style={{ display: "block", maxWidth: "100%", maxHeight: "100%" }}
                onMouseDown={onMD} 
                onMouseMove={onMM}
                onMouseUp={onMU} 
                onMouseLeave={onMU} 
              />

              {!frames.length && !loading && (
                <div style={{ 
                  position: "absolute", 
                  inset: 0, 
                  display: "flex", 
                  flexDirection: "column",
                  alignItems: "center", 
                  justifyContent: "center", 
                  gap: 12,
                  color: "var(--text-muted)", 
                  pointerEvents: "none" 
                }}>
                  <div style={{ 
                    width: 64, 
                    height: 64, 
                    borderRadius: "var(--radius)", 
                    background: "var(--surface2)",
                    display: "flex", 
                    alignItems: "center", 
                    justifyContent: "center",
                    color: "var(--text-muted)"
                  }}>
                    {Icons.zap}
                  </div>
                  <div style={{ fontSize: 15, fontWeight: 600 }}>No simulation data</div>
                  <div style={{ fontSize: 12 }}>Click "Run Simulation" to start</div>
                </div>
              )}
              {loading && <LoadingOverlay progress={loadPct} />}

              {frames.length > 0 && !loading && (
                <>
                  <div style={{ 
                    position: "absolute", 
                    bottom: 12, 
                    right: 14, 
                    fontSize: 11,
                    color: "var(--text-muted)", 
                    background: "rgba(10,10,10,0.8)",
                    backdropFilter: "blur(4px)",
                    padding: "6px 10px", 
                    borderRadius: "var(--radius-sm)", 
                    pointerEvents: "none",
                    fontFamily: "var(--font-mono)"
                  }}>
                    20km x 20km | {totalFleet} taxis | {stations.length} stations
                  </div>
                  <div style={{ 
                    position: "absolute", 
                    top: 12, 
                    left: 14, 
                    fontSize: 11,
                    color: "var(--text-muted)", 
                    background: "rgba(10,10,10,0.8)",
                    backdropFilter: "blur(4px)",
                    padding: "6px 10px", 
                    borderRadius: "var(--radius-sm)", 
                    pointerEvents: "none",
                    display: "flex", 
                    alignItems: "center", 
                    gap: 8 
                  }}>
                    <span style={{ 
                      color: algorithm === "ai" ? "var(--emerald)" : "var(--rose)", 
                      fontWeight: 600 
                    }}>
                      {algorithm === "ai" ? "PPO AI" : "Greedy"}
                    </span>
                    <span style={{ color: "var(--border)" }}>|</span>
                    <span>{PROFILE_NAMES[profile]}</span>
                    <span style={{ color: "var(--border)" }}>|</span>
                    <span style={{ color: "var(--accent)", fontFamily: "var(--font-mono)" }}>
                      {fmtMin(curMin)}
                    </span>
                  </div>
                </>
              )}
            </div>
          </div>

          {/* ── Bottom center: Day phase + per-hour demand ── */}
          {(frames.length > 0 || queueTS.length > 0) && (
            <div className="panel" style={{ 
              padding: "14px 18px", 
              flexShrink: 0, 
              display: "flex",
              flexDirection: "column", 
              gap: 12 
            }}>
              <div>
                <div style={{ 
                  fontSize: 10, 
                  color: "var(--text-muted)", 
                  textTransform: "uppercase",
                  letterSpacing: "0.1em", 
                  marginBottom: 8, 
                  fontWeight: 600 
                }}>
                  Day Phases
                </div>
                <DayPhaseBar currentMinute={curMin} />
              </div>
              {queueTS.length > 0 && (
                <div>
                  <div style={{ 
                    fontSize: 10, 
                    color: "var(--text-muted)", 
                    textTransform: "uppercase",
                    letterSpacing: "0.1em", 
                    marginBottom: 8, 
                    fontWeight: 600, 
                    display: "flex",
                    alignItems: "center", 
                    gap: 12 
                  }}>
                    Hourly Queue Peaks
                    <span style={{ display: "flex", alignItems: "center", gap: 4 }}>
                      <span style={{ width: 6, height: 6, borderRadius: 2, background: "var(--rose)" }} />
                      <span style={{ fontSize: 9, textTransform: "none", letterSpacing: 0, fontWeight: 400 }}>peak &gt;75%</span>
                    </span>
                    <span style={{ display: "flex", alignItems: "center", gap: 4 }}>
                      <span style={{ width: 6, height: 6, borderRadius: 2, background: "var(--amber)" }} />
                      <span style={{ fontSize: 9, textTransform: "none", letterSpacing: 0, fontWeight: 400 }}>medium</span>
                    </span>
                    <span style={{ display: "flex", alignItems: "center", gap: 4 }}>
                      <span style={{ width: 6, height: 6, borderRadius: 2, background: "var(--sky)" }} />
                      <span style={{ fontSize: 9, textTransform: "none", letterSpacing: 0, fontWeight: 400 }}>low</span>
                    </span>
                  </div>
                  <DemandHeatRow queueTS={queueTS} currentMin={curMin} />
                </div>
              )}
            </div>
          )}
        </div>

        {/* ── RIGHT PANEL ────────────────────────────────────────────────── */}
        <aside style={{ 
          width: 340, 
          flexShrink: 0, 
          background: "var(--bg-elevated)",
          borderLeft: "1px solid var(--border)", 
          padding: "20px",
          display: "flex", 
          flexDirection: "column", 
          gap: 18, 
          overflowY: "auto" 
        }}>

          <section>
            <SectionHeader accent="var(--accent)">Live Metrics</SectionHeader>
            <div style={{ display: "flex", gap: 8 }}>
              <div className="stat-card">
                <div style={{ 
                  fontSize: 10, 
                  color: "var(--text-muted)", 
                  textTransform: "uppercase",
                  letterSpacing: "0.08em", 
                  marginBottom: 6 
                }}>
                  Avg SoC
                </div>
                {avgSocNow != null ? (
                  <>
                    <div style={{ 
                      fontSize: 22, 
                      fontWeight: 700, 
                      fontFamily: "var(--font-sans)",
                      color: avgSocNow < 0.3 ? "var(--rose)" : avgSocNow < 0.5 ? "var(--amber)" : "var(--emerald)",
                      letterSpacing: "-0.02em"
                    }}>
                      {(avgSocNow * 100).toFixed(1)}
                      <span style={{ fontSize: 12, fontWeight: 500, color: "var(--text-muted)", marginLeft: 2 }}>%</span>
                    </div>
                    <div style={{ height: 4, background: "var(--surface3)", borderRadius: 99, marginTop: 8, overflow: "hidden" }}>
                      <div style={{ 
                        height: "100%", 
                        width: `${avgSocNow * 100}%`, 
                        borderRadius: 99, 
                        transition: "width 0.3s",
                        background: avgSocNow < 0.3 ? "var(--rose)" : avgSocNow < 0.5 ? "var(--amber)" : "var(--emerald)" 
                      }} />
                    </div>
                  </>
                ) : (
                  <div style={{ color: "var(--text-muted)", fontSize: 14 }}>--</div>
                )}
              </div>
              <div className="stat-card">
                <div style={{ 
                  fontSize: 10, 
                  color: "var(--text-muted)", 
                  textTransform: "uppercase",
                  letterSpacing: "0.08em", 
                  marginBottom: 6 
                }}>
                  Station Queue
                </div>
                <div style={{ 
                  fontSize: 22, 
                  fontWeight: 700, 
                  fontFamily: "var(--font-sans)",
                  color: (totalQNow ?? 0) > 30 ? "var(--rose)" : "var(--text)",
                  letterSpacing: "-0.02em"
                }}>
                  {totalQNow != null ? totalQNow : <span style={{ color: "var(--text-muted)" }}>--</span>}
                  <span style={{ fontSize: 12, fontWeight: 500, color: "var(--text-muted)", marginLeft: 4 }}>taxis</span>
                </div>
              </div>
            </div>
          </section>

          {/* Fleet Health Gauges */}
          {totalFleet > 0 && (
            <section>
              <SectionHeader accent="var(--emerald)">Fleet Health</SectionHeader>
              <div className="panel" style={{ padding: "14px 16px" }}>
                <FleetHealthGauges stateCounts={stateCounts} totalFleet={totalFleet} avgSocNow={avgSocNow} />
              </div>
            </section>
          )}

          {totalFleet > 0 && (
            <section>
              <SectionHeader accent="var(--sky)">Fleet Distribution</SectionHeader>
              <StateDistBar counts={stateCounts} total={totalFleet} />
            </section>
          )}

          {stations.length > 0 && curFrame && (
            <section>
              <SectionHeader accent="var(--violet)">Station Queues</SectionHeader>
              <StationList stations={stations} queues={currentQueues} />
            </section>
          )}

          {queueTS.length > 0 && (
            <section>
              <SectionHeader accent="var(--rose)">Queue Over Time</SectionHeader>
              <div className="panel" style={{ padding: "14px 16px" }}>
                <Sparkline data={queueTS} color="#f43f5e" height={52} width={280} cursorIdx={tsIdx} />
                <div style={{ 
                  display: "flex", 
                  justifyContent: "space-between", 
                  marginTop: 6, 
                  fontSize: 11, 
                  color: "var(--text-muted)",
                  fontFamily: "var(--font-mono)"
                }}>
                  <span>00:00</span>
                  <span style={{ color: "var(--rose)", fontWeight: 600 }}>
                    {queueTS[tsIdx] != null ? `now: ${queueTS[tsIdx]}` : "--"}
                  </span>
                  <span>23:45</span>
                </div>
              </div>
            </section>
          )}

          {socTS.length > 0 && (
            <section>
              <SectionHeader accent="var(--emerald)">Avg SoC Over Time</SectionHeader>
              <div className="panel" style={{ padding: "14px 16px" }}>
                <Sparkline data={socTS.map(v => v * 100)} color="#10b981" height={52} width={280} cursorIdx={tsIdx} />
                <div style={{ 
                  display: "flex", 
                  justifyContent: "space-between", 
                  marginTop: 6, 
                  fontSize: 11, 
                  color: "var(--text-muted)",
                  fontFamily: "var(--font-mono)"
                }}>
                  <span>00:00</span>
                  <span style={{ color: "var(--emerald)", fontWeight: 600 }}>
                    {socTS[tsIdx] != null ? `now: ${(socTS[tsIdx] * 100).toFixed(1)}%` : "--"}
                  </span>
                  <span>23:45</span>
                </div>
              </div>
            </section>
          )}

          {frames.length > 0 && (
            <section>
              <SectionHeader>Day Progress</SectionHeader>
              <DayPhaseBar currentMinute={curMin} />
              <div style={{ marginTop: 10, height: 5, background: "var(--surface3)", borderRadius: 99, overflow: "hidden" }}>
                <div style={{ 
                  height: "100%",
                  width: `${(frameIdx / Math.max(frames.length - 1, 1)) * 100}%`,
                  background: "linear-gradient(90deg, var(--accent), var(--emerald))",
                  borderRadius: 99, 
                  transition: "width 0.2s" 
                }} />
              </div>
              <div style={{ 
                display: "flex", 
                justifyContent: "space-between", 
                marginTop: 6,
                fontSize: 10, 
                color: "var(--text-muted)",
                fontFamily: "var(--font-mono)"
              }}>
                <span>00:00</span>
                <span style={{ color: "var(--accent)", fontWeight: 600 }}>{fmtMin(curMin)}</span>
                <span>23:45</span>
              </div>
            </section>
          )}

          <section>
            <SectionHeader>Algorithm Details</SectionHeader>
            <AlgoInfoCard algorithm={algorithm} />
          </section>

          <div style={{ flex: 1 }} />

          <div style={{ 
            padding: "14px 16px", 
            borderRadius: "var(--radius)", 
            background: "var(--surface2)",
            border: "1px solid var(--border)", 
            fontSize: 12, 
            color: "var(--text-secondary)",
            lineHeight: 1.8, 
            flexShrink: 0 
          }}>
            <div style={{ 
              color: "var(--accent)", 
              fontWeight: 600, 
              marginBottom: 8,
              fontSize: 13 
            }}>
              Map Controls
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "auto 1fr", gap: "4px 14px" }}>
              <span style={{ color: "var(--accent)", fontFamily: "var(--font-mono)", fontSize: 11 }}>Scroll</span>
              <span>Zoom In / Out</span>
              <span style={{ color: "var(--accent)", fontFamily: "var(--font-mono)", fontSize: 11 }}>Drag</span>
              <span>Pan the map</span>
              <span style={{ color: "var(--accent)", fontFamily: "var(--font-mono)", fontSize: 11 }}>Heatmap</span>
              <span>Taxi density</span>
              <span style={{ color: "var(--accent)", fontFamily: "var(--font-mono)", fontSize: 11 }}>Reset</span>
              <span>Reset view</span>
            </div>
            <div style={{ marginTop: 12, display: "flex", gap: 14, flexWrap: "wrap", fontSize: 11 }}>
              <span><span style={{ color: "var(--amber)" }}>●</span> SUPER-HUB</span>
              <span><span style={{ color: "var(--violet)" }}>●</span> Station</span>
              <span><span style={{ color: "var(--rose)" }}>●</span> Stranded</span>
              <span><span style={{ color: "var(--sky)" }}>●</span> With customer</span>
            </div>
          </div>
        </aside>
      </div>

      {/* ══ STATUS BAR ══════════════════════════════════════════════════════ */}
      <footer style={{ 
        height: 32, 
        flexShrink: 0,
        background: "var(--bg-elevated)", 
        borderTop: "1px solid var(--border)",
        display: "flex", 
        alignItems: "center", 
        padding: "0 20px", 
        gap: 16,
        fontSize: 12, 
        color: "var(--text-muted)", 
        overflow: "hidden" 
      }}>
        <span style={{ 
          color: algorithm === "ai" ? "var(--emerald)" : "var(--rose)", 
          fontWeight: 600 
        }}>
          {algorithm === "ai" ? "PPO AI" : "Greedy"}
        </span>
        <span style={{ color: "var(--border)", fontSize: 10 }}>|</span>
        <span>Profile: <strong style={{ color: "var(--text-secondary)" }}>{PROFILE_NAMES[profile]}</strong></span>
        {seed !== "" && (
          <>
            <span style={{ color: "var(--border)", fontSize: 10 }}>|</span>
            <span>Seed: <strong style={{ color: "var(--text-secondary)", fontFamily: "var(--font-mono)" }}>{seed}</strong></span>
          </>
        )}
        <div style={{ flex: 1 }} />
        {curFrame && Object.entries(STATE_META).map(([k, v]) => {
          const n = stateCounts[k] || 0;
          if (!n) return null;
          return (
            <span key={k} style={{ display: "flex", alignItems: "center", gap: 4 }}>
              <span style={{ 
                width: 6, 
                height: 6, 
                borderRadius: "50%", 
                background: v.color,
                display: "inline-block",
                boxShadow: v.glow ? `0 0 4px ${v.color}` : "none" 
              }} />
              <span style={{ color: v.color, fontWeight: 600, fontFamily: "var(--font-mono)", fontSize: 11 }}>{n}</span>
            </span>
          );
        })}
        <div style={{ flex: 1 }} />
        {frames.length > 0 && (
          <>
            <span style={{ fontFamily: "var(--font-mono)" }}>
              <strong style={{ color: "var(--accent)" }}>{fmtMin(curMin)}</strong>
            </span>
            <span style={{ color: "var(--border)", fontSize: 10 }}>|</span>
            <span style={{ fontFamily: "var(--font-mono)" }}>
              Frame <strong style={{ color: "var(--text-secondary)" }}>{frameIdx + 1}/{frames.length}</strong>
            </span>
            <span style={{ color: "var(--border)", fontSize: 10 }}>|</span>
          </>
        )}
        <span style={{ fontFamily: "var(--font-mono)" }}>
          <strong style={{ color: "var(--text-secondary)" }}>{zoom.toFixed(1)}x</strong>
        </span>
      </footer>
    </div>
  );
}
