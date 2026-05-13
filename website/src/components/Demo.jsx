import { useState, useEffect, useRef } from 'react';

const FF = { x: 90, y: 200 };

const SCENARIOS = [
  {
    label: 'Room 201',
    firePos: { x: 110, y: 80 },
    temp: '112°F', o2: '79%', hr: '141', timer: '18:33',
    arrow: '↗', arrowLabel: 'Reroute',
    alert: { text: 'HIGH HEAT — ROOM 201', color: '#E8523A' },
    status: 'HIGH HEAT', statusColor: '#E8523A',
    desc: 'Fire detected in Room 201. FF is south in the hall — compass points NNE toward blaze. HUD reroutes FF away from Room 201.',
  },
  {
    label: 'Room 202',
    firePos: { x: 290, y: 80 },
    temp: '108°F', o2: '83%', hr: '128', timer: '14:10',
    arrow: '↑', arrowLabel: 'Advance north',
    alert: { text: 'HIGH HEAT — ROOM 202', color: '#E8523A' },
    status: 'HIGH HEAT', statusColor: '#E8523A',
    desc: 'Fire in Room 202 — northeast of FF position. Compass bearing NE. FF routed north through hall to avoid Room 202 directly.',
  },
  {
    label: 'Hall',
    firePos: { x: 70, y: 230 },
    temp: '121°F', o2: '61%', hr: '158', timer: '31:50',
    arrow: '↑', arrowLabel: 'Retreat now',
    alert: { text: 'EVACUATE — FIRE IN HALL', color: '#E8523A' },
    status: 'EVACUATE', statusColor: '#E8523A',
    desc: 'CRITICAL: Fire has entered the Hall — FF position compromised. O₂ critically low. HUD screams retreat north immediately.',
  },
  {
    label: 'Room 201 (spread)',
    firePos: { x: 110, y: 80 },
    temp: '116°F', o2: '72%', hr: '149', timer: '26:04',
    arrow: '↙', arrowLabel: 'Fall back',
    alert: { text: 'SPREAD — ROOM 201', color: '#e0b800' },
    status: 'SPREADING', statusColor: '#e0b800',
    desc: 'Fire has re-intensified and spread in Room 201. Conditions deteriorating — vitals worsening. FF ordered to fall back to stairwell.',
  },
  {
    label: 'Stairwell',
    firePos: { x: 190, y: 380 },
    temp: '107°F', o2: '76%', hr: '136', timer: '22:47',
    arrow: '↗', arrowLabel: 'Find alt exit',
    alert: { text: 'EXIT BLOCKED — STAIRWELL', color: '#e0b800' },
    status: 'EXIT BLOCKED', statusColor: '#e0b800',
    desc: 'Fire in the Stairwell — primary exit cut off. Compass points south toward blaze. HUD routes FF northeast to find alternate exit.',
  },
];

function getBearing(from, to) {
  return Math.atan2(to.x - from.x, -(to.y - from.y));
}

function hexToRgb(hex) {
  const r = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
  return r ? `${parseInt(r[1], 16)},${parseInt(r[2], 16)},${parseInt(r[3], 16)}` : '0,229,204';
}

export default function Demo() {
  const canvasRef = useRef(null);
  const hudRef = useRef(null);
  const [idx, setIdx] = useState(0);
  const scenario = SCENARIOS[idx];

  // Keep a ref so the canvas loop always reads the latest without restarting
  hudRef.current = { ...scenario, bearing: getBearing(FF, scenario.firePos) };

  // Auto-advance every 4 seconds
  useEffect(() => {
    const t = setInterval(() => setIdx(i => (i + 1) % SCENARIOS.length), 4000);
    return () => clearInterval(t);
  }, []);

  // Canvas animation loop — runs once, reads hudRef each frame
  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    let raf;

    function draw() {
      const s = hudRef.current;
      const W = canvas.width, H = canvas.height;
      const hudColor = s.alert ? s.alert.color : '#00e5cc';

      ctx.clearRect(0, 0, W, H);
      ctx.fillStyle = '#020508';
      ctx.fillRect(0, 0, W, H);
      for (let y = 0; y < H; y += 3) { ctx.fillStyle = 'rgba(0,255,120,0.01)'; ctx.fillRect(0, y, W, 1); }

      const vg = ctx.createRadialGradient(W / 2, H / 2, H * 0.25, W / 2, H / 2, H * 0.8);
      vg.addColorStop(0, 'transparent');
      vg.addColorStop(1, 'rgba(0,0,0,0.55)');
      ctx.fillStyle = vg;
      ctx.fillRect(0, 0, W, H);

      // Vitals
      const vitals = [['TEMP', s.temp, 60, 88], ['O₂', s.o2, 125, 152], ['HR', s.hr, 190, 217]];
      vitals.forEach(([lbl, val, ly, vy]) => {
        ctx.fillStyle = 'rgba(255,255,255,0.22)';
        ctx.font = '500 9px Helvetica Neue,sans-serif';
        ctx.textAlign = 'left';
        ctx.fillText(lbl, 24, ly);
        const warn = (lbl === 'O₂' && parseInt(val) < 80) || (lbl === 'HR' && parseInt(val) > 120) || (lbl === 'TEMP' && parseInt(val) > 110);
        ctx.fillStyle = warn ? '#ff4444' : hudColor;
        ctx.font = '700 22px Helvetica Neue,sans-serif';
        ctx.fillText(val, 24, vy);
      });

      // Compass
      const cx = W / 2, cy = H / 2 - 6, r = 50;
      ctx.strokeStyle = `rgba(${hexToRgb(hudColor)},0.15)`;
      ctx.lineWidth = 0.5;
      ctx.beginPath(); ctx.arc(cx, cy, r, 0, Math.PI * 2); ctx.stroke();
      ['N', 'E', 'S', 'W'].forEach((d, i) => {
        const a = (i * Math.PI / 2) - Math.PI / 2;
        ctx.fillStyle = d === 'N' ? hudColor : `rgba(${hexToRgb(hudColor)},0.3)`;
        ctx.font = '700 9px Helvetica Neue'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
        ctx.fillText(d, cx + Math.cos(a) * (r + 11), cy + Math.sin(a) * (r + 11));
      });
      // Needle points toward fire
      ctx.save(); ctx.translate(cx, cy); ctx.rotate(s.bearing);
      ctx.fillStyle = hudColor;
      ctx.beginPath(); ctx.moveTo(0, -r * 0.58); ctx.lineTo(3.5, 0); ctx.lineTo(0, r * 0.3); ctx.lineTo(-3.5, 0); ctx.closePath(); ctx.fill();
      ctx.restore();
      ctx.fillStyle = '#fff'; ctx.beginPath(); ctx.arc(cx, cy, 3, 0, Math.PI * 2); ctx.fill();

      // Timer
      ctx.fillStyle = 'rgba(255,255,255,0.22)'; ctx.font = '500 9px Helvetica Neue'; ctx.textAlign = 'center'; ctx.textBaseline = 'alphabetic';
      ctx.fillText('TIME', cx, cy + r + 18);
      ctx.fillStyle = hudColor; ctx.font = '500 16px Helvetica Neue';
      ctx.fillText(s.timer, cx, cy + r + 36);

      // Nav arrow
      const ax = W - 80, ay = H / 2 - 8;
      ctx.fillStyle = hudColor; ctx.font = '700 52px Helvetica Neue'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillText(s.arrow, ax, ay);
      ctx.fillStyle = 'rgba(255,255,255,0.3)'; ctx.font = '500 8px Helvetica Neue'; ctx.textBaseline = 'alphabetic';
      ctx.fillText(s.arrowLabel.toUpperCase(), ax, ay + 36);

      // Horizon line
      ctx.strokeStyle = `rgba(${hexToRgb(hudColor)},0.1)`; ctx.lineWidth = 0.5; ctx.setLineDash([4, 8]);
      ctx.beginPath(); ctx.moveTo(0, H * 0.62); ctx.lineTo(W, H * 0.62); ctx.stroke();
      ctx.setLineDash([]);

      // Alert border + banner
      if (s.alert) {
        const t = (Math.sin(Date.now() / 380) + 1) / 2;
        ctx.globalAlpha = 0.1 + t * 0.25;
        ctx.strokeStyle = s.alert.color; ctx.lineWidth = 3;
        ctx.strokeRect(2, 2, W - 4, H - 4); ctx.globalAlpha = 1;
        ctx.font = '700 10px Helvetica Neue'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
        const tw = ctx.measureText('⚠ ' + s.alert.text).width + 20;
        ctx.globalAlpha = 0.9; ctx.fillStyle = 'rgba(0,0,0,0.8)'; ctx.fillRect(W / 2 - tw / 2, 10, tw, 22);
        ctx.globalAlpha = 1; ctx.fillStyle = s.alert.color;
        ctx.fillText('⚠ ' + s.alert.text, W / 2, 21);
      }
      ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic';
      raf = requestAnimationFrame(draw);
    }

    draw();
    return () => cancelAnimationFrame(raf);
  }, []);

  const statusBadgeStyle = {
    fontSize: '0.62rem', letterSpacing: '0.08em', textTransform: 'uppercase',
    padding: '2px 7px',
    background: scenario.statusColor + '14',
    border: `0.5px solid ${scenario.statusColor}4d`,
    color: scenario.statusColor,
  };

  return (
    <section id="demo" style={{ background: 'var(--bg2)', borderTop: '0.5px solid var(--border)', borderBottom: '0.5px solid var(--border)', padding: '7rem 3rem' }}>
      <div className="section-eyebrow">Interactive Demo</div>
      <h2 className="section-title">see both sides of the system</h2>
      <div className="divider" />

      {/* Scenario selector */}
      <div style={{ marginTop: '2rem', display: 'flex', gap: '1rem', alignItems: 'center' }}>
        <span style={{ fontSize: '0.72rem', letterSpacing: '0.1em', textTransform: 'uppercase', color: 'var(--muted)' }}>Fire location</span>
        <button
          className="drag-chip"
          onClick={() => setIdx(i => (i + 1) % SCENARIOS.length)}
          style={{ borderColor: scenario.statusColor, color: scenario.statusColor }}
        >
          🔥 {scenario.label}
        </button>
        <span style={{ fontSize: '0.72rem', color: 'var(--muted)', letterSpacing: '0.05em' }}>{idx + 1} / {SCENARIOS.length}</span>
      </div>

      <div style={{ marginTop: '1.5rem', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1px', background: 'var(--border)', border: '0.5px solid var(--border)' }}>

        {/* Commander View */}
        <div style={{ background: '#0d0d0d', position: 'relative', overflow: 'hidden', minHeight: '520px' }}>
          <div style={{ background: '#1a1a1a', padding: '0.55rem 1rem', display: 'flex', alignItems: 'center', justifyContent: 'space-between', borderBottom: '0.5px solid var(--border)' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <div style={{ width: 6, height: 6, borderRadius: '50%', background: '#4caf79', animation: 'blink 1.5s ease-in-out infinite' }} />
              <span style={{ fontSize: '0.65rem', letterSpacing: '0.1em', textTransform: 'uppercase', color: '#4caf79' }}>Commander View · Live</span>
            </div>
            <span style={{ fontSize: '0.65rem', letterSpacing: '0.08em', textTransform: 'uppercase', color: 'var(--muted2)' }}>FF1 · Floor 2</span>
          </div>

          <div style={{ position: 'relative', width: '100%', height: '460px', overflow: 'hidden' }}>
            <svg width="100%" height="100%" style={{ position: 'absolute', inset: 0 }} xmlns="http://www.w3.org/2000/svg">
              <defs>
                <pattern id="fp-grid" width="40" height="40" patternUnits="userSpaceOnUse">
                  <path d="M40 0L0 0 0 40" fill="none" stroke="rgba(255,255,255,0.04)" strokeWidth="0.5" />
                </pattern>
              </defs>
              <rect width="100%" height="100%" fill="url(#fp-grid)" />
              <rect x="20" y="20" width="180" height="120" fill="none" stroke="rgba(255,255,255,0.08)" strokeWidth="1" />
              <rect x="220" y="20" width="140" height="120" fill="none" stroke="rgba(255,255,255,0.08)" strokeWidth="1" />
              <rect x="20" y="160" width="100" height="140" fill="none" stroke="rgba(255,255,255,0.08)" strokeWidth="1" />
              <rect x="140" y="160" width="220" height="140" fill="none" stroke="rgba(255,255,255,0.08)" strokeWidth="1" />
              <rect x="20" y="320" width="340" height="120" fill="none" stroke="rgba(255,255,255,0.08)" strokeWidth="1" />
              <text x="110" y="90" textAnchor="middle" fill="rgba(255,255,255,0.12)" fontSize="10" fontFamily="Helvetica Neue,sans-serif" letterSpacing="1">ROOM 201</text>
              <text x="290" y="90" textAnchor="middle" fill="rgba(255,255,255,0.12)" fontSize="10" fontFamily="Helvetica Neue,sans-serif" letterSpacing="1">ROOM 202</text>
              <text x="70" y="240" textAnchor="middle" fill="rgba(255,255,255,0.12)" fontSize="10" fontFamily="Helvetica Neue,sans-serif" letterSpacing="1">HALL</text>
              <text x="250" y="240" textAnchor="middle" fill="rgba(255,255,255,0.12)" fontSize="10" fontFamily="Helvetica Neue,sans-serif" letterSpacing="1">ROOM 203</text>
              <text x="190" y="385" textAnchor="middle" fill="rgba(255,255,255,0.12)" fontSize="10" fontFamily="Helvetica Neue,sans-serif" letterSpacing="1">STAIRWELL</text>
              <line x1="80" y1="140" x2="110" y2="140" stroke="rgba(255,255,255,0.25)" strokeWidth="1.5" />
              <line x1="250" y1="140" x2="280" y2="140" stroke="rgba(255,255,255,0.25)" strokeWidth="1.5" />
              <line x1="120" y1="230" x2="120" y2="260" stroke="rgba(255,255,255,0.25)" strokeWidth="1.5" />
              <line x1="200" y1="300" x2="230" y2="300" stroke="rgba(255,255,255,0.25)" strokeWidth="1.5" />
            </svg>

            {/* Fire emoji */}
            <div style={{
              position: 'absolute',
              left: scenario.firePos.x,
              top: scenario.firePos.y,
              transform: 'translate(-50%, -50%)',
              fontSize: 26,
              zIndex: 10,
              filter: 'drop-shadow(0 0 10px #ff6b00)',
              transition: 'left 0.4s ease, top 0.4s ease',
              pointerEvents: 'none',
            }}>
              🔥
            </div>

            {/* Firefighter blip */}
            <div style={{ position: 'absolute', left: FF.x, top: FF.y, transform: 'translate(-50%,-50%)', zIndex: 5, pointerEvents: 'none' }}>
              <div style={{ width: 26, height: 26, borderRadius: '50%', background: 'var(--accent)', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 10, fontWeight: 700, color: '#fff', boxShadow: '0 0 0 6px rgba(224,85,53,0.2)', animation: 'ping 2s ease-in-out infinite' }}>FF</div>
              <div style={{ position: 'absolute', top: '100%', left: '50%', transform: 'translateX(-50%)', marginTop: 2, fontSize: 8, letterSpacing: '0.08em', textTransform: 'uppercase', color: 'rgba(224,85,53,0.7)', whiteSpace: 'nowrap' }}>33.644, -117.840</div>
            </div>
          </div>
        </div>

        {/* HUD View */}
        <div style={{ background: '#000', position: 'relative', overflow: 'hidden', minHeight: '520px', display: 'flex', flexDirection: 'column' }}>
          <div style={{ background: '#111', padding: '0.55rem 1rem', display: 'flex', alignItems: 'center', justifyContent: 'space-between', borderBottom: '0.5px solid rgba(0,229,204,0.15)' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <div style={{ width: 6, height: 6, borderRadius: '50%', background: '#00e5cc', animation: 'blink 1.5s ease-in-out infinite' }} />
              <span style={{ fontSize: '0.65rem', letterSpacing: '0.1em', textTransform: 'uppercase', color: '#00e5cc' }}>In-Mask HUD · FF1</span>
            </div>
            <span style={statusBadgeStyle}>{scenario.status}</span>
          </div>
          <canvas ref={canvasRef} width={480} height={460} style={{ width: '100%', flex: 1, display: 'block' }} />
        </div>
      </div>

      <div style={{ marginTop: '1px', background: 'var(--bg3)', border: '0.5px solid var(--border)', padding: '0.9rem 1.25rem', display: 'flex', alignItems: 'center', gap: '1.5rem', flexWrap: 'wrap' }}>
        <span style={{ fontSize: '0.65rem', letterSpacing: '0.1em', textTransform: 'uppercase', color: 'var(--muted)' }}>Scene</span>
        <span style={{ fontSize: '0.82rem', color: 'var(--text)' }}>{scenario.desc}</span>
      </div>
    </section>
  );
}
