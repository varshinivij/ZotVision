import { useState, useEffect, useRef, useCallback, useMemo } from 'react';

const CHIPS = [
  { type: 'fire', label: '🔥 Fire' },
  { type: 'person', label: '🧑 Trapped Person' },
  { type: 'hazmat', label: '⚠️ Hazmat' },
  { type: 'exit', label: '🚪 Exit' },
  { type: 'waypoint', label: '📍 Waypoint' },
];

const ELEMENT_CONFIG = {
  fire:     { icon: '🔥', label: 'Fire' },
  person:   { icon: '🧑', label: 'Trapped' },
  hazmat:   { icon: '⚠️', label: 'Hazmat' },
  exit:     { icon: '🚪', label: 'Exit' },
  waypoint: { icon: '📍', label: 'Waypoint' },
};

function getHUDState(placed) {
  const hasFire = placed.fire?.length > 0;
  const hasPerson = placed.person?.length > 0;
  const hasHazmat = placed.hazmat?.length > 0;
  const hasExit = placed.exit?.length > 0;
  const hasWaypoint = placed.waypoint?.length > 0;
  const count = Object.values(placed).flat().length;

  if (count === 0) return { temp: '98°F', o2: '95%', hr: '72', timer: '08:42', arrow: '↑', arrowLabel: 'Patrol', alert: null, status: 'NOMINAL', statusColor: '#00e5cc', desc: 'No elements placed — drag something into the commander view to trigger a HUD response.' };
  if (hasHazmat) return { temp: '103°F', o2: '14%', hr: '128', timer: '22:10', arrow: '↗', arrowLabel: 'Evacuate', alert: { text: 'HAZMAT ZONE', color: '#e0b800' }, status: 'HAZMAT', statusColor: '#e0b800', desc: 'Hazmat detected. O₂ dangerously low. HUD screaming visual alert — silent, no alarm tone. Arrow routing FF out.' };
  if (hasFire && hasPerson) return { temp: '112°F', o2: '79%', hr: '147', timer: '31:50', arrow: '↑', arrowLabel: 'To victim', alert: { text: 'RESCUE ACTIVE', color: '#E8523A' }, status: 'RESCUE', statusColor: '#E8523A', desc: 'Commander sees fire and a trapped person. Override waypoint sent to FF — HUD routes toward victim with fire boundary shown.' };
  if (hasPerson) return { temp: '101°F', o2: '88%', hr: '98', timer: '14:22', arrow: '↑', arrowLabel: 'To victim', alert: { text: 'VICTIM NEARBY', color: '#4A8FD4' }, status: 'RESCUE', statusColor: '#4A8FD4', desc: 'Commander spots a trapped person and sends override waypoint. FF HUD arrow updates instantly to route toward victim.' };
  if (hasFire) return { temp: '118°F', o2: '71%', hr: '141', timer: '27:33', arrow: '↓', arrowLabel: 'Retrace path', alert: { text: 'HIGH HEAT', color: '#E8523A' }, status: 'HIGH HEAT', statusColor: '#E8523A', desc: 'Active fire in scene. Temp spiked past safe range. HUD triggers silent visual alert — arrows redirect FF away from blaze.' };
  if (hasWaypoint) return { temp: '99°F', o2: '93%', hr: '80', timer: '11:05', arrow: '↗', arrowLabel: 'New waypoint', alert: { text: 'NEW ROUTE', color: '#4A8FD4' }, status: 'OVERRIDE', statusColor: '#4A8FD4', desc: 'Commander set a waypoint override. FF HUD arrow updates immediately to new bearing — no radio chatter needed.' };
  if (hasExit) return { temp: '100°F', o2: '91%', hr: '76', timer: '09:18', arrow: '↙', arrowLabel: 'Exit marked', alert: null, status: 'EXIT MARKED', statusColor: '#4caf79', desc: 'Exit marked by commander. HUD shows safe exit direction. All vitals nominal — routine navigation assist.' };
  return { temp: '98°F', o2: '95%', hr: '72', timer: '08:42', arrow: '↑', arrowLabel: 'Continue', alert: null, status: 'NOMINAL', statusColor: '#00e5cc', desc: 'Monitoring.' };
}

function hexToRgb(hex) {
  const r = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
  return r ? `${parseInt(r[1], 16)},${parseInt(r[2], 16)},${parseInt(r[3], 16)}` : '0,229,204';
}

export default function Demo() {
  const canvasRef = useRef(null);
  const sceneRef = useRef(null);
  const [elements, setElements] = useState([]);
  const elementsRef = useRef([]);
  const dragTypeRef = useRef(null);
  const draggingRef = useRef(null);
  const idRef = useRef(0);

  useEffect(() => { elementsRef.current = elements; }, [elements]);

  const placed = useMemo(() => {
    const p = {};
    elements.forEach(el => { if (!p[el.type]) p[el.type] = []; p[el.type].push(el); });
    return p;
  }, [elements]);

  const hudState = useMemo(() => getHUDState(placed), [placed]);

  // Canvas animation loop
  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    let raf;

    function draw() {
      const els = elementsRef.current;
      const p = {};
      els.forEach(el => { if (!p[el.type]) p[el.type] = []; p[el.type].push(el); });
      const s = getHUDState(p);
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
        const warn = (lbl === 'O₂' && parseInt(val) < 20) || (lbl === 'HR' && parseInt(val) > 120) || (lbl === 'TEMP' && parseInt(val) > 110);
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
      ctx.save(); ctx.translate(cx, cy); ctx.rotate(-Math.PI / 6);
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

      // Alert
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

  const addElement = useCallback((type, x, y) => {
    const id = ++idRef.current;
    setElements(prev => [...prev, { id, type, x, y }]);
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    if (!dragTypeRef.current) return;
    const rect = sceneRef.current.getBoundingClientRect();
    addElement(dragTypeRef.current, e.clientX - rect.left, e.clientY - rect.top);
    dragTypeRef.current = null;
  }, [addElement]);

  const handleMouseMove = useCallback((e) => {
    if (!draggingRef.current) return;
    const rect = sceneRef.current.getBoundingClientRect();
    const nx = e.clientX - rect.left - draggingRef.current.offX;
    const ny = e.clientY - rect.top - draggingRef.current.offY;
    const id = draggingRef.current.id;
    setElements(prev => prev.map(el => el.id === id
      ? { ...el, x: Math.max(0, Math.min(rect.width, nx)), y: Math.max(0, Math.min(rect.height, ny)) }
      : el
    ));
  }, []);

  const handleChipTouchEnd = useCallback((e, type) => {
    const t = e.changedTouches[0];
    const rect = sceneRef.current?.getBoundingClientRect();
    if (rect && t.clientX >= rect.left && t.clientX <= rect.right && t.clientY >= rect.top && t.clientY <= rect.bottom) {
      addElement(type, t.clientX - rect.left, t.clientY - rect.top);
    }
    dragTypeRef.current = null;
  }, [addElement]);

  const statusBadgeStyle = {
    fontSize: '0.62rem', letterSpacing: '0.08em', textTransform: 'uppercase',
    padding: '2px 7px',
    background: hudState.statusColor + '14',
    border: `0.5px solid ${hudState.statusColor}4d`,
    color: hudState.statusColor,
  };

  return (
    <section id="demo" style={{ background: 'var(--bg2)', borderTop: '0.5px solid var(--border)', borderBottom: '0.5px solid var(--border)', padding: '7rem 3rem' }}>
      <div className="section-eyebrow">Interactive Demo</div>
      <h2 className="section-title">see both sides of the system</h2>
      <div className="divider" />

      <div style={{ marginTop: '2rem', display: 'flex', gap: '1rem', flexWrap: 'wrap', alignItems: 'center' }}>
        <span style={{ fontSize: '0.72rem', letterSpacing: '0.1em', textTransform: 'uppercase', color: 'var(--muted)' }}>Drag elements into the scene →</span>
        <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
          {CHIPS.map(chip => (
            <div
              key={chip.type}
              className="drag-chip"
              draggable
              onDragStart={() => { dragTypeRef.current = chip.type; }}
              onTouchStart={() => { dragTypeRef.current = chip.type; }}
              onTouchEnd={e => handleChipTouchEnd(e, chip.type)}
            >
              {chip.label}
            </div>
          ))}
        </div>
        <button className="clear-btn" onClick={() => setElements([])}>Clear Scene</button>
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

          <div
            ref={sceneRef}
            style={{ position: 'relative', width: '100%', height: '460px', cursor: 'crosshair', overflow: 'hidden' }}
            onDragOver={e => e.preventDefault()}
            onDrop={handleDrop}
            onMouseMove={handleMouseMove}
            onMouseUp={() => { draggingRef.current = null; }}
            onMouseLeave={() => { draggingRef.current = null; }}
          >
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

            {/* Firefighter blip */}
            <div style={{ position: 'absolute', left: 90, top: 200, transform: 'translate(-50%,-50%)', zIndex: 5, pointerEvents: 'none' }}>
              <div style={{ width: 26, height: 26, borderRadius: '50%', background: 'var(--accent)', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 10, fontWeight: 700, color: '#fff', boxShadow: '0 0 0 6px rgba(224,85,53,0.2)', animation: 'ping 2s ease-in-out infinite' }}>FF</div>
              <div style={{ position: 'absolute', top: '100%', left: '50%', transform: 'translateX(-50%)', marginTop: 2, fontSize: 8, letterSpacing: '0.08em', textTransform: 'uppercase', color: 'rgba(224,85,53,0.7)', whiteSpace: 'nowrap' }}>33.644, -117.840</div>
            </div>

            {/* Dropped elements */}
            {elements.map(el => {
              const cfg = ELEMENT_CONFIG[el.type];
              return (
                <div
                  key={el.id}
                  className="scene-el"
                  style={{ left: el.x, top: el.y }}
                  onMouseDown={e => {
                    if (e.button !== 0) return;
                    const rect = sceneRef.current.getBoundingClientRect();
                    draggingRef.current = { id: el.id, offX: e.clientX - rect.left - el.x, offY: e.clientY - rect.top - el.y };
                    e.stopPropagation();
                  }}
                  onContextMenu={e => { e.preventDefault(); setElements(prev => prev.filter(item => item.id !== el.id)); }}
                  onDoubleClick={() => setElements(prev => prev.filter(item => item.id !== el.id))}
                >
                  <div className="scene-el-icon">{cfg.icon}</div>
                  <div className="scene-el-label">{cfg.label}</div>
                </div>
              );
            })}
          </div>
        </div>

        {/* HUD View */}
        <div style={{ background: '#000', position: 'relative', overflow: 'hidden', minHeight: '520px', display: 'flex', flexDirection: 'column' }}>
          <div style={{ background: '#111', padding: '0.55rem 1rem', display: 'flex', alignItems: 'center', justifyContent: 'space-between', borderBottom: '0.5px solid rgba(0,229,204,0.15)' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <div style={{ width: 6, height: 6, borderRadius: '50%', background: '#00e5cc', animation: 'blink 1.5s ease-in-out infinite' }} />
              <span style={{ fontSize: '0.65rem', letterSpacing: '0.1em', textTransform: 'uppercase', color: '#00e5cc' }}>In-Mask HUD · FF1</span>
            </div>
            <span style={statusBadgeStyle}>{hudState.status}</span>
          </div>
          <canvas ref={canvasRef} width={480} height={460} style={{ width: '100%', flex: 1, display: 'block' }} />
        </div>
      </div>

      <div style={{ marginTop: '1px', background: 'var(--bg3)', border: '0.5px solid var(--border)', padding: '0.9rem 1.25rem', display: 'flex', alignItems: 'center', gap: '1.5rem', flexWrap: 'wrap' }}>
        <span style={{ fontSize: '0.65rem', letterSpacing: '0.1em', textTransform: 'uppercase', color: 'var(--muted)' }}>Scene</span>
        <span style={{ fontSize: '0.82rem', color: 'var(--text)' }}>{hudState.desc}</span>
      </div>
    </section>
  );
}
