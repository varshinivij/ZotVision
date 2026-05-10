const ITEMS = [
  { icon: '📷', title: 'Infrared Camera', text: 'Top-mounted; feeds live video to command.' },
  { icon: '🖥️', title: 'OLED HUD', text: 'In-mask display; path arrows + vitals.' },
  { icon: '📍', title: 'GPS Module', text: 'Logs breadcrumb trail for retracing.' },
  { icon: '⚙️', title: 'ESP32 MCU', text: 'Processes all sensor data onboard.' },
  { icon: '📡', title: 'Wi-Fi Transceiver', text: 'Streams data to commander dashboard.' },
  { icon: '🔋', title: '4-hr Battery', text: 'Vest-mounted; sealed cable to mask.' },
];

export default function Components() {
  return (
    <section id="components" style={{ background: 'var(--bg2)', borderTop: '0.5px solid var(--border)', borderBottom: '0.5px solid var(--border)' }}>
      <div className="section-eyebrow">Under the Hood</div>
      <h2 className="section-title">what's inside the system</h2>
      <div className="divider" />
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '1px', background: 'var(--border)', border: '0.5px solid var(--border)', marginTop: '4rem' }}>
        {ITEMS.map(item => (
          <div
            key={item.title}
            style={{ background: 'var(--bg2)', padding: '2rem 1.75rem', transition: 'background 0.2s' }}
            onMouseOver={e => e.currentTarget.style.background = 'var(--bg3)'}
            onMouseOut={e => e.currentTarget.style.background = 'var(--bg2)'}
          >
            <div style={{ fontSize: '1.5rem', marginBottom: '1rem' }}>{item.icon}</div>
            <div style={{ fontSize: '1rem', fontWeight: 500, letterSpacing: '-0.015em', marginBottom: '0.5rem' }}>{item.title}</div>
            <div style={{ fontSize: '0.85rem', color: 'var(--muted)', lineHeight: 1.65 }}>{item.text}</div>
          </div>
        ))}
      </div>
    </section>
  );
}
