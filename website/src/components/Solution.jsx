import { useState } from 'react';

const SlideIcon = () => (
  <svg viewBox="0 0 24 24" style={{ width: 22, height: 22, stroke: 'var(--accent)', fill: 'none', strokeWidth: 1.5, strokeLinecap: 'round', strokeLinejoin: 'round' }}>
    <rect x="3" y="3" width="18" height="18" rx="2" />
    <circle cx="8.5" cy="8.5" r="1.5" />
    <path d="m21 15-5-5L5 21" />
  </svg>
);

const CARDS = [
  { emoji: '🔔', title: 'Visual-only alerts', text: 'In cases of poor vitals, a visual-only HUD popup ensures steady heart rates.' },
  { emoji: '🎯', title: 'Path retracing', text: 'Our GPS system leads navigation, showing the way they came in with arrows.' },
  { emoji: '👥', title: 'Human-in-the-loop', text: 'The commander supervises the routes from a simple laptop interface.' },
];

export default function Solution() {
  const [current, setCurrent] = useState(0);
  const [images] = useState([
    '/photos/sol-1.png',
    '/photos/sol-2.png',
    '/photos/sol-3.png',
    null,
  ]);

  const step = (dir) => setCurrent(c => (c + dir + 4) % 4);

  return (
    <section id="solution">
      <div className="section-eyebrow">Our Solution</div>
      <h2 className="section-title">a visual HUD that keeps hearts calm and paths clear</h2>
      <div className="divider" />

      <div className="solution-grid">
        {CARDS.map(card => (
          <div className="solution-card" key={card.title}>
            <span className="solution-emoji">{card.emoji}</span>
            <div className="solution-card-title">{card.title}</div>
            <div className="solution-card-text">{card.text}</div>
          </div>
        ))}
      </div>

      <div style={{ marginTop: '3rem' }}>
        <div style={{ position: 'relative', overflow: 'hidden', background: 'var(--bg3)', border: '0.5px solid var(--border)' }}>
          <div style={{ display: 'flex', transition: 'transform 0.4s cubic-bezier(0.4,0,0.2,1)', transform: `translateX(-${current * 100}%)` }}>
            {[0, 1, 2, 3].map(idx => (
              <div className="sol-slide" key={idx}>
                <div className="sol-slide-inner">
                  {images[idx]
                    ? <img src={images[idx]} alt={`SOL ${idx + 1}`} style={{ width: '100%', height: '100%', objectFit: 'cover', position: 'absolute', inset: 0 }} />
                    : <div className="sol-slide-placeholder">
                        <div className="sol-slide-icon"><SlideIcon /></div>
                        <div style={{ fontSize: '0.72rem', letterSpacing: '0.1em', textTransform: 'uppercase', color: 'var(--muted)' }}>Coming Soon</div>
                      </div>
                  }
                </div>
              </div>
            ))}
          </div>

          <button
            onClick={() => step(-1)}
            style={{ position: 'absolute', left: 12, top: '50%', transform: 'translateY(-50%)', background: 'rgba(10,10,10,0.7)', border: '0.5px solid var(--border)', color: 'var(--text)', width: 36, height: 36, cursor: 'pointer', fontSize: '1rem', display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 5 }}
            onMouseOver={e => e.currentTarget.style.background = 'rgba(224,85,53,0.2)'}
            onMouseOut={e => e.currentTarget.style.background = 'rgba(10,10,10,0.7)'}
          >‹</button>
          <button
            onClick={() => step(1)}
            style={{ position: 'absolute', right: 12, top: '50%', transform: 'translateY(-50%)', background: 'rgba(10,10,10,0.7)', border: '0.5px solid var(--border)', color: 'var(--text)', width: 36, height: 36, cursor: 'pointer', fontSize: '1rem', display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 5 }}
            onMouseOver={e => e.currentTarget.style.background = 'rgba(224,85,53,0.2)'}
            onMouseOut={e => e.currentTarget.style.background = 'rgba(10,10,10,0.7)'}
          >›</button>

          <div style={{ position: 'absolute', bottom: 12, left: '50%', transform: 'translateX(-50%)', display: 'flex', gap: 6, zIndex: 5 }}>
            {[0, 1, 2, 3].map(i => (
              <div key={i} className={`sol-dot${i === current ? ' active' : ''}`} onClick={() => setCurrent(i)} />
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}
