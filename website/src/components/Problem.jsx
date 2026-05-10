import { useState } from 'react';

const PhotoIcon = () => (
  <svg viewBox="0 0 24 24">
    <rect x="3" y="3" width="18" height="18" rx="2" />
    <circle cx="8.5" cy="8.5" r="1.5" />
    <path d="m21 15-5-5L5 21" />
  </svg>
);

export default function Problem() {
  const [photo, setPhoto] = useState(null);

  const handlePhoto = (e) => {
    const file = e.target.files[0];
    if (file) setPhoto(URL.createObjectURL(file));
  };

  return (
    <section className="problem" id="problem">
      <div className="section-eyebrow">The Problem</div>
      <h2 className="section-title">disorientation impacts firefighters</h2>

      <div className="problem-quote-row">
        <div className="problem-quote-wrap">
          <blockquote className="problem-quote">
            "The only thing we hate more than change is how things are. Until you've been in the environment, it's hard to describe where you are."
          </blockquote>
          <div className="problem-quote-attr">— David Forrester, Chief Captain, Orange County Fire Authority</div>
        </div>

        <div className="problem-photo-slot">
          <input type="file" accept="image/*" onChange={handlePhoto} />
          {photo
            ? <img src={photo} alt="Firefighter photo" />
            : <>
                <div className="placeholder-icon"><PhotoIcon /></div>
                <div className="placeholder-label">Add Photo</div>
                <div className="placeholder-sub">Firefighter in action</div>
              </>
          }
        </div>
      </div>

      <div className="problem-stats">
        <div className="problem-stat">
          <div className="problem-stat-num">53,575</div>
          <div className="problem-stat-text">firefighter injuries in 2024 — lack of visibility disorients firefighters, leading to 30% of deaths on the fireground.</div>
        </div>
        <div className="problem-stat">
          <div className="problem-stat-num">~45%</div>
          <div className="problem-stat-text">of firefighter deaths on duty stem from sudden cardiac events. Audible alarm tones spike heart rates further.</div>
        </div>
        <div className="problem-stat">
          <div className="problem-stat-num">Zero</div>
          <div className="problem-stat-text">affordable, universal, HUD-based navigation tools exist today for firefighters.</div>
        </div>
      </div>
    </section>
  );
}
