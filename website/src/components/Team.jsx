import { useState } from 'react';

const PhotoIcon = () => (
  <svg viewBox="0 0 24 24">
    <rect x="3" y="3" width="18" height="18" rx="2" />
    <circle cx="8.5" cy="8.5" r="1.5" />
    <path d="m21 15-5-5L5 21" />
  </svg>
);

const MEMBERS = [
  { name: 'Varshini Vijay', role: 'Computer Science', dept: 'UC Irvine' },
  { name: 'Nitya Chauhan', role: 'Mechanical Engineering', dept: 'UC Irvine' },
  { name: 'Kaushik Saravannan', role: 'Computer Engineering', dept: 'UC Irvine' },
  { name: 'Aidan Stone-Walsh', role: 'Electrical Engineering', dept: 'UC Irvine' },
  { name: 'Vincent Pham', role: 'Computer Engineering', dept: 'UC Irvine' },
  { name: 'Kevin Huynh', role: 'Mechanical Engineering', dept: 'UC Irvine' },
];

function TeamCard({ member }) {
  const [photo, setPhoto] = useState(null);

  const handlePhoto = (e) => {
    const file = e.target.files[0];
    if (file) setPhoto(URL.createObjectURL(file));
  };

  return (
    <div className="team-card">
      <div className="team-photo-slot">
        <input type="file" accept="image/*" onChange={handlePhoto} />
        {photo
          ? <img src={photo} alt={member.name} />
          : <>
              <div className="team-photo-icon"><PhotoIcon /></div>
              <div className="team-photo-label">Add Photo</div>
            </>
        }
      </div>
      <div className="team-name">{member.name}</div>
      <div className="team-role">{member.role}</div>
      <span className="team-dept">{member.dept}</span>
    </div>
  );
}

export default function Team() {
  return (
    <section className="team" id="team">
      <div className="section-eyebrow">The Team</div>
      <h2 className="section-title">meet the team</h2>
      <div className="divider" />
      <div className="team-grid">
        {MEMBERS.map(m => <TeamCard key={m.name} member={m} />)}
      </div>
    </section>
  );
}
