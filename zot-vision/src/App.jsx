import { useState, useEffect } from 'react'
import FramePlayer from './FramePlayer.jsx'
import './App.css'

function App() {
  const [firefighters, setFirefighters] = useState(
    Array.from({ length: 5 }, (_, i) => ({
      id: i,
      live: false,
      image_url: null,
      lat: 0,
      lon: 0,
      alt: 0,
      label: null,
    }))
  );

  useEffect(() => {
    const interval = setInterval(async () => {
      try {
        const res = await fetch('/api/state');
        const data = await res.json();
        setFirefighters(data.firefighters);
      } catch {
        // backend not available yet
      }
    }, 1000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="dashboard">
      <div className="dashboard-header">
        <div className="dashboard-badge">ZOT VISION</div>
        <h1 className="dashboard-title">Live Camera Feeds</h1>
      </div>
      <div className="feeds-grid">
        {firefighters.map((ff) => (
          <FramePlayer
            key={ff.id}
            id={ff.id + 1}
            url={ff.image_url}
            live={ff.live}
            coordinates={{ lat: ff.lat, lng: ff.lon }}
            altitude={ff.alt}
            label={ff.label}
          />
        ))}
      </div>
    </div>
  );
}

export default App;
