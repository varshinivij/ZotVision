import FramePlayer from './FramePlayer.jsx'
import './App.css'

const images = import.meta.glob('./assets/Firefighter*.png', { eager: true })

function App() {
  const numFireFighters = 4;

  const array = [];
  for (let i = 1; i <= numFireFighters; i++) {
    const key = `./assets/Firefighter${i}.png`;
    array.push(images[key]?.default || null);
  }

  return (
    <div className="dashboard">
      <div className="dashboard-header">
        <div className="dashboard-badge">ZOT VISION</div>
        <h1 className="dashboard-title">Live Camera Feeds</h1>
      </div>
      <div className="feeds-grid">
        {array.map((IMAGE_PATH, idx) => (
          <FramePlayer key={idx} id={idx + 1} url={IMAGE_PATH} live={idx === 0} />
        ))}
      </div>
    </div>
  );
}

export default App;
