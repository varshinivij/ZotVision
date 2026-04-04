import FramePlayer from './FramePlayer.jsx'
import './App.css'

function App() {
  const numFireFighters = 2;

  const array = [];
  for (let i = 1; i <= numFireFighters; i++) {
    array.push(`./assets/Firefighter${i}`);
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
