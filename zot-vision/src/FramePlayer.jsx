import { useState } from 'react';

function FramePlayer({ id, url, live, coordinates, altitude, label }) {
  const [activeCmd, setActiveCmd] = useState(null);
  const [msgInput, setMsgInput] = useState('');

  const sendCommand = async (cmd) => {
    // Pressing the active button again clears the command
    const next = activeCmd === cmd ? 'none' : cmd;
    setActiveCmd(next === 'none' ? null : cmd);
    try {
      await fetch(`/api/control/${id - 1}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ command: next }),
      });
    } catch (err) {
      console.error('Control POST failed:', err);
    }
  };

  const sendMessage = async () => {
    const text = msgInput.trim();
    try {
      await fetch(`/api/message/${id - 1}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text }),
      });
      setMsgInput('');
    } catch (err) {
      console.error('Message POST failed:', err);
    }
  };

  const handleMsgKey = (e) => {
    if (e.key === 'Enter') sendMessage();
  };

  return (
    <div className={`camera-feed ${live ? 'is-live' : 'is-offline'}`}>
      <div className="camera-label">
        <span className={`camera-dot ${live ? 'dot-live' : 'dot-offline'}`} />
        <span>FF{id} CAMERA - {live ? 'LIVE' : 'OFFLINE'}</span>
      </div>

      {live ? (
        <div className="camera-body">
          <img src={url} alt={`Camera ${id}`} className="camera-image" />
          <div className="camera-controls">
            <text>{`(X, Y): ${coordinates.lat}, ${coordinates.lng}`}</text>
            <text>{`Z: ${altitude}`}</text>
            {label && <text className="hazard-label">{`Hazard: ${label}`}</text>}
            <button
              className={`ctrl-btn ctrl-left${activeCmd === 'left' ? ' ctrl-active' : ''}`}
              title="Navigate Left"
              onClick={() => sendCommand('left')}
            />
            <button
              className={`ctrl-btn ctrl-obstacle${activeCmd === 'warning' ? ' ctrl-active' : ''}`}
              title="Obstacle Warning"
              onClick={() => sendCommand('warning')}
            />
            <button
              className={`ctrl-btn ctrl-right${activeCmd === 'right' ? ' ctrl-active' : ''}`}
              title="Navigate Right"
              onClick={() => sendCommand('right')}
            />
          </div>
          <div className="msg-row">
            <input
              className="msg-input"
              type="text"
              maxLength={20}
              placeholder="Message (20 chars)..."
              value={msgInput}
              onChange={(e) => setMsgInput(e.target.value)}
              onKeyDown={handleMsgKey}
            />
            <button className="msg-send" onClick={sendMessage}>Send</button>
          </div>
        </div>
      ) : (
        <div className="camera-offline">
          <div className="offline-icon" />
          <p>FF{id} - Waiting for feed...</p>
        </div>
      )}
    </div>
  );
}

export default FramePlayer;
