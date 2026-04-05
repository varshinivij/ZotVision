function FramePlayer({ id, url, live, coordinates, altitude, label }) {
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
            <button className="ctrl-btn ctrl-left" title="Left" />
            <button className="ctrl-btn ctrl-obstacle" title="Obstacle" />
            <button className="ctrl-btn ctrl-right" title="Right" />
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
