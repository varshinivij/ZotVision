import { useEffect, useRef } from 'react';
import { MapContainer, TileLayer, Marker, Popup, useMap } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';

// Adjusts map bounds whenever active positions change
function FitBounds({ positions }) {
  const map = useMap();
  const prevKey = useRef('');

  useEffect(() => {
    const key = JSON.stringify(positions);
    if (key === prevKey.current || positions.length === 0) return;
    prevKey.current = key;

    if (positions.length === 1) {
      map.setView(positions[0], 17);
    } else {
      map.fitBounds(L.latLngBounds(positions), { padding: [50, 50] });
    }
  }, [positions, map]);

  return null;
}

function makeMarker(displayId) {
  return L.divIcon({
    className: '',
    html: `<div class="ff-pin">FF${displayId}</div>`,
    iconSize: [40, 40],
    iconAnchor: [20, 20],
    popupAnchor: [0, -26],
  });
}

function MapView({ firefighters }) {
  const active = firefighters.filter(
    ff => ff.live && (ff.lat !== 0 || ff.lon !== 0)
  );
  const positions = active.map(ff => [ff.lat, ff.lon]);
  const defaultCenter = [33.6405, -117.8443]; // UCI campus fallback

  return (
    <div className="map-card">
      <div className="map-header">
        <div className="map-header-left">
          <div className="map-dot" />
          <span className="map-title">LIVE POSITIONS</span>
        </div>
        <span className="map-badge">
          {active.length} / {firefighters.length} active
        </span>
      </div>

      <MapContainer
        center={defaultCenter}
        zoom={16}
        className="map-container"
        zoomControl
      >
        <TileLayer
          url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
          attribution='&copy; <a href="https://openstreetmap.org">OpenStreetMap</a> &copy; <a href="https://carto.com">CARTO</a>'
          maxZoom={19}
        />

        {active.map(ff => (
          <Marker
            key={ff.id}
            position={[ff.lat, ff.lon]}
            icon={makeMarker(ff.id + 1)}
          >
            <Popup>
              <div className="map-popup">
                <div className="map-popup-title">FF{ff.id + 1}</div>
                <div>{ff.lat.toFixed(6)}, {ff.lon.toFixed(6)}</div>
                <div>Alt: {ff.alt.toFixed(1)} m</div>
              </div>
            </Popup>
          </Marker>
        ))}

        <FitBounds positions={positions} />
      </MapContainer>

      {active.length === 0 && (
        <div className="map-empty">No active GPS signals</div>
      )}
    </div>
  );
}

export default MapView;
