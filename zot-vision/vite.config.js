import { defineConfig } from 'vite'
import react, { reactCompilerPreset } from '@vitejs/plugin-react'
import babel from '@rolldown/plugin-babel'
import http from 'http'

// https://vite.dev/config/
export default defineConfig({
  plugins: [
    react(),
    babel({ presets: [reactCompilerPreset()] })
  ],
  server: {
    proxy: {
      '/api': {
        target: 'http://localhost:5000',
        // Flask dev server sends Connection: close — disable keep-alive to
        // prevent Node's HTTP parser from seeing data on a "closed" socket.
        agent: new http.Agent({ keepAlive: false }),
        configure: (proxy) => {
          proxy.on('error', (err, _req, res) => {
            if (res && !res.headersSent) {
              res.writeHead(502, { 'Content-Type': 'application/json' })
              res.end(JSON.stringify({ error: 'backend unavailable' }))
            }
          })
        }
      }
    }
  },
})
