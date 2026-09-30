# phd-helper shell

The React shell (SPEC §1/§3): section tree, transcript with raw find/replace
diff cards, composer, and the corpus panel, over one WebSocket per tab. The
turn-taking logic is a pure reducer (`src/protocol/reducer.ts`) — the only
place shell state changes; everything else is wiring.

## Commands

Run from this directory (`web/app`):

```bash
npm install     # once
npm run dev     # Vite dev server on :5173, proxies API/WS to the backend
npm test        # vitest, once (test:watch for the loop)
npx tsc -b      # typecheck
npm run build   # emits dist/, which the FastAPI app serves at /
```

The backend is expected at `http://localhost:8000` (see repo root for how to
start it). Point the dev proxy elsewhere with `PHD_BACKEND`:

```bash
PHD_BACKEND=http://10.0.0.5:8000 npm run dev
```

## Layout

- `src/protocol/` — wire types, control-frame builders, the pure reducer (+ tests)
- `src/ws/` — the socket hook: connect, heartbeat, dispatch server events into the reducer
- `src/voice/` + `public/capture.js` — mic capture worklet; PCM rides the socket only while armed
- `src/api/` — HTTP doors (sections tree, health, corpus upload/status/pin/retry)
- `src/components/` — presentational panels; `App.tsx` owns polling and wiring
