# Website (React + Vite + Tailwind)

Front end for the project. A ready-built copy is already in `web/dist/` and is served by
`python main.py serve` at http://localhost:8000, so **Node.js is not needed just to use the site**.

| Page | What it does |
|---|---|
| `/` (Home) | Project overview and the 5-agent pipeline |
| `/playground` | Runs Requirement → Developer → Bug Predictor → Tester → Fixer live on your own task or an MBPP / HumanEval problem |
| `/scanner` | Paste Python code to get the CodeBERT bug score, per-function scores and highlighted lines |

## Editing the front end (needs Node.js 18+)

From the project folder, in two terminals:

```bash
python main.py serve
```

```bash
cd web
npm install
npm run dev
```

Open http://localhost:5173 (requests to `/api` are forwarded to port 8000).

After changing the code, rebuild the copy served by `main.py serve`:

```bash
cd web
npm run build
```
