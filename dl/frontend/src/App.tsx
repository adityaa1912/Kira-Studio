import { useEffect, useState } from 'react'
import { ArrowDownToLine, CircleHelp, Dice5, Download, ImagePlus, RefreshCw, Sparkles, WandSparkles } from 'lucide-react'

type DatasetStatus = {
  available: boolean
  image_count: number
  source: 'archive' | 'extracted' | 'missing'
  path: string
}

type HealthStatus = {
  status: string
  ready: boolean
  model_available: boolean
  model_name: string
  output_size: number
  ai_upscale_available: boolean
  ai_upscale_model: string
  ai_upscale_output_size: number
}

type GenerationResult = {
  images: string[]
  seed: number
  count: number
  variation: number
  ai_upscaled: boolean
  output_size: number
}

const resultCounts = [1, 4, 8]

function App() {
  const [dataset, setDataset] = useState<DatasetStatus | null>(null)
  const [health, setHealth] = useState<HealthStatus | null>(null)
  const [statusChecked, setStatusChecked] = useState(false)
  const [count, setCount] = useState(4)
  const [seed, setSeed] = useState('')
  const [variation, setVariation] = useState(1)
  const [aiUpscale, setAiUpscale] = useState(true)
  const [results, setResults] = useState<GenerationResult | null>(null)
  const [isGenerating, setIsGenerating] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    const loadStatus = async () => {
      try {
        const [healthResult, datasetResult] = await Promise.allSettled([
          fetch('/api/health'),
          fetch('/api/dataset'),
        ])
        if (healthResult.status === 'fulfilled' && healthResult.value.ok) {
          setHealth(await healthResult.value.json() as HealthStatus)
        } else {
          setHealth(null)
        }
        if (datasetResult.status === 'fulfilled' && datasetResult.value.ok) {
          setDataset(await datasetResult.value.json() as DatasetStatus)
        } else {
          setDataset(null)
        }
      } catch {
        setHealth(null)
        setDataset(null)
      } finally {
        setStatusChecked(true)
      }
    }
    void loadStatus()
  }, [])

  const generate = async (options?: { count: number; seed: number | null; variation: number; aiUpscale: boolean }) => {
    const requestedSeed = options ? options.seed : seed.trim() === '' ? null : Number(seed)
    const requestedCount = options ? options.count : count
    const requestedVariation = options ? options.variation : variation
    const requestedUpscale = options ? options.aiUpscale : aiUpscale && Boolean(health?.ai_upscale_available)
    if (requestedSeed !== null && (!Number.isSafeInteger(requestedSeed) || requestedSeed < 0 || requestedSeed > 2_147_483_647)) {
      setError('Seed must be a whole number between 0 and 2,147,483,647.')
      return
    }
    setIsGenerating(true)
    setError('')
    try {
      const response = await fetch('/api/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          count: requestedCount,
          seed: requestedSeed,
          variation: requestedVariation,
          ai_upscale: requestedUpscale,
        }),
      })
      const body = await response.json().catch(() => null)
      if (!response.ok) throw new Error(body?.detail ?? 'Generation failed. Check the backend model.')
      if (
        !body ||
        !Array.isArray(body.images) ||
        !body.images.every((image: unknown) => typeof image === 'string') ||
        body.images.length !== requestedCount ||
        body.count !== requestedCount ||
        !Number.isSafeInteger(body.seed) ||
        body.seed < 0 ||
        body.seed > 2_147_483_647 ||
        !Number.isFinite(body.variation) ||
        body.ai_upscaled !== requestedUpscale ||
        !Number.isSafeInteger(body.output_size)
      ) {
        throw new Error('The backend returned an invalid generation result.')
      }
      setResults(body as GenerationResult)
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'Could not reach the local generator.')
    } finally {
      setIsGenerating(false)
    }
  }

  const randomizeSeed = () => setSeed(String(Math.floor(Math.random() * 2_147_483_647)))

  const downloadImage = (image: string, index: number) => {
    const link = document.createElement('a')
    link.href = `data:image/jpeg;base64,${image}`
    link.download = `kira-face-${String(index + 1).padStart(2, '0')}.jpg`
    link.click()
  }

  const backendOnline = health?.status === 'ok'

  return (
    <main className="app-shell">
      <header className="topbar">
        <a className="wordmark" href="#top" aria-label="Kira Studio home">
          <span className="brand-mark"><Sparkles size={17} strokeWidth={2.2} /></span>
          <span>KIRA<span className="wordmark-light"> / STUDIO</span></span>
        </a>
        <div className="topbar-meta">
          <span className={`connection ${backendOnline ? 'is-online' : ''}`}>
            <span className="connection-dot" />
            {backendOnline ? 'LOCAL ENGINE CONNECTED' : statusChecked ? 'ENGINE OFFLINE' : 'CONNECTING TO ENGINE'}
          </span>
          <a className="help-link" href="https://github.com/NeilGeorge1/Anime-Face-Generator" target="_blank" rel="noreferrer" aria-label="Open source project" title="Open source project">
            <CircleHelp size={18} />
          </a>
        </div>
      </header>

      <section className="workspace" id="top">
        <aside className="control-panel">
          <div className="eyebrow"><span>MODEL LAB</span><span className="eyebrow-rule" /></div>
          <h1>Make a<br /><em>new face.</em></h1>
          <p className="intro">A little noise, a whole new character. Shape a batch and let the GAN do the rest.</p>

          <div className="field-block">
            <div className="field-heading"><span>IMAGES</span><span className="field-value">{count} outputs</span></div>
            <div className="segmented-control" role="group" aria-label="Number of images">
              {resultCounts.map((option) => (
                <button key={option} className={count === option ? 'selected' : ''} onClick={() => setCount(option)} aria-pressed={count === option}>
                  {String(option).padStart(2, '0')}
                </button>
              ))}
            </div>
          </div>

          <div className="field-block">
            <div className="field-heading"><label htmlFor="variation">VARIATION</label><span className="field-value">{variation.toFixed(2)}</span></div>
            <input id="variation" className="range-input" type="range" min="0.5" max="1.5" step="0.05" value={variation} onChange={(event) => setVariation(Number(event.target.value))} />
            <div className="range-ends"><span>SUBTLE</span><span>WILD</span></div>
          </div>

          <div className="field-block seed-block">
            <div className="field-heading"><label htmlFor="seed">SEED</label><span className="field-value">OPTIONAL</span></div>
            <div className="seed-input-wrap">
              <input id="seed" type="number" min="0" max="2147483647" placeholder="Random each time" value={seed} onChange={(event) => setSeed(event.target.value)} />
              <button className="icon-button" type="button" onClick={randomizeSeed} title="Choose a random seed" aria-label="Choose a random seed"><Dice5 size={17} /></button>
            </div>
          </div>

          <div className="upscale-control">
            <label>
              <input
                type="checkbox"
                checked={aiUpscale}
                onChange={(event) => setAiUpscale(event.target.checked)}
                disabled={!health?.ai_upscale_available}
              />
              <span>AI upscale to {health?.ai_upscale_output_size ?? 672}px</span>
            </label>
            <p>{health?.ai_upscale_available ? 'Adds learned detail; takes a little longer.' : 'AI upscaler unavailable. Install requirements and restart the backend.'}</p>
          </div>

          <button className="generate-button" onClick={() => void generate()} disabled={isGenerating || !health?.ready}>
            {isGenerating ? <span className="spinner" /> : <WandSparkles size={18} />}
            <span>{isGenerating ? 'Creating faces' : 'Generate faces'}</span>
            {!isGenerating && <span className="button-arrow">↗</span>}
          </button>
          {!health?.ready && <p className="status-message" role="status">{!statusChecked ? 'Checking the local generator…' : backendOnline ? 'Add a valid model file before generating faces.' : 'Start the local API server to generate faces.'}</p>}
          {error && <p className="error-message" role="alert">{error}</p>}

          <div className="engine-note">
            <div className="engine-icon"><ImagePlus size={17} /></div>
            <div>
              <strong>{!statusChecked ? 'Checking generator' : health?.model_available ? 'Pretrained generator ready' : backendOnline ? 'Model unavailable' : 'Engine offline'}</strong>
              <span>{health?.model_available ? `${health.output_size} × ${health.output_size} output · ${health.model_name}` : backendOnline ? `Could not load ${health?.model_name ?? 'the configured model'}; check the backend terminal` : 'Start the local API server'}</span>
            </div>
            <span className={`engine-status ${health?.model_available ? 'ready' : ''}`} />
          </div>
        </aside>

        <section className="output-panel" aria-label="Generated faces">
          <div className="output-heading">
            <div>
              <div className="eyebrow"><span>CANVAS / 01</span><span className="eyebrow-rule" /></div>
              <h2>{results ? 'Your latest batch' : 'Waiting for a spark'}</h2>
            </div>
            {results && <div className="result-meta">SEED <b>{results.seed}</b></div>}
          </div>

          {isGenerating && results ? <p className="status-message" role="status">Generating your next batch…</p> : null}
          {results ? (
            <div className={`face-grid count-${results.images.length}`}>
              {results.images.map((image, index) => (
                <article className="face-tile" key={`${results.seed}-${index}`}>
                  <img src={`data:image/jpeg;base64,${image}`} alt={`Generated anime face ${index + 1}`} loading="lazy" decoding="async" width={results.output_size} height={results.output_size} />
                  <span className="tile-index">{String(index + 1).padStart(2, '0')}</span>
                  <button className="tile-download" onClick={() => downloadImage(image, index)} aria-label={`Download face ${index + 1}`} title="Download image"><ArrowDownToLine size={16} /></button>
                </article>
              ))}
            </div>
          ) : (
            <div className="empty-state">
              <div className="empty-orbit orbit-one" />
              <div className="empty-orbit orbit-two" />
              <div className="empty-spark"><Sparkles size={28} /></div>
              <span className="empty-caption">{health?.ai_upscale_available && aiUpscale ? health.ai_upscale_output_size : health?.output_size ?? 256} × {health?.ai_upscale_available && aiUpscale ? health.ai_upscale_output_size : health?.output_size ?? 256} PX</span>
              <p>Generated portraits<br />will appear here</p>
            </div>
          )}

          <div className="output-footer">
            <div className="dataset-status">
              <span className={`dataset-mark ${dataset?.available ? 'available' : ''}`} />
              <span>{dataset ? `${dataset.image_count.toLocaleString()} TRAINING IMAGES` : 'CHECKING DATASET'}</span>
              <span className="dataset-source">{dataset?.source === 'archive' ? 'ZIP READY' : dataset?.source === 'extracted' ? 'EXTRACTED' : dataset?.source === 'missing' ? 'NOT FOUND' : ''}</span>
            </div>
            {results && <div className="result-actions">
              <button className="text-action" onClick={() => void generate({ count: results.count, seed: results.seed, variation: results.variation, aiUpscale: results.ai_upscaled })} disabled={isGenerating}><RefreshCw size={15} /> Repeat batch</button>
              <button className="text-action" onClick={() => results.images.forEach(downloadImage)}><Download size={15} /> Download batch</button>
            </div>}
          </div>
        </section>
      </section>

      <footer className="page-footer">
        <span>ANIME FACE GENERATION / DCGAN</span>
        <span>RUNNING ON YOUR MACHINE <span className="footer-star">✳</span></span>
      </footer>
    </main>
  )
}

export default App