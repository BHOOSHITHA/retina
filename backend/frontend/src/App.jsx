import React, { useState, useRef, useEffect } from 'react';
import { Upload, Zap, Eye, Terminal, Play } from 'lucide-react';
import './App.css';

const API_BASE = "http://127.0.0.1:8000";

function LiveTuningDashboard() {
  const [logs, setLogs] = useState([]);
  const [status, setStatus] = useState('idle');
  const [runId, setRunId] = useState(null);

  const startTuning = async () => {
    try {
      setStatus('running');
      setLogs(['[System] Initializing Swarm Auto-Tuner...']);
      const res = await fetch(`${API_BASE}/optimize`, { method: "POST" });
      const data = await res.json();
      setRunId(data.run_id);
      pollStatus(data.run_id);
    } catch (err) {
      setLogs(l => [...l, "[Error] Failed to connect to backend"]);
      setStatus('idle');
    }
  };

  const pollStatus = (id) => {
    const timer = setInterval(async () => {
      try {
        const res = await fetch(`${API_BASE}/runs/${id}`);
        const data = await res.json();
        
        if (data.logs) {
           setLogs(data.logs);
        }
        
        if (data.status === 'completed' || data.error) {
           clearInterval(timer);
           setStatus(data.status);
        }
      } catch (err) {
        clearInterval(timer);
        setStatus('idle');
      }
    }, 1000);
  };

  return (
    <div className="glass-card" style={{ padding: '3rem' }}>
      <div className="card-title" style={{ justifyContent: 'center', fontSize: '1.5rem', marginBottom: '2rem' }}>
        <Terminal size={28} className="icon" color="var(--primary-color)" />
        MHO Swarm Tuning Dashboard
      </div>
      
      <div style={{ textAlign: 'center', marginBottom: '2rem' }}>
        <button 
          className="primary-btn" 
          onClick={startTuning} 
          disabled={status === 'running'}
          style={{ padding: '1rem 2rem', fontSize: '1.1rem' }}
        >
           {status === 'running' ? <div className="loader" style={{ width: '20px', height: '20px', borderWidth: '3px' }}></div> : <Play size={20} />}
           {status === 'running' ? "Swarm is Optimizing..." : "Start Swarm Auto-Tuner"}
        </button>
      </div>

      <div className="terminal-window" style={{ background: '#0f172a', padding: '1.5rem', borderRadius: '12px', minHeight: '300px', border: '1px solid #334155', fontFamily: 'monospace', color: '#10b981', textAlign: 'left', overflowY: 'auto' }}>
         {logs.length === 0 && <span style={{ color: '#64748b' }}>Waiting to start system...</span>}
         {logs.map((log, i) => (
           <div key={i} className="log-line" style={{ marginBottom: '0.5rem' }}>{log}</div>
         ))}
         {status === 'running' && <div className="blinking-cursor" style={{ display: 'inline-block', width: '8px', height: '15px', background: '#10b981', animation: 'blink 1s step-end infinite' }}></div>}
      </div>
    </div>
  );
}

function App() {
  const [selectedImage, setSelectedImage] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [currentView, setCurrentView] = useState('segmenter');
  
  const [segSteps, setSegSteps] = useState(null);
  const [segMetrics, setSegMetrics] = useState(null);
  const [mhoSpecs, setMhoSpecs] = useState(null);
  const [segLoading, setSegLoading] = useState(false);
  const fileInputRef = useRef(null);

  const handleImageSelect = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedImage(file);
      setImagePreview(URL.createObjectURL(file));
      setSegSteps(null);
      setSegMetrics(null);
      setMhoSpecs(null);
    }
  };

  const handleSegmentation = async () => {
    if (!selectedImage) return;
    setSegLoading(true);
    
    const formData = new FormData();
    formData.append("file", selectedImage);

    try {
      const res = await fetch(`${API_BASE}/segment`, {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      setSegSteps({
          green: `data:image/jpeg;base64,${data.steps.green_channel}`,
          clahe: `data:image/jpeg;base64,${data.steps.clahe}`,
          overlay: `data:image/jpeg;base64,${data.steps.overlay}`
      });
      setSegMetrics(data.metrics);
      setMhoSpecs(data.mho_specs);
    } catch (err) {
      alert("Segmentation API failed. Ensure the Python backend is running.");
    } finally {
      setSegLoading(false);
    }
  };

  return (
    <div className="app-container" style={{ maxWidth: '1000px' }}>
      <header className="header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '2rem' }}>
        <div style={{ textAlign: 'left' }}>
          <h1>Clinical Retinal Diagnosis System</h1>
          <p>Instant System-Automated Vessel Segmentation Workflow</p>
        </div>
        <div style={{ display: 'flex', gap: '1rem', flexShrink: 0 }}>
          <button 
             onClick={() => setCurrentView('segmenter')}
             style={{ background: currentView === 'segmenter' ? 'var(--primary-color)' : 'rgba(255,255,255,0.1)', color: 'white', border: 'none', padding: '0.6rem 1.2rem', borderRadius: '8px', cursor: 'pointer', fontWeight: 'bold' }}
          >
             Segmenter
          </button>
          <button 
             onClick={() => setCurrentView('tuner')}
             style={{ background: currentView === 'tuner' ? 'var(--primary-color)' : 'rgba(255,255,255,0.1)', color: 'white', border: 'none', padding: '0.6rem 1.2rem', borderRadius: '8px', cursor: 'pointer', fontWeight: 'bold' }}
          >
             Live Tuning Dashboard
          </button>
        </div>
      </header>

      {currentView === 'segmenter' ? (
      <div className="glass-card" style={{ padding: '3rem' }}>
        <div className="card-title" style={{ justifyContent: 'center', fontSize: '1.5rem', marginBottom: '2rem' }}>
          <Eye size={28} className="icon" color="var(--success-color)" />
          Diagnostic Control Center
        </div>

        {!imagePreview && (
          <div 
             className="upload-zone"
             onClick={() => fileInputRef.current?.click()}
             style={{ padding: '4rem 2rem' }}
          >
             <Upload size={48} className="upload-icon" />
             <div className="upload-text" style={{ fontSize: '1.25rem' }}>Securely Upload Patient Eye Scan</div>
             <div className="upload-sub" style={{ fontSize: '1rem' }}>Drag & Drop or Click to Browse</div>
             <input type="file" hidden ref={fileInputRef} onChange={handleImageSelect} accept="image/*" />
          </div>
        )}

        {imagePreview && !segSteps && (
          <div style={{ textAlign: 'center' }}>
            <div className="result-image-container mb-4" style={{ display: 'inline-block', maxWidth: '400px' }}>
              <img src={imagePreview} className="result-image" alt="Preview" />
              <button 
                className="primary-btn" 
                style={{ margin: 0, borderRadius: '0 0 12px 12px', padding: '1rem', fontSize: '1.1rem' }} 
                onClick={handleSegmentation} 
                disabled={segLoading}
              >
                {segLoading ? <div className="loader"></div> : <Zap size={24} />}
                {segLoading ? "Executing AI Diagnosis Model..." : "Run AI Diagnosis Pipeline"}
              </button>
            </div>
            <div style={{ marginTop: '1rem' }}>
                <button 
                  onClick={() => { setImagePreview(null); setSelectedImage(null); }}
                  style={{ background: 'transparent', color: 'var(--text-secondary)', textDecoration: 'underline' }}
                >
                  Upload a different image
                </button>
            </div>
          </div>
        )}

        {segSteps && (
          <>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '1.5rem', alignItems: 'flex-start' }}>
              <div className="step-card">
                 <h4 style={{ color: 'var(--text-secondary)', marginBottom: '0.75rem', fontSize: '0.9rem', textAlign: 'center' }}>1. Original Upload</h4>
                 <div className="result-image-container"><img src={imagePreview} className="result-image" /></div>
              </div>
              <div className="step-card">
                 <h4 style={{ color: 'var(--text-secondary)', marginBottom: '0.75rem', fontSize: '0.9rem', textAlign: 'center' }}>2. Green Channel</h4>
                 <div className="result-image-container"><img src={segSteps.green} className="result-image" /></div>
              </div>
              <div className="step-card">
                 <h4 style={{ color: 'var(--text-secondary)', marginBottom: '0.75rem', fontSize: '0.9rem', textAlign: 'center' }}>3. CLAHE Enhancement</h4>
                 <div className="result-image-container"><img src={segSteps.clahe} className="result-image" /></div>
              </div>
              <div className="step-card">
                 <h4 style={{ color: '#fff', marginBottom: '0.75rem', fontSize: '0.9rem', textAlign: 'center', fontWeight: 'bold' }}>4. Deep AI Output</h4>
                 <div className="result-image-container" style={{ border: '2px solid var(--success-color)', boxShadow: '0 0 15px rgba(16, 185, 129, 0.4)' }}>
                    <img src={segSteps.overlay} className="result-image" />
                 </div>
              </div>
            </div>

            {segMetrics && (
              <div className="metrics-box" style={{ maxWidth: '400px', margin: '2rem auto 0' }}>
                <div className="metric-row">
                  <span className="metric-label">System Confidence (Dice)</span>
                  <span className="metric-value">{(segMetrics.estimated_dice * 100).toFixed(1)}%</span>
                </div>
                <div className="metric-row">
                  <span className="metric-label">Pipeline Execution Time</span>
                  <span className="metric-value">{segMetrics.inference_time_ms.toFixed(1)} ms</span>
                </div>
              </div>
            )}
            
            {mhoSpecs && (
              <div className="mho-badge" style={{ maxWidth: '450px', margin: '1.5rem auto 0', padding: '1.25rem', background: 'rgba(16, 185, 129, 0.1)', border: '1px solid var(--success-color)', borderRadius: '12px' }}>
                <h4 style={{ color: 'var(--success-color)', marginBottom: '1rem', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.5rem', fontSize: '1.1rem', marginTop: 0 }}>
                  <Zap size={20} /> MHO Optimized Pipeline Used
                </h4>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.8rem', fontSize: '0.9rem', color: 'var(--text-secondary)' }}>
                  <div><strong>Algorithm:</strong> <span style={{color: '#fff'}}>{mhoSpecs.algorithm}</span></div>
                  <div><strong>Architecture:</strong> <span style={{color: '#fff'}}>{mhoSpecs.architecture}</span></div>
                  <div><strong>Batch Size:</strong> <span style={{color: '#fff'}}>{mhoSpecs.batch_size}</span></div>
                  <div><strong>Learning Rate:</strong> <span style={{color: '#fff'}}>{mhoSpecs.learning_rate}</span></div>
                </div>
              </div>
            )}
            
            <div style={{ marginTop: '2rem', textAlign: 'center' }}>
                <button 
                  className="primary-btn" 
                  style={{ display: 'inline-flex', width: 'auto', padding: '0.75rem 2rem' }}
                  onClick={() => { setImagePreview(null); setSelectedImage(null); setSegSteps(null); }}
                >
                  Analyze New Patient Scan
                </button>
            </div>
          </>
        )}

      </div>
      ) : (
        <LiveTuningDashboard />
      )}
    </div>
  );
}

export default App;
