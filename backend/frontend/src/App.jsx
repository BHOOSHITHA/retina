import React, { useState, useRef, useEffect } from 'react';
import { Upload, Zap, Eye, Terminal, Play, Activity, Shield, ChevronRight, Settings, Info } from 'lucide-react';
import './App.css';

const API_BASE = "http://127.0.0.1:8000";

function LiveTuningDashboard() {
  const [logs, setLogs] = useState([]);
  const [status, setStatus] = useState('idle');
  const [runId, setRunId] = useState(null);
  const scrollRef = useRef(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [logs]);

  const startTuning = async () => {
    try {
      setStatus('running');
      setLogs(['[System] Initializing Swarm Auto-Tuner...', '[System] Connecting to MHO optimization engine...']);
      const res = await fetch(`${API_BASE}/optimize`, { method: "POST" });
      const data = await res.json();
      setRunId(data.run_id);
      pollStatus(data.run_id);
    } catch (err) {
      setLogs(l => [...l, "[Error] Failed to connect to backend engine"]);
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
    <div className="glass-card dashboard-card">
      <div className="card-header">
        <div className="card-title-group">
          <Terminal size={20} className="icon-accent" />
          <h3>MHO Swarm Tuning Intelligence</h3>
        </div>
        <div className={`status-indicator ${status}`}>
          <span className="dot"></span>
          {status.charAt(0).toUpperCase() + status.slice(1)}
        </div>
      </div>
      
      <div className="dashboard-content">
        <div className="terminal-container">
          <div className="terminal-header">
            <div className="terminal-controls">
              <span></span><span></span><span></span>
            </div>
            <div className="terminal-title">mho-swarm-optimizer --verbose</div>
          </div>
          <div className="terminal-body" ref={scrollRef}>
             {logs.length === 0 && <div className="terminal-placeholder">System idle. Awaiting optimization command...</div>}
             {logs.map((log, i) => (
               <div key={i} className="log-line">
                 <span className="log-timestamp">[{new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit', second:'2-digit'})}]</span>
                 <span className="log-text">{log}</span>
               </div>
             ))}
             {status === 'running' && <div className="terminal-cursor"></div>}
          </div>
        </div>

        <div className="dashboard-actions">
          <button 
            className="action-btn-primary" 
            onClick={startTuning} 
            disabled={status === 'running'}
          >
             {status === 'running' ? <div className="loading-spinner-small"></div> : <Play size={18} />}
             <span>{status === 'running' ? "Optimizing Neural Weights..." : "Initialize Swarm Tuner"}</span>
          </button>
          
          <div className="info-panel">
            <div className="info-item">
              <Activity size={16} />
              <span>Real-time Swarm Feedback</span>
            </div>
            <div className="info-item">
              <Shield size={16} />
              <span>Secure Adaptive Architecture</span>
            </div>
          </div>
        </div>
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
      console.error(err);
      alert("Pipeline failure. Ensure backend services are active.");
    } finally {
      setSegLoading(false);
    }
  };

  return (
    <div className="app-shell">
      <nav className="side-nav">
        <div className="nav-logo">
          <div className="logo-icon"><Zap size={24} /></div>
          <span className="logo-text">Retina<span>AI</span></span>
        </div>
        
        <div className="nav-items">
          <button 
            className={`nav-item ${currentView === 'segmenter' ? 'active' : ''}`}
            onClick={() => setCurrentView('segmenter')}
          >
            <Eye size={20} />
            <span>Diagnostics</span>
          </button>
          <button 
            className={`nav-item ${currentView === 'tuner' ? 'active' : ''}`}
            onClick={() => setCurrentView('tuner')}
          >
            <Settings size={20} />
            <span>Swarm Tuner</span>
          </button>
        </div>
        
        <div className="nav-footer">
          <div className="user-profile">
            <div className="avatar">RD</div>
            <div className="user-info">
              <span className="user-name">Dr. Retina</span>
              <span className="user-role">Lead Clinician</span>
            </div>
          </div>
        </div>
      </nav>

      <main className="main-content">
        <header className="top-header">
          <div className="header-breadcrumbs">
            <span>Dashboard</span>
            <ChevronRight size={14} />
            <span className="current">{currentView === 'segmenter' ? 'Vessel Segmentation' : 'Neural Tuning'}</span>
          </div>
          
          <div className="header-actions">
            <button className="icon-btn"><Info size={20} /></button>
            <div className="system-status">
              <span className="pulse"></span>
              System Active
            </div>
          </div>
        </header>

        <section className="content-area">
          {currentView === 'segmenter' ? (
            <div className="segmenter-view">
              <div className="view-intro">
                <h1>Clinical Retinal Analysis</h1>
                <p>Advanced vessel segmentation using MHO-optimized neural architectures.</p>
              </div>

              <div className="glass-card main-card">
                {!imagePreview ? (
                  <div 
                    className="drop-zone"
                    onClick={() => fileInputRef.current?.click()}
                  >
                    <div className="drop-zone-inner">
                      <div className="upload-illustration">
                        <Upload size={40} />
                      </div>
                      <h3>Import Retinal Scan</h3>
                      <p>Drag and drop DICOM or high-resolution JPEG/PNG files</p>
                      <button className="btn-secondary">Browse Files</button>
                    </div>
                    <input type="file" hidden ref={fileInputRef} onChange={handleImageSelect} accept="image/*" />
                  </div>
                ) : (
                  <div className="diagnostic-workflow">
                    {!segSteps ? (
                      <div className="preview-stage">
                        <div className="image-preview-wrapper">
                          <img src={imagePreview} alt="Preview" />
                          <div className="image-overlay-info">Original Scan</div>
                        </div>
                        <div className="preview-controls">
                          <button 
                            className="btn-primary-large" 
                            onClick={handleSegmentation} 
                            disabled={segLoading}
                          >
                            {segLoading ? <div className="loading-spinner"></div> : <Activity size={20} />}
                            <span>{segLoading ? "Processing Neural Layers..." : "Execute AI Diagnosis"}</span>
                          </button>
                          <button 
                            className="btn-ghost"
                            onClick={() => { setImagePreview(null); setSelectedImage(null); }}
                            disabled={segLoading}
                          >
                            Remove Image
                          </button>
                        </div>
                      </div>
                    ) : (
                      <div className="result-stage">
                        <div className="result-grid">
                          <div className="result-item">
                            <div className="result-label">Original</div>
                            <div className="result-img-box"><img src={imagePreview} alt="Original" /></div>
                          </div>
                          <div className="result-item">
                            <div className="result-label">Channel Ops</div>
                            <div className="result-img-box"><img src={segSteps.green} alt="Green" /></div>
                          </div>
                          <div className="result-item">
                            <div className="result-label">Enhanced</div>
                            <div className="result-img-box"><img src={segSteps.clahe} alt="CLAHE" /></div>
                          </div>
                          <div className="result-item highlight">
                            <div className="result-label">AI Segmentation</div>
                            <div className="result-img-box primary"><img src={segSteps.overlay} alt="Overlay" /></div>
                          </div>
                        </div>

                        <div className="analysis-summary">
                          <div className="metrics-container">
                            <div className="metric-card">
                              <span className="m-label">Confidence Score</span>
                              <span className="m-value">{(segMetrics.estimated_dice * 100).toFixed(1)}%</span>
                              <div className="m-bar"><div style={{ width: `${segMetrics.estimated_dice * 100}%` }}></div></div>
                            </div>
                            <div className="metric-card">
                              <span className="m-label">Inference Latency</span>
                              <span className="m-value">{segMetrics.inference_time_ms.toFixed(0)}<small>ms</small></span>
                              <div className="m-bar"><div className="latency" style={{ width: '40%' }}></div></div>
                            </div>
                          </div>

                          <div className="mho-specs-panel">
                            <div className="specs-header">
                              <Zap size={16} />
                              <span>MHO Optimization Profile</span>
                            </div>
                            <div className="specs-grid">
                              <div className="spec-item"><label>Engine</label><span>{mhoSpecs.algorithm}</span></div>
                              <div className="spec-item"><label>Model</label><span>{mhoSpecs.architecture}</span></div>
                              <div className="spec-item"><label>Batch</label><span>{mhoSpecs.batch_size}</span></div>
                              <div className="spec-item"><label>LR Rate</label><span>{mhoSpecs.learning_rate}</span></div>
                            </div>
                          </div>
                        </div>

                        <div className="result-footer">
                          <button 
                            className="btn-primary"
                            onClick={() => { setImagePreview(null); setSelectedImage(null); setSegSteps(null); }}
                          >
                            New Patient Analysis
                          </button>
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          ) : (
            <div className="tuner-view">
              <div className="view-intro">
                <h1>Neural Swarm Optimization</h1>
                <p>Fine-tune model hyperparameters using Multi-Objective Swarm Intelligence.</p>
              </div>
              <LiveTuningDashboard />
            </div>
          )}
        </section>
      </main>
    </div>
  );
}

export default App;
