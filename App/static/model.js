/* ────────────────── Estado global ────────────────── */
  let selectedFile  = null;
  let currentMode   = 'file';
  let stream        = null;
  let facingMode    = 'environment';   // 'user' = frontal, 'environment' = trasera

  /* ────────────────── Referencias DOM ────────────────── */
  const fileInput      = document.getElementById('file-input');
  const previewImg     = document.getElementById('preview');
  const previewWrap    = document.getElementById('preview-wrap');
  const scanBtn        = document.getElementById('scan-btn');
  const loader         = document.getElementById('loader');
  const resultDiv      = document.getElementById('result');
  const video          = document.getElementById('camera-video');
  const snapCanvas     = document.getElementById('snap');
  const snapPreview    = document.getElementById('snap-preview');
  const snapPreviewWrap= document.getElementById('snap-preview-wrap');
  const captureBtn     = document.getElementById('capture-btn');

  /* ════════════════ CAMBIO DE MODO ════════════════ */
  async function switchMode(mode) {
    currentMode = mode;

    document.getElementById('tab-file').classList.toggle('active', mode === 'file');
    document.getElementById('tab-cam').classList.toggle('active',  mode === 'camera');
    document.getElementById('file-section').style.display   = mode === 'file'   ? 'flex'  : 'none';
    document.getElementById('camera-section').style.display = mode === 'camera' ? 'flex'  : 'none';

    // Ajustar layout de file-section a columna
    document.getElementById('file-section').style.flexDirection = 'column';
    document.getElementById('file-section').style.gap = '16px';

    clearResult();

    if (mode === 'camera') {
      await startCamera();
    } else {
      stopCamera();
      snapPreviewWrap.style.display = 'none';
    }
  }

  /* ════════════════ MODO ARCHIVO ════════════════ */
  function loadFile(file) {
    if (!file || !file.type.startsWith('image/')) return;
    selectedFile = file;
    previewImg.src = URL.createObjectURL(file);
    previewWrap.style.display = 'block';
    scanBtn.style.display = 'block';
    clearResult();
  }

  fileInput.addEventListener('change', e => loadFile(e.target.files[0]));

  const dropArea = document.getElementById('drop-area');
  dropArea.addEventListener('dragover', e => {
    e.preventDefault();
    dropArea.style.background = '#e8f0fb';
    dropArea.style.borderStyle = 'solid';
  });
  dropArea.addEventListener('dragleave', () => {
    dropArea.style.background = '';
    dropArea.style.borderStyle = 'dashed';
  });
  dropArea.addEventListener('drop', e => {
    e.preventDefault();
    dropArea.style.background = '';
    dropArea.style.borderStyle = 'dashed';
    loadFile(e.dataTransfer.files[0]);
  });

  async function scanFile() {
    if (!selectedFile) return;
    await sendToAPI(selectedFile);
  }

  /* ════════════════ MODO CÁMARA ════════════════ */
  async function startCamera() {
    stopCamera();
    captureBtn.disabled = true;
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode, width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false
      });
      video.srcObject = stream;
      await video.play();
      captureBtn.disabled = false;
    } catch (err) {
      showError('No se pudo acceder a la cámara. Asegúrate de dar permiso en el navegador.<br/><small>' + err.message + '</small>');
      captureBtn.disabled = false;
    }
  }

  function stopCamera() {
    if (stream) {
      stream.getTracks().forEach(t => t.stop());
      stream = null;
    }
    video.srcObject = null;
  }

  async function flipCamera() {
    facingMode = facingMode === 'environment' ? 'user' : 'environment';
    await startCamera();
  }

  function captureFrame() {
    if (!stream) return;

    // Dibuja el fotograma actual en el canvas
    snapCanvas.width  = video.videoWidth  || 640;
    snapCanvas.height = video.videoHeight || 480;
    const ctx = snapCanvas.getContext('2d');
    ctx.drawImage(video, 0, 0, snapCanvas.width, snapCanvas.height);

    // Muestra previsualización
    const dataUrl = snapCanvas.toDataURL('image/jpeg', 0.92);
    snapPreview.src = dataUrl;
    snapPreviewWrap.style.display = 'block';

    // Convierte a Blob y envía
    snapCanvas.toBlob(async blob => {
      const file = new File([blob], 'captura.jpg', { type: 'image/jpeg' });
      await sendToAPI(file);
    }, 'image/jpeg', 0.92);
  }

  /* ════════════════ API CALL ════════════════ */
  async function sendToAPI(file) {
    setLoading(true);

    const form = new FormData();
    form.append('file', file);

    try {
      const res  = await fetch('/identify', { method: 'POST', body: form });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      renderResult(data);
    } catch (e) {
      showError('Error al conectar con la API. Inténtalo de nuevo.<br/><small>' + e.message + '</small>');
    } finally {
      setLoading(false);
    }
  }

  /* ════════════════ HELPERS ════════════════ */
  function setLoading(on) {
    loader.style.display = on ? 'block' : 'none';
    if (currentMode === 'file') scanBtn.disabled = on;
    if (currentMode === 'camera') captureBtn.disabled = on;
    if (on) clearResult();
  }

  function clearResult() {
    resultDiv.style.display = 'none';
    resultDiv.innerHTML = '';
  }

  function showError(msg) {
    resultDiv.style.display = 'flex';
    resultDiv.innerHTML = `<div class="error-card">⚠️ ${msg}</div>`;
  }

  function renderResult(data) {
    resultDiv.style.display = 'flex';
    const best = data.best_match;
    const conf = best.confidence;

    if (conf < 0.60) {
      resultDiv.innerHTML = `
        <div class="unsure-card">
          <strong>No estoy seguro 🤔</strong><br/>
          No he podido identificar el producto con suficiente confianza.<br/>
          Intenta acercar más la cámara al mueble o mejorar la iluminación.
        </div>`;
      return;
    }

    let html = `<div class="result-title">Mejor coincidencia</div>`;
    html += `
      <div class="product-card best">
        <div class="info">
          <div class="name">${best.nombre}</div>
          <div class="cat">${best.subtitulo}</div>
        </div>
        <div class="badge">${best.confidence_pct}</div>
      </div>`;

    if (data.alternatives && data.alternatives.length > 0) {
      html += `<div class="result-title" style="margin-top:4px">Otras posibilidades</div>`;
      data.alternatives.forEach(alt => {
        html += `
          <div class="product-card alt">
            <div class="info">
              <div class="name">${alt.nombre}</div>
              <div class="cat">${alt.subtitulo}</div>
            </div>
            <div class="badge">${alt.confidence_pct}</div>
          </div>`;
      });
    }

    resultDiv.innerHTML = html;
    // Hacer scroll suave hacia los resultados
    resultDiv.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }

  /* ── Inicialización: mostrar sección archivo por defecto ── */
  document.getElementById('camera-section').style.display = 'none';