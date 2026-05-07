// ═══════════════════════════════ DATA ═══════════════════════════════

let IKEA_DB = {};        // { "Puerta | VOXTORP": { ...datos completos } }
let allProducts = [];   // Array plano para iteración
let catalogLoaded = false;

// Cargar catálogo desde el servidor
async function loadCatalog() {
  try {
    const res = await fetch('/catalog');
    if (res.ok) {
      const data = await res.json();
      IKEA_DB = {};
      allProducts = [];

      let idCounter = 1;
      Object.entries(data).forEach(([key, product]) => {
        // La key es el identificador único del JSON (ej: "Puerta | VOXTORP")
        const priceNum = typeof product.precio === 'number' ? product.precio :
                        (parseFloat(String(product.precio).replace(/[^\d,]/g, '').replace(',', '.')) || 0);
        const loc = product.ubicacion || { pasillo: '-', estanteria: '-' };

        // Guardar con la key completa como identificador único
        IKEA_DB[key] = {
          key: key,
          nombre: product.nombre || key.split('|')[0].trim(),
          subtitulo: product.subtitulo || '',
          descripcion: product.descripcion || product.subtitulo || '',
          categoria: product.categoria || '',
          price: priceNum,
          priceStr: priceNum > 0 ? priceNum.toFixed(2).replace('.', ',') + ' €' : 'Consultar',
          location: `Pasillo ${loc.pasillo}·${loc.estanteria}`,
          pasillo: String(loc.pasillo || '-'),
          estanteria: String(loc.estanteria || '-'),
          peso: product.peso || '-',
          image: product.imagen || '',
          url: product.url || '#',
          // Campos legacy para compatibilidad
          name: product.nombre || key.split('|')[0].trim(),
          desc: product.subtitulo || '',
          longDesc: product.descripcion || product.subtitulo || '',
          emoji: product.emoji || getCategoryEmoji(product.categoria || ''),
          weight: product.peso || '-',
          location: `Pasillo ${loc.pasillo}·${loc.estanteria}`,
          stock: product.stock || 10
        };
      });

      // Crear array plano con IDs únicos
      allProducts = Object.keys(IKEA_DB).map((key, i) => ({
        id: i + 1,
        key: key,
        ...IKEA_DB[key]
      }));

      catalogLoaded = true;
      console.log('✅ Catálogo cargado:', allProducts.length, 'productos');
      console.log('📦 Primer producto:', allProducts[0]?.key);
    }
  } catch (e) {
    console.error('❌ Error cargando catálogo:', e);
    showToast('⚠️ Error cargando catálogo');
  }

  // Inicializar UI con los datos disponibles
  initUIWithCatalog();
}

// Helper para obtener emoji según categoría
function getCategoryEmoji(categoria) {
  const emojiMap = {
    'Puerta': '🚪', 'Puerta armario': '🚪', 'Silla': '🪑', 'Sofá': '🛋️',
    'Sofá cama': '🛋️', 'Mesa': '🪑', 'Lámpara': '💡', 'Marco': '🖼️',
    'Fregadero': '🚰', 'Armario': '🗄️', 'Cama': '🛏️', 'Estantería': '📚',
    'Cómoda': '🗄️', 'Escritorio': '🖥️', 'Silla de oficina': '🪑',
    'Funda': '🧵', 'Alfombra': '🧵', 'Decoración': '🏠'
  };
  return emojiMap[categoria] || '📦';
}

// Función para buscar producto por key, nombre, o partial match
function findProduct(searchTerm) {
  if (!searchTerm || !allProducts.length) return null;

  // Buscar por key exacta
  if (IKEA_DB[searchTerm]) return IKEA_DB[searchTerm];

  // Buscar por key parcial
  const lowerSearch = searchTerm.toLowerCase();
  const byKey = Object.keys(IKEA_DB).find(k => k.toLowerCase().includes(lowerSearch));
  if (byKey) return IKEA_DB[byKey];

  // Buscar por nombre
  const byName = Object.values(IKEA_DB).find(p =>
    p.nombre && p.nombre.toLowerCase() === lowerSearch
  );
  if (byName) return byName;

  // Buscar por nombre parcial
  const byNamePartial = Object.values(IKEA_DB).find(p =>
    p.nombre && p.nombre.toLowerCase().includes(lowerSearch)
  );
  if (byNamePartial) return byNamePartial;

  return null;
}

// Helper para formatear precios
function formatPrice(price) {
  if (typeof price === 'string') return price;
  if (typeof price !== 'number') return 'Consultar';
  return price > 0 ? price.toFixed(2).replace('.', ',') + ' €' : 'Consultar';
}

// Helper para obtener imagen de producto
function getProductImage(product) {
  return product.image || product.imagen || '';
}

// Helper para obtener nombre para mostrar
function getProductDisplayName(product) {
  return product.key || product.nombre || 'Producto';
}

// Inicializar UI después de cargar catálogo
function initUIWithCatalog() {
  if (typeof renderCart === 'function') renderCart();
  if (typeof renderFavs === 'function') renderFavs();
  if (typeof renderShopping === 'function') renderShopping();
  if (typeof renderSearch === 'function') renderSearch('');
  if (typeof renderPaySummary === 'function') renderPaySummary();
  if (typeof renderRoute === 'function') renderRoute();
  if (typeof updateAllBadges === 'function') updateAllBadges();
  if (typeof renderProfile === 'function') renderProfile();
  if (typeof renderHomeProducts === 'function') renderHomeProducts();
}

// ═══════════════════════════════ DATOS DE USUARIO ═══════════════════════════════

let cartItems = [];    // Se llena dinámicamente desde el JSON
let favItems = [];     // Se llena dinámicamente desde el JSON
let routeItems = [];   // Se genera dinámicamente desde el carrito
let orderHistory = []; // Historial de pedidos completados

let userProfile = { name:'Nombre Usuario', email:'usuario@ikea.es', phone:'+34 600 000 000', bday:'15 / 03 / 1990' };
let currentProductId = null;
let currentProductKey = null;  // Key única del JSON para el producto actual
let currentScreen = 'login';
let screenHistory = [];

// ═══════════════════════════════ NAVIGATION ═══════════════════════════════

function goTo(screen) {
  const prev = document.getElementById('screen-' + currentScreen);
  const next = document.getElementById('screen-' + screen);
  if (!next || screen === currentScreen) return;
  
  screenHistory.push(currentScreen);
  
  next.style.transition = 'none'; 
  next.classList.remove('exit-left', 'active'); 

  void next.offsetWidth; 

  next.style.transition = ''; 

  if (prev) {
      prev.classList.remove('active');
      prev.classList.add('exit-left');
  }

  next.classList.add('active');
  
  onScreenLeave(currentScreen);
  currentScreen = screen;
  onScreenEnter(screen);
  
  if (screen === 'mapa') {
    setTimeout(() => {
      renderRoute();
    }, 50);
  }
  
}

function goBack() {
  if (!screenHistory.length) return;
  const prevScreen = screenHistory.pop();
  const prevEl = document.getElementById('screen-' + prevScreen);
  const currEl = document.getElementById('screen-' + currentScreen);

  if (currEl) {
      currEl.classList.remove('active', 'exit-left');
  }
  
  if (prevEl) {

      prevEl.style.transition = 'none';
      prevEl.classList.add('exit-left');
      prevEl.classList.remove('active');
      
      void prevEl.offsetWidth;
      
      prevEl.style.transition = '';

      prevEl.classList.remove('exit-left');
      prevEl.classList.add('active');
  }
  
  onScreenLeave(currentScreen);
  currentScreen = prevScreen;
  onScreenEnter(prevScreen);
}
function navTo(screen) { screenHistory = []; goTo(screen); }

function onScreenLeave(s) {
  if (s === 'ar')     stopCamera('ar');
  if (s === 'escaner') stopCamera('qr');
}
function onScreenEnter(s) {
  updateAllBadges();
  if (s === 'inicio')      renderHomeProducts();
  if (s === 'cesta')        renderCart();
  if (s === 'favoritos')    renderFavs();
  if (s === 'shopping')    renderShopping();
  if (s === 'buscar')       renderSearch('');
  if (s === 'pagar')        renderPaySummary();
  if (s === 'mapa')         renderRoute();
  if (s === 'ar')           { startCamera('ar'); renderARSuggestedProducts(); }
  if (s === 'escaner')      startCamera('qr');
  if (s === 'perfil')       renderProfile();
  if (s === 'historial')    renderHistorial();
}

// ═══════════════════════════════ HOME PRODUCTS ═══════════════════════════════

function renderHomeProducts() {
  const container = document.getElementById('home-products');
  if (!container) return;

  if (!catalogLoaded || !allProducts.length) {
    container.innerHTML = '<div style="text-align:center;padding:30px;color:var(--gray);">Cargando productos...</div>';
    return;
  }

  // Seleccionar productos aleatorios (máximo 8)
  const shuffled = [...allProducts].sort(() => Math.random() - 0.5);
  const randomProducts = shuffled.slice(0, 8);

  container.innerHTML = randomProducts.map(p => {
    const dbData = IKEA_DB[p.key] || p;
    const displayName = dbData.nombre || dbData.key || 'Producto';
    const priceStr = dbData.priceStr || formatPrice(dbData.price);
    const descStr = dbData.subtitulo || dbData.desc || '';
    const emojiStr = dbData.emoji || '📦';
    const imageStr = dbData.image || '';
    const locationStr = dbData.location || 'Consultar';

    return `
      <div class="product-card" style="min-width:140px;flex-direction:column;gap:6px;" onclick="openProduct('${p.key}')">
        ${imageStr ? `<img src="${imageStr}" alt="${displayName}" style="width:60px;height:60px;object-fit:contain;border-radius:8px;" onerror="this.style.display='none';this.nextElementSibling.style.display='flex';"><div class="product-img" style="display:none;width:60px;height:60px;">${emojiStr}</div>` : `<div class="product-img" style="width:60px;height:60px;">${emojiStr}</div>`}
        <div style="font-size:11px;font-weight:700;color:var(--text);text-align:center;">${displayName}</div>
        <div style="font-size:10px;color:var(--gray);text-align:center;">${descStr.substring(0, 25)}${descStr.length > 25 ? '...' : ''}</div>
        <div style="font-size:12px;font-weight:900;color:var(--blue);text-align:center;">${priceStr}</div>
        <div style="font-size:9px;color:var(--green);text-align:center;">📍 ${locationStr}</div>
      </div>
    `;
  }).join('');
}

// ═══════════════════════════════ AR SUGGESTED PRODUCTS ═══════════════════════════════

function renderARSuggestedProducts() {
  const container = document.getElementById('ar-suggested-products');
  if (!container) return;

  if (!catalogLoaded || !allProducts.length) {
    container.innerHTML = '<div style="text-align:center;padding:20px;color:var(--gray);">Cargando...</div>';
    return;
  }

  // Seleccionar 4 productos aleatorios
  const shuffled = [...allProducts].sort(() => Math.random() - 0.5);
  const randomProducts = shuffled.slice(0, 4);

  container.innerHTML = randomProducts.map(p => {
    const dbData = IKEA_DB[p.key] || p;
    const displayName = dbData.nombre || dbData.key || 'Producto';
    const priceStr = dbData.priceStr || formatPrice(dbData.price);
    const descStr = dbData.subtitulo || dbData.desc || '';
    const emojiStr = dbData.emoji || '📦';
    const imageStr = dbData.image || '';
    const locationStr = dbData.location || 'Consultar';

    return `
      <div class="product-card" style="min-width:160px;flex-direction:column;gap:4px;" onclick="openProduct('${p.key}')">
        ${imageStr ? `<img src="${imageStr}" alt="${displayName}" style="width:50px;height:50px;object-fit:contain;border-radius:8px;" onerror="this.style.display='none';this.nextElementSibling.style.display='flex';"><div class="product-img" style="display:none;width:50px;height:50px;">${emojiStr}</div>` : `<div class="product-img" style="width:50px;height:50px;">${emojiStr}</div>`}
        <div style="font-size:10px;font-weight:700;color:var(--text);">${displayName}</div>
        <div style="font-size:9px;color:var(--gray);">${descStr.substring(0, 20)}${descStr.length > 20 ? '...' : ''}</div>
        <div style="font-size:11px;font-weight:900;color:var(--blue);">${priceStr}</div>
      </div>
    `;
  }).join('');
}

// ═══════════════════════════════ AUTH ═══════════════════════════════

function doLogin() {
  const email = document.getElementById('login-email').value.trim();
  const pass  = document.getElementById('login-pass').value.trim();
  if (!email || !pass) { showToast('⚠️ Rellena todos los campos'); return; }
  userProfile.email = email;
  userProfile.name  = email.split('@')[0];
  goTo('inicio');
}
function doRegister() {
  const name  = document.getElementById('reg-name').value.trim();
  const email = document.getElementById('reg-email').value.trim();
  const pass  = document.getElementById('reg-pass').value.trim();
  if (!name||!email||!pass) { showToast('⚠️ Rellena todos los campos'); return; }
  if (pass.length < 8)      { showToast('⚠️ Contraseña mínimo 8 caracteres'); return; }
  userProfile = { name, email, phone:'+34 600 000 000', bday:'-- / -- / ----' };
  showToast('✅ ¡Cuenta creada!');
  goTo('inicio');
}

// ═══════════════════════════════ PRODUCT DETAIL ═══════════════════════════════

function openProduct(idOrName) {
  let p = null;
  let productKey = null;

  if (typeof idOrName === 'number') {
    // Buscar por ID
    p = allProducts.find(prod => prod.id === idOrName);
    if (p) productKey = p.key;
  } else {
    // Buscar por key o nombre usando la nueva función
    p = findProduct(idOrName);
    if (p) productKey = p.key;
  }

  if (!p) {
    showToast('Producto no encontrado');
    console.error('Producto no encontrado:', idOrName);
    return;
  }

  currentProductId = p.id;
  currentProductKey = productKey;

  // Obtener datos completos del producto
  const dbData = p;

  // Mostrar imagen o emoji
  const heroEmoji = document.getElementById('prod-hero-emoji');
  const heroImage = document.getElementById('prod-hero-image');
  if (dbData.image) {
    heroEmoji.style.display = 'none';
    heroImage.src = dbData.image;
    heroImage.style.display = 'block';
    heroImage.onerror = function() {
      this.style.display = 'none';
      heroEmoji.style.display = 'flex';
      heroEmoji.textContent = dbData.emoji || '📦';
    };
  } else {
    heroEmoji.style.display = 'flex';
    heroEmoji.textContent = dbData.emoji || '📦';
    if (heroImage) heroImage.style.display = 'none';
  }

  document.getElementById('prod-header-title').textContent = dbData.nombre || dbData.key || 'Producto';
  document.getElementById('prod-name').textContent = dbData.nombre || dbData.key || '';
  document.getElementById('prod-desc').textContent = dbData.subtitulo || dbData.desc || '';
  document.getElementById('prod-price').textContent = dbData.priceStr || formatPrice(dbData.price);
  document.getElementById('prod-long-desc').textContent = dbData.descripcion || dbData.longDesc || dbData.subtitulo || '';

  const stock = dbData.stock || 10;
  document.getElementById('prod-badge').textContent = stock <= 3 ? '⚠️ POCAS UNIDADES' : 'EN TIENDA';
  document.getElementById('prod-badge').style.background = stock <= 3 ? 'var(--orange)' : 'var(--yellow)';

  // Pills
  document.getElementById('prod-pills').innerHTML = `
    <div class="info-pill"><span>📍</span><div><span class="pill-label">PASILLO</span><span class="pill-val">${dbData.location}</span></div></div>
    <div class="info-pill"><span>⚖️</span><div><span class="pill-label">PESO</span><span class="pill-val">${dbData.peso || dbData.weight || '-'}</span></div></div>
    <div class="info-pill"><span>📦</span><div><span class="pill-label">CATEGORÍA</span><span class="pill-val">${dbData.categoria || '-'}</span></div></div>
  `;

  // Fav button
  const isFav = favItems.some(f => f.key === productKey);
  document.getElementById('prod-fav-btn').textContent = isFav ? '⭐' : '☆';

  // Add to cart button
  const inCart = cartItems.some(c => c.key === productKey);
  const addBtn = document.getElementById('prod-actions').querySelector('.btn-primary');
  if (addBtn) {
    addBtn.textContent = inCart ? '✅ En la cesta' : '+ Añadir a la cesta';
    addBtn.className = 'btn ' + (inCart ? 'btn-success' : 'btn-primary');
  }

  // Related products
  const related = allProducts.filter(x => x.key !== productKey).slice(0, 4);
  document.getElementById('prod-related').innerHTML = related.map(r => `
    <div class="related-card" onclick="openProduct('${r.key}')">
      ${r.image ? `<img src="${r.image}" alt="${r.nombre}" style="width:40px;height:40px;object-fit:contain;border-radius:8px;">` : `<span class="related-emoji">${r.emoji || '📦'}</span>`}
      <div class="related-name">${r.nombre || r.key}</div>
      <div class="related-price">${r.priceStr || formatPrice(r.price)}</div>
    </div>
  `).join('');

  goTo('producto');
  
  refreshProductButtonState();
  
}

function addProductToCart() {
  if (!currentProductKey) return;
  const p = IKEA_DB[currentProductKey];
  if (!p) return;

  addToCart(currentProductKey, p.price, p.location, p.emoji || '📦', p.image);
  showToast(`✅ ${p.nombre || currentProductKey} añadido a la cesta`);
  const btn = document.getElementById('prod-actions').querySelector('button');
  if (btn) { btn.textContent = '✅ En la cesta'; btn.className = 'btn btn-success'; }
}

function toggleProductFav() {
  
  if (!currentProductKey) return;
  const p = IKEA_DB[currentProductKey];
  if (!p) return;

  const idx = favItems.findIndex(f => f.key === currentProductKey);
  const btn = document.getElementById('prod-fav-btn');

  if (idx >= 0) {
    favItems.splice(idx, 1);
    btn.textContent = '☆';
    showToast('✖️ Eliminado de Favoritos');
  } else {
    favItems.push({
      id: Date.now(),
      key: currentProductKey,
      name: p.nombre || currentProductKey,
      price: p.price,
      location: p.location,
      peso: p.peso,
      emoji: p.emoji,
      image: p.image,
      inCart: cartItems.some(c => c.key === currentProductKey)
    });
    btn.textContent = '⭐';
    showToast('⭐ Guardado en Favoritos');
  }
}

// ═══════════════════════════════ CART ═══════════════════════════════

function renderCart() {
  const list = document.getElementById('cart-list');
  if (!list) return;
  if (!cartItems.length) {
    list.innerHTML = '<div style="text-align:center;padding:48px 20px;color:var(--gray);"><div style="font-size:48px;margin-bottom:12px;">🛒</div><div style="font-weight:700;">Tu cesta está vacía</div></div>';
    updateTotal();
    return;
  }
  list.innerHTML = cartItems.map(item => {
    // Obtener datos del catálogo o usar datos del item
    const dbData = IKEA_DB[item.key] || {};
    const displayName = item.name || dbData.nombre || item.key || 'Producto';
    const priceStr = item.priceStr || dbData.priceStr || formatPrice(item.price);
    const descStr = item.desc || dbData.subtitulo || dbData.desc || '';
    const emojiStr = item.emoji || dbData.emoji || '📦';
    const imageStr = item.image || dbData.image || '';
    const weightStr = item.peso || dbData.peso || item.weight || dbData.weight || '-';

    return `
      <div class="product-card" id="cart-item-${item.id}" style="flex-direction:column;gap:0;" onclick="openProduct('${item.key}')">
        <div style="display:flex;gap:11px;align-items:flex-start;">
          ${imageStr ? `<img src="${imageStr}" alt="${displayName}" style="width:50px;height:50px;object-fit:contain;border-radius:8px;flex-shrink:0;" onerror="this.style.display='none';this.nextElementSibling.style.display='flex';"><div class="product-img" style="display:none;">${emojiStr}</div>` : `<div class="product-img">${emojiStr}</div>`}
          <div style="flex:1;">
            <div class="product-name">${displayName}</div>
            <div class="product-desc">${descStr}</div>
            <div style="display:flex;gap:5px;margin-top:5px;flex-wrap:wrap;">
              <span class="product-location">📍 ${item.location}</span>
              <span class="product-location" style="background:rgba(82,32,125,0.1);color:var(--purple);">⚖️ ${weightStr}</span>
            </div>
          </div>
          <div style="text-align:right;flex-shrink:0;"><div class="product-price">${priceStr}</div></div>
        </div>
        <div style="display:flex;align-items:center;justify-content:space-between;gap:10px;margin-top:10px;padding-top:10px;border-top:1px solid var(--border);" onclick="event.stopPropagation()">
          <div style="display:flex;gap:8px;align-items:center;">
            <button style="width:36px;height:36px;border-radius:10px;border:none;background:rgba(239,68,68,0.1);color:var(--red);font-size:16px;cursor:pointer;display:flex;align-items:center;justify-content:center;flex-shrink:0;transition:all 0.15s;" onclick="removeFromCart(${item.id})" title="Eliminar">🗑️</button>
            <button style="width:36px;height:36px;border-radius:10px;border:none;background:${item.inFav ? 'var(--yellow)' : 'rgba(255,219,0,0.18)'};color:#9a6500;font-size:16px;cursor:pointer;display:flex;align-items:center;justify-content:center;flex-shrink:0;transition:all 0.15s;" onclick="toggleFavFromCart(${item.id})" title="Favorito">${item.inFav ? '⭐' : '☆'}</button>
            <button style="width:36px;height:36px;border-radius:10px;border:none;background:var(--bg);color:var(--blue);font-size:16px;cursor:pointer;display:flex;align-items:center;justify-content:center;flex-shrink:0;transition:all 0.15s;" onclick="goTo('mapa');showToast('📍 ${displayName}')" title="Ver en mapa">📍</button>
          </div>
          <div class="qty-controls">
            <button class="qty-btn" onclick="changeQty(${item.id},-1)">−</button>
            <span class="qty-num" id="qty-${item.id}">${item.qty}</span>
            <button class="qty-btn" onclick="changeQty(${item.id},1)">+</button>
          </div>
        </div>
      </div>
    `;
  }).join('');
  updateTotal();
  updateRouteFromCart();
}

function addToCart(key, price, location, emoji, image) {
  const existing = cartItems.find(i => i.key === key);
  if (existing) {
    existing.qty++;
  } else {
    const dbData = IKEA_DB[key] || {};
    cartItems.push({
      id: Date.now(),
      key: key,
      name: dbData.nombre || key.split('|')[0].trim(),
      price: Number(price) || dbData.price || 0,
      priceStr: dbData.priceStr || formatPrice(price),
      location: location || dbData.location || 'Consultar',
      peso: dbData.peso || dbData.weight || '-',
      emoji: emoji || dbData.emoji || '📦',
      image: image || dbData.image || '',
      desc: dbData.subtitulo || dbData.desc || '',
      qty: 1,
      inFav: false
    });
  }
  updateAllBadges();
  renderCart();
}

function removeFromCart(id) {
  const el = document.getElementById('cart-item-'+id);
  if (el) { el.style.opacity='0'; el.style.transform='translateX(-100%)'; el.style.transition='all 0.28s'; }
  setTimeout(() => { cartItems = cartItems.filter(i=>i.id!==id); renderCart(); }, 300);
  updateAllBadges();
  updateRouteFromCart();
}
function changeQty(id, delta) {
  const item = cartItems.find(i=>i.id===id); if(!item) return;
  item.qty = Math.max(1, item.qty+delta);
  document.getElementById('qty-'+id).textContent = item.qty;
  updateTotal();
}
function toggleFavFromCart(id) {
  const item = cartItems.find(i=>i.id===id); if(!item) return;
  item.inFav = !item.inFav;

  // Sincronizar con favItems
  const favIdx = favItems.findIndex(f => f.key === item.key);
  if (item.inFav && favIdx < 0) {
    favItems.push({
      id: Date.now(),
      key: item.key,
      name: item.name,
      price: item.price,
      location: item.location,
      peso: item.peso,
      emoji: item.emoji,
      image: item.image,
      inCart: true
    });
  } else if (!item.inFav && favIdx >= 0) {
    favItems.splice(favIdx, 1);
  }

  renderCart();
  showToast(item.inFav ? '⭐ Guardado en Favoritos' : '✖️ Eliminado de Favoritos');
}
function updateTotal() {
  const subtotal = cartItems.reduce((s,i)=>s+i.price*i.qty, 0);
  const total = subtotal + 1;
  const fmt = n => n.toFixed(2).replace('.',',')+' €';
  ['subtotal-val','pay-subtotal'].forEach(id => { const el=document.getElementById(id); if(el) el.textContent=fmt(subtotal); });
  ['total-val','pay-total'].forEach(id => { const el=document.getElementById(id); if(el) el.textContent=fmt(total); });
  const pb = document.getElementById('pay-confirm-btn');
  if (pb) pb.textContent = 'Confirmar y pagar · '+fmt(total);
}
function updateAllBadges() {
  const count = cartItems.reduce((s,i)=>s+i.qty, 0);
  document.querySelectorAll('.cart-badge').forEach(b => {
    b.textContent=count; b.classList.add('badge-animate');
    setTimeout(()=>b.classList.remove('badge-animate'),300);
  });
}
function clearCart() { cartItems=[]; updateAllBadges(); renderCart(); updateRouteFromCart(); }
function sortCart(btn, mode) {
    // 1. Manejo del estilo visual de los botones
    document.querySelectorAll('#screen-cesta .aisle-btn').forEach(b => b.classList.remove('active'));
    
    if (btn) {
        // Si hemos hecho clic en un botón físico, lo marcamos
        btn.classList.add('active');
    } else {
        // Si viene por voz, buscamos el botón correspondiente en el HTML y lo marcamos
        const targetBtn = document.querySelector(`#screen-cesta .aisle-btn[onclick*="${mode}"]`);
        if (targetBtn) targetBtn.classList.add('active');
    }

    // 2. Lógica de ordenado a prueba de fallos
    if (mode === 'price') {
        // De mayor a menor precio
        cartItems.sort((a, b) => (b.price || 0) - (a.price || 0));
    } else if (mode === 'weight') {
        // De mayor a menor peso (si prefieres de menor a mayor, cambia el orden de a y b)
        cartItems.sort((a, b) => (b.weight || 0) - (a.weight || 0));
    } else if (mode === 'az') {
        // El || '' evita que crashee si un producto no tiene nombre
        cartItems.sort((a, b) => (a.name || '').localeCompare(b.name || ''));
    } else if (mode === 'route') {
        // El || '' evita el crasheo que tenías con las ubicaciones vacías
        cartItems.sort((a, b) => (a.location || '').localeCompare(b.location || ''));
    }
    
    // 3. Renderizamos la cesta actualizada
    if (typeof renderCart === 'function') renderCart();
}

// Generar ruta desde el carrito
function updateRouteFromCart() {
  routeItems = cartItems.map((item, idx) => {
    const dbData = IKEA_DB[item.key] || {};
    return {
      step: idx + 1,
      key: item.key,
      name: item.name || dbData.nombre || item.key,
      location: item.location,
      done: false,
      emoji: item.emoji || dbData.emoji || '📦'
    };
  });
}

// ═══════════════════════════════ FAVS ═══════════════════════════════

function renderFavs() {
  const list = document.getElementById('fav-list');
  if (!list) return;
  if (!catalogLoaded) {
    list.innerHTML = '<div style="text-align:center;padding:48px 20px;color:var(--gray);"><div style="font-size:48px;margin-bottom:12px;">⭐</div><div style="font-weight:700;">Cargando favoritos...</div></div>';
    return;
  }
  if (!favItems.length) {
    list.innerHTML = '<div style="text-align:center;padding:48px 20px;color:var(--gray);"><div style="font-size:48px;margin-bottom:12px;">⭐</div><div style="font-weight:700;">Sin favoritos todavía</div></div>';
    return;
  }
  list.innerHTML = favItems.map(item => {
    const dbData = IKEA_DB[item.key] || {};
    const displayName = item.name || dbData.nombre || item.key || 'Producto';
    const priceStr = item.priceStr || dbData.priceStr || formatPrice(item.price);
    const descStr = dbData.subtitulo || dbData.desc || item.desc || '';
    const emojiStr = item.emoji || dbData.emoji || '📦';
    const imageStr = item.image || dbData.image || '';

    return `
      <div class="fav-product-card">
        <div class="fav-card-top" onclick="openProduct('${item.key}')" style="cursor:pointer;">
          ${imageStr ? `<img src="${imageStr}" alt="${displayName}" style="width:44px;height:44px;object-fit:contain;border-radius:8px;flex-shrink:0;" onerror="this.style.display='none';this.nextElementSibling.style.display='flex';"><div class="product-img" style="display:none;">${emojiStr}</div>` : `<div class="product-img">${emojiStr}</div>`}
          <div style="flex:1;">
            <div class="product-name">${displayName}</div>
            <div class="product-desc">${descStr}</div>
            <span class="product-location">📍 ${item.location || dbData.location || 'Consultar'}</span>
          </div>
          <div style="text-align:right;"><div class="product-price">${priceStr}</div></div>
        </div>
        <div style="display:flex;gap:7px;">
          <button class="btn ${item.inCart ? 'btn-success' : 'btn-primary'}" style="flex:2;"
            onclick="${item.inCart ? '' : `addToCart('${item.key}',${item.price || dbData.price || 0},'${item.location || dbData.location || 'Consultar'}','${emojiStr}','${imageStr}');this.textContent='✅ En cesta';this.className='btn btn-success';`}">
            ${item.inCart ? '✅ En cesta' : '+ Añadir a la cesta'}
          </button>
          <button class="btn btn-danger" onclick="removeFav(${item.id})">🗑️</button>
        </div>
      </div>
    `;
  }).join('');
}
function removeFav(id) { favItems=favItems.filter(i=>i.id!==id); renderFavs(); showToast('✖️ Eliminado de Favoritos'); }

// ═══════════════════════════════ SEARCH ═══════════════════════════════

function renderSearch(query) {
  const list = document.getElementById('search-results');
  if (!list) return;
  if (!catalogLoaded || !allProducts.length) {
    list.innerHTML = '<div style="text-align:center;padding:40px;color:var(--gray);">Cargando catálogo...</div>';
    return;
  }

  let results = [];

  if (!query || query.trim() === '') {
    // Si no hay búsqueda: desordenamos una copia del array y cogemos solo 20
    const shuffled = [...allProducts].sort(() => 0.5 - Math.random());
    results = shuffled.slice(0, 20);
  } else {
    // Si hay búsqueda: filtramos sobre los +6000 productos
    const lowerQuery = query.toLowerCase();
    results = allProducts.filter(p =>
      (p.name && p.name.toLowerCase().includes(lowerQuery)) ||
      (p.desc && p.desc.toLowerCase().includes(lowerQuery)) ||
      (IKEA_DB[p.name]?.desc && IKEA_DB[p.name].desc.toLowerCase().includes(lowerQuery))
    );

    results = results.slice(0, 20); 
  }

  list.innerHTML = results.map(p => {
    const dbData = IKEA_DB[p.key] || p;
    const priceStr = dbData.priceStr || formatPrice(dbData.price);
    const locationStr = dbData.location || 'Consultar';
    const weightStr = dbData.peso || dbData.weight || '-';
    const descStr = dbData.subtitulo || dbData.desc || '';
    const emojiStr = dbData.emoji || '📦';
    const imageStr = dbData.image || '';
    const displayName = dbData.nombre || dbData.key || 'Producto';

    return `
      <div class="product-card" style="flex-direction:column;gap:7px;" onclick="openProduct('${p.key}')">
        <div style="display:flex;gap:11px;align-items:center;">
          ${imageStr ? `<img src="${imageStr}" alt="${displayName}" style="width:50px;height:50px;object-fit:contain;border-radius:8px;flex-shrink:0;" onerror="this.style.display='none';this.nextElementSibling.style.display='flex';"><div class="product-img" style="display:none;">${emojiStr}</div>` : `<div class="product-img">${emojiStr}</div>`}
          <div style="flex:1;">
            <div class="product-name">${displayName}</div>
            <div class="product-desc">${descStr}</div>
            <span class="product-location">📍 ${locationStr} · ⚖️ ${weightStr}</span>
          </div>
          <div class="product-price">${priceStr}</div>
        </div>
        <div style="display:flex;gap:5px;" onclick="event.stopPropagation()">
          <button class="btn btn-primary" style="flex:2;" onclick="event.stopPropagation();addToCart('${p.key}',${dbData.price || 0},'${locationStr}','${emojiStr}','${imageStr}');showToast('✅ ${displayName} añadido')">+ Añadir</button>
          <button class="btn btn-outline" onclick="event.stopPropagation();openProduct('${p.key}')">Ver</button>
        </div>
      </div>
    `;
  }).join('') || `<div style="text-align:center;padding:40px;color:var(--gray);">Sin resultados para "${query}"</div>`;
}
function filterProducts(q) { renderSearch(q); }

function renderShopping() {
  const list = document.getElementById('shopping-list');
  if (!list) return;
  if (!catalogLoaded || !allProducts.length) {
    list.innerHTML = '<div style="text-align:center;padding:40px;color:var(--gray);">Cargando catálogo...</div>';
    return;
  }

  // 1. Clocar el array, desordenarlo y quedarnos solo con 20 productos para el escaparate
  const shuffled = [...allProducts].sort(() => 0.5 - Math.random());
  const randomProducts = shuffled.slice(0, 20);

  // 2. Mapear SOLO esos 20 productos aleatorios
  list.innerHTML = randomProducts.map(p => {
    const dbData = IKEA_DB[p.key] || p;
    const priceStr = dbData.priceStr || formatPrice(dbData.price);
    const locationStr = dbData.location || 'Consultar';
    const descStr = dbData.subtitulo || dbData.desc || '';
    const emojiStr = dbData.emoji || '📦';
    const imageStr = dbData.image || '';
    const displayName = dbData.nombre || dbData.key || 'Producto';

    return `
      <div class="product-card" onclick="openProduct('${p.key}')">
        ${imageStr ? `<img src="${imageStr}" alt="${displayName}" style="width:60px;height:60px;object-fit:contain;border-radius:8px;flex-shrink:0;" onerror="this.style.display='none';this.nextElementSibling.style.display='flex';" onload="this.style.display='block';this.nextElementSibling.style.display='none';"><div class="product-img" style="display:none;">${emojiStr}</div>` : `<div class="product-img">${emojiStr}</div>`}
        <div class="product-info">
          <div class="product-name">${displayName}</div>
          <div class="product-desc">${descStr}</div>
          <span class="product-location">📍 ${locationStr}</span>
          <div class="product-actions" onclick="event.stopPropagation()">
            <button class="btn btn-primary" onclick="addToCart('${p.key}',${dbData.price || 0},'${locationStr}','${emojiStr}','${imageStr}');showToast('✅ ${displayName} añadido')">+ Añadir</button>
          </div>
        </div>
        <div style="text-align:right;flex-shrink:0;"><div class="product-price">${priceStr}</div></div>
      </div>
    `;
  }).join('');
}

// ═══════════════════════════════ PAY ═══════════════════════════════

function renderPaySummary() {
  const list = document.getElementById('pay-summary-list');
  if (!list) return;
  list.innerHTML = cartItems.map(item => {
    const dbData = IKEA_DB[item.key] || item;
    const displayName = item.name || dbData.nombre || item.key || 'Producto';
    const emojiStr = item.emoji || dbData.emoji || '📦';
    const imageStr = item.image || dbData.image || '';
    return `
      <div class="pay-item" onclick="openProduct('${item.key}')">
        ${imageStr ? `<img src="${imageStr}" alt="${displayName}" style="width:40px;height:40px;object-fit:contain;border-radius:6px;margin-right:10px;" onerror="this.style.display='none';">` : `<div class="pay-item-img">${emojiStr}</div>`}
        <div class="pay-item-info"><div class="pay-item-name">${displayName} x${item.qty}</div><div class="pay-item-loc">📍 ${item.location}</div></div>
        <div class="pay-item-price">${formatPrice(item.price * item.qty)}</div>
      </div>
    `;
  }).join('');
  updateTotal();
}
function selectPayment(el) { document.querySelectorAll('.payment-option').forEach(o=>o.classList.remove('selected')); el.classList.add('selected'); }
function confirmPayment() {
  const btn = document.getElementById('pay-confirm-btn');
  btn.textContent = '⏳ Procesando...';
  btn.disabled = true;

  setTimeout(() => {
    const MONTHS = ['ENE','FEB','MAR','ABR','MAY','JUN','JUL','AGO','SEP','OCT','NOV','DIC'];
    const now      = new Date();
    const subtotal = cartItems.reduce((s, i) => s + i.price * i.qty, 0);
    const bagFee   = cartItems.length > 0 ? 1 : 0;
    const total    = subtotal + bagFee;
    const nItems   = cartItems.reduce((s, i) => s + i.qty, 0);
    const orderId  = '#IK-' + now.getFullYear() + '-' + String(orderHistory.length + 1001).padStart(4, '0');

    if (!cartItems.length) {
      btn.textContent = 'Confirmar y pagar · 0,00 €';
      btn.disabled = false;
      showToast('⚠️ La cesta está vacía');
      return;
    }

    orderHistory.push({
      id: orderId,
      day: now.getDate(),
      month: MONTHS[now.getMonth()],
      year: now.getFullYear(),
      items: cartItems.map(i => ({ ...i })),
      subtotal,
      bagFee,
      total,
      nItems
    });

    const oid  = document.getElementById('success-order-id');
    if (oid) oid.textContent = orderId;
    const osum = document.getElementById('success-order-summary');
    if (osum) osum.textContent = `${formatPrice(total)} · ${nItems} artículo${nItems !== 1 ? 's' : ''} · IKEA Bilbao`;

    btn.disabled = false;
    renderHistorial();
    renderProfile();
    goTo('success');
  }, 1800);
}

// ═══════════════════════════════ HISTORIAL ═══════════════════════════════

function renderHistorial() {
  // --- Estadísticas globales ---
  const totalVisits = orderHistory.length;
  const totalItems  = orderHistory.reduce((s, o) => s + o.nItems, 0);
  const totalSpent  = orderHistory.reduce((s, o) => s + o.total, 0);

  const elVisits = document.getElementById('hist-visits');
  const elItems  = document.getElementById('hist-items');
  const elSpent  = document.getElementById('hist-spent');
  if (elVisits) elVisits.textContent = totalVisits;
  if (elItems)  elItems.textContent  = totalItems;
  if (elSpent)  elSpent.textContent  = Math.round(totalSpent) + '€';

  // --- Lista de pedidos ---
  const listEl = document.getElementById('hist-orders-list');
  if (!listEl) return;

  if (!orderHistory.length) {
    listEl.innerHTML = `
      <div style="text-align:center;padding:40px;color:var(--gray);">
        <div style="font-size:48px;margin-bottom:12px;">📋</div>
        <div style="font-weight:700;">Sin pedidos todavía</div>
        <div style="font-size:12px;margin-top:4px;">Tus compras aparecerán aquí</div>
      </div>`;
    return;
  }

  // Mostrar los pedidos del más reciente al más antiguo
  listEl.innerHTML = [...orderHistory].reverse().map(order => {
    const itemsHtml = order.items.map(item => {
      const d = IKEA_DB[item.key] || {};
      const name     = item.name || d.nombre || item.key || 'Producto';
      const imageStr = item.image || d.image || '';
      const emojiStr = item.emoji || d.emoji || '📦';
      return `
        <div style="padding:9px 14px;display:flex;gap:10px;align-items:center;border-bottom:1px solid var(--border);">
          ${imageStr
            ? `<img src="${imageStr}" alt="${name}" style="width:42px;height:42px;object-fit:contain;border-radius:9px;flex-shrink:0;" onerror="this.style.display='none';">`
            : `<div style="width:42px;height:42px;border-radius:9px;background:#dde;display:flex;align-items:center;justify-content:center;font-size:20px;flex-shrink:0;">${emojiStr}</div>`}
          <div style="flex:1;min-width:0;">
            <div style="font-size:12.5px;font-weight:700;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">${name}${item.qty > 1 ? ' x' + item.qty : ''}</div>
            <div style="font-size:10.5px;color:var(--gray);">📍 ${item.location || 'Consultar'}</div>
          </div>
          <div style="font-size:13px;font-weight:900;flex-shrink:0;">${formatPrice(item.price * item.qty)}</div>
        </div>`;
    }).join('');

    return `
      <div style="background:var(--card);border-radius:16px;margin:0 14px 10px;overflow:hidden;box-shadow:0 2px 8px rgba(0,0,0,0.07);">
        <!-- Cabecera del pedido -->
        <div style="padding:11px 14px;background:rgba(0,88,163,0.05);display:flex;align-items:center;gap:11px;border-bottom:1px solid var(--border);">
          <div style="width:48px;height:48px;border-radius:11px;background:var(--blue);color:white;display:flex;flex-direction:column;align-items:center;justify-content:center;font-size:9.5px;font-weight:900;flex-shrink:0;">
            <span style="font-size:17px;font-weight:900;line-height:1;">${order.day}</span>
            <span>${order.month}</span>
          </div>
          <div style="flex:1;">
            <div style="font-size:13.5px;font-weight:700;">Pedido ${order.id}</div>
            <div style="font-size:10.5px;color:var(--gray);margin-top:2px;">${order.nItems} artículo${order.nItems !== 1 ? 's' : ''}</div>
          </div>
          <div style="font-size:14px;font-weight:900;color:var(--blue);">${formatPrice(order.total)}</div>
        </div>
        <!-- Líneas de productos -->
        ${itemsHtml}
        <!-- Totales -->
        <div style="padding:8px 14px;display:flex;justify-content:space-between;font-size:12px;color:var(--gray);border-top:1px solid var(--border);">
          <span>Subtotal</span><span style="font-weight:700;color:var(--text);">${formatPrice(order.subtotal)}</span>
        </div>
        <div style="padding:8px 14px 10px;display:flex;justify-content:space-between;font-size:12px;color:var(--gray);">
          <span>Bolsa IKEA</span><span style="font-weight:700;color:var(--text);">1,00 €</span>
        </div>
      </div>`;
  }).join('');
}
let computedRouteData = [];
let mapActiveFilter = 'todos';

function renderRoute() {
    computeShortestRoute();
    renderMapSectionHighlights();
    renderMapPinsAndPath();
    renderOrderedRouteList();
}

// Filtros del mapa
function setAisle(btn, filter) {
    mapActiveFilter = filter;
    document.querySelectorAll('#screen-mapa .aisle-btn').forEach(b => b.classList.remove('active'));
    if (btn) btn.classList.add('active');
    renderRoute();
}

function getFilteredRouteData() {
    if (mapActiveFilter === 'todos') return computedRouteData;
    return computedRouteData.filter(item => item.aisle === mapActiveFilter);
}

function renderMapSectionHighlights() {
    const sections = document.querySelectorAll('#screen-mapa .map-section, #screen-mapa .corridor');
    sections.forEach(section => section.classList.remove('highlighted'));

    if (mapActiveFilter === 'todos') return;

    document.querySelectorAll('#screen-mapa .corridor').forEach(corridor => {
        if ((corridor.textContent || '').trim() === mapActiveFilter) {
            corridor.classList.add('highlighted');
        }
    });
}

function computeShortestRoute() {
    const items = (typeof cartItems !== 'undefined') ? cartItems : [];

    if (!items || items.length === 0) {
        computedRouteData = [];
        updateRouteStats();
        return;
    }

    // 1. Preparar los productos con sus coordenadas finales
    const xCoords = { A: 55.5, B: 67.5, C: 79.5, D: 91.5 };
    let pending = items.map(item => {
        const dbData = IKEA_DB[item.key] || {};
        const shelf = parseInt(dbData.estanteria) || 1;
        return {
            key: item.key,
            name: dbData.nombre || item.name || item.key.split('|')[0],
            price: item.price || dbData.price || 0,
            aisle: (dbData.pasillo || 'A').toUpperCase(),
            shelf: shelf,
            recogido: localStorage.getItem(`recogido_${item.key}`) === 'true',
            coords: {
                xPerc: xCoords[(dbData.pasillo || 'A').toUpperCase()] || 55.5,
                // AJUSTE: Inicio 8% + Rango 80% (Termina en 88%)
                yPerc: 8 + ((shelf - 1) * (80 / 19))
            }
        };
    });

    // 2. ALGORITMO POR DISTANCIA
    // Empezamos en la posición de la entrada (Almacén)
    
    let currentPos = { x: 50, y: 15 }; 
    let orderedRoute = [];

    while (pending.length > 0) {
        let nearestIndex = -1;
        let minDistance = Infinity;

        pending.forEach((prod, index) => {
            // Calculamos distancia Manhattan (pasillos + profundidad)
            const dist = Math.abs(currentPos.x - prod.coords.xPerc) + 
                         Math.abs(currentPos.y - prod.coords.yPerc);

            if (dist < minDistance) {
                minDistance = dist;
                nearestIndex = index;
            }
        });

        // Extraemos el más cercano y actualizamos posición actual
        const nextProd = pending.splice(nearestIndex, 1)[0];
        orderedRoute.push(nextProd);
        currentPos = { x: nextProd.coords.xPerc, y: nextProd.coords.yPerc };
    }

    computedRouteData = orderedRoute;
    updateRouteStats();
}

function updateRouteStats() {
    const visibleRoute = getFilteredRouteData();
    const totalProducts = visibleRoute.length;
    const totalPrice = visibleRoute.reduce((sum, p) => sum + (p.price || 0), 0);

    // Ej: 1.5 min por producto + base
    const estimatedTime = totalProducts > 0 ? Math.round(totalProducts * 1.5 + 2) : 0;

    document.getElementById('route-prod-count').textContent = totalProducts;
    document.getElementById('route-time').textContent = estimatedTime;
    document.getElementById('route-total').textContent =
        totalPrice.toFixed(2).replace('.', ',') + ' €';
}

function renderMapPinsAndPath() {
    const mapArea = document.getElementById('store-map-area');
    const svgArea = document.getElementById('map-route-svg');

    if (!mapArea || !svgArea) return;

    mapArea.querySelectorAll('.dynamic-route-dot').forEach(el => el.remove());
    svgArea.innerHTML = '';

    const visibleRoute = getFilteredRouteData();

    if (!visibleRoute.length) {
        return;
    }

    const startPos = { xPerc: 10, yPerc: 90 };
    const endPos = { xPerc: 50, yPerc: 98 };

    let routePoints = [`${startPos.xPerc}%,${startPos.yPerc}%`];
    let currentPos = startPos;

    visibleRoute.forEach((r, i) => {
        const pin = document.createElement('div');
        pin.className = `route-dot dynamic-route-dot ${r.recogido ? 'status-done' : ''}`;
        pin.textContent = r.recogido ? '✓' : (i + 1);
        pin.style.left = `${r.coords.xPerc}%`;
        pin.style.top = `${r.coords.yPerc}%`;
        pin.title = `${r.name} · Pasillo ${r.aisle} · Estantería ${r.shelf}`;
        mapArea.appendChild(pin);

        routePoints.push(`${r.coords.xPerc}%,${currentPos.yPerc}%`);
        routePoints.push(`${r.coords.xPerc}%,${r.coords.yPerc}%`);
        currentPos = r.coords;
    });

    routePoints.push(`${endPos.xPerc}%,${currentPos.yPerc}%`);
    routePoints.push(`${endPos.xPerc}%,${endPos.yPerc}%`);

    const productPolyline = document.createElementNS("http://www.w3.org/2000/svg", "polyline");
    productPolyline.setAttribute("points", routePoints.join(' '));
    productPolyline.setAttribute("fill", "none");
    productPolyline.setAttribute("stroke", "var(--blue, #0058a3)");
    productPolyline.setAttribute("stroke-width", "2.5");
    productPolyline.setAttribute("stroke-dasharray", "6 4");
    svgArea.appendChild(productPolyline);

    const personPathPoints = [];
    currentPos = startPos;
    personPathPoints.push(`${startPos.xPerc}%,${startPos.yPerc}%`);

    visibleRoute.forEach(r => {
        personPathPoints.push(`${currentPos.xPerc}%,${r.coords.yPerc}%`);
        personPathPoints.push(`${r.coords.xPerc}%,${r.coords.yPerc}%`);
        currentPos = r.coords;
    });

    personPathPoints.push(`${currentPos.xPerc}%,${endPos.yPerc}%`);
    personPathPoints.push(`${endPos.xPerc}%,${endPos.yPerc}%`);

    const personPolyline = document.createElementNS("http://www.w3.org/2000/svg", "polyline");
    personPolyline.setAttribute("points", personPathPoints.join(' '));
    personPolyline.setAttribute("fill", "none");
    personPolyline.setAttribute("stroke", "var(--green, #1f8423)");
    personPolyline.setAttribute("stroke-width", "3");
    personPolyline.setAttribute("stroke-dasharray", "10 6");
    personPolyline.setAttribute("stroke-linecap", "round");
    personPolyline.setAttribute("opacity", "0.9");
    svgArea.appendChild(personPolyline);
}

function renderOrderedRouteList() {
    const list = document.getElementById('route-list');
    if (!list) return;

    const visibleRoute = getFilteredRouteData();

    if (!computedRouteData.length) {
        list.innerHTML = '<div style="padding:20px;text-align:center;">Cesta vacía</div>';
        return;
    }

    if (!visibleRoute.length) {
        list.innerHTML = `<div style="padding:20px;text-align:center;color:var(--gray);">No hay productos en ${mapActiveFilter === 'todos' ? 'la ruta' : 'el pasillo ' + mapActiveFilter}</div>`;
        return;
    }

    list.innerHTML = visibleRoute.map((r, i) => {

        const cleanName = r.name.includes('|')
            ? r.name.split('|')[0].trim()
            : r.name;

        return `
        <div class="route-item-card">
            <div class="route-step-number ${r.recogido ? 'status-done' : ''}">
                ${r.recogido ? '✓' : (i + 1)}
            </div>

            <div style="flex:1;">
                <div style="font-weight:700;">${cleanName}</div>
                <div style="font-size:10px;color:var(--blue);">
                    📍 Pasillo ${r.aisle} · Estantería ${r.shelf}
                </div>
            </div>

            <div style="font-size:11px;font-weight:700;">
                ${(r.price || 0).toFixed(2).replace('.', ',')} €
            </div>
        </div>
        `;
    }).join('');
}

// AUX
function updateRouteHeader(count, time) {
    document.getElementById('route-prod-count').textContent = count;
    document.getElementById('route-time').textContent = time;
}

// ═══════════════════════════════ PERFIL ═══════════════════════════════

function renderProfile() {
  const totalVisits = orderHistory.length;
  const totalItems = orderHistory.reduce((sum, order) => sum + (order.nItems || 0), 0);
  const totalSpent = orderHistory.reduce((sum, order) => sum + (order.total || 0), 0);
  const totalPoints = Math.round(totalSpent * 10);

  document.getElementById('perfil-name-display').textContent  = userProfile.name;
  document.getElementById('perfil-email-display').textContent = userProfile.email;
  document.getElementById('family-name-display').textContent  = userProfile.name;
  document.getElementById('pf-name').textContent  = userProfile.name;
  document.getElementById('pf-email').textContent = userProfile.email;
  document.getElementById('pf-phone').textContent = userProfile.phone;
  document.getElementById('pf-bday').textContent  = userProfile.bday;
  document.getElementById('cfg-name').textContent  = userProfile.name;
  document.getElementById('cfg-email').textContent = userProfile.email;

  const perfilStats = document.querySelector('#screen-perfil .perfil-stats');
  if (perfilStats) {
    const statsHtml = [
      `<div class="perfil-stat-card"><span class="perfil-stat-emoji">🏬</span><div class="perfil-stat-val">${totalVisits}</div><div class="perfil-stat-lbl">VISITAS</div></div>`,
      `<div class="perfil-stat-card"><span class="perfil-stat-emoji">⭐</span><div class="perfil-stat-val" id="perfil-points">${totalPoints.toLocaleString('es-ES')}</div><div class="perfil-stat-lbl">PUNTOS</div></div>`,
      `<div class="perfil-stat-card"><span class="perfil-stat-emoji">💳</span><div class="perfil-stat-val">${formatPrice(totalSpent) === "Consultar" ? "0,00 €" : formatPrice(totalSpent)}</div><div class="perfil-stat-lbl">GASTADO</div></div>`
    ];

    perfilStats.innerHTML = statsHtml.join('');
    
  }

  const familyPoints = document.getElementById('family-points-display');
  if (familyPoints) {
    familyPoints.textContent = totalPoints.toLocaleString('es-ES');
  }

  const activityCard = document.getElementById('activity-card');
  
  if (activityCard) {
    activityCard.innerHTML = `
      <div class="perfil-field" onclick="goTo('historial')"><div class="perfil-field-icon">🧾</div><div class="perfil-field-info"><div class="perfil-field-label">HISTORIAL</div><div class="perfil-field-val">${orderHistory.length > 0 ? `Ver ${orderHistory.length} compra${orderHistory.length !== 1 ? 's' : ''} realizada${orderHistory.length !== 1 ? 's' : ''}` : 'Sin compras'}</div></div><span style="color:var(--gray);">›</span></div>
      <div class="perfil-field" onclick="navTo('favoritos')"><div class="perfil-field-icon">⭐</div><div class="perfil-field-info"><div class="perfil-field-label">FAVORITOS</div><div class="perfil-field-val">${favItems.length > 0 ? favItems.length : 'Sin'} artículo${favItems.length !== 1 ? 's' : ''} guardado${favItems.length !== 1 ? 's' : ''}</div></div><span style="color:var(--gray);">›</span></div>
    `;
  }
}

let editField = null;
const editConfigs = {
  name:  { title:'Editar nombre',         field:'name',  label:'NOMBRE',               type:'text',  placeholder:'Tu nombre completo' },
  email: { title:'Editar correo',         field:'email', label:'CORREO ELECTRÓNICO',   type:'email', placeholder:'ejemplo@correo.com' },
  phone: { title:'Editar teléfono',       field:'phone', label:'TELÉFONO',             type:'tel',   placeholder:'+34 600 000 000' },
  bday:  { title:'Fecha de nacimiento',   field:'bday',  label:'FECHA (DD / MM / AAAA)',type:'text', placeholder:'15 / 03 / 1990' }
};

function openEditSheet(field) {
  const cfg = field ? editConfigs[field] : null;
  editField = field || null;
  const title = cfg ? cfg.title : 'Editar perfil';
  const body  = cfg ? `
    <div class="form-group">
      <label class="form-label">${cfg.label}</label>
      <input class="form-input" id="edit-field-input" type="${cfg.type}" placeholder="${cfg.placeholder}" value="${userProfile[cfg.field]||''}">
    </div>
  ` : Object.entries(editConfigs).map(([k,c]) => `
    <div class="form-group">
      <label class="form-label">${c.label}</label>
      <input class="form-input edit-all-input" data-field="${k}" type="${c.type}" placeholder="${c.placeholder}" value="${userProfile[k]||''}">
    </div>
  `).join('');
  document.getElementById('edit-sheet-title').textContent = title;
  document.getElementById('edit-sheet-body').innerHTML    = body;
  document.getElementById('edit-overlay').classList.add('open');
  setTimeout(() => { const inp = document.getElementById('edit-field-input'); if(inp) inp.focus(); }, 300);
}

function closeEditSheet() { document.getElementById('edit-overlay').classList.remove('open'); }
function closeEditSheetOnBg(e) { if(e.target.id==='edit-overlay') closeEditSheet(); }

function saveEditSheet() {
  if (editField) {
    const inp = document.getElementById('edit-field-input');
    if (inp && inp.value.trim()) { userProfile[editField] = inp.value.trim(); showToast('✅ Cambios guardados'); }
  } else {
    document.querySelectorAll('.edit-all-input').forEach(inp => {
      if (inp.value.trim()) userProfile[inp.dataset.field] = inp.value.trim();
    });
    showToast('✅ Perfil actualizado');
  }
  closeEditSheet();
  renderProfile();
}

// ═══════════════════════════════ CAMERA ═══════════════════════════════

let streams = { ar: null, qr: null };
let arFacing = 'environment';
let qrDetected = false;
let qrScanning = false;
const AR_API = window.location.origin + '/identify';
// Canvas compartido para captura de frames
let captureCanvas = null;
let captureCtx = null;

async function startCamera(type) {
  const video = document.getElementById(type + '-video');
  const noCam = document.getElementById(type + '-no-cam');
  const statusEl = document.getElementById(type + '-status');

  // Detener cámara anterior
  stopCamera(type);

  // Crear canvas para captura si no existe
  if (!captureCanvas) {
    captureCanvas = document.createElement('canvas');
    captureCtx = captureCanvas.getContext('2d', { willReadFrequently: true });
  }

  if (type === 'ar') {
    const btn = document.getElementById('ar-capture-btn');
    if (btn) btn.disabled = true;
  }

  document.getElementById(type + '-camera-wrap').classList.add('expanded');

  try {
    const facing = type === 'ar' ? arFacing : 'environment';
    const mediaStream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: facing,
        width: { ideal: 1280 },
        height: { ideal: 720 }
      },
      audio: false
    });

    streams[type] = mediaStream;
    video.srcObject = mediaStream;
    await video.play();

    // Esperar a que el video esté listo
    await new Promise(resolve => {
      if (video.readyState >= 2) resolve();
      else video.onloadedmetadata = resolve;
    });

    noCam.style.display = 'none';

    if (type === 'ar') {
      const btn = document.getElementById('ar-capture-btn');
      if (btn) btn.disabled = false;
      if (statusEl) {
        statusEl.textContent = '📷 Centra el mueble';
        statusEl.className = 'cam-status';
      }
    }

    if (type === 'qr') {
      qrDetected = false;
      qrScanning = true;
      const qrDetectedEl = document.getElementById('qr-detected');
      if (qrDetectedEl) qrDetectedEl.style.display = 'none';
      if (statusEl) {
        statusEl.textContent = '📷 Buscando código QR...';
        statusEl.className = 'cam-status';
      }
      startQRScanning();
    }

  } catch (err) {
    console.error('Error accessing camera:', err);
    if (noCam) noCam.style.display = 'flex';
    if (statusEl) {
      statusEl.textContent = '⚠️ Sin cámara';
      statusEl.className = 'cam-status';
    }
  }
}

function stopCamera(type) {
  if (type === 'qr') {
    qrScanning = false;
    stopQRScanning();
  }

  if (streams[type]) {
    streams[type].getTracks().forEach(track => track.stop());
    streams[type] = null;
  }

  const video = document.getElementById(type + '-video');
  if (video) {
    video.srcObject = null;
    video.pause();
  }
}

// ─── QR SCAN ───

let qrAnimationFrameId = null;
let qrLastScanTime = 0;
const QR_SCAN_INTERVAL = 100; // ms entre scans

function startQRScanning() {
  if (qrAnimationFrameId) {
    cancelAnimationFrame(qrAnimationFrameId);
  }
  qrLastScanTime = 0;
  qrScanning = true;
  scanQRLoop();
}

function stopQRScanning() {
  qrScanning = false;
  if (qrAnimationFrameId) {
    cancelAnimationFrame(qrAnimationFrameId);
    qrAnimationFrameId = null;
  }
}

function scanQRLoop(timestamp) {
  if (!qrScanning || qrDetected || !streams['qr']) {
    qrAnimationFrameId = null;
    return;
  }

  if (timestamp - qrLastScanTime >= QR_SCAN_INTERVAL) {
    qrLastScanTime = timestamp;
    scanQRFrame();
  }

  qrAnimationFrameId = requestAnimationFrame(scanQRLoop);
}

function scanQRFrame() {
  if (qrDetected || !qrScanning) return;

  const video = document.getElementById('qr-video');
  if (!video || !video.srcObject || video.readyState < 2) return;

  try {
    let w = video.videoWidth;
    let h = video.videoHeight;

    if (!w || !h) {
      w = video.clientWidth || 320;
      h = video.clientHeight || 240;
    }

    if (!w || !h || w === 0 || h === 0) return;

    // Solo redimensionar si es necesario
    if (captureCanvas.width !== w || captureCanvas.height !== h) {
      captureCanvas.width = w;
      captureCanvas.height = h;
    }

    captureCtx.drawImage(video, 0, 0, w, h);

    const imageData = captureCtx.getImageData(0, 0, w, h);

    if (typeof jsQR !== 'undefined') {
      const code = jsQR(imageData.data, w, h, {
        inversionAttempts: 'dontInvert'
      });

      if (code && code.data) {
        qrDetected = true;
        handleQRDetected(code.data);
        return;
      }
    }
  } catch (e) {
    // Ignorar errores de contexto - son normales durante transiciones
  }
}

function handleQRDetected(data) {
  qrDetected = true;
  stopQRScanning();

  document.getElementById('qr-camera-wrap').classList.remove('expanded');

  const statusEl = document.getElementById('qr-status');
  if (statusEl) {
    statusEl.textContent = '✅ ¡Detectado!';
    statusEl.className = 'cam-status found';
  }

  const cleanData = data.trim();

  const product = Object.values(IKEA_DB).find(p => p.url === cleanData)
               ?? Object.values(IKEA_DB).find(p => p.url?.includes(cleanData))
               ?? Object.values(IKEA_DB).find(p => cleanData.includes(p.url))
               ?? findProduct(cleanData);

  const qrDetectedEl = document.getElementById('qr-detected');
  if (!qrDetectedEl) return;

  if (product) {
    const displayName = product.nombre || product.key || 'Producto';
    const priceStr    = product.priceStr || formatPrice(product.price);
    const locationStr = product.location || 'Consultar';
    const emojiStr    = product.emoji || '📦';
    const descStr     = product.subtitulo || product.desc || '';
    const imageStr    = product.image || '';
    const isFav       = favItems.some(f => f.key === product.key);

    qrDetectedEl.innerHTML = `
      <div class="qr-result">
        ${imageStr
          ? `<img src="${imageStr}" alt="${displayName}"
               style="width:54px;height:54px;object-fit:contain;border-radius:10px;flex-shrink:0;"
               onerror="this.style.display='none';this.nextElementSibling.style.display='flex';">
             <div class="product-img" style="display:none;border:2px solid var(--green);">${emojiStr}</div>`
          : `<div class="product-img" style="border:2px solid var(--green);">${emojiStr}</div>`
        }
        <div style="flex:1;min-width:0;cursor:pointer;" onclick="openProduct('${product.key}')">
          <div class="qr-result-name">${displayName}</div>
          <div class="qr-result-detail">${descStr}</div>
          <span class="product-location">📍 ${locationStr}</span>
        </div>
        <div style="text-align:right;flex-shrink:0;">
          <div class="product-price" style="color:var(--green);">${priceStr}</div>
          <button class="btn btn-primary"
                  style="margin-top:5px;padding:5px 9px;font-size:10.5px;"
                  onclick="addToCart('${product.key}',${product.price || 0},'${locationStr}','${emojiStr}','${imageStr}');showToast('✅ ${displayName} añadido')">
            + Cesta
          </button>
        </div>
      </div>
      <div style="display:flex;gap:8px;padding:0 14px 12px;">
        <button class="btn btn-fav" id="qr-fav-btn" onclick="qrToggleFav()">${isFav ? '⭐' : '☆'}</button>
        <button class="btn btn-outline" style="flex:1;" onclick="qrViewProduct()">Ver producto →</button>
        <button class="btn btn-outline" onclick="resetQRScanner()">🔄</button>
      </div>
    `;

    window._qrProduct = product;
    qrDetectedEl.style.display = 'block';
    qrDetectedEl.style.animation = 'fadeInUp 0.36s ease-out';
    showToast(`✅ ${displayName}`);

  } else {
    // No encontrado — tarjeta de aviso igual que AR
    qrDetectedEl.innerHTML = `
      <div class="qr-result" style="background:rgba(255,165,0,0.08);border-color:rgba(255,165,0,0.3);">
        <div class="product-img" style="border:2px solid var(--orange);">❓</div>
        <div style="flex:1;">
          <div class="qr-result-name" style="color:var(--orange);">Código: ${cleanData}</div>
          <div class="qr-result-detail">Producto no encontrado en el catálogo</div>
        </div>
      </div>
      <div style="display:flex;gap:8px;padding:0 14px 12px;">
        <button class="btn btn-outline" style="flex:1;" onclick="resetQRScanner()">🔄 Escanear de nuevo</button>
      </div>
    `;

    window._qrProduct = null;
    qrDetectedEl.style.display = 'block';
    qrDetectedEl.style.animation = 'fadeInUp 0.36s ease-out';
    showToast('⚠️ Producto no encontrado en el catálogo');
  }
}

function qrAddToCart() {
  const p = window._qrProduct;
  if (!p) {
    showToast('⚠️ Primero detecta un producto');
    return;
  }
  addToCart(p.key, p.price, p.location, p.emoji, p.image);
  showToast(`✅ ${p.nombre || p.key} añadido a la cesta`);
}

function qrViewProduct() {
  const p = window._qrProduct;
  if (p) {
    openProduct(p.key);
  } else {
    showToast('⚠️ Primero detecta un producto');
  }
}

function qrToggleFav() {
  const p = window._qrProduct;
  if (!p) {
    showToast('⚠️ Primero detecta un producto');
    return;
  }
  const idx = favItems.findIndex(f => f.key === p.key);
  const btn = document.getElementById('qr-fav-btn');
  if (idx >= 0) {
    favItems.splice(idx, 1);
    if (btn) btn.textContent = '☆';
    showToast('✖️ Eliminado de Favoritos');
  } else {
    favItems.push({
      id: Date.now(),
      key: p.key,
      name: p.nombre || p.key,
      price: p.price,
      location: p.location,
      peso: p.peso,
      emoji: p.emoji,
      image: p.image
    });
    if (btn) btn.textContent = '⭐';
    showToast('⭐ Guardado en Favoritos');
  }
}

function resetQRScanner() {
  qrDetected = false;
  const qrDetectedEl = document.getElementById('qr-detected');
  if (qrDetectedEl) qrDetectedEl.style.display = 'none';
  const statusEl = document.getElementById('qr-status');
  if (statusEl) {
    statusEl.textContent = '📷 Buscando código QR...';
    statusEl.className = 'cam-status';
  }
  window._qrProduct = null;
  startQRScanning();
}

function toggleQRFlash() {
  const btn = document.getElementById('qr-flash-btn');
  const isOn = btn.style.background.includes('219');
  btn.style.background = isOn ? 'rgba(255,255,255,0.18)' : 'rgba(255,219,0,0.4)';
  showToast('💡 Flash ' + (isOn ? 'desactivado' : 'activado'));
}

// ─── AR IA ───

async function arFlipCamera() {
  arFacing = arFacing === 'environment' ? 'user' : 'environment';
  await startCamera('ar');
  showToast('🔄 Cámara cambiada');
}

async function arCapture() {
  const video = document.getElementById('ar-video');
  const captBtn = document.getElementById('ar-capture-btn');
  const statusEl = document.getElementById('ar-status');
  const loader = document.getElementById('ar-loader');

  if (!streams['ar'] || !video.srcObject) {
    showToast('⚠️ No hay cámara activa');
    return;
  }

  // Deshabilitar botón inmediatamente
  if (captBtn) captBtn.disabled = true;

  try {
    // Esperar un frame para asegurar que el video tiene dimensiones
    await new Promise(resolve => requestAnimationFrame(resolve));

    // Capturar frame sin pausar el video
    let w = video.videoWidth;
    let h = video.videoHeight;

    // Si las dimensiones no están disponibles, usar las del video element
    if (!w || !h) {
      w = video.clientWidth || 640;
      h = video.clientHeight || 480;
    }

    // Solo redimensionar canvas si las dimensiones cambiaron significativamente
    if (captureCanvas.width !== w || captureCanvas.height !== h) {
      captureCanvas.width = w;
      captureCanvas.height = h;
    }

    // Dibujar frame actual
    captureCtx.drawImage(video, 0, 0, w, h);

    if (statusEl) {
      statusEl.textContent = '🔍 Identificando…';
      statusEl.className = 'cam-status detecting';
    }
    if (loader) loader.classList.add('on');

    // Crear blob de forma asíncrona
    const blob = await new Promise(resolve => {
      captureCanvas.toBlob(resolve, 'image/jpeg', 0.85);
    });

    if (!blob) {
      throw new Error('Error al crear imagen');
    }

    const formData = new FormData();
    formData.append('file', new File([blob], 'ar_capture.jpg', { type: 'image/jpeg' }));

    const response = await fetch(AR_API, {
      method: 'POST',
      body: formData
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    const data = await response.json();
    renderARResults(data);

    document.getElementById('ar-camera-wrap').classList.remove('expanded');

    if (statusEl) {
      statusEl.textContent = '✅ Detectado';
      statusEl.className = 'cam-status found';
    }

  } catch (e) {
    console.error('Error en AR:', e);
    if (statusEl) {
      statusEl.textContent = '⚠️ API no disponible';
      statusEl.className = 'cam-status';
    }

    const list = document.getElementById('ar-detected-list');
    if (list) {
      list.innerHTML = `
        <div style="padding:16px;text-align:center;color:var(--gray);">
          <div style="font-size:28px;margin-bottom:7px;">🔌</div>
          <div style="font-size:12.5px;font-weight:700;">Sin conexión con la API</div>
          <div style="font-size:10.5px;margin-top:3px;">Asegúrate de que app.py está corriendo.</div>
        </div>`;
    }
  } finally {
    // Reactivar botón y ocultar loader
    if (captBtn) captBtn.disabled = false;
    if (loader) loader.classList.remove('on');
    // NO modificar el video stream - la cámara sigue funcionando
  }
}

function renderARResults(data) {
  const list = document.getElementById('ar-detected-list');
  if (!list) return;

  const all = [data.best_match, ...(data.alternatives || [])].filter(Boolean);
  window._arResults = all;
  if (!all.length) {
    list.innerHTML = '<div style="padding:18px;text-align:center;color:var(--gray);">Sin resultados</div>';
    return;
  }

  list.innerHTML = all.map((item, i) => {
    // La respuesta de la API viene con product_name que puede ser la key completa
    let searchTerm = item.product_name || item.id || '';

    // Buscar en el catálogo usando la nueva función
    const product = findProduct(searchTerm);

    if (product) {
      // Producto encontrado en el catálogo
      const displayName = product.nombre || product.key || 'Producto';
      const priceStr = product.priceStr || formatPrice(product.price);
      const locationStr = product.location || 'Consultar';
      const emojiStr = product.emoji || '📦';
      const descStr = product.subtitulo || product.desc || '';
      const imageStr = product.image || '';
      const conf = Math.round((item.confidence || item.confidence_pct || 0) * 100);

      return `
        <div style="padding:11px 13px;display:flex;gap:9px;align-items:center;border-bottom:${i < all.length - 1 ? '1px solid var(--border)' : 'none'};">
          ${imageStr ? `<img src="${imageStr}" alt="${displayName}" style="width:50px;height:50px;object-fit:contain;border-radius:8px;flex-shrink:0;" onerror="this.style.display='none';this.nextElementSibling.style.display='flex';"><div class="product-img" style="display:none;${i === 0 ? 'border:2px solid var(--blue);' : ''}">${emojiStr}</div>` : `<div class="product-img" style="${i === 0 ? 'border:2px solid var(--blue);' : ''}">${emojiStr}</div>`}
          <div style="flex:1;cursor:pointer;" onclick="openProduct('${product.key}')">
            <div class="product-name">${displayName}</div>
            <div class="product-desc">${descStr}</div>
            <span class="product-location">📍 ${locationStr}</span>
            <div><span class="ar-conf-pill ${conf >= 65 ? '' : 'low'}">⚡ ${conf}% confianza</span></div>
          </div>
          <div style="text-align:right;">
            <div class="product-price">${priceStr}</div>
            <button class="btn btn-primary" style="margin-top:4px;padding:5px 9px;font-size:10.5px;"
              onclick="addToCart('${product.key}',${product.price || 0},'${locationStr}','${emojiStr}','${imageStr}');showToast('✅ ${displayName} añadido')">+</button>
          </div>
        </div>
      `;
    } else {
      // Producto NO encontrado - mostrar con la información de la API
      const displayName = searchTerm || 'Desconocido';
      const conf = Math.round((item.confidence || item.confidence_pct || 0) * 100);
      const priceStr = item.precio ? formatPrice(item.precio) : 'Consultar';

      return `
        <div style="padding:11px 13px;display:flex;gap:9px;align-items:center;border-bottom:${i < all.length - 1 ? '1px solid var(--border)' : 'none'};background:rgba(255,165,0,0.1);">
          <div class="product-img" style="${i === 0 ? 'border:2px solid var(--orange);' : ''}">❓</div>
          <div style="flex:1;">
            <div class="product-name">${displayName}</div>
            <div class="product-desc" style="color:var(--gray);">No encontrado en el catálogo</div>
            <div><span class="ar-conf-pill ${conf >= 65 ? '' : 'low'}">⚡ ${conf}% confianza</span></div>
          </div>
          <div style="text-align:right;">
            <div class="product-price">${priceStr}</div>
          </div>
        </div>
      `;
    }
  }).join('');
}

// Expande la cámara al hacer clic
function expandCamera(type, event) {
    // Si hacemos clic en el flash o el botón de AR, no hacemos nada
    if (event && (event.target.closest('button') || event.target.closest('.ar-controls-bar'))) return;

    const wrap = document.getElementById(type + '-camera-wrap');
    
    // Solo actuamos si está encogida
    if (!wrap.classList.contains('expanded')) {
        wrap.classList.add('expanded');
        
        // UX Top: Si expandes la de QR, asumimos que quieres volver a escanear
        if (type === 'qr' && qrDetected) {
            resetQRScanner();
        }
    }
}

// ═══════════════════════════════ PAYMENT ═══════════════════════════════

function confirmPaymentLegacyDisabled() {
  const btn=document.getElementById('pay-confirm-btn'); btn.textContent='⏳ Procesando...'; btn.disabled=true;
  setTimeout(()=>goTo('success'), 1800);
}

// ═══════════════════════════════ TOAST ═══════════════════════════════

let toastTimeout;
function showToast(msg) {
  const t=document.getElementById('toast');
  t.textContent=msg; t.classList.add('show');
  clearTimeout(toastTimeout);
  toastTimeout=setTimeout(()=>t.classList.remove('show'), 2500);
}

// ═══════════════════════════════ RIPPLE ═══════════════════════════════

document.addEventListener('click', e => {
  const btn = e.target.closest('.ripple-btn'); if(!btn) return;
  const r = document.createElement('div'); r.className='ripple-effect';
  const rect = btn.getBoundingClientRect();
  r.style.left=(e.clientX-rect.left)+'px'; r.style.top=(e.clientY-rect.top)+'px';
  btn.appendChild(r); setTimeout(()=>r.remove(), 600);
});

// ═══════════════════════════════ INIT ═══════════════════════════════

// Cargar catálogo primero, luego inicializar UI
document.addEventListener('DOMContentLoaded', () => {
  loadCatalog();
});

// Inicialización legacy (para cuando el catálogo ya está cargado localmente)
// Esto se ejecuta después de loadCatalog()
window.addEventListener('load', () => {
  // Si por alguna razón el catálogo no se cargó, usar datos locales
  setTimeout(() => {
    if (!catalogLoaded) {
      console.log('⚠️ Usando catálogo local');
      initLocalCatalog();
      initUIWithCatalog();
    }
  }, 1000);
});

// ═══════════════════════════════ RELOJ ═══════════════════════════════

function actualizarReloj() {
    const ahora = new Date();
    let horas = ahora.getHours();
    let minutos = ahora.getMinutes();
    
    // Añadir un cero a la izquierda si los minutos son menores de 10
    minutos = minutos < 10 ? '0' + minutos : minutos;
    
    const horaTexto = `${horas}:${minutos}`;
    
    // Busca TODOS los elementos que tengan la clase 'status-time' y les cambia el texto
    document.querySelectorAll('.status-time').forEach(reloj => {
        reloj.textContent = horaTexto;
    });
}

// Ejecutar nada más cargar la app
actualizarReloj();
// Actualizar cada 60 segundos
setInterval(actualizarReloj, 60000);

// ═══════════════════════════════ GESTOS ═══════════════════════════════

let gestosActivos = false;

function getActiveScreenId() {
    const activeScreen = document.querySelector('.screen.active');
    return activeScreen ? activeScreen.id : null;
}

const PANTALLAS_GESTO_BORRAR   = ['screen-inicio', 'screen-buscar', 'screen-shopping', 'screen-cesta', 'screen-favoritos', 'screen-producto', 'screen-ar', 'screen-escaner'];
const PANTALLAS_GESTO_FAVORITO = ['screen-producto', 'screen-ar', 'screen-escaner'];

function updateAllCartBadges() {
    const total = typeof cartItems !== 'undefined' ? cartItems.reduce((s, i) => s + i.qty, 0) : 0;
    document.querySelectorAll('.cart-badge').forEach(badge => {
        badge.textContent = total;
        badge.classList.add('badge-animate');
        setTimeout(() => badge.classList.remove('badge-animate'), 300);
    });
}

function updateAllCartBadges() {
    // Calcula el total real sumando las cantidades de cada item
    const total = typeof cartItems !== 'undefined' ? cartItems.reduce((s, i) => s + (i.qty || 1), 0) : 0;
    document.querySelectorAll('.cart-badge').forEach(badge => {
        badge.textContent = total;
        badge.classList.add('badge-animate');
        setTimeout(() => badge.classList.remove('badge-animate'), 300);
    });
}

function refreshProductButtonState() {
  
    if (getActiveScreenId() !== 'screen-producto' || !currentProductKey) return;
    
    const actionsContainer = document.getElementById('prod-actions');
    if (!actionsContainer) return;
    
    const addBtn = actionsContainer.querySelector('button.btn');
    if (!addBtn) return;

    const isInCart = cartItems.some(item => item.key === currentProductKey);

    if (isInCart) {
        addBtn.textContent = '✅ En la cesta';
        addBtn.className = 'btn btn-success ripple-btn'; 
    } else {
        addBtn.textContent = '+ Añadir a la cesta';
        addBtn.className = 'btn btn-primary ripple-btn'; // Tu clase original azul
    }
}

function refreshCurrentScreen(screenId) {
  
    refreshProductButtonState();

    switch (screenId) {
        case 'screen-cesta':
            if (typeof renderCart === 'function') renderCart();
            break;
        case 'screen-favoritos':
            if (typeof renderFavs === 'function') renderFavs();
            break;
        case 'screen-buscar':
            if (typeof renderSearch === 'function') renderSearch();
            break;
    }
    updateAllCartBadges();
    
}

function undoLastCartItem() {
    const screenId = getActiveScreenId();
    if (!PANTALLAS_GESTO_BORRAR.includes(screenId)) return;

    if (!cartItems || cartItems.length === 0) {
        showToast('ℹ️ Tu cesta ya está vacía');
        return;
    }

    if (screenId === 'screen-producto') {
        if (!currentProductKey) return;
        const index = cartItems.findIndex(item => item.key === currentProductKey);

        if (index !== -1) {
            const removed = cartItems.splice(index, 1)[0];
            showToast(`🗑️ "${removed.name || removed.key}" eliminado de la cesta`);
        } else {
            showToast('ℹ️ Este producto no está en tu cesta');
        }
        refreshCurrentScreen(screenId);
        return;
    }

    const removed = cartItems.pop();
    showToast(`🗑️ "${removed.name || removed.key}" eliminado de la cesta`);
    refreshCurrentScreen(screenId);
}

function favoriteCurrentItem() {
    const screenId = getActiveScreenId();
    if (!PANTALLAS_GESTO_FAVORITO.includes(screenId)) return;

    let product = null;

    if (screenId === 'screen-producto') {
        product = IKEA_DB[currentProductKey];
    } else if (screenId === 'screen-ar') {
        if (window._arResults && window._arResults.length > 0) {
            const topResult = window._arResults[0];
            const searchTerm = topResult.product_name || topResult.name || topResult.id || '';
            product = typeof findProduct === 'function' ? findProduct(searchTerm) : null;
        }
        if (!product) {
            showToast('ℹ️ No hay producto detectado por la IA');
            return;
        }
    } else if (screenId === 'screen-escaner') {
        product = window._qrProduct || null;
        if (!product) {
            showToast('ℹ️ No se ha escaneado ningún producto');
            return;
        }
    }

    if (!product) return;

    const exists = favItems.find(f => f.key === product.key);
    if (exists) {
        showToast('ℹ️ Ya está en favoritos');
        return;
    }

    favItems.push({
        id: Date.now(),
        key: product.key,
        name: product.nombre || product.key,
        price: product.price,
        location: product.location,
        peso: product.peso || product.weight || '-',
        emoji: product.emoji || '📦',
        image: product.image || '',
        inCart: cartItems.some(c => c.key === product.key)
    });

    showToast(`⭐ "${product.nombre || product.key}" guardado en Favoritos`);

    if (screenId === 'screen-producto') {
        const favBtn = document.getElementById('prod-fav-btn');
        if (favBtn) favBtn.textContent = '⭐';
    }

    if (typeof renderFavs === 'function') renderFavs();
}

class ShakeDetector {
    constructor(options) {
        this.threshold = options.threshold || 15;
        this.timeout = options.timeout || 1000;
        this.onShake = options.onShake;
        
        this.lastTime = Date.now();
        this.lastX = null;
        this.lastY = null;
        this.lastZ = null;
        this.lastShake = Date.now();
        
        this.handler = this.devicemotion.bind(this);
    }

    start() { window.addEventListener('devicemotion', this.handler, { passive: true }); }
    stop() { window.removeEventListener('devicemotion', this.handler); }

    devicemotion(e) {
        const current = e.accelerationIncludingGravity || e.acceleration;
        if (!current) return;
        
        const now = Date.now();
        // Comprobamos cada 100ms (filtra el ruido microscópico)
        if (now - this.lastTime > 100) { 
            const x = current.x, y = current.y, z = current.z;
            
            if (this.lastX === null) {
                this.lastX = x; this.lastY = y; this.lastZ = z; 
                return;
            }
            
            // Calculamos el latigazo en cada eje
            const deltaX = Math.abs(this.lastX - x);
            const deltaY = Math.abs(this.lastY - y);
            const deltaZ = Math.abs(this.lastZ - z);
            
            // Magia pura: Un agitado real mueve al menos DOS ejes a la vez. 
            // Un frenazo caminando o el giro del Favorito suele mover solo uno fuerte.
            if (((deltaX > this.threshold) && (deltaY > this.threshold)) || 
                ((deltaX > this.threshold) && (deltaZ > this.threshold)) || 
                ((deltaY > this.threshold) && (deltaZ > this.threshold))) {
                
                if (now - this.lastShake > this.timeout) {
                    if (typeof this.onShake === 'function') this.onShake();
                    this.lastShake = now;
                }
            }
            
            this.lastTime = now;
            this.lastX = x; this.lastY = y; this.lastZ = z;
        }
    }
}

// --- Lógica de Agitar ---
let miShakeEvent = null;

function initShakeLibrary() {
    if (miShakeEvent) return; 
    
    // Instanciamos nuestro propio motor. 12 es un buen equilibrio.
    miShakeEvent = new ShakeDetector({
        threshold: 12, 
        timeout: 1000, 
        onShake: undoLastCartItem // Llamamos directamente a tu función de borrar
    });
    
    miShakeEvent.start();
}

// --- Lógica de Flick (Giro rápido y suave a la derecha) ---
let tiltCooldownTime = 0;
let gammaHistory = []; 

function handleOrientation(event) {
    const screenId = getActiveScreenId();
    if (!PANTALLAS_GESTO_FAVORITO.includes(screenId)) return;

    const now = Date.now();
    if (now - tiltCooldownTime < 2000) return; // Cooldown post-favorito

    const gamma = event.gamma; // Rotación izquierda/derecha (-90 a 90)
    const beta = event.beta;   // Inclinación adelante/atrás

    if (gamma === null || beta === null) return;

    // Condición de seguridad: El usuario debe estar sosteniendo el móvil frente a él
    // Si está totalmente boca abajo, ignoramos.
    if (beta > 60) {
        gammaHistory = []; // Limpiamos historial si la postura no es natural
        return; 
    }

    // Mantenemos un historial de los últimos 250ms para analizar la fluidez del movimiento
    gammaHistory.push({ time: now, val: gamma });
    gammaHistory = gammaHistory.filter(h => now - h.time < 200);

    if (gammaHistory.length > 2) {
        const oldest = gammaHistory[0];
        const recentDelta = gamma - oldest.val; // Cuántos grados ha girado
        const timeDelta = now - oldest.time;    // En cuánto tiempo

        // Detectar un giro RÁPIDO (>35 grados en menos de 250ms) hacia la DERECHA
        // recentDelta > 35 asegura que es un movimiento intencionado
        // La división (recentDelta / timeDelta) asegura que haya velocidad (flick) y no sea un giro lento
        if (recentDelta > 35 && timeDelta > 50 && (recentDelta / timeDelta) > 0.18) {
            tiltCooldownTime = now;
            gammaHistory = []; 
            favoriteCurrentItem();
        }
    }
}

function requestSensorPermissions() {
    if (gestosActivos) {
        showToast('✅ Los sensores ya están activos');
        return;
    }

    if (location.protocol !== 'https:' && location.hostname !== 'localhost' && location.hostname !== '127.0.0.1') {
        showToast('⚠️ Los gestos requieren conexión HTTPS');
        return;
    }

    const activarSensores = () => {
      
        initShakeLibrary();
        window.addEventListener('deviceorientation', handleOrientation, { passive: true });
        gestosActivos = true;
        activarInterfazGestos();
        
    };

    // iOS 13+ requiere permiso por interacción del usuario
    if (typeof DeviceMotionEvent !== 'undefined' && typeof DeviceMotionEvent.requestPermission === 'function') {
        DeviceMotionEvent.requestPermission()
            .then(permissionState => {
                if (permissionState === 'granted') activarSensores();
                else showToast('⚠️ Permiso de sensores denegado');
            })
            .catch(console.error);
    } else {
        activarSensores();
    }
}

function activarInterfazGestos() {
    const panel = document.getElementById('panel-gestos');
    if (panel) {
        panel.innerHTML = `
            <div class="gestures-card fade-in-up s5" style="margin: 15px; min-height: 135px;">
              <div class="gestures-title" style="font-size:12px;font-weight:bold;margin-bottom:10px;color:#666;">GESTOS ACTIVOS</div>
              <div class="gesture-grid" style="display:grid;grid-template-columns:1fr 1fr;gap:8px;">
                <div class="gesture-item" style="background:rgba(82,32,125,0.1);border:1px solid rgba(82,32,125,0.3);padding:10px;border-radius:6px;display:flex;align-items:center;justify-content:center;gap:5px;">
                  <span>🎙️</span><span style="color:var(--purple);font-size:10px;">Voz → Buscar</span>
                </div>
                <div class="gesture-item" style="background:rgba(0,88,163,0.1);border:1px solid rgba(0,88,163,0.3);padding:10px;border-radius:6px;display:flex;align-items:center;justify-content:center;gap:5px;">
                  <span>📷</span><span style="color:var(--blue);font-size:10px;">Cámara → Añadir</span>
                </div>
                <div class="gesture-item" style="background:rgba(239,119,68,0.1);border:1px solid rgba(239,119,68,0.3);padding:10px;border-radius:6px;display:flex;align-items:center;justify-content:center;gap:5px;">
                  <span>↪️</span><span style="color:var(--orange);font-size:10px;">Giro Der → Fav</span>
                </div>
                <div class="gesture-item" style="background:rgba(239,68,68,0.1);border:1px solid rgba(239,68,68,0.3);padding:10px;border-radius:6px;display:flex;align-items:center;justify-content:center;gap:5px;">
                  <span>📳</span><span style="color:var(--red);font-size:10px;">Agitar → Borrar</span>
                </div>
              </div>
            </div>
        `;
    }
    showToast('✅ ¡Sensores activados!');
}

// Helper para inicializar el micro
function getSpeechRecognition() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
        showToast('⚠️ Tu navegador no soporta dictado por voz');
        return null;
    }
    const recognition = new SpeechRecognition();
    recognition.lang = 'es-ES'; // Idioma español
    recognition.interimResults = false;
    return recognition;
}

function startVoiceSearch() {
    const recognition = getSpeechRecognition();
    if (!recognition) return;

    recognition.onstart = () => showToast('🎙️ Escuchando... Di lo que buscas');
    
    recognition.onresult = (event) => {
        // Pillamos lo que has dicho y le quitamos el punto final si lo pone
        let transcript = event.results[0][0].transcript;
        if (transcript.endsWith('.')) transcript = transcript.slice(0, -1);
        
        showToast(`✅ Buscando: "${transcript}"`);
        
        // 1. Te llevamos a la pantalla de búsqueda
        goTo('buscar');
        
        setTimeout(() => {
            // 2. Metemos el texto en la barra principal
            const searchInput = document.getElementById('search-input');
            if (searchInput) {
                searchInput.value = transcript;
                
                // 3. Forzamos a que salte la búsqueda (el oninput no salta solo por JS)
                if (typeof filterProducts === 'function') {
                    filterProducts(transcript);
                }
            }
        }, 100); // Mismo margen para asegurar que la pantalla ha cargado
    };
    
    recognition.start();
}

function startVoiceSort() {
    const recognition = getSpeechRecognition();
    if (!recognition) return;

    recognition.onstart = () => showToast('🎙️ Di: Precio, Peso, Ruta o Nombre...');
    
    recognition.onresult = (event) => {
        // Pasamos todo a minúsculas y quitamos espacios extra para que no falle
        const command = event.results[0][0].transcript.toLowerCase().trim();
        
        if (command.includes('precio') || command.includes('dinero') || command.includes('caro')) {
            showToast('🗣️ Ordenando por: Precio');
            sortCart(null, 'price');
        } 
        else if (command.includes('peso') || command.includes('gramos') || command.includes('kilos')) {
            showToast('🗣️ Ordenando por: Peso');
            sortCart(null, 'weight');
        } 
        else if (command.includes('ruta') || command.includes('pasillo') || command.includes('ubicación')) {
            showToast('🗣️ Ordenando por: Ruta');
            sortCart(null, 'route');
        } 
        else if (command.includes('a z') || command.includes('alfabético') || command.includes('nombre')) {
            showToast('🗣️ Ordenando por: Nombre');
            sortCart(null, 'az');
        } 
        else {
            showToast(`⚠️ Comando no reconocido: "${command}"`);
        }
    };
    
    recognition.start();
}

function openSearchAndFocus() {
    goTo('buscar');
    
    // Le damos 100ms de margen para que la pantalla termine de hacerse visible
    setTimeout(() => {
        const searchInput = document.getElementById('search-input');
        if (searchInput) {
            searchInput.focus();
        }
    }, 100);
}