const $ = (id) => document.getElementById(id);

const htmlRoot = $('htmlRoot');
const chatForm = $('chatForm');
const promptInput = $('prompt');
const sendBtn = $('sendBtn');
const typingIndicator = $('typingIndicator');
const chatLog = $('chatLog');
const clarifyChips = $('clarifyChips');
const guardrailBanner = $('guardrailBanner');
const recommendationsNode = $('recommendations');
const stage = $('stage');
const room = $('room');
const divider = $('divider');
const swatchesNode = $('swatches');
const ledgerItems = $('ledgerItems');
const quotePanel = $('quotePanel');
const quoteBtn = $('quoteBtn');

const SURFACES = ['floor', 'wall', 'balcony'];
const LABEL = { floor: 'Floor', wall: 'Wall', balcony: 'Balcony' };
const PLURAL = { floor: 'floors', wall: 'walls', balcony: 'balconies' };
const RATING = { floor: 'floor', wall: 'wall', balcony: 'outdoor' };
const PX_PER_METRE = 150; // grout scale in the visualizer

const newSurface = () => ({ item: null, area: null, edited: false, calc: null, inv: null, pending: 0 });

const state = {
  items: [],            // latest recommendations from the assistant
  catalog: new Map(),   // every product the assistant has shown, by SKU
  requirements: {},
  target: 'floor',      // surface that clicks on products apply to
  mode: 'room',
  surfaces: { floor: newSurface(), wall: newSurface(), balcony: newSurface() },
  userAssigned: false,
  quote: null,
  photos: {},           // sku -> { room?: url, texture?: url } from /api/v1/product-images
};

/* Helpers -------------------------------------------------------------- */

const money = (value) => {
  const digits = Number.isInteger(Number(value)) ? 0 : 2;
  return `$${Number(value).toLocaleString('en-US', { minimumFractionDigits: digits, maximumFractionDigits: digits })}`;
};

const pretty = (code) => String(code || '').replace(/_/g, ' ').toLowerCase();

function clearChildren(node) {
  while (node.firstChild) node.removeChild(node.firstChild);
}

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text; // never innerHTML (FR-6.2)
  return node;
}

// Placement rules mirror backend/app/models/catalog.py (_PLACEMENT).
const WALL_RATED_GENERAL = ['Indoor', 'Premium indoor'];
const suitsWall = (item) => /wall/i.test(item.application || '') || WALL_RATED_GENERAL.includes(item.application);
const suitsFloor = (item) => !/wall/i.test(item.application || '');
// A balcony is an outdoor floor: it needs an outdoor-rated tile.
const suitsBalcony = (item) => /outdoor/i.test(item.application || '');
const suits = (item, surface) => {
  if (surface === 'wall') return suitsWall(item);
  if (surface === 'balcony') return suitsBalcony(item);
  return suitsFloor(item);
};

function tileSizeCm(item) {
  const parts = String(item.size || '').split(/[×x]/).map(Number);
  return parts.length === 2 && parts.every((n) => n > 0) ? parts : [60, 60];
}

/* Procedural surface textures (the catalogue has no photography) -------- */

function surfaceTexture(item) {
  const style = `${item.style || ''} ${item.type || ''}`.toLowerCase();
  const glossy = /polished|gloss/.test(item.finish || '');
  const sheen = glossy
    ? 'linear-gradient(120deg, rgba(255,255,255,0.4) 0%, rgba(255,255,255,0) 38%, rgba(255,255,255,0.22) 70%, rgba(255,255,255,0) 100%), '
    : '';

  let base;
  let color;
  if (/wood/.test(style)) {
    color = '#b8864f';
    base = 'repeating-linear-gradient(90deg, #b8864f 0 18px, #a7753f 18px 20px, #c29561 20px 44px, #9c6c3a 44px 46px)';
  } else if (/onyx|black|luxury$/.test(style) && !/marble/.test(style)) {
    color = '#23262b';
    base = 'linear-gradient(115deg, transparent 40%, rgba(212,175,95,0.55) 41%, transparent 43%), linear-gradient(65deg, transparent 60%, rgba(212,175,95,0.35) 61%, transparent 62.5%), #23262b';
  } else if (/marble/.test(style)) {
    color = '#e8e6e1';
    base = 'linear-gradient(118deg, transparent 32%, rgba(120,120,125,0.35) 33%, transparent 34.5%), linear-gradient(62deg, transparent 58%, rgba(150,150,155,0.3) 59%, transparent 60.5%), linear-gradient(170deg, #f4f3ef, #dedbd4)';
  } else if (/subway/.test(style)) {
    color = '#f1f1ef';
    base = 'repeating-linear-gradient(0deg, transparent 0 22px, #d5d6d4 22px 24px), repeating-linear-gradient(90deg, transparent 0 46px, #d5d6d4 46px 48px), #f4f4f2';
  } else if (/decorative|mosaic/.test(style)) {
    color = '#c9d6dc';
    base = 'repeating-linear-gradient(0deg, transparent 0 14px, #fff 14px 16px), repeating-linear-gradient(90deg, transparent 0 14px, #fff 14px 16px), linear-gradient(135deg, #cfdde3, #a9bfc9)';
  } else if (/stone|slate|rustic|natural|sand/.test(style)) {
    color = '#a89a86';
    base = 'radial-gradient(rgba(70,60,50,0.18) 1px, transparent 1.5px) 0 0 / 7px 7px, linear-gradient(150deg, #b9ac98, #8f8372)';
  } else if (/industrial|concrete/.test(style)) {
    color = '#7c8086';
    base = 'radial-gradient(rgba(0,0,0,0.12) 1px, transparent 1.5px) 0 0 / 9px 9px, linear-gradient(160deg, #8b8f95, #6b6f75)';
  } else if (/minimal|contemporary/.test(style)) {
    color = '#ecebe7';
    base = 'linear-gradient(160deg, #f3f2ee, #dcdad4)';
  } else {
    color = '#b4b9bf';
    base = 'linear-gradient(160deg, #c3c8cd, #a1a7ae)';
  }
  return { css: `${sheen}${base}`, color };
}

const EMPTY_TEXTURE = {
  floor: 'linear-gradient(160deg, #cfccc6, #b9b5ae)',
  wall: 'linear-gradient(170deg, #efece6, #e2ded7)',
};

function applyTexture(node, item) {
  node.style.background = surfaceTexture(item).css;
}

// Lifestyle photo when one exists for this SKU, generated texture otherwise.
function paintPhoto(node, item) {
  const photo = state.photos[item.sku] && state.photos[item.sku].room;
  if (photo) {
    node.style.background = `url("${photo}") center / cover no-repeat, ${EMPTY_TEXTURE.wall}`;
  } else {
    applyTexture(node, item);
  }
}

function paintTiled(node, item, fallback) {
  if (item) {
    const [w, l] = tileSizeCm(item);
    const tex = state.photos[item.sku] && state.photos[item.sku].texture;
    node.style.background = tex
      ? `url("${tex}") 0 0 / ${(w / 100) * PX_PER_METRE}px ${(l / 100) * PX_PER_METRE}px repeat`
      : surfaceTexture(item).css;
    node.style.setProperty('--tw', `${(w / 100) * PX_PER_METRE}px`);
    node.style.setProperty('--tl', `${(l / 100) * PX_PER_METRE}px`);
    node.style.setProperty('--grout', '1');
  } else {
    node.style.background = fallback;
    node.style.setProperty('--grout', '0');
  }
}

/* Chat ------------------------------------------------------------------ */

function timeNow() {
  return new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

// Returns the column node for assistant messages so bundles can be attached.
function appendMessage(role, text) {
  const msg = el('div', `msg ${role}`);
  const bubble = el('div', 'bubble', text);
  bubble.setAttribute('dir', 'auto');
  let col = msg;
  if (role === 'assistant') {
    msg.appendChild(el('span', 'msg-avatar', 'CN'));
    col = el('div', 'msg-col');
    col.appendChild(bubble);
    msg.appendChild(col);
  } else {
    msg.appendChild(bubble);
  }
  col.appendChild(el('div', 'msg-time', role === 'user' ? `${timeNow()} ✓✓` : timeNow()));
  chatLog.appendChild(msg);
  chatLog.scrollTop = chatLog.scrollHeight;
  return col;
}

function applyButtons(item) {
  const wrap = el('div', 'bundle-apply');
  SURFACES.forEach((surface) => {
    const btn = el('button', 'apply-btn', LABEL[surface]);
    btn.type = 'button';
    btn.dataset.sku = item.sku;
    btn.dataset.surface = surface;
    if (!suits(item, surface)) {
      btn.disabled = true;
      btn.title = `${item.name} is rated for ${item.application} — not for ${PLURAL[surface]}`;
    } else {
      btn.title = `Apply ${item.name} to the ${surface}`;
    }
    btn.addEventListener('click', () => assign(surface, item.sku));
    wrap.appendChild(btn);
  });
  return wrap;
}

function appendBundle(col, items) {
  const sourced = el('span', 'sourced', '✓ Catalog-sourced: Ceramica Nova product catalogue');
  col.insertBefore(sourced, col.lastChild);

  const bundle = el('div', 'bundle');
  const head = el('div', 'bundle-head');
  head.appendChild(el('b', '', 'Recommended picks'));
  head.appendChild(el('span', 'match-tag', `${items.length} matched`));
  bundle.appendChild(head);

  items.forEach((item) => {
    const row = el('div', 'bundle-row');
    const thumb = el('div', 'bundle-thumb');
    paintPhoto(thumb, item);
    row.appendChild(thumb);
    const text = el('div', 'bundle-text');
    text.appendChild(el('b', '', item.name));
    text.appendChild(el('small', '', `${money(item.price_per_m2)}/m² · ${item.size} · ${item.finish} · ${item.slip_rating}`));
    row.appendChild(text);
    row.appendChild(applyButtons(item));
    bundle.appendChild(row);
  });
  col.insertBefore(bundle, col.lastChild);
  refreshApplyButtons();
  chatLog.scrollTop = chatLog.scrollHeight;
}

function refreshApplyButtons() {
  document.querySelectorAll('.apply-btn').forEach((btn) => {
    const current = state.surfaces[btn.dataset.surface].item;
    btn.classList.toggle('on', !!current && current.sku === btn.dataset.sku);
  });
}

function renderClarifyingChips(questions) {
  clearChildren(clarifyChips);
  (questions || []).forEach((q) => {
    const chip = el('button', 'chip', q);
    chip.type = 'button';
    chip.addEventListener('click', () => {
      promptInput.value = q;
      autosize();
      promptInput.focus();
    });
    clarifyChips.appendChild(chip);
  });
}

function renderGuardrail(guardrail) {
  if (guardrail && guardrail.triggered) {
    guardrailBanner.textContent = `A safety guardrail adjusted this reply (${guardrail.rule || 'policy'}).`;
    guardrailBanner.classList.remove('hidden');
  } else {
    guardrailBanner.classList.add('hidden');
  }
}

/* Surface assignment ---------------------------------------------------- */

let toastTimer;
function toast(name, message) {
  const node = $('stageToast');
  clearChildren(node);
  if (name) node.appendChild(el('b', '', name));
  node.appendChild(document.createTextNode(name ? ` ${message}` : message));
  node.classList.remove('hidden');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => node.classList.add('hidden'), 2600);
}

function clearSurface(surface) {
  const slot = state.surfaces[surface];
  slot.item = null;
  slot.calc = null;
  slot.inv = null;
  slot.pending += 1;
  state.quote = null;
  renderAll();
}

function setTarget(surface) {
  state.target = surface;
  stage.dataset.target = surface;
  renderStage();
  renderLedger();
}

function assign(surface, sku) {
  const item = state.catalog.get(sku);
  if (!item) return;
  if (!suits(item, surface)) {
    toast(item.name, `is rated for ${item.application.toLowerCase()} use — pick a ${RATING[surface]}-rated tile.`);
    return;
  }
  const slot = state.surfaces[surface];
  if (slot.item && slot.item.sku === sku) {
    setTarget(surface);
    return;
  }
  slot.item = item;
  slot.calc = null;
  slot.inv = null;
  state.target = surface;
  stage.dataset.target = surface;
  state.userAssigned = true;
  state.quote = null;
  if (state.mode !== 'room') setMode('room');
  recalc(surface);
  renderAll();
}

async function postJson(url, body) {
  const response = await fetch(url, {
    method: 'POST',
    credentials: 'same-origin',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  const data = await response.json();
  if (!response.ok) throw new Error((data.error && data.error.message) || 'Request failed');
  return data;
}

// Box counts and prices always come from the server calculator (FR-2).
async function recalc(surface) {
  const slot = state.surfaces[surface];
  slot.calc = null;
  slot.inv = null;
  const ticket = ++slot.pending;
  if (!slot.item || !(slot.area > 0)) {
    renderLedger();
    return;
  }
  try {
    const calc = await postJson('/api/v1/tools/calculate-quantity', { sku: slot.item.sku, area_m2: slot.area });
    const inv = await postJson('/api/v1/tools/check-inventory', {
      sku: slot.item.sku,
      required_area_m2: calc.total_purchasable_area_m2,
    });
    if (ticket !== slot.pending) return; // superseded by a newer edit
    slot.calc = calc;
    slot.inv = inv;
  } catch (error) {
    if (ticket === slot.pending) toast('', 'Could not price that surface. Check the area value.');
  }
  renderLedger();
  renderSteps();
}

/* Rendering ------------------------------------------------------------- */

const SKY = 'linear-gradient(to bottom, #8fb8de 0%, #c4dcef 55%, #e6eef3 100%)';

// The balcony target switches the box to an open-air scene: sky and a glass
// railing instead of tiled walls, with the balcony tile on the floor.
function renderRoom() {
  const outdoor = state.target === 'balcony';
  room.classList.toggle('outdoor', outdoor);
  const floorItem = state.surfaces[outdoor ? 'balcony' : 'floor'].item;
  const wall = state.surfaces.wall.item;
  paintTiled(room.querySelector('.face.floor'), floorItem, EMPTY_TEXTURE.floor);
  room.querySelectorAll('.face.wall').forEach((face) => {
    if (outdoor) {
      face.style.background = SKY;
      face.style.setProperty('--grout', '0');
    } else {
      paintTiled(face, wall, EMPTY_TEXTURE.wall);
    }
  });

  const tagWall = $('tagWall');
  const tagFloor = $('tagFloor');
  const floorLabel = outdoor ? 'Balcony' : 'Floor';
  if (outdoor) {
    tagWall.textContent = 'Open-air balcony view';
  } else {
    tagWall.textContent = wall ? `Wall · ${wall.name} (${wall.finish})` : 'Wall · not selected';
  }
  tagFloor.textContent = floorItem
    ? `${floorLabel} · ${floorItem.name} (${floorItem.finish}, ${floorItem.slip_rating})`
    : `${floorLabel} · not selected`;
  tagWall.classList.toggle('ok', outdoor || !!wall);
  tagFloor.classList.toggle('ok', !!floorItem);
}

function renderCompare() {
  const targetItem = state.surfaces[state.target].item;
  const pool = state.items.length ? state.items : [...state.catalog.values()];
  const first = targetItem || pool[0];
  const second = pool.find((i) => first && i.sku !== first.sku);
  const a = $('stageA');
  const b = $('stageB');
  if (first) paintPhoto(a, first); else a.style.background = EMPTY_TEXTURE.wall;
  if (second) paintPhoto(b, second); else b.style.background = a.style.background;
  $('labelA').textContent = first ? `${first.name} · ${first.finish}` : '';
  $('labelB').textContent = second ? `${second.name} · ${second.finish}` : '';
  divider.disabled = !second;
}

function renderZoom() {
  const item = state.surfaces[state.target].item || state.items[0];
  const zoom = $('zoom');
  paintTiled(zoom, item, EMPTY_TEXTURE[state.target]);
  return item;
}

function renderStage() {
  renderRoom();
  renderCompare();
  const zoomItem = renderZoom();

  const badge = $('stageBadge');
  const sub = $('stageSub');
  if (state.mode === 'room') {
    badge.textContent = 'AI Space Preview · Concept Visual';
    sub.textContent = `Targeting ${state.target}`;
  } else if (state.mode === 'compare') {
    badge.textContent = 'Side-by-Side · Surface Comparison';
    sub.textContent = 'Drag to compare';
  } else {
    badge.textContent = `Surface Texture · ${zoomItem ? zoomItem.name : 'none selected'}`;
    sub.textContent = zoomItem ? `${zoomItem.size} cm · ${zoomItem.finish}` : `Pick a ${state.target} tile`;
  }

  document.querySelectorAll('.seg-btn').forEach((b) =>
    b.classList.toggle('active', b.dataset.surface === state.target));
  $('swatchHint').textContent = `Apply to ${state.target}:`;

  clearChildren(swatchesNode);
  const pool = uniqueItems([state.surfaces.floor.item, state.surfaces.wall.item, ...state.items]);
  const current = state.surfaces[state.target].item;
  pool.forEach((item) => {
    const sw = el('button', `swatch${current && item.sku === current.sku ? ' active' : ''}`);
    sw.type = 'button';
    sw.style.background = surfaceTexture(item).color;
    const ok = suits(item, state.target);
    sw.title = ok ? `${item.name} → ${state.target}` : `${item.name} is not rated for ${PLURAL[state.target]}`;
    sw.setAttribute('aria-label', sw.title);
    sw.disabled = !ok;
    sw.addEventListener('click', () => assign(state.target, item.sku));
    swatchesNode.appendChild(sw);
  });
}

function uniqueItems(list) {
  const seen = new Set();
  return list.filter((item) => item && !seen.has(item.sku) && seen.add(item.sku));
}

function renderCards() {
  clearChildren(recommendationsNode);
  const assigned = SURFACES.map((s) => state.surfaces[s].item);
  const list = uniqueItems([...assigned, ...state.items]);

  const specified = assigned.filter(Boolean).length;
  const setTag = $('setTag');
  setTag.textContent = specified > 1
    ? `Coordinated set (${SURFACES.filter((s) => state.surfaces[s].item).join(' + ')})`
    : `${specified} of ${SURFACES.length} surfaces specified`;
  setTag.classList.toggle('hidden', !list.length);

  list.forEach((item) => {
    const on = SURFACES.filter((s) => state.surfaces[s].item && state.surfaces[s].item.sku === item.sku);
    const card = el('div', `product${on.length ? ' applied' : ''}`);

    const img = el('div', 'product-img');
    paintPhoto(img, item);
    const placed = on.map((s) => LABEL[s]).join(' + ');
    img.appendChild(el('span', `product-tag${placed ? ' on' : ''}`, placed ? `On ${placed.toLowerCase()}` : item.type || 'tile'));
    card.appendChild(img);

    const body = el('div', 'product-body');
    body.appendChild(el('h3', '', item.name));
    body.appendChild(el('p', 'spec', `${item.sku} · ${item.size} cm · ${item.finish} · ${item.slip_rating}`));
    body.appendChild(el('p', 'spec', item.application));
    const price = el('p', 'price', money(item.price_per_m2));
    price.appendChild(el('small', '', ' / m²'));
    body.appendChild(price);

    const badges = el('div', 'badges');
    if (item.status) badges.appendChild(el('span', `badge ${item.status.toLowerCase()}`, pretty(item.status)));
    if (item.over_budget_by) badges.appendChild(el('span', 'badge over', `+${money(item.over_budget_by)} over`));
    if (badges.childNodes.length) body.appendChild(badges);
    card.appendChild(body);

    const foot = el('div', 'product-foot');
    foot.appendChild(applyButtons(item));
    card.appendChild(foot);

    recommendationsNode.appendChild(card);
  });
  refreshApplyButtons();
}

function renderLedgerLine(surface) {
  const slot = state.surfaces[surface];
  const line = el('div', `line${slot.item ? '' : ' empty'}${state.target === surface ? ' active' : ''}`);
  line.addEventListener('click', (event) => {
    if (!event.target.closest('input, button')) setTarget(surface);
  });

  const thumb = el('div', 'line-img');
  if (slot.item) paintPhoto(thumb, slot.item);
  line.appendChild(thumb);

  const text = el('div', 'line-text');
  text.appendChild(el('span', 'line-surface', surface));
  text.appendChild(el('b', '', slot.item ? slot.item.name : `Choose a ${surface} tile`));
  let detail = slot.item ? `${money(slot.item.price_per_m2)}/m² · ${slot.item.size} cm` : 'Apply one from chat or cards';
  if (slot.calc) detail = `${slot.calc.boxes} boxes · ${slot.calc.total_purchasable_area_m2} m² · ${money(slot.calc.price_per_m2)}/m²`;
  else if (slot.item && !(slot.area > 0)) detail = 'Enter the area to price it';
  text.appendChild(el('small', '', detail));
  line.appendChild(text);

  const end = el('div', 'line-end');
  const amtRow = el('div', 'line-amt-row');
  amtRow.appendChild(el('span', 'line-amt', slot.calc ? money(slot.calc.material_subtotal) : '—'));
  if (slot.item) {
    const remove = el('button', 'line-remove', '×');
    remove.type = 'button';
    remove.title = `Remove ${slot.item.name} from the ${surface}`;
    remove.setAttribute('aria-label', remove.title);
    remove.addEventListener('click', () => clearSurface(surface));
    amtRow.appendChild(remove);
  }
  end.appendChild(amtRow);
  const field = el('label', 'area-field');
  const input = el('input');
  input.type = 'number';
  input.min = '0';
  input.step = '0.5';
  input.inputMode = 'decimal';
  input.placeholder = '0';
  input.value = slot.area > 0 ? String(slot.area) : '';
  input.setAttribute('aria-label', `${surface} area in square metres`);
  input.addEventListener('input', () => onAreaInput(surface, input.value));
  field.appendChild(input);
  field.appendChild(document.createTextNode('m²'));
  end.appendChild(field);
  line.appendChild(end);
  return line;
}

const areaTimers = {};
function onAreaInput(surface, raw) {
  const slot = state.surfaces[surface];
  const value = parseFloat(raw);
  slot.area = value > 0 ? value : null;
  slot.edited = true;
  state.quote = null;
  clearTimeout(areaTimers[surface]);
  areaTimers[surface] = setTimeout(() => recalc(surface), 450);
}

function renderLedger() {
  const req = state.requirements || {};
  const slots = SURFACES.map((s) => state.surfaces[s]);
  const priced = slots.filter((s) => s.item && s.calc);

  $('ledgerSub').textContent = req.application || req.indoor_outdoor
    ? [req.application, req.indoor_outdoor].filter(Boolean).join(' · ')
    : 'Awaiting project details';

  const netArea = slots.filter((s) => s.item && s.area > 0).reduce((sum, s) => sum + s.area, 0);
  const gross = priced.reduce((sum, s) => sum + s.calc.total_purchasable_area_m2, 0);
  const wastes = [...new Set(priced.map((s) => s.calc.wastage_percent))];
  $('statArea').textContent = netArea ? `${+netArea.toFixed(2)} m²` : '—';
  $('statWaste').textContent = wastes.length ? wastes.map((w) => `${w}%`).join(' / ') : '—';
  $('statQty').textContent = gross ? `${+gross.toFixed(2)} m²` : '—';

  // Keep the focused area input alive while the user is typing in it.
  const focused = document.activeElement && ledgerItems.contains(document.activeElement)
    ? document.activeElement.getAttribute('aria-label') : null;
  clearChildren(ledgerItems);
  SURFACES.forEach((surface) => ledgerItems.appendChild(renderLedgerLine(surface)));
  if (focused) {
    const again = ledgerItems.querySelector(`input[aria-label="${focused}"]`);
    if (again) again.focus();
  }
  $('itemCount').textContent = `${slots.filter((s) => s.item).length} of ${SURFACES.length} surfaces`;

  const subtotal = priced.reduce((sum, s) => sum + s.calc.material_subtotal, 0);
  const quote = state.quote;
  $('subtotal').textContent = priced.length ? money(subtotal) : '—';

  let total = priced.length ? subtotal : null;
  const discountRow = $('discountRow');
  if (quote && quote.discount_amount > 0) {
    discountRow.classList.remove('hidden');
    $('discountLabel').textContent = `Discount (${quote.discount_percent}%)`;
    $('discountAmt').textContent = `− ${money(quote.discount_amount)}`;
    total = quote.total;
  } else {
    discountRow.classList.add('hidden');
  }
  if (quote) total = quote.total;
  $('projectTotal').textContent = total !== null ? money(total) : '—';

  renderBudget(total);
  renderInfoTiles();

  quoteBtn.disabled = !priced.length;
  renderQuote(quote);
  $('docsBom').textContent = `BOM — ${priced.length} surface${priced.length === 1 ? '' : 's'} priced`;
  $('docsBom').classList.toggle('ready', priced.length > 0);
}

function renderBudget(total) {
  const budget = (state.requirements || {}).budget;
  const pill = $('budgetPill');
  const track = $('budgetTrack');
  const note = $('budgetNote');
  if (!budget) {
    pill.classList.add('hidden');
    track.classList.add('hidden');
    note.textContent = '';
    return;
  }
  const perM2 = budget.basis === 'per_m2';
  note.textContent = `${money(budget.amount)}${perM2 ? '/m²' : ''} target`;

  const assigned = SURFACES.map((s) => state.surfaces[s].item).filter(Boolean);
  if (total === null && !(perM2 && assigned.length)) {
    pill.classList.add('hidden');
    track.classList.add('hidden');
    return;
  }

  const within = perM2
    ? assigned.every((i) => i.price_per_m2 <= budget.amount)
    : total <= budget.amount;
  pill.textContent = within ? '✓ Within target' : 'Over target';
  pill.className = `budget-pill ${within ? 'ok' : 'over'}`;

  if (perM2 || total === null) {
    track.classList.add('hidden');
    return;
  }
  track.classList.remove('hidden');
  $('budgetTarget').textContent = `Budget (your target ${money(budget.amount)})`;
  $('budgetRatio').textContent = `${money(total)} / ${money(budget.amount)}`;
  const bar = $('budgetBar');
  bar.style.width = `${Math.min(100, (total / budget.amount) * 100)}%`;
  bar.classList.toggle('over', !within);
}

function renderInfoTiles() {
  const order = ['OUT_OF_STOCK', 'PARTIALLY_AVAILABLE', 'AVAILABLE'];
  const statuses = SURFACES.map((s) => state.surfaces[s].inv && state.surfaces[s].inv.status).filter(Boolean);
  const worst = order.find((s) => statuses.includes(s));
  const stockTile = $('stockTile');
  $('stockText').textContent = worst ? (worst === 'AVAILABLE' ? 'In stock for all surfaces' : pretty(worst)) : 'Price a surface to check';
  stockTile.className = `info-tile${!worst ? ' idle' : worst === 'AVAILABLE' ? '' : ' warn'}`;

  // Slip resistance matters on walked surfaces; outdoor needs R11 or better.
  const walked = ['floor', 'balcony'].filter((s) => state.surfaces[s].item);
  const weak = walked.filter((s) => s === 'balcony' && !/R1[1-3]/.test(state.surfaces[s].item.slip_rating));
  const slipTile = $('slipTile');
  $('slipText').textContent = walked.length
    ? walked.map((s) => `${LABEL[s]} ${state.surfaces[s].item.slip_rating}`).join(' · ') + (weak.length ? ' — low for outdoor' : '')
    : 'No floor tile yet';
  slipTile.className = `info-tile${!walked.length ? ' idle' : weak.length ? ' warn' : ''}`;
}

function renderQuote(quote) {
  clearChildren(quotePanel);
  const docs = $('docsQuote');
  if (!quote) {
    quotePanel.classList.add('hidden');
    docs.textContent = 'Quotation — not generated yet';
    docs.classList.remove('ready');
    return;
  }
  quotePanel.appendChild(el('h3', '', `Quote ${quote.quotation_id}`));
  if (quote.approval_required) quotePanel.appendChild(el('span', 'pending-badge', 'Pending staff approval'));
  (quote.lines || []).forEach((line) => {
    quotePanel.appendChild(el('p', '', `${line.product_name}: ${line.boxes} boxes · ${money(line.subtotal)}`));
  });
  quotePanel.appendChild(el('p', '', `Total ${money(quote.total)} (${pretty(quote.status)})`));
  quotePanel.classList.remove('hidden');
  docs.textContent = `Quotation ${quote.quotation_id} — ${pretty(quote.status)}`;
  docs.classList.add('ready');
}

function renderSteps() {
  const req = state.requirements || {};
  const priced = SURFACES.filter((s) => state.surfaces[s].calc);
  const assigned = SURFACES.filter((s) => state.surfaces[s].item);
  const done = {
    discover: !!(req.area_m2 || req.floor_or_wall || state.catalog.size),
    curate: state.catalog.size > 0,
    visualize: state.userAssigned,
    configure: state.userAssigned && assigned.length > 0 && priced.length === assigned.length,
    quote: !!state.quote,
  };
  const order = ['discover', 'curate', 'visualize', 'configure', 'quote'];
  const current = order.find((s) => !done[s]);
  document.querySelectorAll('#steps li').forEach((li) => {
    const step = li.dataset.step;
    li.classList.toggle('done', done[step] && step !== current);
    li.classList.toggle('current', step === current);
  });

  const ref = $('projectRef');
  clearChildren(ref);
  if (state.quote) {
    ref.appendChild(document.createTextNode('Quotation '));
    ref.appendChild(el('b', '', state.quote.quotation_id));
  } else {
    ref.appendChild(document.createTextNode(`Draft project · ${assigned.length} of ${SURFACES.length} surfaces`));
  }
}

function renderAll() {
  renderCards();
  renderStage();
  renderLedger();
  renderSteps();
}

function setMode(mode) {
  state.mode = mode;
  stage.dataset.mode = mode;
  document.querySelectorAll('.vtab').forEach((t) => {
    t.classList.toggle('active', t.dataset.mode === mode);
    t.setAttribute('aria-selected', String(t.dataset.mode === mode));
  });
  renderStage();
}

/* Applying an assistant turn ------------------------------------------- */

function applyResult(data, col) {
  state.requirements = data.requirements || {};
  const req = state.requirements;
  const items = data.recommendations || [];
  items.forEach((item) => state.catalog.set(item.sku, item));

  // Seed the area of the surface the customer described, unless they've
  // already typed one in the ledger.
  let described = req.floor_or_wall === 'wall' ? 'wall' : 'floor';
  if (req.indoor_outdoor === 'outdoor' && described === 'floor') described = 'balcony';
  const slot = state.surfaces[described];
  const areaChanged = req.area_m2 && !slot.edited && slot.area !== req.area_m2;
  if (areaChanged) slot.area = req.area_m2;

  // Keep the previous shortlist on screen when a reply has no products
  // (e.g. a clarifying question) so the studio doesn't flash empty.
  if (items.length) {
    state.items = items;
    appendBundle(col, items);
    const firstFit = items.find((i) => suits(i, described));
    if (!slot.item && firstFit) {
      slot.item = firstFit;
      state.target = described;
      stage.dataset.target = described;
      recalc(described);
    } else if (areaChanged) {
      recalc(described);
    }
  } else if (areaChanged) {
    recalc(described);
  }

  if (data.quote) state.quote = data.quote;
  renderAll();
}

/* Server calls ---------------------------------------------------------- */

async function sendMessage(message) {
  appendMessage('user', message);
  sendBtn.disabled = true;
  typingIndicator.classList.remove('hidden');
  renderClarifyingChips([]);

  try {
    const response = await fetch('/api/v1/chat', {
      method: 'POST',
      credentials: 'same-origin',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message }),
    });

    const data = await response.json();
    if (!response.ok) {
      appendMessage('assistant', (data.error && data.error.message) || 'Something went wrong.');
      return;
    }

    const col = appendMessage('assistant', data.reply || data.response || 'No response available.');
    renderGuardrail(data.guardrail);
    renderClarifyingChips(data.clarifying_questions);
    applyResult(data, col);

    if (/[؀-ۿ]/.test(message)) setLanguage('ar');
  } catch (error) {
    appendMessage('assistant', 'There was an error reaching the sales assistant.');
  } finally {
    sendBtn.disabled = false;
    typingIndicator.classList.add('hidden');
  }
}

function sendPrompt(event) {
  event.preventDefault();
  const message = promptInput.value.trim();
  if (!message) return;
  promptInput.value = '';
  autosize();
  sendMessage(message);
}

async function generateQuote() {
  const lines = SURFACES
    .map((s) => state.surfaces[s])
    .filter((s) => s.item && s.area > 0)
    .map((s) => ({ sku: s.item.sku, area_m2: s.area }));
  if (!lines.length) return;

  quoteBtn.disabled = true;
  try {
    state.quote = await postJson('/api/v1/quotes', { lines, requested_discount_percent: 0 });
    renderLedger();
    renderSteps();
  } catch (error) {
    appendMessage('assistant', `I could not generate that quotation (${error.message}).`);
  } finally {
    quoteBtn.disabled = false;
  }
}

/* UI wiring ------------------------------------------------------------- */

function setLanguage(lang) {
  htmlRoot.setAttribute('lang', lang);
  htmlRoot.setAttribute('dir', lang === 'ar' ? 'rtl' : 'ltr');
  $('langEn').classList.toggle('active', lang === 'en');
  $('langAr').classList.toggle('active', lang === 'ar');
}

function autosize() {
  promptInput.style.height = 'auto';
  promptInput.style.height = `${Math.min(promptInput.scrollHeight, 100)}px`;
}

function runAction(action) {
  if (action === 'texture') setMode('zoom');
  else if (action === 'compare') setMode('compare');
  else if (action === 'finishes') sendMessage('What finishes do you offer?');
  else if (action === 'stock') {
    const item = state.surfaces[state.target].item || state.items[0];
    sendMessage(item ? `Is ${item.name} (${item.sku}) in stock for my project?` : 'Which tiles do you have in stock?');
  }
}

document.querySelectorAll('.lang').forEach((btn) =>
  btn.addEventListener('click', () => setLanguage(btn.dataset.lang)));

document.querySelectorAll('.tier').forEach((btn) =>
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tier').forEach((t) => t.classList.toggle('active', t === btn));
    const amount = Number(btn.dataset.amount);
    sendMessage(btn.dataset.tier === 'value'
      ? `My total budget is under $${amount}.`
      : `My total budget is around $${amount}.`);
  }));

document.querySelectorAll('#quickChips .chip').forEach((chip) =>
  chip.addEventListener('click', () => sendMessage(chip.dataset.prompt)));

document.querySelectorAll('[data-action]').forEach((btn) =>
  btn.addEventListener('click', () => runAction(btn.dataset.action)));

document.querySelectorAll('.vtab').forEach((tab) =>
  tab.addEventListener('click', () => setMode(tab.dataset.mode)));

document.querySelectorAll('.seg-btn').forEach((btn) =>
  btn.addEventListener('click', () => setTarget(btn.dataset.surface)));

room.querySelectorAll('.face[data-surface]').forEach((face) =>
  face.addEventListener('click', () => {
    // On the balcony the floor is the balcony; the "walls" are open sky.
    if (state.target === 'balcony') {
      if (face.dataset.surface === 'floor') setTarget('balcony');
      return;
    }
    setTarget(face.dataset.surface);
  }));

divider.addEventListener('input', () => stage.style.setProperty('--split', `${divider.value}%`));

$('resetBtn').addEventListener('click', () => {
  SURFACES.forEach((s) => { state.surfaces[s] = newSurface(); });
  state.userAssigned = false;
  state.quote = null;
  setMode('room');
  renderAll();
});

quoteBtn.addEventListener('click', generateQuote);
$('printBtn').addEventListener('click', () => window.print());
$('shareBtn').addEventListener('click', async () => {
  const label = $('shareLabel');
  try {
    await navigator.clipboard.writeText(window.location.href);
    label.textContent = 'Copied';
  } catch (error) {
    label.textContent = 'Copy failed';
  }
  setTimeout(() => { label.textContent = 'Share'; }, 1800);
});

chatForm.addEventListener('submit', sendPrompt);
promptInput.addEventListener('input', autosize);
promptInput.addEventListener('keydown', (event) => {
  if (event.key === 'Enter' && !event.shiftKey) sendPrompt(event);
});

// Room depth and perspective follow the stage size so the box keeps its shape.
new ResizeObserver(() => {
  const h = stage.clientHeight;
  room.style.setProperty('--d', `${Math.round(h * 1.5)}px`);
  room.style.setProperty('--p', `${Math.round(h * 1.05)}px`);
}).observe(stage);

fetch('/api/v1/product-images', { credentials: 'same-origin' })
  .then((r) => (r.ok ? r.json() : {}))
  .then((photos) => { state.photos = photos; renderAll(); })
  .catch(() => {}); // photos are an enhancement; generated textures remain

appendMessage('assistant', 'Hello! Tell me about your project — room, floor and walls, and the area in m² — and I will curate surfaces you can try on the room visualizer.');
renderAll();
