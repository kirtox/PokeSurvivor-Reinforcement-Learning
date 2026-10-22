/* ============ rl_boot.js — arranque headless para entrenamiento RL ============
 * Sustituye a main.js: salta login/test de personalidad/menús y expone
 * window.RL para que Playwright/Python controle el juego y lea su estado.
 */
(() => {
  // 9 direcciones (8 + quieto), mismas que WASD/flechas combinadas.
  const DIRS = [[0, 0], [0, -1], [1, -1], [1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1]];
  let curAxis = [0, 0];

  // 1 si un disparo del jugador llegaría a `e` en línea recta; 0 si una pared, árbol o roca lo para
  // antes (mismo criterio que combat.js: World.projectileBlock). Se mira hasta el borde del enemigo.
  const LOS_STEP = 14;
  function losClear(pl, e) {
    const dx = e.x - pl.x, dy = e.y - pl.y, d = Math.hypot(dx, dy);
    const end = d - (e.r || 10);
    for (let s = LOS_STEP; s < end; s += LOS_STEP) {
      if (G.World.projectileBlock(pl.x + dx / d * s, pl.y + dy / d * s, 6)) return 0;
    }
    return 1;
  }

  // Velocidad (px/s) por diferencia de posición entre dos observaciones. Es idempotente: si se llama
  // varias veces en el mismo instante devuelve la última velocidad calculada, sin deformarla.
  const vel = new Map();
  function velOf(key, x, y, t) {
    let r = vel.get(key);
    if (!r || t < r.t) { r = { x, y, t, vx: 0, vy: 0 }; vel.set(key, r); return r; }
    const dt = t - r.t;
    if (dt > 0.05) {
      r.vx = Math.max(-400, Math.min(400, (x - r.x) / dt));
      r.vy = Math.max(-400, Math.min(400, (y - r.y) / dt));
      r.x = x; r.y = y; r.t = t;
    }
    return r;
  }
  function pruneVel(t) {
    if (vel.size < 300) return;
    for (const [k, r] of vel) if (k !== 'pl' && r.t < t - 3) vel.delete(k);
  }

  // Comportamientos de enemigo (los normales andan hacia ti y no tienen ninguno de estos).
  const BEHAVIORS = ['charger', 'ranged', 'bomber', 'shielder', 'jumper', 'thief', 'runner'];

  // Fase de ataque: aim = preparando (carga/lanza), dash = embistiendo, fuse = mecha de bomba,
  // jump = en salto; left = segundos que quedan de esa fase. Sirve igual para jefes (ai.cast).
  function enemyPhase(e) {
    const c = e.ai && e.ai.cast;
    let aim = e.phase === 'aim' ? 1 : 0, dash = e.phase === 'dash' ? 1 : 0, fuse = e.phase === 'fuse' ? 1 : 0;
    let left = aim || dash || fuse ? Math.max(0, e.pt || 0) : 0;
    if (c) {
      if (c.t < c.wind) { aim = 1; left = c.wind - c.t; }
      else if (c.id === 'dash' && c.t < c.wind + c.dur) { dash = 1; left = c.wind + c.dur - c.t; }
    }
    const jump = e.jump ? 1 : 0;
    if (jump) left = Math.max(left, e.jump.dur - e.jump.t);
    return { aim, dash, fuse, jump, left };
  }

  function enemyObs(pl, t) {
    const tg = pl.target;
    return G.EnemyMgr.all().filter(e => !e.dead).map(e => {
      const dx = e.x - pl.x, dy = e.y - pl.y;
      return { e, dx, dy, d2: dx * dx + dy * dy };
    }).sort((a, b) => a.d2 - b.d2).slice(0, 8).map(o => {
      const e = o.e, v = velOf('e' + e.id, e.x, e.y, t);
      return Object.assign({
        dx: o.dx, dy: o.dy, hp: e.hp / e.maxHp, boss: e.boss ? 1 : 0,
        fly: e.flying ? 1 : 0, los: losClear(pl, e), tgt: e === tg ? 1 : 0,
        vx: v.vx, vy: v.vy, beh: BEHAVIORS.indexOf(e.behavior),
        fury: e.ai && e.ai.fury ? 1 : 0, shield: e.shield > 0 ? 1 : 0
      }, enemyPhase(e));
    });
  }

  // Todos los enemigos vivos (no sólo los 8 más cercanos): total y cuántos hay en cada una de las
  // 8 direcciones de movimiento (mismo orden que DIRS[1..8]) a menos de 450 px.
  const DENSITY_R = 450;
  function densityObs(pl) {
    const sect = [0, 0, 0, 0, 0, 0, 0, 0];
    let total = 0;
    for (const e of G.EnemyMgr.all()) {
      if (e.dead) continue;
      total++;
      const dx = e.x - pl.x, dy = e.y - pl.y;
      if (dx * dx + dy * dy > DENSITY_R * DENSITY_R) continue;
      sect[(Math.round(Math.atan2(dy, dx) / (Math.PI / 4)) + 10) % 8]++;
    }
    return { total, sect };
  }

  // Estado propio: velocidad real, lentitud, invulnerabilidad, poder del altar, mejoras temporales y
  // charco bajo los pies (tipo, frenazo, daño).
  const POWER_IDS = ['dmg', 'spd', 'mag', 'cd'];
  const BUFF_KEYS = ['atk', 'patk', 'satk', 'spd', 'def', 'regen'];
  const LIQUID_KINDS = ['water', 'swamp', 'lava', 'poison', 'spark', 'ice'];
  function selfObs(pl, t) {
    const v = velOf('pl', pl.x, pl.y, t);
    const eff = pl.inLiquid ? G.World.liquidEffect(pl.mon.types) : {};
    const pw = pl.power;
    return {
      vx: v.vx, vy: v.vy, spd: pl.spd,
      slow: pl.slowT > 0 ? pl.slowT : 0, invuln: pl.invuln > 0 ? pl.invuln : 0,
      power: pw && pw.def ? POWER_IDS.indexOf(pw.def.id) : -1, powerLeft: pw && pw.dur ? pw.t / pw.dur : 0,
      buffs: BUFF_KEYS.map(k => { const b = pl.buffs[k]; return b && b.t > 0 ? b.n : 0; }),
      inLiquid: pl.inLiquid ? 1 : 0, liqSlow: eff.slow || 1, liqDmg: eff.dmg ? 1 : 0, liqSlide: eff.slide ? 1 : 0,
      liquid: LIQUID_KINDS.indexOf(G.World.liquidKind())
    };
  }

  // Manantiales, altares y cofres grandes cercanos (las trampas van aparte). state: carga del
  // manantial / progreso de apertura del cofre / 1 el altar. kind: poder del altar.
  function specialObs(pl) {
    const CHEST_TIME = 3.5;
    return G.World.specialsNear(pl.x, pl.y, 600).filter(p => (p.type === 'spring' || p.type === 'altar' || p.type === 'bigchest') && !p.used).map(p => {
      const dx = p.x - pl.x, dy = (p.type === 'spring' ? p.y - 8 : p.y) - pl.y;
      return {
        dx, dy, d2: dx * dx + dy * dy, type: p.type,
        state: p.type === 'spring' ? p.charge : p.type === 'bigchest' ? Math.min(1, p.open / CHEST_TIME) : 1,
        kind: p.type === 'altar' ? (p.kind | 0) : 0
      };
    }).sort((a, b) => a.d2 - b.d2).slice(0, 4).map(p => ({ dx: p.dx, dy: p.dy, type: p.type, state: p.state, kind: p.kind }));
  }

  // Distancia al primer charco (cualquiera) en cada una de las 8 direcciones; RAY_MAX si no hay
  // o si una pared se interpone.
  function liquidRays(pl) {
    const out = [];
    for (let i = 1; i < DIRS.length; i++) {
      const [dx, dy] = G.U.norm(DIRS[i][0], DIRS[i][1]);
      let d = RAY_MAX;
      for (let s = RAY_STEP; s <= RAY_MAX; s += RAY_STEP) {
        const x = pl.x + dx * s, y = pl.y + dy * s;
        if (G.World.isLiquid(x, y)) { d = s; break; }
        if (!G.World.isFree(x, y, pl.r, true)) break;
      }
      out.push(d);
    }
    return out;
  }

  // Clima actual: kind (-1 = ninguno), intensidad 0..1 y si te quita vida con él.
  const WEATHER_KINDS = ['sun', 'rain', 'sand', 'snow', 'fog'];
  function weatherObs(pl) {
    const w = G.Weather.current;
    if (!w) return { kind: -1, level: 0, hurts: 0 };
    return {
      kind: WEATHER_KINDS.indexOf(w.kind), level: Math.max(0, Math.min(1, w.t / 3, (w.dur - w.t) / 3)),
      hurts: G.Weather.hurts(pl.mon.types) ? 1 : 0
    };
  }

  // Grieta (portal a la arena de un legendario) y arena. Posiciones relativas al jugador.
  const RIFT_TTL = 45, ARENA_TIME = 120;
  function riftObs(pl) {
    const r = G.Rift.rift, a = G.Rift.arena;
    const o = { rift: 0, rdx: 0, rdy: 0, ttl: 0, arena: 0, left: 0, cdx: 0, cdy: 0, edge: 0, won: 0 };
    if (r) { o.rift = 1; o.rdx = r.x - pl.x; o.rdy = r.y - pl.y; o.ttl = Math.max(0, r.ttl) / RIFT_TTL; }
    if (a) {
      const cdx = a.x - pl.x, cdy = a.y - pl.y, rad = a.r * G.World.TS;
      o.arena = 1; o.left = G.Rift.timeLeft / ARENA_TIME; o.cdx = cdx; o.cdy = cdy;
      o.edge = Math.max(0, rad - Math.hypot(cdx, cdy)) / rad;
      o.won = a.phase === 'won' ? 1 : 0;
    }
    return o;
  }

  // Distancia del jugador a la zona peligrosa (0 = dentro), para ordenar por amenaza real.
  function hazardGap(h, pl) {
    if (h.kind === 'zone') return Math.max(0, Math.hypot(h.x - pl.x, h.y - pl.y) - h.r);
    const ca = Math.cos(h.angle), sa = Math.sin(h.angle);
    const px = pl.x - h.x, py = pl.y - h.y;
    const along = Math.max(0, Math.min(h.len, px * ca + py * sa));
    return Math.max(0, Math.hypot(px - ca * along, py - sa * along) - h.w / 2);
  }

  function hazardObs(pl) {
    return (G.Hazards.all || []).filter(h => !h.hit).map(h => {
      const beam = h.kind !== 'zone';
      return {
        dx: h.x - pl.x, dy: h.y - pl.y, gap: hazardGap(h, pl),
        r: beam ? (h.w || 26) / 2 : (h.r || 40), frac: Math.min(1, h.t / (h.delay || 1)),
        beam: beam ? 1 : 0, cos: beam ? Math.cos(h.angle) : 0, sin: beam ? Math.sin(h.angle) : 0,
        len: beam ? h.len : 0, left: Math.max(0, (h.delay || 0) - h.t),
        inside: G.Hazards.inside(h, pl.x, pl.y, pl.r) ? 1 : 0
      };
    }).sort((a, b) => a.gap - b.gap).slice(0, 6);
  }

  // Disparos enemigos en vuelo (los más cercanos): posición y velocidad relativas al jugador.
  function shotObs(pl) {
    return G.Projectiles.all().filter(p => !p.friendly).map(p => {
      const dx = p.x - pl.x, dy = p.y - pl.y;
      return { dx, dy, d2: dx * dx + dy * dy, vx: p.vx, vy: p.vy };
    }).sort((a, b) => a.d2 - b.d2).slice(0, 8).map(p => ({ dx: p.dx, dy: p.dy, vx: p.vx, vy: p.vy }));
  }

  // Trampas del suelo cercanas. kind: 0=veneno, 1=pegajosa, 2=explosiva.
  function trapObs(pl) {
    return G.World.specialsNear(pl.x, pl.y, 600).filter(p => p.type === 'trap').map(p => {
      const dx = p.x - pl.x, dy = p.y - 8 - pl.y;
      return { dx, dy, d2: dx * dx + dy * dy, kind: p.kind };
    }).sort((a, b) => a.d2 - b.d2).slice(0, 4).map(p => ({ dx: p.dx, dy: p.dy, kind: p.kind }));
  }

  // 1 si el movimiento activo es de proyectil/rayo y el blanco que el juego eligió está tapado.
  function aimBlockedObs(pl) {
    const m = pl.activeMove();
    if (!m || (m.kind !== 'projectile' && m.kind !== 'beam')) return 0;
    const t = pl.target;
    return t && t.hurt && !t.dead && !losClear(pl, t) ? 1 : 0;
  }

  // Orbes de experiencia y objetos útiles cercanos. Sin esto el agente no puede
  // "ver" la experiencia y sólo aprende a esquivar. kind: 1=xp, 2=xpBig, 3=heal, 4=magnet.
  const PICK_KIND = { xp: 1, xpBig: 2, heal: 3, magnet: 4 };
  function pickupObs(pl) {
    return (G.Pickups.all() || []).filter(p => PICK_KIND[p.kind]).map(p => {
      const dx = p.x - pl.x, dy = p.y - pl.y;
      return { dx, dy, d2: dx * dx + dy * dy, kind: PICK_KIND[p.kind] };
    }).sort((a, b) => a.d2 - b.d2).slice(0, 6).map(p => ({ dx: p.dx, dy: p.dy, kind: p.kind }));
  }

  // Hasta 4 huecos de movimiento: tipo de ataque, nivel, tamaño del grupo (vínculo),
  // si está listo para lanzarse y su daño/recarga base. Sin esto el agente no sabe
  // qué hace cada hueco y cambiar de movimiento es puro azar.
  function moveObs(pl) {
    return pl.slots.slice(0, 4).map(g => {
      const m = g[0];
      const full = Math.max(0.12, m.cd * pl.cdMul);
      return {
        kind: m.kind, lvl: m.lvl / m.def.maxLvl, size: g.length,
        ready: m.kind === 'orbit' ? 1 : 1 - Math.max(0, Math.min(1, m.t / full)),
        dmg: m.dmg || 0, cd: m.cd || 0
      };
    });
  }

  // Carta de subida de nivel -> números. type: new | up | evo | stat.
  // Las cartas de stat que desbloquean una evolución se pintan en dorado (#ffd23f).
  function cardObs(pl, c) {
    const id = c.key.split(':')[1];
    const o = {
      type: c.type, stat: c.type === 'stat' ? id : null,
      evoStat: c.type === 'stat' && c.color === '#ffd23f' ? 1 : 0,
      kind: null, lvl: 0, dmg: 0, cd: 0, title: c.title
    };
    if (c.type !== 'stat') {
      const def = G.Moves.BY_ID[id], own = pl.moves.find(m => m.id === id);
      if (def) { o.kind = def.kind; o.dmg = (own || def).dmg || 0; o.cd = (own || def).cd || 0; }
      if (own) o.lvl = own.lvl / own.def.maxLvl;
    }
    return o;
  }

  // Elección de carta de nivel pendiente: { list, cb }. Mientras exista, el juego
  // está parado (state 'levelup') y espera a que step() traiga el índice elegido.
  let pending = null;
  let lastPick = null;

  // Terreno: distancia libre en cada una de las 8 direcciones de movimiento (mismo orden que
  // DIRS[1..8]) hasta la primera pared, roca/árbol/cofre o charco que daña (lava, veneno,
  // descarga). El agua y el pantano frenan pero no bloquean, así que no cuentan.
  const RAY_MAX = 192, RAY_STEP = 12;
  function blockedAt(pl, x, y) {
    if (!G.World.isFree(x, y, pl.r, true)) return true;
    return G.World.isLiquid(x, y) && !!G.World.liquidEffect(pl.mon.types).dmg;
  }
  function terrainObs(pl) {
    const out = [];
    for (let i = 1; i < DIRS.length; i++) {
      const [dx, dy] = G.U.norm(DIRS[i][0], DIRS[i][1]);
      let d = RAY_MAX;
      for (let s = RAY_STEP; s <= RAY_MAX; s += RAY_STEP) {
        if (blockedAt(pl, pl.x + dx * s, pl.y + dy * s)) { d = s - RAY_STEP; break; }
      }
      out.push(d);
    }
    return out;
  }

  // 1 si el último step() pedía moverse pero el jugador avanzó menos del 30% de lo esperado
  // (pegado a una pared u objeto). Deslizar en diagonal contra una pared no cuenta.
  let stuck = 0;
  // Trampas que han saltado lo bastante cerca como para afectar al jugador (acumulado de la partida).
  let trapHits = 0;

  function getObs() {
    const pl = G.Game.player;
    if (!pl) return null;
    const t = G.Game.st.time;
    pruneVel(t);
    return {
      time: t,
      state: G.Game.state,
      hp: pl.hp, maxHp: pl.maxHp,
      level: pl.level, xp: pl.xp, xpNext: pl.xpNext,
      kills: pl.kills, bosses: pl.bosses,
      active: pl.active, numSlots: pl.slots.length, maxMoves: pl.maxMoves,
      dead: pl.dead,
      enemies: enemyObs(pl, t),
      density: densityObs(pl),
      self: selfObs(pl, t),
      specials: specialObs(pl),
      liqRays: liquidRays(pl),
      weather: weatherObs(pl),
      rift: riftObs(pl),
      hazards: hazardObs(pl),
      shots: shotObs(pl),
      traps: trapObs(pl),
      aimBlocked: aimBlockedObs(pl),
      trapHits,
      pickups: pickupObs(pl),
      moves: moveObs(pl),
      cards: pending ? pending.list.map(c => cardObs(pl, c)) : [],
      picked: lastPick,
      rays: terrainObs(pl),
      rayMax: RAY_MAX,
      stuck
    };
  }

  /** Aplica la carta `idx` (se recorta al rango) y reanuda la partida. */
  function resolveCard(idx) {
    if (!pending) return;
    const { list, cb } = pending;
    pending = null;
    const card = list[Math.max(0, Math.min(list.length - 1, idx | 0))];
    lastPick = { title: card.title, type: card.type };
    cb(card);
  }

  // Las cartas de nivel las elige el agente (ver step). Objeto y vínculo siguen
  // siendo automáticos: se resuelven en el acto para no bloquear step().
  function deferLevelUp(level, list, cb) { pending = { list, cb }; }
  function autoPickItem(cur, next, cb) { cb(next); }
  function autoPickBond(opt, cb) { cb(opt.mode === 'add' ? [opt.loose[0]] : [opt.loose[0], opt.loose[1]]); }

  function boot() {
    G.I18n.startDom();
    G.UI.initChrome();
    G.Game.init();
    G.DB.playAsGuest();
    // La pantalla "Cargando…" (#boot, fija a pantalla completa) la oculta normalmente
    // G.Flow.boot(), que aquí no se ejecuta: sin esto tapa el juego en la ventana visible.
    const bootEl = document.getElementById('boot');
    if (bootEl) bootEl.classList.add('hidden');

    // El movimiento lo decide Python/Playwright, no el teclado real.
    G.Input.axis = () => curAxis;
    // Las cartas de nivel las decide el agente (se quedan en `pending` hasta el
    // siguiente step). Objeto y vínculo se auto-eligen: si esperasen un clic real,
    // step() se quedaría colgado para siempre.
    G.RunUI.showLevelUp = deferLevelUp;
    G.RunUI.showItemChoice = autoPickItem;
    G.RunUI.showBond = autoPickBond;

    // Cuenta las trampas que saltan dentro del radio que afecta al jugador (TRAP_R * 0.7 en interact.js).
    const origTrap = G.Interact.trapEffect;
    G.Interact.trapEffect = function (p) {
      const pl = G.Game.player;
      if (pl && !pl.dead && Math.hypot(p.x - pl.x, p.y - 8 - pl.y) < 74 * 0.7) trapHits++;
      return origTrap.apply(this, arguments);
    };

    window.RL = {
      /** @param dex  Pokémon inicial  @param shiny  versión shiny  @param upgrades  { id -> nivel } de G.Upgrades (mejoras permanentes) */
      reset(dex = 132, shiny = false, upgrades = {}) {
        pending = null; lastPick = null; stuck = 0; trapHits = 0; vel.clear();
        if (G.Game.state !== 'ui' && G.Game.state !== 'over') G.Game.quit();
        curAxis = [0, 0];
        // El shiny sólo se aplica si la cuenta lo posee: se concede aquí (guest local).
        if (shiny) G.DB.grantShiny(dex, 'rl');
        G.DB.save.partner = dex;
        G.DB.save.partnerShiny = !!shiny;
        // G.Upgrades.perks() recorta cada nivel a su máximo, así que no hace falta validar aquí.
        G.DB.save.pupg = G.DB.save.pupg || {};
        G.DB.save.pupg[dex] = Object.assign({}, upgrades);
        G.DB.save.pbind = {};
        G.DB.save.pban = {};
        G.Game.startRun();
        return getObs();
      },
      /** k=1 tiempo real; k>1 adelanta la simulación (entrenamiento sin pantalla). */
      setSpeed(k) { G.Game.debugSpeed(k); },
      /** @param moveIdx 0-8 (9 direcciones) @param skillIdx 0 = no cambiar, 1-4 = hueco */
      setInput(moveIdx, skillIdx) {
        const [x, y] = DIRS[moveIdx] || [0, 0];
        curAxis = G.U.norm(x, y);
        const pl = G.Game.player;
        if (pl && skillIdx > 0 && skillIdx <= pl.slots.length) pl.setActive(skillIdx - 1);
      },
      getObs,
      isDone() { return !G.Game.player || G.Game.player.dead || G.Game.state === 'over'; },
      /**
       * Un paso de entrenamiento: aplica la carta pendiente (si hay), fija la acción,
       * avanza `frameSkip` fotogramas reales (con setSpeed ya acelerados) y devuelve
       * la observación resultante. Todo en una sola llamada para no pagar ida-vuelta
       * Python↔JS por fotograma. Si aparece una elección de carta, devuelve enseguida
       * (el juego está parado) para que el agente decida en el siguiente paso.
       * @param cardIdx  0-2, sólo se usa si hay una carta de nivel pendiente
       */
      step(moveIdx, skillIdx, frameSkip = 4, cardIdx = 0) {
        lastPick = null;
        resolveCard(cardIdx);
        window.RL.setInput(moveIdx, skillIdx);
        const p0 = G.Game.player;
        const x0 = p0 ? p0.x : 0, y0 = p0 ? p0.y : 0, t0 = G.Game.st.time;
        const wantsMove = moveIdx > 0;
        return new Promise(resolve => {
          let n = 0;
          const finish = () => {
            const pl = G.Game.player;
            const dt = G.Game.st.time - t0;
            stuck = 0;
            if (pl && wantsMove && dt > 0.05) {
              const moved = Math.hypot(pl.x - x0, pl.y - y0);
              stuck = moved < 0.3 * pl.spd * dt ? 1 : 0;
            }
            resolve(getObs());
          };
          const tick = () => {
            n++;
            if (n >= frameSkip || window.RL.isDone() || pending) finish();
            else requestAnimationFrame(tick);
          };
          requestAnimationFrame(tick);
        });
      }
    };
  }

  if (document.readyState === 'loading') addEventListener('DOMContentLoaded', boot);
  else boot();
})();
