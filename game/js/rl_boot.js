/* ============ rl_boot.js — arranque headless para entrenamiento RL ============
 * Sustituye a main.js: salta login/test de personalidad/menús y expone
 * window.RL para que Playwright/Python controle el juego y lea su estado.
 */
(() => {
  // 9 direcciones (8 + quieto), mismas que WASD/flechas combinadas.
  const DIRS = [[0, 0], [0, -1], [1, -1], [1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1]];
  let curAxis = [0, 0];

  function enemyObs(pl) {
    return G.EnemyMgr.all().filter(e => !e.dead).map(e => {
      const dx = e.x - pl.x, dy = e.y - pl.y;
      return { dx, dy, d2: dx * dx + dy * dy, hp: e.hp / e.maxHp, boss: e.boss ? 1 : 0 };
    }).sort((a, b) => a.d2 - b.d2).slice(0, 8).map(e => ({ dx: e.dx, dy: e.dy, hp: e.hp, boss: e.boss }));
  }

  function hazardObs(pl) {
    return (G.Hazards.all || []).filter(h => !h.hit).map(h => {
      const dx = h.x - pl.x, dy = h.y - pl.y;
      return { dx, dy, d2: dx * dx + dy * dy, r: h.r || h.w || 40, frac: Math.min(1, h.t / (h.delay || 1)) };
    }).sort((a, b) => a.d2 - b.d2).slice(0, 6).map(h => ({ dx: h.dx, dy: h.dy, r: h.r, frac: h.frac }));
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

  function getObs() {
    const pl = G.Game.player;
    if (!pl) return null;
    return {
      time: G.Game.st.time,
      state: G.Game.state,
      hp: pl.hp, maxHp: pl.maxHp,
      level: pl.level, xp: pl.xp, xpNext: pl.xpNext,
      kills: pl.kills, bosses: pl.bosses,
      active: pl.active, numSlots: pl.slots.length, maxMoves: pl.maxMoves,
      dead: pl.dead,
      enemies: enemyObs(pl),
      hazards: hazardObs(pl),
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

    window.RL = {
      /** @param dex  Pokémon inicial  @param shiny  versión shiny  @param upgrades  { id -> nivel } de G.Upgrades (mejoras permanentes) */
      reset(dex = 132, shiny = false, upgrades = {}) {
        pending = null; lastPick = null; stuck = 0;
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
