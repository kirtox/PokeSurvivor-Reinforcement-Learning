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

  function getObs() {
    const pl = G.Game.player;
    if (!pl) return null;
    return {
      time: G.Game.st.time,
      state: G.Game.state,
      hp: pl.hp, maxHp: pl.maxHp,
      level: pl.level, xp: pl.xp, xpNext: pl.xpNext,
      kills: pl.kills, bosses: pl.bosses,
      active: pl.active, numSlots: pl.slots.length,
      dead: pl.dead,
      enemies: enemyObs(pl),
      hazards: hazardObs(pl)
    };
  }

  // v1: nada de agente para esto — se elige por heurística para no bloquear step().
  function autoPickLevelUp(level, list, cb) {
    const pick = list.find(c => c.key === 'stat:hp') || list.find(c => c.key === 'stat:def') || list[0];
    cb(pick);
  }
  function autoPickItem(cur, next, cb) { cb(next); }
  function autoPickBond(opt, cb) { cb(opt.mode === 'add' ? [opt.loose[0]] : [opt.loose[0], opt.loose[1]]); }

  function boot() {
    G.I18n.startDom();
    G.UI.initChrome();
    G.Game.init();
    G.DB.playAsGuest();

    // El movimiento lo decide Python/Playwright, no el teclado real.
    G.Input.axis = () => curAxis;
    // Las cartas (subida de nivel, objeto, vínculo) se auto-eligen: si esperasen
    // un clic real, step() se quedaría colgado para siempre.
    G.RunUI.showLevelUp = autoPickLevelUp;
    G.RunUI.showItemChoice = autoPickItem;
    G.RunUI.showBond = autoPickBond;

    window.RL = {
      /** @param dex  Pokémon fijo para la v1 (132 = Ditto). */
      reset(dex = 132) {
        if (G.Game.state !== 'ui' && G.Game.state !== 'over') G.Game.quit();
        curAxis = [0, 0];
        G.DB.save.partner = dex;
        G.DB.save.partnerShiny = false;
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
       * Un paso de entrenamiento: fija la acción, avanza `frameSkip` fotogramas
       * reales (con setSpeed ya acelerados) y devuelve la observación resultante.
       * Todo en una sola llamada para no pagar ida-vuelta Python↔JS por fotograma.
       */
      step(moveIdx, skillIdx, frameSkip = 4) {
        window.RL.setInput(moveIdx, skillIdx);
        return new Promise(resolve => {
          let n = 0;
          const tick = () => {
            n++;
            if (n >= frameSkip || window.RL.isDone()) resolve(getObs());
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
