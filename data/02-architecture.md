# 系統架構與檔案結構

```
Reinforcement_Learning/
├── game/                     # pokesurvivor 原始碼（clone 自 github.com/JPTER78/pokesurvivor）
│   ├── rl_harness.html       # dev.html 的副本，載入 js/rl_boot.js 取代一般開機流程
│   │                         # （跳過登入/性格測驗/選單）
│   └── js/
│       ├── game.js           # 主迴圈；debugSpeed(k) 用來加速模擬時間
│       ├── rl_boot.js        # ★ Python ↔ JS 的橋接層，見下方說明
│       ├── systems/levelup.js
│       └── ui/run-ui.js      # showLevelUp/showItemChoice/showBond 在 rl_boot.js 中被覆寫
├── rl/                       # Python 端
│   ├── env.py                 # PokeSurvivorEnv(gymnasium.Env)：用 Playwright 開瀏覽器分頁、
│   │                           # 呼叫 window.RL.step(...)，組裝 79 維觀察向量、計算 reward
│   ├── baseline.py             # 隨機 / 原地不動 baseline 量測腳本
│   ├── train.py                 # PPO 訓練入口（SubprocVecEnv 平行環境）
│   ├── play.py                   # 載入訓練好的模型，開視窗 A（真實瀏覽器）+ 視窗 B（telemetry）
│   ├── smoke_test.py              # 快速驗證 env 能不能跑的手動測試腳本
│   ├── models/                     # 訓練完成的模型（.zip）
│   └── tb_logs/                     # TensorBoard 訓練紀錄
├── requirements.txt
└── data/                      # 本文件夾：專案規劃與使用說明
```

## `window.RL` 橋接介面（`game/js/rl_boot.js`）

| 方法 | 說明 |
|---|---|
| `reset(dex=132)` | 重設/開始一場新 run，固定夥伴 dex（v1 固定百變怪=132），回傳初始觀察值 |
| `setSpeed(k)` | 呼叫 `G.Game.debugSpeed(k)`，`k>1` 加速模擬時間（訓練用；展示時設回 1） |
| `setInput(moveIdx, skillIdx)` | 設定移動方向（0-8）與是否切換招式（0=不切，1-4=招式槽） |
| `step(moveIdx, skillIdx, frameSkip=4)` | **主要介面**：設定動作 → 推進 `frameSkip` 個畫面更新 → 回傳觀察值；單一 Promise 呼叫完成，減少 Python↔JS 來回次數 |
| `getObs()` | 回傳目前觀察值（hp/maxHp/level/xp/xpNext/kills/active/dead/enemies[]/hazards[]） |
| `isDone()` | 這場 run 是否已經結束（死亡或回到選單） |

升級卡片／道具選擇／傳說連結選擇都被 `rl_boot.js` 覆寫成自動挑選（heuristic），
不會跳出 UI 等待點擊，所以 `step()` 永遠不會卡住。

## Python 端資料流

```
PokeSurvivorEnv.step(action)
  └─ page.evaluate("window.RL.step(moveIdx, skillIdx, frameSkip)")
       └─ (JS 端) 設定輸入 → requestAnimationFrame × frameSkip → 回傳 obs dict
  └─ _encode(obs) → 79 維 numpy 向量（給 PPO 當作 observation）
  └─ _reward(prev_obs, obs) → shaped reward（見 01-decisions.md）
```

## 觀察向量組成（79 維，`rl/env.py::_encode`）
- `hp/maxHp`、`level/50`、`xp/xpNext`、`time/1800`（4 維）
- 目前啟用招式 one-hot（4 維）
- `kills/200`（1 維）
- 最近 8 隻敵人：`dx, dy, hp_frac, boss, 存在flag` × 8（40 維）
- 最近 6 個地面警示圈：`dx, dy, 半徑, 倒數進度, 存在flag` × 6（30 維）
