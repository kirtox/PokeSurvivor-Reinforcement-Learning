# 核心決策（透過 /grilling 訪談確認）

## 1. 遊戲版本：本地 clone，不碰正式站
- 原始碼：github.com/JPTER78/pokesurvivor，PolyForm Noncommercial 1.0.0 授權，非商用研究使用沒問題。
- 正式站 pokesurvivor.com 有排行榜（撐多久），自動化腳本若打正式站會污染排行榜 —— 避免。
- 遊戲本身是純 HTML + CSS + 原生 JS（Canvas 2D），無建置工具，可直接用 `file://` 離線執行，
  存檔只在 `localStorage`，不需要登入/網路。

## 2. 可程式步進的遊戲迴圈（而非純即時）
- 原始迴圈用 `requestAnimationFrame` + `G.Game.debugSpeed(k)`：`k` 會乘上每幀的 `dt`，
  只要畫面維持在 <1/30s 一幀，整體模擬速度就能提升到約 `k` 倍。
- 在 `game/js/rl_boot.js` 加了 `window.RL.step(moveIdx, skillIdx, frameSkip)`：
  一次 Promise 呼叫內完成「設定動作 → 推進 frameSkip 個 rAF → 回傳觀察值」，
  避免每一幀都要 Python↔JS 來回一次造成的額外延遲。

## 3. 觀察值：結構化向量，不用像素
- 不用 CNN/截圖，直接從遊戲物件讀數值：玩家血量/等級/經驗、目前招式、
  最近 8 隻敵人（相對座標/血量/是否 boss）、最近 6 個地面警示圈（相對座標/半徑/倒數進度）。
- 原因：訓練樣本效率高很多，且配合「無畫面加速」不用渲染截圖。

## 4. 動作空間
- `MultiDiscrete([9, 5])`：9 = 8 方向 + 不動；5 = 不切招式 + 招式 1-4。
- Frame skip = 4（約等於人類 15 次/秒操作頻率）。

## 5. 獎勵函數（shaped reward）
- 每步小額存活獎勵 + 升級/擊殺加分 + 損血懲罰（依血量比例）+ 死亡大額懲罰。
- 最終評估仍看存活時間（跟遊戲排行榜指標一致），但訓練訊號用 shaped 版本加速收斂。

## 6. v1 範圍縮小（避免一次處理全部複雜度）
- 固定角色：**百變怪（Ditto, dex 132）**。確認過在這個遊戲裡它沒有「變身」特殊機制，
  純一般系招式池（tackle/quick-attack/body-slam/hyper-voice + 增益技），機制單純。
- 升級三選一卡片／道具選擇／傳說連結：**不交給 agent 決策**，由 `rl_boot.js` 內建heuristic
  自動選（優先選血量 HP / 防禦 DEF 卡），避免訓練時卡在等待 UI 點擊。
- 關閉：天氣、裂隙 Boss 戰、連線合作、轉蛋。

## 7. 演算法與框架
- **PPO**（Stable-Baselines3）+ **Gymnasium** 自訂環境 + **Playwright** 橋接本地 JS 遊戲。
- 沒有 GPU（本機只有 CPU），但結構化向量 + MLP 的 PPO 在 CPU 上完全可行；
  真正的瓶頸是 Playwright headless 瀏覽器的 step 呼叫，不是神經網路運算本身。

## 8. 成功標準／訓練預算
- 先建立 baseline（隨機動作、原地不動）作比較基準，而不是盲目看「存活幾秒」這個數字。
- 小規模試跑（幾十萬到百萬步級）先看學習曲線是否上升，再決定要不要加碼長時間訓練。
- v1 目標：明顯超越 baseline + 看得到主動閃避行為，**不是**「破關」或「衝到某個名次」。
- 「存活 30 分鐘」列為 v2/v3 的長期目標（因為這類遊戲敵人強度隨時間上升，光靠閃避、
  沒有角色強化跟不上強度曲線，所以 30 分鐘需要先讓 agent 也能控制升級選卡）。

## 9. 視覺化（進階需求）
- **視窗 A**：headed（有畫面）的 Playwright Chromium 視窗，直接顯示本地遊戲的真實 Canvas
  畫面，用訓練好的 policy 以真實速度（speed=1）操作 —— 這就是使用者要的「另開視窗呈現網頁遊戲畫面」。
- **視窗 B**：Python/pygame 遙測儀表板，顯示 reward 曲線、存活時間/等級/血量，
  以及**目前動作的視覺化**（移動方向箭頭 + 目前啟用招式高亮），依使用者要求加碼的細節。

## 事實查核紀錄
- Ditto 在此遊戲中沒有特殊「變身」實作，走一般系招式池（`js/data/starters.js` /
  `js/data/moves.js` 確認過）。
- 升級卡選擇若不處理會真的卡住遊戲（`state = 'levelup'` 會暫停 `update(dt)`，等待
  `G.RunUI.showLevelUp` 的回呼），已在 `rl_boot.js` 覆寫處理。
