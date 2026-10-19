# 執行指南：baseline / 訓練 / 展示

所有指令都在專案根目錄 `C:\Python\Reinforcement_Learning` 底下執行，直接呼叫 venv
裡的 python（不需要額外 activate）。訓練/推論本身不需要連網，只有第一次安裝套件時
需要公司 proxy（見 `/memories/network.md`：`--proxy "http://proxy-dmz.intel.com:912"`）。

## 0. 先決條件（已完成，供之後重建環境參考）
```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt --proxy "http://proxy-dmz.intel.com:912"
$env:HTTPS_PROXY = "http://proxy-dmz.intel.com:912"; $env:HTTP_PROXY = "http://proxy-dmz.intel.com:912"
.\.venv\Scripts\python.exe -m playwright install chromium
```

## 1. 快速驗證環境是否正常（非必要，已驗證過一次）
```powershell
.\.venv\Scripts\python.exe -m rl.smoke_test
```

## 2. Baseline（已跑過一次，結果記錄在 `01-decisions.md`）
```powershell
.\.venv\Scripts\python.exe -m rl.baseline
```
會印出「隨機動作」和「原地不動只切招式」兩組 10 回合的平均存活時間，
之後訓練完的 agent 要拿這兩個數字當比較基準。

## 3. 啟動 30 萬步訓練試驗（下一步要做的事）
```powershell
.\.venv\Scripts\python.exe -m rl.train --timesteps 300000 --n-envs 6 --name trial_v1_300k
```
- `--n-envs 6`：這台機器 8 個邏輯核心，留一些餘裕給主行程和你自己的桌面瀏覽器。
- 預估：先前小規模試跑（2 環境、4096 步）量到含啟動開銷在內約 27 fps；30 萬步在
  6 環境下粗估 1-2 小時，實際時間請用 TensorBoard 的 `fps` 數字即時校正。
- 這個指令會直接佔用終端機前景執行；若要背景執行，用工具內建的背景模式啟動即可，
  完成後模型會存到 `rl/models/trial_v1_300k.zip`，訓練紀錄在 `rl/tb_logs/`。

### 訓練中怎麼看學習曲線
另開一個終端機：
```powershell
.\.venv\Scripts\python.exe -m tensorboard.main --logdir rl/tb_logs
```
然後瀏覽器開 http://localhost:6006，看 `rollout/ep_rew_mean`（平均回合獎勵）跟
`rollout/ep_len_mean`（平均回合長度，粗略對應存活時間）是否隨步數上升。

### 判斷要不要加碼訓練
- 如果 `ep_rew_mean` 明顯上升、且推算存活時間超過 baseline（32.7s）數倍 → 值得加碼到
  完整的 150 萬步：`--timesteps 1500000 --n-envs 6 --name ppo_ditto_v1`。
- 如果完全平坦或震盪不收斂 → 先不要加長訓練時間，回頭檢查 reward/環境設計
  （見 `01-decisions.md` 的 reward 設計、`02-architecture.md` 的觀察向量組成）。

## 4. 用訓練好的模型展示（視窗 A + 視窗 B）
```powershell
.\.venv\Scripts\python.exe -m rl.play --model rl/models/trial_v1_300k
```
- 會開一個真實的 Chromium 視窗（視窗 A）用真實速度播放遊戲，agent 即時操作；
- 同時開一個 pygame 小視窗（視窗 B），顯示 reward 曲線、血量/等級/存活時間、
  以及目前動作（移動方向箭頭 + 目前啟用招式高亮）。
- 關閉 pygame 視窗或按視窗的關閉鈕即可結束。

## 常見問題
- **訓練跑很久都沒印出東西**：SubprocVecEnv 每個子行程要重新載入 torch/stable-baselines3
  並啟動自己的 Chromium，第一次 rollout 收集完（預設 PPO `n_steps=2048` × `n_envs`）才會
  印出第一行統計，屬正常現象，不是卡住。
- **要重跑/清掉某次訓練結果**：`rl/models/<name>.zip` 和 `rl/tb_logs/<name>_*/` 直接刪掉即可。
