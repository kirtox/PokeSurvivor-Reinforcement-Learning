# PokeSurvivor RL — 專案文件索引

這個 `data/` 資料夾整理了透過 `/grilling` 訪談確認的專案方案、目前實作狀態、
以及後續 v2/v3 的規劃。

- [01-decisions.md](./01-decisions.md) — 核心決策與理由（為什麼選這些技術方案）
- [02-architecture.md](./02-architecture.md) — 目前的系統架構與檔案結構
- [03-roadmap-v2-v3.md](./03-roadmap-v2-v3.md) — v2、v3 預計要做的事
- [04-training-guide.md](./04-training-guide.md) — 如何執行 baseline / 訓練 / 展示視窗

## 目前狀態（2026-10-19）
- v1 環境、訓練管線、baseline 已完成並驗證可跑通。
- Baseline 結果：隨機動作平均存活 **32.7 秒**；原地不動只切招式平均存活 **27.9 秒**。
- 下一步：執行一次 6 個平行環境、30 萬步的小規模訓練試驗，觀察學習曲線是否優於 baseline。
