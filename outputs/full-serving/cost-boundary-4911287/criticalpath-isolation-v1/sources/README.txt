CS471：單因素失敗隔離與 CPU 分析修正（審查後選取程式）

此目錄是可審查的選取來源，不是完整 serving 部署或主測量啟動器。
原始 traces、logs、prompts、outputs、token arrays 與私有逐檔雜湊收據不在此目錄。
每個來源檔的原始 commit 與檔案 SHA256 見 ../code-provenance.json。

CPU 測試：在此 sources 目錄，以 PYTHONPATH=. python -m pytest -q tests 執行。
已核對 27 項 CPU 測試。測試不代表 GPU 或原設定 charged admission 通過。

boundary_cuda_event_analysis：查詢式 stream 身分與 native CUDA API correlation；索引加速保留既有 ambiguity／tie 規則。
boundary_gated_capture：原始捕捉邏輯及線性 CPU 切片。這是保留的診斷來源，不宣稱捕捉穩定性已修復。
boundary_output_wait_isolation：在輸出產生端等待既有 copy_event；該全模型診斷仍失敗。不得作為建議的 serving 優化。
boundary_native_record_reproducer：小型 native48層 map/record、LogicalHistory、CpuGpuEvent、CUDA graph 和輸出拷貝案例。GPU 執行需原實驗對應的 vLLM/PyTorch、八 rank Slurm 環境與原生函式；未包含真實模型 attention/MoE、完整 planner／weight transfer、allocator/KV 壓力或完整非阻塞 readiness 行為。

其他檔案為上述測試與重現案例的必要支援。檔案來自不同明示來源版本，並非可直接執行的完整實驗快照。完整部署與失敗紀錄保留於私有研究環境。
