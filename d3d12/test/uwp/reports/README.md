# Xbox test app reports

Series X, OS 10.0.26100.9445, 2026-10-18 (console clock). Each run was deployed and launched through Device Portal. The Dev Home app type was checked on the console.

| file | package | capabilities | Dev Home app type | result |
|---|---|---|---|---|
| `A_hevc_app_bench_first.txt` | 1.0.4.0 | `hevcPlayback` | App | media-app tier; device hung on the first decode after the micro-benchmarks |
| `B_hevc_game.txt` | 1.0.4.0 | `hevcPlayback` | Game | media-app tier, no Game Mode; same hang |
| `C_none_game.txt` | 1.0.4.1 | none | Game | Game Mode; all pass |
| `D_expanded_app.txt` | 1.0.4.2 | `expandedResources` | App | Game Mode; all pass |
| `E_both_game.txt` | 1.0.4.3 | `hevcPlayback` + `expandedResources` | Game | Game Mode; all pass |
| `diag_hevc_nobench_xboxfirst.txt` | 1.0.4.4 (diagnostic) | `hevcPlayback` | App | micro-benchmarks skipped: all pass |
| `A_hevc_app_fixed.txt` | 1.0.4.5 | `hevcPlayback` | App | micro-benchmarks moved last: all pass |

`sept_hevc_build_1.0.1.3-1.0.2.9.txt` was left on the console by a September build. Every build from that range declares only `hevcPlayback`, yet it reports the Game tier (4147 MB, 5.95 TFLOPS). Its Dev Home app type and OS version weren't recorded. On today's OS, `hevcPlayback` with the Game app type gives the media-app tier (`B_hevc_game.txt`), so this run is unexplained.
