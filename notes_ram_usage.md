# RAM usage

Measured on this Mac mini (24 GB) on 4 Oct 2026.

Loaded size is what the runtime reports in `/api/ps` while the model is in memory. Weight size is the file size from `/api/tags`. For the Ollaya GGUF and ONNX models here, those two numbers match. Qwen’s loaded size is larger than the file because the context window sits on top of the weights.

Jev runs on TypeSafe’s servers, so it uses no local RAM.

## Loaded right now

`GET http://127.0.0.1:11435/api/ps`

| Model | Loaded | Device | Context | Bytes |
|---|---:|---|---:|---:|
| Winnow 12B (`winnow:12b`) | 11.8 GiB (12.7 GB) | metal | 8192 | 12,669,660,829 |

`size` and `size_vram` are the same, so the whole allocation is on the GPU. While this model is loaded, macOS reports about 19.2 GB in use and 1.3 GB available.

Ollama has nothing loaded (`GET http://127.0.0.1:11434/api/ps`).

## Every system in this POC

| System | Tag | Where | Weight size | Loaded, when seen |
|---|---|---|---:|---|
| Qwen | `qwen3:8b` | Ollama | 4.9 GiB | About 5.5 GiB with a 4096 context, earlier in this project |
| Jev | `jev-latest` | TypeSafe | — | 0 on this machine |
| Laya | `laya:en` | Ollaya | 0.8 GiB | Same as the weight file |
| Laya typed | `laya:typed-decisions` | Ollaya | 0.8 GiB | Same as the weight file |
| Kev 0.8B | `kev:0.8b` | Ollaya | 1.7 GiB | Same as the weight file |
| Kev 4B | `kev:4b` | Ollaya | 8.8 GiB | 8.8 GiB on CPU (ONNX F32, context 8192) when it was loaded |
| Winnow e4b | `winnow:e4b` | Ollaya | 7.5 GiB | Same as the weight file |
| Winnow 12B | `winnow:12b` | Ollaya | 11.8 GiB | 11.8 GiB on Metal, measured above |

Bytes behind those GiB figures:

| Tag | Bytes |
|---|---:|
| `qwen3:8b` | 5,225,388,164 |
| `laya:en` | 852,602,879 |
| `laya:typed-decisions` | 852,494,175 |
| `kev:0.8b` | 1,822,775,186 |
| `kev:4b` | 9,489,173,088 |
| `winnow:e4b` | 8,005,451,502 |
| `winnow:12b` | 12,669,660,829 |
