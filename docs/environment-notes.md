# 実行環境（GPU / herdr）に関するメモ

## herdr上でもGPUは利用できる

このセッションで使用している `herdr`（AIコーディングエージェント向けターミナル
マルチプレクサ、<https://herdr.dev/）は、tmux相当のセッション管理ツールであり、>
セキュリティサンドボックスではない。GPUアクセスをブロックする仕組みは持たない。

## WSL2環境でGPUが見つからないように見える場合の確認方法

WSL2 + NVIDIA GPUパススルー環境では、以下の理由でGPUの存在を見落としやすい。

1. `nvidia-smi` が標準の `PATH` に入っていないことがある。
   実体は `/usr/lib/wsl/lib/nvidia-smi` にあるため、`command -v nvidia-smi` が
   失敗しても、フルパスで直接実行すれば動作することがある。
2. `lspci` はWSL2のGPUパススルーを正しく列挙しない。
   仮想ディスプレイ用の `Basic Render Driver` (3D controller) だけが見え、
   実GPUは出てこない。`lspci` の結果だけでGPU不在と判断しないこと。

確実な確認方法は、対象のPython仮想環境から直接 `torch.cuda.is_available()` を
呼び出すこと。

```shell
mise exec -- uv run python -c "
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no GPU')
"
```

GPUデバイスノード `/dev/dxg` の存在も、WSL2でGPUパススルーが有効かどうかの
簡易な確認手段になる。

## GPUメモリ使用量の実測方法

理論値やモデルカードの記載だけで「メモリに収まる」と判断せず、実際に対象環境で
モデルをロード・推論して `torch.cuda.memory_allocated()` /
`torch.cuda.max_memory_allocated()` を計測することを優先する。

```python
import torch
print(f"before: {torch.cuda.memory_allocated(0) / 1024**2:.1f} MiB")
# モデルロード・推論
print(f"peak:   {torch.cuda.max_memory_allocated(0) / 1024**2:.1f} MiB")
```

`cross-encoder/ms-marco-MiniLM-L-6-v2` を候補文書10件で実行した実測例:
モデルロード後 約87 MiB、推論ピーク 約102 MiB（RTX 3070, VRAM 8,192 MiBの
環境で、他プロセス使用中の約2,400 MiBを差し引いても十分な余裕がある）。
