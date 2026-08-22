# ColBERT スタックの選定記録

パート14（ColBERT）で使うライブラリを、PyLate から
`sentence-transformers` v6 + `fast-plaid` へ移行した際の判断記録。

## 変遷

| 版 | エンコーダ | インデックス |
| --- | --- | --- |
| 元ノート（langchain-ai/rag-from-scratch） | RAGatouille | RAGatouille |
| 旧版 | PyLate `models.ColBERT` | PyLate `indexes.PLAID` |
| 現在 | `sentence-transformers` `MultiVectorEncoder` | `fast-plaid` `FastPlaid` |

## 検討した3案の比較（すべて隔離環境で実測）

| | 旧版：PyLate 1.4.0 | 案A：ST 6.0 のみ | **案B：ST 6.0 + fast-plaid（採用）** |
| --- | --- | --- | --- |
| エンコーダ | `pylate.models.ColBERT` | `MultiVectorEncoder` | `MultiVectorEncoder` |
| インデックス | `pylate.indexes.PLAID` | **なし**（総当たりのみ） | `fast_plaid.search.FastPlaid` |
| 実体のPLAID実装 | fast-plaid `<=1.3.0.290` | — | fast-plaid `1.6.0.2110` |
| sentence-transformers | **`==5.1.1` 固定** | `6.0.0` | `6.0.0` |
| transformers | `4.56.2` | `5.15.1` | `5.15.1` |
| torch | `2.9.0` | `2.13.0` | `2.11.0` |
| パッケージ数 | 218 | 201 | 205 |
| MaxSimの可視化 | 不可 | 可 | 可 |
| GPUピークメモリ | 未計測 | 599 MiB | 687.7 MiB |

## なぜ移行したか

### 1. PyLate は sentence-transformers を厳密固定していた

PyLate はどのバージョンも `sentence-transformers` を完全一致で固定する。

| pylate | 固定先 |
| --- | --- |
| 1.4.0（旧版で使用） | `sentence-transformers==5.1.1` |
| 1.6.0（当時の最新） | `sentence-transformers==5.3.0` |

`sentence-transformers` v6 は `transformers>=5.0.0` を要求するため、
**PyLate と v6 は共存できない**。パート15の `CrossEncoder` も
同じ `sentence-transformers` を使うため、PyLate を入れている限り
そちらも 5.1.1 に縛られていた。

### 2. インデックスの実装は変わっていない

`pylate` 1.4.0 のメタデータには次の記載がある。

```text
Requires-Dist: fast-plaid<=1.3.0.290,>=1.2.4.260
```

つまり旧版の `indexes.PLAID` は `fast-plaid` のラッパーだった。
案Bはエンジンを変えたのではなく、**ラッパーを外して同じエンジンを直接呼ぶ**構成であり、
機能的な後退はない。むしろ fast-plaid は `1.3.0` 上限から `1.6.0` へ上がった。

### 3. MaxSim が教材として見えるようになった

旧版は PLAID インデックスの内部で採点が完結しており、ColBERT の核心である
「トークンごとのベクトル」と「MaxSim による採点」が見えなかった。
`MultiVectorEncoder` では次が確認できる。

- `encode_query()` / `encode_document()` が返す `(トークン数, 次元)` の形状
- `similarity()` が返す MaxSim スコアそのもの
- PLAID の近似スコアと総当たりスコアの比較

## 案Aを採らなかった理由

`sentence-transformers` v6 は本格的な検索インデックスを同梱しない。
[公式ブログ](https://huggingface.co/blog/multi-vector-encoder)に明記がある。

> Past that size you want a real late-interaction index, which Sentence
> Transformers doesn't ship.

案Aでは総当たり検索のみになり、教材で見せていた PLAID インデックスの構築が
失われる。案Bならその損失がない。

## 移行時に確認したこと（実測ログ）

```text
torch: 2.11.0+cu130
cuda available: True / NVIDIA GeForce RTX 3070

query shape : (12, 128)
doc shapes  : [(22, 128), (19, 128), (18, 128), (13, 128), (18, 128)]

exhaustive MaxSim : [11.202, 10.9952, 11.2743, 10.6625, 10.764]
PLAID results     : [(2, 11.27392578125), (0, 11.201171875), (1, 10.9951171875)]

index size        : 84.1 KiB
peak GPU memory   : 687.7 MiB
```

- 旧版と同じチェックポイント `lightonai/GTE-ModernColBERT-v1` が
  `MultiVectorEncoder` へ自動変換されてそのまま動作する
- PLAID の検索結果は、この規模では総当たり MaxSim と一致する
- パート15の `CrossEncoder` も新しい torch 上で動作する

## 残る制約

- `fast-plaid` は `torch` を厳密固定する（現在 `torch==2.11.0`）。
  torch を上げたい場合は fast-plaid の更新を待つ必要がある。
- PLAID は近似検索であり、コーパスが大きくなると総当たりとスコアが乖離しうる。

## 使用モデルについて

元ノートは `colbert-ir/colbertv2.0` を使っていたが、本リポジトリは
`lightonai/GTE-ModernColBERT-v1` を使う（環境変数 `COLBERT_MODEL` で変更可能）。
ライブラリとモデルの両方が異なるため、検索結果を元ノートと直接比較することはできない。
