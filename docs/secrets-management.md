# SOPSによるシークレット管理の運用メモ

## 基本構成

- シークレットは `.env.json` にSOPS（age）で暗号化して保存し、Gitで追跡する。
- `mise.toml` の `[env] _.file = { path = ".env.json", redact = true }` により、
  `mise exec --` 経由の実行時に自動復号されて環境変数として渡される。
  復号済みの平文ファイルはディスク上に生成されない。
- `.sops.yaml` の `creation_rules` で `.env.json` に対応するage recipientを指定する。
- 実際の秘密鍵は `~/.config/sops/age/keys.txt` に配置し、絶対にリポジトリへ
  コミットしない。

## よく使うコマンド

```shell
# 暗号化ファイルを編集（保存時に自動で再暗号化される）
mise exec -- sops .env.json

# 特定のキーだけを追加/更新（file index の順で指定する）
mise exec -- sops set .env.json '["NEW_KEY"]' '"value"'

# 特定のキーだけを削除
mise exec -- sops unset .env.json '["OLD_KEY"]'

# 暗号化されているか確認（トップレベルの "encrypted" を見る。中身は表示しない）
mise exec -- sops filestatus .env.json
```

`sops unset` / `sops set` のCLI引数は「サブコマンド → ファイルパス → インデックス」
の順であり、`--set` / `--unset` という単独フラグではない点に注意する
（`sops --unset ...` は存在しないエラーになる）。

## pre-commit hookによる誤コミット防止

`.pre-commit-config.yaml` に `sops-encrypted` フックを登録しており、
`.env.json` / `.sops.yaml` / `mise.toml` を変更した際に
`scripts/check-sops-encrypted.sh` が `sops filestatus` の出力で
`"encrypted": true` を確認する。平文化した状態でコミットしようとすると
このフックが失敗して止める。

## 秘密鍵のバックアップ（未対応・要対応）

age秘密鍵（`~/.config/sops/age/keys.txt`）を紛失すると `.env.json` は復号
不能になる。リポジトリ外のパスワードマネージャー等へバックアップすることを
強く推奨する。複数人で利用する場合は、各利用者のage recipientを
`.sops.yaml` に追加するか、KMSベースの鍵管理へ移行する。
