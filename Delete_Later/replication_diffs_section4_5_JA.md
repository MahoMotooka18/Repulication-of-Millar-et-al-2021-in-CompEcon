# レプリケーション差分レポート（更新版）

このノートは、更新されたレプリケーション指示ファイルに対して現在の実装を照合し、相違点と修正案を整理したものです。
- `Delete_Later/new_section4_math.md`
- `Delete_Later/new_section5_math.md`

---

## セクション4（Consumption–Saving）

1) **Euler/Bellman 残差で `w'` の更新に `y_t` を使っている**
- **該当コード:**
  - `Lab_Section4_ConsumptionSaving/model_consumption_saving.py:79-98`（`exp(y_t)` を使用）
  - `Lab_Section4_ConsumptionSaving/objectives.py:110-118`（Euler で `state_transition(..., y_batch)`）
  - `Lab_Section4_ConsumptionSaving/objectives.py:200-207`（Bellman で `state_transition(..., y_batch)`）
  - `Lab_Section4_ConsumptionSaving/evaluator.py:120-128`（評価でも `y_t`）
- **指示との不一致:** `new_section4_math.md` の 4.6 に「Bellman/Euler 残差評価では `w' = r(w-c)+e^{y'}` を使う」とある。
- **修正案:**
  - 残差評価用に `state_transition_next_y(w_t, c_t, y_next)` を追加。
  - Euler/Bellman objective と evaluator で `y_next` を先に計算し、それを使って `w_next` を更新。
  - シミュレーション用の遷移は必要なら現行のまま維持。

2) **Bellman objective の FB 項と乗数項が指示と異なる**
- **該当コード:**
  - `Lab_Section4_ConsumptionSaving/objectives.py:224-233`（FB を `λ` で評価し `fb_1 * fb_2`）
  - `Lab_Section4_ConsumptionSaving/objectives.py:235-239`（`a * λ` の積）
- **指示との不一致:** `new_section4_math.md`（Eq. 32）は
  - FB 項を `PsiFB(1-c/w, 1-h)^2`（ショック別の積ではない）、
  - 乗数項を `[(βr dV/dw'/u'(c) - h)_1 * (βr dV/dw'/u'(c) - h)_2]` としている。
- **修正案:**
  - `h = policy.forward_h(...)` を使い、FB は `PsiFB(1-c/w, 1-h)` を **二乗**。
  - `a * λ` をやめて `(βr dV/dw'/u'(c) - h)` の AiO 積に置き換え。
  - Bellman 残差の AiO 積は現行のまま維持。

---

## セクション5（Krusell–Smith）

1) **価格・集計で「平均労働」と `z_t` を使う指示だが、実装は「合計労働」と `exp(z_t)`**
- **該当コード:**
  - `Lab_Section5_Krusell_and_Smith_1998/model_ks1998.py:160-215`
- **指示との不一致:** `new_section5_math.md`（Eq. 42）は
  - `R_t, W_t` を `z_t` で計算し、労働は `(1/ℓ)∑ exp(y_i)`（平均）を使う。
- **修正案:**
  - 指示に厳密に合わせるなら、労働は `mean(exp(y))` を用い、価格計算で `z_t` をそのまま使う。
  - もし `z_t` をログ TFP と解釈するなら、ノート側を `exp(z_t)` に合わせるなど、どちらかに統一して明記。

2) **政策関数のパラメータ化が指示と一致しない（切片固定＋定常シフト）**
- **該当コード:**
  - `Lab_Section5_Krusell_and_Smith_1998/nn_policy_ks.py:62-75`（`phi_intercept` が学習不可、`phi_logit_shift` 追加）
- **指示との不一致:** `new_section5_math.md` は `zeta_0 + eta(...)` を共有し、`zeta_0` は 0 初期化（固定ではない）で、定常シフトの記述はない。
- **修正案:**
  - `phi_intercept` を学習可能にする、または 3 ヘッドで同一切片を共有。
  - `phi_logit_shift` をオプション化（デフォルト無効）。

3) **Euler objective で `1 - h` を非負にクリップしている**
- **該当コード:**
  - `Lab_Section5_Krusell_and_Smith_1998/objectives_ks.py:160-165`
- **指示との不一致:** `new_section5_math.md`（Eq. 44）は `PsiFB(1-c/w, 1-h)` をそのまま使う。クリップすると `h>1` へのペナルティが消える。
- **修正案:**
  - `1-h` の clamp を削除（`w` の数値安定化ガードは残してよい）。

4) **Bellman objective の FB 項が `h` ではなく `λ` を使っている**
- **該当コード:**
  - `Lab_Section5_Krusell_and_Smith_1998/objectives_ks.py:287-296`
- **指示との不一致:** `new_section5_math.md`（Eq. 45）は FB 項を `PsiFB(1-c/w, 1-h)` とし、FOC の整合性は別の `G` 項で扱う。
- **修正案:**
  - FB 項は `1-h` を使う。
  - 乗数整合性項 `(β R dV/dw'/u'(c) - h)` の AiO 積は維持。

5) **生産性の正規化とショックの平均シフトが追加仕様**
- **該当コード:**
  - `Lab_Section5_Krusell_and_Smith_1998/model_ks1998.py:274-291`（正規化）
  - `Lab_Section5_Krusell_and_Smith_1998/model_ks1998.py:153-158`（平均シフト）
  - `Lab_Section5_Krusell_and_Smith_1998/main_section5.py:392-470`（トレーニングで正規化使用）
- **指示との不一致:** 更新後のノートではそのような補正が記載されていない。
- **修正案:**
  - 厳密再現なら `use_log_shock_shift=false`、`enforce_bounds=false` とし、`normalize_productivity` 呼び出しを外す。
  - 維持するなら「学習安定化のための差分」として明記。

6) **入力スケーリングが指示にない**
- **該当コード:**
  - `Lab_Section5_Krusell_and_Smith_1998/policy_utils_ks.py`
  - `Lab_Section5_Krusell_and_Smith_1998/main_section5.py:176-238`
- **指示との不一致:** `new_section5_math.md` は生の状態入力を想定。
- **修正案:**
  - 厳密再現なら `input_scaling.enabled: false`。
  - 維持するなら「学習安定化のための差分」として明記。

