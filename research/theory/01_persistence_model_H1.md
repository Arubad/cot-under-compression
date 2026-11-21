# Theory for H1: how a quantisation error becomes a wrong derivation

_Written 2026-11-21, before any sweep. Extends README §4 and REVIEW §3.1. Changes are logged at the bottom._

## 0. Setup and notation

Greedy decoding produces a reference trajectory $y_{1:T}$ under the FP32 reference model. A perturbed model (KV-quantised or weight-quantised) decodes the same prompt. At step $t$, as long as the two trajectories agree so far, both models condition on the **same** context. Write $\ell_t \in \mathbb{R}^V$ for the reference logits and $\tilde\ell_t = \ell_t + \delta_t$ for the perturbed logits.

- **Margin:** $m_t = \ell_t[y_t] - \max_{v \ne y_t} \ell_t[v] \ge 0$.
- **Flip at t:** the argmax changes, which happens when the perturbation projected on the top-2 gap exceeds the margin: $\delta_t[v^*] - \delta_t[y_t] > m_t$.
- **First divergence:** $T_\text{div} = \min\{t : \text{flip at } t\}$. This is the primary H1 statistic.

The key structural fact: **before the first flip, the context is identical.** The perturbation at step $t$ is therefore a deterministic function of the intervention and the shared prefix. It does not accumulate through the generated tokens, because no generated token has changed yet. Autoregressive compounding starts only *after* $T_\text{div}$, and it is the same for every intervention.

## 1. Noise injected by a quantised cache entry

Asymmetric uniform $b$-bit quantisation over a group with range $R$ has step $\Delta = R/(2^b - 1)$. Under the usual high-resolution approximation, the reconstruction error is uniform on $[-\Delta/2, \Delta/2]$, with variance

$$\sigma_b^2 = \frac{\Delta^2}{12} = \frac{R^2}{12\,(2^b-1)^2}.$$

So $\sigma_8 : \sigma_4 : \sigma_2 = 1 : 17 : 85$, using $255/15$ and $255/3$. The ranges $R$ differ by scheme: per-token uses one scale per token over head_dim, and per-channel uses one scale per channel over a 64-token block. Channel outliers in $K$ ([KIVI](https://arxiv.org/abs/2402.02750), [KVQuant](https://arxiv.org/abs/2401.18079)) make per-token $R$ for keys much larger than per-channel $R$. That is why per-channel keys are expected to be better, and it is a measurable quantity: σ² can be estimated directly from cached K/V in I1, without any decoding.

**Value noise.** With $o_t = \sum_i a_{t,i}(v_i + \epsilon_i)$ and $\epsilon_i$ i.i.d. with per-coordinate variance $\sigma_v^2$:

$$\operatorname{Var}[\delta o_t] = \sigma_v^2 \sum_i a_{t,i}^2 = \sigma_v^2\, e^{-H_2(a_t)},$$

where $H_2$ is the Rényi-2 entropy of the attention row. A spread-out head averages the noise away, and a concentrated head passes it through at full strength.

**Key noise.** $\delta s_{t,i} = q_t^\top \eta_i / \sqrt d$ perturbs the scores. Through the softmax Jacobian, $\delta a_{t,i} = a_{t,i}(\delta s_{t,i} - \bar{\delta s}_t)$, so

$$\delta o_t = \sum_i a_{t,i}\,(\delta s_{t,i} - \bar{\delta s}_t)(v_i - o_t) = \operatorname{Cov}_{a_t}(\delta s, v).$$

Key noise therefore hurts most where attention sits on keys whose values differ from the current output. That describes a retrieval head reading a specific number from the question. It hurts little in a sink or streaming head whose attended values are similar to each other.

## 2. From per-step noise to a divergence hazard

The per-step perturbation $\delta o_t$ propagates through the remaining layers to $\delta_t$. To first order $\delta_t = J_t\,\delta o_t$, summed over layers and heads, and its projection on the top-2 direction has some scale $\tau_t$, roughly linear in σ. The per-step flip probability (the hazard) is

$$h_t = P(\text{flip at } t \mid \text{no flip before}) = P\big(|\xi_t| > m_t\big), \quad \xi_t \sim \text{scale } \tau_t.$$

The margins $m_t$ vary across steps. Let $f_m$ be their density. Two regimes follow:

- **Small noise (expected at INT8, maybe INT4):** if $f_m(0) > 0$, meaning some tokens are near-ties, then $h \approx 2 f_m(0)\,\mathbb{E}|\xi| \propto \tau \propto \sigma$. The hazard is **linear in σ**, and $\mathbb{E}[T_\text{div}] \approx 1/h \propto 1/\sigma$.
- **Large noise (INT2):** $h \to P(|\xi| > m)$ saturates, and $T_\text{div}$ collapses to the first few tokens.

> **Prediction T1.1.** In the linear regime, $\mathbb{E}[T_\text{div}]$ for INT8 : INT4 : INT2 scales as $1 : 1/17 : 1/85$, up to the ratio of ranges $R$ across bit-widths, which is the same quantiser so roughly 1. Each step is clipped from below by 1 and from above by censoring (§5).
>
> **Note against the pre-registration.** The pre-registered secondary analysis fits $\mathbb{E}[T_\text{div}]$ against $1/\sigma^2$. This derivation gives $1/\sigma$ whenever near-tie tokens exist. $1/\sigma^2$ would follow only if the first-order term vanished, for example if all margins were bounded away from zero and the flip came from a second-order effect. **Proposal:** fit $\log \mathbb{E}[T_\text{div}] = \alpha - \beta \log \sigma$ and report β. β ≈ 1 supports the near-tie mechanism and β ≈ 2 supports a second-order one. This is secondary and not pass/fail. It needs an amendment before I3 runs (see `open_questions.md` Q3).

## 3. What makes KV different from weights in this model

Weight quantisation also produces some $\tau^W$ at every step and therefore also a hazard. The H1 contrast is "KV first divergence moves earlier with fewer bits, weight first divergence is flat". In this model that contrast is a claim about **magnitudes**, not about different laws. INT4 weights with group-wise scales and NF4 are known to be near-lossless ([Dettmers & Zettlemoyer](https://arxiv.org/abs/2212.09720)), so their $\tau^W$ is small and the INT8→INT4 step barely moves $h$. Per-token INT4 KV has a large $R$ (key outliers), so its τ is large. The theory predicts that the contrast holds when $\tau^W_4 \ll \tau^{KV}_4$, and $\sigma^{KV}$ can be measured cheaply to check this.

## 4. The length prediction needs more than persistence

Given the reference trajectory, the probability of surviving to the end without a flip is

$$S(T) = \prod_{t \le T}(1 - h_t) \approx \exp\Big(-\sum_{t\le T} h_t\Big).$$

If $h_t = h$ is constant, $S(T) = e^{-hT}$. Longer chains then diverge more often under **both** interventions, simply from more exposure. The tertile test (KV long-tertile gap ≥ 2× short-tertile gap) can therefore pass even with a constant hazard. What distinguishes "persistent, position-specific error" from "the same error every step" is the **shape** of $h_t$ in $t$, and under greedy decoding persistence does *not* by itself make $h_t$ grow:

- A persistent corrupted anchor is read at every step, but no flip means no change of context. A correlated error makes the steps' flip events correlated without raising the marginal hazard.
- Attention dilution goes the *other* way: as the context grows, $\sum_i a^2$ falls, so value noise per step falls (§1).
- $h_t$ grows with $t$ only if the attention mass on heavily corrupted positions grows with $t$. Examples: retrieval heads that re-read the question's numbers late in the derivation, or the fp16 residual tail of the per-channel scheme sliding over, so that more of the attended context becomes quantised.

> **Prediction T1.2.** For per-token KV, where every cached position is quantised from prefill on, $h_t$ is flat or slightly decreasing in $t$. For per-channel KV, $h_t$ steps up once generation passes the residual-tail length. For weights, $h_t$ is flat. If the KV length gap grows faster than $1 - e^{-hT}$ predicts, the persistence story needs an anchor-reading mechanism, which I5's head statistics can identify.
>
> **Proposed secondary analysis (needs amendment, Q3):** estimate $h_t$ per configuration with a Nelson–Aalen estimator on first-divergence times. Test whether $h_t$ is flat with a log-rank trend test, and compare hazard shapes between KV and weights. This statistic separates the two mechanisms. The tertile contrast cannot, because exposure alone produces it.

## 5. Censoring

A problem that does not diverge before the generation ends (EOS or 512 tokens) is **right-censored**: we only know $T_\text{div} > T_\text{end}$. The harness currently records `first_divergence_or_len`, which counts censored problems as diverging at $T_\text{end}$ and biases the median downwards. With INT8 KV most problems will probably be censored, so the bias is not small.

> **Prediction T1.3.** At INT8 KV, more than 50 % of problems do not diverge within 512 tokens. The "median first-divergence token" is then undefined in the uncensored sense, and a Kaplan–Meier median (or "> 512") must be reported. This affects the *primary* H1 statistic. The monotonicity test should be run on Kaplan–Meier medians with bootstrap CIs (`open_questions.md` Q3).

## 6. Divergence is not error

A flip can be benign, for example "so" versus "therefore", or an equivalent reordering of arithmetic. Accuracy depends on

$$P(\text{wrong}) = P(\text{diverge}) \cdot P(\text{wrong} \mid \text{diverge}).$$

A secondary quantity worth reporting is $P(\text{wrong} \mid \text{diverge at } t)$ as a function of $t$. Early flips, in the problem restatement or the first equation, should be far more damaging than late ones, which fall in the final sentence after the answer.

## Changelog
- 2026-11-21 — first version. Derived the linear-hazard regime (1/σ, not 1/σ²), showed that persistence alone does not produce length dependence under greedy decoding, and flagged censoring of the primary statistic.
