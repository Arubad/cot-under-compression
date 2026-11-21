# Theory for H2: when is dropping keys from one head harmless?

_Written 2026-11-21, before any sweep._

## 1. Exact error of keeping a subset of keys

Take one head at one decode step. Its attention weights are $a_i$ over $n$ cached positions, its values are $v_i$, and its output is $o = \sum_i a_i v_i$. Sparsification keeps a set $K$ and renormalises. Let $A_K = \sum_{i \in K} a_i$. The kept output is $o_K = \frac{1}{A_K}\sum_{i\in K} a_i v_i$, and the dropped part is $\bar o_{\bar K} = \frac{1}{1-A_K}\sum_{i \notin K} a_i v_i$. Then $o = A_K\,o_K + (1-A_K)\,\bar o_{\bar K}$, so

$$\boxed{\,o - o_K = (1 - A_K)\,(\bar o_{\bar K} - o_K)\,} \qquad \|o - o_K\| \le (1-A_K)\cdot \operatorname{diam}\{v_i\} \le 2(1-A_K)\max_i\|v_i\|.$$

Two factors control the error:
1. **Dropped mass** $\varepsilon = 1 - A_K$. This is the heavy-attention quantity in LevAttention and H2O.
2. **Value contrast** $\|\bar o_{\bar K} - o_K\|$. Dropping a lot of mass costs nothing if the dropped values look like the kept ones. This is the case for "streaming" heads over repetitive text.

## 2. Entropy versus top-k mass as the predictor

For top-k heavy-hitter selection, $\varepsilon$ is the smallest dropped mass any $k$-set can achieve: $\varepsilon_{\text{top-}k} = 1 - \text{mass}_k$. So **top-k mass is the quantity the bound actually contains**. Entropy enters only through an inequality. A grouping argument, in the style of Fano, gives

$$H(a) \le h(\varepsilon) + (1-\varepsilon)\log k + \varepsilon \log(n-k),$$

which forces $\varepsilon$ to be large when $H$ is large: high entropy means sparsification cannot be safe. Low entropy does *not* force small $\varepsilon$ at a given $k$. A head can spread mass evenly over 2k keys with low entropy and still lose half its mass at budget k.

> **Prediction T2.1.** $|\rho(\text{mass}_k, \Delta)| \ge |\rho(H, \Delta)|$. The pre-registration sets the entropy threshold (ρ ≥ 0.6) above the mass threshold (≥ 0.5). This theory would rank them the other way. The pre-registered thresholds stand, but if the result is "entropy fails, mass passes", it is the *expected* outcome and not a surprise.

## 3. Leverage versus heavy-hitter: query dependence

The leverage sparsifier keeps the keys with the largest $\|k_i\|^2$, the same set for every query. The heavy-hitter sparsifier keeps each query's own top-k. For a given head, the gap

$$\varepsilon_\text{lev} - \varepsilon_\text{top-}k \ge 0$$

measures how **query-dependent** the head's attention is. It is near 0 for sink and positional heads and large for retrieval heads. This is a cheap, interpretable per-head statistic. It should be recorded in I1 from the cached attention, at no extra decode cost, and it is the natural explanation for wherever leverage is much worse than top-k.

## 4. Single-head effects are small, so the outcome variable matters

Removing one head out of 336 (28 × 12) adds a perturbation $W_O^{(h)}(o - o_K)$ to the residual stream. That perturbation feeds into the same hazard model as H1 (theory/01 §2), with $\tau \propto \|W_O^{(h)}\|\,\varepsilon\,\text{contrast}$. For most heads the accuracy change on 300 problems will be a handful of flipped answers. A paired difference on 300 binary outcomes, with a discordance rate $d$, has standard error $\approx \sqrt{d/300}$. Even $d$ = 5 % gives ~1.3 pt, comparable to the effect itself.

> **Prediction T2.2.** Raw Δacc per head will be dominated by noise for most heads, which attenuates Spearman ρ toward 0. That can cause a false "refuted" (|ρ| < 0.3) on accuracy while the mechanism holds. Continuous outcomes, namely mean truncated KL to the reference before divergence and the per-head hazard, have far less noise per problem.
>
> **Proposal (needs amendment before I5, Q4):** report ρ against (i) Δacc, which stays the pre-registered pass/fail test, (ii) mean KL and (iii) the Kaplan–Meier median $T_\text{div}$ as secondary outcomes. Also report a partial correlation that controls for $\|W_O^{(h)}\|_F$, because head importance confounds head concentration.

## 5. Choosing the 6-layer subset

The pre-registration requires a subset that spans low and high entropy. To avoid choosing layers on outcome-adjacent information, the rule should be fixed before I5 runs: take per-layer mean entropy from I1, sort, and pick layers at the 0, 20, 40, 60, 80 and 100th percentiles, ties broken toward the lower index. Write this rule into the I5 page before launch.

## Changelog
- 2026-11-21 — first version. Derived the exact sparsification error, argued that top-k mass, not entropy, is the primary predictor, flagged the single-head noise floor, and fixed the layer-selection rule.
