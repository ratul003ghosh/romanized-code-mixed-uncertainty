"""Two-channel uncertainty decomposition on a teacher-forced pivot (proposal §4.2-4.3, §4.5).

Input: log-probabilities of K same-tokenizer teachers at each pivot position, shape [K, T, V].

    H(p_bar_t) = A_t + E_t
    A_t = (1/K) sum_k H(p_k,t)               aleatoric: every teacher is unsure
    E_t = H(p_bar_t) - A_t = JSD(p_1..p_K)   epistemic: teachers confident but disagree

Uniform weights are used for the decomposition on purpose (entropy weights would
suppress A_t). Entropy-based weights (Eq. 6) are used ONLY for the distillation target.
All entropies are in nats.
"""
import torch


@torch.no_grad()
def decompose(logp, target_ids, tau=1.0, top_k=10, alt_k=3):
    K, T, V = logp.shape
    p = logp.exp()
    H_k = -(p * logp).sum(-1)                                  # [K, T]
    A = H_k.mean(0)                                            # [T]
    p_bar = p.mean(0)                                          # [T, V]
    H_tot = -(p_bar * torch.log(p_bar.clamp_min(1e-30))).sum(-1)
    E_raw = H_tot - A
    residual_min = float(E_raw.min())                          # JSD >= 0 up to float error
    E = E_raw.clamp_min(0.0)

    y = torch.as_tensor(target_ids, device=logp.device)
    idx = torch.arange(T, device=logp.device)
    nll_k = -logp[:, idx, y]                                   # [K, T] teacher NLL of y*
    p_bar_y = p_bar[idx, y]                                    # [T] ensemble prob of y*

    w = torch.softmax(-H_k / tau, dim=0)                       # Eq. 6, [K, T]
    p_w = (w.unsqueeze(-1) * p).sum(0)                         # [T, V] KD target
    top_w = p_w.topk(top_k, dim=-1)
    alt = p_bar.topk(alt_k, dim=-1)                            # readable alternatives

    return {
        "A": A, "E": E, "H_total": H_tot, "H_k": H_k, "nll_k": nll_k, "p_bar_y": p_bar_y,
        "kd_top_ids": top_w.indices, "kd_top_probs": top_w.values,
        "alt_ids": alt.indices, "alt_probs": alt.values,
        "jsd_min_before_clamp": residual_min,
    }
