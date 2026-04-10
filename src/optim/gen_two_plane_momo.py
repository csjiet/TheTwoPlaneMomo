from typing import Dict, Iterable, Optional, Union

import torch
from torch.optim import Optimizer


class GenTwoPlaneMoMo(Optimizer):
    """
    Two-Plane MoMo
    Works with: loss.backward(); opt.step(loss)

    If you set use_loss_ema=False, you may call opt.step() without passing loss,
    and the planes use b1 = <m1, w_t> - gamma1, b2 = <m2, w_t> - gamma2 (no loss EMA term).
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        beta_short: float = 0.9,  # short-horizon EMA decay for the gradient/ fast timescale
        beta_long: float = 0.999,  # long-horizon EMA decay for the gradient /slow timescale
        eps: float = 1e-12,
        clip_alpha: bool = True,
        use_loss_ema: bool = True,
        preconditioner: str = "identity",  # string options: {"identity", "adam"}
        precond_beta2: float = 0.999,  # Adam second-moment beta2
        weight_decay_factor: float = 0.0,
        eps_precond: float = 1e-12,
        decoupled_weight_decay: bool = False,
        alpha_scope: str = "network",
    ):
        if lr <= 0:
            raise ValueError("lr must be > 0")
        if not (0.0 <= beta_short < 1.0):
            raise ValueError("beta_short in [0,1)")
        if not (0.0 <= beta_long < 1.0):
            raise ValueError("beta_long in [0,1)")
        if eps <= 0:
            raise ValueError("eps must be > 0")
        if preconditioner not in ("identity", "adam"):
            raise ValueError("preconditioner must be either {'identity', 'adam'}")
        if not (0.0 <= precond_beta2 < 1.0):
            raise ValueError("precond_beta2 must be in [0,1)")
        if weight_decay_factor < 0:
            raise ValueError("weight_decay_factor must be non-negative")
        if eps_precond <= 0:
            raise ValueError("eps_precond must be > 0")
        if not isinstance(decoupled_weight_decay, bool):
            raise ValueError("decoupled_weight_decay must be a bool")
        if alpha_scope not in ("network", "parameter"):
            raise ValueError("alpha_scope must be one of {'network', 'parameter'}")

        defaults = dict(
            lr=lr,
            beta_short=beta_short,  # short-horizon EMA decay for the gradient
            beta_long=beta_long,  # long-horizon EMA decay for the gradient /slower timescale
            eps=eps,
            eps_precond=eps_precond,
            preconditioner=preconditioner,
            precond_beta2=precond_beta2,  # adam style second-moment EMA decay
            weight_decay_factor=weight_decay_factor,
            decoupled_weight_decay=decoupled_weight_decay,
            alpha_scope=alpha_scope,
            tp_clip_alpha=clip_alpha,
            tp_use_loss_ema=use_loss_ema,
        )

        super().__init__(params, defaults)

        for g in self.param_groups:
            g.setdefault("tp_barf1", 1.0)
            g.setdefault("tp_barf2", 1.0)
            g.setdefault("tp_gamma1", 1.0)
            g.setdefault("tp_gamma2", 1.0)
            g.setdefault("tp_step", 0)
            g.setdefault("tp_clip_alpha", clip_alpha)
            g.setdefault("tp_use_loss_ema", use_loss_ema)
            g.setdefault("preconditioner", preconditioner)
            g.setdefault("precond_beta2", precond_beta2)
            g.setdefault("weight_decay_factor", weight_decay_factor)
            g.setdefault("eps_precond", eps_precond)
            g.setdefault("decoupled_weight_decay", decoupled_weight_decay)
            g.setdefault("alpha_scope", alpha_scope)

        self.last_alpha1: Optional[float] = None

    #### Network-wide version ####
    @torch.no_grad()
    def _step_network(self, loss: Optional[Union[float, torch.Tensor]], log_dict: Optional[Dict] = None):
        """
        Usage:
          loss.backward()
          opt.step(loss)   # if use_loss_ema=True (default)

        If use_loss_ema=False, you may call:
          loss.backward()
          opt.step()       # loss is ignored
        """
        # fetch hparams
        lr = self.param_groups[0]["lr"]
        mu = self.param_groups[0]["weight_decay_factor"]
        beta_s = self.param_groups[0]["beta_short"]
        beta_l = self.param_groups[0]["beta_long"]
        eps = self.param_groups[0]["eps"]
        eps_precond = self.param_groups[0]["eps_precond"]
        precond = self.param_groups[0]["preconditioner"]
        precond_beta2 = self.param_groups[0]["precond_beta2"]
        decoupled_wd = self.param_groups[0]["decoupled_weight_decay"]
        g = self.param_groups[0]
        barf1 = g["tp_barf1"]
        barf2 = g["tp_barf2"]
        gamma1 = g["tp_gamma1"]
        gamma2 = g["tp_gamma2"]
        clip_alpha = g["tp_clip_alpha"]
        use_loss_ema = g["tp_use_loss_ema"]
        tp_step = g["tp_step"]

        # ensure all parameter groups must share the same optimizer-wide hyperparameters and scalar-state semantics
        # since this implementation computes one network-wide alpha using param_groups[0] as the global owner.
        for group in self.param_groups[1:]:
            if group["lr"] != lr:
                raise ValueError("All parameter groups must share the same lr for network-wide GenTwoPlaneMoMo.")
            if group["beta_short"] != beta_s:
                raise ValueError("All parameter groups must share the same beta_short for network-wide GenTwoPlaneMoMo.")
            if group["beta_long"] != beta_l:
                raise ValueError("All parameter groups must share the same beta_long for network-wide GenTwoPlaneMoMo.")
            if group["eps"] != eps:
                raise ValueError("All parameter groups must share the same eps for network-wide GenTwoPlaneMoMo.")
            if group["eps_precond"] != eps_precond:
                raise ValueError("All parameter groups must share the same eps_precond for network-wide GenTwoPlaneMoMo.")
            if group["preconditioner"] != precond:
                raise ValueError("All parameter groups must share the same preconditioner for network-wide GenTwoPlaneMoMo.")
            if group["precond_beta2"] != precond_beta2:
                raise ValueError("All parameter groups must share the same precond_beta2 for network-wide GenTwoPlaneMoMo.")
            if group["weight_decay_factor"] != mu:
                raise ValueError("All parameter groups must share the same weight_decay_factor for network-wide GenTwoPlaneMoMo.")
            if group["decoupled_weight_decay"] != decoupled_wd:
                raise ValueError("All parameter groups must share the same decoupled_weight_decay for network-wide GenTwoPlaneMoMo.")
            if group["tp_clip_alpha"] != clip_alpha:
                raise ValueError("All parameter groups must share the same tp_clip_alpha for network-wide GenTwoPlaneMoMo.")
            if group["tp_use_loss_ema"] != use_loss_ema:
                raise ValueError("All parameter groups must share the same tp_use_loss_ema for network-wide GenTwoPlaneMoMo.")
            if group["alpha_scope"] != "network":
                raise ValueError("All parameter groups must use alpha_scope='network' for network-wide GenTwoPlaneMoMo.")
            if group["tp_barf1"] != barf1:
                raise ValueError("All parameter groups must share the same tp_barf1 for network-wide GenTwoPlaneMoMo.")
            if group["tp_barf2"] != barf2:
                raise ValueError("All parameter groups must share the same tp_barf2 for network-wide GenTwoPlaneMoMo.")
            if group["tp_gamma1"] != gamma1:
                raise ValueError("All parameter groups must share the same tp_gamma1 for network-wide GenTwoPlaneMoMo.")
            if group["tp_gamma2"] != gamma2:
                raise ValueError("All parameter groups must share the same tp_gamma2 for network-wide GenTwoPlaneMoMo.")
            if group["tp_step"] != tp_step:
                raise ValueError("All parameter groups must share the same tp_step for network-wide GenTwoPlaneMoMo.")

        lr_safe = max(lr, eps)
        mu_model = 0.0 if decoupled_wd else mu  # turn off the proximal coupled subproblem when using decoupled weight decay

        # For: lambda_1,unc (Factor in the first term in the numerator)
        num_fac = (1 + (lr_safe * mu_model)) / lr_safe

        # For: \hat{v}_t --- the bias corrected second moment estimate --- used in the Adam preconditioner.
        bias_correction_denom = 1.0 - (precond_beta2 ** (tp_step + 1))
        bias_correction_denom = max(bias_correction_denom, eps)

        if use_loss_ema:
            if loss is None:
                raise RuntimeError(
                    "TwoPlaneMoMo.step(loss=...) requires the current loss (or set use_loss_ema=False in the constructor)."
                )
            loss_t = float(loss.detach().item() if isinstance(loss, torch.Tensor) else loss)
        else:
            loss_t = None

        # accumulators (global)
        logged_mom_vec_1_squared_norm = 0.0
        logged_mom_vec_2_squared_norm = 0.0
        logged_mom_vec1_vec2_dot_prod = 0.0

        # For: lambda_1,unc
        m1_dot_w_t = 0.0
        m2_dot_w_t = 0.0
        g_t_dot_w_t = 0.0

        # For: "lambda_1,unc" with preconditioner (P_t^{-1})
        denom_m1_m2_precond_inv_m1_m2 = 0.0 # (m1-m2)^T Pinv (m1-m2)
        numer_m1_m2_precond_inv_m2 = 0.0 # (m1-m2)^T Pinv m2

        # 1) update per-parameter EMAs m1/m2 and accumulate inner products
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("TwoPlaneMoMo does not support sparse gradients.")
                state = self.state[p]

                grad = p.grad.detach()
                if grad.dtype != torch.float32:
                    grad = grad.float()

                # Check if per-parameter state doesn't exist
                if "m1" not in state:
                    state["m1"] = torch.zeros_like(p, dtype=torch.float32, memory_format=torch.preserve_format)
                if "m2" not in state:
                    state["m2"] = torch.zeros_like(p, dtype=torch.float32, memory_format=torch.preserve_format)
                if precond == "adam" and "v_t" not in state:
                    state["v_t"] = torch.zeros_like(p, dtype=torch.float32, memory_format=torch.preserve_format)

                # For: lambda_unc, and w_{t+1}
                m1 = state["m1"]  # m_t^{(1)}
                m2 = state["m2"]  # m_t^{(2)}

                # two EMAs of the gradient
                m1.mul_(beta_s).add_(grad, alpha=1 - beta_s)  # m_t^{(1)} = \Beta_short m_{t-1}^{(1)} + (1 - \Beta_short) g_t
                m2.mul_(beta_l).add_(grad, alpha=1 - beta_l)  # m_t^{(2)} = \Beta_long m_{t-1}^{(2)} + (1 - \Beta_long) g_t

                # For: lambda_unc, and w_{t+1}
                # Choose preconditioner
                if precond == "adam":
                    # v_t = beta2 * v_{t-1} + (1 - beta2) * (grad \odot grad)
                    v_t = state["v_t"]
                    v_t.mul_(precond_beta2).addcmul_(grad, grad, value=(1 - precond_beta2))

                    # P_t := diag( sqrt(\hat{v_t}) + eps )
                    # => inverse diag entries (P_t^{-1})_{i,i} = 1 / (sqrt(v_t)_i + eps)
                    # store only the diagonal entries in a tensor, not a full diagonal matrix.
                    # precond_t_inv_flattened = (P_t^{-1}_{1,1}, ..., P_t^{-1}_{n,n})
                    v_t_hat = v_t / bias_correction_denom # \hat{v_t}
                    precond_t_inv_flattened = v_t_hat.sqrt().add_(eps_precond).reciprocal()
                else:
                    precond_t_inv_flattened = None

                # global inner products (only for wandb logging analysis)
                logged_mom_vec_1_squared_norm += torch.dot(m1.flatten(), m1.flatten()).item()
                logged_mom_vec_2_squared_norm += torch.dot(m2.flatten(), m2.flatten()).item()
                logged_mom_vec1_vec2_dot_prod += torch.dot(m1.flatten(), m2.flatten()).item()

                # For: lambda_1,unc
                w_t = p.detach().float()
                m1_dot_w_t += torch.dot(m1.flatten(), w_t.flatten()).item() # <m_t^{(1)}, w_t> (second term in the numerator) 
                m2_dot_w_t += torch.dot(m2.flatten(), w_t.flatten()).item() # <m_t^{(2)}, w_t> (second term in the numerator)

                # For: lambda_1,unc --- MoMo: by first building \gamma_{t}^{(i)} -> build b_t^{(1)} and b_t^{(2)} -> build lambda_1_unc.
                g_t_dot_w_t += torch.dot(grad.flatten(), w_t.flatten()).item()

                # preconditioned quadratic forms for lambda_1,unc
                if precond == "adam":
                    # For: lambda_1,unc  (numerator and denominator)
                    m1_minus_m2 = (m1 - m2) # m_t^{(1)} - m_t^{(2)} 
                    # For: lambda_1,unc (denominator)
                    # (m1_minus_m2)^T P_t^{-1} (m1_minus_m2) --- quadratic form!
                    # Since P_t^{-1} is diagonal we can compute this equivalently as the sum over coordinates:
                    #        = \sum_i (P_t^{-1})_{i,i} * ((m1_minus_m2)_i)^2
                    denom_m1_m2_precond_inv_m1_m2 += torch.sum(m1_minus_m2 * precond_t_inv_flattened * m1_minus_m2).item()
                    # For: lambda_1,unc (3rd term in the numerator)
                    # (m1_minus_m2^T) P_t^{-1} (m_t^{(2)})
                    # with diagonal P_t^{-1}, this equals:
                    #   numer_m1_m2_precond_inv_m2 = \sum_i (P_t^{-1})_{i,i} * (m1_minus_m2)_i * (m_t^{(2)})_i
                    numer_m1_m2_precond_inv_m2 += torch.sum(m1_minus_m2 * precond_t_inv_flattened * m2).item()
                else:
                    m1_minus_m2 = (m1 - m2)
                    denom_m1_m2_precond_inv_m1_m2 += torch.dot(m1_minus_m2.flatten(), m1_minus_m2.flatten()).item()
                    numer_m1_m2_precond_inv_m2 += torch.dot(m1_minus_m2.flatten(), m2.flatten()).item()

        # For: lambda_1,unc --- MoMo: by first building b_t^{(1)} and b_t^{(2)} -> build lambda_1_unc.
        if use_loss_ema:
            # \bar{l}_{t}^{(1)} = \Beta_1 (\bar{l}_{t}^{(1)}) + (1 - \Beta_1) l_t
            barf1 = beta_s * barf1 + (1 - beta_s) * loss_t
            # \bar{\ell}_{t}^{(2)} = \Beta_2 (\bar{l}_{t}^{(2)}) + (1 - \Beta_2) l_t
            barf2 = beta_l * barf2 + (1 - beta_l) * loss_t

        # \gamma_{t}^{(1)} (fast EMA of <g_t, w_t>)
        gamma1 = beta_s * gamma1 + (1 - beta_s) * g_t_dot_w_t
        # \gamma_{t}^{(2)} (slow EMA of <g_t, w_t>)
        gamma2 = beta_l * gamma2 + (1 - beta_l) * g_t_dot_w_t

        # For: lambda_1,unc --- MoMo
        if use_loss_ema:
            b1 = barf1 - gamma1 + m1_dot_w_t
            b2 = barf2 - gamma2 + m2_dot_w_t
        else:
            b1 = m1_dot_w_t - gamma1
            b2 = m2_dot_w_t - gamma2

        # For: lambda_1,unc (2nd term in the numerator: (m1 - m2)^T w_t)
        m1_minus_m2_dot_wt = (m1_dot_w_t - m2_dot_w_t)

        # Constrained lambda_{1,unc} -> lambda_{1} CLIPPED!
        alpha_max = 0.9
        alpha_min = 0.1

        # Unconstrained lambda_{1,unc}
        # - Denominator:
        #   final_denom = (m_t^{(1)} - m_t^{(2)})^\top  P_t^{-1}  (m_t^{(1)} - m_t^{(2)}).
        # - Numerator (coupled/proximal-\mu version):
        #   final_numer = ((1 + \eta*\mu)/\eta) * (b_t^{(1)} - b_t^{(2)}) - \mu * (m_t^{(1)} - m_t^{(2)})^\top w_t - (m_t^{(1)} - m_t^{(2)})^\top P_t^{-1} m_t^{(2)}.
        # - Final form:
        #   lambda_{1,unc} = final_numer / final_denom.
        final_denom = denom_m1_m2_precond_inv_m1_m2
        alpha1_unc = 0.0
        if final_denom <= eps:
            alpha1 = alpha_max if b1 >= b2 else alpha_min
            alpha1_unc = 1.0 if b1 >= b2 else 0.0
        else:
            alpha1_unc = (num_fac * (b1 - b2) - mu_model * m1_minus_m2_dot_wt - numer_m1_m2_precond_inv_m2) / max(final_denom, eps)
            alpha1 = alpha1_unc
            if clip_alpha:
                alpha1 = min(alpha_max, max(alpha_min, alpha1))
        alpha2 = 1.0 - alpha1
        self.last_alpha1 = alpha1

        # wandb logging
        if log_dict is not None:
            log_dict["two_plane_momo/network/alpha1"] = float(alpha1)
            log_dict["two_plane_momo/network/alpha2"] = float(alpha2)
            log_dict["two_plane_momo/network/final_denom"] = float(final_denom)
            log_dict["two_plane_momo/network/alpha1_unclipped"] = float(alpha1_unc)
            log_dict["two_plane_momo/network/||m_t^{(1)}||_{2}^{2}"] = float(logged_mom_vec_1_squared_norm)
            log_dict["two_plane_momo/network/||m_t^{(2)}||_{2}^{2}"] = float(logged_mom_vec_2_squared_norm)
            log_dict["two_plane_momo/network/<m_t^(1), m_t^(2)>"] = float(logged_mom_vec1_vec2_dot_prod)

        # apply iterate update w_{t+1}
        # w_{t+1} = (1 / (1 + \eta*\mu)) w_t - (\eta / (1 + \eta*\mu)) P_t^{-1} (\lambda_1 m_t^{(1)} + (1 - \lambda_1) m_t^{(2)})
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                m1 = self.state[p]["m1"]
                m2 = self.state[p]["m2"]

                # convex-combo momentum m = alpha1*m1 + alpha2*m2
                mom_vec_cvx_combo = alpha1 * m1 + alpha2 * m2

                # apply preconditioner as Pinv
                if precond == "adam":
                    v_t = self.state[p]["v_t"]
                    v_t_hat = v_t / bias_correction_denom
                    precond_t_inv_flattened = v_t_hat.sqrt().add(eps_precond).reciprocal()
                    step_dir = mom_vec_cvx_combo * precond_t_inv_flattened
                else:
                    step_dir = mom_vec_cvx_combo

                # Decoupled weight decay (AdamW): apply decay directly to the parameters!
                # then apply the preconditioned momentum step w/ no proximal shrinkage factor
                if decoupled_wd:
                    if mu != 0.0:
                        p.add_(p, alpha=-lr * mu)
                    p.add_(step_dir.to(p.dtype), alpha=-lr)
                    continue

                # The new step for the optimizer: proximal coupled weight decay step
                # w_{t+1} = (1/(1+ (eta * mu))) w_t - (η/(1+(eta * mu))) Pinv( alpha1*m1 + (1-alpha1)*m2 )
                denom = (1.0 + lr * mu)
                p.mul_(1.0 / denom)
                p.add_(step_dir.to(p.dtype), alpha=-lr / denom)

        # write back optimizer-global state to all param groups for run resume correctness
        next_step = tp_step + 1
        for _grp in self.param_groups:
            _grp["tp_barf1"] = barf1
            _grp["tp_barf2"] = barf2
            _grp["tp_gamma1"] = gamma1
            _grp["tp_gamma2"] = gamma2
            _grp["tp_step"] = next_step

        return None

    #### Per-parameter version ####
    @torch.no_grad()
    def _step_parameter(self, loss: Optional[Union[float, torch.Tensor]], log_dict: Optional[Dict] = None):
        """
        Usage:
          loss.backward()
          opt.step(loss)   # if use_loss_ema=True (default)

        If use_loss_ema=False, you may call:
          loss.backward()
          opt.step()       # loss is ignored
        """
        # Use param_groups[0] as the single "global" owner of the MoMo loss-EMA state in the per-parameter version 
        shared_group = self.param_groups[0]
        shared_barf1 = shared_group["tp_barf1"]
        shared_barf2 = shared_group["tp_barf2"]
        shared_use_loss_ema = shared_group["tp_use_loss_ema"]

        beta_s_global = self.param_groups[0]["beta_short"]
        beta_l_global = self.param_groups[0]["beta_long"]

        # global loss-EMA states 
        for group in self.param_groups[1:]:
            if group["beta_short"] != beta_s_global:
                raise ValueError("All parameter groups must share the same beta_short when using global loss EMA in parameter mode.")
            if group["beta_long"] != beta_l_global:
                raise ValueError("All parameter groups must share the same beta_long when using global loss EMA in parameter mode.")
            if group["tp_use_loss_ema"] != shared_use_loss_ema:
                raise ValueError("All parameter groups must share the same tp_use_loss_ema when using global loss EMA in parameter mode.")
            if group["alpha_scope"] != "parameter":
                raise ValueError("All parameter groups must use alpha_scope='parameter' for parameter-specific GenTwoPlaneMoMo.")

        # update the global loss-EMA state exactly once per optimizer step
        if shared_use_loss_ema:
            if loss is None:
                raise RuntimeError("TwoPlaneMoMo.step(loss=...) requires the current loss (or set use_loss_ema=False in the constructor).")
            loss_t = float(loss.detach().item() if isinstance(loss, torch.Tensor) else loss)
            # \bar{l}_{t}^{(1)} = \Beta_1 (\bar{l}_{t}^{(1)}) + (1 - \Beta_1) l_t
            shared_barf1 = beta_s_global * shared_barf1 + (1 - beta_s_global) * loss_t
            # \bar{\ell}_{t}^{(2)} = \Beta_2 (\bar{l}_{t}^{(2)}) + (1 - \Beta_2) l_t
            shared_barf2 = beta_l_global * shared_barf2 + (1 - beta_l_global) * loss_t
        else:
            loss_t = None

        for group_idx, group in enumerate(self.param_groups):
            lr = group["lr"]
            mu = group["weight_decay_factor"]
            beta_s = group["beta_short"]
            beta_l = group["beta_long"]
            eps = group["eps"]
            eps_precond = group["eps_precond"]
            precond = group["preconditioner"]
            precond_beta2 = group["precond_beta2"]
            decoupled_wd = group["decoupled_weight_decay"]
            clip_alpha = group["tp_clip_alpha"]
            use_loss_ema = group["tp_use_loss_ema"]

            # using the globally defined shared loss-EMA states from MoMo (since there is no notion of a "local" parameter-subset loss)
            barf1 = shared_barf1
            barf2 = shared_barf2


            lr_safe = max(lr, eps)
            mu_model = 0.0 if decoupled_wd else mu
            num_fac = (1 + (lr_safe * mu_model)) / lr_safe 

            param_log_idx = 0
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("TwoPlaneMoMo does not support sparse gradients.")

                state = self.state[p]

                grad = p.grad.detach()
                if grad.dtype != torch.float32:
                    grad = grad.float()

                # Check if per-parameter state doesn't exist
                if "m1" not in state:
                    state["m1"] = torch.zeros_like(p, dtype=torch.float32, memory_format=torch.preserve_format)
                if "m2" not in state:
                    state["m2"] = torch.zeros_like(p, dtype=torch.float32, memory_format=torch.preserve_format)
                if "gamma1" not in state:
                    state["gamma1"] = 1.0
                if "gamma2" not in state:
                    state["gamma2"] = 1.0
                if "tp_step" not in state:
                    state["tp_step"] = 0
                if precond == "adam" and "v_t" not in state:
                    state["v_t"] = torch.zeros_like(p, dtype=torch.float32, memory_format=torch.preserve_format)

                m1 = state["m1"]
                m2 = state["m2"]
                gamma1 = state["gamma1"]
                gamma2 = state["gamma2"]
                tp_step = state["tp_step"]

                # bias-correction factor for AdamW second moment estimate for beta2
                bias_correction_denom = 1.0 - (precond_beta2 ** (tp_step + 1))
                bias_correction_denom = max(bias_correction_denom, eps)

                # two EMAs of the gradient
                m1.mul_(beta_s).add_(grad, alpha=1 - beta_s)
                m2.mul_(beta_l).add_(grad, alpha=1 - beta_l)

                # Choose preconditioner
                if precond == "adam":
                    v_t = state["v_t"]
                    v_t.mul_(precond_beta2).addcmul_(grad, grad, value=(1 - precond_beta2))
                    v_t_hat = v_t / bias_correction_denom
                    precond_t_inv_flattened = v_t_hat.sqrt().add_(eps_precond).reciprocal()
                else:
                    precond_t_inv_flattened = None

                logged_mom_vec_1_squared_norm = torch.dot(m1.flatten(), m1.flatten()).item()
                logged_mom_vec_2_squared_norm = torch.dot(m2.flatten(), m2.flatten()).item()
                logged_mom_vec1_vec2_dot_prod = torch.dot(m1.flatten(), m2.flatten()).item()

                w_t = p.detach().float()
                m1_dot_w_t = torch.dot(m1.flatten(), w_t.flatten()).item()
                m2_dot_w_t = torch.dot(m2.flatten(), w_t.flatten()).item()
                g_t_dot_w_t = torch.dot(grad.flatten(), w_t.flatten()).item()

                if precond == "adam":
                    m1_minus_m2 = (m1 - m2)
                    final_denom = torch.sum(m1_minus_m2 * precond_t_inv_flattened * m1_minus_m2).item()
                    numer_m1_m2_precond_inv_m2 = torch.sum(m1_minus_m2 * precond_t_inv_flattened * m2).item()
                else:
                    m1_minus_m2 = (m1 - m2)
                    final_denom = torch.dot(m1_minus_m2.flatten(), m1_minus_m2.flatten()).item()
                    numer_m1_m2_precond_inv_m2 = torch.dot(m1_minus_m2.flatten(), m2.flatten()).item()

                # \gamma_{t}^{(1)} (fast EMA of <g_t, w_t>)
                gamma1 = beta_s * gamma1 + (1 - beta_s) * g_t_dot_w_t
                # \gamma_{t}^{(2)} (slow EMA of <g_t, w_t>)
                gamma2 = beta_l * gamma2 + (1 - beta_l) * g_t_dot_w_t

                if use_loss_ema:
                    b1 = barf1 - gamma1 + m1_dot_w_t
                    b2 = barf2 - gamma2 + m2_dot_w_t
                else:
                    b1 = m1_dot_w_t - gamma1
                    b2 = m2_dot_w_t - gamma2

                # second term in the numerator: (m1 - m2)^T w_t
                m1_minus_m2_dot_wt = (m1_dot_w_t - m2_dot_w_t)

                alpha_max = 0.9
                alpha_min = 0.1

                alpha1_unc = 0.0
                if final_denom <= eps:
                    alpha1 = alpha_max if b1 >= b2 else alpha_min
                    alpha1_unc = 1.0 if b1 >= b2 else 0.0
                else:
                    alpha1_unc = (num_fac * (b1 - b2) - mu_model * m1_minus_m2_dot_wt - numer_m1_m2_precond_inv_m2) / max(final_denom, eps)
                    alpha1 = alpha1_unc
                    if clip_alpha:
                        alpha1 = min(alpha_max, max(alpha_min, alpha1))
                alpha2 = 1.0 - alpha1
                self.last_alpha1 = alpha1

                if log_dict is not None:
                    prefix = f"two_plane_momo/group_{group_idx}/parameter_{param_log_idx}"
                    log_dict[f"{prefix}/alpha1"] = float(alpha1)
                    log_dict[f"{prefix}/alpha2"] = float(alpha2)
                    log_dict[f"{prefix}/final_denom"] = float(final_denom)
                    log_dict[f"{prefix}/alpha1_unclipped"] = float(alpha1_unc)
                    log_dict[f"{prefix}/||m_t^{(1)}||_{2}^{2}"] = float(logged_mom_vec_1_squared_norm)
                    log_dict[f"{prefix}/||m_t^{(2)}||_{2}^{2}"] = float(logged_mom_vec_2_squared_norm)
                    log_dict[f"{prefix}/<m_t^(1), m_t^(2)>"] = float(logged_mom_vec1_vec2_dot_prod)

                # convex-combo momentum m = alpha1*m1 + alpha2*m2
                mom_vec_cvx_combo = alpha1 * m1 + alpha2 * m2

                # apply preconditioner as Pinv
                if precond == "adam":
                    v_t = self.state[p]["v_t"]
                    v_t_hat = v_t / bias_correction_denom
                    precond_t_inv_flattened = v_t_hat.sqrt().add(eps_precond).reciprocal()
                    step_dir = mom_vec_cvx_combo * precond_t_inv_flattened
                else:
                    step_dir = mom_vec_cvx_combo

                # Decoupled weight decay (AdamW): apply decay directly to the parameters!
                # then apply the preconditioned momentum step w/ no proximal shrinkage factor
                if decoupled_wd:
                    if mu != 0.0:
                        p.add_(p, alpha=-lr * mu)
                    p.add_(step_dir.to(p.dtype), alpha=-lr)
                else:
                    # The new step for the optimizer: proximal coupled weight decay step
                    # w_{t+1} = (1/(1+ (eta * mu))) w_t - (η/(1+(eta * mu))) Pinv( alpha1*m1 + (1-alpha1)*m2 )
                    denom = (1.0 + lr * mu)
                    p.mul_(1.0 / denom)
                    p.add_(step_dir.to(p.dtype), alpha=-lr / denom)

                # write back gamma1 and gamma2 to "state", since each gamma_j here is computed per-parameter
                # Unlike barf_j, which has only a global meaning, hence it is not written to "state"
                state["gamma1"] = gamma1
                state["gamma2"] = gamma2
                state["tp_step"] = tp_step + 1
                param_log_idx += 1

        # write back barf1, barf2 to "self.param_groups[0]" --- the shared global loss-EMA state we designated --- exactly once
        shared_group["tp_barf1"] = shared_barf1
        shared_group["tp_barf2"] = shared_barf2

        # ensure to update the shared global loss-EMA state into the remaining groups for resume consistency
        for group in self.param_groups[1:]:
            group["tp_barf1"] = shared_barf1
            group["tp_barf2"] = shared_barf2
        return None

    # Step function calls either _step_network (network-wide alpha scope computation) or _step_parameter (per-parameter alpha scope computation)
    @torch.no_grad()
    def step(self, loss: Optional[Union[float, torch.Tensor]] = None, log_dict: Optional[Dict] = None):
        alpha_scope = self.param_groups[0]["alpha_scope"]
        for group in self.param_groups[1:]:
            if group["alpha_scope"] != alpha_scope:
                raise ValueError("All parameter groups must share the same alpha_scope.")

        if alpha_scope == "network":
            return self._step_network(loss, log_dict)
        if alpha_scope == "parameter":
            return self._step_parameter(loss, log_dict)

        raise ValueError("alpha_scope must be one of {'network', 'parameter'}!")
