from typing import Dict, Iterable, Optional, Union
import math
import torch
from torch.optim import Optimizer


####################################################
# Sanity check only. Comment out once done: forces alpha1 and alpha2 to match ademamix's for consistency check 
# AdEMAMix alpha scheduler
def linear_warmup_scheduler(step, alpha_end, alpha_start=0, warmup=1):
    if step < warmup:
        a = step / float(warmup)
        return (1.0 - a) * alpha_start + a * alpha_end
    return alpha_end
####################################################

# AdEMAMix beta3 scheduler
def linear_hl_warmup_scheduler(step, beta_end, beta_start=0, warmup=1):
    def f(beta, eps=1e-8):
        return math.log(0.5) / math.log(beta + eps) - 1

    def f_inv(t):
        return math.pow(0.5, 1 / (t + 1))

    if step < warmup:
        a = step / float(warmup)
        return f_inv((1.0 - a) * f(beta_start) + a * f(beta_end))
    return beta_end

def dual_function(lambda_1, lambda_2, lambda_3, mu_model, eta, w_t_precond_w_t, lambda_combo_dot_w_t, lambda_combo_precond_inv_norm_sq, b1, b2, fstar):
    fac1 = mu_model / (2.0 * (1.0 + eta * mu_model))
    fac2 = (eta * mu_model) / (1.0 + eta * mu_model)
    fac3 = eta / (2.0 * (1.0 + eta * mu_model))
    dual = (fac1 * w_t_precond_w_t) - (fac2 * lambda_combo_dot_w_t) - (fac3 * lambda_combo_precond_inv_norm_sq) + (lambda_1 * b1) + (lambda_2 * b2) + (lambda_3 * fstar)
    return float(dual)


class GenThreeNesterovPlaneMoMo(Optimizer):
    """
    Three-Plane MoMo
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
        beta_long_start: float = 0.5,
        beta_long_warmup_steps: Optional[int] = None,
        eps: float = 1e-12,
        clip_alpha: bool = True,
        use_loss_ema: bool = True,
        alpha_denom_correction: float = 0.0,
        preconditioner: str = "identity",  # string options: {"identity", "adam"}
        precond_beta2: float = 0.999,  # Adam second-moment beta2
        weight_decay_factor: float = 0.0,
        eps_precond: float = 1e-12,
        decoupled_weight_decay: bool = False,
        alpha_scope: str = "network",
        fstar: float = 3.0,
        rho_reliability: float = 0.99,
        reliability_lambda: float = 1.0, # 1.0 turns on reliability reward and penalization
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
        if not (0.0 <= beta_long_start < 1.0): # omit the degen no update beta=1 case
            raise ValueError("beta_long_start in [0,1)")
        if beta_long_warmup_steps is not None and beta_long_warmup_steps < 1:
            raise ValueError("beta_long_warmup must be >= 1 or None")
        if not (0.0 <= rho_reliability < 1.0):
            raise ValueError("rho_reliability must be in [0,1)")
        if reliability_lambda < 0:
            raise ValueError("reliability_lambda must be non-negative")

        defaults = dict(
            lr=lr,
            beta_short=beta_short,  # short-horizon EMA decay for the gradient
            beta_long=beta_long,  # long-horizon EMA decay for the gradient /slower timescale
            beta_long_start=beta_long_start,
            beta_long_warmup_steps=beta_long_warmup_steps,
            eps=eps,
            eps_precond=eps_precond,
            preconditioner=preconditioner,
            precond_beta2=precond_beta2,  # adam style second-moment EMA decay
            weight_decay_factor=weight_decay_factor,
            decoupled_weight_decay=decoupled_weight_decay,
            alpha_scope=alpha_scope,
            fstar=fstar,
            tp_clip_alpha=clip_alpha,
            tp_use_loss_ema=use_loss_ema,
            alpha_denom_correction = alpha_denom_correction,
            rho_reliability=rho_reliability,
            reliability_lambda=reliability_lambda,
        )

        super().__init__(params, defaults)

        for g in self.param_groups:
            g["tp_barf1"] = 0.0
            g["tp_gamma1"] = 0.0
            # g["tp_barf2"] = 1.0
            ###########################################
            # Temporary: Trying to inject Nesterov to the second plane!! (comment the line below, uncomment above)
            ###########################################
            g["tp_barf2"] = 0.0
            # g["tp_gamma2"] = 1.0
            ###########################################
            # Temporary: Trying to inject Nesterov to the second plane!! (comment the line below, uncomment above)
            ###########################################
            g["tp_gamma2"] = 0.0
            g["tp_barf2_gs"] = 0.0
            g["tp_gamma2_gs"] = 0.0

            g["tp_reliability_ema1"] = 0.0
            g["tp_reliability_ema2"] = 0.0

            g["tp_prev_intercept1"] = 0.0
            g["tp_prev_intercept2"] = 0.0
            g["tp_reliability_initialized"] = False

            g["tp_step"] = 0
            g.setdefault("tp_clip_alpha", clip_alpha)
            g.setdefault("tp_use_loss_ema", use_loss_ema)
            g.setdefault("alpha_denom_correction", alpha_denom_correction)
            g.setdefault("preconditioner", preconditioner)
            g.setdefault("precond_beta2", precond_beta2)
            g.setdefault("weight_decay_factor", weight_decay_factor)
            g.setdefault("eps_precond", eps_precond)
            g.setdefault("decoupled_weight_decay", decoupled_weight_decay)
            g.setdefault("alpha_scope", alpha_scope)
            g.setdefault("fstar", fstar)

            g.setdefault("rho_reliability", rho_reliability)
            g.setdefault("reliability_lambda", reliability_lambda)

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
        beta_l_final = self.param_groups[0]["beta_long"]
        beta_l_start = self.param_groups[0]["beta_long_start"]
        beta_l_warmup_steps = self.param_groups[0]["beta_long_warmup_steps"]

        rho_reliability = self.param_groups[0]["rho_reliability"]
        reliability_lambda = self.param_groups[0]["reliability_lambda"]
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

        barf2_gs = g["tp_barf2_gs"]
        gamma2_gs = g["tp_gamma2_gs"]

        reliability_ema1 = g["tp_reliability_ema1"]
        reliability_ema2 = g["tp_reliability_ema2"]

        prev_intercept1 = g["tp_prev_intercept1"]
        prev_intercept2 = g["tp_prev_intercept2"]
        reliability_initialized = g["tp_reliability_initialized"]
        clip_alpha = g["tp_clip_alpha"]
        use_loss_ema = g["tp_use_loss_ema"]
        alpha_denom_correction = g["alpha_denom_correction"]
        stored_tp_step = g["tp_step"]
        tp_step = stored_tp_step + 1 # To match AdEMAMix which uses pre-increment timeline

        # sync optimizer-wide scalar state from param_groups[0] to all other groups
        for group in self.param_groups[1:]:
            group["tp_barf1"] = barf1
            group["tp_barf2"] = barf2
            group["tp_gamma1"] = gamma1
            group["tp_gamma2"] = gamma2
            group["tp_barf2_gs"] = barf2_gs
            group["tp_gamma2_gs"] = gamma2_gs
            group["tp_reliability_ema1"] = reliability_ema1
            group["tp_reliability_ema2"] = reliability_ema2
            group["tp_prev_intercept1"] = prev_intercept1
            group["tp_prev_intercept2"] = prev_intercept2
            group["tp_reliability_initialized"] = reliability_initialized
            group["tp_step"] = stored_tp_step

        if beta_l_warmup_steps is not None:
            beta_l = linear_hl_warmup_scheduler(
                tp_step,
                beta_end=beta_l_final,
                beta_start=beta_l_start,
                warmup=beta_l_warmup_steps,
            )
        else:
            beta_l = beta_l_final

        # Flipped weights v1: slow
        # nesterov_decay = (2.0 * beta_l) - (beta_l * beta_l)
        # nesterov_new_weight = 1.0 - beta_l

        # Flipped weights v2: reactive
        nesterov_decay = (1.0 - beta_l) + (beta_l * beta_l)
        nesterov_new_weight = beta_l

        if log_dict is not None:
            log_dict["three_plane_momo/beta_long_used"] = float(beta_l)
            log_dict["three_plane_momo/nesterov_decay_used"] = float(nesterov_decay)
            log_dict["three_plane_momo/nesterov_new_weight_used"] = float(nesterov_new_weight)

        # ensure all parameter groups must share the same optimizer-wide hyperparameters and scalar-state semantics
        # since this implementation computes one network-wide alpha using param_groups[0] as the global owner.
        for group in self.param_groups[1:]:
            if group["lr"] != lr:
                raise ValueError("All parameter groups must share the same lr for network-wide GenThreePlaneMoMo.")
            if group["beta_short"] != beta_s:
                raise ValueError("All parameter groups must share the same beta_short for network-wide GenThreePlaneMoMo.")
            if group["beta_long"] != beta_l_final:
                raise ValueError("All parameter groups must share the same beta_long for network-wide GenThreePlaneMoMo.")
            if group["beta_long_start"] != beta_l_start:
                raise ValueError("All parameter groups must share the same beta_long_start for network-wide GenThreePlaneMoMo.")
            if group["beta_long_warmup_steps"] != beta_l_warmup_steps:
                raise ValueError("All parameter groups must share the same beta_long_warmup_steps for network-wide GenThreePlaneMoMo.")
            if group["rho_reliability"] != rho_reliability:
                raise ValueError("All parameter groups must share the same rho_reliability for network-wide GenThreePlaneMoMo.")
            if group["reliability_lambda"] != reliability_lambda:
                raise ValueError("All parameter groups must share the same reliability_lambda for network-wide GenThreePlaneMoMo.")
            if group["eps"] != eps:
                raise ValueError("All parameter groups must share the same eps for network-wide GenThreePlaneMoMo.")
            if group["eps_precond"] != eps_precond:
                raise ValueError("All parameter groups must share the same eps_precond for network-wide GenThreePlaneMoMo.")
            if group["preconditioner"] != precond:
                raise ValueError("All parameter groups must share the same preconditioner for network-wide GenThreePlaneMoMo.")
            if group["precond_beta2"] != precond_beta2:
                raise ValueError("All parameter groups must share the same precond_beta2 for network-wide GenThreePlaneMoMo.")
            if group["weight_decay_factor"] != mu:
                raise ValueError("All parameter groups must share the same weight_decay_factor for network-wide GenThreePlaneMoMo.")
            if group["decoupled_weight_decay"] != decoupled_wd:
                raise ValueError("All parameter groups must share the same decoupled_weight_decay for network-wide GenThreePlaneMoMo.")
            if group["tp_clip_alpha"] != clip_alpha:
                raise ValueError("All parameter groups must share the same tp_clip_alpha for network-wide GenThreePlaneMoMo.")
            if group["tp_use_loss_ema"] != use_loss_ema:
                raise ValueError("All parameter groups must share the same tp_use_loss_ema for network-wide GenThreePlaneMoMo.")
            if group["alpha_denom_correction"] != alpha_denom_correction:
                raise ValueError("All parameter groups must share the same alpha_denom_correction for network-wide GenThreePlaneMoMo.")
            if group["alpha_scope"] != "network":
                raise ValueError("All parameter groups must use alpha_scope='network' for network-wide GenThreePlaneMoMo.")

        lr_safe = max(lr, eps)
        mu_model = 0.0 if decoupled_wd else mu  # turn off the proximal coupled subproblem when using decoupled weight decay

        # For: lambda_1,unc (Factor in the first term in the numerator)
        num_fac = (1 + (lr_safe * mu_model)) / lr_safe

        # For: \hat{v}_t --- the bias corrected second moment estimate --- used in the Adam preconditioner.
        # bias_correction_denom = 1.0 - (precond_beta2 ** (tp_step + 1))
        bias_correction_denom = 1.0 - (precond_beta2 ** (tp_step)) # To match AdEMAMix which uses pre-increment timeline

        bias_correction_denom = max(bias_correction_denom, eps)

        if use_loss_ema:
            if loss is None:
                raise RuntimeError(
                    "ThreePlaneMoMo.step(loss=...) requires the current loss (or set use_loss_ema=False in the constructor)."
                )
            loss_t = float(loss.detach().item() if isinstance(loss, torch.Tensor) else loss)
        else:
            loss_t = None

        # accumulators (global)
        logged_mom_vec_1_squared_norm = 0.0
        logged_mom_vec_2_squared_norm = 0.0
        logged_mom_vec1_vec2_dot_prod = 0.0
        logged_grad_m1_dot_prod = 0.0
        logged_grad_m2_dot_prod = 0.0

        logged_grad_squared_norm = 0.0
        logged_m1_minus_m2_squared_norm = 0.0

        pinv_diag_entry_sum = 0.0
        pinv_diag_entry_max = 0.0
        pinv_diag_entry_count = 0
        pinv_m2_squared_norm = 0.0
        pinv_m1_minus_m2_squared_norm = 0.0 

        w_t_precond_w_t = 0.0
        denom_m1_precond_inv_m1 = 0.0
        denom_m2_precond_inv_m2 = 0.0
        inner_m1_precond_inv_m2 = 0.0

        # For: lambda_1,unc
        m1_dot_w_t = 0.0
        m2_dot_w_t = 0.0
        g_t_dot_w_t = 0.0

        prev_m1_dot_w_t_for_reliability = 0.0
        prev_m2_dot_w_t_for_reliability = 0.0

        # For: "lambda_1,unc" with preconditioner (P_t^{-1})
        denom_m1_m2_precond_inv_m1_m2 = 0.0 # (m1-m2)^T Pinv (m1-m2)
        numer_m1_m2_precond_inv_m2 = 0.0 # (m1-m2)^T Pinv m2

        # 1) update per-parameter EMAs m1/m2 and accumulate inner products
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("ThreePlaneMoMo does not support sparse gradients.")
                state = self.state[p]

                grad = p.grad.detach()
                if grad.dtype != torch.float32:
                    grad = grad.float()

                # Check if per-parameter state doesn't exist
                if "m1" not in state:
                    state["m1"] = torch.zeros_like(p, dtype=torch.float32, memory_format=torch.preserve_format)
                if "m2" not in state:
                    state["m2"] = torch.zeros_like(p, dtype=torch.float32, memory_format=torch.preserve_format)

                if "m2_gs" not in state:
                    state["m2_gs"] = 0.0
                if precond == "adam" and "v_t" not in state:
                    state["v_t"] = torch.zeros_like(p, dtype=torch.float32, memory_format=torch.preserve_format)

                # For: lambda_unc, and w_{t+1}
                m1 = state["m1"]  # m_t^{(1)}
                m2 = state["m2"]  # m_t^{(2)}

                # pred reliability: uses the previous-step planes evaluated at the current iterate w_t
                # computed before updating m1 and m2 with the current gradient
                w_t_for_reliability = p.detach().float()

                # prev_m1_for_reliability = m1 
                ####################################################
                # Temporary: Potentially consider bias correction to momentum 1.
                prev_m1_for_reliability = m1
                ####################################################
                prev_m2_for_reliability = m2 / max(state["m2_gs"], eps)
                prev_m1_dot_w_t_for_reliability += torch.dot(prev_m1_for_reliability.flatten(), w_t_for_reliability.flatten()).item()
                prev_m2_dot_w_t_for_reliability += torch.dot(prev_m2_for_reliability.flatten(), w_t_for_reliability.flatten()).item()

                # two EMAs of the gradient
                # m1.mul_(beta_s).add_(grad, alpha=1 - beta_s)  # m_t^{(1)} = \Beta_short m_{t-1}^{(1)} + (1 - \Beta_short) g_t
                # m2.mul_(nesterov_decay).add_(grad, alpha=nesterov_new_weight)
                # state["m2_gs"] = (nesterov_decay * state["m2_gs"]) + nesterov_new_weight
                if stored_tp_step == 0:
                    m1.copy_(grad)
                    m2.copy_(grad)
                    state["m2_gs"] = 1.0
                else:
                    m1.mul_(beta_s).add_(grad, alpha=1 - beta_s)  # m_t^{(1)} = \Beta_short m_{t-1}^{(1)} + (1 - \Beta_short) g_t
                    m2.mul_(nesterov_decay).add_(grad, alpha=nesterov_new_weight)
                    state["m2_gs"] = (nesterov_decay * state["m2_gs"]) + nesterov_new_weight
                # m2.mul_(beta_l).add_(grad, alpha=1 - beta_l)  # m_t^{(2)} = \Beta_long m_{t-1}^{(2)} + (1 - \Beta_long) g_t
                m2_nesterov = m2 / max(state["m2_gs"], eps)

                # m1_for_update = m1
                ####################################################
                # Temporary: Potentially consider bias correction to momentum 1.
                #modify
                # m1_for_update = m1 / max(1.0 - beta_s ** tp_step, eps)
                m1_for_update = m1
                ####################################################

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
                    pinv_diag_entry_sum += precond_t_inv_flattened.sum().item()
                    pinv_diag_entry_max = max(pinv_diag_entry_max, precond_t_inv_flattened.max().item())
                    pinv_diag_entry_count += precond_t_inv_flattened.numel()
                    # pinv_m2_squared_norm += torch.dot((precond_t_inv_flattened * m2).flatten(), (precond_t_inv_flattened * m2).flatten()).item()  
                    pinv_m2_squared_norm += torch.dot((precond_t_inv_flattened * m2_nesterov).flatten(), (precond_t_inv_flattened * m2_nesterov).flatten()).item()  
                    # pinv_m1_minus_m2_squared_norm += torch.dot((precond_t_inv_flattened * (m1_for_update - m2)).flatten(), (precond_t_inv_flattened * (m1_for_update - m2)).flatten()).item()  
                    pinv_m1_minus_m2_squared_norm += torch.dot((precond_t_inv_flattened * (m1_for_update - m2_nesterov)).flatten(), (precond_t_inv_flattened * (m1_for_update - m2_nesterov)).flatten()).item()  
                else:
                    precond_t_inv_flattened = None

                # global inner products (only for wandb logging analysis)
                logged_mom_vec_1_squared_norm += torch.dot(m1_for_update.flatten(), m1_for_update.flatten()).item()
                # logged_mom_vec_2_squared_norm += torch.dot(m2.flatten(), m2.flatten()).item()
                logged_mom_vec_2_squared_norm += torch.dot(m2_nesterov.flatten(), m2_nesterov.flatten()).item()
                # logged_mom_vec1_vec2_dot_prod += torch.dot(m1_for_update.flatten(), m2.flatten()).item()
                logged_mom_vec1_vec2_dot_prod += torch.dot(m1_for_update.flatten(), m2_nesterov.flatten()).item()
                logged_grad_m1_dot_prod += torch.dot(grad.flatten(), m1_for_update.flatten()).item()
                # logged_grad_m2_dot_prod += torch.dot(grad.flatten(), m2.flatten()).item()
                logged_grad_m2_dot_prod += torch.dot(grad.flatten(), m2_nesterov.flatten()).item()
                logged_grad_squared_norm += torch.dot(grad.flatten(), grad.flatten()).item()
                logged_m1_minus_m2_squared_norm += torch.dot((m1_for_update - m2_nesterov).flatten(), (m1_for_update - m2_nesterov).flatten()).item()  

                # For: lambda_1,unc
                w_t = p.detach().float()
                if precond == "adam":
                    w_t_precond_w_t += torch.sum(w_t * precond_t_inv_flattened.reciprocal() * w_t).item()
                else:
                    w_t_precond_w_t += torch.dot(w_t.flatten(), w_t.flatten()).item()

                m1_dot_w_t += torch.dot(m1_for_update.flatten(), w_t.flatten()).item() # <m_t^{(1)}, w_t> (second term in the numerator) 
                m2_dot_w_t += torch.dot(m2_nesterov.flatten(), w_t.flatten()).item() # <m_t^{(2)}, w_t> (second term in the numerator)

                # For: lambda_1,unc --- MoMo: by first building \gamma_{t}^{(i)} -> build b_t^{(1)} and b_t^{(2)} -> build lambda_1_unc.
                g_t_dot_w_t += torch.dot(grad.flatten(), w_t.flatten()).item()

                # preconditioned quadratic forms for lambda_1,unc
                if precond == "adam":
                    # For: lambda_1,unc  (numerator and denominator)
                    m1_minus_m2 = (m1_for_update - m2_nesterov) # m_t^{(1)} - m_t^{(2)} 
                    # For: lambda_1,unc (denominator)
                    # (m1_minus_m2)^T P_t^{-1} (m1_minus_m2) --- quadratic form!
                    # Since P_t^{-1} is diagonal we can compute this equivalently as the sum over coordinates:
                    #        = \sum_i (P_t^{-1})_{i,i} * ((m1_minus_m2)_i)^2
                    denom_m1_m2_precond_inv_m1_m2 += torch.sum(m1_minus_m2 * precond_t_inv_flattened * m1_minus_m2).item()

                    denom_m1_precond_inv_m1 += torch.sum(m1_for_update * precond_t_inv_flattened * m1_for_update).item()
                    denom_m2_precond_inv_m2 += torch.sum(m2_nesterov * precond_t_inv_flattened * m2_nesterov).item()
                    inner_m1_precond_inv_m2 += torch.sum(m1_for_update * precond_t_inv_flattened * m2_nesterov).item()

                    # For: lambda_1,unc (3rd term in the numerator)
                    # (m1_minus_m2^T) P_t^{-1} (m_t^{(2)})
                    # with diagonal P_t^{-1}, this equals:
                    #   numer_m1_m2_precond_inv_m2 = \sum_i (P_t^{-1})_{i,i} * (m1_minus_m2)_i * (m_t^{(2)})_i
                    numer_m1_m2_precond_inv_m2 += torch.sum(m1_minus_m2 * precond_t_inv_flattened * m2_nesterov).item()
                else:
                    m1_minus_m2 = (m1_for_update - m2_nesterov)
                    denom_m1_precond_inv_m1 += torch.dot(m1_for_update.flatten(), m1_for_update.flatten()).item()
                    denom_m2_precond_inv_m2 += torch.dot(m2_nesterov.flatten(), m2_nesterov.flatten()).item()
                    inner_m1_precond_inv_m2 += torch.dot(m1_for_update.flatten(), m2_nesterov.flatten()).item()
                    denom_m1_m2_precond_inv_m1_m2 += torch.dot(m1_minus_m2.flatten(), m1_minus_m2.flatten()).item()
                    numer_m1_m2_precond_inv_m2 += torch.dot(m1_minus_m2.flatten(), m2_nesterov.flatten()).item()

        if use_loss_ema and reliability_initialized:
            # prev_ell1_of_w_t_for_reliability = prev_b1 + prev_m1_dot_w_t_for_reliability
            # prev_ell2_of_w_t_for_reliability = prev_b2 + prev_m2_dot_w_t_for_reliability
            prev_ell1_of_w_t_for_reliability = prev_intercept1 + prev_m1_dot_w_t_for_reliability
            prev_ell2_of_w_t_for_reliability = prev_intercept2 + prev_m2_dot_w_t_for_reliability
            reliability_error1 = abs(loss_t - prev_ell1_of_w_t_for_reliability)
            reliability_error2 = abs(loss_t - prev_ell2_of_w_t_for_reliability)
            reliability_ema1 = rho_reliability * reliability_ema1 + (1.0 - rho_reliability) * reliability_error1
            reliability_ema2 = rho_reliability * reliability_ema2 + (1.0 - rho_reliability) * reliability_error2
        else:
            prev_ell1_of_w_t_for_reliability = 0.0
            prev_ell2_of_w_t_for_reliability = 0.0
            reliability_error1 = 0.0
            reliability_error2 = 0.0

        # For: lambda_1,unc --- MoMo: by first building b_t^{(1)} and b_t^{(2)} -> build lambda_1_unc.
        if use_loss_ema:
            # \bar{l}_{t}^{(1)} = \Beta_1 (\bar{l}_{t}^{(1)}) + (1 - \Beta_1) l_t
            # barf1 = beta_s * barf1 + (1 - beta_s) * loss_t
            # \bar{\ell}_{t}^{(2)} = \Beta_2 (\bar{l}_{t}^{(2)}) + (1 - \Beta_2) l_t
            # barf2 = beta_l * barf2 + (1 - beta_l) * loss_t
            # barf2 = nesterov_decay * barf2 + nesterov_new_weight * loss_t
            # barf2_gs = nesterov_decay * barf2_gs + nesterov_new_weight
            if stored_tp_step == 0:
                barf1 = loss_t
                barf2 = loss_t
                barf2_gs = 1.0
            else:
                barf1 = beta_s * barf1 + (1 - beta_s) * loss_t
                barf2 = nesterov_decay * barf2 + nesterov_new_weight * loss_t
                barf2_gs = nesterov_decay * barf2_gs + nesterov_new_weight
            barf2_nesterov = barf2 / max(barf2_gs, eps)

        # \gamma_{t}^{(1)} (fast EMA of <g_t, w_t>)
        # gamma1 = beta_s * gamma1 + (1 - beta_s) * g_t_dot_w_t
        # \gamma_{t}^{(2)} (slow EMA of <g_t, w_t>)
        # gamma2 = beta_l * gamma2 + (1 - beta_l) * g_t_dot_w_t
        # gamma2 = nesterov_decay * gamma2 + nesterov_new_weight * g_t_dot_w_t
        # gamma2_gs = nesterov_decay * gamma2_gs + nesterov_new_weight
        if stored_tp_step == 0:
            gamma1 = g_t_dot_w_t
            gamma2 = g_t_dot_w_t
            gamma2_gs = 1.0
        else:
            gamma1 = beta_s * gamma1 + (1 - beta_s) * g_t_dot_w_t
            gamma2 = nesterov_decay * gamma2 + nesterov_new_weight * g_t_dot_w_t
            gamma2_gs = nesterov_decay * gamma2_gs + nesterov_new_weight
        gamma2_nesterov = gamma2 / max(gamma2_gs, eps)

        # For: lambda_1,unc
        if use_loss_ema:
            # b1 = barf1 - gamma1 + m1_dot_w_t
            # b2 = barf2 - gamma2 + m2_dot_w_t
            # b2 = barf2_nesterov - gamma2_nesterov + m2_dot_w_t
            current_intercept1 = barf1 - gamma1
            current_intercept2 = barf2_nesterov - gamma2_nesterov
            b1_raw = current_intercept1 + m1_dot_w_t
            b2_raw = current_intercept2 + m2_dot_w_t
        else:
            # b1 = m1_dot_w_t - gamma1
            # b2 = m2_dot_w_t - gamma2
            # b2 = m2_dot_w_t - gamma2_nesterov
            current_intercept1 = -gamma1
            current_intercept2 = -gamma2_nesterov
            b1_raw = current_intercept1 + m1_dot_w_t
            b2_raw = current_intercept2 + m2_dot_w_t

        # Prediction-reliability-adjusted plane heights.
        # To recover the previous version, set reliability_lambda=0.0 or comment out these two lines and use b1=b1_raw, b2=b2_raw.
        reliability_ema_avg = 0.5 * (reliability_ema1 + reliability_ema2)
        if reliability_lambda > 0.0:
            reliability_centered_adjustment1 = reliability_lambda * (reliability_ema1 - reliability_ema_avg)
            reliability_centered_adjustment2 = reliability_lambda * (reliability_ema2 - reliability_ema_avg)
            b1 = b1_raw - reliability_centered_adjustment1
            b2 = b2_raw - reliability_centered_adjustment2
        else:
            reliability_centered_adjustment1 = 0.0
            reliability_centered_adjustment2 = 0.0
            b1 = b1_raw
            b2 = b2_raw
        # store the intercepts, not b_i.  The next reliability check evaluates
        # ell_t^{(i)}(w_{t+1/current}) = intercept_t^{(i)} + <m_t^{(i)}, w_{next/current}>.
        # prev_b1 = b1_raw
        # prev_b2 = b2_raw
        prev_intercept1 = current_intercept1
        prev_intercept2 = current_intercept2
        reliability_initialized = True

        # Remove this! Doesn't make MoMo sense. But we try: Force b1 and b2 (this might not make theoretical sense. But makes empirical sense)
        # b1 = b2

        # wandb logging
        if log_dict is not None:
            log_dict["three_plane_momo/network/barf1"] = float(barf1)
            log_dict["three_plane_momo/network/barf2"] = float(barf2)
            log_dict["three_plane_momo/network/gamma1"] = float(gamma1)
            log_dict["three_plane_momo/network/gamma2"] = float(gamma2)
            log_dict["three_plane_momo/network/barf2_gs"] = float(barf2_gs)
            log_dict["three_plane_momo/network/gamma2_gs"] = float(gamma2_gs)
            log_dict["three_plane_momo/network/barf2_nesterov_corrected"] = float(barf2_nesterov) if use_loss_ema else 0.0
            log_dict["three_plane_momo/network/gamma2_nesterov_corrected"] = float(gamma2_nesterov)
            log_dict["three_plane_momo/network/b1"] = float(b1)
            log_dict["three_plane_momo/network/b2"] = float(b2)
            log_dict["three_plane_momo/network/b1_raw_before_reliability_adjustment"] = float(b1_raw)
            log_dict["three_plane_momo/network/b2_raw_before_reliability_adjustment"] = float(b2_raw)
            log_dict["three_plane_momo/network/reliability_ema1"] = float(reliability_ema1)
            log_dict["three_plane_momo/network/reliability_ema2"] = float(reliability_ema2)
            log_dict["three_plane_momo/network/reliability_error1"] = float(reliability_error1)
            log_dict["three_plane_momo/network/reliability_error2"] = float(reliability_error2)
            log_dict["three_plane_momo/network/reliability_lambda"] = float(reliability_lambda)
            log_dict["three_plane_momo/network/rho_reliability"] = float(rho_reliability)
            log_dict["three_plane_momo/network/prev_ell1_of_w_t_for_reliability"] = float(prev_ell1_of_w_t_for_reliability)
            log_dict["three_plane_momo/network/prev_ell2_of_w_t_for_reliability"] = float(prev_ell2_of_w_t_for_reliability)
            log_dict["three_plane_momo/network/reliability_ema_avg"] = float(reliability_ema_avg)
            log_dict["three_plane_momo/network/reliability_centered_adjustment1"] = float(reliability_centered_adjustment1)
            log_dict["three_plane_momo/network/reliability_centered_adjustment2"] = float(reliability_centered_adjustment2)
            log_dict["three_plane_momo/network/reliability_adjustment1"] = float(reliability_centered_adjustment1)
            log_dict["three_plane_momo/network/reliability_adjustment2"] = float(reliability_centered_adjustment2)
            log_dict["three_plane_momo/network/<m_t^(1), w_t>"] = float(m1_dot_w_t)  
            log_dict["three_plane_momo/network/<m_t^(2), w_t>"] = float(m2_dot_w_t)  
            log_dict["three_plane_momo/network/<m_t^(1), w_t>-<m_t^(2), w_t>"] = float(m1_dot_w_t - m2_dot_w_t)  
            log_dict["three_plane_momo/network/<g_t, w_t>"] = float(g_t_dot_w_t)  
            log_dict["three_plane_momo/network/barf1_minus_barf2"] = float(barf1 - barf2)  
            log_dict["three_plane_momo/network/gamma1_minus_gamma2"] = float(gamma1 - gamma2)  

        # For: lambda_1,unc (2nd term in the numerator: (m1 - m2)^T w_t)
        m1_minus_m2_dot_wt = (m1_dot_w_t - m2_dot_w_t)

        # Constrained lambda_{1,unc} -> lambda_{1} CLIPPED!
        alpha_max = 1.0
        alpha_min = 0.0
        # For our three-plane simplex subproblem, projection is required, the projected edge candidates must always lie on the simplex edges


        # Compute all terms required for the 3 candidate dual function 
        fstar = self.param_groups[0]["fstar"]

        #### Candidate A ####
        cand_A_numer_term_A_num_fac_times_b_minus_fstar = num_fac * (b1 - fstar)
        cand_A_numer_term_B_mu_times_m1_dot_wt = mu_model * m1_dot_w_t
        cand_A_final_numer = cand_A_numer_term_A_num_fac_times_b_minus_fstar - cand_A_numer_term_B_mu_times_m1_dot_wt
        cand_A_final_denom = denom_m1_precond_inv_m1
        cand_A_corrected_denom = max(cand_A_final_denom, eps) + alpha_denom_correction
        # Unconstrained alpha/ lambda
        # Degenerate edge: objective is linear/constant, so the projected maximizer is an endpoint
        if cand_A_final_denom <= eps:
            cand_A_alpha1_unc = 1.0 if cand_A_final_numer >= 0.0 else 0.0
        else:
            cand_A_alpha1_unc = cand_A_final_numer / cand_A_corrected_denom
        # Ascertain the alphas/ lambdas
        cand_A_alpha1 = min(alpha_max, max(alpha_min, cand_A_alpha1_unc))
        cand_A_alpha2 = 0.0
        cand_A_alpha3 = 1.0 - cand_A_alpha1
        # Things that the dual function require
        cand_A_lambda_combo_dot_w_t = cand_A_alpha1 * m1_dot_w_t
        cand_A_lambda_combo_precond_inv_norm_sq = (cand_A_alpha1 * cand_A_alpha1) * denom_m1_precond_inv_m1
        # Compute dual value
        cand_A_dual = dual_function(cand_A_alpha1, cand_A_alpha2, cand_A_alpha3, mu_model, lr_safe, w_t_precond_w_t, cand_A_lambda_combo_dot_w_t, cand_A_lambda_combo_precond_inv_norm_sq, b1, b2, fstar)

        #### Candidate B ####
        cand_B_numer_term_A_num_fac_times_b_minus_fstar = num_fac * (b2 - fstar)
        cand_B_numer_term_B_mu_times_m2_dot_wt = mu_model * m2_dot_w_t
        cand_B_final_numer = cand_B_numer_term_A_num_fac_times_b_minus_fstar - cand_B_numer_term_B_mu_times_m2_dot_wt
        cand_B_final_denom = denom_m2_precond_inv_m2
        cand_B_corrected_denom = max(cand_B_final_denom, eps) + alpha_denom_correction
        # Unconstrained alpha/ lambda
        # Degenerate edge: objective is linear/constant, so the projected maximizer is an endpoint
        if cand_B_final_denom <= eps:
            cand_B_alpha2_unc = 1.0 if cand_B_final_numer >= 0.0 else 0.0
        else:
            cand_B_alpha2_unc = cand_B_final_numer / cand_B_corrected_denom
        # Ascertain the alphas/ lambdas
        cand_B_alpha2 = min(alpha_max, max(alpha_min, cand_B_alpha2_unc))
        cand_B_alpha1 = 0.0
        cand_B_alpha3 = 1.0 - cand_B_alpha2
        # Things that the dual function require
        cand_B_lambda_combo_dot_w_t = cand_B_alpha2 * m2_dot_w_t
        cand_B_lambda_combo_precond_inv_norm_sq = (cand_B_alpha2 * cand_B_alpha2) * denom_m2_precond_inv_m2
        # Compute dual value
        cand_B_dual = dual_function(cand_B_alpha1, cand_B_alpha2, cand_B_alpha3, mu_model, lr_safe, w_t_precond_w_t, cand_B_lambda_combo_dot_w_t, cand_B_lambda_combo_precond_inv_norm_sq, b1, b2, fstar)

        #### Candidate C (same two plane momo) ####
        cand_C_numer_term_A_num_fac_times_b_gap = num_fac * (b1 - b2)  
        cand_C_numer_term_B_mu_times_m1_minus_m2_dot_wt = mu_model * m1_minus_m2_dot_wt  
        cand_C_numer_term_C_m1_minus_m2_dot_Pinv_m2 = numer_m1_m2_precond_inv_m2  
        cand_C_final_numer = cand_C_numer_term_A_num_fac_times_b_gap - cand_C_numer_term_B_mu_times_m1_minus_m2_dot_wt - cand_C_numer_term_C_m1_minus_m2_dot_Pinv_m2  
        cand_C_final_denom = denom_m1_m2_precond_inv_m1_m2
        cand_C_corrected_denom = max(cand_C_final_denom, eps) + alpha_denom_correction
        # Unconstrained alpha/ lambda
        # Degenerate edge: objective is linear/constant, so the projected maximizer is an endpoint
        if cand_C_final_denom <= eps:
            cand_C_alpha1_unc = 1.0 if cand_C_final_numer >= 0.0 else 0.0
        else:
            cand_C_alpha1_unc = cand_C_final_numer / cand_C_corrected_denom
        # Ascertain the alphas/ lambdas
        cand_C_alpha1 = min(alpha_max, max(alpha_min, cand_C_alpha1_unc))
        cand_C_alpha2 = 1.0 - cand_C_alpha1
        cand_C_alpha3 = 0.0
        # Things that the dual function require
        cand_C_lambda_combo_dot_w_t = cand_C_alpha1 * m1_dot_w_t + cand_C_alpha2 * m2_dot_w_t
        cand_C_lambda_combo_precond_inv_norm_sq = (cand_C_alpha1 * cand_C_alpha1 * denom_m1_precond_inv_m1) + (2.0 * cand_C_alpha1 * cand_C_alpha2 * inner_m1_precond_inv_m2) + (cand_C_alpha2 * cand_C_alpha2 * denom_m2_precond_inv_m2)
        # Compute dual value
        cand_C_dual = dual_function(cand_C_alpha1, cand_C_alpha2, cand_C_alpha3, mu_model, lr_safe, w_t_precond_w_t, cand_C_lambda_combo_dot_w_t, cand_C_lambda_combo_precond_inv_norm_sq, b1, b2, fstar)

        # Solve the 2x2 KKT system in the least-squares sense, then check the residual.
        # This handles both the invertible case and the singular-but-consistent case.
        interior_matrix_a11 = denom_m1_precond_inv_m1
        interior_matrix_a12 = inner_m1_precond_inv_m2
        interior_matrix_a22 = denom_m2_precond_inv_m2
        interior_rhs1 = cand_A_final_numer
        interior_rhs2 = cand_B_final_numer

        # M^{T}P_{t}^{-1}M (2x2 matrix)
        interior_A = torch.tensor(
            [
                [interior_matrix_a11, interior_matrix_a12],
                [interior_matrix_a12, interior_matrix_a22],
            ],
            dtype=torch.float64,
        )

        # F^{T}
        interior_rhs = torch.tensor(
            [interior_rhs1, interior_rhs2],
            dtype=torch.float64,
        )

        # Check if determinant is non-zero => invertible
        interior_det = float(torch.linalg.det(interior_A).item())
        # interior state variables
        interior_feasible = False
        interior_alpha1 = 0.0
        interior_alpha2 = 0.0
        interior_alpha3 = 0.0
        interior_dual = float("-inf")
        interior_solve_attempted = False
        interior_solve_success = False
        interior_solve_residual = float("inf")
        interior_solve_residual_tol = float("inf")
        interior_rank = 0

        interior_solve_attempted = True
        try:
            # Try to solve the system
            # We solve via least squares that will give us the unique solution if the matrix is invertible, else the smallest norm solution if there are infinite solution
            interior_lstsq_result = torch.linalg.lstsq(interior_A, interior_rhs)
            interior_solution = interior_lstsq_result.solution

            # wandb logging
            interior_solve_residual = float(torch.linalg.norm(interior_A @ interior_solution - interior_rhs).item())
            interior_A_norm = float(torch.linalg.norm(interior_A).item())
            interior_rhs_norm = float(torch.linalg.norm(interior_rhs).item())
            interior_singular_values = torch.linalg.svdvals(interior_A)
            interior_rank_tol = 1e-10 * max(1.0, float(interior_singular_values.max().item()))
            interior_rank = int((interior_singular_values > interior_rank_tol).sum().item())
            interior_alpha1 = float(interior_solution[0].item())
            interior_alpha2 = float(interior_solution[1].item())
            interior_alpha3 = 1.0 - interior_alpha1 - interior_alpha2

            # Get current solution, \lambda
            interior_current_solution = torch.tensor(
                [interior_alpha1, interior_alpha2],
                dtype=torch.float64,
            )
            # Get the residual, r = ||A\lambda - y||
            interior_solve_residual = float(torch.linalg.norm(interior_A @ interior_current_solution - interior_rhs).item())
            # Get ||\lambda||
            interior_current_solution_norm = float(torch.linalg.norm(interior_current_solution).item())
            # Reisdual tolerance score (10^8)(1 + ||r|| + ||A|||*|\lambda||) 
            interior_solve_residual_tol = 1e-8 * (1.0 + interior_rhs_norm + interior_A_norm * interior_current_solution_norm)

            # Accept interior lambda point with finite entries and small residual below the threshold
            interior_solve_success = (
                math.isfinite(interior_alpha1)
                and math.isfinite(interior_alpha2)
                and math.isfinite(interior_alpha3)
                and math.isfinite(interior_solve_residual)
                and interior_solve_residual <= interior_solve_residual_tol
)
            if interior_solve_success:
                # If A is rank deficient but consistent, torch.linalg.lstsq still returns one solution, but needs to check feasibility
                # Check if interior solution falls in the feasible set
                # If feasible, the least-squares solution is already strictly inside the simplex, it should be interpolatable within the triangle
                interior_feasible = (interior_alpha1 > 0.0) and (interior_alpha2 > 0.0) and (interior_alpha3 > 0.0)

                # If rank = 0 (that means A approx 0_{2x2}) 
                # if the residual check passed, rhs is also approximately zero
                # => every alpha solves A alpha = rhs, since any input works
                # => any input is valid and always maps to y = 0 (kernel)
                # so choose the symmetric interior point in this case.
                if (not interior_feasible) and interior_rank == 0:
                    interior_alpha1 = 1.0 / 3.0
                    interior_alpha2 = 1.0 / 3.0
                    interior_alpha3 = 1.0 / 3.0
                    interior_feasible = True
                # If rank = 1 and Solvable => The solution set contains: base solution + input that does to kernel
                elif (not interior_feasible) and interior_rank == 1:
                    # Get the SVD USV^T
                    interior_svd_u, interior_svd_s, interior_svd_vh = torch.linalg.svd(interior_A)

                    # The rows of V^T are the right singular vectors 
                    # singular values ordered from largest to smallest, hence
                    # last row of V^T corresponds to the smallest singular value, since it is rank 1, then it has singular value zero
                    interior_null_vec = interior_svd_vh[-1, :]
                    null_v1 = float(interior_null_vec[0].item())
                    null_v2 = float(interior_null_vec[1].item())
                    null_v3 = -(null_v1 + null_v2)

                    t_lower = float("-inf")
                    t_upper = float("inf")

                    def _update_interval_for_positive_coordinate(current_value, direction_value, lower, upper):
                        if abs(direction_value) <= 1e-14:
                            if current_value > 0.0:
                                return lower, upper, True
                            return lower, upper, False
                        boundary_t = -current_value / direction_value
                        if direction_value > 0.0:
                            lower = max(lower, boundary_t)
                        else:
                            upper = min(upper, boundary_t)
                        return lower, upper, True

                    t_lower, t_upper, ok1 = _update_interval_for_positive_coordinate(interior_alpha1, null_v1, t_lower, t_upper)
                    t_lower, t_upper, ok2 = _update_interval_for_positive_coordinate(interior_alpha2, null_v2, t_lower, t_upper)
                    t_lower, t_upper, ok3 = _update_interval_for_positive_coordinate(interior_alpha3, null_v3, t_lower, t_upper)

                    if ok1 and ok2 and ok3 and (t_lower < t_upper):
                        if t_lower < 0.0 < t_upper:
                            interior_t = 0.0
                        elif math.isfinite(t_lower) and math.isfinite(t_upper):
                            interior_t = 0.5 * (t_lower + t_upper)
                        elif math.isfinite(t_lower):
                            interior_t = t_lower + 1.0
                        elif math.isfinite(t_upper):
                            interior_t = t_upper - 1.0
                        else:
                            interior_t = 0.0

                        interior_alpha1 = interior_alpha1 + interior_t * null_v1
                        interior_alpha2 = interior_alpha2 + interior_t * null_v2
                        interior_alpha3 = 1.0 - interior_alpha1 - interior_alpha2
                        interior_feasible = (interior_alpha1 > 0.0) and (interior_alpha2 > 0.0) and (interior_alpha3 > 0.0)

                # After any singular-system interior recovery, recheck the adjusted candidate against the stationarity system.
                if interior_feasible:
                    interior_adjusted_solution = torch.tensor(
                        [interior_alpha1, interior_alpha2],
                        dtype=torch.float64,
                    )
                    interior_solve_residual = float(torch.linalg.norm(interior_A @ interior_adjusted_solution - interior_rhs).item())
                    interior_adjusted_solution_norm = float(torch.linalg.norm(interior_adjusted_solution).item())
                    interior_solve_residual_tol = 1e-8 * (1.0 + interior_rhs_norm + interior_A_norm * interior_adjusted_solution_norm)
                    interior_solve_success = (
                        math.isfinite(interior_alpha1)
                        and math.isfinite(interior_alpha2)
                        and math.isfinite(interior_alpha3)
                        and math.isfinite(interior_solve_residual)
                        and interior_solve_residual <= interior_solve_residual_tol
                    )
                    interior_feasible = (
                        interior_solve_success
                        and (interior_alpha1 > 0.0)
                        and (interior_alpha2 > 0.0)
                        and (interior_alpha3 > 0.0)
                    )
                if interior_feasible:
                    # Prepare for final update rule
                    interior_lambda_combo_dot_w_t = interior_alpha1 * m1_dot_w_t + interior_alpha2 * m2_dot_w_t
                    # Prepare for the Dual function (just in case of numerical error of the solved interior point)
                    interior_lambda_combo_precond_inv_norm_sq = (interior_alpha1 * interior_alpha1 * denom_m1_precond_inv_m1) + (2.0 * interior_alpha1 * interior_alpha2 * inner_m1_precond_inv_m2) + (interior_alpha2 * interior_alpha2 * denom_m2_precond_inv_m2)
                    interior_dual = dual_function(interior_alpha1, interior_alpha2, interior_alpha3, mu_model, lr_safe, w_t_precond_w_t, interior_lambda_combo_dot_w_t, interior_lambda_combo_precond_inv_norm_sq, b1, b2, fstar)

        # Step 2.2
        # If we reach here => matrix solve failed numerically
        # fall back to the projected boundary candidates
        except RuntimeError:
            interior_feasible = False
            interior_solve_success = False
            interior_dual = float("-inf")

        # Initialize candidate list with the projected boundary candidates
        # these will cover our 3 simplex edges: A = (plane 1, fstar), B = (plane 2, fstar), C = (plane 1, plane 2)
        candidate_records = [
            (cand_A_dual, cand_A_alpha1, cand_A_alpha2, cand_A_alpha3, cand_A_alpha1_unc, cand_A_final_denom, cand_A_corrected_denom, "A"),
            (cand_B_dual, cand_B_alpha1, cand_B_alpha2, cand_B_alpha3, cand_B_alpha2_unc, cand_B_final_denom, cand_B_corrected_denom, "B"),
            (cand_C_dual, cand_C_alpha1, cand_C_alpha2, cand_C_alpha3, cand_C_alpha1_unc, cand_C_final_denom, cand_C_corrected_denom, "C"),
        ]
        if interior_feasible:
            candidate_records.append((interior_dual, interior_alpha1, interior_alpha2, interior_alpha3, interior_alpha1, interior_det, max(abs(interior_det), eps) + alpha_denom_correction, "interior"))

        # Argmax of the dual function and keep only the candidate with the max dual value
        # Max on the x[0] --- dual value as criteria
        best_dual, alpha1, alpha2, alpha3, alpha1_unc, final_denom, corrected_denom, best_candidate_name = max(candidate_records, key=lambda x: x[0])

        # Final optimizer update rule state variables with explicit casting
        alpha1 = float(alpha1)
        alpha2 = float(alpha2)
        alpha3 = float(alpha3)
        alpha1_unc = float(alpha1_unc)
        final_denom = float(final_denom)
        corrected_denom = float(corrected_denom)
        best_dual = float(best_dual)

        if best_candidate_name == "A":
            selected_alpha_unc_from_final_numer_over_corrected_denom = cand_A_final_numer / cand_A_corrected_denom if cand_A_corrected_denom > 0.0 else 0.0
        elif best_candidate_name == "B":
            selected_alpha_unc_from_final_numer_over_corrected_denom = cand_B_final_numer / cand_B_corrected_denom if cand_B_corrected_denom > 0.0 else 0.0
        elif best_candidate_name == "C":
            selected_alpha_unc_from_final_numer_over_corrected_denom = cand_C_final_numer / cand_C_corrected_denom if cand_C_corrected_denom > 0.0 else 0.0
        else:
            selected_alpha_unc_from_final_numer_over_corrected_denom = alpha1_unc

        # This old alpha1_unc_from_final_numer_over_corrected_denom name is kept below for backward-compatible logging
        alpha1_unc_from_final_numer_over_corrected_denom = selected_alpha_unc_from_final_numer_over_corrected_denom
        eps_cos = 1e-12

        # wandb logging
        abs_alpha1_unc_minus_alpha1_unc_from_final_numer_over_corrected_denom = abs(alpha1_unc - selected_alpha_unc_from_final_numer_over_corrected_denom)
        indicator_alpha1_unc_less_than_0 = float(alpha1_unc < 0.0)
        indicator_alpha1_unc_greater_than_1 = float(alpha1_unc > 1.0)
        indicator_final_denom_raw_less_than_eps = float(final_denom < eps)
        cos_sim_m_t1_m_t2 = logged_mom_vec1_vec2_dot_prod / ((logged_mom_vec_1_squared_norm**0.5) * (logged_mom_vec_2_squared_norm**0.5) + eps_cos)
        cos_sim_g_t_m_t1 = logged_grad_m1_dot_prod / ((logged_grad_squared_norm**0.5) * (logged_mom_vec_1_squared_norm**0.5) + eps_cos)
        cos_sim_g_t_m_t2 = logged_grad_m2_dot_prod / ((logged_grad_squared_norm**0.5) * (logged_mom_vec_2_squared_norm**0.5) + eps_cos)


        ####################################################
        # Temporary: sliding-window min-max normalization for alpha1_unc
        # alpha_window_size = 500

        # if "tp_alpha1_unc_window" not in g:
        #     g["tp_alpha1_unc_window"] = []

        # g["tp_alpha1_unc_window"].append(float(alpha1_unc))

        # if len(g["tp_alpha1_unc_window"]) > alpha_window_size:
        #     g["tp_alpha1_unc_window"] = g["tp_alpha1_unc_window"][-alpha_window_size:]

        # alpha1_window_min = min(g["tp_alpha1_unc_window"])
        # alpha1_window_max = max(g["tp_alpha1_unc_window"])
        # alpha1_window_range = alpha1_window_max - alpha1_window_min

        # if alpha1_window_range <= eps:
        #     alpha1_norm_01 = 0.5
        # else:
        #     alpha1_norm_01 = (float(alpha1_unc) - alpha1_window_min) / alpha1_window_range
        #     alpha1_norm_01 = min(1.0, max(0.0, alpha1_norm_01))

        # alpha1 = alpha_min + (alpha_max - alpha_min) * alpha1_norm_01

        # if clip_alpha:
        #     alpha1 = min(alpha_max, max(alpha_min, alpha1))
        ####################################################

        ####################################################
        # Sanity check only. Comment out once done: forces alpha1 and alpha2 to match ademamix's for consistency check 
        # schedule_step = tp_step
        # bias_correction1_for_ademamix_match = 1.0 - (beta_s ** schedule_step)
        # bias_correction1_for_ademamix_match = max(bias_correction1_for_ademamix_match, eps)

        # alpha1 = 1.0 / bias_correction1_for_ademamix_match
        # alpha2 = linear_warmup_scheduler(
        #         schedule_step,
        #         alpha_end=8.0,
        #         alpha_start=0,
        #         warmup=16000,
        #         )
        ####################################################
        self.last_alpha1 = alpha1

        # wandb logging
        if log_dict is not None:
            log_dict["three_plane_momo/network/alpha1"] = float(alpha1)
            log_dict["three_plane_momo/network/alpha2"] = float(alpha2)
            log_dict["three_plane_momo/network/alpha3"] = float(alpha3)
            log_dict["three_plane_momo/network/best_candidate"] = best_candidate_name
            log_dict["three_plane_momo/network/best_dual"] = float(best_dual)
            log_dict["three_plane_momo/network/candidate_A_dual"] = float(cand_A_dual)
            log_dict["three_plane_momo/network/candidate_B_dual"] = float(cand_B_dual)
            log_dict["three_plane_momo/network/candidate_C_dual"] = float(cand_C_dual)
            log_dict["three_plane_momo/network/interior_dual"] = float(interior_dual)
            log_dict["three_plane_momo/network/interior_feasible"] = float(interior_feasible)
            log_dict["three_plane_momo/network/interior_solve_attempted"] = float(interior_solve_attempted)
            log_dict["three_plane_momo/network/interior_solve_success"] = float(interior_solve_success)
            log_dict["three_plane_momo/network/interior_solve_residual"] = float(interior_solve_residual)
            log_dict["three_plane_momo/network/interior_solve_residual_tol"] = float(interior_solve_residual_tol)
            log_dict["three_plane_momo/network/interior_rank"] = float(interior_rank)
            log_dict["three_plane_momo/network/interior_det"] = float(interior_det)
            log_dict["three_plane_momo/network/final_denom"] = float(final_denom)
            log_dict["three_plane_momo/network/alpha1_unclipped"] = float(alpha1_unc)
            log_dict["three_plane_momo/network/||m_t^{(1)}||_{2}^{2}"] = float(logged_mom_vec_1_squared_norm)
            log_dict["three_plane_momo/network/||m_t^{(2)}||_{2}^{2}"] = float(logged_mom_vec_2_squared_norm)
            log_dict["three_plane_momo/network/<m_t^(1), m_t^(2)>"] = float(logged_mom_vec1_vec2_dot_prod)
            log_dict["three_plane_momo/network/<g_t, m_t^(1)>"] = float(logged_grad_m1_dot_prod)
            log_dict["three_plane_momo/network/<g_t, m_t^(2)>"] = float(logged_grad_m2_dot_prod)
            log_dict["three_plane_momo/network/corrected_denom"] = float(corrected_denom)
            log_dict["three_plane_momo/network/b1_minus_b2"] = float(b1 - b2)
            log_dict["three_plane_momo/network/b1_raw_minus_b2_raw_before_reliability_adjustment"] = float(b1_raw - b2_raw)
            log_dict["three_plane_momo/network/reliability_adjusted_b_gap_minus_raw_b_gap"] = float((b1 - b2) - (b1_raw - b2_raw))
            log_dict["three_plane_momo/network/candidate_C_final_numer"] = float(cand_C_final_numer)  
            log_dict["three_plane_momo/network/candidate_C_numer_term_A_num_fac_times_b_gap"] = float(cand_C_numer_term_A_num_fac_times_b_gap)  
            log_dict["three_plane_momo/network/candidate_C_numer_term_B_mu_times_m1_minus_m2_dot_wt"] = float(cand_C_numer_term_B_mu_times_m1_minus_m2_dot_wt)
            log_dict["three_plane_momo/network/candidate_C_numer_term_C_m1_minus_m2_dot_Pinv_m2"] = float(cand_C_numer_term_C_m1_minus_m2_dot_Pinv_m2)
            log_dict["three_plane_momo/network/selected_alpha_unc_from_final_numer_over_corrected_denom"] = float(selected_alpha_unc_from_final_numer_over_corrected_denom)
            log_dict["three_plane_momo/network/alpha1_unc_from_final_numer_over_corrected_denom"] = float(alpha1_unc_from_final_numer_over_corrected_denom)
            log_dict["three_plane_momo/network/|alpha1_unc - alpha1_unc_from_final_numer_over_corrected_denom|"] = float(abs_alpha1_unc_minus_alpha1_unc_from_final_numer_over_corrected_denom)
            log_dict["three_plane_momo/network/indicator_alpha1_unc_less_than_0"] = float(indicator_alpha1_unc_less_than_0)  
            log_dict["three_plane_momo/network/indicator_alpha1_unc_greater_than_1"] = float(indicator_alpha1_unc_greater_than_1)  
            log_dict["three_plane_momo/network/indicator_final_denom_raw_less_than_eps"] = float(indicator_final_denom_raw_less_than_eps)
            log_dict["three_plane_momo/network/||g_t||_{2}^{2}"] = float(logged_grad_squared_norm)  
            log_dict["three_plane_momo/network/||m_t^{(1)}-m_t^{(2)}||_{2}^{2}"] = float(logged_m1_minus_m2_squared_norm)  
            log_dict["three_plane_momo/network/cos_sim(m_t^(1), m_t^(2))"] = float(cos_sim_m_t1_m_t2)  
            log_dict["three_plane_momo/network/cos_sim(g_t, m_t^(1))"] = float(cos_sim_g_t_m_t1)  
            log_dict["three_plane_momo/network/cos_sim(g_t, m_t^(2))"] = float(cos_sim_g_t_m_t2)  
            if precond == "adam":  
                pinv_diag_entry_mean = pinv_diag_entry_sum / max(1, pinv_diag_entry_count)
                log_dict["three_plane_momo/network/pinv_diag_entry_mean"] = float(pinv_diag_entry_mean)
                log_dict["three_plane_momo/network/pinv_diag_entry_max"] = float(pinv_diag_entry_max)
                log_dict["three_plane_momo/network/||P_t^{-1} m_t^(2)||_{2}^{2}"] = float(pinv_m2_squared_norm)
                log_dict["three_plane_momo/network/||P_t^{-1} (m_t^(1)-m_t^(2))||_{2}^{2}"] = float(pinv_m1_minus_m2_squared_norm)  

        m_t_1_dot_w_t_plus_1_minus_w_t = 0.0
        m_t_2_dot_w_t_plus_1_minus_w_t = 0.0

        # Final optimizer update rule, w_{t+1}
        # w_{t+1} = (1 / (1 + \eta*\mu)) w_t - (\eta / (1 + \eta*\mu)) P_t^{-1} (\lambda_1 m_t^{(1)} + \lambda_2 m_t^{(2)})
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                m1 = self.state[p]["m1"]
                m2 = self.state[p]["m2"]

                # m1_for_update = m1
                ####################################################
                # Temporary: Potentially consider bias correction to momentum 1.
                # m1_for_update = m1 / max(1.0 - beta_s ** tp_step, eps)
                m1_for_update = m1
                ####################################################

                grad = p.grad.detach()
                if grad.dtype != torch.float32:
                    grad = grad.float()

                m2_nesterov = m2 / max(self.state[p]["m2_gs"], eps)

                # projected dual mixture direction:
                # alpha1*m1 + alpha2*m2, with remaining mass alpha3 assigned to the zero-slope fstar plane
                # mom_vec_cvx_combo = alpha1 * m1_for_update + alpha2 * m2
                mom_vec_cvx_combo = alpha1 * m1_for_update + alpha2 * m2_nesterov

                # apply preconditioner as Pinv
                if precond == "adam":
                    # Setup preconditioner 
                    v_t = self.state[p]["v_t"]
                    v_t_hat = v_t / bias_correction_denom
                    precond_t_inv_flattened = v_t_hat.sqrt().add(eps_precond).reciprocal()
                    # Apply step transformed by preconditioner
                    step_dir = mom_vec_cvx_combo * precond_t_inv_flattened
                else:
                    # No transformation if we use the identity
                    step_dir = mom_vec_cvx_combo

                # DECOUPLED
                # Decoupled weight decay (AdamW): apply decay directly to the parameters!
                # then apply the preconditioned momentum step w/ no proximal shrinkage factor
                if decoupled_wd:
                    # Current iterate that will used in the update rule
                    p_old = p.detach().float()

                    # wandb logging
                    w_t_plus_1_minus_w_t = (-lr * mu) * p_old - lr * step_dir.detach().float()
                    m_t_1_dot_w_t_plus_1_minus_w_t += torch.dot(m1_for_update.detach().float().flatten(), w_t_plus_1_minus_w_t.flatten()).item()
                    # m_t_2_dot_w_t_plus_1_minus_w_t += torch.dot(m2.detach().float().flatten(), w_t_plus_1_minus_w_t.flatten()).item()
                    m_t_2_dot_w_t_plus_1_minus_w_t += torch.dot(m2_nesterov.detach().float().flatten(), w_t_plus_1_minus_w_t.flatten()).item()

                    # Decoupled form after distributing the learning rate
                    # Final optimizer update rule minus current iterate
                    # w_{t+1} = (1 - lr*mu) w_t - lr*step_dir  <=>  w_{t+1} - w_t = -lr*mu*w_t - lr*step_dir
                    if mu != 0.0:
                        p.add_(p, alpha=-lr * mu)
                    p.add_(step_dir.to(p.dtype), alpha=-lr)
                    continue

                # COUPLED
                # Current iterate that will used in the update rule
                p_old = p.detach().float()
                # prepare coupled shrink factor
                denom_for_dw = (1.0 + lr * mu)
                shrink = (1.0 / denom_for_dw)
                scale = (lr / denom_for_dw)

                # wandb logging
                w_t_plus_1_minus_w_t = (shrink - 1.0) * p_old - scale * step_dir.detach().float()
                m_t_1_dot_w_t_plus_1_minus_w_t += torch.dot(m1_for_update.detach().float().flatten(), w_t_plus_1_minus_w_t.flatten()).item()
                # m_t_2_dot_w_t_plus_1_minus_w_t += torch.dot(m2.detach().float().flatten(), w_t_plus_1_minus_w_t.flatten()).item()
                m_t_2_dot_w_t_plus_1_minus_w_t += torch.dot(m2_nesterov.detach().float().flatten(), w_t_plus_1_minus_w_t.flatten()).item()

                # The new step for the optimizer: proximal coupled weight decay step
                # w_{t+1} = (1/(1+ (eta * mu))) w_t - (η/(1+(eta * mu))) Pinv( alpha1*m1 + alpha2*m2 )
                denom = (1.0 + lr * mu)
                p.mul_(1.0 / denom)
                p.add_(step_dir.to(p.dtype), alpha=-lr / denom)

        # wandb logging
        ell_t_1_of_w_t_plus_1 = b1 + m_t_1_dot_w_t_plus_1_minus_w_t
        ell_t_2_of_w_t_plus_1 = b2 + m_t_2_dot_w_t_plus_1_minus_w_t
        ell_t_1_of_w_t_plus_1_minus_ell_t_2_of_w_t_plus_1 = ell_t_1_of_w_t_plus_1 - ell_t_2_of_w_t_plus_1 

        if log_dict is not None:  
            log_dict["three_plane_momo/network/ell_t^(1)(w_{t+1})"] = float(ell_t_1_of_w_t_plus_1)
            log_dict["three_plane_momo/network/ell_t^(2)(w_{t+1})"] = float(ell_t_2_of_w_t_plus_1)
            log_dict["three_plane_momo/network/ell_t^(1)(w_{t+1})-ell_t^(2)(w_{t+1})"] = float(ell_t_1_of_w_t_plus_1_minus_ell_t_2_of_w_t_plus_1)
            log_dict["three_plane_momo/network/<m_t^(1), w_{t+1}-w_t>"] = float(m_t_1_dot_w_t_plus_1_minus_w_t)
            log_dict["three_plane_momo/network/<m_t^(2), w_{t+1}-w_t>"] = float(m_t_2_dot_w_t_plus_1_minus_w_t)

        # write back optimizer-global state to all param groups for run resume correctness
        # next_step = tp_step + 1
        next_step = tp_step # To match AdEMAMix which uses pre-increment timeline

        for _grp in self.param_groups:
            _grp["tp_barf1"] = barf1
            _grp["tp_barf2"] = barf2
            _grp["tp_gamma1"] = gamma1
            _grp["tp_gamma2"] = gamma2
            _grp["tp_barf2_gs"] = barf2_gs
            _grp["tp_gamma2_gs"] = gamma2_gs
            _grp["tp_reliability_ema1"] = reliability_ema1
            _grp["tp_reliability_ema2"] = reliability_ema2
            # _grp["tp_prev_b1"] = prev_b1
            # _grp["tp_prev_b2"] = prev_b2
            _grp["tp_prev_intercept1"] = prev_intercept1
            _grp["tp_prev_intercept2"] = prev_intercept2
            _grp["tp_reliability_initialized"] = reliability_initialized
            _grp["tp_step"] = next_step

        return None

    @torch.no_grad()
    def step(self, loss: Optional[Union[float, torch.Tensor]] = None, log_dict: Optional[Dict] = None):
        alpha_scope = self.param_groups[0]["alpha_scope"]
        if alpha_scope == "network":
            return self._step_network(loss=loss, log_dict=log_dict)
        raise NotImplementedError("This pasted GenThreePlaneMoMo block currently contains the network-wide implementation only.")

