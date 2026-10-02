"""
Cross-feeding consumer-resource model for a 2-species + 3-compound microbial community.

State variables:
- R: limiting resource (with logistic growth, autoregenerated/externally supplied)
- A: species A biomass (consumes R, produces M)
- M: intermediate metabolite (produced by A, consumed by B)
- B: species B biomass (consumes M, produces P) 
- P: product of interest (produced by B at a cost)

when considering evolutionary instability:
- Bp: biomass of species B that are producers
- Bc: biomass of species B that are not producers ("cheaters")

It can illustrate three qualitatively different kinds of instability and interventions to fix them:
- evolutionary: producing P is costly for B. A mutation that skips production can have advantage and invade the system
- ecological: paradox of enrichment (changes in resource availability can lead to big oscillations)
- environmental: increased death rate for a limited amount of time can lead to collapse

All models are variations on the same skeleton with some added feedback.
"""

import numpy as np
from scipy.integrate import solve_ivp

def monod(x, K):
    return x / (K + x)

class Params:
    def __init__(self, **kwargs):
        # R -> A
        self.rR = 1.0       # renewal rate of R
        self.Rcap = 2.0     # max amount of R
        self.qR = 1.3       # resource consumed per unit A growth
        self.muA_max = 1.0  # max growth rate of A
        self.KA = 0.5       # half-saturation of A on R
        self.dA = 0.15      # death rate of A

        # A -> M 
        self.pM = 1.0       # production rate of M (M produced per unit A growth)
        self.dM = 0.05      # M degradation

        # M -> B
        self.muB_max = 1.0  # max growth rate of A
        self.KB = 0.5       # half-saturation of B on M
        self.dB = 0.15      # death rate of B
        self.qM = 1.0       # M consumed per unit B growth
        self.Ballee = 0.0   # Allee threshold: B below this goes extinct

        # B -> P 
        self.pP = 1.0       # production rate of P (P produced per unit B growth)
        self.dP = 0.02      # P degradation
        self.cost = 0.35    # fraction of growth diverted to making P (f)

        # feedbacks (default off)
        self.selfreg = 0.0       # B self regulation negative feedback (ecological fix)
        self.coupled_production = False  # obligate P production tied to growth (evolutionary fix)
        self.homeostasis = 0 #intensity of compensation (h)
        self.costM = 0

        for k, v in kwargs.items():
            setattr(self, k, v)

def base_model(t, y, p: Params, pulse=None):
    #base 5-state model (for ecological & environmental)
    R, A, M, B, P = np.maximum(y, 0)

    muA = p.muA_max * monod(R, p.KA)
    muB = p.muB_max * monod(M, p.KB)

    dRdt = p.rR * R * (1 - R / p.Rcap) - p.qR * muA * A

    reg = p.homeostasis / (M+1)
    extraM = muA * A * reg
    dAdt = muA * A - p.costM * extraM - p.dA * A
    dMdt = muA * A * (1 + reg) - p.qM * muB * B - p.dM * M

    allee = B / (B + p.Ballee) if p.Ballee > 0 else 1.0
    dBdt = muB * (1 - p.cost) * B * allee - p.dB * B - p.selfreg * B**2
    dPdt = p.pP * muB * B * p.cost - p.dP * P

    dydt = np.array([dRdt, dAdt, dMdt, dBdt, dPdt])

    return dydt


def evo_model(t, y, p: Params):
    #6-state model: R, A, M, Bp (producer), Bc (cheater), P (used for evolutionary and all at once)
    R, A, M, Bp, Bc, P = np.maximum(y, 0)
    B = Bp + Bc

    muA = p.muA_max * monod(R, p.KA)
    dRdt = p.rR * R * (1 - R / p.Rcap) - p.qR * muA * A

    reg = p.homeostasis / (M+1)
    extraM = muA * A * reg
    dAdt = muA * A - p.costM * extraM - p.dA * A
    
    muB = p.muB_max * monod(M, p.KB)
    allee = B / (B + p.Ballee) if p.Ballee > 0 else 1.0

    if p.coupled_production:
        # skipping production also skips growth, so there is no fitness advantage to cheating
        growth_Bp = muB * (1 - p.cost) * allee
        growth_Bc = muB * (1 - p.cost) * allee   # same penalty, obligate
    else:
        # decoupled: producers pay a cost that cheaters avoid
        growth_Bp = muB * (1 - p.cost) * allee
        growth_Bc = muB * allee

    dMdt = muA * A * (1 + reg) - p.qM * muB * B - p.dM * M
    dBpdt = growth_Bp * Bp - p.dB * Bp - p.selfreg * Bp * B
    dBcdt = growth_Bc * Bc - p.dB * Bc - p.selfreg * Bc * B
    dPdt = (p.pP * muB * Bp * p.cost) - p.dP * P

    return np.array([dRdt, dAdt, dMdt, dBpdt, dBcdt, dPdt])

def simulate(model, y0, t_span, p, n=400, **kwargs):
    t_eval = np.linspace(*t_span, n)
    sol = solve_ivp(model, t_span, y0, args=(p,), t_eval=t_eval, method="LSODA", rtol=1e-8, atol=1e-10, **kwargs)
    return sol

def simulate_with_extinction(model, y0, t_span, p, n=400, threshold=0.001):
    #Used for the case of only ecological instability (since Ballee=0, and we want the population to go exctinct if it reaches 0, even if mathematically it bounces back)
    t_eval = np.linspace(*t_span, n)
    sol = solve_ivp(model, t_span, y0, args=(p,), t_eval=t_eval, method="LSODA", rtol=1e-8, atol=1e-10,dense_output=True)
    
    t = sol.t
    y = sol.y.copy()
    
    t_dense = np.linspace(t_span[0], t_span[1], 100000)
    B_dense = sol.sol(t_dense)[3] 
    
    extinct_indices = np.where(B_dense <= threshold)[0]
    
    if len(extinct_indices) > 0:
        t_extinct = t_dense[extinct_indices[0]]
        # Set to zero all points in t_eval from t_extinct onward
        first_extinct_idx = np.searchsorted(t, t_extinct)
        y[3, first_extinct_idx:] = 0.0
    return t, y

def simulate_rate_stress(model, y0, t_total, p, stress_start, stress_end, rate_name, stress_value, n=1500):
    #environmental instability, increase death rate of B for a fixed amount of time
    baseline_value = getattr(p, rate_name)

    t1 = np.linspace(0, stress_start, max(int(n * stress_start / t_total), 2))
    sol1 = solve_ivp(model, (0, stress_start), y0, args=(p,), t_eval=t1, method="LSODA", rtol=1e-8, atol=1e-10)

    setattr(p, rate_name, stress_value)
    t2 = np.linspace(stress_start, stress_end, max(int(n * (stress_end - stress_start) / t_total), 2))
    sol2 = solve_ivp(model, (stress_start, stress_end), sol1.y[:, -1], args=(p,), t_eval=t2, method="LSODA", rtol=1e-8, atol=1e-10)

    setattr(p, rate_name, baseline_value)
    t3 = np.linspace(stress_end, t_total, max(int(n * (t_total - stress_end) / t_total), 2))
    sol3 = solve_ivp(model, (stress_end, t_total), sol2.y[:, -1], args=(p,), t_eval=t3, method="LSODA", rtol=1e-8, atol=1e-10)

    t = np.concatenate([sol1.t, sol2.t, sol3.t])
    y = np.concatenate([sol1.y, sol2.y, sol3.y], axis=1)
    return t, y

def rate_stress_and_switch(model, y0, t_total, p_pre, p_post, stress_start, stress_end, rate_name, stress_value, n=1500):
    #to see whether the environmental fix rescues the population during recovery or only because of previous buffering 
    #activate fix only when B has experienced an increased death rate, and not continuously
    baseline_value = getattr(p_pre, rate_name)

    t1 = np.linspace(0, stress_start, max(int(n*stress_start/t_total), 2))
    sol1 = solve_ivp(model, (0, stress_start), y0, args=(p_pre,), t_eval=t1, method="LSODA", rtol=1e-8, atol=1e-10)

    setattr(p_pre, rate_name, stress_value)
    t2 = np.linspace(stress_start, stress_end, max(int(n*(stress_end-stress_start)/t_total), 2))
    sol2 = solve_ivp(model, (stress_start, stress_end), sol1.y[:, -1], args=(p_pre,), t_eval=t2, method="LSODA", rtol=1e-8, atol=1e-10)
    setattr(p_pre, rate_name, baseline_value)   

    t3 = np.linspace(stress_end, t_total, max(int(n*(t_total-stress_end)/t_total), 2))
    sol3 = solve_ivp(model, (stress_end, t_total), sol2.y[:, -1], args=(p_post,), t_eval=t3, method="LSODA", rtol=1e-8, atol=1e-10)

    t = np.concatenate([sol1.t, sol2.t, sol3.t])
    y = np.concatenate([sol1.y, sol2.y, sol3.y], axis=1)
    return t, y
