### A Pluto.jl notebook ###
# v0.19.25

using Markdown
using InteractiveUtils

# ╔═╡ c6e50d2a-df8e-11ed-2859-b1d221d06c6d
begin
	using Distributions, Plots, Random
	using LinearAlgebra
	# using StatsBase
	using StatsFuns
	using SpecialFunctions
	using PlutoUI
end

# ╔═╡ bfe253c8-3ab4-4629-9a6b-423a64d8fe32
TableOfContents()

# ╔═╡ 5e08403b-2475-45d2-950a-ab4cf2aada5a
md"""
# VBEM Derivation for DLM
"""

# ╔═╡ 7971c689-4438-47ff-a896-6e61fc62d10b
md"""
A linear Gaussian state space model (LDS / DLM) can be defined as follows:

State Transition Model (**State Equation**): 

$$x_t = Ax_{t-1} + w_t, \  w_t \sim \mathcal N(0, Q)$$ 

where $A (K \times K)$ is the state transition matrix, $Q (K \times K)$

Observation Model (**Observation Equation**): 

$$y_t = Cx_t + v_t, \ v_t \sim \mathcal N(0, R)$$ 
	
where $C (D \times K)$ is the observation/emission matrix, $R (D \times D)$.

We can fix $Q$ to the identity matrix $I$ and denote the set of DLM parameters as:

$$\mathbf{θ} = (A, C, R)$$

Suppose $R$ is defined through a precision vector $\mathbf{ρ}$, such that $diag(\mathbf{ρ})$ = $R^{-1}$, and

$p(\mathbf{ρ}|a, b) = \prod_{s=1}^D \mathcal Gam(ρ_s| a, b)$

The joint probability of hidden and observed sequences are:

$p(x_{1:T}, y_{1:T}) = p(x_1)p(y_1|x_1) \prod_{t=2}^T p(x_t|x_{t-1}) p(y_t|x_t)$

It should be noted that, **SSM and HMM have closely related basic assumptions and recursive computations**. 

"""

# ╔═╡ 9000f45e-0f28-4b72-ba19-34dd944b274b
md"""
## Prior specification

### Rows of the state transition matrix:

For each row j $(j = 1, 2,... K)$ of A, we can define a prior as follows:

$a_j \sim \mathcal N(\mathbf{0}, \ diag(\mathbf{α})^{-1})$

Here, $\mathbf{α}$ is a K-dimensional precision hyperparameter vector, and $diag(\mathbf{α})$ is a K x K diagonal matrix formed from the precision vector $\mathbf{α}$. This prior implies that the transition dynamics are regularized towards zero, and the precision hyperparameters $\mathbf{α}$ control the strength of the regularization.

### Rows of the observation/emission matrix:

For each row s $(s = 1, 2 ... D)$ of C, we can define a prior as follows:

$c_s \sim \mathcal N(\mathbf{0}, \ diag(ρ_s \mathbf{γ})^{-1})$

Here, $ρ_s$ is a scalar gamma-distributed random variable with hyperparameters a and b:

$ρ_s \sim \mathcal Gam(a, b)$

The vector $\mathbf{γ}$ is a K-dimensional precision hyperparameter vector, and $diag(ρ_s \mathbf{γ})$ is a K x K diagonal matrix formed by element-wise multiplication of $ρ_s$ and $$\mathbf{γ}$$.

This prior on the rows of C assumes that the mapping between hidden states and observations is regularized towards zero, with the strength of the regularization controlled by the precision parameters $ρ_s$ and $$\mathbf{γ}$$.

### Hidden State Sequence

Let $p(x_0 | μ_0, Σ_0) \sim \mathcal N(x_0 | μ_0, Σ_0)$ be an auxiliary hidden state at time $t=0$, we can express $p(x_1)$ as:

$\begin{align}
p(x_1 |μ_0, Σ_0, \mathbf{θ}) &= \int dx_0 \ p(x_0 | μ_0, Σ_0) p(x_1|x_0,\mathbf{θ})\\
&= \mathcal N(x_1|Aμ_0, A^TΣ_0A + Q)
\end{align}$

where $Σ_0$ is a multiple of the identity matrix.
"""

# ╔═╡ 477f8dbf-6797-4e7c-91be-31387f82ece7
md"""
## Variational Treatment/Approximation

Full joint probability:

$p(A, C, \mathbf{ρ}, x_{0:T}, y_{1:T}) = p(A|\mathbf{α}) p(\mathbf{ρ}|a, b) p(C|\mathbf{ρ}, \mathbf{γ}) p(x_0 | μ_0, Σ_0) \prod_{t=1}^T p(x_t|x_{t-1}, A) p(y_t|x_t, C, \mathbf{ρ})$

Log marginal likelihood:

$\begin{align}
\ln p(y_{1:T}) &= \ln \int dA \ dC \ d\mathbf{ρ} \ dx_{0:T} \ p(A, C, \mathbf{ρ}, x_{0:T}, y_{1:T}) \\

&\geq \int dA \ dC \ d\mathbf{ρ} \ dx_{0:T} \ q(A, C, \mathbf{ρ}, x_{0:T}) \ln \frac{p(A, C, \mathbf{ρ}, x_{0:T}, y_{1:T})}{q(A, C, \mathbf{ρ}, x_{0:T})}\\

&= \mathcal F
\end{align}$
"""

# ╔═╡ c5fc190c-63a1-4d94-ac29-f56d0556452f
md"""
Choose $q(...)$ such that $\mathcal F$ is of tractable form:

$q(A, C, \mathbf{ρ}, x_{0:T}) = q(A) q(C, \mathbf{ρ}) q(x_{0:T})$

$q(C, \mathbf{ρ}) = q( \mathbf{ρ}) q (C| \mathbf{ρ})$

From the prior specification:

$q(A, C, \mathbf{ρ}) = \prod_{j=1}^K q(a_j) \prod_{s=1}^D q(ρ_s) q(c_s|ρ_s)$
"""

# ╔═╡ 74482089-10fe-446b-b3f6-dc1b81b1a424
md"""
## VBM Step: θ distributions

Given expected complete data sufficient statistics: $W_A, S_A, W_C, S_C$, all are matrices obtained from VBE Step.

c.f. Beale Chap 5 5.36

$q(A) = \prod_{j=1}^K \mathcal N(a_j|Λ_A^{-1} S_{A, j}, \ Λ_A^{-1})$

where $Λ_A = diag(\mathbf{α}) + W_A$, $S_{A,j}$ is the jth column of matrix $S_A$

cf. Beale Chap 5 5.42, 5.44

$q(\mathbf{ρ}) = \prod_{s=1}^D \mathcal Ga(ρ_s| a + \frac{T}{2}, b + \frac{G_{ss}}{2})$

$q(C|\mathbf{ρ}) = \prod_{s=1}^D \mathcal N(c_s|Λ_C^{-1} S_{C, s}, \ ρ_s^{-1} Λ_C^{-1})$

where $Λ_C = diag(\mathbf{γ}) + W_C$, $S_{C,s}$ is the sth column of matrix $S_C$,

$G = \sum_{t=1}^T y_t y_t^T - S_C^T Λ_C^{-1} S_C$

After integrating out the precision vector $\mathbf{ρ}$, full marginal of C should be Student-t distributed.

### Natural parameterisation

$ϕ(\mathbf{θ}) = ϕ(A, C, R) = \{A, \ A^TA, \ C, \ R^{-1},  C^TR^{-1}C, \ R^{-1}C \}$

The VBM Step computes the expected natural parameter: $\langle ϕ(\mathbf{θ}) \rangle_{q_θ(θ)}$
"""

# ╔═╡ 9381e183-5b4e-489f-a109-4e606212986e
# hidden state sufficient statistics
struct HSS
    W_A::Matrix{Float64}
    S_A::Matrix{Float64}
    W_C::Matrix{Float64}
    S_C::Matrix{Float64}

	# can be extended to incorporate driving input/ HSSMs
end

# ╔═╡ 45b0255d-72fd-4fa7-916a-4f73e730f4d5
# expected natural parameters
struct Exp_ϕ
	A
	AᵀA
	C
	R⁻¹
	CᵀR⁻¹C
	R⁻¹C
	CᵀR⁻¹

	# TO-DO: ELBO computation and Convergence check
	"""
	log_det_R⁻¹
	"""
end

# ╔═╡ 6c1a13d3-9089-4b54-a0e5-a02fb5fdf4a1
# hyper-prior parameters
struct HPP
    α::Vector{Float64} # precision vector for transition A
    γ::Vector{Float64}  # precision vector for emission C
    a::Float64 # gamma rate of ρ
    b::Float64 # gamma inverse scale of ρ
    μ_0::Vector{Float64} # auxiliary hidden state mean
    Σ_0::Matrix{Float64} # auxiliary hidden state co-variance
end

# ╔═╡ a8b50581-e5f3-449e-803e-ab31e6e0b812
# input: data, hyperprior, and e-step suff stats
# infer parameter posterior q_θ(θ)
function vb_m(ys, hps::HPP, ss::HSS)
	D, T = size(ys)

	W_A = ss.W_A
	S_A = ss.S_A
	W_C = ss.W_C
	S_C = ss.S_C
	α = hps.α
	γ = hps.γ
	a = hps.a
	b = hps.b

	K = length(α)
	
	# q(A), q(ρ), q(C|ρ)
    Σ_A = inv(diagm(α) + W_A)
	Σ_C = inv(diagm(γ) + W_C)
	
	G = sum(ys[:, t] * ys[:, t]' for t in 1:T) - S_C' * Σ_C * S_C
	a_ = a + 0.5 * T
	a_s = a_ * ones(D)
    b_s = [b + 0.5 * G[i, i] for i in 1:D]

	q_ρ = Gamma.(a_s, 1 ./ b_s)
	ρ̄ = mean.(q_ρ)
	
	# Exp_ϕ 
	Exp_A = S_A'*Σ_A
	Exp_AᵀA = Exp_A'*Exp_A + K*Σ_A
	Exp_C = S_C'*Σ_C
	Exp_R⁻¹ = diagm(ρ̄)
	
	Exp_CᵀR⁻¹C = Exp_C'*Exp_R⁻¹*Exp_C + D*Σ_C
	Exp_R⁻¹C = Exp_R⁻¹*Exp_C
	Exp_CᵀR⁻¹ = Exp_C'*Exp_R⁻¹

	# update hyperparameter (after m-step)
	α_n = [K/((K*Σ_A + Σ_A*S_A*S_A'*Σ_A)[j, j]) for j in 1:K]
	γ_n = [D/((D*Σ_C + Σ_C*S_C*Exp_R⁻¹*S_C'*Σ_C)[j, j]) for j in 1:K]

	# for updating gamma hyperparam a, b 
	exp_ρ = a_s ./ b_s
	exp_log_ρ = [(digamma(a_) - log(b_s[i])) for i in 1:D]
	
	# return expected natural parameters :: Exp_ϕ (for e-step)
	return Exp_ϕ(Exp_A, Exp_AᵀA, Exp_C, Exp_R⁻¹, Exp_CᵀR⁻¹C, Exp_R⁻¹C, Exp_CᵀR⁻¹), α_n, γ_n, exp_ρ, exp_log_ρ
end

# ╔═╡ 85e2ada1-0adc-41a8-ab34-8043379ca0a4
md"""
### Testing M-step
"""

# ╔═╡ 01b6b048-6bd6-4c5a-8586-066cecf3ed51
md"""
## VBE Step: Forward Backward


"""

# ╔═╡ 781d041c-1e4d-4354-b240-12511207bde0
md"""
### Forward recursion (filtering)

**Variational derivation**

$\begin{align}
α_t(x_t) &= \frac{1}{ζ_t^{'}} \int dx_{t-1} \ \mathcal N(x_{t-1}|μ_{t-1}, Σ_{t-1}) \ \langle p(x_t| x_{t-1}) p(y_t|x_t) \rangle_{q_θ(θ)} \\
&= \frac{1}{ζ_t^{'}} \int dx_{t-1} \ \mathcal N(x_{t-1}|μ_{t-1}, Σ_{t-1}) \\
&* \exp \frac{-1}{2} (\langle (x_t - Ax_{t-1})^T I (x_t - Ax_{t-1}) + (y_t - Cx_t)^T R^{-1} (y_t - Cx_t) + K \ln|2π| + \ln|2π R| \rangle_{q_θ(θ)}) \\
&= \frac{1}{ζ_t^{'}} \int dx_{t-1} \ \mathcal N(x_{t-1}|μ_{t-1}, Σ_{t-1}) \\
&* \exp \frac{-1}{2} \{ x_{t-1}^T \langle A^TA \rangle_{q_θ(θ)} x_{t-1} - 2x_{t-1}^T \langle A \rangle_{q_θ(θ)} x_t + x_t^T \langle C^TR^{-1}C \rangle_{q_θ(θ)} x_t - 2x_t^T\langle C^TR^{-1}\rangle_{q_θ(θ)} y_t + const. \}\\
&= \mathcal N(x_t|μ_t, Σ_t)
\end{align}$

where $\langle \cdot \rangle_{q_θ(θ)}$, expectation under the variational posterior parameters are calculated from the VBM step.

Recognizing quadratic terms of $x_t$ and $x_{t-1}$ in the exponent. Analog to point-parameter derviation above, parameter expectations $\langle \cdot \rangle_{q_θ(θ)}$ from VBM step now take the place of fixed $A,C,R$, yielding:

$\mathbf{Σ^*} = (Σ_{t-1}^{-1} + \langle A^TA \rangle)^{-1}$
$m^* = Σ^* (Σ_{t-1}^{-1}μ_{t-1} + \langle A \rangle^Tx_t)$

and filtered mean and co-variance as:

$Σ_t = (I + \langle C^T R^{-1} C \rangle - \langle A \rangle \mathbf{Σ^*} \langle A \rangle^T)^{-1}$

$μ_t = Σ_t (\langle C^TR^{-1} \rangle y_t + \langle A \rangle \mathbf{Σ^*}Σ_{t-1}^{-1}μ_{t-1})$

where $t = \{1, ... T\}$
"""

# ╔═╡ cb1a9949-59e1-4ccb-8efc-aa2ffbadaab2
function v_forward(ys::Matrix{Float64}, exp_np::Exp_ϕ, hpp::HPP)
    D, T = size(ys)
    K = size(exp_np.A, 1)

    μs = zeros(K, T)
    Σs = zeros(K, K, T)
	Σs_ = zeros(K, K, T)
	
	# TO-DO: ELBO and convergence check
	#Qs = zeros(D, D, T)
	#fs = zeros(D, T)

	# Extract μ_0 and Σ_0 from the HPP struct
    μ_0 = hpp.μ_0
    Σ_0 = hpp.Σ_0

	# initialise for t=1
	Σ₀_ = inv(inv(Σ_0) + exp_np.AᵀA)
	Σs_[:, :, 1] = Σ₀_
	
    Σs[:, :, 1] = inv(I + exp_np.CᵀR⁻¹C - exp_np.A*Σ₀_*exp_np.A')
    μs[:, 1] = Σs[:, :, 1]*(exp_np.CᵀR⁻¹*ys[:, 1] + exp_np.A*Σ₀_*inv(Σ_0)μ_0)

	# iterate over T
	for t in 2:T
		Σₜ₋₁_ = inv(inv(Σs[:, :, t-1]) + exp_np.AᵀA)
		Σs_[:, :, t] = Σₜ₋₁_
		
		Σs[:, :, t] = inv(I + exp_np.CᵀR⁻¹C - exp_np.A*Σₜ₋₁_*exp_np.A')
    	μs[:, t] = Σs[:, :, t]*(exp_np.CᵀR⁻¹*ys[:, t] + exp_np.A*Σₜ₋₁_*inv(Σs[:, :, t-1])μs[:, t-1])

	end

	return μs, Σs, Σs_
end

# ╔═╡ c9d3b75e-e1ff-4ad6-9c66-6a1b89a1b426
md"""
### Backward recursion (smoothing)

**Variational analysis** (parallel implementation)

Similarly, we use expectation of natural parameter under variational distribution (VBM outputs) to replace the fixed point $A,C,R$ in Point-parameter implementation.


$\mathbf{Ψ_t^*} = (I + \langle C^TR^{-1}C \rangle + Ψ_t^{-1})^{-1}$

$Ψ_{t-1} = (\langle A^TA \rangle - \langle A \rangle^T\mathbf{Ψ_t^*} \langle A \rangle)^{-1}$

$η_{t-1} = Ψ_{t-1} \langle A \rangle^T \mathbf{Ψ_t^*}(\langle C^TR^{-1} \rangle y_t + Ψ_t^{-1}η_t)$

where, $t = \{T, ..., 1\}$
"""

# ╔═╡ 8cb62a79-7dbc-4c94-ae7b-2e2cc12764f4
function v_backward(ys::Matrix{Float64}, exp_np::Exp_ϕ)
    D, T = size(ys)
    K = size(exp_np.A, 1)

	ηs = zeros(K, T)
    Ψs = zeros(K, K, T)

    # Initialize the filter, t=T, β(x_T-1)
	Ψs[:, :, T] = zeros(K, K)
    ηs[:, T] = ones(K)
	
	Ψₜ = inv(I + exp_np.CᵀR⁻¹C)
	Ψs[:, :, T-1] = inv(exp_np.AᵀA - exp_np.A'*Ψₜ*exp_np.A)
	ηs[:, T-1] = Ψs[:, :, T-1]*exp_np.A'*Ψₜ*exp_np.CᵀR⁻¹*ys[:, T]
	
	for t in T-2:-1:1
		Ψₜ₊₁ = inv(I + exp_np.CᵀR⁻¹C + inv(Ψs[:, :, t+1]))
		
		Ψs[:, :, t] = inv(exp_np.AᵀA - exp_np.A'*Ψₜ₊₁*exp_np.A)
		ηs[:, t] = Ψs[:, :, t]*exp_np.A'*Ψₜ₊₁*(exp_np.CᵀR⁻¹*ys[:, t+1] + inv(Ψs[:, :, t+1])ηs[:, t+1])
	end

	# for t=1, this correspond to β(x_0), the probability of all the data given the setting of the auxiliary x_0 hidden state.
	Ψ₁ = inv(I + exp_np.CᵀR⁻¹C + inv(Ψs[:, :, 1]))
		
	Ψ_0 = inv(exp_np.AᵀA - exp_np.A'*Ψ₁*exp_np.A)
	η_0 = Ψs[:, :, 1]*exp_np.A'*Ψ₁*(exp_np.CᵀR⁻¹*ys[:, 1] + inv(Ψs[:, :, 1])ηs[:, 1])
	
	return ηs, Ψs, η_0, Ψ_0
end

# ╔═╡ d5457335-bc65-4bf1-b6ed-796dd5e2ab69
md"""
### Marginal and Pairwise beliefs

**Variational Bayesian**

We get **marginal beliefs** ($\mathbf{γ}$) by combining α and β messages from forward and backward recursion:

$\begin{align}
p(x_t|y_{1:T}) &\propto p(x_t|y_{1:t}) \ p(y_{t+1:T}|x_t) \\
&= α_t(x_t)\ β_t(x_t) \\
&= \mathcal N(x_t|ω_t, Υ_{t,t})
\end{align}$

Product of two Normal distributions are still normally distributed, with

$Υ_{t,t} = (Σ_t^{-1} + Ψ_t^{-1})^{-1}$
$ω_t = Υ_{t,t}(Σ_t^{-1}μ_t + Ψ_t^{-1}η_t)$

We get **pairwise beliefs** ($\mathbf{ξ}$) as:

$\begin{align}
p(x_t, x_{t+1}|y_{1:T}) &\propto p(x_t|y_{1:t}) \ p(x_{t+1}|x_t)\ p(y_{t+1}|x_{t+1})\ p(y_{t+2:T}|x_{t+1}) \\
&= α_t(x_t)\ p(x_{t+1}|x_t) \ p(y_{t+1}|x_{t+1}) \ β_{t+1}(x_{t+1}) \\
&\xrightarrow{VB} α_t(x_t) \ \exp \langle \ln p(x_{t+1}|x_t) + \ln p(y_{t+1}|x_{t+1}) \rangle \ β_{t+1}(x_{t+1}) \\
&= \mathcal N(\begin{bmatrix} x_t \\ x_{t+1} \end{bmatrix}|\begin{bmatrix} ω_t \\ ω_{t+1} \end{bmatrix}, \begin{bmatrix} Υ_{t,t} \ \ Υ_{t,t+1} \\ Υ_{t,t+1}^T \ \ Υ_{t+1,t+1}\end{bmatrix})\\
\end{align}$

where $Υ_{t,t+1} = \mathbf{Σ^*} \langle A \rangle^T (I + \langle C^T R^{-1} C \rangle + Ψ_{t+1}^{-1} - \langle A \rangle \mathbf{Σ^*} \langle A \rangle^T)^{-1}$
"""

# ╔═╡ 96ff4afb-fe7f-471a-b15e-26676c600090
# combine forward and backward pass
# marginals beliefs t = 0, ..., T (REUSE for VB)
function parallel_smoother(μs, Σs, ηs, Ψs, η_0, Ψ_0, μ_0, Σ_0)
	K, T = size(μs)
	Υs = zeros(K, K, T)
	ωs = zeros(K, T)

	# ending condition t=T
	Υs[:, :, T] = Σs[:, :, T]
	ωs[:, T] = μs[:, T]
	
	for t in 1:(T-1)
		Υs[:, :, t] = inv(inv(Σs[:, :, t]) + inv(Ψs[:, :, t]))
		ωs[:, t] = Υs[:, :, t]*(inv(Σs[:, :, t])μs[:, t] + inv(Ψs[:, :, t])ηs[:, t])
	end

	# t = 0
	Υ_0 = inv(inv(Σ_0) + inv(Ψ_0))
	ω_0 = Υ_0*(inv(Σ_0)μ_0 + inv(Ψ_0)η_0)
	
	return ωs, Υs, ω_0, Υ_0
end

# ╔═╡ 14d4e0a3-8db0-4c57-bb33-497e1bd3c64c
md"""
In summary, VBE Step consists of a forward pass (α_messages) and a backward pass (β-messages), and computing the marginal and pair-wise beliefs. 

N.B. **calculating marginals straight after each $β_t(x_t)$ could be more efficient**
"""

# ╔═╡ c1640ff6-3047-42a5-a5cd-d0c77aa41179
function v_pairwise_x(Σs_, exp_np::Exp_ϕ, Ψs)
	T = size(Σs_, 3)
    K = size(exp_np.A, 1)

	# cross-covariance is then computed for all time steps t = 0, ..., T−1
	Υ_ₜ₋ₜ₊₁ = zeros(K, K, T)
	
	for t in 1:T-2
		Υ_ₜ₋ₜ₊₁[:, :, t+1] = Σs_[:, :, t+1]*exp_np.A'*inv(I + exp_np.CᵀR⁻¹C + inv(Ψs[:, :, t+1]) - exp_np.A*Σs_[:, :, t+1]*exp_np.A')
	end

	# t=0, the cross-covariance between the zeroth and first hidden states.
	Υ_ₜ₋ₜ₊₁[:, :, 1] = Σs_[:, :, 1]*exp_np.A'*inv(I + exp_np.CᵀR⁻¹C + inv(Ψs[:, :, 1]) - exp_np.A*Σs_[:, :, 1]*exp_np.A')

	# t=T-1, Ψs[T] = 0 special case
	Υ_ₜ₋ₜ₊₁[:, :, T] = Σs_[:, :, T]*exp_np.A'*inv(I + exp_np.CᵀR⁻¹C - exp_np.A*Σs_[:, :, T]*exp_np.A')
	
	return Υ_ₜ₋ₜ₊₁
end

# ╔═╡ 6749ddf8-633b-498e-aecb-9e1592050ed2
md"""
### Expected sufficient statistics $W_A, S_A, W_C, S_C$

$W_A = \sum_{t=1}^T \langle x_{t-1} x_{t-1}^T \rangle = \sum_{t=1}^T Υ_{t-1,t-1} + ω_{t-1} ω_{t-1}^T$

$S_A = \sum_{t=1}^T \langle x_{t-1} x_t^T \rangle = \sum_{t=1}^T Υ_{t-1,t} + ω_{t-1} ω_t^T$

$W_C = \sum_{t=1}^T \langle x_t x_t^T \rangle = \sum_{t=1}^T Υ_{t,t} + ω_t ω_t^T$

$S_C = \sum_{t=1}^T \langle x_t \rangle y_t^T = \sum_{t=1}^T ω_ty_t^T$
"""

# ╔═╡ 52a70be4-fb8c-40d8-9c7a-226649ada6e3
md"""
**Debug notes**: if all hidden states are treated as observed, i.e. parse true x from data generation


$\langle x_t x_t^T \rangle = \int x_t x_t^T q(x_t) \ dx_t$

if $x_t$ is observed, $q(x_t)$ can be viewed as direct measure (delta function), hence 

$\langle x_t x_t^T \rangle = x_t x_t^T$
"""

# ╔═╡ fbc24a7c-48a0-43cc-9dd2-440acfb41c39
# infer hidden state distribution q_x(x_0:T)
function vb_e(ys::Matrix{Float64}, exp_np::Exp_ϕ, hpp::HPP)
    _, T = size(ys)
	# forward pass α_t(x_t)
	μs, Σs, Σs_ = v_forward(ys, exp_np, hpp)

	# backward pass β_t(x_t)
	ηs, Ψs, η₀, Ψ₀ = v_backward(ys, exp_np)

	# marginal (smoothed) means, covs, and pairwise beliefs 
	ωs, Υs, ω_0, Υ_0 = parallel_smoother(μs, Σs, ηs, Ψs, η₀, Ψ₀, hpp.μ_0, hpp.Σ_0)

	Υ_ₜ₋ₜ₊₁ = v_pairwise_x(Σs_, exp_np, Ψs)
	
	# hidden state sufficient stats 
	W_A = sum(Υs[:, :, t-1] + ωs[:, t-1] * ωs[:, t-1]' for t in 2:T)
	W_A += Υ_0 + ω_0*ω_0'

	S_A = sum(Υ_ₜ₋ₜ₊₁[:, :, t] + ωs[:, t-1] * ωs[:, t]' for t in 2:T)
	S_A += Υ_ₜ₋ₜ₊₁[:, :, 1] + ω_0*ωs[:, 1]'
	
	W_C = sum(Υs[:, :, t] + ωs[:, t] * ωs[:, t]' for t in 1:T)
	S_C = sum(ωs[:, t] * ys[:, t]' for t in 1:T)
	
	return HSS(W_A, S_A, W_C, S_C), ω_0, Υ_0
end

# ╔═╡ a810cf76-2c64-457c-b5ea-eaa8bf4b1d42
md"""
### Testing E-step
"""

# ╔═╡ 408dd6d8-cb5f-49ce-944b-50a0d9cebef5
md"""
Ground truth HSS using x_true
"""

# ╔═╡ 9373df69-ba17-46e0-a48a-ab1ca7dc3a9f
md"""
### Hyper-param learning - Hierachical Model

N.B. should be cautious with model comparison and structure discovery tasks.

Our prior specification involves a few hyper-parameters $\mathbf{α}, \mathbf{γ}$, and the prior parameters $Σ_0$ and $μ_0$

Straight after the VBM Step, the following can be updated:

$α_j^{-1} = \frac{1}{K}\{K Σ_A + Σ_AS_AS_A^TΣ_A\}_{j, j}$

$γ_j^{-1} = \frac{1}{D}\{D Σ_C + Σ_C S_C diag(\mathbf{\bar{ρ}}) S_C^T Σ_C \}_{j, j}$

$Σ_0 = Υ_{0, 0}$

$μ_0 = ω_0$

The **+ve** hyperparameters $a, b$ governing the prior distribution over the output noise,
$R = diag (\mathbf{ρ})$, are set to the fixed point of the equations [See Beal Appendix C.2 for how to solve]:

$ψ(a) = \ln b + \frac{1}{D} \sum_{s=1}^D \bar{\ln ρ_s}$

$\frac{1}{b} = \frac{1}{D \times a} \sum_{s=1}^D \bar{ρ_s}$

**TO-DO**: 
These are also updated during learning, note for a valid gamma distribution, both a and b needs to be positive! 

Using properties of Gamma distribution (shape $a$, inverse scale $b$), expectation and log expectation are given by:

$\bar{ρ_s} \equiv \langle ρ_s \rangle = \frac{a + 0.5 T}{b + 0.5 G_{ss}}$

$\bar{\ln ρ_s} \equiv \langle \ln ρ_s \rangle = ψ(a + 0.5T) - \ln(b + 0.5 G_{ss})$

To avoid solving for -ve values of $a$, we can re-parameterize $a$ to $a'$ such that $a = \exp(a')$ to ensure $a$ is always positive, and then solve a different fixed point equation for $a'$:

$a_n' = a' - \frac{g(a')}{g'(a')}$

$\exp(a_n') = \exp(a') \exp(-\frac{g(a')}{g'(a')})$

$g(a') = ψ(\exp(a')) - a' + \ln d - c$

$g'(a') = ψ'(\exp(a')) \exp(a') - 1$
"""

# ╔═╡ 87667a9e-02aa-4104-b5a0-0f6b9e98ba96
# cf. Newton's method
function update_ab(hpp::HPP, exp_ρ::Vector{Float64}, exp_log_ρ::Vector{Float64})
    D = length(exp_ρ)
    d = mean(exp_ρ)
    c = mean(exp_log_ρ)
    
    # Update `a` using fixed point iteration
	a = hpp.a		

    for _ in 1:100
        ψ_a = digamma(a)
        ψ_a_p = trigamma(a)
        
        a_new = a * exp(-(ψ_a - log(a) + log(d) - c) / (a * ψ_a_p - 1))
		a = a_new

		# check convergence
        if abs(a_new - a) < 1e-6
            break
        end
    end
    
    # Update `b` using the converged value of `a`
    b = a/d

	return a, b
end

# ╔═╡ b0b1f14d-4fbd-4995-845f-f19990460329
md"""
## VB DLM
"""

# ╔═╡ 1902c1f1-1246-4ab3-88e6-35619d685cdd
function vb_dlm(ys::Matrix{Float64}, hpp::HPP, hpp_learn = false, max_iter=100, r_seed=99)
	D, T = size(ys)
	K = length(hpp.α)
	
	Random.seed!(r_seed) # different seed? sensitive to inialisation/local minima is expected 

	W_A = rand(K, K)
	S_A = rand(K, K)
	W_C = rand(K, K)
	S_C = rand(D, K)
	
	#W_A = sum(x_true[:, t-1] * x_true[:, t-1]' for t in 2:T)
	#W_A += μ_0*μ_0'
	#S_A = sum(x_true[:, t-1] * x_true[:, t]' for t in 2:T)
	#S_A += μ_0*x_true[:, 1]'
	#W_C = sum(x_true[:, t] * x_true[:, t]' for t in 1:T)
	#S_C = sum(x_true[:, t] * y[:, t]' for t in 1:T)
	
	hss = HSS(W_A, S_A, W_C, S_C)
	exp_np = missing

	for i in 1:max_iter

		exp_np, α_n, γ_n, exp_ρ, exp_log_ρ = vb_m(ys, hpp, hss)
		
		a, b = update_ab(hpp, exp_ρ, exp_log_ρ)
		
		hss, ω_0, Υ_0 = vb_e(ys, exp_np, hpp)

		if (hpp_learn)
			hpp = HPP(α_n, γ_n, a, b, ω_0, Υ_0)
		end

		#TO-DO: ELBO and Convergence
	end
	return exp_np
end

# ╔═╡ 3c63b27b-76e3-4edc-9b56-345738b97c41
md"""
Ground truth values of DLM parameters
"""

# ╔═╡ dc2c58de-98b9-4621-b465-064e8ab3caf1
md"""
Result from VB DLM:
"""

# ╔═╡ dab1fe9c-20a4-4376-beaf-02b5292ca7cd
md"""
For initialisation with seed = 99, vb dlm is able to yield some reasonable learning results, although adding hyperparameter learning does not seem to improve much.
"""

# ╔═╡ 24de2bcb-cf9d-44f7-b1d7-f80ae8c08ed1
md"""
## Testing notes:

Case **matrix variate linear regression**, by choosing A = I (the identity matrix) and Q = 0 (the zero matrix) for the Dynamic Linear Model (DLM) setup has specific implications:

    A = I: This choice implies that the state vector at time t is equal to the state vector at time t-1. In other words, the states do not change over time. This is a strong assumption and may not be applicable for all situations. However, it simplifies the model and can be useful in cases where we believe that the underlying state does not change significantly over time.

    Q = 0: This choice implies that there is no process noise in the model. In other words, we are assuming that the evolution of the state over time is deterministic and not influenced by any unobserved random effects. This is also a strong assumption and may not be suitable in cases where there are unobserved influences on the state that we want to model.
"""

# ╔═╡ e3e78fb1-00aa-4399-8330-1d4a08742b42
md"""
Case **probabilistic PCA**, To reduce a DLM to PPCA, we should set:

    The state transition matrix A to be a zero matrix. This means that the hidden state does not depend on the previous hidden state, which is consistent with the assumption of PPCA where hidden states are independently drawn from a Gaussian distribution.

    The process noise covariance matrix Q to be an identity matrix. This means that the hidden states are sampled from a standard Gaussian distribution.

    The initial state mean to be a zero vector and the initial state covariance to be an identity matrix. This is consistent with the assumption in PPCA where the hidden states are sampled from a standard Gaussian distribution.

With these settings, the DLM essentially becomes a model where the observed variables are linear functions of the hidden states (with some Gaussian noise), and the hidden states are independently drawn from a Gaussian distribution. This is the setting of PPCA.
"""

# ╔═╡ be042373-ed3e-4e2e-b714-b4f9e5964b57
md"""
## Debugging notes: 

-> check vb-m with HSS using x_true (✓)

-> check vb-e with Exp_ϕ using A, C, R (✓)

-> check forward, backward with StateSpaceModels -> consider first uni-variate local level model (see separate notebook)

-> verify with MCMC and Turing (✓ - see separate notebook)
"""

# ╔═╡ b2818ed9-6ef8-4398-a9d4-63b1d399169c
md"""
## Appendix
"""

# ╔═╡ 8fed847c-93bc-454b-94c7-ba1d13c73b04
md"""
Generate test data
"""

# ╔═╡ baca3b20-16ac-4e37-a2bb-7512d1c99eb8
md"""
### Kalman Filter
"""

# ╔═╡ e7ca9061-64dc-44ef-854e-45b8015abad1
md"""
Aside, **Point-parameter** Kalman Filter for DLM (cf. Beale Chap 5 5.80, 5.81, 5.85)

Filtering density: $α_t(x_t) \equiv p(x_t|y_{1:t})$

$\begin{align}
α_t(x_t) &= \frac{p(x_t|y_{1:t-1}) p(y_t|x_t)}{p(y_t|y_{1:t-1})}\\

&= \int dx_{t-1} \ p(x_{t-1}|y_{1:t-1}) p(x_t|x_{t-1}) \frac{p(y_t|x_t)}{p(y_t|y_{1:t-1})} \\

&= \frac{1}{p(y_t|y_{1:t-1})} \int dx_{t-1} \ α_{t-1}(x_{t-1}) \  p(x_t|x_{t-1}) \ p(y_t|x_t) \\

&= \frac{1}{ζ_t} \int dx_{t-1} \ \mathcal N(x_{t-1}|μ_{t-1}, Σ_{t-1}) \ \mathcal N(x_t|Ax_{t-1}, I) \ \mathcal N(y_t| Cx_t, R)\\

&= \mathcal N(x_t|μ_t, Σ_t)
\end{align}$

Complete the square of qudratic terms involving $x_{t-1}$ form a Normal distribution $\mathcal N(x_{t-1}|m^*, Σ^*)$

$\mathbf{Σ^*} = (Σ_{t-1}^{-1} + A^TA)^{-1}$
$m^* = Σ^* (Σ_{t-1}^{-1}μ_{t-1} + A^T x_t)$

Filtered mean $μ_t$ and co-variance $Σ_t$ are computed by marginalising $x_{t-1}$, using the expression $\mathbf{Σ^*}$:

$Σ_t = (I + C^T R^{-1} C - A\mathbf{Σ^*}A^T)^{-1}$

$μ_t = Σ_t (C^TR^{-1}y_t + A\mathbf{Σ^*}Σ_{t-1}^{-1}μ_{t-1})$

The denominator of the normalising constant $ζ_t(y_t)$ is also a Normal distribution, and we have (using DLM with R notations):

$p(y_{1:T}) = p(y_1) \prod_{t=2}^T p(y_t|y_{1:t-1}) = \prod_{t=1}^T ζ_t(y_t)$

$ζ_t(y_t) \sim \mathcal N(y_t|f_t, Q_t)$

$Q_t = (R^{-1} - R^{-1}CΣ_tC^TR^{-1})^{-1}$ 

$f_t = Q_tR^{-1}CΣ_tA\mathbf{Σ^*}Σ_{t-1}^{-1}μ_{t-1}$

N.B. Unlike `Dynamic Linear Model with R` Chap 2.7.2, results above taken from Beal's thesis have not been simplified, this is a purposeful choice in order to get an analog with the **Variational Bayesian derviation** next.
"""

# ╔═╡ 59bcc9bf-276c-47e1-b6a9-86f90571c0fb
# using Beale 5.77, 5.80, 5.81, forward pass - Kalman-filter
function p_forward(ys, A, C, R, μ₀, Σ₀)
    D, T = size(ys)
    K = size(A, 1)
	
    μs = zeros(K, T)
    Σs = zeros(K, K, T)
	Σs_ = zeros(K, K, T)
	Qs = zeros(D, D, T)
	fs = zeros(D, T)
	
    # Initialize the filter, t=1
	Σ₀_ = inv(inv(Σ₀) + A'A)
	Σs_[:, :, 1] = Σ₀_
	
    Σs[:, :, 1] = inv(I + C'inv(R)C - A*Σ₀_*A')
    μs[:, 1] = Σs[:, :, 1]*(C'inv(R)ys[:, 1] + A*Σ₀_*inv(Σ₀)μ₀)

	Qs[:, :, 1] = inv(inv(R) - inv(R)*C*Σs[:, :, 1]*C'*inv(R))
	fs[:, 1] = Qs[:, :, 1]*inv(R)*C*Σs[:, :, 1]*A*Σ₀_*inv(Σ₀)*μ₀
	#fs[:, 1] = C*A*μ₀
	#Qs[:, :, 1] = R + C*(I+A*Σ₀*A')*C'
	
    for t in 2:T
		Σₜ₋₁_ = inv(inv(Σs[:, :, t-1]) + A'A)
		Σs_[:, :, 1] = Σₜ₋₁_

        Σs[:, :, t] = inv(I + C'inv(R)C - A*Σₜ₋₁_*A')
		μs[:, t] = Σs[:, :, t]*(C'inv(R)ys[:, t] + A*Σₜ₋₁_*inv(Σs[:, :, t-1])μs[:, t-1])

		Qs[:, :, t] = inv(inv(R) - inv(R)*C*Σs[:, :, t]*C'*inv(R))
		fs[:, t] = Qs[:, :, t]*inv(R)*C*Σs[:, :, t]*A*Σₜ₋₁_*inv(Σs[:, :, t-1])*μs[:, t-1]

		#fs[:, t] = C*A*μs[:, t-1]
		#Qs[:, :, 1] = R + C*(I+A*Σs[:, :, t-1]*A')*C'
    end

    return μs, Σs, fs, Qs, Σs_
end

# ╔═╡ a5ae35dc-cc4b-48bd-869e-37823b8073d2
begin
	function gen_data(A, C, Q, R, μ_0, Σ_0, T)
	    K, _ = size(A)
	    D, _ = size(C)
	    x = zeros(K, T)
	    y = zeros(D, T)
	
	    x[:, 1] = rand(MvNormal(A*μ_0, A'*Σ_0*A + Q))
	    y[:, 1] = C * x[:, 1] + rand(MvNormal(zeros(D), R))
	
	    for t in 2:T
			if (tr(Q) != 0)
	        	x[:, t] = A * x[:, t-1] + rand(MvNormal(zeros(K), Q))
			else
				x[:, t] = A * x[:, t-1] # Q zero matrix special case
			end
	        y[:, t] = C * x[:, t] + rand(MvNormal(zeros(D), R)) 
	    end
	
	    return y, x
	end
	
	# Ground truth values
	A = [0.8 -0.1; 0.2 0.75]
	C = [1.0 0.0; 0.0 1.0]
	
	R = Diagonal([0.33, 0.33]) # prefer small R to get better filtered accuracy, c.f signal to noise ratio (DLM with R Chap 2)
	
	μ_0 = [0.0, 0.0]
	Σ_0 = Diagonal([1.0, 1.0])
	T = 10000
	
	# Generate the toy dataset
	Random.seed!(100)
	y, x_true = gen_data(A, C, Diagonal([1.0, 1.0]), R, μ_0, Σ_0, T)
	
	# Test the Kalman filter
	x_hat, Px, y_hat, Py, _ = p_forward(y, A, C, R, μ_0, Σ_0)
end;

# ╔═╡ 2c9a233f-3a96-43dc-b783-b82642a82590
A, C, R

# ╔═╡ d9cb7c74-007d-4229-a576-a7a41fff565b
# ╠═╡ disabled = true
#=╠═╡
let
	D, T = size(y)
	K = size(A, 1)

	# DEBUG, initialise HSS using real x from data generation
	W_A = sum(x_true[:, t-1] * x_true[:, t-1]' for t in 2:T)
	W_A += μ_0*μ_0'

	S_A = sum(x_true[:, t-1] * x_true[:, t]' for t in 2:T)
	S_A += μ_0*x_true[:, 1]'

	W_C = sum(x_true[:, t] * x_true[:, t]' for t in 1:T)

	S_C = sum(x_true[:, t] * y[:, t]' for t in 1:T)

	hss = HSS(W_A, S_A, W_C, S_C)
	α = ones(K)
	γ = ones(K)
	a = 0.1
	b = 0.1
	μ_0 = zeros(K)
	Σ_0 = Matrix{Float64}(I, K, K)

	hpp = HPP(α, γ, a, b, μ_0, Σ_0)

	# should recover values of A, C, R close to ground truth
	exp_np = vb_m(y, hpp, hss)[1]
end
  ╠═╡ =#

# ╔═╡ 8a73d154-236d-4660-bb21-24681ed7d315
# ╠═╡ disabled = true
#=╠═╡
let
	D, T = size(y)
	K = size(A, 1)

	# use fixed A,C,R from ground truth for the exp_np::Exp_ϕ
	e_A = A
	e_AᵀA = A'A
	e_C = C
	e_R⁻¹ = inv(R)
	e_CᵀR⁻¹C = e_C'*e_R⁻¹*e_C
	e_R⁻¹C = e_R⁻¹*e_C
	e_CᵀR⁻¹ = e_C'*e_R⁻¹

	exp_np = Exp_ϕ(e_A, e_AᵀA, e_C, e_R⁻¹, e_CᵀR⁻¹C, e_R⁻¹C, e_CᵀR⁻¹)
	α = ones(K)
	γ = ones(K)
	a = 0.1
	b = 0.1
	μ_0 = zeros(K)
	Σ_0 = Matrix{Float64}(I, K, K)

	hpp = HPP(α, γ, a, b, μ_0, Σ_0)

	# should recover very similar hss using ground-truth xs
	vb_e(y, exp_np, hpp)[1]
end
  ╠═╡ =#

# ╔═╡ fb472969-3c3c-4787-8cf1-296f2c13ddf5
# ╠═╡ disabled = true
#=╠═╡
let
	W_A = sum(x_true[:, t-1] * x_true[:, t-1]' for t in 2:T)
	W_A += μ_0*μ_0'
	S_A = sum(x_true[:, t-1] * x_true[:, t]' for t in 2:T)
	S_A += μ_0*x_true[:, 1]'
	W_C = sum(x_true[:, t] * x_true[:, t]' for t in 1:T)
	S_C = sum(x_true[:, t] * y[:, t]' for t in 1:T)
	hss = HSS(W_A, S_A, W_C, S_C)
end
  ╠═╡ =#

# ╔═╡ f871da95-6710-4c0f-a3a1-890dd59a41a1
A, C, R

# ╔═╡ 079cd7ef-632d-41d0-866d-6678808a8f4c
let
	K = size(A, 1)
	D = size(y, 1)
	
	# specify initial priors (hyper-params)
	α = ones(K)
	γ = ones(K)
	a = 0.1
	b = 0.1
	μ_0 = zeros(K)
	Σ_0 = Matrix{Float64}(I, K, K)
	hpp = HPP(α, γ, a, b, μ_0, Σ_0)

	l_off = vb_dlm(y, hpp) 
	l_on = vb_dlm(y, hpp, true)

	# results with hyperparam learning on and off
	l_on, l_off
end

# ╔═╡ ed704f46-779c-4369-8a3b-d3e8cf0f4dd1
begin
	Random.seed!(99)
	y_ml, xs_ml = gen_data(Diagonal([1.0, 1.0]), C, zeros(2, 2), R, μ_0, Σ_0, 1000)
end

# ╔═╡ 6550261c-a3b8-40bc-a4ac-c43ae33215ca
begin
	Random.seed!(99)
	y_pca, xs_pca = gen_data(zeros(2, 2), C, Diagonal([1.0, 1.0]), R, μ_0, Σ_0, 1000)
end

# ╔═╡ 1a129b6f-74f0-404c-ae4f-3ae39c8431aa
y, x_true

# ╔═╡ 14a209dd-be4c-47f0-a343-1cfb97b7d04a
# ╠═╡ disabled = true
#=╠═╡
let
	T = size(y, 2)
	p1 = plot(1:T, x_true[1, :], label="True xs[1]", linewidth=2)
	plot!(1:T, x_hat[1, :], label="Filtered xs[1]", linewidth=2, linestyle=:dash)
end
  ╠═╡ =#

# ╔═╡ 5c221210-e1df-4015-b959-6d330b47be29
# ╠═╡ disabled = true
#=╠═╡
let
	T = size(y, 2)
	p2 = plot(1:T, x_true[2, :], label="True xs[2]", linewidth=2)
	plot!(1:T, x_hat[2, :], label="Filtered xs[2]", linewidth=2, linestyle=:dash)
end
  ╠═╡ =#

# ╔═╡ 7c20c3ab-b0ae-48fc-b2f0-9cde30559bf5
# ╠═╡ disabled = true
#=╠═╡
let
	T = size(y, 2)
	_, _, y_hat, y_cov = p_forward(y, A, C, R, μ_0, Σ_0)
	y_hat = circshift(y_hat, (0, -1))
	y_hat[:, T] .= NaN  # Set the first column to NaN to avoid connecting the last point to the first

	p1 = plot(1:T, y[1, :], label="True y[1]", linewidth=2)
	plot!(1:T, y_hat[1, :], label="Filtered y[1]", linewidth=2, linestyle=:dash)
end
  ╠═╡ =#

# ╔═╡ e02d0dd5-6bab-4548-8bbe-d9b1759688c5
# ╠═╡ disabled = true
#=╠═╡
let
	T = size(y, 2)
	_, _, y_hat, y_cov = p_forward(y, A, C, R, μ_0, Σ_0)
	y_hat = circshift(y_hat, (0, -1))
	y_hat[:, T] .= NaN

	p2 = plot(1:T, y[2, :], label="True y[2]", linewidth=2)
	plot!(1:T, y_hat[2, :], label="Filtered y[2]", linewidth=2, linestyle=:dash)
end
  ╠═╡ =#

# ╔═╡ c417e618-41c2-454c-9b27-470988215d48
md"""
Aside, **Point-parameter** Parallel Smoother

Define, β-messages as:

$β_t(x_t) \equiv p(y_{t+1:T}|x_t)$

Then, analogous to the α-message from the Kalman Filter:

$\begin{align}
β_{t-1}(x_{t-1}) &= p(y_{t:T}|x_{t-1})\\
&= \int dx_t \ p(x_t, y_t, y_{t+1:T}|x_{t-1})\\
&= \int dx_t \ p(x_t|x_{t-1}) \ p(y_t|x_t) \ p(y_{t+1:T}|x_t)\\
&= \int dx_t \ p(x_t|x_{t-1}) \ p(y_t|x_t) \ β_t(x_t)\\
&\propto \mathcal N(x_{t-1}|η_{t-1}, Ψ_{t-1})
\end{align}$

Ending condition: $β_T(x_T) = 1$, $Ψ_T^{-1} = \mathbf{0}$

cf. Beale 5.110-5.112

$\mathbf{Ψ_t^*} = (I + C^TR^{-1}C + Ψ_t^{-1})^{-1}$

$Ψ_{t-1} = (A^TA - A^T\mathbf{Ψ_t^*}A)^{-1}$

$η_{t-1} = Ψ_{t-1}A^T\mathbf{Ψ_t^*}(C^TR^{-1}y_t + Ψ_t^{-1}η_t)$

"""

# ╔═╡ 8950aa50-22b2-4299-83b2-b9abfd1d5303
# from t=T-1 to 0, point-parameter approach cf. Beal pg 180
function parallel_backward(y, A, C, R)
	D, T = size(y)
	K = size(A, 1)

	ηs = zeros(K, T)
    Ψs = zeros(K, K, T)

    # Initialize the filter, t=T
    Ψs[:, :, T] = zeros(K, K)
    ηs[:, T] = ones(K)

	Ψₜ = inv(I + C'inv(R)C)
	
	Ψs[:, :, T-1] = inv(A'A - A'*Ψₜ*A)
	ηs[:, T-1] = Ψs[:, :, T-1]*A'*Ψₜ*C'inv(R)y[:, T]

	for t in (T - 2):-1:1
		Ψₜ₊₁ = inv(I + C'inv(R)C + inv(Ψs[:, :, t+1]))
		
		Ψs[:, :, t] = inv(A'A - A'*Ψₜ₊₁*A)
		ηs[:, t] = Ψs[:, :, t]*A'*Ψₜ₊₁*(C'inv(R)y[:, t+1] + inv(Ψs[:, :, t+1])ηs[:, t+1])
	end

	Ψ₁ = inv(I + C'inv(R)C + inv(Ψs[:, :, 1]))
	Ψ_0 = inv(A'A - A'*Ψ₁*A)
	η_0 = Ψs[:, :, 1]*A'Ψ₁*(C'inv(R)y[:, 1] + inv(Ψs[:, :, 1])ηs[:, 1])
	
	return ηs, Ψs, η_0, Ψ_0
end

# ╔═╡ 30502079-9684-4144-8bcd-a70f2cb5928a
function p_pairwise_x(Σs_, A, Υs)
	T = size(Σs_, 3)
	K = size(A, 1)

	# cross-covariance is then computed for all time steps t = {0, . . . , T − 1}
	Υ_ₜ₋ₜ₊₁ = zeros(K, K, T)

	for t in 1:T
		Υ_ₜ₋ₜ₊₁[:, :, t] = Σs_[:, :, t]*A'*Υs[:, :, t]
	end

	return Υ_ₜ₋ₜ₊₁
end

# ╔═╡ ca825009-564e-43e0-9014-cce87c46533b
# ╠═╡ disabled = true
#=╠═╡
function error_metrics(true_means, smoothed_means)
    T = size(true_means, 2)
    mse = sum((true_means .- smoothed_means).^2) / T
    mad = sum(abs.(true_means .- smoothed_means)) / T
    mape = sum(abs.((true_means .- smoothed_means) ./ true_means)) / T * 100

	# mean squared error (MSE), mean absolute deviation (MAD), and mean absolute percentage error (MAPE) 
    return mse, mad, mape
end
  ╠═╡ =#

# ╔═╡ 8bd60367-2007-4d50-9d25-c12acd73be96
md"""
MSE, MAD, MAPE error with Kalman Filter
"""

# ╔═╡ f1cea551-4feb-44b4-a77e-03621c9b37b9
# ╠═╡ disabled = true
#=╠═╡
error_metrics(x_true, x_hat)
  ╠═╡ =#

# ╔═╡ 4c8259f1-d3ae-4400-93cb-0a09b22a14ae
md"""
MSE, MAD, MAPE error with **Kalman smoother**
"""

# ╔═╡ a3677e9f-837b-4ba0-a29f-e60bf3712323
# ╠═╡ disabled = true
#=╠═╡
let
	ηs, Ψs, η_0, Ψ_0 = parallel_backward(y, A, C, R)
	ωs, Υs, ω_0, Υ_0 = parallel_smoother(x_hat, Px, ηs, Ψs, η_0 , Ψ_0, μ_0, Σ_0)
	error_metrics(x_true, ωs) #lower error compared to filtered xs
end
  ╠═╡ =#

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b"
StatsFuns = "4c63d2b9-4356-54db-8cca-17b64c39e42c"

[compat]
Distributions = "~0.25.87"
Plots = "~1.38.9"
PlutoUI = "~0.7.50"
SpecialFunctions = "~2.2.0"
StatsFuns = "~1.3.0"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.9.0"
manifest_format = "2.0"
project_hash = "d5999de436990a28cca4f7280892b159fec7b049"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "8eaf9f1b4921132a4cff3f36a1d9ba923b14a481"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.1.4"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.BitFlags]]
git-tree-sha1 = "43b1a4a8f797c1cddadf60499a8a077d4af2cd2d"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.7"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "4b859a208b2397a7a623a03449e4636bdb17bcf2"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+1"

[[deps.Calculus]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f641eb0a4f00c343bbc32346e1217b86f3ce9dad"
uuid = "49dc2e85-a5d0-5ad3-a950-438e2897f1b9"
version = "0.5.1"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "9c209fb7536406834aa938fb149964b985de6c83"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.1"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "be6ab11021cd29f0344d5c4357b163af05a48cba"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.21.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "eb7f0f8307f71fac7c606984ea5fb2817275d6e4"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.4"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "SpecialFunctions", "Statistics", "TensorCore"]
git-tree-sha1 = "600cc5508d66b78aae350f7accdb58763ac18589"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.9.10"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "fc08e5930ee9a4e03f84bfb5211cb54e7769758a"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.10"

[[deps.Compat]]
deps = ["UUIDs"]
git-tree-sha1 = "7a60c856b9fa189eb34f5f8a6f6b5529b7942957"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.6.1"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.0.2+0"

[[deps.ConcurrentUtilities]]
deps = ["Serialization", "Sockets"]
git-tree-sha1 = "96d823b94ba8d187a6d8f0826e731195a74b90e9"
uuid = "f0e56b4a-5159-44fe-b623-3e5288b988bb"
version = "2.2.0"

[[deps.Contour]]
git-tree-sha1 = "d05d9e7b7aedff4e5b51a029dced05cfb6125781"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.2"

[[deps.DataAPI]]
git-tree-sha1 = "e8119c1a33d267e16108be441a287a6981ba1630"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.14.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "d1fff3a548102f48987a52a2e0d114fa97d730f0"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.13"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.Distributions]]
deps = ["FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns", "Test"]
git-tree-sha1 = "eead66061583b6807652281c0fbf291d7a9dc497"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.90"

    [deps.Distributions.extensions]
    DistributionsChainRulesCoreExt = "ChainRulesCore"
    DistributionsDensityInterfaceExt = "DensityInterface"

    [deps.Distributions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DensityInterface = "b429d917-457f-4dbc-8f4c-0cc954292b1d"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.DualNumbers]]
deps = ["Calculus", "NaNMath", "SpecialFunctions"]
git-tree-sha1 = "5837a837389fccf076445fce071c8ddaea35a566"
uuid = "fa6b7ba4-c1ee-5f82-b5fc-ecf0adba8f74"
version = "0.6.8"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bad72f730e9e91c08d9427d5e8db95478a3c323d"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.4.8+0"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "b57e3acbe22f8484b4b5ff66a7499717fe1a9cc8"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.1"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Pkg", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "74faea50c1d007c85837327f6775bea60b5492dd"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.2+2"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "fc86b4fd3eff76c3ce4f5e96e2fdfa6282722885"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.0.0"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "21efd19106a55620a188615da6d3d06cd7f6ee03"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.93+0"

[[deps.Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "87eb71354d8ec1a96d4a7636bd57a7347dde3ef9"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.10.4+0"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pkg", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "d972031d28c8c8d9d7b41a536ad7bb0c2579caca"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.8+0"

[[deps.GR]]
deps = ["Artifacts", "Base64", "DelimitedFiles", "Downloads", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Preferences", "Printf", "Random", "Serialization", "Sockets", "TOML", "Tar", "Test", "UUIDs", "p7zip_jll"]
git-tree-sha1 = "efaac003187ccc71ace6c755b197284cd4811bfe"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.72.4"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Qt5Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "4486ff47de4c18cb511a0da420efebb314556316"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.72.4+0"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "d3b3624125c1474292d0d8ed0f65554ac37ddb23"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.74.0+2"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HTTP]]
deps = ["Base64", "CodecZlib", "ConcurrentUtilities", "Dates", "Logging", "LoggingExtras", "MbedTLS", "NetworkOptions", "OpenSSL", "Random", "SimpleBufferStream", "Sockets", "URIs", "UUIDs"]
git-tree-sha1 = "877b7bc42729aa2c90bbbf5cb0d4294bd6d42e5a"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.9.1"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "129acf094d168394e80ee1dc4bc06ec835e510a3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+1"

[[deps.HypergeometricFunctions]]
deps = ["DualNumbers", "LinearAlgebra", "OpenLibm_jll", "SpecialFunctions"]
git-tree-sha1 = "84204eae2dd237500835990bcade263e27674a93"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.16"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "c47c5fa4c5308f27ccaac35504858d8914e102f9"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.4"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "f7be53659ab06ddc986428d3a9dcc95f6fa6705a"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.2"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.IrrationalConstants]]
git-tree-sha1 = "630b497eafcc20001bba38a4651b327dcfc491d2"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.2"

[[deps.JLFzf]]
deps = ["Pipe", "REPL", "Random", "fzf_jll"]
git-tree-sha1 = "f377670cda23b6b7c1c0b3893e37451c5c1a2185"
uuid = "1019f520-868f-41f5-a6de-eb00f4b6a39c"
version = "0.1.5"

[[deps.JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "abc9885a7ca2052a736a600f7fa66209f96506e1"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.4.1"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6f2675ef130a300a112286de91973805fcc5ffbc"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.91+0"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6250b16881adf048549549fba48b1161acdac8c"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.1+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bf36f528eec6634efc60d7ec062008f171071434"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "3.0.0+1"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "f2355693d6778a178ade15952b7ac47a4ff97996"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.0"

[[deps.Latexify]]
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Printf", "Requires"]
git-tree-sha1 = "099e356f267354f46ba65087981a77da23a279b7"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.16.0"

    [deps.Latexify.extensions]
    DataFramesExt = "DataFrames"
    SymEngineExt = "SymEngine"

    [deps.Latexify.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    SymEngine = "123dc426-2d89-5057-bbad-38513e3affd8"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.3"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "7.84.0+0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.10.2+0"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0b4a5d71f3e5200a7dff793393e09dfc2d874290"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+1"

[[deps.Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll", "Pkg"]
git-tree-sha1 = "64613c82a59c120435c067c2b809fc61cf5166ae"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.7+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "6f73d1dd803986947b2c750138528a999a6c7733"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.6.0+0"

[[deps.Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c333716e46366857753e273ce6a69ee0945a6db9"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.42.0+0"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c7cb1f5d892775ba13767a87c7ada0b980ea0a71"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.16.1+2"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9c30530bf0effd46e15e0fdcf2b8636e78cbbd73"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.35.0+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "Pkg", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "3eb79b0ca5764d4799c06699573fd8f533259713"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.4.0+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7f3efec06033682db852f8b3bc3c1d2b0a0ab066"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.36.0+0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "0a1b7c2863e44523180fdb3146534e265a91870b"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.23"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "cedb76b37bc5a6c702ade66be44f831fa23c681e"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "1.0.0"

[[deps.MIMEs]]
git-tree-sha1 = "65f28ad4b594aebe22157d6fac869786a255b7eb"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "0.1.4"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "42324d08725e200c23d4dfb549e0d5d89dede2d2"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.10"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "Random", "Sockets"]
git-tree-sha1 = "03a9b9718f5682ecb107ac9f7308991db4ce395b"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.7"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.2+0"

[[deps.Measures]]
git-tree-sha1 = "c13304c81eec1ed3af7fc20e75fb6b26092a1102"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.2"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "f66bdc5de519e8f8ae43bdc598782d35a25b1272"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.1.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2022.10.11"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "0877504529a3e5c3343c6f8b4c0381e57e4387e4"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.2"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.21+4"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+0"

[[deps.OpenSSL]]
deps = ["BitFlags", "Dates", "MozillaCACerts_jll", "OpenSSL_jll", "Sockets"]
git-tree-sha1 = "51901a49222b09e3743c65b8847687ae5fc78eb2"
uuid = "4d8831e6-92b7-49fb-bdf8-b643e874388c"
version = "1.4.1"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9ff31d101d987eb9d66bd8b176ac7c277beccd09"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.20+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "d321bf2de576bf25ec4d3e4360faca399afca282"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.6.0"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.42.0+0"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "67eae2738d63117a196f497d7db789821bce61d1"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.17"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "7302075e5e06da7d000d9bfa055013e3e85578ca"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.5.9"

[[deps.Pipe]]
git-tree-sha1 = "6842804e7867b115ca9de748a0cf6b364523c16d"
uuid = "b98c9c47-44ae-5843-9183-064241ee97a0"
version = "1.3.0"

[[deps.Pixman_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b4f5d02549a10e20780a24fce72bea96b6329e29"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.40.1+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.9.0"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Statistics"]
git-tree-sha1 = "1f03a2d339f42dca4a4da149c7e15e9b896ad899"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "3.1.0"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "PrecompileTools", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "f92e1315dadf8c46561fb9396e525f7200cdc227"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.3.5"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "JLFzf", "JSON", "LaTeXStrings", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "PrecompileTools", "Preferences", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "RelocatableFolders", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs", "UnicodeFun", "Unzip"]
git-tree-sha1 = "6c7f47fd112001fc95ea1569c2757dffd9e81328"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.38.11"

    [deps.Plots.extensions]
    FileIOExt = "FileIO"
    GeometryBasicsExt = "GeometryBasics"
    IJuliaExt = "IJulia"
    ImageInTerminalExt = "ImageInTerminal"
    UnitfulExt = "Unitful"

    [deps.Plots.weakdeps]
    FileIO = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
    GeometryBasics = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
    IJulia = "7073ff75-c697-5162-941a-fcdaad2a7d2a"
    ImageInTerminal = "d8c32880-2388-543b-8c61-d9f865259254"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "b478a748be27bd2f2c73a7690da219d0844db305"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.51"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "259e206946c293698122f63e2b513a7c99a244e8"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.1.1"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "7eb1686b4f04b82f96ed7a4ea5890a4f0c7a09f1"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.Qt5Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "xkbcommon_jll"]
git-tree-sha1 = "0c03844e2231e12fda4d0086fd7cbe4098ee8dc5"
uuid = "ea2cea3b-5b76-57ae-a6ef-0a8af62496e1"
version = "5.15.3+2"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "6ec7ac8412e83d57e313393220879ede1740f9ee"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.8.2"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.RecipesBase]]
deps = ["PrecompileTools"]
git-tree-sha1 = "5c3d09cc4f31f5fc6af001c250bf1278733100ff"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.4"

[[deps.RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "PrecompileTools", "RecipesBase"]
git-tree-sha1 = "45cf9fd0ca5839d06ef333c8201714e888486342"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.6.12"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "90bc7a7c96410424509e4263e277e43250c05691"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.0"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "f65dcb5fa46aee0cf9ed6274ccbd597adc49aa7b"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.1"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6ed52fdd3382cf21947b15e8870ac0ddbff736da"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.4.0+0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "30449ee12237627992a99d5e30ae63e4d78cd24a"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.2.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.SimpleBufferStream]]
git-tree-sha1 = "874e8867b33a00e784c8a7e4b60afe9e037b74e1"
uuid = "777ac1f9-54b0-4bf8-805c-2214025038e7"
version = "1.1.0"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "a4ada03f999bd01b3a25dcaa30b2d929fe537e00"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.1.0"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "ef28127915f4229c971eb43f3fc075dd3fe91880"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.2.0"

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

    [deps.SpecialFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.9.0"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "45a7769a04a3cf80da1c1c7c60caf932e6f4c9f7"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.6.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "d1bf48bfcc554a3761a133fe3a9bb01488e06916"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.21"

[[deps.StatsFuns]]
deps = ["HypergeometricFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "f625d686d5a88bcd2b15cd81f18f98186fdc0c9a"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.3.0"

    [deps.StatsFuns.extensions]
    StatsFunsChainRulesCoreExt = "ChainRulesCore"
    StatsFunsInverseFunctionsExt = "InverseFunctions"

    [deps.StatsFuns.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "Pkg", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "5.10.1+6"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.TranscodingStreams]]
deps = ["Random", "Test"]
git-tree-sha1 = "9a6ae7ed916312b41236fcef7e0af564ef934769"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.9.13"

[[deps.Tricks]]
git-tree-sha1 = "aadb748be58b492045b4f56166b5188aa63ce549"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.7"

[[deps.URIs]]
git-tree-sha1 = "074f993b0ca030848b897beff716d93aca60f06a"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.4.2"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unzip]]
git-tree-sha1 = "ca0969166a028236229f63514992fc073799bb78"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.2.0"

[[deps.Wayland_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "ed8d92d9774b077c53e1da50fd81a36af3744c1c"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.21.0+0"

[[deps.Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4528479aa01ee1b3b4cd0e6faef0e04cf16466da"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.25.0+0"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "93c41695bc1c08c46c5899f4fe06d6ead504bb73"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.10.3+0"

[[deps.XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "5be649d550f3f4b95308bf0183b82e2582876527"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.6.9+4"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4e490d5c960c314f33885790ed410ff3a94ce67e"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.9+4"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "12e0eb3bc634fa2080c1c37fccf56f7c22989afd"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.0+4"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fe47bd2247248125c428978740e18a681372dd4"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.3+4"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "b7c0aa8c376b31e4852b360222848637f481f8c3"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.4+4"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "0e0dc7431e7a0587559f9294aeec269471c991a4"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "5.0.3+4"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "89b52bc2160aadc84d707093930ef0bffa641246"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.7.10+4"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll"]
git-tree-sha1 = "26be8b1c342929259317d8b9f7b53bf2bb73b123"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.4+4"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "34cea83cb726fb58f325887bf0612c6b3fb17631"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.2+4"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "19560f30fd49f4d4efbe7002a1037f8c43d43b96"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.10+4"

[[deps.Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6783737e45d3c59a4a4c4091f5f88cdcf0908cbb"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.0+3"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "daf17f441228e7a3833846cd048892861cff16d6"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.13.0+3"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "926af861744212db0eb001d9e40b5d16292080b2"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.0+4"

[[deps.Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "0fab0a40349ba1cba2c1da699243396ff8e94b97"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll"]
git-tree-sha1 = "e7fd7b2881fa2eaa72717420894d3938177862d1"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "d1151e2c45a544f32441a567d1690e701ec89b00"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "dfd7a8f38d4613b6a575253b3174dd991ca6183e"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.9+1"

[[deps.Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "e78d10aab01a4a154142c5006ed44fd9e8e31b67"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.1+1"

[[deps.Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "4bcbf660f6c2e714f87e960a171b119d06ee163b"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.2+4"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "5c8424f8a67c3f2209646d4425f3d415fee5931d"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.27.0+4"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "79c31e7844f6ecf779705fbc12146eb190b7d845"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.4.0+3"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+0"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "49ce682769cd5de6c72dcf1b94ed7790cd08974c"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.5+0"

[[deps.fzf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "868e669ccb12ba16eaf50cb2957ee2ff61261c56"
uuid = "214eeab7-80f7-51ab-84ad-2988db7cef09"
version = "0.29.0+0"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3a2ea60308f0996d26f1e5354e10c24e9ef905d4"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.4.0+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.7.0+0"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "94d180a6d2b5e55e447e2d27a29ed04fe79eb30c"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.38+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "b910cb81ef3fe6e78bf6acee440bda86fd6ae00c"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+1"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.48.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+0"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"

[[deps.xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "9ebfc140cc56e8c2156a15ceac2f0302e327ac0a"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "1.4.1+0"
"""

# ╔═╡ Cell order:
# ╠═c6e50d2a-df8e-11ed-2859-b1d221d06c6d
# ╟─bfe253c8-3ab4-4629-9a6b-423a64d8fe32
# ╟─5e08403b-2475-45d2-950a-ab4cf2aada5a
# ╟─7971c689-4438-47ff-a896-6e61fc62d10b
# ╟─9000f45e-0f28-4b72-ba19-34dd944b274b
# ╟─477f8dbf-6797-4e7c-91be-31387f82ece7
# ╟─c5fc190c-63a1-4d94-ac29-f56d0556452f
# ╟─74482089-10fe-446b-b3f6-dc1b81b1a424
# ╠═9381e183-5b4e-489f-a109-4e606212986e
# ╠═45b0255d-72fd-4fa7-916a-4f73e730f4d5
# ╠═6c1a13d3-9089-4b54-a0e5-a02fb5fdf4a1
# ╠═a8b50581-e5f3-449e-803e-ab31e6e0b812
# ╟─85e2ada1-0adc-41a8-ab34-8043379ca0a4
# ╠═2c9a233f-3a96-43dc-b783-b82642a82590
# ╠═d9cb7c74-007d-4229-a576-a7a41fff565b
# ╟─01b6b048-6bd6-4c5a-8586-066cecf3ed51
# ╟─781d041c-1e4d-4354-b240-12511207bde0
# ╠═cb1a9949-59e1-4ccb-8efc-aa2ffbadaab2
# ╟─c9d3b75e-e1ff-4ad6-9c66-6a1b89a1b426
# ╠═8cb62a79-7dbc-4c94-ae7b-2e2cc12764f4
# ╟─d5457335-bc65-4bf1-b6ed-796dd5e2ab69
# ╠═96ff4afb-fe7f-471a-b15e-26676c600090
# ╟─14d4e0a3-8db0-4c57-bb33-497e1bd3c64c
# ╠═c1640ff6-3047-42a5-a5cd-d0c77aa41179
# ╟─6749ddf8-633b-498e-aecb-9e1592050ed2
# ╟─52a70be4-fb8c-40d8-9c7a-226649ada6e3
# ╠═fbc24a7c-48a0-43cc-9dd2-440acfb41c39
# ╟─a810cf76-2c64-457c-b5ea-eaa8bf4b1d42
# ╠═8a73d154-236d-4660-bb21-24681ed7d315
# ╟─408dd6d8-cb5f-49ce-944b-50a0d9cebef5
# ╠═fb472969-3c3c-4787-8cf1-296f2c13ddf5
# ╟─9373df69-ba17-46e0-a48a-ab1ca7dc3a9f
# ╠═87667a9e-02aa-4104-b5a0-0f6b9e98ba96
# ╟─b0b1f14d-4fbd-4995-845f-f19990460329
# ╠═1902c1f1-1246-4ab3-88e6-35619d685cdd
# ╟─3c63b27b-76e3-4edc-9b56-345738b97c41
# ╠═f871da95-6710-4c0f-a3a1-890dd59a41a1
# ╟─dc2c58de-98b9-4621-b465-064e8ab3caf1
# ╠═079cd7ef-632d-41d0-866d-6678808a8f4c
# ╟─dab1fe9c-20a4-4376-beaf-02b5292ca7cd
# ╟─24de2bcb-cf9d-44f7-b1d7-f80ae8c08ed1
# ╠═ed704f46-779c-4369-8a3b-d3e8cf0f4dd1
# ╟─e3e78fb1-00aa-4399-8330-1d4a08742b42
# ╠═6550261c-a3b8-40bc-a4ac-c43ae33215ca
# ╟─be042373-ed3e-4e2e-b714-b4f9e5964b57
# ╟─b2818ed9-6ef8-4398-a9d4-63b1d399169c
# ╟─8fed847c-93bc-454b-94c7-ba1d13c73b04
# ╠═1a129b6f-74f0-404c-ae4f-3ae39c8431aa
# ╠═a5ae35dc-cc4b-48bd-869e-37823b8073d2
# ╟─baca3b20-16ac-4e37-a2bb-7512d1c99eb8
# ╟─e7ca9061-64dc-44ef-854e-45b8015abad1
# ╠═59bcc9bf-276c-47e1-b6a9-86f90571c0fb
# ╟─14a209dd-be4c-47f0-a343-1cfb97b7d04a
# ╟─5c221210-e1df-4015-b959-6d330b47be29
# ╟─7c20c3ab-b0ae-48fc-b2f0-9cde30559bf5
# ╟─e02d0dd5-6bab-4548-8bbe-d9b1759688c5
# ╟─c417e618-41c2-454c-9b27-470988215d48
# ╠═8950aa50-22b2-4299-83b2-b9abfd1d5303
# ╠═30502079-9684-4144-8bcd-a70f2cb5928a
# ╟─ca825009-564e-43e0-9014-cce87c46533b
# ╟─8bd60367-2007-4d50-9d25-c12acd73be96
# ╠═f1cea551-4feb-44b4-a77e-03621c9b37b9
# ╟─4c8259f1-d3ae-4400-93cb-0a09b22a14ae
# ╟─a3677e9f-837b-4ba0-a29f-e60bf3712323
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
