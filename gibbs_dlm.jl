### A Pluto.jl notebook ###
# v0.19.26

using Markdown
using InteractiveUtils

# ╔═╡ ee858a0c-1414-11ee-3b47-2fb4b9112c53
begin
	using Distributions, Plots, Random
	plotly()
	using LinearAlgebra
	using StatsFuns
	using PlutoUI
	using MCMCChains
	using Statistics
	using DataFrames
	using StatsPlots
	using SpecialFunctions
end

# ╔═╡ 72f82a78-828a-42f2-9b63-9950af4c7be3
using PDMats

# ╔═╡ 6c0ecec8-afdc-4072-9dac-4658af3706d5
TableOfContents()

# ╔═╡ 73e449fb-81d2-4a9e-a89d-38909093863b
md"""
# Gibbs sampling analog to VBEM-DLM
"""

# ╔═╡ e1f22c73-dee8-4507-af03-3d2d0ceb9011
# A -> transition matrix, C -> emission matrix, Q -> process noise, Q -> observation noise, μ_0, Σ_0 -> auxiliary hidden state x_0
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

# ╔═╡ 544ac3d9-a2b8-4950-a501-40c14c84b2d8
function error_metrics(true_means, smoothed_means)
    T = size(true_means)[2]
    mse = sum((true_means .- smoothed_means).^2) / T
    mad = sum(abs.(true_means .- smoothed_means)) / T

	# mean squared error (MSE), mean absolute deviation (MAD)
    return mse, mad
end

# ╔═╡ 46d87386-7c36-486f-ba59-15d71e88869c
md"""
Establish **ground-truth** and test data
"""

# ╔═╡ e1bd9dd3-855e-4aa6-91aa-2695da07ba48
begin
	A = [0.8 -0.1; 0.1 0.9]
	C = [1.0 0.0; 0.0 1.0]
	Q = Diagonal([0.5, 0.5])
	R = Diagonal([0.1, 0.1])
	T = 500
	Random.seed!(111)
	μ_0 = [0.0, 0.0]
	Σ_0 = Diagonal([1.0, 1.0])
	y, x_true = gen_data(A, C, Q, R, μ_0, Σ_0, T)
end

# ╔═╡ df41dbec-6f39-437c-9f59-8a74a7f5a8dd
md"""
Some assumptions to keep in-line with VBEM `Beale 2003 Thesis`:

	Row-wise MVN Prior: A row-wise MVN prior for A or C assumes that the elements of each row are jointly normally distributed, but it treats each row as independent from the others. This means that there are dependencies between the elements within each row, but not between different rows. This is a simpler assumption that can be easier to handle computationally, but it may not capture all the relationships in the data if there are important dependencies between different rows.

	Diagonal Covariance Matrix: If R is a diagonal matrix, then the noise in different dimensions of the observations is assumed to be uncorrelated. This means that high (or low) noise values in one dimension are independent of the noise values in other dimensions. This is a simpler assumption that can be easier to handle computationally, but it may not capture all the noise structures in the data if there are important correlations between different dimensions.

	Known and Fixed Q: Q is the identity matrix. This is possible since noise co-varaince can be incorportated in the state dynamics governed by A.
"""

# ╔═╡ fbf64d6a-0cc7-4150-9891-e659f43a3b39
md"""
## Sample A
"""

# ╔═╡ cfe224e3-dbb3-42bc-ac1b-b96fea5da00d
md"""

For sampling $A$, we need according to Gibbs sampling: 

$$\log p(A∣x_{1:T}, y_{1:T}, C, R, Q)$$

$$= \log p(A∣x_{1:T}, Q)$$

The state transition model in a multivariate dynamic linear model (DLM) is given by:

$$x_t = A x_{t−1} + w_t, w_t \sim \mathcal N (\mathbf{0}, Q)$$

The likelihood of the states $x_{1:T}$ given $A$ and the previous states $x_{1:T−1}$ is a product of MVNs:

$$p(x_{1:T}∣ A, Q)= \prod_{t=2}^T \mathcal N(x_t|A x_{t−1}, Q)$$

Taking the log likelihood and keeping only terms that relate to $A$:

$$\log p(x_{1:T}∣ A, Q) \propto -\frac{1}{2} \sum_{t=2}^T (x_t − Ax_{t−1})^\top Q^{−1}(x_t − A x_{t−1})$$

Given $A$ has row-wise MVN prior $\mathcal N(μ_a, Σ_a)$, we can express the log prior as:

$$\log p(A) \propto-\frac{1}{2} \sum_{i=1}^K (A_i − μ_a)^\top Σ_a^{-1} (A_i - μ_a)$$

Adding the log-likelihood and the log-prior, we obtain the log-posterior for sampling $A$:

$$\log p(A∣x_{1:T}, Q) = \log p(x_{1:T}∣ A, Q) + \log p(A) + C$$

"""

# ╔═╡ fc535acb-afd6-4f2a-a9f1-15dc83e4a53c
function sample_A(Xs, μ_A, Σ_A, Q)
    K, T = size(Xs)
	A = zeros(K, K)
	Σ_A_inv = inv(Σ_A)
	for i in 1:K
     	Σ_post = inv(Σ_A_inv + (Xs[:, 1:T-1] * Xs[:, 1:T-1]') ./ Q[i, i])
        μ_post = Σ_post * (Σ_A_inv * μ_A + (Xs[:, 1:T-1] * Xs[i, 2:T]) ./ Q[i, i])
		A[i, :] = rand(MvNormal(μ_post, Symmetric(Σ_post)))
    end
    return A
end

# ╔═╡ e9318f52-e918-42c6-9aa9-45a39ad73ec7
let
	μ_A = zeros(2)
	Σ_A = Matrix{Float64}(I, 2, 2)
	sample_A(x_true, μ_A, Σ_A, Q), A
end

# ╔═╡ bb13a886-0877-42fb-876e-38709f041d65
md"""
## Sample C
"""

# ╔═╡ 3308752f-d770-4951-9a21-73f1c3886df4
md"""
For sampling $C$, 
"""

# ╔═╡ 4b9cad0a-7ec4-4a58-bf4c-4f103371de33
function sample_C(Xs, Ys, μ_C, Σ_C, R)
    P, T = size(Ys)
	K, _ = size(Xs)
    C_sampled = zeros(P, K)
	
    for i in 1:P
        Y = Ys[i, :]
        Σ_C_inv = inv(Σ_C)
        Σ_post = inv(Σ_C_inv + (Xs * Xs') ./ R[i, i])
        μ_post = Σ_post * (Σ_C_inv * μ_C + Xs * Y / R[i, i])
        C_sampled[i, :] = rand(MvNormal(μ_post, Symmetric(Σ_post)))
    end
    return C_sampled
end

# ╔═╡ 9e73e982-a4ae-4e9b-9650-3cf7c519657c
let
	μ_C = zeros(2)
	Σ_C = Matrix{Float64}(I, 2, 2)
	sample_C(x_true, y, μ_C, Σ_C, R), C
end

# ╔═╡ a41bd4a5-a7be-48fe-a222-6e8b3cf98dec
md"""
## Sample R
"""

# ╔═╡ d8c05722-a79b-4132-b1c2-982ef39af257
function sample_R(Xs, Ys, C, α_ρ, β_ρ)
    P, T = size(Ys)
    ρ_sampled = zeros(P)
    for i in 1:P
        Y = Ys[i, :]
        α_post = α_ρ + T / 2
        β_post = β_ρ + 0.5 * sum((Y' - C[i, :]' * Xs).^2)
		
        ρ_sampled[i] = rand(Gamma(α_post, 1 / β_post))
    end
    return diagm(1 ./ ρ_sampled)
end

# ╔═╡ 65b7e5f4-aff8-4671-9a3a-7aeebef6b83e
let
	α = 0.01
	β = 0.01
	sample_R(x_true, y, C, α, β), R
end

# ╔═╡ e2a46e7b-0e83-4275-bf9d-bc1a84fa2e87
md"""
## Gibbs sampling
"""

# ╔═╡ 6a4af386-bfe0-48bb-8d40-300e02680703
md"""
VBEM Analog with Beale Chap 5
"""

# ╔═╡ 56cad0cb-352b-4612-b3a3-ddb34de607ad
md"""
Analyse Gibbs results with Chains
"""

# ╔═╡ 0a9c1721-6901-4dc1-a93d-8d8e18f7f375
md"""
# TO-DO:

- Single-move sampler of X v.s. FFBS for latent state $x$ inference
- Gibbs sampling for infering unknown $Q$ and $R$ with **known** A, C - cf `DLM with R`
- VBEM for infering unknown $Q$ and $R$ with **known** A, C - cf `Beale 2003 Thesis`
- Design tests/experiments to verify and compare both approaches
"""

# ╔═╡ 91892a1b-b55c-4f83-91b3-dab4132b1863
md"""
## Closed-form Bayesian Inference
"""

# ╔═╡ 9d2a6daf-2b06-409e-b034-6e787e64fea8
md"""
According to the formulation in `DLM with R` _4.3_:

- Known A, C
- Q, R have identitcal scaling factor σ², which is Unknown
- This allows for conjugate inference (closed-form)
"""

# ╔═╡ 3e3d4c01-8a97-4ede-a341-27ab6fd07b95
md"""
Choosing a Normal Gamma prior for $\frac{1}{σ^2}$ and initial state $x_0$

Posterior mean of $σ^2$ can be computed as $\frac{β_T}{(α_T−1)}$
"""

# ╔═╡ 369366a2-b4c7-44b5-8f64-d11616e99290
function kf_ng_σ(Ys, A, C, Q̃, R̃, μ_0, Σ_0, α_0, β_0)
    p, T = size(Ys)
    d, _ = size(A)
	
    # Initialize
    m = zeros(d, T)
    P = zeros(d, d, T)
	
    m[:, 1] = μ_0
    P[:, :, 1] = Σ_0
	
    α_T = α_0
    β_T = β_0
	
    # Kalman filter (Prep 4.1)
    for t in 2:T
        # Prediction
        a_t = A * m[:, t-1]
        P_t = A * P[:, :, t-1] * A' + Q̃
        
		# Update
        f_t = C * a_t
        S_t = C * P_t * C' + R̃

		# filter 
        m[:, t] = a_t + P_t * C' * inv(S_t) * (Ys[:, t] - f_t)
        P[:, :, t] = P_t - P_t * C' * inv(S_t) * P_t'
        
		# Update Normal-Gamma parameters
        α_T += p / 2
        β_T += 0.5 * (Ys[:, t] - f_t)' * inv(S_t) * (Ys[:, t] - f_t)
    end
	
    return m[:, T], P[:, :, T], α_T, β_T
end

# ╔═╡ 57c87102-04bc-4414-9258-e2220f9d2e22
let
	A = [1.0 0.0; 0.0 1.0]
	C = [1.0 0.0; 0.0 1.0]
	Q = Diagonal([1.0, 1.0])
	R = Diagonal([0.1, 0.1])
	T = 500
	Random.seed!(888)
	μ_0 = [0.0, 0.0]
	Σ_0 = Diagonal([1.0, 1.0])

	y, x_true = gen_data(A, C, Q, R, μ_0, Σ_0, T)
	
	σ_true = 2.0
	println("true scale: ", σ_true)
	Q̃ = Q ./ σ_true
	R̃ = R ./ σ_true
	Σ_0̃  = Σ_0 ./ σ_true
	
	 _, _, α_t, β_t = kf_ng_σ(y, A, C, Q̃, R̃, μ_0, Σ_0̃ , 0.1, 0.1)
	println("unknown_scale_mean: ", β_t / (α_t - 1))
	println("unknown_scale_variance: ", β_t^2 / ((α_t - 1)^2 * (α_t - 2)))	
end

# ╔═╡ 11700202-40fe-408b-a8b8-5c073daec12d
md"""
## FFBS
"""

# ╔═╡ 8c9357c8-8339-4889-8a91-b62e542f0407
md"""
DLM with R< _4.4_: Simulation-based Bayesian Inference.

Joint posterior of all unknown variables, where $ψ$ is model specific parameters: 

$$p(ψ, x_{0: T}|y_{1:T})$$

Gibbs sampling involves two main full conditionals:

$$p(x_{0: T} | y_{1:T}, ψ)$$

$$p(ψ | x_{0: T}, y_{1:T})$$

The first can be obtained via **FFBS**, essentially a simulation version of the smoothing recursions

$\begin{align}
p(x_{0: T} | y_{1:T}) &= \prod_{t=0}^T p(x_t| x_{t+1:T}, y_{1:T}) \\
&= \prod_{t=0}^T p(x_t| x_{t+1}, y_{1:T})
\end{align}$

Numerical stability? - Symmetric() on co-variance matrices
"""

# ╔═╡ a9621810-e0cb-4925-8b6a-726f78d13510
function ffbs_x(Ys, A, C, R, Q, μ_0, Σ_0)
	p, T = size(Ys)
    d, _ = size(A)
	
    # Initialize
    m = zeros(d, T)
	P = zeros(d, d, T)
	
	a = zeros(d, T)
	RR = zeros(d, d, T)
	X = zeros(d, T)

	a[:, 1] = A * μ_0
	P_1 = A * Σ_0 * A' + Q
	RR[:, :, 1] = P_1
	f_1 = C * a[:, 1]
    S_1 = C * P_1 * C' + R
    m[:, 1] = a[:, 1] + RR[:, :, 1] * C' * inv(S_1) * (Ys[:, 1] - f_1)
    P[:, :, 1] = RR[:, :, 1] - RR[:, :, 1] * C' * inv(S_1) * C * RR[:, :, 1]
		
		# Kalman filter (Prep 4.1)
    for t in 2:T
        # Prediction
        a[:, t] = A * m[:, t-1]
        P_t = A * P[:, :, t-1] * A' + Q
		RR[:, :, t] = P_t
		
		# Update
        f_t = C * a[:, t]
        S_t = C * P_t * C' + R

		# filter 
        m[:, t] = a[:, t] + RR[:, :, t] * C' * inv(S_t) * (Ys[:, t] - f_t)

       	Σ_t = RR[:, :, t] - RR[:, :, t] * C' * inv(S_t) * C * RR[:, :, t]
		P[:, :, t] = Σ_t
	end
	
		X[:, T] = rand(MvNormal(m[:, T], Symmetric(P[:, :, T])))
	
	# backward sampling
	for t in (T-1):-1:1
		h_t = m[:, t] +  P[:, :, t] * A' * inv(RR[:, :, t+1])*(X[:, t+1] - a[:, t+1])
		H_t = P[:, :, t] - P[:, :, t] * A' * inv(RR[:, :, t+1]) * A * P[:, :, t]
	
		X[:, t] = rand(MvNormal(h_t, Symmetric(H_t)))
	end

	# sample initial x_0
	h_0 = μ_0 + Σ_0 * A' * inv(RR[:, :, 1])*(X[:, 1] - a[:, 1])
	H_0 = Σ_0 - Σ_0 * A' * inv(RR[:, :, 1]) * A * Σ_0

	x_0 = rand((MvNormal(h_0, Symmetric(H_0))))

	return X, x_0
end

# ╔═╡ 36cb2dd6-19af-4a1f-aa19-7646c2c9cbab
let
	Random.seed!(177)
	xss = zeros(2, T, 3000)
	x_0s = zeros(2, 3000)
	for i in 1:3000
		x_ffbs, x_0 = ffbs_x(y, A, C, R, Matrix{Float64}(I, 2, 2), zeros(2), Matrix{Float64}(I, 2, 2))
		xss[:, :, i] = x_ffbs
		x_0s[:, i] = x_0
	end
	xs_m = mean(xss, dims=3)[:, :, 1]
	println("MSE, MAD of MCMC mean: ", error_metrics(x_true, xs_m))
	reshaped_samples = reshape(xss, 3000, 2*T, 1)
	chains = Chains(reshaped_samples)
	
	ess = ess(chains)
	rhat = rhat(chains)
	# Convert to DataFrame
	ess_df = DataFrame(ess)
	rhat_df = DataFrame(rhat)
	
	# Get the mean ESS and Rhat across all parameters
	mean_ess = mean(skipmissing(ess_df.ess))
	mean_rhat = mean(skipmissing(rhat_df.rhat))
	
	println("Mean ESS: $mean_ess")
	println("Mean Rhat: $mean_rhat")

end

# ╔═╡ a95ed94c-5fe2-4c31-a7a6-e45e841af528
let
	Random.seed!(177)
	mcmc = 1000
	xss = zeros(2, T, mcmc)
	x_0s = zeros(2, mcmc)
	for i in 1:mcmc
		x_ffbs, x_0 = ffbs_x(y, A, C, R, Matrix{Float64}(I, 2, 2), zeros(2), Matrix{Float64}(I, 2, 2))
		xss[:, :, i] = x_ffbs
	end
	
	# Select the first 50 time steps
	true_latent_50 = x_true[:, 1:50]
	sampled_latent_50 = xss[:, 1:50, :]
	
	# Create a new plot
	p = plot()
	
	# Plot the true latent states with a thick line
	plot!(p, true_latent_50[1, :], linewidth=1.5, alpha=5, label="True x_d1", color=:blue)
	plot!(p, true_latent_50[2, :], linewidth=1.5, alpha=5, label="True x_d2", color=:red)
	
	# Plot the sampled latent states with a thin line
	for i in 1:size(sampled_latent_50, 3)
	    plot!(p, sampled_latent_50[1, :, i], linewidth=0.1, alpha=0.1, label=false, color=:violet)
	    plot!(p, sampled_latent_50[2, :, i], linewidth=0.1, alpha=0.1, label=false, color=:orange)
	end
	p
end

# ╔═╡ fa0dd0fd-7b8a-47a4-bb22-c05c9b70bff3
md"""
## Compare with Single-move
""" 

# ╔═╡ e9f3b9e2-5689-40ce-b5b5-bc571ba35c10
md"""
Case at both end, $t=1$, $t=T$

$$p(x_1| y_1, x_2) ∝ p(x_2| x_1) p(y_1|x_1)$$

$$p(x_T| y_T, x_{t-1}) ∝ p(x_T| x_{T-1}) p(y_T|x_T)$$

General case:

$$p(x_i| y_i, x_{i-1}, x_{i+1}) ∝ p(x_i| x_{i-1}) p(x_{i+1}| x_i) p(y_i|x_i)$$

Derivation for the general case:

$$p(y_i|x_i) \sim \mathcal N(Cx_i, R)$$

$$p(x_i|x_{i-1}) \sim \mathcal N(Ax_{i-1}, Q)$$

$$p(x_{i+1}|x_{i}) \sim \mathcal N(Ax_i, Q)$$

"""

# ╔═╡ c9f6fad7-518c-442b-a385-e3fa74431cb1
function sample_x_i(y_i, x_i_minus_1, x_i_plus_1, A, C, Q, R)
    Σ_x_i_inv = C' * inv(R) * C + inv(Q) + A' * inv(Q) * A
	Σ_x_i = inv(Σ_x_i_inv)
    μ_x_i = Σ_x_i * (C' * inv(R) * y_i + inv(Q) * A * x_i_minus_1 + A' * inv(Q) * x_i_plus_1)
	
    return rand(MvNormal(μ_x_i, Symmetric(Σ_x_i)))
end

# ╔═╡ 3a8ceb49-403e-424f-bedb-49f5b01c8d7a
function sample_x_1(y_1, x_2, A, C, Q, R)
    Σ_x_1_inv = C' * inv(R) * C + A' * inv(Q) * A
	Σ_x_1 = inv(Σ_x_1_inv)
    μ_x_1 = Σ_x_1 * (C' * inv(R) * y_1 + A' * inv(Q) * x_2)
    return rand(MvNormal(μ_x_1, Symmetric(Σ_x_1)))
end

# ╔═╡ 0a609b97-7859-4053-900d-c1be5d61e68c
function sample_x_T(y_T, x_T_1, A, C, Q, R)
    Σ_x_T_inv = C' * inv(R) * C + inv(Q)
	Σ_x_T = inv(Σ_x_T_inv)
    μ_x_T = Σ_x_T * (C' * inv(R) * y_T + inv(Q) * A * x_T_1)
    return rand(MvNormal(μ_x_T, Symmetric(Σ_x_T)))
end

# ╔═╡ a8971bb3-cf38-4445-b66e-65ff35ca13ca
function single_move_sampler(Ys, A, C, Q, R, mcmc=2000)
    p, T = size(Ys)
    d, _ = size(A)
	xs = rand(d, T)
    # Sample each latent state one at a time

	xss = zeros(d, T, mcmc)
	xss[:, :, 1] = xs
	
    for m in 2:mcmc
		xs[:, 1] = sample_x_1(Ys[:, 1], xs[:, 2], A, C, Q, R)

		for i in 2:T-2
			xs[:, i] = sample_x_i(Ys[:, i], xs[:, i-1], xs[:, i+1], A, C, Q, R)
		end
		xs[:, T] = sample_x_T(Ys[:, T], xs[:, T-1], A, C, Q, R)
		xss[:, :, m] = xs
    end
	
    return xss
end

# ╔═╡ 120d3c31-bba9-476d-8a63-95cdf2457a1b
function gibbs_dlm(y, K, single_move=false, Q=Matrix{Float64}(I, K, K), mcmc=3000, burn_in=1500, thinning=1)
	P, T = size(y)
	n_samples = Int.(mcmc/thinning)
    A_samples = zeros(K, K, n_samples)
    C_samples = zeros(P, K, n_samples)
    R_samples = zeros(P, P, n_samples)
    Xs_samples = zeros(K, T, n_samples)
	
    # Initialize A, C, and R, using fixed prior 
    A_init = rand(MvNormal(zeros(K), Matrix{Float64}(I, K, K)), K)'
    C_init = rand(MvNormal(zeros(K), Matrix{Float64}(I, K, K)), P)'
	a = 0.1
	b = 0.1
	ρ = rand(Gamma(a, b), 2)
    R_init = Diagonal(1 ./ ρ)
    A = A_init
    C = C_init
    R = R_init
	μ₀, Σ₀ = vec(mean(y, dims=2)), Matrix{Float64}(I, K, K)
	
    for iter in 1:(mcmc + burn_in)
		Xs = missing
        # Sample latent states Xs
		if single_move # not working well in Gibbs
			Xs = single_move_sampler(y, A, C, Q, R, 1)[:, :, 1]
		else
        	Xs, _ = ffbs_x(y, A, C, R, Q, μ₀, Σ₀)
		end
		
        # Sample model parameters A, C, and R
        A = sample_A(Xs, zeros(K), Σ₀, Q)
        C = sample_C(Xs, y, zeros(P), Σ₀, R)
        R = sample_R(Xs, y, C, a, b)
        # Store samples after burn-in
        if iter > burn_in && mod(iter - burn_in, thinning) == 0
			index = div(iter - burn_in, thinning)
            A_samples[:, :, index] = A
            C_samples[:, :, index] = C
            R_samples[:, :, index] = R
            Xs_samples[:, :, index] = Xs
        end
    end
    return A_samples, C_samples, R_samples, Xs_samples
end

# ╔═╡ df166b81-a6c3-490b-8cbc-4061f19b750b
begin
	Random.seed!(199)
	A_samples, C_samples, R_samples, X_samples = gibbs_dlm(y, 2)
end

# ╔═╡ af9c5548-14f2-4771-84cf-bf93eebcd3f2
 chn_A = Chains(reshape(A_samples, 4, 3000)');

# ╔═╡ 779e0cef-0865-4087-b3d1-563aec15a734
describe(chn_A)

# ╔═╡ 611e868a-e808-4a1f-8dd3-2d7ef64e2984
chn_C = Chains(reshape(C_samples, 4, 3000)');

# ╔═╡ 69efb78d-1297-46b4-a6bb-218c07c9b2af
describe(chn_C)

# ╔═╡ 9f1a120d-80ac-46e0-ae7c-949d2f571b98
chn_R = Chains(reshape(R_samples, 4, 3000)');

# ╔═╡ f08a6391-24da-4d3e-8a3e-55806bb9efbb
describe(chn_R)

# ╔═╡ 39ecddfa-89a0-49ec-86f1-4794336215d0
let
	xs_m = mean(X_samples, dims=3)[:, :, 1]
	println("MSE, MAD of MCMC mean: ", error_metrics(x_true, xs_m))
	reshaped_samples = reshape(X_samples, 3000, 2*T, 1)
	xss_chains = Chains(reshaped_samples)
	
	ess = ess(xss_chains)
	rhat = rhat(xss_chains)
	# Convert to DataFrame
	ess_df = DataFrame(ess)
	rhat_df = DataFrame(rhat)
	
	# Get the mean ESS and Rhat across all parameters
	mean_ess = mean(skipmissing(ess_df.ess))
	mean_rhat = mean(skipmissing(rhat_df.rhat))
	
	println("X Mean ESS: $mean_ess")
	println("X Mean Rhat: $mean_rhat")
	ess, rhat

	#plot latent samples out against x_true
end

# ╔═╡ ad1eab82-0fa5-462e-ad17-8cb3b787aaf0
let
	Random.seed!(177)
	mcmc = 1000
	xss = single_move_sampler(y, A, C, Q, R, mcmc)

	# Select the first 100 time steps
	true_latent_50 = x_true[:, 1:50]
	sampled_latent_50 = xss[:, 1:50, :]
	
	# Create a new plot
	p = plot()
	
	# Plot the true latent states with a thick line
	plot!(p, true_latent_50[1, :], linewidth=1.5, alpha=5, label="True x_d1", color=:blue)
	plot!(p, true_latent_50[2, :], linewidth=1.5, alpha=5, label="True x_d2", color=:red)
	
	# Plot the sampled latent states with a thin line
	for i in 1:size(sampled_latent_50, 3)
	    plot!(p, sampled_latent_50[1, :, i], linewidth=0.1, alpha=0.1, label=false, color=:violet)
	    plot!(p, sampled_latent_50[2, :, i], linewidth=0.1, alpha=0.1, label=false, color=:orange)
	end
	p
end

# ╔═╡ 13007ba3-7ce2-4201-aa93-559fcbf9d12f
let
	Random.seed!(177)
	@time xss = single_move_sampler(y, A, C, Q, R, 3000)
	xs_m = mean(xss, dims=3)[:, :, 1]
	println("MSE, MAD of MCMC mean: ", error_metrics(x_true, xs_m))

	reshaped_samples = reshape(xss, 3000, 2*T, 1)
	chains = Chains(reshaped_samples)

	ess = ess(chains)
	rhat = rhat(chains)
	ess_df = DataFrame(ess)
	rhat_df = DataFrame(rhat)
	mean_ess = mean(skipmissing(ess_df.ess))
	mean_rhat = mean(skipmissing(rhat_df.rhat))
	
	println("Mean ESS: $mean_ess")
	println("Mean Rhat: $mean_rhat")
	ess_df, rhat_df
end

# ╔═╡ b4c11a46-438d-4653-89e7-bc2b99e84f48
md"""
## Multi-variate DLM with unknown $R, Q$

Assume the $P \times P$ observation precision matrix $Φ_0 = R^{-1}$ is Wishart distributed with prior:

$$Φ_0 \sim \mathcal Wi(v_0, S_0)$$

The Gibbs step full-conditional is 

$$p(Φ_0| ...) \sim \mathcal Wi(v_0 + T/2, S_0 + 1/2 * S S_y)$$

$$SS_y = \sum_{t=1}^T (y_t - C x_t)(y_t - C x_t)^\top$$
"""

# ╔═╡ 494eed09-a6e8-488b-bea2-55b7ddb37082
function sample_R_(y, x, C, v_0, S_0)
    T = size(y, 2)
	
    residuals = [y[:, t] - C * x[:, t] for t in 1:T]
	SS_y = sum([residuals[t] * residuals[t]' for t in 1:T])
	
    scale_posterior = S_0 + SS_y .* 0.5
    v_p = v_0 + 0.5 * T

	S_p = PDMat(Symmetric(inv(scale_posterior)))
	R⁻¹ = rand(Wishart(v_p, S_p))
    return inv(R⁻¹)
end

# ╔═╡ 060ae93d-12fe-47c6-abe1-ff7728bda572
let
	sample_R_(y, x_true, C, 3, Matrix{Float64}(0.01 * I, 2, 2)), R
end

# ╔═╡ 425782f9-4764-4880-ab72-3b481a2cf55a
md"""
Assume the $K \times K$ system precision matrix $Φ_1 = Q^{-1}$ is also Wishart distributed with prior:

$$Φ_1 \sim \mathcal Wi(v_1, S_1)$$

The Gibbs step full-conditional is 

$$p(Φ_1| ...) \sim \mathcal Wi(v_1 + T/2, S_1 + 1/2 * S S_1)$$

$$SS_1 = \sum_{t=1}^T (x_t - A x_t)(x_t - A x_t)^\top$$
"""

# ╔═╡ cc216d03-956c-45bb-a6ef-38bf79d6a597
function sample_Q_(x, A, v_1, S_1, x_0)
    T = size(x, 2)
	
    residuals = [x[:, t] - A * x[:, t-1] for t in 2:T]
	SS_1 = sum([residuals[t] * residuals[t]' for t in 1:T-1])
    scale_posterior = S_1 + SS_1 .* 0.5

	scale_posterior += (x[:, 1] - A * x_0) * (x[:, 1] - A * x_0)' .* 0.5
    v_p = v_1 + 0.5 * T
	S_p = PDMat(Symmetric(inv(scale_posterior)))

	Q⁻¹ = rand(Wishart(v_p, S_p))
    return inv(Q⁻¹)
end

# ╔═╡ 7dfa5042-227b-43aa-a55c-30decee08413
let
	sample_Q_(x_true, A, 3, Matrix{Float64}(0.01 * I, 2, 2), μ_0), Q
end

# ╔═╡ 68c26d99-8f54-4580-8357-5598eb1c8cdf
md"""
## Gibbs sampling
"""

# ╔═╡ f0dc526c-b221-4652-a877-58a959d97019
function gibbs_dlm_cov(y, A, C, mcmc=3000, burn_in=100, thinning=1)
	P, T = size(y)
	K = size(A, 2)
	
	μ_0 = vec(mean(y, dims=2)) 
	λ_0 = Matrix{Float64}(I, K, K)
	
	v_0 = P + 1.0 
	S_0 = Matrix{Float64}(0.01 * I, P, P)

	v_1 = K + 1.0
	S_1 = Matrix{Float64}(0.01 * I, K, K)
	
	# Initial values for the parameters
	R⁻¹ = rand(Wishart(v_0, inv(S_0)))
	Q⁻¹ = rand(Wishart(v_1, inv(S_1)))

	R, Q = inv(R⁻¹), inv(Q⁻¹)
	
	n_samples = Int.(mcmc/thinning)
	# Store the samples
	samples_X = zeros(n_samples, K, T)
	samples_Q = zeros(n_samples, K, K)
	samples_R = zeros(n_samples, P, P)
	
	# Gibbs sampler
	for i in 1:mcmc+burn_in
	    # Update the states
		x, x_0 = ffbs_x(y, A, C, R, Q, μ_0, λ_0)
		
		# Update the system noise
		Q = sample_Q_(x, A, v_1, S_1, x_0)
		
	    # Update the observation noise
		R = sample_R_(y, x, C, v_0, S_0)
	
	    # Store the samples
		if i > burn_in && mod(i - burn_in, thinning) == 0
			index = div(i - burn_in, thinning)
		    samples_X[index, :, :] = x
			samples_Q[index, :, :] = Q
		    samples_R[index, :, :] = R
		end
	end

	return samples_X, samples_Q, samples_R
end

# ╔═╡ ad6d0997-43c1-42f3-b997-765c594794b4
begin
	Random.seed!(123)
	Xs_samples, Qs_samples, Rs_samples = gibbs_dlm_cov(y, A, C, 3000, 1000, 1)
end

# ╔═╡ f0551220-d7c1-4312-b6c5-e0c432889494
begin
	Q_chain = Chains(reshape(Qs_samples, 3000, 4))
	R_chain = Chains(reshape(Rs_samples, 3000, 4))
	summarystats(Q_chain), summarystats(R_chain)
end

# ╔═╡ 0b4d5ac8-5a7b-4363-9dbe-2edc517708a0
md"""
Q, R sample means
"""

# ╔═╡ b337a706-cbbf-4acd-8a8f-26fdbc137e8e
begin
	Q_m = mean(Qs_samples, dims=1)[1, :, :]

	R_m = mean(Rs_samples, dims=1)[1, :, :]

	Q_m, R_m
end

# ╔═╡ ab8ec1ee-b28c-4010-9087-aaeb6a022fa9
begin
	xs_m = mean(Xs_samples, dims=1)[1, :, :]
	println("MSE, MAD of MCMC X mean: ", error_metrics(x_true, xs_m))
	println("MSE, MAD of MCMC X end: ", error_metrics(x_true, Xs_samples[end, :, :]))

	X_chain = Chains(reshape(Xs_samples, 3000, 2*T))
	x_ess = ess(X_chain)
	x_rhat = rhat(X_chain)
	# Convert to DataFrame
	ess_df = DataFrame(x_ess)
	rhat_df = DataFrame(x_rhat)
	
	# Get the mean ESS and Rhat across all parameters
	mean_ess = mean(skipmissing(ess_df.ess))
	mean_rhat = mean(skipmissing(rhat_df.rhat))
	
	println("Mean ESS: $mean_ess")
	println("Mean Rhat: $mean_rhat")
end

# ╔═╡ 282b71f4-4848-426d-b8fc-0e3656d01767
md"""
FFBS Sample Quality in Gibbs
"""

# ╔═╡ 2837effd-25f2-4f49-829e-8fc191db8460
let
	# Select the first 50 time steps
	true_latent_50 = x_true[:, 1:50]
	ffbs_sampled_latent_50 = Xs_samples[:, :, 1:50]
	
	# Create a new plot
	p = plot()
	
	# Plot the true latent states with a thick line
	plot!(p, true_latent_50[1, :], linewidth=1.5, alpha=5, label="True x_d1", color=:blue)
	plot!(p, true_latent_50[2, :], linewidth=1.5, alpha=5, label="True x_d2", color=:red)
	
	# Plot the sampled latent states with a thin line
	for i in 1:size(ffbs_sampled_latent_50, 1)
	    plot!(p, ffbs_sampled_latent_50[i, 1, :], linewidth=0.1, alpha=0.1, label=false, color=:violet)
	    plot!(p, ffbs_sampled_latent_50[i, 2, :], linewidth=0.1, alpha=0.1, label=false, color=:orange)
	end
	p
end

# ╔═╡ 0d8327f7-beb8-42de-ad0a-d7e2ebae81ac
md"""
Compare with single-move for sampling latent states
"""

# ╔═╡ 4baa0604-712a-448f-b3ee-56543bfc0d71
# ╠═╡ disabled = true
#=╠═╡
function gibbs_smx(y, A, C, mcmc=3000, burn_in=100, thinning=1)
	P, T = size(y)
	K = size(A, 2)
	
	μ_0 = zeros(K)  
	λ_0 = Matrix{Float64}(I, K, K)
	
	v_0 = P + 1.0 
	S_0 = Matrix{Float64}(0.01 * I, P, P)

	v_1 = K + 1.0
	S_1 = Matrix{Float64}(0.01 * I, K, K)
	
	# Initial values for the parameters
	R⁻¹ = rand(Wishart(v_0, inv(S_0)))
	Q⁻¹ = rand(Wishart(v_1, inv(S_1)))

	R, Q = inv(R⁻¹), inv(Q⁻¹)
	
	n_samples = Int.(mcmc/thinning)
	# Store the samples
	samples_X = zeros(n_samples, K, T)
	samples_Q = zeros(n_samples, K, K)
	samples_R = zeros(n_samples, P, P)
	
	# Gibbs sampler
	for i in 1:mcmc+burn_in

		x = single_move_sampler(y, A, C, Q, R, burn_in+1)[:, :, end]
		
		# Update the system noise
		Q = sample_Q_(x, A, v_1, S_1, μ_0)
		
	    # Update the observation noise
		R = sample_R_(y, x, C, v_0, S_0)
		
	    # Store the samples
		if i > burn_in && mod(i - burn_in, thinning) == 0
			index = div(i - burn_in, thinning)
		    samples_X[index, :, :] = x
			samples_Q[index, :, :] = Q
		    samples_R[index, :, :] = R
		end
	end
	return samples_X, samples_Q, samples_R
end
  ╠═╡ =#

# ╔═╡ de7046da-e361-41d0-b2d7-12439b571795
#=╠═╡
begin
	Random.seed!(99)
	Xs_samples_sm, Qs_samples_sm, Rs_samples_sm = gibbs_smx(y, A, C)
	mean(Qs_samples_sm, dims=1)[1, :, :], mean(Rs_samples_sm, dims=1)[1, :, :]
end
  ╠═╡ =#

# ╔═╡ 05828da6-bc3c-45de-b059-310159038d5d
md"""
similar to inference with unknown A, C, R, gibbs via single-move learning is poor compared to FFBS? 
"""

# ╔═╡ 69061e7f-8a6d-4fac-b187-4d6ff16cf777
#=╠═╡
let
	# Select the first 50 time steps
	true_latent_50 = x_true[:, 1:50]
	sm_sampled_latent_50 = Xs_samples_sm[:, :, 1:50]
	
	# Create a new plot
	p = plot()
	
	# Plot the true latent states with a thick line
	plot!(p, true_latent_50[1, :], linewidth=1.5, alpha=5, label="True x_d1", color=:blue)
	plot!(p, true_latent_50[2, :], linewidth=1.5, alpha=5, label="True x_d2", color=:red)
	
	# Plot the sampled latent states with a thin line
	for i in 1:size(sm_sampled_latent_50, 1)
	    plot!(p, sm_sampled_latent_50[i, 1, :], linewidth=0.1, alpha=0.1, label=false, color=:violet)
	    plot!(p, sm_sampled_latent_50[i, 2, :], linewidth=0.1, alpha=0.1, label=false, color=:orange)
	end
	p
end
  ╠═╡ =#

# ╔═╡ a9cba95e-9a9e-46c1-8f66-0a9b4ee0fcf0
#=╠═╡
let
	xs_m = mean(Xs_samples_sm, dims=1)[1, :, :]
	println("MSE, MAD of MCMC X mean: ", error_metrics(x_true, xs_m))
	println("MSE, MAD of MCMC X end: ", error_metrics(x_true, Xs_samples_sm[end, :, :]))

	X_chain = Chains(reshape(Xs_samples_sm, 3000, 2*T))
	x_ess = ess(X_chain)
	x_rhat = rhat(X_chain)
	ess_df = DataFrame(x_ess)
	rhat_df = DataFrame(x_rhat)
	
	mean_ess = mean(skipmissing(ess_df.ess))
	mean_rhat = mean(skipmissing(rhat_df.rhat))
	
	println("Mean ESS: $mean_ess")
	println("Mean Rhat: $mean_rhat")
end
  ╠═╡ =#

# ╔═╡ 6bea5f44-2abd-47b9-9db5-c5f70dd12c4f
md"""
# VBEM for general Q, R
"""

# ╔═╡ 37e0d499-bda5-4a5e-8b81-9b3f9384c2fb
md"""
Akin to the uni-variate case, we assume state transition $A$ and observation $C$ are fixed, and we want to infer the unknown matrices $Q$ and $R$

Full joint probability:

$$p(R, Q, x_{0:T}, y_{1:T}) = p(R) p(Q) p(x_0) \prod_{t=1}^{T} p(x_t | x_{t-1}, Q) \prod_{t=1}^{T} p(y_t | x_t, R)$$

$\begin{align}
\log p(y_{1:T}) &= \log \int p(R, Q, x_{0:T}, y_{1:T}) \ dR \ dQ \ dx_{0:T} \\
&\geq \int \ dR \ dQ \ dx_{0:T} \  q(R, Q, x_{0:T}) \log \frac{p(R, Q, x_{0:T}, y_{1:T})}{q(R, Q, x_{0:T})} \\
&= \mathcal F
\end{align}$

$$q(R, Q, x_{0:T}) = q(R) \ q(Q) \ q(x_{0:T})$$

"""

# ╔═╡ 456f0f6f-1c39-4a50-be3a-0db68d61a95f
md"""
## Prior specification

The prior for the precision matrices $R^{-1} = Λ_R$ and $Q^{-1} = Λ_Q$ can be expressed as follows:


$Λ_R \sim \mathcal{W}(W_R, \nu_R)$
$Λ_Q \sim \mathcal{W}(W_Q, \nu_Q)$

where $\mathcal{W}(W, \nu)$ denotes a Wishart distribution with scale matrix $W$ and degrees of freedom $\nu$. 


Log full joint:

$\ln p(Λ_Q, Λ_R, x_{0:T}, y_{1:T}) = \ln p(y_{1:T}|x_{0:T}, Λ_R) + \ln p(x_{0:T}| Λ_Q) + \ln p(Λ_R) + \ln p(Λ_Q)$

"""

# ╔═╡ 89fb5821-17c5-46be-8e3c-94439d295220
md"""
## VB-M 

$\ln q(Λ_Q, Λ_R) = E_q[\ln p(Λ_Q, Λ_R, x_{0:T}, y_{1:T})] + const$


### Update emission precision $Λ_R$
$\ln q(Λ_R) =  \langle \ln p(x_{0:T}, y_{1:T}, Λ_Q, Λ_R) \rangle_{\hat q(\mathbf{x}, Λ_Q)} + c$

We need only the terms that involve $Λ_R$ from the log full joint:

$\begin{align}
\ln q(Λ_R) &= \langle \ln p(Λ_R) + \ln p(y_{1:T}|x_{1:T}, Λ_R) \rangle_{\hat q(\mathbf{x}, Λ_Q)} + c \\

&= \ln p(Λ_R) + \langle \sum_{t=1}^T \ln  p(y_t|x_t, Λ_R) \rangle_{\hat q(\mathbf{x})} + c \\

&= \frac{\nu_R - d - 1}{2} \ln|\Lambda_R| -\frac{1}{2} \text{tr}(W_R^{-1} \Lambda_R) \\

&+ \frac{T}{2} \ln|\Lambda_R| -\frac{1}{2} \langle \sum_{t=1}^T (y_t - C x_t)^\top \Lambda_R (y_t - C x_t) \rangle_{\hat q(\mathbf{x})} 

\end{align}$ 

The variational posterior is therefore Wishart distributed:

$\ln q(Λ_R) = \ln \mathcal{W}(Λ_R; W_{Rn}, \nu_{Rn})$

$$W_{Rn}^{-1} = W_R^{-1} + \langle \sum_{t=1}^T (y_t - C x_t)(y_t - C x_t)^\top\rangle_{\hat q(\mathbf{x})}$$

$$\nu_{Rn} = \nu_R + T$$
"""

# ╔═╡ c9d18a43-5984-45f5-b558-368368212355
md"""
### Update emission precision $Λ_Q$
$\ln q(Λ_Q) =  \langle \ln p(x_{0:T}, y_{1:T}, Λ_Q, Λ_R) \rangle_{\hat q(\mathbf{x}, Λ_R)} + c$

We need only the terms that involve $Λ_Q$ from the log full joint:

$\begin{align}
\ln q(Λ_Q) &= \langle \ln p(Λ_Q) + \ln p(x_{0:T}| Λ_Q) \rangle_{\hat q(\mathbf{x}, Λ_R)} + c \\
&=  \ln p(Λ_Q) + \sum_{t=1}^{T} \langle \ln p(x_t| x_{t-1}, Λ_Q) \rangle_{\hat q(\mathbf{x})} + c \\
&= \frac{\nu_Q - d - 1}{2} \ln|\Lambda_Q| -\frac{1}{2} \text{tr}(W_Q^{-1} \Lambda_Q) \\
&+ \frac{T-1}{2} \ln |\Lambda_Q| - \langle \frac{1}{2} \sum_{t=1}^{T} (x_t - A x_{t-1})^\top \Lambda_Q (x_t - A x_{t-1}) \rangle_{\hat q(\mathbf{x})} + c
\end{align}$

The variational posterior is therefore also Wishart distributed:

$\ln q(Λ_Q) = \ln \mathcal{W}(Λ_Q; W_{Qn}, \nu_{Qn})$

$$W_{Qn}^{-1} = W_Q^{-1} + \langle \sum_{t=1}^{T} (x_t - A x_{t-1})(x_t - A x_{t-1})^\top\rangle_{\hat q(\mathbf{x})}$$
$$\nu_{Qn} = \nu_Q + T$$

"""

# ╔═╡ 99aab1db-2156-4bd4-9b54-3bb0d4a1620b
md"""
Hidden State Sufficient Statistics (HSS)
Calculated in E-step and used in M-step

$W_A = \sum_{t=1}^T \langle x_{t-1} x_{t-1}^\top \rangle = \sum_{t=1}^T Υ_{t-1,t-1} + ω_{t-1} ω_{t-1}^\top$

$S_A = \sum_{t=1}^T \langle x_{t-1} x_t^\top \rangle = \sum_{t=1}^T Υ_{t-1,t} + ω_{t-1} ω_t^\top$

$W_C = \sum_{t=1}^T \langle x_t x_t^\top \rangle = \sum_{t=1}^T Υ_{t,t} + ω_t ω_t^\top$

$S_C = \sum_{t=1}^T \langle x_t \rangle y_t^\top = \sum_{t=1}^T ω_ty_t^\top$

$W_Y = \sum_{t=1}^T \ y_t \ y_t^\top$

In $q(Λ_R)$ update:

$$E_q[\sum_{t=1}^{T} (y_t - Cx_t)(y_t - Cx_t)^\top] = W_Y - S_C C^\top - C S_C^\top + C W_C C^\top$$


In $q(Λ_Q)$ update:

$$E_q[\sum_{t=1}^{T} (x_t - Ax_{t-1})(x_t - Ax_{t-1})^\top] = W_C - S_A A^\top - A S_A^\top + A W_A A^\top$$
"""

# ╔═╡ 95ab5440-82dd-4fc4-be08-b1a851caf9ca
begin
	struct HSS
	    W_C::Array{Float64, 2}
	    W_A::Array{Float64, 2}
	    S_C::Array{Float64, 2}
	    S_A::Array{Float64, 2}
	end
	
	struct Prior
	    ν_R::Float64
	    W_R::Array{Float64, 2}
	    ν_Q::Float64
	    W_Q::Array{Float64, 2}
	    μ_0::Array{Float64, 1}
	    Λ_0::Array{Float64, 2}
	end

	struct Q_Wishart
		ν_R_q
		W_R_q
		ν_Q_q
		W_Q_q
	end
end

# ╔═╡ 0f2e6c3a-04c4-4f6b-8ccd-ed18c41e2bc4
function vb_m_step(y::Array{Float64, 2}, hss::HSS, prior::Prior, A::Array{Float64, 2}, C::Array{Float64, 2})
    _, T = size(y)
    
    # Compute the new parameters for the variational posterior of Λ_R
    ν_Rn = prior.ν_R + T
	W_Rn_inv = inv(prior.W_R) + y*y' - hss.S_C * C' - C * hss.S_C' + C * hss.W_C * C'
    
    # Compute the new parameters for the variational posterior of Λ_Q
    ν_Qn = prior.ν_Q + T
	W_Qn_inv = inv(prior.W_Q) + hss.W_C - hss.S_A * A' - A * hss.S_A' + A * hss.W_A * A'

	# Return expectations for E-step, Eq[R], E_q[Q], co-variance matrices
	return W_Rn_inv ./ ν_Rn, W_Qn_inv ./ ν_Qn, Q_Wishart(ν_Rn, inv(W_Rn_inv), ν_Qn, inv(W_Qn_inv))
	#return ν_Rn .* inv(W_Rn_inv), ν_Qn .* inv(W_Qn_inv) # precision matrices
end

# ╔═╡ f35c8af8-00b0-45ad-8910-04f656cecfa3
md"""
Test M-step update with true latent states, A, C assumed fixed

- A, C as identity matrices $I$

- A, C as general matrices
"""

# ╔═╡ aaf8f3a7-9549-4d02-ba99-e223fda5252a
let
	A = [1.0 0.0; 0.0 1.0]
	C = [1.0 0.0; 0.0 1.0]
	Q = Diagonal([1.0, 1.0])
	R = Diagonal([0.1, 0.1])
	T = 500
	Random.seed!(99)
	μ_0 = [0.0, 0.0]
	Σ_0 = Diagonal([1.0, 1.0])
	y, x_true = gen_data(A, C, Q, R, μ_0, Σ_0, T)
	D, T = size(y)
	K = size(A, 1)

	W_A = sum(x_true[:, t-1] * x_true[:, t-1]' for t in 2:T)
	S_A = sum(x_true[:, t-1] * x_true[:, t]' for t in 2:T)
	W_C = sum(x_true[:, t] * x_true[:, t]' for t in 1:T)
	S_C = sum(x_true[:, t] * y[:, t]' for t in 1:T)
	
	W_Q = Matrix{Float64}(I, K, K)
	W_R = Matrix{Float64}(I, D, D)
	prior = Prior(D + 1.0, W_R, K + 1.0, W_Q, zeros(K), Matrix{Float64}(I, K, K))
	hss = HSS(W_C, W_A, S_C, S_A)

	E_q_R, E_q_Q = vb_m_step(y, hss, prior, A, C), R, Q
end

# ╔═╡ a6470873-26b3-4981-80eb-12a59bd3695d
let
	A = [0.8 -0.05; 0.1 0.75]
	C = [1.0 0.0; 0.0 1.0]
	Q = Diagonal([0.5, 0.5])
	R = Diagonal([0.01, 0.01])
	T = 1000
	Random.seed!(111)
	μ_0 = [0.0, 0.0]
	Σ_0 = Diagonal([1.0, 1.0])
	y, x_true = gen_data(A, C, Q, R, μ_0, Σ_0, T)
	D, T = size(y)
	K = size(A, 1)

	W_A = sum(x_true[:, t-1] * x_true[:, t-1]' for t in 2:T)
	S_A = sum(x_true[:, t-1] * x_true[:, t]' for t in 2:T)
	W_C = sum(x_true[:, t] * x_true[:, t]' for t in 1:T)
	S_C = sum(x_true[:, t] * y[:, t]' for t in 1:T)
	
	W_Q = Matrix{Float64}(10*I, K, K)
	W_R = Matrix{Float64}(10*I, D, D)
	prior = Prior(D + 1.0, W_R, K + 1.0, W_Q, zeros(K), Matrix{Float64}(I, K, K))
	hss = HSS(W_C, W_A, S_C, S_A)

	vb_m_step(y, hss, prior, A, C), R, Q
end

# ╔═╡ c096bbab-4009-4995-8f45-dc7ffab7ccfa
md"""
## VB E

$\ln q(x_{0:T}) = E_q[\ln p(Λ_Q, Λ_R, x_{0:T}, y_{1:T})] + const$

This requires the expectations of M-step posterior:

$$E_q[Λ_R] = \nu_{Rn}W_{Rn}$$
$$E_q[R] = W_{Rn}^{-1} / \nu_{Rn}$$


$$E_q[Λ_Q] = \nu_{Qn}W_{Qn}$$
$$E_q[Q] = W_{Qn}^{-1} / \nu_{Qn}$$

Similar to uni-variate local level model, we use the forward-backward algorithm for $ω_t$, $Υ_{t, t}$
"""

# ╔═╡ 63bc1239-1a6a-4f3b-9d2c-9b904aec573c
md"""
### Forward
"""

# ╔═╡ 2f760ffd-1fc5-485b-8e7c-8b49ab7217e3
function forward_(y::Array{Float64, 2}, A::Array{Float64, 2}, C::Array{Float64, 2}, E_R::Array{Float64, 2}, E_Q::Array{Float64, 2}, prior::Prior)
    P, T = size(y)
    K = size(A, 1)
    
    # Unpack the prior parameters
    μ_0, Λ_0 = prior.μ_0, prior.Λ_0
    
    # Initialize the filtered means and covariances
    μ_f = zeros(K, T)
    Σ_f = zeros(K, K, T)
    f_s = zeros(K, T)
	S_s = zeros(K, K, T)
    # Set the initial filtered mean and covariance to their prior values
	A_1 = A * μ_0
	R_1 = A * inv(Λ_0) * A' + E_Q
	
	f_1 = C * A_1
	S_1 = C * R_1 * C' + E_R
	f_s[:, 1] = f_1
	S_s[:, :, 1] = S_1
	
    μ_f[:, 1] = A_1 + R_1 * C'* inv(S_1) * (y[:, 1] - f_1)
    Σ_f[:, :, 1] = R_1 - R_1*C'*inv(S_1)*C*R_1 
    
    # Forward pass (kalman filter)
    for t = 2:T
        # Predicted state mean and covariance
        μ_p = A * μ_f[:, t-1]
        Σ_p = A * Σ_f[:, :, t-1] * A' + E_Q

		# marginal y - normalization
		f_t = C * μ_p
		S_t = C * Σ_p * C' + E_R
		f_s[:, t] = f_t
		S_s[:, :, t] = S_t
		
		# Filtered state mean and covariance (2.8a - 2.8c DLM with R)
		μ_f[:, t] = μ_p + Σ_p * C' * inv(S_t) * (y[:, t] - f_t)
		Σ_f[:, :, t] = Σ_p - Σ_p * C * inv(S_t) * C * Σ_p
			
		# Kalman gain
        #K_t = Σ_p * C' / (C * Σ_p * C' + E_R)
        #μ_f[:, t] = μ_p + K_t * (y[:, t] - C * μ_p)
        #Σ_f[:, :, t] = (I - K_t * C) * Σ_p
    end
	
    log_z = sum(logpdf(MvNormal(f_s[:, i], Symmetric(S_s[:, :, i])), y[:, i]) for i in 1:T)
    return μ_f, Σ_f, log_z
end

# ╔═╡ 1f5d6cbd-43a2-4a17-996e-d4d2b1a7769c
begin
	### Beale VBEM
	struct Exp_ϕ
		A
		AᵀA
		C
		R⁻¹
		CᵀR⁻¹C
		R⁻¹C
		CᵀR⁻¹
		log_ρ
	end
	
	
	# hyper-prior parameters
	struct HPP
	    α::Vector{Float64} # precision vector for transition A
	    γ::Vector{Float64}  # precision vector for emission C
	    a::Float64 # gamma rate of ρ
	    b::Float64 # gamma inverse scale of ρ
	    μ_0::Vector{Float64} # auxiliary hidden state mean
	    Σ_0::Matrix{Float64} # auxiliary hidden state co-variance
	end
	
	function v_forward(ys::Matrix{Float64}, exp_np::Exp_ϕ, hpp::HPP)
	    D, T = size(ys)
	    K = size(exp_np.A, 1)
	
	    μs = zeros(K, T)
	    Σs = zeros(K, K, T)
		Σs_ = zeros(K, K, T)
		
		Qs = zeros(D, D, T)
		fs = zeros(D, T)
	
		# Extract μ_0 and Σ_0 from the HPP struct
	    μ_0 = hpp.μ_0
	    Σ_0 = hpp.Σ_0
	
		# initialise for t = 1
		Σ₀_ = inv(inv(Σ_0) + exp_np.AᵀA)
		Σs_[:, :, 1] = Σ₀_
		
	    Σs[:, :, 1] = inv(I + exp_np.CᵀR⁻¹C - exp_np.A*Σ₀_*exp_np.A')
	    μs[:, 1] = Σs[:, :, 1]*(exp_np.CᵀR⁻¹*ys[:, 1] + exp_np.A*Σ₀_*inv(Σ_0)μ_0)
	
		Qs[:, :, 1] = inv(exp_np.R⁻¹ - exp_np.R⁻¹C*Σs[:, :, 1]*exp_np.R⁻¹C')
		fs[:, 1] = Qs[:, :, 1]*exp_np.R⁻¹C*Σs[:, :, 1]*exp_np.A*Σ₀_*inv(Σ_0)*μ_0
			
		# iterate over T
		for t in 2:T
			Σₜ₋₁_ = inv(inv(Σs[:, :, t-1]) + exp_np.AᵀA)
			Σs_[:, :, t] = Σₜ₋₁_
			
			Σs[:, :, t] = inv(I + exp_np.CᵀR⁻¹C - exp_np.A*Σₜ₋₁_*exp_np.A')
	    	μs[:, t] = Σs[:, :, t]*(exp_np.CᵀR⁻¹*ys[:, t] + exp_np.A*Σₜ₋₁_*inv(Σs[:, :, t-1])μs[:, t-1])
	
			Qs[:, :, t] = inv(exp_np.R⁻¹ - exp_np.R⁻¹C*Σs[:, :, t]*exp_np.R⁻¹C')
			fs[:, t] = Qs[:, :, t]*exp_np.R⁻¹C*Σs[:, :, t]*exp_np.A*Σₜ₋₁_*inv(Σs[:, :, t-1])μs[:, t-1]
		end
	
		return μs, Σs, Σs_, fs, Qs
	end

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
	
		# for t = 1, this correspond to β(x_0), the probability of all the data given the setting of the auxiliary x_0 hidden state.
		Ψ₁ = inv(I + exp_np.CᵀR⁻¹C + inv(Ψs[:, :, 1]))
			
		Ψ_0 = inv(exp_np.AᵀA - exp_np.A'*Ψ₁*exp_np.A)
		η_0 = Ψs[:, :, 1]*exp_np.A'*Ψ₁*(exp_np.CᵀR⁻¹*ys[:, 1] + inv(Ψs[:, :, 1])ηs[:, 1])
		
		return ηs, Ψs, η_0, Ψ_0
	end

	function parallel_smoother(μs, Σs, ηs, Ψs, η_0, Ψ_0, μ_0, Σ_0)
		K, T = size(μs)
		Υs = zeros(K, K, T)
		ωs = zeros(K, T)
	
		# ending condition t = T
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
end;

# ╔═╡ 3100b411-e2de-4a43-be80-bcfcb42cef40
md"""
Test forward
"""

# ╔═╡ ff46a86f-5c18-4d83-8f0f-4d13fe7b3df2
let
	A = [0.8 -0.1; 0.3 0.6]
	C = [1.0 0.0; 0.0 1.0]
	Q = Diagonal([1.0, 1.0])
	R = Diagonal([0.01, 0.01])
	T = 500
	
	Random.seed!(111)
	μ_0 = [0.0, 0.0]
	Σ_0 = Diagonal([1.0, 1.0])
	y, x_true = gen_data(A, C, Q, R, μ_0, Σ_0, T)
	D, T = size(y)
	K = size(A, 1)

	# use fixed A, C, R from ground truth for the exp_np::Exp_ϕ
	e_A = A
	e_AᵀA = A'A
	e_C = C
	e_R⁻¹ = inv(R)
	e_CᵀR⁻¹C = e_C'*e_R⁻¹*e_C
	e_R⁻¹C = e_R⁻¹*e_C
	e_CᵀR⁻¹ = e_C'*e_R⁻¹
	e_log_ρ = log.(1 ./ diag(R))
	
	exp_np = Exp_ϕ(e_A, e_AᵀA, e_C, e_R⁻¹, e_CᵀR⁻¹C, e_R⁻¹C, e_CᵀR⁻¹, e_log_ρ)
	α = ones(K)
	γ = ones(K)
	a = 0.1
	b = 0.1
	μ_0 = zeros(K)
	Σ_0 = Matrix{Float64}(I, K, K)

	hpp = HPP(α, γ, a, b, μ_0, Σ_0)
	prior = Prior(3.0, Matrix{Float64}(I, K, K), 3.0, Matrix{Float64}(I, K, K), zeros(K), Matrix{Float64}(I, K, K))
    
	mean_f, cov_f, log_z = forward_(y, A, C, [0.01 0.0; 0.0 0.01], [1.0 0.0; 0.0 1.0], prior)

	m_f, c_f, _ , _, _ = v_forward(y, exp_np, hpp) #beale

	mean_f, m_f, cov_f, c_f, log_z
end

# ╔═╡ 1edc58de-db69-4dbd-bcc5-c72a07e841be
md"""
### Backward
"""

# ╔═╡ 9ca2a2bf-27a9-461b-ae74-1c28ac883168
function backward_(μ_f::Array{Float64, 2}, Σ_f::Array{Float64, 3}, A::Array{Float64, 2}, E_Q::Array{Float64, 2})
    K, T = size(μ_f)
    
    # Initialize the smoothed means, covariances, and cross-covariances
    μ_s = zeros(K, T)
    Σ_s = zeros(K, K, T)
    Σ_s_cross = zeros(K, K, T)
    
    # Set the final smoothed mean and covariance to their filtered values
    μ_s[:, T] = μ_f[:, T]
    Σ_s[:, :, T] = Σ_f[:, :, T]
    
    # Backward pass
    for t = T-1:-1:1
        # Compute the gain J_t
        J_t = Σ_f[:, :, t] * A' / (A * Σ_f[:, :, t] * A' + E_Q)

        # Update the smoothed mean μ_s and covariance Σ_s
        μ_s[:, t] = μ_f[:, t] + J_t * (μ_s[:, t+1] - A * μ_f[:, t])
        Σ_s[:, :, t] = Σ_f[:, :, t] + J_t * (Σ_s[:, :, t+1] - A * Σ_f[:, :, t] * A' - E_Q) * J_t'

        # Compute the cross covariance Σ_s_cross
        #Σ_s_cross[:, :, t+1] = inv(inv(Σ_f[:, :, t]) + A'*A) * A' * Σ_s[:, :, t+1]
		Σ_s_cross[:, :, t+1] = J_t * Σ_s[:, :, t+1]
    end
	
	Σ_s_cross[:, :, 1] = inv(I + A'*A) * A' * Σ_s[:, :, 1]
	#J_1 = I * A' / (A * I * A' + E_Q)
	#Σ_s_cross[:, :, 1] = J_1 * Σ_s[:, :, 1]
    return μ_s, Σ_s, Σ_s_cross
end

# ╔═╡ 098bc646-7300-4ac6-88af-08a599ba774a
md"""
Test backward, cross-covariance
""" 

# ╔═╡ 11796ca9-30e2-4ba7-b8dc-9a0eda90e14e
let
	A = [0.8 -0.1; 0.3 0.6]
	C = [1.0 0.0; 0.0 1.0]
	Q = Diagonal([1.0, 1.0])
	R = Diagonal([0.01, 0.01])
	T = 500
	
	Random.seed!(111)
	μ_0 = [0.0, 0.0]
	Σ_0 = Diagonal([1.0, 1.0])
	y, x_true = gen_data(A, C, Q, R, μ_0, Σ_0, T)
	D, T = size(y)
	K = size(A, 1)
	e_A = A
	e_AᵀA = A'A
	e_C = C
	e_R⁻¹ = inv(R)
	e_CᵀR⁻¹C = e_C'*e_R⁻¹*e_C
	e_R⁻¹C = e_R⁻¹*e_C
	e_CᵀR⁻¹ = e_C'*e_R⁻¹
	e_log_ρ = log.(1 ./ diag(R))
	exp_np = Exp_ϕ(e_A, e_AᵀA, e_C, e_R⁻¹, e_CᵀR⁻¹C, e_R⁻¹C, e_CᵀR⁻¹, e_log_ρ)
	α = ones(K)
	γ = ones(K)
	a = 0.1
	b = 0.1
	μ_0 = zeros(K)
	Σ_0 = Matrix{Float64}(I, K, K)
	hpp = HPP(α, γ, a, b, μ_0, Σ_0)


	prior = Prior(3.0, Matrix{Float64}(I, K, K), 3.0, Matrix{Float64}(I, K, K), zeros(K), Matrix{Float64}(I, K, K))
    
	μ_f, Σ_f = forward_(y, A, C, [0.01 0.0; 0.0 0.01], [1.0 0.0; 0.0 1.0], prior)
	μ_s, Σ_s, Σ_ss = backward_(μ_f, Σ_f, A, [1.0 0.0; 0.0 1.0])

	
	μs, Σs, Σs_, fs, Qs = v_forward(y, exp_np, hpp)
	ηs, Ψs, η_0, Ψ_0 = v_backward(y, exp_np)
	ωs, Υs, ω_0, Υ_0 = parallel_smoother(μs, Σs, ηs, Ψs, η_0, Ψ_0, μ_0, Σ_0)
	Σ_cross = v_pairwise_x(Σs_, exp_np::Exp_ϕ, Ψs)
	μ_s, ωs, Σ_s, Υs, Σ_ss, Σ_cross
end

# ╔═╡ e68dbe27-95ea-4710-9999-d2c4de0db914
function vb_e_step(y::Array{Float64, 2}, A::Array{Float64, 2}, C::Array{Float64, 2}, E_R::Array{Float64, 2}, E_Q::Array{Float64, 2}, prior::Prior)
    # Run the forward pass
    μ_f, Σ_f, log_Z = forward_(y, A, C, E_R, E_Q, prior)

    # Run the backward pass
    μ_s, Σ_s, Σ_s_cross = backward_(μ_f, Σ_f, A, E_Q)

    # Compute the hidden state sufficient statistics
    W_C = sum(Σ_s, dims=3)[:, :, 1] + μ_s * μ_s'
    W_A = sum(Σ_s[:, :, 1:end-1], dims=3)[:, :, 1] + μ_s[:, 1:end-1] * μ_s[:, 1:end-1]'
    S_C = y * μ_s'
    S_A = sum(Σ_s_cross, dims=3)[:, :, 1] + μ_s[:, 1:end-1] * μ_s[:, 2:end]'
    W_Y = y * y'

	# Return the hidden state sufficient statistics
    return HSS(W_C, W_A, S_C, S_A), log_Z
end

# ╔═╡ 488d1200-1ddf-4f06-9643-2eecb2072263
md"""
Test E-step
"""

# ╔═╡ 0591883c-49af-4201-b2f1-49f208506ece
let
	K = size(A, 1)
	prior = Prior(3.0, Matrix{Float64}(I, K, K), 3.0, Matrix{Float64}(I, K, K), zeros(K), Matrix{Float64}(I, K, K))

	hss_e = vb_e_step(y, A, C, [0.01 0.0; 0.0 0.01], [1.0 0.0; 0.0 1.0], prior)

	W_A = sum(x_true[:, t-1] * x_true[:, t-1]' for t in 2:T)
	S_A = sum(x_true[:, t-1] * x_true[:, t]' for t in 2:T)
	W_C = sum(x_true[:, t] * x_true[:, t]' for t in 1:T)
	S_C = sum(x_true[:, t] * y[:, t]' for t in 1:T)
	hss_t = HSS(W_C, W_A, S_C, S_A)
	hss_t,hss_e
end

# ╔═╡ 0dfa4d60-577a-4631-bd24-c05aee2969d0
function vbem_(y::Array{Float64, 2}, A::Array{Float64, 2}, C::Array{Float64, 2}, prior::Prior, max_iter=300)
	
	hss = HSS(ones(size(A)), ones(size(A)), ones(size(C)), ones(size(A)))
	E_R, E_Q  = missing, missing
	
	for i in 1:max_iter
		E_R, E_Q, _ = vb_m_step(y, hss, prior, A, C)
				
		hss, _ = vb_e_step(y, A, C, E_R, E_Q, prior)
	end

	return E_R, E_Q
end

# ╔═╡ bb26ac74-da64-47be-a49a-4519101cffce
md"""
### KL Divergence 
"""

# ╔═╡ dd8f1c15-8915-4503-85f0-d3378f8e4751
function kl_Wishart(ν_q, S_q, ν_0, S_0)
	k = size(S_0, 1)
	term1 = (ν_0 - ν_q)*k*log(2) + ν_0*logdet(S_0) - ν_q*logdet(S_q) + sum(loggamma((ν_0 + 1 - i)/2) for i in 1:k) - sum(loggamma((ν_q + 1 - i)/2) for i in 1:k)
	
    term2 = (ν_q - ν_0) * sum(digamma((ν_q + 1 - i)/2) + k*log(2) + logdet(S_q) for i in 1:k)
	
    term3 = ν_q * tr(inv(S_0) * S_q - I)
    return 0.5 * (term1 + term2 + term3) 
end

# ╔═╡ 37e01d43-b804-4736-8d91-fb9c7e0ab493
let
	A = [1.0 0.0; 0.0 1.0]
	C = [1.0 0.0; 0.0 1.0]
	Q = Diagonal([1.0, 1.0])
	R = Diagonal([0.2, 0.2])
	T = 500
	Random.seed!(111)
	μ_0 = [0.0, 0.0]
	Σ_0 = Diagonal([1.0, 1.0])
	y, x_true = gen_data(A, C, Q, R, μ_0, Σ_0, T)
	D, T = size(y)
	K = size(A, 1)
	W_A = sum(x_true[:, t-1] * x_true[:, t-1]' for t in 2:T)
	S_A = sum(x_true[:, t-1] * x_true[:, t]' for t in 2:T)
	W_C = sum(x_true[:, t] * x_true[:, t]' for t in 1:T)
	S_C = sum(x_true[:, t] * y[:, t]' for t in 1:T)
	
	W_Q = Matrix{Float64}(I, K, K)
	W_R = Matrix{Float64}(I, D, D)
	prior = Prior(D + 1.0, W_R, K + 1.0, W_Q, zeros(K), Matrix{Float64}(I, K, K))
	hss = HSS(W_C, W_A, S_C, S_A)
	E_q_R, E_q_Q, Q_ = vb_m_step(y, hss, prior, A, C)

	kl_R = kl_Wishart(Q_.ν_R_q, Q_.W_R_q, prior.ν_R, prior.W_R)
	kl_Q = kl_Wishart(Q_.ν_Q_q, Q_.W_Q_q, prior.ν_Q, prior.W_Q)

	_ , _ , log_z = forward_(y, A, C, [0.01 0.0; 0.0 0.01], [1.0 0.0; 0.0 1.0], prior)

	println(log_z)
	println(kl_Q)
	println(kl_R)
	log_z - kl_Q - kl_R
end

# ╔═╡ 5e10db40-5c2e-41c3-a431-e0a4c81d2718
function vbem_his_plot(y::Array{Float64, 2}, A::Array{Float64, 2}, C::Array{Float64, 2}, prior::Prior, max_iter=100)
    P, T = size(y)
    K, _ = size(A)

    W_C = zeros(K, K)
    W_A = zeros(K, K)
    S_C = zeros(P, K)
    S_A = zeros(K, K)
    hss = HSS(W_C, W_A, S_C, S_A)
	E_R, E_Q  = missing, missing
	
    # Initialize the history of E_R and E_Q
    E_R_history = zeros(P, P, max_iter)
    E_Q_history = zeros(K, K, max_iter)

    # Repeat until convergence
    for iter in 1:max_iter
		E_R, E_Q, _ = vb_m_step(y, hss, prior, A, C)
				
		hss, _ = vb_e_step(y, A, C, E_R, E_Q, prior)

        # Store the history of E_R and E_Q
        E_R_history[:, :, iter] = E_R
        E_Q_history[:, :, iter] = E_Q
    end

	p1 = plot(title = "Learning of E_R")
    for i in 1:P
        plot!(10:max_iter, [E_R_history[i, i, t] for t in 10:max_iter], label = "E_R[$i, $i]")
    end

    p2 = plot(title = "Learning of E_Q")
    for i in 1:K
        plot!(10:max_iter, [E_Q_history[i, i, t] for t in 10:max_iter], label = "E_Q[$i, $i]")
    end
	
	plot(p1, p2, layout = (1, 2))
end

# ╔═╡ e1edfb29-8f82-418b-948b-5542fd6d5b24
md"""
## With Convergence Check
"""

# ╔═╡ 907e0fea-bad1-49f5-aa98-e2524e93e191
function vbem_c(y::Array{Float64, 2}, A::Array{Float64, 2}, C::Array{Float64, 2}, prior::Prior, max_iter=100, tol=1e-3)

	# different initialisation?
	#hss = HSS(ones(size(A)), ones(size(A)), ones(size(C)), ones(size(A)))
	D, _ = size(y)
	K, _ = size(A)
	hss = HSS(Matrix{Float64}(I, K, K), Matrix{Float64}(I, K, K), Matrix{Float64}(I, K, D), Matrix{Float64}(I, K, K))
	
	E_R, E_Q  = missing, missing
	elbo_prev = -Inf
	el_s = zeros(max_iter)
	for i in 1:max_iter
		E_R, E_Q, Q_Wi = vb_m_step(y, hss, prior, A, C)
				
		hss, log_Z = vb_e_step(y, A, C, E_R, E_Q, prior)

		kl_Wi = kl_Wishart(Q_Wi.ν_R_q, Q_Wi.W_R_q, prior.ν_R, prior.W_R) + kl_Wishart(Q_Wi.ν_Q_q, Q_Wi.W_Q_q, prior.ν_Q, prior.W_Q)
		elbo = log_Z - kl_Wi
		el_s[i] = elbo
		
		if abs(elbo - elbo_prev) < tol
			println("Stopped at iteration: $i")
			el_s = el_s[1:i]
            break
		end
		
        elbo_prev = elbo

		if (i == max_iter)
			println("Warning: VB have not necessarily converged at $max_iter iterations with tolerance $tol")
		end
	end

	return E_R, E_Q, el_s
end

# ╔═╡ e1676b43-b8f0-409f-a6c3-7f6c8852f7ae
md"""
A, C is identity matrix, converged elbo around $785$
"""

# ╔═╡ ffd75de1-b9fd-4883-810d-1d4f79775f0d
let
	# A, C identity matrix (cf. local level model)
	A = [1.0 0.0; 0.0 1.0]
	C = [1.0 0.0; 0.0 1.0]
	Q = Diagonal([1.0, 1.0])
	R = Diagonal([0.1, 0.1])
	T = 1000
	Random.seed!(133)
	μ_0 = [0.0, 0.0]
	Σ_0 = Diagonal([1.0, 1.0])
	y, x_true = gen_data(A, C, Q, R, μ_0, Σ_0, T)
	D, _ = size(y)
	K = size(A, 1)
	W_Q = Matrix{Float64}(10*I, K, K)
	W_R = Matrix{Float64}(10*I, D, D)
	prior = Prior(D + 1.0, W_R, K + 1.0, W_Q, zeros(K), Matrix{Float64}(I, K, K))
	vbem_his_plot(y, A, C, prior)
end

# ╔═╡ 9ebaf094-75ae-48fa-8c3a-280dfbf24dcd
let
	A = [1.0 0.0; 0.0 1.0]
	C = [1.0 0.0; 0.0 1.0]
	Q = Diagonal([1.0, 1.0])
	R = Diagonal([0.1, 0.1])
	T = 1000
	Random.seed!(133)
	μ_0 = [0.0, 0.0]
	Σ_0 = Diagonal([1.0, 1.0])
	y, x_true = gen_data(A, C, Q, R, μ_0, Σ_0, T)
	D, _ = size(y)
	K = size(A, 1)
	W_Q = Matrix{Float64}(10*I, K, K)
	W_R = Matrix{Float64}(10*I, D, D)
	prior = Prior(D + 1.0, W_R, K + 1.0, W_Q, zeros(K), Matrix{Float64}(I, K, K))
	@time R, Q, elbos = vbem_c(y, A, C, prior)
	p = plot(elbos, label = "elbo", title = "ElBO progression")
	R, Q, p
end

# ╔═╡ b29576d1-12ed-4346-8114-1e6575f3ee7f
md"""
Initial setup (Compare with Gibbs)
"""

# ╔═╡ 8443d615-15a4-4bc3-b40d-c103db279d70
A, C, Q, R

# ╔═╡ 4190732e-a3d8-4623-9e47-46b6385335a9
let
	A = [0.8 -0.1; 0.1 0.9]
	C = [1.0 0.0; 0.0 1.0]
	Q = Diagonal([0.5, 0.5])
	R = Diagonal([0.1, 0.1])
	T = 1000
	Random.seed!(77)
	μ_0 = [0.0, 0.0]
	Σ_0 = Diagonal([1.0, 1.0])
	y, x_true = gen_data(A, C, Q, R, μ_0, Σ_0, T)
	D, _ = size(y)
	K = size(A, 1)
	W_Q = Matrix{Float64}(100*I, K, K)
	W_R = Matrix{Float64}(100*I, D, D)
	prior = Prior(D + 1.0, W_R, K + 1.0, W_Q, zeros(K), Matrix{Float64}(I, K, K))
	vbem_his_plot(y, A, C, prior, 100)
end

# ╔═╡ d1a0c9e6-b4d2-411c-8634-88c05ac81eb2
let
	A = [0.8 -0.1; 0.1 0.9]
	C = [1.0 0.0; 0.0 1.0]
	Q = Diagonal([0.5, 0.5])
	R = Diagonal([0.1, 0.1])
	T = 500
	Random.seed!(77)
	μ_0 = [0.0, 0.0]
	Σ_0 = Diagonal([1.0, 1.0])
	y, x_true = gen_data(A, C, Q, R, μ_0, Σ_0, T)
	D, _ = size(y)
	K = size(A, 1)
	W_Q = Matrix{Float64}(100*I, K, K)
	W_R = Matrix{Float64}(100*I, D, D)
	prior = Prior(D + 1.0, W_R, K + 1.0, W_Q, zeros(K), Matrix{Float64}(I, K, K))
	@time R, Q, elbos = vbem_c(y, A, C, prior)
	p = plot(elbos, label = "elbo", title = "ElBO progression")
	R, Q, p
end

# ╔═╡ 1d5b1be7-b05b-466b-9580-67c69e60fc40
let	
	A = [0.8 -0.1; 0.1 0.9]
	C = [1.0 0.0; 0.0 1.0]
	Q = Diagonal([0.5, 0.5])
	R = Diagonal([0.1, 0.1])
	T = 500
	Random.seed!(77)
	μ_0 = [0.0, 0.0]
	Σ_0 = Diagonal([1.0, 1.0])
	y, x_true = gen_data(A, C, Q, R, μ_0, Σ_0, T)
	D, _ = size(y)
	K = size(A, 1)
	W_Q = Matrix{Float64}(100*I, K, K)
	W_R = Matrix{Float64}(100*I, D, D)
	prior = Prior(D + 1.0, W_R, K + 1.0, W_Q, zeros(K), Matrix{Float64}(I, K, K))
	vbem_(y, A, C, prior, 25)
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
MCMCChains = "c7f686f2-ff18-58e9-bc7b-31028e88f75d"
PDMats = "90014a1f-27ba-587c-ab20-58faa44d9150"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
StatsFuns = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
StatsPlots = "f3b207a7-027a-5e70-b257-86293d7955fd"

[compat]
DataFrames = "~1.5.0"
Distributions = "~0.25.96"
MCMCChains = "~6.0.3"
PDMats = "~0.11.17"
Plots = "~1.38.16"
PlutoUI = "~0.7.51"
SpecialFunctions = "~2.2.0"
StatsFuns = "~1.3.0"
StatsPlots = "~0.15.5"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.9.2"
manifest_format = "2.0"
project_hash = "75424a04a867dbd875891d24d5b22a0c728bd4fd"

[[deps.AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "16b6dbc4cf7caee4e1e75c49485ec67b667098a0"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.3.1"
weakdeps = ["ChainRulesCore"]

    [deps.AbstractFFTs.extensions]
    AbstractFFTsChainRulesCoreExt = "ChainRulesCore"

[[deps.AbstractMCMC]]
deps = ["BangBang", "ConsoleProgressMonitor", "Distributed", "LogDensityProblems", "Logging", "LoggingExtras", "ProgressLogging", "Random", "StatsBase", "TerminalLoggers", "Transducers"]
git-tree-sha1 = "87e63dcb990029346b091b170252f3c416568afc"
uuid = "80f14c24-f653-4e6a-9b94-39d6b0f70001"
version = "4.4.2"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "8eaf9f1b4921132a4cff3f36a1d9ba923b14a481"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.1.4"

[[deps.AbstractTrees]]
git-tree-sha1 = "faa260e4cb5aba097a73fab382dd4b5819d8ec8c"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.4.4"

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "76289dc51920fdc6e0013c872ba9551d54961c24"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.6.2"
weakdeps = ["StaticArrays"]

    [deps.Adapt.extensions]
    AdaptStaticArraysExt = "StaticArrays"

[[deps.ArgCheck]]
git-tree-sha1 = "a3a402a35a2f7e0b87828ccabbd5ebfbebe356b4"
uuid = "dce04be8-c92d-5529-be00-80e4d2c0e197"
version = "2.3.0"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.Arpack]]
deps = ["Arpack_jll", "Libdl", "LinearAlgebra", "Logging"]
git-tree-sha1 = "9b9b347613394885fd1c8c7729bfc60528faa436"
uuid = "7d9fca2a-8960-54d3-9f78-7d1dccf2cb97"
version = "0.5.4"

[[deps.Arpack_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "OpenBLAS_jll", "Pkg"]
git-tree-sha1 = "5ba6c757e8feccf03a1554dfaf3e26b3cfc7fd5e"
uuid = "68821587-b530-5797-8361-c406ea357684"
version = "3.5.1+1"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "66771c8d21c8ff5e3a93379480a2307ac36863f7"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.0.1"

[[deps.AxisArrays]]
deps = ["Dates", "IntervalSets", "IterTools", "RangeArrays"]
git-tree-sha1 = "1dd4d9f5beebac0c03446918741b1a03dc5e5788"
uuid = "39de3d68-74b9-583c-8d2d-e117c070f3a9"
version = "0.4.6"

[[deps.BangBang]]
deps = ["Compat", "ConstructionBase", "InitialValues", "LinearAlgebra", "Requires", "Setfield", "Tables"]
git-tree-sha1 = "e28912ce94077686443433c2800104b061a827ed"
uuid = "198e06fe-97b7-11e9-32a5-e1d131e6ad66"
version = "0.3.39"

    [deps.BangBang.extensions]
    BangBangChainRulesCoreExt = "ChainRulesCore"
    BangBangDataFramesExt = "DataFrames"
    BangBangStaticArraysExt = "StaticArrays"
    BangBangStructArraysExt = "StructArrays"
    BangBangTypedTablesExt = "TypedTables"

    [deps.BangBang.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    TypedTables = "9d95f2ec-7b3d-5a63-8d20-e2491e220bb9"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.Baselet]]
git-tree-sha1 = "aebf55e6d7795e02ca500a689d326ac979aaf89e"
uuid = "9718e550-a3fa-408a-8086-8db961cd8217"
version = "0.1.1"

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

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "e30f2f4e20f7f186dc36529910beaedc60cfa644"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.16.0"

[[deps.Clustering]]
deps = ["Distances", "LinearAlgebra", "NearestNeighbors", "Printf", "Random", "SparseArrays", "Statistics", "StatsBase"]
git-tree-sha1 = "42fe66dbc8f1d09a44aa87f18d26926d06a35f84"
uuid = "aaaa29a8-35af-508c-8bc3-b662a17a0fe5"
version = "0.15.3"

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
version = "1.0.5+0"

[[deps.CompositionsBase]]
git-tree-sha1 = "802bb88cd69dfd1509f6670416bd4434015693ad"
uuid = "a33af91c-f02d-484b-be07-31d278c5ca2b"
version = "0.1.2"

    [deps.CompositionsBase.extensions]
    CompositionsBaseInverseFunctionsExt = "InverseFunctions"

    [deps.CompositionsBase.weakdeps]
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.ConcurrentUtilities]]
deps = ["Serialization", "Sockets"]
git-tree-sha1 = "96d823b94ba8d187a6d8f0826e731195a74b90e9"
uuid = "f0e56b4a-5159-44fe-b623-3e5288b988bb"
version = "2.2.0"

[[deps.ConsoleProgressMonitor]]
deps = ["Logging", "ProgressMeter"]
git-tree-sha1 = "3ab7b2136722890b9af903859afcf457fa3059e8"
uuid = "88cd18e8-d9cc-4ea6-8889-5259c0d15c8b"
version = "0.1.2"

[[deps.ConstructionBase]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "738fec4d684a9a6ee9598a8bfee305b26831f28c"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.5.2"
weakdeps = ["IntervalSets", "StaticArrays"]

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseStaticArraysExt = "StaticArrays"

[[deps.Contour]]
git-tree-sha1 = "d05d9e7b7aedff4e5b51a029dced05cfb6125781"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.2"

[[deps.Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[deps.DataAPI]]
git-tree-sha1 = "8da84edb865b0b5b0100c0666a9bc9a0b71c553c"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.15.0"

[[deps.DataFrames]]
deps = ["Compat", "DataAPI", "Future", "InlineStrings", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrettyTables", "Printf", "REPL", "Random", "Reexport", "SentinelArrays", "SnoopPrecompile", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "aa51303df86f8626a962fccb878430cdb0a97eee"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.5.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "d1fff3a548102f48987a52a2e0d114fa97d730f0"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.13"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DefineSingletons]]
git-tree-sha1 = "0fba8b706d0178b4dc7fd44a96a92382c9065c2c"
uuid = "244e2a9f-e319-4986-a169-4d1fe445cd52"
version = "0.1.2"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.Distances]]
deps = ["LinearAlgebra", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "49eba9ad9f7ead780bfb7ee319f962c811c6d3b2"
uuid = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
version = "0.10.8"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.Distributions]]
deps = ["FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns", "Test"]
git-tree-sha1 = "4ed4a6df2548a72f66e03f3a285cd1f3b573035d"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.96"

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

[[deps.ExceptionUnwrapping]]
deps = ["Test"]
git-tree-sha1 = "e90caa41f5a86296e014e148ee061bd6c3edec96"
uuid = "460bff9d-24e4-43bc-9d9f-a8973cb893f4"
version = "0.1.9"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "4558ab818dcceaab612d1bb8c19cee87eda2b83c"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.5.0+0"

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

[[deps.FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "b4fbdd20c889804969571cc589900803edda16b7"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.7.1"

[[deps.FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c6033cc3892d0ef5bb9cd29b7f2f0331ea5184ea"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.10+0"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "e17cc4dc2d0b0b568e80d937de8ed8341822de67"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.2.0"

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

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pkg", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "d972031d28c8c8d9d7b41a536ad7bb0c2579caca"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.8+0"

[[deps.GR]]
deps = ["Artifacts", "Base64", "DelimitedFiles", "Downloads", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Preferences", "Printf", "Random", "Serialization", "Sockets", "TOML", "Tar", "Test", "UUIDs", "p7zip_jll"]
git-tree-sha1 = "8b8a2fd4536ece6e554168c21860b6820a8a83db"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.72.7"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Qt5Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "19fad9cd9ae44847fe842558a744748084a722d1"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.72.7+0"

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
deps = ["Base64", "CodecZlib", "ConcurrentUtilities", "Dates", "ExceptionUnwrapping", "Logging", "LoggingExtras", "MbedTLS", "NetworkOptions", "OpenSSL", "Random", "SimpleBufferStream", "Sockets", "URIs", "UUIDs"]
git-tree-sha1 = "2613d054b0e18a3dea99ca1594e9a3960e025da4"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.9.7"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "129acf094d168394e80ee1dc4bc06ec835e510a3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+1"

[[deps.HypergeometricFunctions]]
deps = ["DualNumbers", "LinearAlgebra", "OpenLibm_jll", "SpecialFunctions"]
git-tree-sha1 = "0ec02c648befc2f94156eaef13b0f38106212f3f"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.17"

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
git-tree-sha1 = "d75853a0bdbfb1ac815478bacd89cd27b550ace6"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.3"

[[deps.InitialValues]]
git-tree-sha1 = "4da0f88e9a39111c2fa3add390ab15f3a44f3ca3"
uuid = "22cec73e-a1b8-11e9-2c92-598750a2cf9c"
version = "0.3.1"

[[deps.InlineStrings]]
deps = ["Parsers"]
git-tree-sha1 = "9cc2baf75c6d09f9da536ddf58eb2f29dedaf461"
uuid = "842dd82b-1e85-43dc-bf29-5d0ee9dffc48"
version = "1.4.0"

[[deps.IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0cb9352ef2e01574eeebdb102948a58740dcaf83"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2023.1.0+0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.Interpolations]]
deps = ["Adapt", "AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "Requires", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "721ec2cf720536ad005cb38f50dbba7b02419a15"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.14.7"

[[deps.IntervalSets]]
deps = ["Dates", "Random", "Statistics"]
git-tree-sha1 = "16c0cc91853084cb5f58a78bd209513900206ce6"
uuid = "8197267c-284f-5f27-9208-e0e47529a953"
version = "0.7.4"

[[deps.InvertedIndices]]
git-tree-sha1 = "0dc7b50b8d436461be01300fd8cd45aa0274b038"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.3.0"

[[deps.IrrationalConstants]]
git-tree-sha1 = "630b497eafcc20001bba38a4651b327dcfc491d2"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.2"

[[deps.IterTools]]
git-tree-sha1 = "4ced6667f9974fc5c5943fa5e2ef1ca43ea9e450"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.8.0"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

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

[[deps.KernelDensity]]
deps = ["Distributions", "DocStringExtensions", "FFTW", "Interpolations", "StatsBase"]
git-tree-sha1 = "90442c50e202a5cdf21a7899c66b240fdef14035"
uuid = "5ab0869b-81aa-558d-bb23-cbf5423bbe9b"
version = "0.6.7"

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

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f689897ccbe049adb19a065c495e75f372ecd42b"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "15.0.4+0"

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
git-tree-sha1 = "f428ae552340899a935973270b8d98e5a31c49fe"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.16.1"

    [deps.Latexify.extensions]
    DataFramesExt = "DataFrames"
    SymEngineExt = "SymEngine"

    [deps.Latexify.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    SymEngine = "123dc426-2d89-5057-bbad-38513e3affd8"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

[[deps.LeftChildRightSiblingTrees]]
deps = ["AbstractTrees"]
git-tree-sha1 = "fb6803dafae4a5d62ea5cab204b1e657d9737e7f"
uuid = "1d6d02ad-be62-4b6b-8a6d-2f90e265016e"
version = "0.2.0"

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

[[deps.LogDensityProblems]]
deps = ["ArgCheck", "DocStringExtensions", "Random"]
git-tree-sha1 = "f9a11237204bc137617194d79d813069838fcf61"
uuid = "6fdf6af0-433a-55f7-b3ed-c6c6e0b8df7c"
version = "2.1.1"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "c3ce8e7420b3a6e071e0fe4745f5d4300e37b13f"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.24"

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

[[deps.MCMCChains]]
deps = ["AbstractMCMC", "AxisArrays", "Dates", "Distributions", "Formatting", "IteratorInterfaceExtensions", "KernelDensity", "LinearAlgebra", "MCMCDiagnosticTools", "MLJModelInterface", "NaturalSort", "OrderedCollections", "PrettyTables", "Random", "RecipesBase", "Statistics", "StatsBase", "StatsFuns", "TableTraits", "Tables"]
git-tree-sha1 = "8778ea7283a0bf0d7e507a0235adfff38071493b"
uuid = "c7f686f2-ff18-58e9-bc7b-31028e88f75d"
version = "6.0.3"

[[deps.MCMCDiagnosticTools]]
deps = ["AbstractFFTs", "DataAPI", "DataStructures", "Distributions", "LinearAlgebra", "MLJModelInterface", "Random", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns", "Tables"]
git-tree-sha1 = "695e91605361d1932c3e89a812be78480a4a4595"
uuid = "be115224-59cd-429b-ad48-344e309966f0"
version = "0.3.4"

[[deps.MIMEs]]
git-tree-sha1 = "65f28ad4b594aebe22157d6fac869786a255b7eb"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "0.1.4"

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "Pkg"]
git-tree-sha1 = "154d7aaa82d24db6d8f7e4ffcfe596f40bff214b"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2023.1.0+0"

[[deps.MLJModelInterface]]
deps = ["Random", "ScientificTypesBase", "StatisticalTraits"]
git-tree-sha1 = "c8b7e632d6754a5e36c0d94a4b466a5ba3a30128"
uuid = "e80e1ace-859a-464e-9ed9-23947d8ae3ea"
version = "1.8.0"

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

[[deps.MicroCollections]]
deps = ["BangBang", "InitialValues", "Setfield"]
git-tree-sha1 = "629afd7d10dbc6935ec59b32daeb33bc4460a42e"
uuid = "128add7d-3638-4c79-886c-908ea0c25c34"
version = "0.1.4"

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

[[deps.MultivariateStats]]
deps = ["Arpack", "LinearAlgebra", "SparseArrays", "Statistics", "StatsAPI", "StatsBase"]
git-tree-sha1 = "68bf5103e002c44adfd71fea6bd770b3f0586843"
uuid = "6f286f6a-111f-5878-ab1e-185364afe411"
version = "0.10.2"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "0877504529a3e5c3343c6f8b4c0381e57e4387e4"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.2"

[[deps.NaturalSort]]
git-tree-sha1 = "eda490d06b9f7c00752ee81cfa451efe55521e21"
uuid = "c020b1a1-e9b0-503a-9c33-f039bfc54a85"
version = "1.0.0"

[[deps.NearestNeighbors]]
deps = ["Distances", "StaticArrays"]
git-tree-sha1 = "2c3726ceb3388917602169bed973dbc97f1b51a8"
uuid = "b8a86587-4115-5ab1-83bc-aa920d37bbce"
version = "0.4.13"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.Observables]]
git-tree-sha1 = "6862738f9796b3edc1c09d0890afce4eca9e7e93"
uuid = "510215fc-4207-5dde-b226-833fc4488ee2"
version = "0.5.4"

[[deps.OffsetArrays]]
deps = ["Adapt"]
git-tree-sha1 = "82d7c9e310fe55aa54996e6f7f94674e2a38fcb4"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.12.9"

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
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1aa4b74f80b01c6bc2b89992b861b5f210e665b5"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.21+0"

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
git-tree-sha1 = "4b2e829ee66d4218e0cef22c0a64ee37cf258c29"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.7.1"

[[deps.Pipe]]
git-tree-sha1 = "6842804e7867b115ca9de748a0cf6b364523c16d"
uuid = "b98c9c47-44ae-5843-9183-064241ee97a0"
version = "1.3.0"

[[deps.Pixman_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "64779bc4c9784fee475689a1752ef4d5747c5e87"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.42.2+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.9.2"

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
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "JLFzf", "JSON", "LaTeXStrings", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "PrecompileTools", "Preferences", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "RelocatableFolders", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs", "UnicodeFun", "UnitfulLatexify", "Unzip"]
git-tree-sha1 = "75ca67b2c6512ad2d0c767a7cfc55e75075f8bbc"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.38.16"

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

[[deps.PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "a6062fe4063cdafe78f4a0a81cfffb89721b30e7"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.2"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "9673d39decc5feece56ef3940e5dafba15ba0f81"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.1.2"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "7eb1686b4f04b82f96ed7a4ea5890a4f0c7a09f1"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.0"

[[deps.PrettyTables]]
deps = ["Crayons", "Formatting", "LaTeXStrings", "Markdown", "Reexport", "StringManipulation", "Tables"]
git-tree-sha1 = "213579618ec1f42dea7dd637a42785a608b1ea9c"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "2.2.4"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.ProgressLogging]]
deps = ["Logging", "SHA", "UUIDs"]
git-tree-sha1 = "80d919dee55b9c50e8d9e2da5eeafff3fe58b539"
uuid = "33c8b6b6-d38a-422a-b730-caa89a2f386c"
version = "0.1.4"

[[deps.ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "d7a7aef8f8f2d537104f170139553b14dfe39fe9"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.7.2"

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

[[deps.RangeArrays]]
git-tree-sha1 = "b9039e93773ddcfc828f12aadf7115b4b4d225f5"
uuid = "b3c3ace0-ae52-54e7-9d0b-2c1406fd6b9d"
version = "0.3.2"

[[deps.Ratios]]
deps = ["Requires"]
git-tree-sha1 = "1342a47bf3260ee108163042310d26f2be5ec90b"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.5"
weakdeps = ["FixedPointNumbers"]

    [deps.Ratios.extensions]
    RatiosFixedPointNumbersExt = "FixedPointNumbers"

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

[[deps.ScientificTypesBase]]
git-tree-sha1 = "a8e18eb383b5ecf1b5e6fc237eb39255044fd92b"
uuid = "30f210dd-8aff-4c5f-94ba-8e64358c1161"
version = "3.0.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "30449ee12237627992a99d5e30ae63e4d78cd24a"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.2.0"

[[deps.SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "04bdff0b09c65ff3e06a05e3eb7b120223da3d39"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.4.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "StaticArraysCore"]
git-tree-sha1 = "e2cc6d8c88613c05e1defb55170bf5ff211fbeac"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "1.1.1"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.SimpleBufferStream]]
git-tree-sha1 = "874e8867b33a00e784c8a7e4b60afe9e037b74e1"
uuid = "777ac1f9-54b0-4bf8-805c-2214025038e7"
version = "1.1.0"

[[deps.SnoopPrecompile]]
deps = ["Preferences"]
git-tree-sha1 = "e760a70afdcd461cf01a575947738d359234665c"
uuid = "66db9d55-30c0-4569-8b51-7e840670fc0c"
version = "1.0.3"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "c60ec5c62180f27efea3ba2908480f8055e17cee"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.1.1"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "ef28127915f4229c971eb43f3fc075dd3fe91880"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.2.0"
weakdeps = ["ChainRulesCore"]

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

[[deps.SplittablesBase]]
deps = ["Setfield", "Test"]
git-tree-sha1 = "e08a62abc517eb79667d0a29dc08a3b589516bb5"
uuid = "171d559e-b47b-412a-8079-5efa626c420e"
version = "0.1.15"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "Random", "StaticArraysCore", "Statistics"]
git-tree-sha1 = "832afbae2a45b4ae7e831f86965469a24d1d8a83"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.5.26"

[[deps.StaticArraysCore]]
git-tree-sha1 = "6b7ba252635a5eff6a0b0664a41ee140a1c9e72a"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.0"

[[deps.StatisticalTraits]]
deps = ["ScientificTypesBase"]
git-tree-sha1 = "30b9236691858e13f167ce829490a68e1a597782"
uuid = "64bff920-2084-43da-a3e6-9bb72801c0c9"
version = "3.2.0"

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
git-tree-sha1 = "75ebe04c5bed70b91614d684259b661c9e6274a4"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.0"

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

[[deps.StatsPlots]]
deps = ["AbstractFFTs", "Clustering", "DataStructures", "Distributions", "Interpolations", "KernelDensity", "LinearAlgebra", "MultivariateStats", "NaNMath", "Observables", "Plots", "RecipesBase", "RecipesPipeline", "Reexport", "StatsBase", "TableOperations", "Tables", "Widgets"]
git-tree-sha1 = "14ef622cf28b05e38f8af1de57bc9142b03fbfe3"
uuid = "f3b207a7-027a-5e70-b257-86293d7955fd"
version = "0.15.5"

[[deps.StringManipulation]]
git-tree-sha1 = "46da2434b41f41ac3594ee9816ce5541c6096123"
uuid = "892a3eda-7b42-436c-8928-eab12a02cf0e"
version = "0.3.0"

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

[[deps.TableOperations]]
deps = ["SentinelArrays", "Tables", "Test"]
git-tree-sha1 = "e383c87cf2a1dc41fa30c093b2a19877c83e1bc1"
uuid = "ab02a1b2-a7df-11e8-156e-fb1833f50b87"
version = "1.2.0"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "OrderedCollections", "TableTraits", "Test"]
git-tree-sha1 = "1544b926975372da01227b382066ab70e574a3ec"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.10.1"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.TerminalLoggers]]
deps = ["LeftChildRightSiblingTrees", "Logging", "Markdown", "Printf", "ProgressLogging", "UUIDs"]
git-tree-sha1 = "f133fab380933d042f6796eda4e130272ba520ca"
uuid = "5d786b92-1e48-4d6f-9151-6b4477ca9bed"
version = "0.1.7"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.TranscodingStreams]]
deps = ["Random", "Test"]
git-tree-sha1 = "9a6ae7ed916312b41236fcef7e0af564ef934769"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.9.13"

[[deps.Transducers]]
deps = ["Adapt", "ArgCheck", "BangBang", "Baselet", "CompositionsBase", "DefineSingletons", "Distributed", "InitialValues", "Logging", "Markdown", "MicroCollections", "Requires", "Setfield", "SplittablesBase", "Tables"]
git-tree-sha1 = "25358a5f2384c490e98abd565ed321ffae2cbb37"
uuid = "28d57a85-8fef-5791-bfe6-a80928e7c999"
version = "0.4.76"

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

[[deps.Unitful]]
deps = ["ConstructionBase", "Dates", "LinearAlgebra", "Random"]
git-tree-sha1 = "ba4aa36b2d5c98d6ed1f149da916b3ba46527b2b"
uuid = "1986cc42-f94f-5a68-af5c-568840ba703d"
version = "1.14.0"

    [deps.Unitful.extensions]
    InverseFunctionsUnitfulExt = "InverseFunctions"

    [deps.Unitful.weakdeps]
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.UnitfulLatexify]]
deps = ["LaTeXStrings", "Latexify", "Unitful"]
git-tree-sha1 = "e2d817cc500e960fdbafcf988ac8436ba3208bfd"
uuid = "45397f5d-5981-4c77-b2b3-fc36d6e9b728"
version = "1.6.3"

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

[[deps.Widgets]]
deps = ["Colors", "Dates", "Observables", "OrderedCollections"]
git-tree-sha1 = "fcdae142c1cfc7d89de2d11e08721d0f2f86c98a"
uuid = "cc8bc4a8-27d6-5769-a93b-9d913e69aa62"
version = "0.6.6"

[[deps.WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "de67fa59e33ad156a590055375a30b23c40299d3"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "0.5.5"

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
version = "5.8.0+0"

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
# ╠═ee858a0c-1414-11ee-3b47-2fb4b9112c53
# ╟─6c0ecec8-afdc-4072-9dac-4658af3706d5
# ╟─73e449fb-81d2-4a9e-a89d-38909093863b
# ╟─e1f22c73-dee8-4507-af03-3d2d0ceb9011
# ╟─544ac3d9-a2b8-4950-a501-40c14c84b2d8
# ╟─46d87386-7c36-486f-ba59-15d71e88869c
# ╠═e1bd9dd3-855e-4aa6-91aa-2695da07ba48
# ╟─df41dbec-6f39-437c-9f59-8a74a7f5a8dd
# ╟─fbf64d6a-0cc7-4150-9891-e659f43a3b39
# ╟─cfe224e3-dbb3-42bc-ac1b-b96fea5da00d
# ╠═fc535acb-afd6-4f2a-a9f1-15dc83e4a53c
# ╠═e9318f52-e918-42c6-9aa9-45a39ad73ec7
# ╟─bb13a886-0877-42fb-876e-38709f041d65
# ╟─3308752f-d770-4951-9a21-73f1c3886df4
# ╠═4b9cad0a-7ec4-4a58-bf4c-4f103371de33
# ╟─9e73e982-a4ae-4e9b-9650-3cf7c519657c
# ╟─a41bd4a5-a7be-48fe-a222-6e8b3cf98dec
# ╠═d8c05722-a79b-4132-b1c2-982ef39af257
# ╟─65b7e5f4-aff8-4671-9a3a-7aeebef6b83e
# ╟─e2a46e7b-0e83-4275-bf9d-bc1a84fa2e87
# ╟─6a4af386-bfe0-48bb-8d40-300e02680703
# ╠═120d3c31-bba9-476d-8a63-95cdf2457a1b
# ╠═df166b81-a6c3-490b-8cbc-4061f19b750b
# ╟─56cad0cb-352b-4612-b3a3-ddb34de607ad
# ╟─af9c5548-14f2-4771-84cf-bf93eebcd3f2
# ╟─779e0cef-0865-4087-b3d1-563aec15a734
# ╟─611e868a-e808-4a1f-8dd3-2d7ef64e2984
# ╟─69efb78d-1297-46b4-a6bb-218c07c9b2af
# ╟─9f1a120d-80ac-46e0-ae7c-949d2f571b98
# ╟─f08a6391-24da-4d3e-8a3e-55806bb9efbb
# ╟─39ecddfa-89a0-49ec-86f1-4794336215d0
# ╟─0a9c1721-6901-4dc1-a93d-8d8e18f7f375
# ╟─91892a1b-b55c-4f83-91b3-dab4132b1863
# ╟─9d2a6daf-2b06-409e-b034-6e787e64fea8
# ╟─3e3d4c01-8a97-4ede-a341-27ab6fd07b95
# ╠═369366a2-b4c7-44b5-8f64-d11616e99290
# ╟─57c87102-04bc-4414-9258-e2220f9d2e22
# ╟─11700202-40fe-408b-a8b8-5c073daec12d
# ╟─8c9357c8-8339-4889-8a91-b62e542f0407
# ╠═a9621810-e0cb-4925-8b6a-726f78d13510
# ╟─36cb2dd6-19af-4a1f-aa19-7646c2c9cbab
# ╟─a95ed94c-5fe2-4c31-a7a6-e45e841af528
# ╟─fa0dd0fd-7b8a-47a4-bb22-c05c9b70bff3
# ╟─ad1eab82-0fa5-462e-ad17-8cb3b787aaf0
# ╟─13007ba3-7ce2-4201-aa93-559fcbf9d12f
# ╟─e9f3b9e2-5689-40ce-b5b5-bc571ba35c10
# ╠═c9f6fad7-518c-442b-a385-e3fa74431cb1
# ╠═3a8ceb49-403e-424f-bedb-49f5b01c8d7a
# ╠═0a609b97-7859-4053-900d-c1be5d61e68c
# ╠═a8971bb3-cf38-4445-b66e-65ff35ca13ca
# ╟─b4c11a46-438d-4653-89e7-bc2b99e84f48
# ╠═72f82a78-828a-42f2-9b63-9950af4c7be3
# ╠═494eed09-a6e8-488b-bea2-55b7ddb37082
# ╟─060ae93d-12fe-47c6-abe1-ff7728bda572
# ╟─425782f9-4764-4880-ab72-3b481a2cf55a
# ╠═cc216d03-956c-45bb-a6ef-38bf79d6a597
# ╟─7dfa5042-227b-43aa-a55c-30decee08413
# ╟─68c26d99-8f54-4580-8357-5598eb1c8cdf
# ╠═f0dc526c-b221-4652-a877-58a959d97019
# ╠═ad6d0997-43c1-42f3-b997-765c594794b4
# ╟─f0551220-d7c1-4312-b6c5-e0c432889494
# ╟─0b4d5ac8-5a7b-4363-9dbe-2edc517708a0
# ╟─b337a706-cbbf-4acd-8a8f-26fdbc137e8e
# ╟─ab8ec1ee-b28c-4010-9087-aaeb6a022fa9
# ╟─282b71f4-4848-426d-b8fc-0e3656d01767
# ╟─2837effd-25f2-4f49-829e-8fc191db8460
# ╟─0d8327f7-beb8-42de-ad0a-d7e2ebae81ac
# ╟─4baa0604-712a-448f-b3ee-56543bfc0d71
# ╟─de7046da-e361-41d0-b2d7-12439b571795
# ╟─05828da6-bc3c-45de-b059-310159038d5d
# ╟─69061e7f-8a6d-4fac-b187-4d6ff16cf777
# ╟─a9cba95e-9a9e-46c1-8f66-0a9b4ee0fcf0
# ╟─6bea5f44-2abd-47b9-9db5-c5f70dd12c4f
# ╟─37e0d499-bda5-4a5e-8b81-9b3f9384c2fb
# ╟─456f0f6f-1c39-4a50-be3a-0db68d61a95f
# ╟─89fb5821-17c5-46be-8e3c-94439d295220
# ╟─c9d18a43-5984-45f5-b558-368368212355
# ╟─99aab1db-2156-4bd4-9b54-3bb0d4a1620b
# ╠═95ab5440-82dd-4fc4-be08-b1a851caf9ca
# ╠═0f2e6c3a-04c4-4f6b-8ccd-ed18c41e2bc4
# ╟─f35c8af8-00b0-45ad-8910-04f656cecfa3
# ╠═aaf8f3a7-9549-4d02-ba99-e223fda5252a
# ╟─a6470873-26b3-4981-80eb-12a59bd3695d
# ╟─c096bbab-4009-4995-8f45-dc7ffab7ccfa
# ╟─63bc1239-1a6a-4f3b-9d2c-9b904aec573c
# ╠═2f760ffd-1fc5-485b-8e7c-8b49ab7217e3
# ╟─1f5d6cbd-43a2-4a17-996e-d4d2b1a7769c
# ╟─3100b411-e2de-4a43-be80-bcfcb42cef40
# ╟─ff46a86f-5c18-4d83-8f0f-4d13fe7b3df2
# ╟─1edc58de-db69-4dbd-bcc5-c72a07e841be
# ╟─9ca2a2bf-27a9-461b-ae74-1c28ac883168
# ╟─098bc646-7300-4ac6-88af-08a599ba774a
# ╟─11796ca9-30e2-4ba7-b8dc-9a0eda90e14e
# ╠═e68dbe27-95ea-4710-9999-d2c4de0db914
# ╟─488d1200-1ddf-4f06-9643-2eecb2072263
# ╟─0591883c-49af-4201-b2f1-49f208506ece
# ╠═0dfa4d60-577a-4631-bd24-c05aee2969d0
# ╟─bb26ac74-da64-47be-a49a-4519101cffce
# ╠═dd8f1c15-8915-4503-85f0-d3378f8e4751
# ╠═37e01d43-b804-4736-8d91-fb9c7e0ab493
# ╟─5e10db40-5c2e-41c3-a431-e0a4c81d2718
# ╟─e1edfb29-8f82-418b-948b-5542fd6d5b24
# ╠═907e0fea-bad1-49f5-aa98-e2524e93e191
# ╟─e1676b43-b8f0-409f-a6c3-7f6c8852f7ae
# ╠═ffd75de1-b9fd-4883-810d-1d4f79775f0d
# ╠═9ebaf094-75ae-48fa-8c3a-280dfbf24dcd
# ╟─b29576d1-12ed-4346-8114-1e6575f3ee7f
# ╠═8443d615-15a4-4bc3-b40d-c103db279d70
# ╠═4190732e-a3d8-4623-9e47-46b6385335a9
# ╠═d1a0c9e6-b4d2-411c-8634-88c05ac81eb2
# ╠═1d5b1be7-b05b-466b-9580-67c69e60fc40
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
