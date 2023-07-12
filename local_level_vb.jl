### A Pluto.jl notebook ###
# v0.19.26

using Markdown
using InteractiveUtils

# ╔═╡ bd315098-f0d4-11ed-21f8-2b6e59a2ee9c
begin
	using StateSpaceModels
	using Distributions
	using Random
	using LinearAlgebra
	using Plots
	using PlutoUI
end

# ╔═╡ 4cc367d1-37a1-4712-a63e-8826b5646a1b
begin
	using Turing
	using MCMCChains
end

# ╔═╡ a4e3933c-f487-4447-8b94-dcb58eaec886
TableOfContents()

# ╔═╡ e56e975c-ccdd-40c1-8c32-30cab01fe5ce
function gen_data(A, C, Q, R, μ_0, Σ_0, T)

	if length(A) == 1 && length(C) == 1 # univariate
		x = zeros(T)
		y = zeros(T)
		
		for t in 1:T
		    if t == 1
		        x[t] = μ_0 + sqrt(Q) * randn()
		    else
		        x[t] = A * x[t-1] + sqrt(Q) * randn()
		    end
		    	y[t] = C * x[t] + sqrt(R) * randn()
		end
		return y, x

	else
		K, _ = size(A)
	    D, _ = size(C)

		x = zeros(K, T)
		y = zeros(D, T)

		x[:, 1] = rand(MvNormal(A*μ_0, A'*Σ_0*A + Q))
		y[:, 1] = C * x[:, 1] + rand(MvNormal(zeros(D), R))
	
		for t in 2:T
			x[:, t] = A * x[:, t-1] + rand(MvNormal(zeros(K), Q))
			y[:, t] = C * x[:, t] + rand(MvNormal(zeros(D), R))
		end

		return y, x
	end
end

# ╔═╡ ab8b0d77-9eb0-4eb7-b047-737fec53bc39
md"""
Ground truth
"""

# ╔═╡ a7d4f30b-d21e-400e-ba13-a58fdb424fa6
begin
	Random.seed!(123)
	T = 200
	A = 1.0
	C = 1.0
	R = 0.2
	Q = 1.0 # assume fixed in Beale
	y, x_true = gen_data(A, C, Q, R, 0.0, 1.0, T)
end

# ╔═╡ d1ab75c8-5d7e-4d42-9b54-c48cd0e61e90
md"""
# SSM package MLE results
"""

# ╔═╡ f108fc16-b3c2-4a89-87e8-f735a1d04301
let	
	model = LocalLevel(y)
	fit!(model) # MLE and uni-variate kalman filter
	print_results(model)
end

# ╔═╡ 05be6f30-1936-40b1-83cc-743f8704610e
md"""
# Uni-variate VB DLM
"""

# ╔═╡ 99198a4a-6322-4c0f-be9f-24c9bb86f2ca
begin
	struct HPP_uni
	    a::Float64
	    b::Float64
	    α::Float64
	    γ::Float64
		μ₀::Float64
		σ₀::Float64
	end
	
	struct HSS_uni
	    W_A::Float64
	    S_A::Float64
	    W_C::Float64
	    S_C::Float64
	end

	struct Exp_ϕ_uni
		A
		C
		R⁻¹
		AᵀA
		CᵀR⁻¹C
		R⁻¹C
		CᵀR⁻¹
	end
end

# ╔═╡ d7a5cb1a-c768-4aed-beec-eaad552735b6
function vb_m_uni(y::Vector{Float64}, hss::HSS_uni, hpp::HPP_uni)
	T = length(y)
    a, b, α, γ, μ_0, σ_0 = hpp.a, hpp.b, hpp.α, hpp.γ, hpp.μ₀, hpp.σ₀
	W_A, S_A, W_C, S_C = hss.W_A, hss.S_A, hss.W_C, hss.S_C

	σ_A = 1 / (α + W_A)
    σ_C = 1 / (γ + W_C)

	#q_A = Normal(σ_A*S_A, sqrt(σ_A))
	
	G = y' * y - S_C * σ_C * S_C

	# Update parameters of Gamma distribution
	a_n = a + 0.5 * T
	b_n = b + 0.5 * G

	q_ρ = Gamma(a_n, 1 / b_n)
	ρ̄ = mean(q_ρ)

	#ρ_s = rand(q_p)
	#q_C = Normal(σ_C*S_C, sqrt(σ_C/ρ_s))

	Exp_A = S_A*σ_A
	Exp_C = S_C*σ_C
	Exp_R⁻¹ = ρ̄

	Exp_AᵀA = Exp_A^2 + σ_A
    Exp_CᵀR⁻¹C = Exp_C^2 * Exp_R⁻¹ + σ_C
    Exp_R⁻¹C = Exp_C * Exp_R⁻¹
    Exp_CᵀR⁻¹ = Exp_R⁻¹C 

	return Exp_ϕ_uni(Exp_A, Exp_C, Exp_R⁻¹, Exp_AᵀA, Exp_CᵀR⁻¹C, Exp_R⁻¹C, Exp_CᵀR⁻¹)
end

# ╔═╡ cd96215e-5ec9-4031-b2e8-39d76b5e5bee
md"""
## Test M-step (uni-variate)
"""

# ╔═╡ 7db0f68b-f27f-4529-961a-7b9a9c1b7500
let
	T = length(y)

	W_A = sum(x_true[t-1] * x_true[t-1] for t in 2:T)
	#W_A += μ_0*μ_0'

	S_A = sum(x_true[t-1] * x_true[t] for t in 2:T)
	#S_A += μ_0*x_true[:, 1]'

	W_C = sum(x_true[t] * x_true[t] for t in 1:T)

	S_C = sum(x_true[t] * y[t] for t in 1:T)

	hss = HSS_uni(W_A, S_A, W_C, S_C)
	α = 1.0
	γ = 1.0
	a = 0.1
	b = 0.1
	μ_0 = 0.0
	σ_0 = 1.0

	hpp = HPP_uni(a, b, α, γ, μ_0, σ_0)

	# should recover values of A, C, R close to ground truth
	exp_np = vb_m_uni(y, hss, hpp)
end

# ╔═╡ a78d2f18-af12-4285-944f-4297a69f2369
function v_forward(y::Vector{Float64}, exp_np::Exp_ϕ_uni, μ_0, σ_0)
	T = length(y)

    μs = zeros(T)
    σs = zeros(T)
	σs_ = zeros(T)
	
	# TO-DO: ELBO and convergence check
	#Qs = zeros(D, D, T)
	#fs = zeros(D, T)

	# initialise for t=1
	σ₀_ = 1 / (σ_0^(-1) + exp_np.AᵀA)
	σs_[1] = σ₀_
	
    σs[1] = 1/ (1.0 + exp_np.CᵀR⁻¹C - exp_np.A*σ₀_*exp_np.A)
    μs[1] = σs[1]*(exp_np.CᵀR⁻¹*y[1] + exp_np.A*σ₀_*σ_0^(-1)*μ_0)

	# iterate over T
	for t in 2:T
		σₜ₋₁_ = 1/ ((σs[t-1])^(-1) + exp_np.AᵀA)
		σs_[t] = σₜ₋₁_
		
		σs[t] = 1/ (1.0 + exp_np.CᵀR⁻¹C - exp_np.A*σₜ₋₁_*exp_np.A)
    	μs[t] = σs[t]*(exp_np.CᵀR⁻¹*y[t] + exp_np.A*σₜ₋₁_*(σs[t-1])^(-1)*μs[t-1])

	end

	return μs, σs, σs_
end

# ╔═╡ 3f882bc0-b27f-48c2-997e-3bfa8fda421e
function error_metrics(true_means, smoothed_means)
    T = length(true_means)
    mse = sum((true_means .- smoothed_means).^2) / T
    mad = sum(abs.(true_means .- smoothed_means)) / T
    mape = sum(abs.((true_means .- smoothed_means) ./ true_means)) / T * 100

	# mean squared error (MSE), mean absolute deviation (MAD), and mean absolute percentage error (MAPE) 
    return mse, mad, mape
end

# ╔═╡ 720ce158-f0b6-4165-a742-938c83146cff
md"""
## Testing filtered output (variational)
"""

# ╔═╡ 2b65eb50-0ff3-441f-9c0e-964bf10d29bc
let
	# A, C = 1.0 , R⁻¹ = 5.0 (R = 0.2)
	true_exp_np = Exp_ϕ_uni(1.0, 1.0, 5.0, 1.0, 5.0, 5.0, 5.0)

	fm = v_forward(y, true_exp_np, 0.0, 1.0)[1]

	println("MSE, MAD, MAPE: ", error_metrics(x_true, fm))
end

# ╔═╡ 602a021c-a473-4267-aaf3-2ed77f05ca07
md"""
Compare SSM package filter and smoother
"""

# ╔═╡ 2bae2d86-2a3f-446c-aa34-7582b383d2b2
let
	model = LocalLevel(y)
	fit!(model)
	fm = get_filtered_state(model)
	filter_err = error_metrics(x_true, fm)

	sm = get_smoothed_state(model)
	smooth_err = error_metrics(x_true, sm)

	println("Filtered MSE, MAD, MAPE: ", filter_err)
	println("Smoother MSE, MAD, MAPE: ", smooth_err)
end

# ╔═╡ 68564ae6-2540-4a6c-aed5-0794d413e6ef
function v_backward(y::Vector{Float64}, exp_np::Exp_ϕ_uni)
	T = length(y)
	ηs = zeros(T)
    Ψs = zeros(T)

    # Initialize the filter, t=T, β(x_T-1)
	Ψs[T] = 0.0
    ηs[T] = 1.0
	
	Ψₜ = 1/(1.0 + exp_np.CᵀR⁻¹C)
	Ψs[T-1] = 1/ (exp_np.AᵀA - exp_np.A*Ψₜ*exp_np.A)
	ηs[T-1] = Ψs[T-1]*exp_np.A*Ψₜ*exp_np.CᵀR⁻¹*y[T]
	
	for t in T-2:-1:1
		Ψₜ₊₁ = 1/(1.0 + exp_np.CᵀR⁻¹C + (Ψs[t+1])^(-1))
		
		Ψs[t] = 1/ (exp_np.AᵀA - exp_np.A*Ψₜ₊₁*exp_np.A)
		ηs[t] = Ψs[t]*exp_np.A*Ψₜ₊₁*(exp_np.CᵀR⁻¹*y[t+1] + (Ψs[t+1])^(-1)*ηs[t+1])
	end

	# for t=1, this correspond to β(x_0), the probability of all the data given the setting of the auxiliary x_0 hidden state.
	Ψ₁ = 1/ (1.0 + exp_np.CᵀR⁻¹C + (Ψs[1])^(-1))
	Ψ_0 = 1/ (exp_np.AᵀA - exp_np.A*Ψ₁*exp_np.A)
	η_0 = Ψs[1]*exp_np.A*Ψ₁*(exp_np.CᵀR⁻¹*y[1] + (Ψs[1])^(-1)*ηs[1])
	
	return ηs, Ψs, η_0, Ψ_0
end

# ╔═╡ cc5e1e9a-9087-437a-b832-1918de6d4c48
function parallel_smoother(μs, σs, ηs, Ψs, η_0, Ψ_0, μ_0, σ_0)
	T = length(μs)
	Υs = zeros(T)
	ωs = zeros(T)

	# ending condition t=T
	Υs[T] = σs[T]
	ωs[T] = μs[T]
	
	for t in 1:(T-1)
		Υs[t] = 1 / ((σs[t])^(-1) + (Ψs[t])^(-1))
		ωs[t] = Υs[t]*((σs[t])^(-1)*μs[t] + (Ψs[t])^(-1)*ηs[t])
	end

	# t = 0
	Υ_0 = 1 / ((σ_0)^(-1) + (Ψ_0)^(-1))
	ω_0 = Υ_0*((σ_0)^(-1)μ_0 + (Ψ_0)^(-1)η_0)
	
	return ωs, Υs, ω_0, Υ_0
end

# ╔═╡ 5d9cf8b5-82f4-496f-ad7d-44b83a6ef157
md"""
## Testing smoothed output (variational)
"""

# ╔═╡ 5758e9c4-4c8c-4573-80a7-8270e8b428e4
let
	true_exp_np = Exp_ϕ_uni(1.0, 1.0, 5.0, 1.0, 5.0, 5.0, 5.0)
	
	μs, σs, _ =  v_forward(y, true_exp_np, 0.0, 1.0)
	
	ηs, Ψs, η_0, Ψ_0 = v_backward(y, true_exp_np)

	ωs, Υs, ω_0, Υ_0 = parallel_smoother(μs, σs, ηs, Ψs, η_0, Ψ_0, 0.0, 1.0)

	println("MSE, MAD, MAPE: ", error_metrics(x_true, ωs))
end

# ╔═╡ ce95e427-eeec-48c6-8e62-bf6f9f892b3e
md"""
Smoother shows improved latent state inference over filtered estimates as expected
"""

# ╔═╡ 2969a12c-8fe1-4b76-bf0e-b7be6425eb21
function v_pairwise_x(σs_, exp_np::Exp_ϕ_uni, Ψs)
	T = length(σs_)

	# cross-covariance is then computed for all time steps t = 0, ..., T−1
	Υ_ₜ₋ₜ₊₁ = zeros(T)
	
	for t in 1:T-2
		Υ_ₜ₋ₜ₊₁[t+1] = σs_[t+1]*exp_np.A*(1.0 + exp_np.CᵀR⁻¹C + (Ψs[t+1])^(-1) - exp_np.A*σs_[t+1]*exp_np.A)^(-1)
	end

	# t=0, the cross-covariance between the zeroth and first hidden states.
	Υ_ₜ₋ₜ₊₁[1] = σs_[1]*exp_np.A*(1.0 + exp_np.CᵀR⁻¹C + (Ψs[1])^(-1) - exp_np.A*σs_[1]*exp_np.A)^(-1)

	# t=T-1, Ψs[T] = 0 special case
	Υ_ₜ₋ₜ₊₁[T] = σs_[T]*exp_np.A*(1.0 + exp_np.CᵀR⁻¹C - exp_np.A*σs_[T]*exp_np.A)^(-1)
	
	return Υ_ₜ₋ₜ₊₁
end

# ╔═╡ 738519e3-c469-4def-ae9d-dbdd92563120
function vb_e_uni(y::Vector{Float64}, hpp::HPP_uni, exp_np::Exp_ϕ_uni, smooth_out = false)
	T = length(y)

	# forward pass
	μs, σs, σs_ = v_forward(y, exp_np, hpp.μ₀, hpp.σ₀)

	# backward pass 
	ηs, Ψs, η₀, Ψ₀ = v_backward(y, exp_np)

	# marginal (smoothed) means, covs, and pairwise beliefs 
	ωs, Υs, ω_0, Υ_0 = parallel_smoother(μs, σs, ηs, Ψs, η₀, Ψ₀, hpp.μ₀, hpp.σ₀)
	Υ_ₜ₋ₜ₊₁ = v_pairwise_x(σs_, exp_np, Ψs)

	# hidden state sufficient stats 
	W_A = sum(Υs[t-1] + ωs[t-1] * ωs[t-1] for t in 2:T)
	W_A += Υ_0 + ω_0*ω_0

	S_A = sum(Υ_ₜ₋ₜ₊₁[t] + ωs[t-1] * ωs[t] for t in 2:T)
	S_A += Υ_ₜ₋ₜ₊₁[1] + ω_0*ωs[1]
	
	W_C = sum(Υs[t] + ωs[t] * ωs[t] for t in 1:T)
	S_C = sum(ωs[t] * y[t] for t in 1:T)

	if (smooth_out)
		return ωs, Υs
	end
	
	return HSS_uni(W_A, S_A, W_C, S_C), ω_0, Υ_0
end

# ╔═╡ 05b29e3f-4493-4829-abba-5db47756b54a
md"""
Test e-step
"""

# ╔═╡ 668197a9-91e4-461d-8552-a20a34b6eb3d
let
	α = 1.0
	γ = 1.0
	a = 0.1
	b = 0.1
	μ_0 = 0.0
	σ_0 = 1.0

	hpp = HPP_uni(a, b, α, γ, μ_0, σ_0)

	true_exp_np = Exp_ϕ_uni(1.0, 1.0, 5.0, 1.0, 5.0, 5.0, 5.0)

	vb_e_uni(y, hpp, true_exp_np)[1]
end

# ╔═╡ 363fbc94-afa6-4a85-9ac5-4d253b21df69
begin
	W_A_true = sum(x_true[t-1] * x_true[t-1] for t in 2:T)
	#W_A += μ_0*μ_0'

	S_A_true = sum(x_true[t-1] * x_true[t] for t in 2:T)
	#S_A += μ_0*x_true[:, 1]'

	W_C_true = sum(x_true[t] * x_true[t] for t in 1:T)

	S_C_true = sum(x_true[t] * y[t] for t in 1:T)

	HSS_uni(W_A_true, S_A_true, W_C_true, S_C_true)
end

# ╔═╡ fbb7a8c1-dc44-4d79-94fc-b7068a7881e1
md"""
Uni-variate VB
"""

# ╔═╡ c3669042-bb56-423e-a077-b0cb82ce74a3
function vb_dlm(y::Vector{Float64}, hpp::HPP_uni, max_iter=1000)
	T = length(y)
	W_A = 1.0
	S_A = 1.0
	W_C = 1.0
	S_C = 1.0
	
	hss = HSS_uni(W_A, S_A, W_C, S_C)
	exp_np = missing
	
	for i in 1:max_iter
		exp_np = vb_m_uni(y, hss, hpp)
				
		hss, ω_0, Υ_0 = vb_e_uni(y, hpp, exp_np)
	end

	return exp_np
end

# ╔═╡ fd2784db-87b3-4c2e-9d4d-7bbca8ec3aba
md"""
## Ploting parameter learning, A, C, R
"""

# ╔═╡ 0e431880-8449-44bc-8996-2c50439b0eac
function vb_ll_his(y::Vector{Float64}, hpp::HPP_uni, max_iter=1000)
	T = length(y)
	
	hss = HSS_uni(1.0, 1.0, 1.0, 1.0)
	exp_np = missing

	history_A = Vector{Float64}(undef, max_iter - 50)
    history_C = Vector{Float64}(undef, max_iter - 50)
    history_R = Vector{Float64}(undef, max_iter - 50)
	
	for i in 1:max_iter
		exp_np = vb_m_uni(y, hss, hpp)
				
		hss, ω_0, Υ_0 = vb_e_uni(y, hpp, exp_np)

		if(i > 50) # discard the first 10 to see better plots
			history_A[i-50] = exp_np.A
			history_C[i-50] = exp_np.C
       		history_R[i-50] = 1/exp_np.R⁻¹
		end
	end

	return exp_np, history_A, history_C, history_R
end

# ╔═╡ e93f8a5f-fd02-4894-aba5-7cabb9dc76b3
let
	α = 1.0
	γ = 1.0
	a = 0.001
	b = 0.001
	μ_0 = 0.0
	σ_0 = 1.0

	hpp = HPP_uni(a, b, α, γ, μ_0, σ_0)
	exp_f, As, Cs, Rs = vb_ll_his(y, hpp, 2000) 

	p1 = plot(As, title = "A learning", legend = false)
	p2 = plot(Cs, title = "C learning", legend = false)
	p3 = plot(Rs, title = "R learning", legend = false)

	# A convergence is quick, but C and R are relatively slow
	plot(p1, p2, p3, layout = (3, 1))
end

# ╔═╡ 0e7c0e3d-297d-4ccd-a71b-54a49fc40212
md"""
Converge around 1500 iterations
"""

# ╔═╡ 418b6277-da40-4232-ab0b-a07c1dc0d5e9
md"""
## Hidden state x inference 
"""

# ╔═╡ 40b0773b-fc7d-4b28-9684-7e4e36ee81c9
begin
	α = 1.0
	γ = 1.0
	a = 0.01
	b = 0.01
	μ_0 = 0.0
	σ_0 = 1.0
	hpp = HPP_uni(a, b, α, γ, μ_0, σ_0)
	@time exp_f = vb_dlm(y, hpp, 1500) 
	exp_f.A, exp_f.C, 1 / exp_f.R⁻¹
	xs, σs = vb_e_uni(y, hpp, exp_f, true)
	println("\nVB uni-DLM latent x error: ", error_metrics(x_true, xs)) # MSE, MAD, MAPE of VB inference
end

# ╔═╡ 06b9d658-7d19-448d-8c02-6fabc5d4a551
md"""
### Compare with package Kalman smoother (MSE, MAD, MAPE)
"""

# ╔═╡ 1ba53bec-658c-4e07-8728-f9799a5514e8
let
	model = LocalLevel(y)
	fit!(model)
	sm = get_smoothed_state(model)
	smooth_err = error_metrics(x_true, sm)
	println("SSM Smoother error: ", smooth_err)
end

# ╔═╡ 5cc8a85a-2542-4b69-b79a-736b87a5a8c4
md"""
# MCMC 
"""

# ╔═╡ 80e4b5da-9d6e-46cd-9f84-59fa86c201b1
md"""
## Gibbs sampling uni-variate DLM
"""

# ╔═╡ 29489270-20ee-4835-8451-db12fe46cf4c
md"""
### Sample state transition $a$

Given prior, $p(a) \sim \mathcal N(0, 1)$, by conjugacy, The mean and variance of this Normal can be computed using standard results for the Normal distribution.

$p(a | y_{1:T}, x_{1:T}, c, r) = p(a | x_{1:T}) \propto p(a)p(x_{1:T}|a) \sim \mathcal{N}\left(\frac{\sum_{t=2}^{T} x_t x_{t-1}}{\sum_{t=2}^{T} x_{t-1}^2}, \frac{1}{\sum_{t=2}^{T} x_{t-1}^2}\right)$
"""

# ╔═╡ 84f6d8f2-6ba5-43d6-9a06-f485975bf208
function sample_a(xs, q)
	T = length(xs)
	return rand(Normal(sum(xs[1:T-1] .* xs[2:T]) / sum(xs[1:T-1].^2), sqrt(q / sum(xs[1:T-1].^2))))
end

# ╔═╡ be028bb1-28a4-45f4-8c52-c919636b2ea4
let
	sample_a(x_true, 1.0)
end

# ╔═╡ b2efefeb-f4a4-40f6-850f-f73b30ce386c
md"""
### Sample emission $c$

Given prior, $p(c) \sim \mathcal N(0, 1)$, by conjugacy, the posteror in which we sample from is: 

$p(c | y_{1:T}, x_{1:T}, a, r) = p(c | y_{1:T}, x_{1:T}, r) \propto p(c) p (y_{1:T}∣c, x_{1:T},r) \sim \mathcal{N}\left(\frac{\sum_{t=1}^{T} y_t x_t}{\sum_{t=1}^{T} x_t^2}, \frac{1}{r \sum_{t=1}^{T} x_t^2}\right)$
"""

# ╔═╡ 29ca031c-5520-4ef5-95c1-2b0c9fa12906
function sample_c(xs, ys, r)
    return rand(Normal(sum(ys .* xs) / sum(xs.^2), sqrt(r / (sum(xs.^2)))))
end

# ╔═╡ a320c6e2-42f3-445f-9585-6e2ff5a43060
let
	sample_c(x_true, y, 0.2)
end

# ╔═╡ 5f3d1c80-d93f-4916-82a8-21205e4d0e26
md"""
### Sample observation noise $r$

Given prior, $p(r) \sim \mathcal IG(a, b)$, by conjugacy, the posteror in which we sample from is: 

$p(r | y_{1:T}, x_{1:T}, a, c) = p(r | y_{1:T}, x_{1:T}, c) \propto p(r) p (y_{1:T}∣x_{1:T}, c, r) \sim \mathcal{IG}\left(\alpha + \frac{T}{2}, \beta + \frac{1}{2} \sum_{t=1}^{T} (y_t - c x_t)^2\right)$

"""

# ╔═╡ 8e3edeee-c940-4975-99e8-bc27c3b18939
function sample_r(xs, ys, c, α_r, β_r)
	T = length(ys)
    α_post = α_r + T / 2
    β_post = β_r + sum((ys - c * xs).^2) / 2
	λ_r = rand(Gamma(α_post, 1 / β_post))
	return 1/λ_r # inverse precision is variance
end

# ╔═╡ 5985ac18-636e-4606-9978-7e1e0ce1fd09
let
	sample_r(x_true, y, 1.0, 0.01, 0.01)
end

# ╔═╡ 392e0306-6e39-4d62-80aa-7c9f837cd0a0
md"""
### Sample latent $x$
"""

# ╔═╡ 1f0d9a8f-94b9-4e87-a837-9c350b905c72
md"""

"""

# ╔═╡ aa404f69-e857-4eb6-87f1-298d62429891
md"""
Multi-move FFBS sampling
"""

# ╔═╡ 7ea26a1d-b6be-4d47-ad30-222ec6ea1a5a
function sample_x_ffbs(y, A, C, Q, R, μ_0, σ_0)
    T = length(y)
    μs = Vector{Float64}(undef, T)
    σs = Vector{Float64}(undef, T)
    μs[1] = μ_0
    σs[1] = σ_0
	
    for t in 2:T #forward
        μ_pred = A * μs[t-1]
        σ_pred = A * σs[t-1] * A + Q
        K = σ_pred * C * (1/ (C^2 * σ_pred + R))
		
        μs[t] = μ_pred + K * (y[t] - C * μ_pred)
        σs[t] = (1 - K * C) * σ_pred
    end

	x = Vector{Float64}(undef, T)
    x[T] = rand(Normal(μs[T], sqrt(σs[T])))

    for t in (T-1):-1:1 #backward
        μ_cond = μs[t] + σs[t] * A * (1/ (A * σs[t] * A + Q)) * (x[t+1] - A * μs[t])
        σ_cond = σs[t] - σs[t] * A * (1/ (A * σs[t] * A + Q)) * A * σs[t]
        x[t] = rand(Normal(μ_cond, sqrt(σ_cond)))
    end
    return x
end

# ╔═╡ 494373d7-3a8c-4717-bc9b-6e57267b3a58
let
	xs_ffbs = sample_x_ffbs(y, 1.0, 1.0, 1.0, 0.2, 0, 1.0)
	println("FFBS latent x error: ", error_metrics(x_true, xs_ffbs))
end

# ╔═╡ 427fcde3-c719-4504-89c3-dfc0491677f9
function gibbs_uni_dlm(y, num_iterations=2000, burn_in=200, thinning=5)
	T = length(y)
	μ_0 = 0.0  # Prior mean for the states
	λ_0 = 1.0  # Prior precision for the states
	α = 0.01  # Shape parameter for Inverse-Gamma prior
	β = 0.01  # Scale parameter for Inverse-Gamma prior
	
	# Initial values for the parameters
	a = rand(Normal(μ_0, λ_0))
	c = rand(Normal(μ_0, λ_0))
	r = rand(InverseGamma(α, β))
	q = 1.0

	n_samples = Int.(num_iterations/thinning)
	# Store the samples
	samples_x = zeros(n_samples, T)
	samples_a = zeros(n_samples)
	samples_c = zeros(n_samples)
	samples_r = zeros(n_samples)
	
	# Gibbs sampler
	for i in 1:num_iterations+burn_in
	    # Update the states
		x = sample_x_ffbs(y, a, c, q, r, μ_0, 1/λ_0)
	
	    # Update the state transition factor
	    a = sample_a(x, q)
	
	    # Update the emission factor
	    c = sample_c(x, y, r)
	
	    # Update the observation noise
		r = sample_r(x, y, c, α, β)
	
	    # Store the samples
		if i > burn_in && mod(i - burn_in, thinning) == 0
			index = div(i - burn_in, thinning)
		    samples_x[index, :] = x
		    samples_a[index] = a
		    samples_c[index] = c
		    samples_r[index] = r
		end
	end
	return samples_x, samples_a, samples_c, samples_r
end

# ╔═╡ 1b1e9945-ad12-4108-a866-f02192eb064f
begin
	Random.seed!(888)
	@time x_ss, a_ss, c_ss, r_ss = gibbs_uni_dlm(y, 3000, 300, 1)
	plot(a_ss, label="a")
	plot!(c_ss, label="c")
	plot!(r_ss, label="r")
end

# ╔═╡ 678b89ee-0302-4ab9-865e-564460f2d691
mean(r_ss)

# ╔═╡ 737782c3-17a7-4684-b912-8ac392422941
println("end chain latent x error: ", error_metrics(x_true, x_ss[end, : ]))

# ╔═╡ b07bcd90-d9c0-4a71-886c-756cb03b9bc1
md"""
Manual Gibbs sampling procedure is faster than NUTS sampler used by Turing,

but c estimates not very stable (sometimes negative 1?), and Gibbs tend to require more samples to converge.
"""

# ╔═╡ 1c0ed9fb-c56a-4aab-a765-49c394123a42
md"""
# Known A, C:
- Beale treatment v.s DLM with R treatment
- Add Inference step for Q, fix A, C as known constants
"""

# ╔═╡ 8dbf29db-cefd-4bd4-95e7-a302c7aa858a
md"""
## Gibbs sampling of local level model
`DLM with R` _4.4.3_

Given gamma prior on the precision (inverse variance) with prior parameter α, β
"""

# ╔═╡ 1075e6dc-be4b-4594-838e-60d44100c92d
function sample_q(xs, a, α_q, β_q, x_0)
	T = length(xs)
    α_post = α_q + T / 2
    β_post = β_q + sum((xs[2:T] .- (a .* xs[1:T-1])).^2) /2 
	
	β_post += (xs[1] - a * x_0)^2 /2
	λ_q = rand(Gamma(α_post, 1 / β_post))
	
	return 1/λ_q # inverse precision is variance
end

# ╔═╡ ce1f3eed-13ab-4fa7-aafc-a97954dd818b
let
	sample_q(x_true, 1.0, 0.01, 0.01, 0.0)
end

# ╔═╡ 7e4fa23d-b05b-4f77-959e-29577e349333
function gibbs_ll(y, a, c, mcmc=3000, burn_in=300, thinning=1)
	T = length(y)
	μ_0 = 0.0  # Prior mean for the states
	λ_0 = 1.0  # Prior precision for the states
	
	α = 0.01  # Shape parameter for Inverse-Gamma prior
	β = 0.01  # Scale parameter for Inverse-Gamma prior
	
	# Initial values for the parameters
	r = rand(InverseGamma(α, β))
	q = rand(InverseGamma(α, β))

	n_samples = Int.(mcmc/thinning)
	# Store the samples
	samples_x = zeros(n_samples, T)
	samples_q = zeros(n_samples)
	samples_r = zeros(n_samples)
	
	# Gibbs sampler
	for i in 1:mcmc+burn_in
	    # Update the states
		x = sample_x_ffbs(y, a, c, q, r, μ_0, 1/λ_0)
		
		# Update the system noise
		q = sample_q(x, a, α, β, μ_0)
		
	    # Update the observation noise
		r = sample_r(x, y, c, α, β)
	
	    # Store the samples
		if i > burn_in && mod(i - burn_in, thinning) == 0
			index = div(i - burn_in, thinning)
		    samples_x[index, :] = x
			samples_q[index] = q
		    samples_r[index] = r
		end
	end

	return samples_x, samples_q, samples_r
end

# ╔═╡ 69e31afc-7893-4099-8d48-937d8ebffa86
md"""
## VB Derivation for local level model

we assume $a = c = 1.0$ is known and fixed in local level model, so the unknown model parameters are $r$ and $q$:

Full joint probability:

$$p(r, q, x_{0:T}, y_{1:T}) = p(r) p(q) p(x_0) \prod_{t=1}^{T} p(x_t | x_{t-1}, q) \prod_{t=1}^{T} p(y_t | x_t, r)$$

Log marginal likelihood:

$\begin{align}
\log p(y_{1:T}) &= \log \int p(r, q, x_{0:T}, y_{1:T}) \ dr \ dq \ dx_{0:T} \\
&\geq \int \ dr \ dq \ dx_{0:T} \  q(r, q, x_{0:T}) \log \frac{p(r, q, x_{0:T}, y_{1:T})}{q(r, q, x_{0:T})} \\
&= \mathcal F
\end{align}$

$\mathcal{F}(q) = E_q[\log p(r, q, x_{0:T}, y_{1:T})] - E_q[\log q(r, q, x_{0:T})]$


To make the variational inference tractable, we can assume that the variational distribution factorizes over the model parameters and the latent states, 

$$q(r, q, x_{0:T}) = q(r) \ q(q) \ q(x_{0:T})$$
"""

# ╔═╡ 94caa09c-59b6-461f-b56f-178992d2bc83
md"""
### Choice of prior
let $τ_r = 1/r, τ_q = 1/q$, a natural prior for the precision is the gamma distribution: 

$$τ_r \sim \mathcal Gamma(\alpha, \beta)$$
$$τ_q \sim \mathcal Gamma(\alpha, \beta)$$
"""

# ╔═╡ bb4de2cf-e4b6-4788-9f8b-ff5fd2ca2570
md"""
### VB-M 

#### Updating $q(τ_r, τ_q)$

$\ln q(τ_r, τ_q) =  \langle \ln p(τ_r, τ_q, x_{0:T}, y_{1:T}) \rangle_{\hat q(x_{0:T})} + c$

Given that we have assumed $τ_r$ and $τ_q$ to follow Gamma distributions in the variational family, their variational posteriors $q(τ_r)$ and $q(τ_q)$ will also be Gamma distributions. The parameters of these Gamma distributions are updated in the M-step:

$\alpha_r' = \alpha + T/2$
$\beta_r' = \beta + 0.5 \sum_{t=1}^{T} E_q[(y_t - x_t)^2]$

$\alpha_q' = \alpha + (T-1)/2$
$\beta_q' = \beta + 0.5 \sum_{t=2}^{T} E_q[(x_t - x_{t-1})^2]$

Here, $E_q[\cdot]$ denotes the expectation with respect to the variational distribution $q(x_{0:T})$, and the expectations are computed using the current parameters of the variational distribution from the E-step --- latent state sufficient statistics (HSS)

$\begin{align}
E_q[(y_t - x_t)^2] &= E_q[y_t^2 - 2y_tx_t + x_t^2] \\
                   &= y_t^2 - 2y_tE_q[x_t] + E_q[x_t^2] \\
                   &= y_t^2 - 2y_t\mu_t + \mu_t^2 + \sigma_t^2
\end{align}$



$\begin{align}
E_q[(x_t - x_{t-1})^2] &= E_q[x_t^2 - 2x_tx_{t-1} + x_{t-1}^2] \\
                       &= \mu_t^2 + \sigma_t^2 - 2 E_q[x_tx_{t-1}] + \mu_{t-1}^2 + \sigma_{t-1}^2
\end{align}$


This motivates E-step to compute the following HSS:

$w_c = \sum_{t=1}^{T} E_q[x_t^2] = \sum_{t=1}^{T} (\sigma_t^2 + \mu_t^2)$
$w_a = \sum_{t=1}^{T} E_q[x_{t-1}^2] = \sum_{t=1}^{T} (\sigma_{t-1}^2 + \mu_{t-1}^2)$
$s_c = \sum_{t=1}^{T} y_t E_q[x_t] = \sum_{t=1}^{T} \mu_t y_t$
$s_a = \sum_{t=1}^{T} E_q[x_{t-1} x_t] = \sum_{t=1}^{T} \sigma_{t-1, t} + \mu_{t-1}\mu_t$

"""

# ╔═╡ 501172ab-203d-4faa-a3b0-3e4fa0c79d10
md"""
TO-DO:
verify cross-variance, cross-moment by hand
"""

# ╔═╡ 981608f2-57f6-44f1-95ed-82e8cca04718
begin
	struct HSS_ll
	    w_c::Float64
	    w_a::Float64
	    s_c::Float64
	    s_a::Float64
	end
	
	# Define the struct for the priors
	struct Priors_ll
	    α_r::Float64
	    β_r::Float64
	    α_q::Float64
	    β_q::Float64
	    μ_0::Float64
	    σ_0::Float64
	end
end

# ╔═╡ 241e587f-b3dd-4bf8-83d0-1459c389fcc0
function vb_m_ll(y, hss::HSS_ll, priors::Priors_ll)
    T = length(y)

    # Update parameters for τ_r
    α_r_p = priors.α_r + T / 2
    β_r_p = priors.β_r + 0.5 * (y' * y - 2 * hss.s_c + hss.w_c)

    # Update parameters for τ_q
    α_q_p = priors.α_q + T / 2
    β_q_p = priors.β_q + 0.5 * (hss.w_a + hss.w_c - 2 * hss.s_a)

    # Compute expectations
    E_τ_r = α_r_p / β_r_p
    E_τ_q = α_q_p / β_q_p

    return E_τ_r, E_τ_q	
end

# ╔═╡ 168d4dbd-f0dd-433b-bb4a-e3bb918fb184
md"""
Test vb m-step, given x_true
"""

# ╔═╡ 2308aa4c-bb99-4546-a108-9fa88fca130b
let
	T = length(y)
	w_a = sum(x_true[t-1] * x_true[t-1] for t in 2:T)
	s_a = sum(x_true[t-1] * x_true[t] for t in 2:T)
	w_c = sum(x_true[t] * x_true[t] for t in 1:T)
	s_c = sum(x_true[t] * y[t] for t in 1:T)

	hss = HSS_ll(w_c, w_a, s_c, s_a)

	hpp = Priors_ll(0.01, 0.01, 0.01, 0.01, 0.0, 1.0)

	# should recover values of A, C, R close to ground truth
	exp_np = vb_m_ll(y, hss, hpp)

	println("r ", (1 ./ exp_np)[1]) # r = 0.2, q = 1.0
	println("q ", (1 ./ exp_np)[2])
end

# ╔═╡ 1adc874e-e024-464a-80d5-5ded04f62f24
md"""
### VB-E
#### Updating $q(x_{0:T})$

$\ln q(x_{0:T}) =  \langle \ln p(r, q, x_{0:T}, y_{1:T}) \rangle_{\hat q(r, q)} + c$

In the VB-E step, we compute the variational distribution $q(x_{0:T})$ for the latent states. This is done using the forward-backward algorithm, which is a dynamic programming algorithm 

We need the following expected parameters computed from vb-m step: 

$E_q[τ_r] = \alpha_r' / \beta_r'$
$E_q[τ_q] = \alpha_q' / \beta_q'$

#### Forward pass
The forward pass computes the filtered distribution for each latent state $x_t$ given the observations up to $y_{1:t}$. The filtered distributions are computed recursively from $t=1$ to $T$, much akin to the forward filter in FFBS, $p(x_t∣y_{1:t})$ 
"""

# ╔═╡ d359f3aa-b238-420f-99d2-52f85ce9ff82
function forward_ll(y, a, c, E_τ_r, E_τ_q, priors::Priors_ll)
    T = length(y)
    μ_f = zeros(T)
    σ_f2 = zeros(T)
	aa = zeros(T)
	rr = zeros(T)
	
	aa[1] = a * priors.μ_0
	rr[1] = a^2 * priors.σ_0 + 1/E_τ_q
	f_1 = c * aa[1]
	s_1 = c^2 * rr[1] + 1/E_τ_r
	
    μ_f[1] = aa[1] + rr[1] * c * (1/s_1) * (y[1] - f_1)
    σ_f2[1] = rr[1] - rr[1]^2 * c^2 * (1/s_1)
	
    for t = 2:T
        # Predict step
        μ_pred = a * μ_f[t-1]
        σ_pred2 = a^2 * σ_f2[t-1] + 1/E_τ_q

        # Update step
        K_t = σ_pred2 / (σ_pred2 + 1/E_τ_r)
        μ_f[t] = μ_pred + K_t * (y[t] - μ_pred)
        σ_f2[t] = (1 - K_t) * σ_pred2
    end
	
    return μ_f, σ_f2
end

# ╔═╡ bca920fc-9535-4eb0-89c2-03a7334df6b6
md"""
test forward
"""

# ╔═╡ 416a607b-26bc-4973-8c1a-489e855a06de
let
	true_exp_np = Exp_ϕ_uni(1.0, 1.0, 5.0, 1.0, 5.0, 5.0, 5.0)

	fm, σs, σs_ = v_forward(y, true_exp_np, 0.0, 1.0)

	E_τ_r = 5.0
	E_τ_q = 1.0
	hpp = Priors_ll(0.01, 0.01, 0.01, 0.01, 0.0, 1.0)

	f_ll, σs_ll = forward_ll(y, 1.0, 1.0, E_τ_r, E_τ_q, hpp)

	fm, f_ll, σs, σs_ll
end

# ╔═╡ ece11b32-cc61-447b-bbcf-018829360b73
md"""
#### Backward pass
The backward pass computes the smoothed distribution for each latent state $x_t$ given all the observations $y_{1:T}$. The smoothed distributions are computed recursively from $t = T ... 1$. $p(x_t∣y_{1:T})$ 

The recursion is initialized with the filtered distribution at time $T$, and it also uses the expected parameters computed in the VB-M step.
"""

# ╔═╡ c29b63f3-0d32-46ad-99a4-3cae4a3f6181
function backward_ll(μ_f, σ_f2, E_τ_q)
    T = length(μ_f)
    μ_s = similar(μ_f)
    σ_s2 = similar(σ_f2)
    σ_s2_cross = zeros(T)
    μ_s[T] = μ_f[T]
    σ_s2[T] = σ_f2[T]
    for t = T-1:-1:1
        # Compute the gain J_t
        J_t = σ_f2[t] / (σ_f2[t] + 1/E_τ_q)

        # Update the smoothed mean μ_s and variance σ_s2
        μ_s[t] = μ_f[t] + J_t * (μ_s[t+1] - μ_f[t])
        σ_s2[t] = σ_f2[t] + J_t^2 * (σ_s2[t+1] - σ_f2[t] - 1/E_τ_q)

        # Compute the cross variance σ_s2_cross
        σ_s2_cross[t+1] = J_t * σ_s2[t+1]
    end
	
    #J_0 = σ_f2[1] / (σ_f2[1] + 1/E_τ_q)
	J_0 = 1.0 / (1.0 + 1/E_τ_q)
	σ_s2_cross[1] = J_0 * σ_s2[1]
    return μ_s, σ_s2, σ_s2_cross
end

# ╔═╡ 135c6b95-c440-45ed-bade-0327bf1e142a
md"""
test backward
"""

# ╔═╡ 8e98a3b4-bc92-43ad-9da3-1323e06cfce6
let
	true_exp_np = Exp_ϕ_uni(1.0, 1.0, 5.0, 1.0, 5.0, 5.0, 5.0)
	μs, σs, _ =  v_forward(y, true_exp_np, 0.0, 1.0)
	ηs, Ψs, η_0, Ψ_0 = v_backward(y, true_exp_np)
	ωs, Υs, ω_0, Υ_0 = parallel_smoother(μs, σs, ηs, Ψs, η_0, Ψ_0, 0.0, 1.0)

	E_τ_r = 5.0
	E_τ_q = 1.0
	hpp = Priors_ll(0.01, 0.01, 0.01, 0.01, 0.0, 1.0)
	μ_f, σ_f2 = forward_ll(y, 1.0, 1.0, E_τ_r, E_τ_q, hpp)
	
	b_ll, σ_ll , _ = backward_ll(μ_f, σ_f2, E_τ_q)

	ωs, b_ll, Υs, σ_ll
end

# ╔═╡ aa5caae7-638d-441f-a306-5442a5c8f75f
md"""
#### Cross-variance

The term $\sigma_{t-1,t}^2$ represents the cross-variance between the latent state at time $t−1$ and the latent state at time $t$, under the variational distribution. This term can be computed from the joint marginal distribution $p(x_{t−1},x_t∣y_{1:T})$ 

"""

# ╔═╡ faed4326-5ee6-41da-9ba4-297e965c242e
md"""
test cross-variance
"""

# ╔═╡ 17136b27-9463-4e5f-a943-d78297f28be7
let
	true_exp_np = Exp_ϕ_uni(1.0, 1.0, 5.0, 1.0, 5.0, 5.0, 5.0)
	μs, σs, σs_ =  v_forward(y, true_exp_np, 0.0, 1.0)
	ηs, Ψs, η_0, Ψ_0 = v_backward(y, true_exp_np)
	Υ_ₜ₋ₜ₊₁ = v_pairwise_x(σs_, true_exp_np, Ψs)

	E_τ_r = 5.0
	E_τ_q = 1.0
	hpp = Priors_ll(0.01, 0.01, 0.01, 0.01, 0.0, 1.0)
	μ_f, σ_f2 = forward_ll(y, 1.0, 1.0, E_τ_r, E_τ_q, hpp)
	_ , _ , css = backward_ll(μ_f, σ_f2, E_τ_q)

	Υ_ₜ₋ₜ₊₁, css
end

# ╔═╡ bee6469f-13a1-4bd8-8f14-f01e8405a949
function vb_e_ll(y, E_τ_r, E_τ_q, priors::Priors_ll)
    # Forward pass (filter)
    μs_f, σs_f2 = forward_ll(y, 1.0, 1.0, E_τ_r, E_τ_q, priors)

    # Backward pass (smoother)
    μs_s, σs_s2, σs_s2_cross = backward_ll(μs_f, σs_f2, E_τ_q)

    # Compute the sufficient statistics
    w_c = sum(σs_s2 .+ μs_s.^2)
    w_a = sum(σs_s2[1:end-1] .+ μs_s[1:end-1].^2)
    s_c = sum(y .* μs_s)
    s_a = sum(σs_s2_cross[1:end-1]) + sum(μs_s[1:end-1] .* μs_s[2:end])

    # Return the sufficient statistics in a HSS struct
    return HSS_ll(w_c, w_a, s_c, s_a)
end

# ╔═╡ 59554e03-ae31-4cc4-a6d1-c307f1f7bd9a
let
	E_τ_r = 5.0
	E_τ_q = 1.0
	hpp = Priors_ll(0.01, 0.01, 0.01, 0.01, 0.0, 1.0)
	
	hss_e = vb_e_ll(y, E_τ_r, E_τ_q, hpp)

	w_a = sum(x_true[t-1] * x_true[t-1] for t in 2:T)
	s_a = sum(x_true[t-1] * x_true[t] for t in 2:T)
	w_c = sum(x_true[t] * x_true[t] for t in 1:T)
	s_c = sum(x_true[t] * y[t] for t in 1:T)

	hss_t = HSS_ll(w_c, w_a, s_c, s_a)

	hss_e, hss_t
end

# ╔═╡ 7a6940ef-56b3-4cb4-bc6b-2c97625965cc
function vb_ll(y::Vector{Float64}, hpp::Priors_ll, max_iter=100)
	hss = HSS_ll(1.0, 1.0, 1.0, 1.0)
	E_τ_r, E_τ_q  = missing, missing
	
	for i in 1:max_iter
		E_τ_r, E_τ_q = vb_m_ll(y, hss, hpp)
				
		hss = vb_e_ll(y, E_τ_r, E_τ_q, hpp)
	end

	return 1/E_τ_r, 1/E_τ_q
end

# ╔═╡ 665c55c3-d4dc-4d13-9517-d1106ea6210f
begin
	hpp_ll = Priors_ll(0.01, 0.01, 0.01, 0.01, 0.0, 1.0)
	@time vb_ll(y, hpp_ll)
end

# ╔═╡ e6930d53-6652-4fea-9a01-a4c87b8058dc
md"""
## Preliminary results
"""

# ╔═╡ 0cce2e6d-4f19-4c50-a4b5-2835c3ed4401
function vb_ll_plot(y::Vector{Float64}, hpp::Priors_ll, max_iter=200)
	T = length(y)
	hss = HSS_ll(1.0, 1.0, 1.0, 1.0)
    history_R = Vector{Float64}(undef, max_iter)
	history_Q = Vector{Float64}(undef, max_iter)
	
	E_τ_r, E_τ_q  = missing, missing
	for i in 1:max_iter
		E_τ_r, E_τ_q = vb_m_ll(y, hss, hpp)
		hss = vb_e_ll(y, E_τ_r, E_τ_q, hpp)

       	history_R[i] = 1/E_τ_r
		history_Q[i] = 1/E_τ_q
	end
	
	p1 = plot(history_Q[20:end], title = "Q learning", legend = false)
	p2 = plot(history_R[20:end], title = "R learning", legend = false)
	plot(p1, p2, layout = (2, 1))
	
end

# ╔═╡ 1c023156-0634-456d-a959-65880fd60c34
let
	hpp_ll = Priors_ll(0.01, 0.01, 0.01, 0.01, 0.0, 1.0)
	vb_ll_plot(y, hpp_ll)
end

# ╔═╡ caa2e633-e044-417c-944c-6a0458475e4f
md"""
VB inference of unknown r, q in local level model
"""

# ╔═╡ 3e64e18a-6446-4fcc-a282-d3d6079e975a
let
	hpp_ll = Priors_ll(0.01, 0.01, 0.01, 0.01, 0.0, 1.0)
	@time r, q = vb_ll(y, hpp_ll, 50)

	μs_f, σs_f2 = forward_ll(y, 1.0, 1.0, 1/r, 1/q, hpp_ll)
    μs_s, σs_s2, _ = backward_ll(μs_f, σs_f2, 1/q)
	
	println("\nlatent x error: " , error_metrics(x_true, μs_s))
end

# ╔═╡ 23301a45-fc89-416f-bcc7-4f3ba41bf04f
md"""
This is considerably faster than Gibbs sampling that takes ~0.05 second to converge (already fast). and latent x inference is on par with end-chain accuracy of Gibbs sampling.
"""

# ╔═╡ d57adc7b-fcd5-468b-aa50-919d884a916a
let
	Random.seed!(123)
	# fix a = c = 1.0 for local level model
	@time x_ss, q_ss, r_ss = gibbs_ll(y, 1.0, 1.0)

	println("\nmean system noise (q): ", mean(q_ss))
	println("mean observation noise (r): ", mean(r_ss))

	x_m = mean(x_ss, dims=1)[1,:]
	println("\naverage x sample error ", error_metrics(x_true, x_ss[end,: ]))
	println("end chain x sample error" , error_metrics(x_true, x_m))
	plot(q_ss, title = "MCMC-Gibbs", label="q samples")
	plot!(r_ss, label="r samples")
end

# ╔═╡ 492c0922-e5f0-435b-9372-27e79570d679
let
	x_ss, q_ss, r_ss = gibbs_ll(y, 1.0, 1.0)
	# Select the first 50 time steps
	true_latent_50 = x_true[1:50]
	sm_sampled_latent_50 = x_ss[:, 1:50]
	
	# Create a new plot
	p = plot()
	# Plot the true latent states with a thick line
	plot!(p, true_latent_50, linewidth=1.5, alpha=2, label="True x", color=:blue)
	
	# Plot the sampled latent states with a thin line
	for i in 1:size(sm_sampled_latent_50, 1)
	    plot!(p, sm_sampled_latent_50[i, :], linewidth=0.1, alpha=0.1, label=false, color=:violet)
	end
	p
end

# ╔═╡ d2efd2a0-f494-4486-8ed3-ffd944b8473f
md"""
# Extras: Using NUTS() from Turing.jl
"""

# ╔═╡ 4c3c2931-e4a8-43d1-b3fa-8bb3b82fb975
begin
	@model function DLM_Turing(y, a, c)
	    T = length(y)
	    
	    # Priors
	    r ~ InverseGamma(0.1, 0.1)
	    q ~ InverseGamma(0.1, 0.1)
		
	    x = Vector(undef, T)
	    x[1] ~ Normal(0, 1.0)
	    y[1] ~ Normal(c * x[1], sqrt(r))
		
	    for t in 2:T
	        # State transition
	        x[t] ~ Normal(a * x[t-1], sqrt(q))
	        
	        # Observation model
	        y[t] ~ Normal(c * x[t], sqrt(r))
	    end
	end
	Random.seed!(888)
	model = DLM_Turing(y, 1.0, 1.0)
	chain = sample(model, NUTS(), 3000)
	chain = chain[100:end]
end;

# ╔═╡ 2ad03b07-68ad-4da8-a5d3-7692502c0e00
describe(chain)

# ╔═╡ 5f9f903b-a72c-4b60-9934-4bd2ced30a2c
md"""
Plot learning of $r, q$
"""

# ╔═╡ 54c537af-06c3-4e12-87b8-33d3f1efa77b
begin
	r_samples = chain[:r]
	q_samples = chain[:q]
	
	# Plot the trace of r and q
	p1 = plot(r_samples, title = "Trace for r", legend = false)
	p2 = plot(q_samples, title = "Trace for q", legend = false)
	plot(p1, p2, layout = (2, 1))
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
MCMCChains = "c7f686f2-ff18-58e9-bc7b-31028e88f75d"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
StateSpaceModels = "99342f36-827c-5390-97c9-d7f9ee765c78"
Turing = "fce5fe82-541a-59a6-adf8-730c64b5f9a0"

[compat]
Distributions = "~0.25.90"
MCMCChains = "~6.0.3"
Plots = "~1.38.11"
PlutoUI = "~0.7.51"
StateSpaceModels = "~0.6.6"
Turing = "~0.26.2"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.9.1"
manifest_format = "2.0"
project_hash = "2ffd268c344238bf57ad2e5b1b1c7b10501acb33"

[[deps.ADTypes]]
git-tree-sha1 = "dcfdf328328f2645531c4ddebf841228aef74130"
uuid = "47edcb42-4c32-4615-8424-f2b9edc5f35b"
version = "0.1.3"

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

[[deps.AbstractPPL]]
deps = ["AbstractMCMC", "DensityInterface", "Random", "Setfield", "SparseArrays"]
git-tree-sha1 = "33ea6c6837332395dbf3ba336f273c9f7fcf4db9"
uuid = "7a57a42e-76ec-4ea3-a279-07e840d6d9cf"
version = "0.5.4"

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
git-tree-sha1 = "cc37d689f599e8df4f464b2fa3870ff7db7492ef"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.6.1"
weakdeps = ["StaticArrays"]

    [deps.Adapt.extensions]
    AdaptStaticArraysExt = "StaticArrays"

[[deps.AdvancedHMC]]
deps = ["AbstractMCMC", "ArgCheck", "DocStringExtensions", "InplaceOps", "LinearAlgebra", "LogDensityProblems", "LogDensityProblemsAD", "ProgressMeter", "Random", "Requires", "Setfield", "SimpleUnPack", "Statistics", "StatsBase", "StatsFuns"]
git-tree-sha1 = "3bf24030e85b1d6d298e4f483f6aeff6f38462db"
uuid = "0bf59076-c3b1-5ca4-86bd-e02cd72cde3d"
version = "0.4.6"

    [deps.AdvancedHMC.extensions]
    AdvancedHMCCUDAExt = "CUDA"
    AdvancedHMCMCMCChainsExt = "MCMCChains"
    AdvancedHMCOrdinaryDiffEqExt = "OrdinaryDiffEq"

    [deps.AdvancedHMC.weakdeps]
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    MCMCChains = "c7f686f2-ff18-58e9-bc7b-31028e88f75d"
    OrdinaryDiffEq = "1dea7af3-3e70-54e6-95c3-0bf5283fa5ed"

[[deps.AdvancedMH]]
deps = ["AbstractMCMC", "Distributions", "LogDensityProblems", "Random", "Requires"]
git-tree-sha1 = "165af834eee68d0a96c58daa950ddf0b3f45f608"
uuid = "5b7e9947-ddc0-4b3f-9b55-0d8042f74170"
version = "0.7.4"
weakdeps = ["DiffResults", "ForwardDiff", "MCMCChains", "StructArrays"]

    [deps.AdvancedMH.extensions]
    AdvancedMHForwardDiffExt = ["DiffResults", "ForwardDiff"]
    AdvancedMHMCMCChainsExt = "MCMCChains"
    AdvancedMHStructArraysExt = "StructArrays"

[[deps.AdvancedPS]]
deps = ["AbstractMCMC", "Distributions", "Libtask", "Random", "Random123", "StatsFuns"]
git-tree-sha1 = "4d73400b3583147b1b639794696c78202a226584"
uuid = "576499cb-2369-40b2-a588-c64705576edc"
version = "0.4.3"

[[deps.AdvancedVI]]
deps = ["Bijectors", "Distributions", "DistributionsAD", "DocStringExtensions", "ForwardDiff", "LinearAlgebra", "ProgressMeter", "Random", "Requires", "StatsBase", "StatsFuns", "Tracker"]
git-tree-sha1 = "1f919a9c59cf3dfc68b64c22c453a2e356fca473"
uuid = "b5ca4192-6429-45e5-a2d9-87aec30a685c"
version = "0.2.4"

[[deps.ArgCheck]]
git-tree-sha1 = "a3a402a35a2f7e0b87828ccabbd5ebfbebe356b4"
uuid = "dce04be8-c92d-5529-be00-80e4d2c0e197"
version = "2.3.0"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.ArrayInterface]]
deps = ["Adapt", "LinearAlgebra", "Requires", "SnoopPrecompile", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "38911c7737e123b28182d89027f4216cfc8a9da7"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "7.4.3"

    [deps.ArrayInterface.extensions]
    ArrayInterfaceBandedMatricesExt = "BandedMatrices"
    ArrayInterfaceBlockBandedMatricesExt = "BlockBandedMatrices"
    ArrayInterfaceCUDAExt = "CUDA"
    ArrayInterfaceGPUArraysCoreExt = "GPUArraysCore"
    ArrayInterfaceStaticArraysCoreExt = "StaticArraysCore"
    ArrayInterfaceTrackerExt = "Tracker"

    [deps.ArrayInterface.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockBandedMatrices = "ffab5731-97b5-5995-9138-79e8c1846df0"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    StaticArraysCore = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Atomix]]
deps = ["UnsafeAtomics"]
git-tree-sha1 = "c06a868224ecba914baa6942988e2f2aade419be"
uuid = "a9b6321e-bd34-4604-b9c9-b65b8de01458"
version = "0.1.0"

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

[[deps.Bijectors]]
deps = ["ArgCheck", "ChainRulesCore", "ChangesOfVariables", "Compat", "Distributions", "Functors", "InverseFunctions", "IrrationalConstants", "LinearAlgebra", "LogExpFunctions", "MappedArrays", "Random", "Reexport", "Requires", "Roots", "SparseArrays", "Statistics"]
git-tree-sha1 = "ff192d037dee3c05fe842a207f8c6b840b04cca2"
uuid = "76274a88-744f-5084-9051-94815aaf08c4"
version = "0.12.8"

    [deps.Bijectors.extensions]
    BijectorsDistributionsADExt = "DistributionsAD"
    BijectorsForwardDiffExt = "ForwardDiff"
    BijectorsLazyArraysExt = "LazyArrays"
    BijectorsReverseDiffExt = "ReverseDiff"
    BijectorsTrackerExt = "Tracker"
    BijectorsZygoteExt = "Zygote"

    [deps.Bijectors.weakdeps]
    DistributionsAD = "ced4e74d-a319-5a8a-b0ac-84af2272839c"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    LazyArrays = "5078a376-72f3-5289-bfd5-ec5146d43c02"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[[deps.BitFlags]]
git-tree-sha1 = "43b1a4a8f797c1cddadf60499a8a077d4af2cd2d"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.7"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[deps.CEnum]]
git-tree-sha1 = "eb4cb44a499229b3b8426dcfb5dd85333951ff90"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.4.2"

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

[[deps.ChainRules]]
deps = ["Adapt", "ChainRulesCore", "Compat", "Distributed", "GPUArraysCore", "IrrationalConstants", "LinearAlgebra", "Random", "RealDot", "SparseArrays", "Statistics", "StructArrays"]
git-tree-sha1 = "61549d9b52c88df34d21bd306dba1d43bb039c87"
uuid = "082447d4-558c-5d27-93f4-14fc19e9eca2"
version = "1.51.0"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "e30f2f4e20f7f186dc36529910beaedc60cfa644"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.16.0"

[[deps.ChangesOfVariables]]
deps = ["LinearAlgebra", "Test"]
git-tree-sha1 = "f84967c4497e0e1955f9a582c232b02847c5f589"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.7"

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

[[deps.Combinatorics]]
git-tree-sha1 = "08c8b6831dc00bfea825826be0bc8336fc369860"
uuid = "861a8166-3701-5b0c-9a16-15d98fcdc6aa"
version = "1.0.2"

[[deps.CommonSolve]]
git-tree-sha1 = "0eee5eb66b1cf62cd6ad1b460238e60e4b09400c"
uuid = "38540f10-b2f7-11e9-35d8-d573e4eb0ff2"
version = "0.2.4"

[[deps.CommonSubexpressions]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "7b8a93dba8af7e3b42fecabf646260105ac373f7"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.0"

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

[[deps.CompositionsBase]]
git-tree-sha1 = "802bb88cd69dfd1509f6670416bd4434015693ad"
uuid = "a33af91c-f02d-484b-be07-31d278c5ca2b"
version = "0.1.2"
weakdeps = ["InverseFunctions"]

    [deps.CompositionsBase.extensions]
    CompositionsBaseInverseFunctionsExt = "InverseFunctions"

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
git-tree-sha1 = "e8119c1a33d267e16108be441a287a6981ba1630"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.14.0"

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

[[deps.DensityInterface]]
deps = ["InverseFunctions", "Test"]
git-tree-sha1 = "80c3e8639e3353e5d2912fb3a1916b8455e2494b"
uuid = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
version = "0.4.0"

[[deps.DiffResults]]
deps = ["StaticArraysCore"]
git-tree-sha1 = "782dd5f4561f5d267313f23853baaaa4c52ea621"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.1.0"

[[deps.DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "a4ad7ef19d2cdc2eff57abbbe68032b1cd0bd8f8"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.13.0"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.Distributions]]
deps = ["FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns", "Test"]
git-tree-sha1 = "eead66061583b6807652281c0fbf291d7a9dc497"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.90"
weakdeps = ["ChainRulesCore", "DensityInterface"]

    [deps.Distributions.extensions]
    DistributionsChainRulesCoreExt = "ChainRulesCore"
    DistributionsDensityInterfaceExt = "DensityInterface"

[[deps.DistributionsAD]]
deps = ["Adapt", "ChainRules", "ChainRulesCore", "Compat", "Distributions", "FillArrays", "LinearAlgebra", "PDMats", "Random", "Requires", "SpecialFunctions", "StaticArrays", "StatsFuns", "ZygoteRules"]
git-tree-sha1 = "1fb4fedd5a407243d535cc50c8c803c2fce0dc26"
uuid = "ced4e74d-a319-5a8a-b0ac-84af2272839c"
version = "0.6.49"

    [deps.DistributionsAD.extensions]
    DistributionsADForwardDiffExt = "ForwardDiff"
    DistributionsADLazyArraysExt = "LazyArrays"
    DistributionsADReverseDiffExt = "ReverseDiff"
    DistributionsADTrackerExt = "Tracker"

    [deps.DistributionsAD.weakdeps]
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    LazyArrays = "5078a376-72f3-5289-bfd5-ec5146d43c02"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

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

[[deps.DynamicPPL]]
deps = ["AbstractMCMC", "AbstractPPL", "BangBang", "Bijectors", "ChainRulesCore", "ConstructionBase", "Distributions", "DocStringExtensions", "LinearAlgebra", "LogDensityProblems", "MacroTools", "OrderedCollections", "Random", "Setfield", "Test", "ZygoteRules"]
git-tree-sha1 = "e3518c62a998defe74c5cc64b2bf2a33b7a788b8"
uuid = "366bfd00-2699-11ea-058f-f148b4cae6d8"
version = "0.23.0"

[[deps.EllipticalSliceSampling]]
deps = ["AbstractMCMC", "ArrayInterface", "Distributions", "Random", "Statistics"]
git-tree-sha1 = "973b4927d112559dc737f55d6bf06503a5b3fc14"
uuid = "cad2338a-1db2-11e9-3401-43bc07c9ede2"
version = "1.1.0"

[[deps.EnumX]]
git-tree-sha1 = "bdb1942cd4c45e3c678fd11569d5cccd80976237"
uuid = "4e289a0a-7415-4d19-859d-a7e5c4648b56"
version = "1.0.4"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bad72f730e9e91c08d9427d5e8db95478a3c323d"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.4.8+0"

[[deps.ExprTools]]
git-tree-sha1 = "c1d06d129da9f55715c6c212866f5b1bddc5fa00"
uuid = "e2ba6199-217a-4e67-a87a-7c52f15ade04"
version = "0.1.9"

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
git-tree-sha1 = "fc86b4fd3eff76c3ce4f5e96e2fdfa6282722885"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.0.0"

[[deps.FiniteDiff]]
deps = ["ArrayInterface", "LinearAlgebra", "Requires", "Setfield", "SparseArrays", "StaticArrays"]
git-tree-sha1 = "6604e18a0220650dbbea7854938768f15955dd8e"
uuid = "6a86dc24-6348-571c-b903-95158fe2bd41"
version = "2.20.0"

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

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions"]
git-tree-sha1 = "00e252f4d706b3d55a8863432e742bf5717b498d"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.35"
weakdeps = ["StaticArrays"]

    [deps.ForwardDiff.extensions]
    ForwardDiffStaticArraysExt = "StaticArrays"

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

[[deps.FunctionWrappers]]
git-tree-sha1 = "d62485945ce5ae9c0c48f124a84998d755bae00e"
uuid = "069b7b12-0de2-55c6-9aab-29f3d0a68a2e"
version = "1.1.3"

[[deps.FunctionWrappersWrappers]]
deps = ["FunctionWrappers"]
git-tree-sha1 = "b104d487b34566608f8b4e1c39fb0b10aa279ff8"
uuid = "77dc65aa-8811-40c2-897b-53d922fa7daf"
version = "0.1.3"

[[deps.Functors]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "478f8c3145bb91d82c2cf20433e8c1b30df454cc"
uuid = "d9f16b24-f501-4c13-a1f2-28368ffc5196"
version = "0.4.4"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pkg", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "d972031d28c8c8d9d7b41a536ad7bb0c2579caca"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.8+0"

[[deps.GPUArraysCore]]
deps = ["Adapt"]
git-tree-sha1 = "2d6ca471a6c7b536127afccfa7564b5b39227fe0"
uuid = "46192b85-c4d5-4398-a991-12ede77f4527"
version = "0.1.5"

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

[[deps.InitialValues]]
git-tree-sha1 = "4da0f88e9a39111c2fa3add390ab15f3a44f3ca3"
uuid = "22cec73e-a1b8-11e9-2c92-598750a2cf9c"
version = "0.3.1"

[[deps.InlineStrings]]
deps = ["Parsers"]
git-tree-sha1 = "9cc2baf75c6d09f9da536ddf58eb2f29dedaf461"
uuid = "842dd82b-1e85-43dc-bf29-5d0ee9dffc48"
version = "1.4.0"

[[deps.InplaceOps]]
deps = ["LinearAlgebra", "Test"]
git-tree-sha1 = "50b41d59e7164ab6fda65e71049fee9d890731ff"
uuid = "505f98c9-085e-5b2c-8e89-488be7bf1f34"
version = "0.3.0"

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

[[deps.Intervals]]
deps = ["Dates", "Printf", "RecipesBase", "Serialization", "TimeZones"]
git-tree-sha1 = "d136858cb539cd6484b15f0e28a761e8d63d6fb3"
uuid = "d8418881-c3e1-53bb-8760-2df7ec849ed5"
version = "1.9.0"

[[deps.InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "6667aadd1cdee2c6cd068128b3d226ebc4fb0c67"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.9"

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

[[deps.KernelAbstractions]]
deps = ["Adapt", "Atomix", "InteractiveUtils", "LinearAlgebra", "MacroTools", "PrecompileTools", "SparseArrays", "StaticArrays", "UUIDs", "UnsafeAtomics", "UnsafeAtomicsLLVM"]
git-tree-sha1 = "b48617c5d764908b5fac493cd907cf33cc11eec1"
uuid = "63c18a36-062a-441e-b654-da1e3ab1ce7c"
version = "0.9.6"

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

[[deps.LLVM]]
deps = ["CEnum", "LLVMExtra_jll", "Libdl", "Printf", "Unicode"]
git-tree-sha1 = "5007c1421563108110bbd57f63d8ad4565808818"
uuid = "929cbde3-209d-540e-8aea-75f648917ca0"
version = "5.2.0"

[[deps.LLVMExtra_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl", "TOML"]
git-tree-sha1 = "1222116d7313cdefecf3d45a2bc1a89c4e7c9217"
uuid = "dad2f222-ce93-54a1-a47d-0025e8a3acab"
version = "0.0.22+0"

[[deps.LRUCache]]
git-tree-sha1 = "48c10e3cc27e30de82463c27bef0b8bdbd1dc634"
uuid = "8ac3fa9e-de4c-5943-b1dc-09c6b5f20637"
version = "1.4.1"

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

[[deps.Lazy]]
deps = ["MacroTools"]
git-tree-sha1 = "1370f8202dac30758f3c345f9909b97f53d87d3f"
uuid = "50d2b5c4-7a5e-59d5-8109-a42b560f39c0"
version = "0.15.1"

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

[[deps.Libtask]]
deps = ["FunctionWrappers", "LRUCache", "LinearAlgebra", "Statistics"]
git-tree-sha1 = "345a40c746404dd9cb1bbc368715856838ab96f2"
uuid = "6f1fad26-d15e-5dc8-ae53-837a1d7b8c9f"
version = "0.8.6"

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

[[deps.LineSearches]]
deps = ["LinearAlgebra", "NLSolversBase", "NaNMath", "Parameters", "Printf"]
git-tree-sha1 = "7bbea35cec17305fc70a0e5b4641477dc0789d9d"
uuid = "d3d80556-e9d4-5f37-9878-2ab0fcc64255"
version = "7.2.0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LinearMaps]]
deps = ["ChainRulesCore", "LinearAlgebra", "SparseArrays", "Statistics"]
git-tree-sha1 = "4af48c3585177561e9f0d24eb9619ad3abf77cc7"
uuid = "7a12625a-238d-50fd-b39a-03d52299707e"
version = "3.10.0"

[[deps.LogDensityProblems]]
deps = ["ArgCheck", "DocStringExtensions", "Random"]
git-tree-sha1 = "f9a11237204bc137617194d79d813069838fcf61"
uuid = "6fdf6af0-433a-55f7-b3ed-c6c6e0b8df7c"
version = "2.1.1"

[[deps.LogDensityProblemsAD]]
deps = ["DocStringExtensions", "LogDensityProblems", "Requires", "SimpleUnPack"]
git-tree-sha1 = "a0512ad65f849536b5a52e59b05c59c25cdad943"
uuid = "996a588d-648d-4e1f-a8f0-a84b347e47b1"
version = "1.5.0"

    [deps.LogDensityProblemsAD.extensions]
    LogDensityProblemsADEnzymeExt = "Enzyme"
    LogDensityProblemsADFiniteDifferencesExt = "FiniteDifferences"
    LogDensityProblemsADForwardDiffBenchmarkToolsExt = ["BenchmarkTools", "ForwardDiff"]
    LogDensityProblemsADForwardDiffExt = "ForwardDiff"
    LogDensityProblemsADReverseDiffExt = "ReverseDiff"
    LogDensityProblemsADTrackerExt = "Tracker"
    LogDensityProblemsADZygoteExt = "Zygote"

    [deps.LogDensityProblemsAD.weakdeps]
    BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"
    FiniteDifferences = "26cc04aa-876d-5657-8c51-4c34ba976000"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "0a1b7c2863e44523180fdb3146534e265a91870b"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.23"
weakdeps = ["ChainRulesCore", "ChangesOfVariables", "InverseFunctions"]

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

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

[[deps.MappedArrays]]
git-tree-sha1 = "2dab0221fe2b0f2cb6754eaa743cc266339f527e"
uuid = "dbb5928d-eab1-5f90-85c2-b9b0edb7c900"
version = "0.4.2"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MatrixEquations]]
deps = ["LinearAlgebra", "LinearMaps"]
git-tree-sha1 = "15838905f0d1eb5d450b12986e64861a82fd6e03"
uuid = "99c1a7ee-ab34-5fd5-8076-27c950a045f4"
version = "2.2.9"

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

[[deps.Mocking]]
deps = ["Compat", "ExprTools"]
git-tree-sha1 = "782e258e80d68a73d8c916e55f8ced1de00c2cea"
uuid = "78c3b35d-d492-501b-9361-3d52fe80e533"
version = "0.7.6"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2022.10.11"

[[deps.MutableArithmetics]]
deps = ["LinearAlgebra", "SparseArrays", "Test"]
git-tree-sha1 = "964cb1a7069723727025ae295408747a0b36a854"
uuid = "d8a4904e-b15c-11e9-3269-09a3773c0cb0"
version = "1.3.0"

[[deps.NLSolversBase]]
deps = ["DiffResults", "Distributed", "FiniteDiff", "ForwardDiff"]
git-tree-sha1 = "a0b464d183da839699f4c79e7606d9d186ec172c"
uuid = "d41bc354-129a-5804-8e4c-c37616107c6c"
version = "7.8.3"

[[deps.NNlib]]
deps = ["Adapt", "Atomix", "ChainRulesCore", "GPUArraysCore", "KernelAbstractions", "LinearAlgebra", "Pkg", "Random", "Requires", "Statistics"]
git-tree-sha1 = "72240e3f5ca031937bd536182cb2c031da5f46dd"
uuid = "872c559c-99b0-510c-b3b7-b6c96a88d5cd"
version = "0.8.21"

    [deps.NNlib.extensions]
    NNlibAMDGPUExt = "AMDGPU"

    [deps.NNlib.weakdeps]
    AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "0877504529a3e5c3343c6f8b4c0381e57e4387e4"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.2"

[[deps.NamedArrays]]
deps = ["Combinatorics", "DataStructures", "DelimitedFiles", "InvertedIndices", "LinearAlgebra", "Random", "Requires", "SparseArrays", "Statistics"]
git-tree-sha1 = "b84e17976a40cb2bfe3ae7edb3673a8c630d4f95"
uuid = "86f7a689-2022-50b4-a561-43c23ac3c673"
version = "0.9.8"

[[deps.NaturalSort]]
git-tree-sha1 = "eda490d06b9f7c00752ee81cfa451efe55521e21"
uuid = "c020b1a1-e9b0-503a-9c33-f039bfc54a85"
version = "1.0.0"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

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
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9ff31d101d987eb9d66bd8b176ac7c277beccd09"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.20+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.Optim]]
deps = ["Compat", "FillArrays", "ForwardDiff", "LineSearches", "LinearAlgebra", "NLSolversBase", "NaNMath", "Parameters", "PositiveFactorizations", "Printf", "SparseArrays", "StatsBase"]
git-tree-sha1 = "a89b11f0f354f06099e4001c151dffad7ebab015"
uuid = "429524aa-4258-5aef-a3af-852621145aeb"
version = "1.7.5"

[[deps.Optimisers]]
deps = ["ChainRulesCore", "Functors", "LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "6a01f65dd8583dee82eecc2a19b0ff21521aa749"
uuid = "3bd65402-5787-11e9-1adc-39752487f4e2"
version = "0.2.18"

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

[[deps.Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

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

[[deps.Polynomials]]
deps = ["Intervals", "LinearAlgebra", "MutableArithmetics", "RecipesBase"]
git-tree-sha1 = "a1f7f4e41404bed760213ca01d7f384319f717a5"
uuid = "f27b6e38-b328-58d1-80ce-0feddd5e7a45"
version = "2.0.25"

[[deps.PositiveFactorizations]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "17275485f373e6673f7e7f97051f703ed5b15b20"
uuid = "85a6dd25-e78a-55b7-8502-1745935b8125"
version = "0.2.4"

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

[[deps.Random123]]
deps = ["Random", "RandomNumbers"]
git-tree-sha1 = "552f30e847641591ba3f39fd1bed559b9deb0ef3"
uuid = "74087812-796a-5b5d-8853-05524746bad3"
version = "1.6.1"

[[deps.RandomNumbers]]
deps = ["Random", "Requires"]
git-tree-sha1 = "043da614cc7e95c703498a491e2c21f58a2b8111"
uuid = "e6cf234a-135c-5ec9-84dd-332b85af5143"
version = "1.5.3"

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

[[deps.RealDot]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "9f0a1b71baaf7650f4fa8a1d168c7fb6ee41f0c9"
uuid = "c1ae055f-0cd5-4b69-90a6-9a35b1a98df9"
version = "0.1.0"

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

[[deps.RecursiveArrayTools]]
deps = ["Adapt", "ArrayInterface", "DocStringExtensions", "GPUArraysCore", "IteratorInterfaceExtensions", "LinearAlgebra", "RecipesBase", "Requires", "StaticArraysCore", "Statistics", "SymbolicIndexingInterface", "Tables"]
git-tree-sha1 = "02ef02926f30d53b94be443bfaea010c47f6b556"
uuid = "731186ca-8d62-57ce-b412-fbd966d074cd"
version = "2.38.5"

    [deps.RecursiveArrayTools.extensions]
    RecursiveArrayToolsMeasurementsExt = "Measurements"
    RecursiveArrayToolsTrackerExt = "Tracker"
    RecursiveArrayToolsZygoteExt = "Zygote"

    [deps.RecursiveArrayTools.weakdeps]
    Measurements = "eff96d63-e80a-5855-80a2-b1b0885c5ab7"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

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

[[deps.Roots]]
deps = ["ChainRulesCore", "CommonSolve", "Printf", "Setfield"]
git-tree-sha1 = "de432823e8aab4dd1a985be4be768f95acf152d4"
uuid = "f2b01f46-fcfa-551c-844a-d8ac1e96c665"
version = "2.0.17"

    [deps.Roots.extensions]
    RootsForwardDiffExt = "ForwardDiff"
    RootsIntervalRootFindingExt = "IntervalRootFinding"
    RootsSymPyExt = "SymPy"

    [deps.Roots.weakdeps]
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    IntervalRootFinding = "d2bf35a9-74e0-55ec-b149-d360ff49b807"
    SymPy = "24249f21-da20-56a4-8eb1-6a02cf4ae2e6"

[[deps.RuntimeGeneratedFunctions]]
deps = ["ExprTools", "SHA", "Serialization"]
git-tree-sha1 = "237edc1563bbf078629b4f8d194bd334e97907cf"
uuid = "7e49a35a-f44a-4d26-94aa-eba1b4ca6b47"
version = "0.5.11"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.SciMLBase]]
deps = ["ADTypes", "ArrayInterface", "CommonSolve", "ConstructionBase", "Distributed", "DocStringExtensions", "EnumX", "FunctionWrappersWrappers", "IteratorInterfaceExtensions", "LinearAlgebra", "Logging", "Markdown", "PrecompileTools", "Preferences", "RecipesBase", "RecursiveArrayTools", "Reexport", "RuntimeGeneratedFunctions", "SciMLOperators", "StaticArraysCore", "Statistics", "SymbolicIndexingInterface", "Tables", "TruncatedStacktraces"]
git-tree-sha1 = "a22a12db91f6a921e28a7ae39a9546eed93fd92e"
uuid = "0bca4576-84f4-4d90-8ffe-ffa030f20462"
version = "1.93.0"

[[deps.SciMLOperators]]
deps = ["ArrayInterface", "DocStringExtensions", "Lazy", "LinearAlgebra", "Setfield", "SparseArrays", "StaticArraysCore", "Tricks"]
git-tree-sha1 = "660322732becf934bf818792f9740984b375d300"
uuid = "c0aeaf25-5076-4817-a8d5-81caf7dfa961"
version = "0.3.1"

[[deps.ScientificTypesBase]]
git-tree-sha1 = "a8e18eb383b5ecf1b5e6fc237eb39255044fd92b"
uuid = "30f210dd-8aff-4c5f-94ba-8e64358c1161"
version = "3.0.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "30449ee12237627992a99d5e30ae63e4d78cd24a"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.2.0"

[[deps.SeasonalTrendLoess]]
deps = ["Statistics"]
git-tree-sha1 = "839dcd8152dc20663349781f7a7e8cf3d3009673"
uuid = "42fb36cb-998a-4034-bf40-4eee476c43a1"
version = "0.1.0"

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

[[deps.ShiftedArrays]]
git-tree-sha1 = "22395afdcf37d6709a5a0766cc4a5ca52cb85ea0"
uuid = "1277b4bf-5013-50f5-be3d-901d8477a67a"
version = "1.0.0"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.SimpleBufferStream]]
git-tree-sha1 = "874e8867b33a00e784c8a7e4b60afe9e037b74e1"
uuid = "777ac1f9-54b0-4bf8-805c-2214025038e7"
version = "1.1.0"

[[deps.SimpleUnPack]]
git-tree-sha1 = "58e6353e72cde29b90a69527e56df1b5c3d8c437"
uuid = "ce78b400-467f-4804-87d8-8f486da07d0a"
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
weakdeps = ["ChainRulesCore"]

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

[[deps.SplittablesBase]]
deps = ["Setfield", "Test"]
git-tree-sha1 = "e08a62abc517eb79667d0a29dc08a3b589516bb5"
uuid = "171d559e-b47b-412a-8079-5efa626c420e"
version = "0.1.15"

[[deps.StateSpaceModels]]
deps = ["Distributions", "LinearAlgebra", "MatrixEquations", "Optim", "OrderedCollections", "Polynomials", "Printf", "RecipesBase", "SeasonalTrendLoess", "ShiftedArrays", "SparseArrays", "Statistics", "StatsBase"]
git-tree-sha1 = "a84cedd2cae8654fe4f78c31f7c2f36b78dd8254"
uuid = "99342f36-827c-5390-97c9-d7f9ee765c78"
version = "0.6.6"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "Random", "StaticArraysCore", "Statistics"]
git-tree-sha1 = "c262c8e978048c2b095be1672c9bee55b4619521"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.5.24"

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
git-tree-sha1 = "d1bf48bfcc554a3761a133fe3a9bb01488e06916"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.21"

[[deps.StatsFuns]]
deps = ["HypergeometricFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "f625d686d5a88bcd2b15cd81f18f98186fdc0c9a"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.3.0"
weakdeps = ["ChainRulesCore", "InverseFunctions"]

    [deps.StatsFuns.extensions]
    StatsFunsChainRulesCoreExt = "ChainRulesCore"
    StatsFunsInverseFunctionsExt = "InverseFunctions"

[[deps.StringManipulation]]
git-tree-sha1 = "46da2434b41f41ac3594ee9816ce5541c6096123"
uuid = "892a3eda-7b42-436c-8928-eab12a02cf0e"
version = "0.3.0"

[[deps.StructArrays]]
deps = ["Adapt", "DataAPI", "GPUArraysCore", "StaticArraysCore", "Tables"]
git-tree-sha1 = "521a0e828e98bb69042fec1809c1b5a680eb7389"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.15"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "Pkg", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "5.10.1+6"

[[deps.SymbolicIndexingInterface]]
deps = ["DocStringExtensions"]
git-tree-sha1 = "f8ab052bfcbdb9b48fad2c80c873aa0d0344dfe5"
uuid = "2efcf032-c050-4f8e-a9bb-153293bab1f5"
version = "0.2.2"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

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

[[deps.TimeZones]]
deps = ["Dates", "Downloads", "InlineStrings", "LazyArtifacts", "Mocking", "Printf", "RecipesBase", "Scratch", "Unicode"]
git-tree-sha1 = "a5404eddfee0cf451cabb8ea8846413323712e25"
uuid = "f269a46b-ccf7-5d73-abea-4c690281aa53"
version = "1.9.2"

[[deps.Tracker]]
deps = ["Adapt", "DiffRules", "ForwardDiff", "Functors", "LinearAlgebra", "LogExpFunctions", "MacroTools", "NNlib", "NaNMath", "Optimisers", "Printf", "Random", "Requires", "SpecialFunctions", "Statistics"]
git-tree-sha1 = "8b552cc0a4132c1ce5cee14197bb57d2109d480f"
uuid = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
version = "0.2.25"
weakdeps = ["PDMats"]

    [deps.Tracker.extensions]
    TrackerPDMatsExt = "PDMats"

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

[[deps.TruncatedStacktraces]]
deps = ["InteractiveUtils", "MacroTools", "Preferences"]
git-tree-sha1 = "7bc1632a4eafbe9bd94cf1a784a9a4eb5e040a91"
uuid = "781d530d-4396-4725-bb49-402e4bee1e77"
version = "1.3.0"

[[deps.Turing]]
deps = ["AbstractMCMC", "AdvancedHMC", "AdvancedMH", "AdvancedPS", "AdvancedVI", "BangBang", "Bijectors", "DataStructures", "Distributions", "DistributionsAD", "DocStringExtensions", "DynamicPPL", "EllipticalSliceSampling", "FillArrays", "ForwardDiff", "Libtask", "LinearAlgebra", "LogDensityProblems", "LogDensityProblemsAD", "MCMCChains", "NamedArrays", "Printf", "Random", "Reexport", "Requires", "SciMLBase", "Setfield", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns", "Tracker"]
git-tree-sha1 = "f262c5a9f1d82011636f4c5e61bf6bdda094577f"
uuid = "fce5fe82-541a-59a6-adf8-730c64b5f9a0"
version = "0.26.2"

[[deps.URIs]]
git-tree-sha1 = "074f993b0ca030848b897beff716d93aca60f06a"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.4.2"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.UnsafeAtomics]]
git-tree-sha1 = "6331ac3440856ea1988316b46045303bef658278"
uuid = "013be700-e6cd-48c3-b4a1-df204f14c38f"
version = "0.2.1"

[[deps.UnsafeAtomicsLLVM]]
deps = ["LLVM", "UnsafeAtomics"]
git-tree-sha1 = "ea37e6066bf194ab78f4e747f5245261f17a7175"
uuid = "d80eeb9a-aca5-4d75-85e5-170c8b632249"
version = "0.1.2"

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

[[deps.ZygoteRules]]
deps = ["ChainRulesCore", "MacroTools"]
git-tree-sha1 = "977aed5d006b840e2e40c0b48984f7463109046d"
uuid = "700de1a5-db45-46bc-99cf-38207098b444"
version = "0.2.3"

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
# ╠═bd315098-f0d4-11ed-21f8-2b6e59a2ee9c
# ╟─a4e3933c-f487-4447-8b94-dcb58eaec886
# ╟─e56e975c-ccdd-40c1-8c32-30cab01fe5ce
# ╟─ab8b0d77-9eb0-4eb7-b047-737fec53bc39
# ╠═a7d4f30b-d21e-400e-ba13-a58fdb424fa6
# ╟─d1ab75c8-5d7e-4d42-9b54-c48cd0e61e90
# ╠═f108fc16-b3c2-4a89-87e8-f735a1d04301
# ╟─05be6f30-1936-40b1-83cc-743f8704610e
# ╟─99198a4a-6322-4c0f-be9f-24c9bb86f2ca
# ╠═d7a5cb1a-c768-4aed-beec-eaad552735b6
# ╟─cd96215e-5ec9-4031-b2e8-39d76b5e5bee
# ╠═7db0f68b-f27f-4529-961a-7b9a9c1b7500
# ╠═a78d2f18-af12-4285-944f-4297a69f2369
# ╟─3f882bc0-b27f-48c2-997e-3bfa8fda421e
# ╟─720ce158-f0b6-4165-a742-938c83146cff
# ╠═2b65eb50-0ff3-441f-9c0e-964bf10d29bc
# ╟─602a021c-a473-4267-aaf3-2ed77f05ca07
# ╟─2bae2d86-2a3f-446c-aa34-7582b383d2b2
# ╠═68564ae6-2540-4a6c-aed5-0794d413e6ef
# ╠═cc5e1e9a-9087-437a-b832-1918de6d4c48
# ╟─5d9cf8b5-82f4-496f-ad7d-44b83a6ef157
# ╟─5758e9c4-4c8c-4573-80a7-8270e8b428e4
# ╟─ce95e427-eeec-48c6-8e62-bf6f9f892b3e
# ╠═2969a12c-8fe1-4b76-bf0e-b7be6425eb21
# ╠═738519e3-c469-4def-ae9d-dbdd92563120
# ╟─05b29e3f-4493-4829-abba-5db47756b54a
# ╟─668197a9-91e4-461d-8552-a20a34b6eb3d
# ╟─363fbc94-afa6-4a85-9ac5-4d253b21df69
# ╟─fbb7a8c1-dc44-4d79-94fc-b7068a7881e1
# ╠═c3669042-bb56-423e-a077-b0cb82ce74a3
# ╟─fd2784db-87b3-4c2e-9d4d-7bbca8ec3aba
# ╟─0e431880-8449-44bc-8996-2c50439b0eac
# ╟─e93f8a5f-fd02-4894-aba5-7cabb9dc76b3
# ╟─0e7c0e3d-297d-4ccd-a71b-54a49fc40212
# ╟─418b6277-da40-4232-ab0b-a07c1dc0d5e9
# ╟─40b0773b-fc7d-4b28-9684-7e4e36ee81c9
# ╟─06b9d658-7d19-448d-8c02-6fabc5d4a551
# ╟─1ba53bec-658c-4e07-8728-f9799a5514e8
# ╟─5cc8a85a-2542-4b69-b79a-736b87a5a8c4
# ╟─80e4b5da-9d6e-46cd-9f84-59fa86c201b1
# ╟─29489270-20ee-4835-8451-db12fe46cf4c
# ╠═84f6d8f2-6ba5-43d6-9a06-f485975bf208
# ╠═be028bb1-28a4-45f4-8c52-c919636b2ea4
# ╟─b2efefeb-f4a4-40f6-850f-f73b30ce386c
# ╠═29ca031c-5520-4ef5-95c1-2b0c9fa12906
# ╟─a320c6e2-42f3-445f-9585-6e2ff5a43060
# ╟─5f3d1c80-d93f-4916-82a8-21205e4d0e26
# ╠═8e3edeee-c940-4975-99e8-bc27c3b18939
# ╟─5985ac18-636e-4606-9978-7e1e0ce1fd09
# ╟─392e0306-6e39-4d62-80aa-7c9f837cd0a0
# ╟─1f0d9a8f-94b9-4e87-a837-9c350b905c72
# ╟─aa404f69-e857-4eb6-87f1-298d62429891
# ╟─7ea26a1d-b6be-4d47-ad30-222ec6ea1a5a
# ╟─494373d7-3a8c-4717-bc9b-6e57267b3a58
# ╠═427fcde3-c719-4504-89c3-dfc0491677f9
# ╠═1b1e9945-ad12-4108-a866-f02192eb064f
# ╠═678b89ee-0302-4ab9-865e-564460f2d691
# ╟─737782c3-17a7-4684-b912-8ac392422941
# ╟─b07bcd90-d9c0-4a71-886c-756cb03b9bc1
# ╟─1c0ed9fb-c56a-4aab-a765-49c394123a42
# ╟─8dbf29db-cefd-4bd4-95e7-a302c7aa858a
# ╠═1075e6dc-be4b-4594-838e-60d44100c92d
# ╟─ce1f3eed-13ab-4fa7-aafc-a97954dd818b
# ╠═7e4fa23d-b05b-4f77-959e-29577e349333
# ╟─69e31afc-7893-4099-8d48-937d8ebffa86
# ╟─94caa09c-59b6-461f-b56f-178992d2bc83
# ╟─bb4de2cf-e4b6-4788-9f8b-ff5fd2ca2570
# ╟─501172ab-203d-4faa-a3b0-3e4fa0c79d10
# ╟─981608f2-57f6-44f1-95ed-82e8cca04718
# ╠═241e587f-b3dd-4bf8-83d0-1459c389fcc0
# ╟─168d4dbd-f0dd-433b-bb4a-e3bb918fb184
# ╠═2308aa4c-bb99-4546-a108-9fa88fca130b
# ╟─1adc874e-e024-464a-80d5-5ded04f62f24
# ╠═d359f3aa-b238-420f-99d2-52f85ce9ff82
# ╟─bca920fc-9535-4eb0-89c2-03a7334df6b6
# ╟─416a607b-26bc-4973-8c1a-489e855a06de
# ╟─ece11b32-cc61-447b-bbcf-018829360b73
# ╠═c29b63f3-0d32-46ad-99a4-3cae4a3f6181
# ╟─135c6b95-c440-45ed-bade-0327bf1e142a
# ╟─8e98a3b4-bc92-43ad-9da3-1323e06cfce6
# ╟─aa5caae7-638d-441f-a306-5442a5c8f75f
# ╟─faed4326-5ee6-41da-9ba4-297e965c242e
# ╟─17136b27-9463-4e5f-a943-d78297f28be7
# ╠═bee6469f-13a1-4bd8-8f14-f01e8405a949
# ╠═59554e03-ae31-4cc4-a6d1-c307f1f7bd9a
# ╠═7a6940ef-56b3-4cb4-bc6b-2c97625965cc
# ╠═665c55c3-d4dc-4d13-9517-d1106ea6210f
# ╟─e6930d53-6652-4fea-9a01-a4c87b8058dc
# ╠═0cce2e6d-4f19-4c50-a4b5-2835c3ed4401
# ╠═1c023156-0634-456d-a959-65880fd60c34
# ╟─caa2e633-e044-417c-944c-6a0458475e4f
# ╠═3e64e18a-6446-4fcc-a282-d3d6079e975a
# ╟─23301a45-fc89-416f-bcc7-4f3ba41bf04f
# ╠═d57adc7b-fcd5-468b-aa50-919d884a916a
# ╟─492c0922-e5f0-435b-9372-27e79570d679
# ╟─d2efd2a0-f494-4486-8ed3-ffd944b8473f
# ╟─4cc367d1-37a1-4712-a63e-8826b5646a1b
# ╠═4c3c2931-e4a8-43d1-b3fa-8bb3b82fb975
# ╠═2ad03b07-68ad-4da8-a5d3-7692502c0e00
# ╟─5f9f903b-a72c-4b60-9934-4bd2ced30a2c
# ╠═54c537af-06c3-4e12-87b8-33d3f1efa77b
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
