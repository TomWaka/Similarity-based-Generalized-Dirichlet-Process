using Distributions, LinearAlgebra, Random, SpecialFunctions, StatsPlots, Distances

include("functions.jl")
using .functions

# Data settings
const N_REGIONS = 40  # Number of regions
const N_DAYS = 15  # Number of days
const P_POINTS = 24  # Number of data points per day
const N_PERIODS = 1  # Number of periods (weekdays, holidays, before holidays)
const TRUE_CLUSTERS = 8  # True number of clusters

# True cluster assignments
true_z = repeat(1:TRUE_CLUSTERS, inner=5)
adj = [i == j for i in true_z, j in true_z] - Matrix{Float64}(I, N_REGIONS, N_REGIONS)

w = ones(N_DAYS, N_PERIODS)

X = reshape(1:P_POINTS, 1, P_POINTS)
true_θ = [rand(MvNormal(zeros(P_POINTS), gaussian_process_cov(X, 2, 5))) for _ in 1:TRUE_CLUSTERS]

y = zeros(N_REGIONS, N_DAYS, P_POINTS)
for i in 1:N_REGIONS
    for t in 1:N_DAYS
        noise = rand(MvNormal(zeros(P_POINTS), gaussian_process_cov(X, 1, 1)))
        y[i, t, :] = true_θ[true_z[i]] .+ noise
    end
end

# using Plots
# plot()
# for i in 1:TRUE_CLUSTERS
#     plot!(true_θ[i], label="$i")
# end

# Hyperparameter settings
hyperparams = (
    a_η = 1,
    b_η = 1,
    a_α = 2,
    b_α = 1,
    a_β = 5,
    b_β = 1,
    a_ϕ = 0.5,
    b_ϕ = 0.5,
    a_τ = 0.5,
    b_τ = 0.5,
    m_m = [0.5 * ones(P_POINTS) for _ in 1:N_PERIODS],
    C_m = 10.0 * I(P_POINTS)
)

# MCMC settings
const N_ITER = 20000

# Running SGDP clustering
α_trace, β_trace, K_trace, z_trace, θ_trace = sgdp_clustering(
    y, adj, X, N_PERIODS, w, hyperparams.m_m, hyperparams.C_m,
    hyperparams.a_η, hyperparams.b_η, hyperparams.a_α, hyperparams.b_α,
    hyperparams.a_β, hyperparams.b_β, hyperparams.a_τ, hyperparams.b_τ,
    hyperparams.a_ϕ, hyperparams.b_ϕ, N_ITER
)
