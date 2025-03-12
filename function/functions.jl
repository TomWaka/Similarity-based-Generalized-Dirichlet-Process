module functions

using Distributions, LinearAlgebra, Random, SpecialFunctions, StatsPlots, Distances

export gaussian_process_cov, sgdp_clustering, gaussian_process_cov_exp

function gaussian_process_cov(X, η, ϕ)
    pdist(x) = pairwise(Euclidean(), x)
    K = η^2 * exp.(-pdist(X).^2 ./ (2 * ϕ^2))
    return K + 1e-6 * Matrix{eltype(K)}(I, size(K)...)
end

function gaussian_process_cov_exp(X, η, ϕ)
    pdist(x) = pairwise(Euclidean(), x)
    K = η^2 * exp.(-pdist(X) ./ (2*ϕ))
    return K + 1e-6 * Matrix{eltype(K)}(I, size(K)...)
end

function similarity_measure(i, j, adj, τ)
    return adj[i, j] == 1 ? 1.0 : τ
end

function sgdp_similarity_weight(i, j, z, τ, adj)
    n = length(z) 
    idx = findall(x -> x < n, 1:length(z))[z[1:n-1] .== j]
    if isempty(idx)
        return τ
    else
        return sum(adj[i, idx] + τ .* (1 .-adj[i, idx])) / sum(adj[i, 1:n-1] .+ τ .* (1 .- adj[i, 1:n-1]))
    end
end

function sgdp_similarity_weight(i, j, z, τ, adj)
    n = length(z) 
    indices = findall(x -> x == j, z)
    if isempty(indices)
        return τ
    else
        # 全体の類似度行列 adj はフルサイズ（例えば n_full x n_full）のものを渡すので、
        # 行 i の、前に割り当てられたインデックス（1:n）に対応する部分を抽出する
        # ここでは、z に対応する部分は、adj[i, 1:n] とする
        similarities_prev = [adj[i, k] for k in 1:n]
        # indices で指定されたクラスタ j に属する部分の類似度
        similarities_j = [adj[i, k] for k in indices]
        denom = sum([sim + τ * (1 - sim) for sim in similarities_prev])
        numer = sum([sim + τ * (1 - sim) for sim in similarities_j])
        return numer / denom
    end
end

function sgdp_conditional_prob(i, j, z, α, β, τ, adj)
    n = length(z)
    k = maximum(z)
    if j <= k
        N_j = count(z .== j)
        ω_j = sgdp_similarity_weight(i, j, z, τ, adj)
        p = (α * β + N_j - 1) / (α + n - 1) * ω_j
        for ℓ in 1:(j-1)
            p *= (α * (1 - β) + sum(z .> ℓ)) / (α + sum(z .> ℓ) - 1)
        end
        return p < 1e-100  ? 1e-100 - p : p
    else
        p = (α * (1 - β)) / (α + n - 2)
        for ℓ in 1:k-1
            p *= (α * (1 - β) + sum(z .> ℓ)) / (α + sum(z .> ℓ) - 1)
        end
        return p
    end
end

function sgdp_joint_log_prob1(z, α, α1, β, τ, adj)
    n = length(z)
    K = maximum(z)

    log_prob = 0.0
    for i in 2:n
        if z[i] <= K
            log_prob += log(sgdp_conditional_prob(i, z[i], z[1:i-1], α, β, τ, adj) / sgdp_conditional_prob(i, z[i], z[1:i-1], α1, β, τ, adj))
        else
            log_prob += log((α * (1 - β)) / (α + i - 2) / (α1 * (1 - β)) * (α1 + i - 1) / (α + i - 1))
        end
    end
    return log_prob
end

function sgdp_joint_log_prob2(z, α, β, β1, τ, adj)
    n = length(z)
    K = maximum(z)

    log_prob = 0.0
    for i in 2:n
        if z[i] <= K
            log_prob += log(sgdp_conditional_prob(i, z[i], z[1:i-1], α, β, τ, adj) / sgdp_conditional_prob(i, z[i], z[1:i-1], α, β1, τ, adj))
        else
            log_prob += log((α * (1 - β)) / (α + i - 2) / (α * (1 - β1)) * (α + i - 1) / (α + i - 1))
        end
    end
    return log_prob
end

function sgdp_joint_log_prob3(z, α, β, τ, τ1, adj)
    n = length(z)
    K = maximum(z)
    log_prob = 0.0
    for i in 2:n
        if z[i] <= K
            log_prob += log(sgdp_conditional_prob(i, z[i], z[1:i-1], α, β, τ, adj) / sgdp_conditional_prob(i, z[i], z[1:i-1], α, β, τ1, adj))
        else
            log_prob += 0 #log((α * (1 - β)) / (α + i - 2) / (α * (1 - β)) * (α + i - 1) / (α + i - 1))
        end
    end
    return log_prob
end

function sample_ϕ1(ϕ, ϕ_new, a_ϕ, b_ϕ, y, mu, η, X)
    N, T, _ = size(y)
    log_ratio = 0.0
    for i in 1:N
        for t in 1:T
            log_ratio += logpdf(MvNormal(mu[i, t, :], gaussian_process_cov(X, η, ϕ_new)), y[i, t, :]) -
                         logpdf(MvNormal(mu[i, t, :], gaussian_process_cov(X, η, ϕ)), y[i, t, :])
        end
    end
    log_ratio += logpdf(InverseGamma(a_ϕ, b_ϕ), ϕ_new) - logpdf(InverseGamma(a_ϕ, b_ϕ), ϕ)
    if log(rand()) < log_ratio
        return ϕ_new
    else
        return ϕ
    end
end

function sample_ϕ2(ϕ, ϕ_new, a_ϕ, b_ϕ, θ, m, η, X)
    K, _ = size(θ)
    log_ratio = 0.0
    for k in 1:K
        log_ratio += logpdf(MvNormal(m, gaussian_process_cov(X, η, ϕ_new)), θ[k, :]) -
                     logpdf(MvNormal(m, gaussian_process_cov(X, η, ϕ)), θ[k, :])
    end
    log_ratio += logpdf(InverseGamma(a_ϕ, b_ϕ), ϕ_new) - logpdf(InverseGamma(a_ϕ, b_ϕ), ϕ)
    if log(rand()) < log_ratio
        return ϕ_new
    else
        return ϕ
    end
end

function sgdp_joint_log_prob_ordered(z_order::Vector{Int}, α, β, τ, adj_order)
    n = length(z_order)
    log_prob = 0.0
    # 1 番目の割当は初期値として確率 1 とする
    for t in 2:n
        # 条件付き確率を計算するため、t-1 個の既割当を用いる
        # sgdp_conditional_prob を使うために、一時的な vector を作成
        z_prev = z_order[1:t-1]
        # 現在の割当
        j_current = z_order[t]
        # 条件付き確率の計算（sgdp_conditional_prob は t を n とみなす）
        p_val = sgdp_conditional_prob(t, j_current, z_prev, α, β, τ, adj_order[1:t-1, 1:t-1])
        log_prob += log(p_val)
    end
    return log_prob
end

function sgdp_clustering(y, adj, X, M, w, m_m, C_m, a_η=1, b_η=1, a_α=5.0, b_α=1.0, a_β=10.0, b_β=1.0, a_τ=0.5, b_τ=0.5, a_ϕ=0.5, b_ϕ=0.5, n_iter=20000)

    N, T, p = size(y)

    # 初期化
    α = [3.0 for _ in 1:M]
    β = [2/3 for _ in 1:M]
    τ = [1e-3 for _ in 1:M]
    z = [1 for _ in 1:N, _ in 1:M]
    K = [maximum(z[:, ℓ]) for ℓ in 1:M]
    θ = [zeros(K[ℓ], p) for ℓ in 1:M]
    m_θ = [zeros(p) for ℓ in 1:M]
    η_y = 2.0
    ϕ_y = 2.0
    η_θ = ones(M) * 1.0
    ϕ_θ = ones(M) * 1.0
    sigma = collect(1:N)          

    # パラメータのトレースを保存する配列
    α_trace = zeros(n_iter÷5, M)
    β_trace = zeros(n_iter÷5, M)
    τ_trace = zeros(n_iter÷5, M)
    η_y_trace = zeros(n_iter÷5)
    η_θ_trace = zeros(n_iter÷5, M)
    ϕ_y_trace = zeros(n_iter÷5)
    ϕ_θ_trace = zeros(n_iter÷5, M)
    z_trace = zeros(Int, n_iter÷5, N, M)
    sigma_trace = zeros(Int, n_iter ÷ 5, N)  
    K_trace = zeros(Int, n_iter÷5, M)
    θ_trace = []
    

    # MCMCサンプリング
    for iter in 1:n_iter
        println(iter)

        # θのサンプリング
        for ℓ in 1:M
            for j in 1:K[ℓ]
                idx = findall(z[:, ℓ] .== j)
                if !isempty(idx)
                    y_j = y[idx, ℓ, :]
                    C_θ = gaussian_process_cov(X, η_θ[ℓ], ϕ_θ[ℓ])
                    C_y = gaussian_process_cov(X, η_y, ϕ_y)
                    C_pos = (C_θ - C_θ * ((C_θ + 1 / length(idx) .* C_y) \ C_θ))
                    C_pos = (C_pos + C_pos') / 2
                    m_pos = C_pos * (C_y \ sum(y_j, dims=1)' + C_θ \ m_θ[ℓ])
                    θ[ℓ][j, :] = rand(MvNormal(vec(m_pos), C_pos))
                end
            end
        end

        # mu 
        mu = zeros(N, T, p)
        for n in 1:N
            for t in 1:T
                for ℓ in 1:M
                    if w[t, ℓ] == 1
                        mu[n, t, :] .+= θ[ℓ][z[n, ℓ], :]
                    end
                end
            end
        end

        # m_θのサンプリング
        for ℓ in 1:M
            C_θ_sum = K[ℓ] .* inv(gaussian_process_cov(X, η_θ[ℓ], ϕ_θ[ℓ]))
            C_pos = inv(C_θ_sum + inv(C_m))
            C_pos = (C_pos + C_pos') / 2
            m_pos = C_pos * ((C_θ_sum) * mean(θ[ℓ], dims=1)' + inv(C_m) * m_m[ℓ])
            m_θ[ℓ] .= rand(MvNormal(vec(m_pos), C_pos))
        end

        # η_yのサンプリング
        a_y = (a_η + N * T * p) / 2
        b_y = b_η / 2
        Ry = gaussian_process_cov(X, 1.0, ϕ_y)^(-1)
        Ry = (Ry + Ry') / 2
        for i in 1:N
            for t in 1:T
                b_y += 0.5 * (y[i, t, :] - mu[i, t, :])' * Ry * (y[i, t, :] - mu[i, t, :])
            end
        end
        η_y = sqrt(rand(InverseGamma(a_y, b_y)))

        # η_θのサンプリング
        for ℓ in 1:M
            a_θ = (a_η + K[ℓ] * p) / 2
            b_θ = b_η / 2
            Rℓ = gaussian_process_cov(X, 1.0, ϕ_θ[ℓ])^(-1)
            Rℓ = (Rℓ + Rℓ') / 2
            for k in 1:K[ℓ]
                b_θ += 0.5 * (θ[ℓ][k, :] - m_θ[ℓ])' * Rℓ * (θ[ℓ][k, :] - m_θ[ℓ])
            end
            η_θ[ℓ] = sqrt(rand(InverseGamma(a_θ, b_θ)))
        end

        # αのサンプリング
        for ℓ in 1:M
            α_prop = α[ℓ] + rand(Normal(0, 1)) / 10
            α_prop = α_prop < 1e-3 ? 2e-3 - α_prop : α_prop
            α_prop = α_prop < 1 ? 2 - α_prop : α_prop
            log_ratio = sgdp_joint_log_prob1(z[:, ℓ], α_prop, α[ℓ], β[ℓ], τ[ℓ], adj) +
                        logpdf(Gamma(a_α, 1 / b_α), α_prop) - logpdf(Gamma(a_α, 1 / b_α), α[ℓ])
            if log(rand()) < log_ratio
                α[ℓ] = α_prop
            end
        end

        # βのサンプリング
        for ℓ in 1:M
            β_prop = β[ℓ] + rand(Normal(0, 1)) / 10
            β_prop = β_prop < 1e-3 ? 2e-3 - β_prop : β_prop
            β_prop = β_prop > 0.999 ? 0.999*2 - β_prop : β_prop
            log_ratio = sgdp_joint_log_prob2(z[:, ℓ], α[ℓ], β_prop, β[ℓ], τ[ℓ], adj) +
                        logpdf(Beta(a_β, b_β), β_prop) - logpdf(Beta(a_β, b_β), β[ℓ])
            if log(rand()) < log_ratio
                β[ℓ] = β_prop
            end
        end

        # τのサンプリング
        for ℓ in 1:M
            τ_prop = τ[ℓ] + rand(Normal(0, 1)) / 400
            τ_prop = τ_prop < 1e-10 ? 2e-10 - τ_prop : τ_prop
            τ_prop = τ_prop > 0.999 ? 2*0.999 - τ_prop : τ_prop
            log_ratio = sgdp_joint_log_prob3(z[:, ℓ], α[ℓ], β[ℓ], τ_prop, τ[ℓ], adj) -
                        logpdf(Beta(a_τ, b_τ), τ_prop) - logpdf(Beta(a_τ, b_τ), τ[ℓ])
            if log(rand()) < log_ratio
                τ[ℓ] = τ_prop
            end
        end

        # ϕ_yのサンプリング
        ϕ_y_new = ϕ_y + rand(Normal(0, 1)) / 5
        ϕ_y_new = ϕ_y_new < 1e-10 ? 2e-10 - ϕ_y_new : ϕ_y_new
        ϕ_y = sample_ϕ1(ϕ_y, ϕ_y_new, a_ϕ, b_ϕ, y, mu, η_y, X)

        # ϕ_θのサンプリング
        for ℓ in 1:M
            ϕ_θ_new = ϕ_θ[ℓ] + rand(Normal(0, 1)) / 5
            ϕ_θ_new = ϕ_θ_new < 1e-10 ? 2e-10 - ϕ_θ_new : ϕ_θ_new
            ϕ_θ[ℓ] = sample_ϕ2(ϕ_θ[ℓ], ϕ_θ_new, a_ϕ, b_ϕ, θ[ℓ], m_θ[ℓ], η_θ[ℓ], X)
        end

        # zのサンプリング
        for i in sigma
            for ℓ in 1:M
                prob = zeros(K[ℓ] + 1)
                yy = y[i, w[:, ℓ].==1, :]
                if M > 1
                    yy  -= sum(cat([(w[:, setdiff(1:M, ℓ)[k]] .* θ[setdiff(1:M, ℓ)[k]][z[i, setdiff(1:M, ℓ)[k]], :]') for k in eachindex(setdiff(1:M, ℓ))]..., dims=1), dims=1)
                end

                for j in 1:K[ℓ]
                    prob[j] =  sgdp_conditional_prob(i, j, z[:, ℓ], α[ℓ], β[ℓ], τ[ℓ], adj) 
                end
                prob[K[ℓ]+1] = sgdp_conditional_prob(i, K[ℓ]+1, z[:, ℓ], α[ℓ], β[ℓ], τ[ℓ], adj) 
                prob .+= 1e-30
                prob[1:K[ℓ]] = prob[1:K[ℓ]] ./ sum(prob[1:K[ℓ]]) .* (1-prob[K[ℓ]+1])

                Σ_y = gaussian_process_cov(X, η_y, ϕ_y)
                for t in 1:T
                    yy_t = view(yy, t, :)
                    for j in 1:K[ℓ]
                        prob[j] *= pdf(MvNormal(θ[ℓ][j, :], Σ_y), yy_t)
                    end
                    prob[K[ℓ]+1] *= pdf(MvNormal(m_θ[ℓ], gaussian_process_cov(X, η_y, ϕ_y) + gaussian_process_cov(X, η_θ[ℓ], ϕ_θ[ℓ])), yy_t)
                    prob ./= sum(prob)
                end
                if any(prob .< 0)
                    prob[prob .< 0] .= 0
                    prob ./= sum(prob)
                end
                if any(isnan, prob)
                    prob .= 1
                    prob ./= sum(prob)
                end 

                z[i, ℓ] = rand(Categorical(prob))
                if z[i, ℓ] == K[ℓ] + 1
                    K[ℓ] += 1
                    θ[ℓ] = vcat(θ[ℓ], vec(mean(yy, dims=1))')
                end
            end
        end

        # number of cluster
        nth = []
        for i in 1:M
            function shift_numbers(numbers)
                min_num = 1
                max_num = maximum(numbers)
                missing_numbers = setdiff(min_num:max_num, numbers)
                while length(missing_numbers) > 0
                    smallest_missing = minimum(missing_numbers)
                    numbers = ifelse.(numbers .> smallest_missing, numbers .- 1, numbers)
                    missing_numbers = setdiff(min_num:maximum(numbers), numbers)
                end
                return numbers
            end
            K[i] = length(unique(z[:, i]))
            uz = unique(z[:, i])
            push!(nth, θ[i][uz, :])
            z[:, i] .= shift_numbers(z[:, i])
        end
        θ = nth

        # --- 順列 σ の更新（Metropolis–Hastings ステップ） ---
        # σ の提案：ランダムに 2 つの位置をスワップ
        sigma_new = copy(sigma)
        i1, i2 = rand(1:N, 2)
        sigma_new[i1], sigma_new[i2] = sigma_new[i2], sigma_new[i1]

        # 順序付きに再配置したクラスタ割当を得る（ここでは M=1 の場合の例）
        z_order_current = z[sigma, 1]
        z_order_new = z[sigma_new, 1]
        # 同様に、隣接行列も σ に沿って並べ替える
        adj_order_current = adj[sigma, sigma]
        adj_order_new = adj[sigma_new, sigma_new]

        # 順序付きの joint log probability を計算（以下の関数を定義しておく）
        log_prob_current = sgdp_joint_log_prob_ordered(z_order_current, α[1], β[1], τ[1], adj_order_current)
        log_prob_new = sgdp_joint_log_prob_ordered(z_order_new, α[1], β[1], τ[1], adj_order_new)

        if log(rand()) < (log_prob_new - log_prob_current)
            sigma = sigma_new   # 提案を受容
        end

        # パラメータの値を保存
        if iter > n_iter/5*4
            i = div(iter - div(n_iter, 5) * 4, 1)
            α_trace[i, :] = α
            β_trace[i, :] = β
            τ_trace[i, :] = τ
            η_y_trace[i] = η_y
            η_θ_trace[i, :] = η_θ
            ϕ_y_trace[i] = ϕ_y
            ϕ_θ_trace[i, :] = ϕ_θ
            z_trace[i, :, :] = z
            K_trace[i, :] = K
            sigma_trace[i, :] = sigma
            push!(θ_trace, deepcopy(θ))
        end
    end

    return α_trace, β_trace, K_trace, z_trace, θ_trace, sigma_trace
end

end
