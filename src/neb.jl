
function NEB{T}(y::AbstractMatrix{T}, u::AbstractMatrix{T},
  r::AbstractMatrix{T}, n::Int, m::Int)
  nᵤ = size(u,2)
  λᵥ, βᵥ, σᵥ, sᵥ, Θ, gₜ, Gₜ, R = _initial_NEB(y,u,r,n,m)
  z  = vcat(u[:],y[:])
  W  = vcat(R, Gₜ*R)
  @inbounds for iter in 1:50
    println(iter)
    _iter_NEB!(λᵥ, βᵥ, σᵥ, sᵥ, Θ, W, R, z, nᵤ)
  end
  return λᵥ, βᵥ, σᵥ, sᵥ, Θ, z
end


function _iter_NEB!{T}(λᵥ::AbstractVector{T}, βᵥ::AbstractVector{T},
  σᵥ::AbstractVector{T}, sᵥ::AbstractMatrix{T}, Θ::AbstractVector{T},
  W::AbstractMatrix{T}, R::AbstractMatrix{T}, z::AbstractVector{T}, nᵤ::Int)
#  @assert eltype(R) == T throw(ArgumentError())
  Ts = 1.0

  nₛ, n, m, nᵣ, N::Int = _get_problem_dims(R, Θ, λᵥ, nᵤ)
  y  = view(z, N*nᵤ+1:N*(nᵤ+1))

  # problematic types
  local Σₑ::SparseMatrixCSC{T,Int}
  local invΣₑ::SparseMatrixCSC{T,Int}
  local Λ::SparseMatrixCSC{T,Int}

  K     = _create_K(βᵥ, n)
  Σₑ    = spdiagm(kron(σᵥ.^2,ones(T,N)))
  invΣₑ = spdiagm(kron(1./σᵥ.^2,ones(T,N)))
  Λ     = spdiagm(kron(λᵥ,ones(T,n)))
  # warning
  P, s  = _create_Ps(K*Λ, invΣₑ, W, z)
  S  = P+s*s'
  û  = R*s

  # θ optimimization
  b = _create_b(y, û, nᵤ, N)
  A  = _create_A(R*S*R.', N)

  df = TwiceDifferentiableFunction(x -> Q₀(x, A, b, N, Ts))
  #options  = OptimizationOptions(autodiff = true, g_tol = 1e-14)
  opt = optimize(df, Θ, Newton(), OptimizationOptions(autodiff = true, g_tol = 1e-14))

  # update hyperparameters
  Θ  = opt.minimum
  gₜ::Vector{T} = impulse(tf(vcat(zeros(T,1), Θ[1:m]), vcat(ones(T,1), Θ[m+1:2m]), Ts), N)
  Gₜ = Toeplitz(gₜ, N)
  W  = vcat(R, Gₜ*R)
  ŷ  = Gₜ*R*s
  P̂  = W*P*W'

  # update λ and β
  for i in 1:nₛ
    idx = (i-1)*n + (1:n)
    λᵥ[i], βᵥ[i] = basicQmin(view(S,idx,idx))
  end

  # update input noise parameters
  for i in 1:nᵤ
    idx = (i-1)*N + (1:N)
    Pₜ = view(P̂, idx, idx)
    uₜ = view(z,idx)
    ûₜ = view(û,idx)
    σᵥ[i] = sqrt((sumabs2(uₜ-ûₜ) + trace(Pₜ))/N)
  end

  # update output noise parameters
  Pₜ = view(P̂, nᵤ*N+(1:N), nᵤ*N+(1:N))
  σᵥ[end] = sqrt((sumabs2(y-ŷ) + trace(Pₜ))/N)

  return nothing
end

function _initial_NEB{T}(y::AbstractMatrix{T}, u::AbstractMatrix{T},
  r::AbstractMatrix{T}, n::Int, m::Int)
  size(y,1) == size(u,1) == size(r,1) || throw(ArgumentError("Data length must be the same"))
  N,nᵤ = size(u)
  nᵣ   = size(r,2)
  nₛ   = nᵤ*nᵣ
  Ts   = 1.0

  λᵥ = zeros(T,nₛ)
  βᵥ = zeros(T,nₛ)
  σᵥ = zeros(T,nᵤ+1)
  sᵥ = zeros(T,nₛ,n)
  û = zeros(N,nᵤ)
  for k = 1:nᵤ, j = 1:nᵣ
    i = nᵣ*(k-1) + j
    λᵥ[i], βᵥ[i], σᵥ[k], sᵥ[i,:] = basicEB(u[:,k], r[:,j], n, λ₀, β₀, σ₀)
    û[:,nᵤ] += filt(sᵥ[i,:],1,r[:,j])
  end

  z = iddata(y[:],û[:], 1.0)
  s = arx(z,[m,m,1], ARX())
  Θ₀ = vcat(numvec(s.G),denvec(s.G)[2:end])

  s₁ = pem(z, [m,m,1], Θ₀, OE())
  σᵥ[end] = sqrt(s₁.info.mse)

  Θ = vcat(numvec(s₁.G),denvec(s₁.G)[2:end])

  R = _create_R(r, nᵤ, nᵣ, nₛ, n, N)
  gₜ = impulse(tf(vcat(zeros(T,1), Θ[1:m]),vcat(ones(T,1), Θ[m+1:2m]),Ts),N)
  Gₜ = Toeplitz(gₜ,N)

  return λᵥ, βᵥ, σᵥ, sᵥ, Θ, gₜ, Gₜ, R
end

function _create_R{T}(r::AbstractMatrix{T}, nᵤ::Int, nᵣ::Int, nₛ::Int,
  n::Int, N::Int)
  # create R = [R_1...R_nr; 0 … 0;
  #             0     ⋱    0 … 0
  #             0  …  0   R_1 …  R_nr]
  idxr = 1:nᵣ
  idxu = 1:nᵤ
  R = zeros(T, nᵤ*N, n*nₛ)
  for  i in 0:nᵤ-1, j in 0:nᵣ-1
    jᵣ = idxr[j+1]
    R[i*N+(1:N), (i*nᵣ+j)n+(1:n)] = Toeplitz(hcat(r[:,jᵣ]),hcat(r[1,jᵣ],zeros(1,n-1)))
  end
  return R
end

function _get_problem_dims{T}(R::AbstractMatrix{T}, Θ::AbstractVector{T},
  λᵥ::AbstractVector{T}, nᵤ::Int)
  nₛ::Int = length(λᵥ)
  n::Int  = round(size(R,2)/nₛ)
  m::Int  = round(length(Θ)/2)
  nᵣ::Int = round(nₛ/nᵤ)
  N::Int  = round(size(R,1)/nᵤ)
  return nₛ, n, m, nᵣ, N
end

function _create_b{T}(y::AbstractVector{T}, û::AbstractVector{T}, nᵤ::Int, N::Int)
  bᵥ = zeros(T,N*nᵤ)
  for i = 1:nᵤ
    idx  = (i-1)*N + (1:N)
    bᵥ[idx] = Toeplitz(û[idx], N)*y
  end
  return bᵥ
end

function _create_D{T}(::Type{T}, N::Int)
  D  = spzeros(T,N*N,N)
  for i in 1:N, j in i:N
    D[(i-1)*N+j,j-i+1] = one(T)
  end
  return D
end

function _create_A{T}(RSRᵀ::AbstractMatrix{T}, N::Int)
  D = _create_D(T,N)
  return D.'*kron(RSRᵀ,speye(T,N))*D
end

function _create_Ps{T}(KΛ::AbstractMatrix{T}, invΣₑ::AbstractMatrix{T},
  W::AbstractMatrix{T}, z::AbstractVector{T})
  P  = inv(W.'*invΣₑ*W + inv(KΛ))
  s  = P*W'*invΣₑ*z
  return P, s
end

function _create_K{T}(βᵥ::AbstractVector{T}, n::Int)
  # create K = [TC_1; 0 … 0;
  #             0     ⋱    0 … 0
  #             0  …  0   TC_nₛ]
  TC  = T[max(i,j) for i in 1:n, j in 1:n]
  nₛ = length(βᵥ)
  K  = zeros(Float64,n*nₛ,n*nₛ)
  for i in 0:nₛ-1
    K[i*n+(1:n), i*n+(1:n)] = βᵥ[i+1].^TC
  end
  return K
end

function Q₀{T}(Θ::AbstractVector{T}, A::AbstractMatrix{T},
  bb::AbstractVector{T}, N::Int, Ts::Float64)
  m::Int = round(length(Θ)/2)
  b::Vector{T} = Θ[1:m]
  a::Vector{T} = Θ[m+1:end]
  gₜ::Vector{T} = impulse(tf(vcat(zeros(1), b),vcat(ones(1), a),Ts),N)

  return dot(gₜ,A*gₜ) - 2*dot(bb,gₜ)
end
