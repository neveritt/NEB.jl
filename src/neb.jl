abstract IdentificationState

immutable NEBstate{T} <: IdentificationState
  Θ::Vector{T}
  σ::Vector{T}
  λ::Vector{T}
  β::Vector{T}
  s::Matrix{T}
end

function neb{T}(data::IdDataObject{T}, n::Int, m::Int; outputidx::Int=1)
  y  = data.y[outputidx:outputidx,:]
  u  = data.y[setdiff(1:data.ny,outputidx),:]
  r  = data.u
  neb(y,u,r,n,m)
end

function neb{T}(y::AbstractMatrix{T}, u::AbstractMatrix{T}, r::AbstractMatrix{T},
    n::Int, m::Int)
  nᵤ = size(u,1)
  nᵣ = size(r,1)
  nₛ = nᵤ*nᵣ
  λᵥ, βᵥ, σᵥ, sᵥ, Θ, gₜ, Gₜ, R = _initial_NEB(y,u,r,n,m)
  z  = vcat(vec(u.'),vec(y.'))
  W  = vcat(R, Gₜ*R)

  # save state
  NEBtrace = [NEBstate(copy(Θ), copy(σᵥ), copy(λᵥ), copy(βᵥ), copy(sᵥ))]
  @inbounds for iter in 1:100
    _iter_NEB!(λᵥ, βᵥ, σᵥ, sᵥ, Θ, W, R, z, nᵤ)
    state = NEBstate(copy(Θ), copy(σᵥ), copy(λᵥ), copy(βᵥ), copy(sᵥ))
    push!(NEBtrace,state)
  end
  return NEBtrace, z
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
  Σₑ    = spdiagm(kron(σᵥ,ones(T,N)))
  invΣₑ = spdiagm(kron(1./σᵥ,ones(T,N)))
  Λ     = spdiagm(kron(λᵥ,ones(T,n)))
  # warning
  P, s  = _create_Ps(K*Λ, invΣₑ, W, z)
  S     = P+s*s'
  û     = R*s
  sᵥ[:] = s

  # θ optimimization
  b = _create_b(y, û, nᵤ, N)
#  A  = _create_A(R*S*R.', N, nᵤ)
  U,sv,V = svd(S)
  sidx = min(10,length(sv)) #length(sv) - length(filter(x-> x > 0.99*sum(sv), cumsum(sv))) + 1
  sv = sv[1:sidx]
  Uᵣ = R*U[:,1:sidx]
  Vᵣ = R*V[:,1:sidx]

#  df = TwiceDifferentiableFunction(x -> Q₀(x, A, b, N, Ts, m, nᵤ))
  df = TwiceDifferentiableFunction(x -> Qₙ(x, Uᵣ, sv, b, N, m, nᵤ))
  #options  = OptimizationOptions(autodiff = true, g_tol = 1e-14)
  opt = optimize(df, Θ, Newton(), Optim.Options(autodiff = true, g_tol = 1e-14))

  # update hyperparameters
  Θ[:]  = opt.minimizer
  gₜ,Gₜ = _create_G(Θ, nᵤ, m, N)
  W  = vcat(R, Gₜ*R)
  ŷ  = Gₜ*R*sᵥ[:]
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
    σᵥ[i] = (sumabs2(uₜ-ûₜ) + trace(Pₜ))/N
  end

  # update output noise parameters
  Pₜ = view(P̂, nᵤ*N+(1:N), nᵤ*N+(1:N))
  σᵥ[end] = (sumabs2(y-ŷ) + trace(Pₜ))/N

  return nothing
end

function _initial_NEB{T}(y::AbstractMatrix{T}, u::AbstractMatrix{T},
  r::AbstractMatrix{T}, n::Int, m::Int)
  size(y,1) == 1 || throw(ArgumentError("_initial_NEB: only one output allowed"))
  size(y,2) == size(u,2) == size(r,2) || throw(ArgumentError("_initial_NEB: Data length must be the same"))
  nᵤ,N = size(u)
  nᵣ   = size(r,1)
  nₛ   = nᵤ*nᵣ
  Ts   = 1.0

  λᵥ = zeros(T,nₛ)
  βᵥ = zeros(T,nₛ)
  σᵥ = zeros(T,nᵤ+1)
  sᵥ = zeros(T,n,nₛ)
  û  = zeros(T,nᵤ,N)
  λ₀, β₀, σ₀ = 100*one(T), 0.9*one(T), 100*one(T)
  for k = 1:nᵤ, j = 1:nᵣ
    i = nᵣ*(k-1) + j
    λᵥ[i], βᵥ[i], σᵥ[k], sᵥ[:,i] = basicEB(u[k,:], r[j,:], n, λ₀, β₀, σ₀)
    û[k,:] += filt(sᵥ[:,i],1,r[j,:])
  end

  # initialization for OE
  options = IdOptions(iterations = 20, autodiff=true, estimate_initial=false)
  model   = ARX(m,m,ones(Int,nᵤ),1,nᵤ)
  zdata   = iddata(y, û, 1.0)
  s       = arx(zdata,model,options)

  Θ = zeros(2m*nᵤ)
  for i in 1:nᵤ
    Θ[(i-1)*m+(1:m)]      = coeffs(s.B[i])[2:m+1]
    Θ[nᵤ*m+(i-1)*m+(1:m)] = coeffs(s.A[1])[2:m+1]
  end

  # OE model
  OEmodel = OE(m*ones(Int,1,nᵤ), m*ones(Int,1,nᵤ), ones(Int,1,nᵤ), 1, nᵤ)
  A,B,F,C,D,info = pem(zdata, OEmodel, Θ, options)

  σᵥ[end] = info.mse[1]

  Θ = zeros(T,2m*nᵤ)
  for i in 1:nᵤ
    Θ[(i-1)*2m+(1:2m)] = vcat(coeffs(B[i])[2:m+1], coeffs(F[i])[2:m+1])
  end
  R      = _create_R(r.', nᵤ, nᵣ, nₛ, n, N)
  gₜ, Gₜ = _create_G(Θ, nᵤ, m, N)

  z  = vcat(vec(u.'),vec(y.'))
  W  = vcat(R, Gₜ*R)

  # with initial Θ, update noise and hyper parameters
  K     = _create_K(βᵥ, n)
  invΣₑ = spdiagm(kron(1./σᵥ,ones(T,N)))
  Λ     = spdiagm(kron(λᵥ,ones(T,n)))
  # warning
  P, s  = _create_Ps(K*Λ, invΣₑ, W, z)
  S     = P+s*s'
  û     = R*s
  sᵥ[:] = s

  ŷ  = Gₜ*R*sᵥ[:]
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
    σᵥ[i] = (sumabs2(uₜ-ûₜ) + trace(Pₜ))/N
  end

  # update output noise parameters
  Pₜ = view(P̂, nᵤ*N+(1:N), nᵤ*N+(1:N))
  σᵥ[end] = (sumabs2(y.'-ŷ) + trace(Pₜ))/N

  return λᵥ, βᵥ, σᵥ, sᵥ, Θ, gₜ, Gₜ, R
end

function _create_G{T}(Θ::AbstractVector{T}, nᵤ::Int, m::Int, N::Int)
  Gₜ = zeros(T,N,N*nᵤ)
  gₜ = zeros(T,N*nᵤ)
  for i in 0:nᵤ-1
    Θᵢ              = Θ[i*2m+(1:2m)]
    gₜ[i*N+(1:N)]   = impulse(vcat(zeros(T,1),Θᵢ[1:m]), vcat(ones(T,1),Θᵢ[m+(1:m)]),Ts,N)
    Gₜ[:,i*N+(1:N)] = Toeplitz(gₜ[i*N+(1:N)],N)
  end
  return gₜ, Gₜ
end

function _create_g{T}(Θ::AbstractVector{T}, nᵤ::Int, m::Int, N::Int)
  gₜ = zeros(T,N*nᵤ)
  for i in 0:nᵤ-1
    Θᵢ              = Θ[i*2m+(1:2m)]
    gₜ[i*N+(1:N)]   = impulse(vcat(zeros(T,1),Θᵢ[1:m]), vcat(ones(T,1),Θᵢ[m+(1:m)]),Ts,N)
  end
  gₜ
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
    R[i*N+(1:N), (i*nᵣ+j)n+(1:n)] = Toeplitz(hcat(r[1:N,jᵣ]),hcat(r[1,jᵣ],zeros(1,n-1)))
  end
  return R
end

function _get_problem_dims{T}(R::AbstractMatrix{T}, Θ::AbstractVector{T},
  λᵥ::AbstractVector{T}, nᵤ::Int)
  nₛ::Int = length(λᵥ)
  n::Int  = round(size(R,2)/nₛ)
  m::Int  = round(length(Θ)/2nᵤ)
  nᵣ::Int = round(nₛ/nᵤ)
  N::Int  = round(size(R,1)/nᵤ)
  return nₛ, n, m, nᵣ, N
end

function _create_b{T}(y::AbstractVector{T}, û::AbstractVector{T}, nᵤ::Int, N::Int)
  bᵥ = zeros(T,N*nᵤ)
  for i = 1:nᵤ
    idx  = (i-1)*N + (1:N)
    bᵥ[idx] = Toeplitz(û[idx], N).'*y
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

function _create_A{T}(RSRᵀ::AbstractMatrix{T}, N::Int, nu::Int=1)
  D = _create_D(T,N)
  return kron(speye(T,nu), D).'*kron(RSRᵀ,speye(T,N))*kron(speye(T,nu), D)
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
  bb::AbstractVector{T}, N::Int, Ts::Float64, m::Int, nᵤ::Int)
  m::Int        = round(length(Θ)/2nᵤ)
  gₜ::Vector{T} = _create_g(Θ, nᵤ, m, N)

  return dot(gₜ,A*gₜ) - 2*dot(bb,gₜ)
end

function Qₙ{T}(Θ::AbstractVector{T}, U::AbstractMatrix{T},
  s::AbstractVector{T}, bb::AbstractVector{T}, N::Int, m::Int, nᵤ::Int)

  sumu = _quad_cost(Θ, U, s, m, nᵤ)
  gₜ   = _create_g(Θ, nᵤ, m, N)
  return sumu - 2*dot(bb,gₜ)
end

function _quad_cost{T}(Θ::AbstractVector{T}, U::AbstractMatrix{T}, s::AbstractVector{T}, m::Int, nᵤ::Int)
  b = Vector{Vector{T}}(nᵤ)
  a = Vector{Vector{T}}(nᵤ)
  for i in 0:nᵤ-1
    Θᵢ = Θ[i*2m+(1:2m)]
    b[i+1]  = vcat(zeros(T,1), Θᵢ[1:m])
    a[i+1]  = vcat(ones(T,1), Θᵢ[m+(1:m)])
  end
  sum = zero(T)
  for i in 1:length(s)
    vj = zeros(T,N)
    for j in 1:nᵤ
      idxj = (j-1)*N+(1:N)
      vj += filt(b[j], a[j], U[idxj,i])
    end
    sum += s[i]*sumabs2(vj)
  end
  return sum
end
