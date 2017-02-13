abstract IdentificationState

immutable NEBstate{T} <: IdentificationState
  Θ::Vector{T}
  σ::Vector{T}
  λ::Vector{T}
  β::Vector{T}
  s::Matrix{T}
end

function neb{T,O}(data::IdDataObject{T}, n::Int, m::Int; outputidx::Int=1,
    options::IdOptions{O}=IdOptions(iterations=100))
  y  = data.y[outputidx:outputidx,:]
  u  = data.y[setdiff(1:data.ny,outputidx),:]
  r  = data.u
  neb(y,u,r,n,m,data.Ts,options=options)
end

function neb{T,O}(y::AbstractMatrix{T}, u::AbstractMatrix{T}, r::AbstractMatrix{T},
    n::Int, m::Int, Ts::Float64; options::IdOptions{O}=IdOptions(iterations=100))
  iterations = options.OptimizationOptions.iterations
  nᵤ = size(u,1)
  nᵣ = size(r,1)
  nₛ = nᵤ*nᵣ
  λᵥ, βᵥ, σᵥ, sᵥ, Θ, gₜ, Gₜ, R = _initial_NEB(y,u,r,n,m,Ts)
  z  = vcat(vec(u.'),vec(y.'))
  W  = vcat(R, Gₜ*R)

  # save state
  NEBtrace = [NEBstate(copy(Θ), copy(σᵥ), copy(λᵥ), copy(βᵥ), copy(sᵥ))]
  η = vcat(Θ, σᵥ, λᵥ, βᵥ)
  @inbounds for iter in 1:iterations
    ηold = η
    _iter_NEB!(λᵥ, βᵥ, σᵥ, sᵥ, Θ, W, R, z, nᵤ, Ts)
    state = NEBstate(copy(Θ), copy(σᵥ), copy(λᵥ), copy(βᵥ), copy(sᵥ))
    push!(NEBtrace,state)
    η = vcat(Θ, σᵥ, λᵥ, βᵥ)
    if norm(η-ηold)/norm(η) < options.OptimizationOptions.x_tol
      return NEBtrace, z
    end
  end
  return NEBtrace, z
end

function _iter_NEB!{T}(λᵥ::AbstractVector{T}, βᵥ::AbstractVector{T},
  σᵥ::AbstractVector{T}, sᵥ::AbstractMatrix{T}, Θ::AbstractVector{T},
  W::AbstractMatrix{T}, R::AbstractMatrix{T}, z::AbstractVector{T}, nᵤ::Int,
  Ts::Float64)
#  @assert eltype(R) == T throw(ArgumentError())

  nₛ, n, m, nᵣ, N::Int = _get_problem_dims(R, Θ, λᵥ, nᵤ)
  y  = view(z, N*nᵤ+1:N*(nᵤ+1))

  # problematic types
  local Σₑ::SparseMatrixCSC{T,Int}
  local invΣₑ::SparseMatrixCSC{T,Int}
  local Λ::SparseMatrixCSC{T,Int}

  K     = _create_K(βᵥ, n)
  Σₑ    = spdiagm(kron(σᵥ,ones(T,N)))
  invΣₑ = spdiagm(kron(1./σᵥ,ones(T,N)))
  iΛ    = spdiagm(kron(1./λᵥ,ones(T,n)))
  P, s  = _create_Ps(iΛ*inv(K), invΣₑ, W, z)
  S     = P+s*s'
  û     = R*s
  sᵥ[:] = s

  # θ optimimization
  b = _create_b(y, û, nᵤ, N)
  U,sv,V = svd(S)
  sidx = length(sv) - length(filter(x-> x > 0.999*sum(sv), cumsum(sv))) + 1
  sv = sv[1:sidx]
  Uᵣ = R*U[:,1:sidx]
  Vᵣ = R*V[:,1:sidx]

  df = TwiceDifferentiableFunction(x -> Qₙ(x, Uᵣ, sv, b, N, Ts, m, nᵤ))
  opt = optimize(df, Θ, Newton(), Optim.Options(autodiff = true, g_tol = 1e-14))

  # update hyperparameters
  Θ[:]  = opt.minimizer
  gₜ,Gₜ = _create_G(Θ, nᵤ, m, Ts, N)

  W[end-N+(1:N),:] = Gₜ*R

  ŷ  = Gₜ*R*sᵥ[:]
  P̂  = W*P*W'

  # update λ and β
  for i in 1:nₛ
    idx = (i-1)*n + (1:n)
    λᵥ[i], βᵥ[i] = basicQmin(view(S,idx,idx), 100)
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
  r::AbstractMatrix{T}, n::Int, m::Int, Ts::Float64)
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

  # fir
  fir_m    = n
  nk       = 0*ones(Int,1,nᵣ)
  firmodel = FIR(fir_m*ones(Int,1,nᵣ), nk, 1, nᵣ)
  options  = IdOptions(iterations = 100, autodiff=true, estimate_initial=false)
  û  = zeros(T,nᵤ,N)
  for k = 0:nᵤ-1
    zdata     = IdentificationToolbox.iddata(u[k+1:k+1,:], r, Ts)
    A,B,F,C,D,info = IdentificationToolbox.pem(zdata,firmodel,zeros(T,nᵣ*fir_m),options)
    û[k+1:k+1,:] += filt(B,F,r)
    σᵥ[k+1] = info.mse[1]
  end

  # initial Θ
  zdata   = iddata(y, û, Ts)
  options = IdOptions(iterations = 20, autodiff=true, estimate_initial=false)
  OEmodel = OE(m*ones(Int,1,nᵤ), m*ones(Int,1,nᵤ), ones(Int,1,nᵤ), 1, nᵤ)
  Θᵢ,_    = IdentificationToolbox._morsm(zdata, OEmodel, options)
  σᵥ[end] = IdentificationToolbox._mse(zdata, OEmodel, Θᵢ, options)[1]

  # sort Θ
  Θ = zeros(Θᵢ)
  for k = 0:nᵤ-1
    Θ[2k*m+(1:m)]     = Θᵢ[k*m+(1:m)]
    Θ[(2k+1)*m+(1:m)] = Θᵢ[nᵤ*m+k*m+(1:m)]
  end

  R      = _create_R(r.', nᵤ, nᵣ, nₛ, n, N)
  gₜ, Gₜ = _create_G(Θ, nᵤ, m, Ts, N)

  W      = vcat(R, Gₜ*R)
  z      = vcat(vec(u.'),vec(y.'))
  y      = view(z, N*nᵤ+1:N*(nᵤ+1))

  K     = _create_K(βᵥ, n)
  Σₑ    = spdiagm(kron(σᵥ,ones(T,N)))
  invΣₑ = spdiagm(kron(1./σᵥ,ones(T,N)))
  iΛ    = spdiagm(kron(1./λᵥ,ones(T,n)))
  P, s  = _create_Ps(iΛ*inv(K), invΣₑ, W, z)
  S     = P+s*s'
  û     = R*s
  sᵥ[:] = s

  ŷ  = Gₜ*R*sᵥ[:]
  P̂  = W*P*W'

  # update λ and β
  for i in 1:nₛ
    idx = (i-1)*n + (1:n)
    λᵥ[i], βᵥ[i] = basicQmin(view(S,idx,idx),100)
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

  return λᵥ, βᵥ, σᵥ, sᵥ, Θ, gₜ, Gₜ, R
end

# NEB cost functions
function Q₀{T}(Θ::AbstractVector{T}, A::AbstractMatrix{T},
  bb::AbstractVector{T}, N::Int, Ts::Float64, m::Int, nᵤ::Int)
  m::Int        = round(length(Θ)/2nᵤ)
  gₜ::Vector{T} = _create_g(Θ, nᵤ, m, Ts, N)

  return dot(gₜ,A*gₜ) - 2*dot(bb,gₜ)
end

function Qₙ{T}(Θ::AbstractVector{T}, U::AbstractMatrix{T},
  s::AbstractVector{T}, bb::AbstractVector{T}, N::Int, Ts::Float64, m::Int, nᵤ::Int)

  sumu = _quad_cost(Θ, U, s, m, N, nᵤ)
  gₜ   = _create_g(Θ, nᵤ, m, Ts, N)
  return sumu - 2*dot(bb,gₜ)
end
