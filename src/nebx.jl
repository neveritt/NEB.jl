immutable NEBXstate{T} <: IdentificationState
  Θ::Vector{T}
  σ::Vector{T}
  λ::Vector{T}
  β::Vector{T}
  s::Vector{T}
  f::Vector{T}
end


# function nebx{T}(y::AbstractVector{T}, u::AbstractMatrix{T},
#   r::AbstractMatrix{T}, n::Int, m::Int, zₜ::AbstractVector{T})
#
#   λₛ, βₛ, σₛ, sₛ, Θ, zₛ = NEB(y,u,r,n,m)
#   nᵤ = size(u,2)
#   nᵣ = size(r,2)
#   N  = size(r,1)
#   nebx(λₛ, βₛ, σₛ, sₛ, Θ, zₛ, zₜ, n, m, nᵤ, nᵣ, N)
#
#
#   λₛ, βₛ, σₛ, sₛ, Θ, zₛ = NEB(y,u,r,n,m)
#
#
#   z  = vcat(vec(u.'),vec(y.'))
#   W  = vcat(R, Gₜ*R)
#
#   # save state
#   NEBXtrace = [NEBXstate(copy(Θ), copy(σᵥ), copy(λᵥ), copy(βᵥ), copy(sᵥ))]
#   @inbounds for iter in 1:10
#     # println("iter", iter)
#     # println("Θ: ", Θ)
#     # println("σᵥ: ", σᵥ)
#     # println("λᵥ: ", λᵥ)
#     # println("βᵥ: ", βᵥ)
#     _iter_NEBX!(λᵥ, βᵥ, σᵥ, sᵥ, Θ, W, R, z, nᵤ)
#     state = NEBXstate(copy(Θ), copy(σᵥ), copy(λᵥ), copy(βᵥ), copy(sᵥ))
#     push!(NEBXtrace,state)
#   end
#   return NEBXtrace, z
#
# end
#
#
# function nebx{T}(λₛ::AbstractVector{T}, βₛ::AbstractVector{T},
#   σₛ::AbstractVector{T}, sₛ::AbstractVector{T}, Θ::AbstractVector{T},
#   zₛ::AbstractVector{T}, zₜ::AbstractVector{T},
#   n::Int, m::Int, nᵤ::Int, nᵣ::Int, N::Int)
#
#   λᵥ, βᵥ, σᵥ, sᵥ, fₛ, Θ, W, R, z = _initial_nebx(y,u,r,zₜ,n,m)
#
#   nebx(λᵥ, βᵥ, σᵥ, sₛ, fₛ[:], Θ, z, orders)
# end

function nebx{T}(y::AbstractMatrix{T}, u::AbstractMatrix{T},
  r::AbstractMatrix{T}, zₜ::AbstractMatrix{T}, orders::Vector{Int})
  n, m, nᵤ, nᵣ, N = orders
  nₛ = nᵤ*nᵣ

  state, z  = _initial_nebx(y,u,r,zₜ,orders)
  nebxtrace = [state]
  Θ  = copy(state.Θ)
  σᵥ = copy(state.σ)
  λᵥ = copy(state.λ)
  βᵥ = copy(state.β)
  sᵥ = copy(state.s)
  fₛ = copy(state.f)

  R      = _create_R(r.', nᵤ, nᵣ, nₛ, n, N)
  gₜ, Gₜ = _create_G(Θ, nᵤ, m, N)
  Xₜ     = full(Toeplitz(vcat(fₛ, zeros(N-n)), N))
  W      = vcat(R, Gₜ*R, Xₜ*Gₜ*R)
  # every iteration
  for iter in 1:40
    println(iter)
    _iter_nebx!(λᵥ, βᵥ, σᵥ, sᵥ, fₛ, Θ, W, R, z, orders)
    state = NEBXstate(copy(Θ), copy(σᵥ), copy(λᵥ), copy(βᵥ), copy(sᵥ), copy(fₛ))
    push!(nebxtrace, state)
  end

  return nebxtrace, z
end

function _initial_nebx{T}(y::AbstractMatrix{T}, u::AbstractMatrix{T},
  r::AbstractMatrix{T}, zₜ::AbstractMatrix{T},  orders::Vector{Int})
  n, m, nᵤ, nᵣ, N = orders
  nₛ = nᵤ*nᵣ

  nebtrace, zₛ = neb(y,u,r,n,m)
  Θ  = last(nebtrace).Θ
  σₛ = last(nebtrace).σ
  λₛ = last(nebtrace).λ
  βₛ = last(nebtrace).β
  sₛ = last(nebtrace).s

  nₛ = nᵤ*nᵣ
  z  = vcat(zₛ[:],zₜ[:])
  σₜ = 0.01
  λₜ = 100.0
  βₜ = 0.94

  R      = _create_R(r.', nᵤ, nᵣ, nₛ, n, N)
  gₜ, Gₜ = _create_G(Θ, nᵤ, m, N)
  û      = Gₜ*R*sₛ[:]

  λₜ, βₜ, σₜ, fₛ = basicEB(zₜ[:], û, n, λₜ, βₜ, σₜ)

  λᵥ = vcat(λₛ,[2*λₜ])
  βᵥ = vcat(βₛ,[βₜ])
  σᵥ = vcat(σₛ,[2*σₜ])
  sₛ = sₛ[:]
  NEBXstate(Θ, σᵥ, λᵥ, βᵥ, sₛ, fₛ), z
end

function _iter_nebx!{T}(λᵥ::AbstractVector{T}, βᵥ::AbstractVector{T},
    σᵥ::AbstractVector{T}, sₛ::AbstractVector{T}, fₛ::AbstractVector{T},
    Θ::AbstractVector{T}, W::AbstractMatrix{T}, R::AbstractMatrix{T},
    z::AbstractVector{T}, orders::AbstractVector{Int})
  n,m,nᵤ,nᵣ,N = orders
  nₛ = nᵤ*nᵣ

  y  = view(z, nᵤ*N+(1:N))
  zₛ = view(z, 1:(nᵤ+1)*N)
  zₜ = view(z, (nᵤ+1)*N+(1:N))
  σₛ = view(σᵥ, 1:nₛ)
  λₛ = view(λᵥ, 1:nₛ)
  βₛ = view(βᵥ, 1:nₛ)
  σₛ = view(σᵥ, 1:nᵤ+1)
  λₜ = view(λᵥ, nₛ+1:nₛ+1)
  βₜ = view(βᵥ, nₛ+1:nₛ+1)
  σₜ = view(σᵥ, nᵤ+2:nᵤ+2)

  burnin = 1000
  nsteps = 2000
  M = burnin + nsteps

  Sₘ = zeros(T, nₛ*n, nsteps)
  Fₘ = zeros(T, n, nsteps)

  Kₛ    = _create_K(βₛ, n)
  Λₛ    = spdiagm(kron(λₛ,ones(T,n)))
  KΛₛ   = Kₛ*Λₛ
  invΣₛ = spdiagm(kron(1./σᵥ,ones(T,N)))

  Kₜ    = _create_K(βₜ, n)
  Λₜ    = spdiagm(kron(λₜ,ones(T,n)))
  KΛₜ   = Kₜ*Λₜ
  invΣₜ = spdiagm(kron(1./σₜ,ones(T,N)))
  gₜ, Gₜ = _create_G(Θ, nᵤ, m, N)

  for iₘ in 1:M
    _NEB_gibbs!(fₛ, sₛ, KΛₛ, invΣₛ, KΛₜ, invΣₜ, W, nᵤ,
    Gₜ, R, Sₘ, Fₘ, iₘ, burnin)
  end

  fₛ    = mean(Fₘ[:,burnin+1:end],2)[:]
  sₛ    = mean(Sₘ[:,burnin+1:end],2)[:]
  Mₜ    = cov(Fₘ[:,burnin+1:end],2,false) + fₛ*fₛ.'
  Mₛ    = cov(Sₘ[:,burnin+1:end],2,false) + sₛ*sₛ.'

  v = R*sₛ
  #V = Toeplitz(v[:],N)
  V = zeros(T,N,nᵤ*N)
  for i = 1:nᵤ
    V[:,(i-1)*N+(1:N)] = Toeplitz(v[(i-1)*N+(1:N)],N)
  end
  F = full(Toeplitz(vcat(fₛ[:],zeros(N-n)),N))
  FV = F*V

  # update Θ
  df = TwiceDifferentiableFunction(x -> Qₜ(
    x, zₜ, zₛ[end-N+1:end], FV, V, σᵥ[end], σᵥ[end-1], N, m, nᵤ))
  options  = Optim.Options(autodiff = true, g_tol = 1e-32)
  opt = optimize(df, Θ, Newton(), options)
  Θ[:] = opt.minimizer

  # update hyper parameters
  for i in 1:nₛ
    idx = (i-1)*n + (1:n)
    λᵥ[i], βᵥ[i] = basicQmin(view(Mₛ,idx,idx))
  end
  λᵥ[end], βᵥ[end] = basicQmin(Mₜ)

  # update noise
  gₜ, Gₜ = _create_G(Θ, nᵤ, m, N)
  Xₜ     = full(Toeplitz(vcat(fₛ[:],zeros(N-n)), N))
  W[:]   = vcat(R, Gₜ*R, Xₜ*Gₜ*R)

  Eᵢ = z-W*sₛ
  for i = 1:nᵤ
    idx    = (i-1)*N + (1:N)
    eᵢ     = view(Eᵢ,idx)
    σᵥ[i]  = sumabs2(eᵢ)/N
  end
  σᵥ[nᵤ+1] = sumabs2(Eᵢ[nᵤ*N+(1:N)])/N
  σᵥ[end]  = sumabs2(Eᵢ[end-N+(1:N)])/N
end

function _NEB_gibbs!{T}(fₛ::AbstractVector{T}, sₛ::AbstractVector{T},
  KΛₛ::AbstractMatrix{T}, invΣₛ::AbstractMatrix{T}, KΛₜ::AbstractMatrix{T},
  invΣₜ::AbstractMatrix{T}, W::AbstractMatrix{T}, nᵤ::Int,
  Gₜ::AbstractMatrix{T}, R::AbstractMatrix{T}, Sₘ::AbstractMatrix{T}, Fₘ::AbstractMatrix{T},
  iₘ::Int, burnin::Int)

  N  = size(Gₜ, 2)
  n  = size(KΛₜ,2)

  ## sample S
  Pₛ, s = _create_Ps(KΛₛ, invΣₛ, W, z)
  Covₛ  = PDMat(cholfact(Hermitian(Pₛ)))

  Ssampler = MvNormal(s[:], Covₛ)
  sₛ[:]    = rand(Ssampler,1)

  ## sample T
  Sₜ    = full(Toeplitz(R*sₛ[:], n))
  Wₜ    = Gₜ*Sₜ
  Pₜ, f = _create_Ps(KΛₜ, invΣₜ, Wₜ, zₜ)
  Covₜ  = PDMat(cholfact(Hermitian(Pₜ)))

  Fsampler = MvNormal(f[:], Covₜ)
  fₛ[:]    = rand(Fsampler,1)

  if iₘ > burnin
    Fₘ[:,iₘ-burnin] = fₛ
    Sₘ[:,iₘ-burnin] = sₛ
  end
end

function Qₜ(Θ, zₜ, zₛ, FV, V, σₜ, σₛ, N, m, nᵤ)
  gₜ = _create_g(Θ, nᵤ, m, N)
  return sumabs2(zₜ - FV*gₜ)/σₜ + sumabs2(zₛ - V*gₜ)/σₛ
end
