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

function nebx{T,O}(y::AbstractMatrix{T}, u::AbstractMatrix{T},
  r::AbstractMatrix{T}, zₜ::AbstractMatrix{T}, orders::Vector{Int}, Ts::Float64;
  options::IdOptions{O}=IdOptions())
  iterations = options.OptimizationOptions.iterations
  n, m, nᵤ, nᵣ, N = orders
  nₛ = nᵤ*nᵣ

  state, z  = _initial_nebx(y,u,r,zₜ,orders, Ts)
  nebxtrace = [state]
  Θ  = copy(state.Θ)
  σᵥ = copy(state.σ)
  λᵥ = copy(state.λ)
  βᵥ = copy(state.β)
  sᵥ = copy(state.s)
  fₛ = copy(state.f)

  R      = _create_R(r.', nᵤ, nᵣ, nₛ, n, N)
  R₂     = _create_R(r.', nᵤ, nᵣ, nₛ, 2n-1, N)
  gₜ, Gₜ = _create_G(Θ, nᵤ, m, Ts, N)
  Fₜ     = full(Toeplitz(vcat(fₛ, zeros(N-n)), N))
  W      = vcat(R, Gₜ*R, Fₜ*Gₜ*R)

  burnin = 1000
  nsteps = 4000
  Sₘ = zeros(T, nₛ*n, nsteps)
  Fₘ = zeros(T, n, nsteps)
  Vₘ = zeros(T, nₛ*(2n-1), nsteps)
  # every iteration
  η = vcat(Θ, σᵥ, λᵥ, βᵥ)
  for iter in 1:iterations
    ηold = η
    _iter_nebx!(λᵥ, βᵥ, σᵥ, sᵥ, fₛ, Θ, W, R, R₂, z, orders, Ts, burnin, nsteps,
                Sₘ, Fₘ, Vₘ)
    state = NEBXstate(copy(Θ), copy(σᵥ), copy(λᵥ), copy(βᵥ), copy(sᵥ), copy(fₛ))
    push!(nebxtrace, state)
    η = vcat(Θ, σᵥ, λᵥ, βᵥ)
    if norm(η-ηold)/norm(η) < options.OptimizationOptions.x_tol
      return NEBtrace, z
    end
  end

  return nebxtrace, z
end

function _initial_nebx{T}(y::AbstractMatrix{T}, u::AbstractMatrix{T},
  r::AbstractMatrix{T}, zₜ::AbstractMatrix{T},  orders::Vector{Int}, Ts::Float64)
  n, m, nᵤ, nᵣ, N = orders
  nₛ = nᵤ*nᵣ

  nebtrace, zₛ = neb(y,u,r,n,m,Ts; options=IdOptions(iterations=50))
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
  gₜ, Gₜ = _create_G(Θ, nᵤ, m, Ts, N)
  û      = Gₜ*R*sₛ[:]

  λₜ, βₜ, σₜ, fₛ = basicEB(zₜ[:], û, n, λₜ, βₜ, σₜ)

  λᵥ = vcat(λₛ,[λₜ])
  βᵥ = vcat(βₛ,[βₜ])
  σᵥ = vcat(σₛ,[σₜ])
  sₛ = sₛ[:]
  NEBXstate(Θ, σᵥ, λᵥ, βᵥ, sₛ, fₛ), z
end

function _iter_nebx!{T}(λᵥ::AbstractVector{T}, βᵥ::AbstractVector{T},
    σᵥ::AbstractVector{T}, sₛ::AbstractVector{T}, fₛ::AbstractVector{T},
    Θ::AbstractVector{T}, W::AbstractMatrix{T}, R::AbstractMatrix{T},
    R₂::AbstractMatrix{T}, z::AbstractVector{T}, orders::AbstractVector{Int},
    Ts::Float64, burnin::Int=1000, nsteps::Int=2000,
    Sₘ=zeros(T,nₛ*n,nsteps), Fₘ=zeros(T,n,nsteps), Vₘ=zeros(T,nₛ*(2n-1),nsteps))
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

  Kₛ    = _create_K(βₛ, n)
  Λₛ    = spdiagm(kron(λₛ,ones(T,n)))
  iKΛₛ  = inv(Kₛ*Λₛ)
  invΣₛ = spdiagm(kron(1./σᵥ,ones(T,N)))

  Kₜ    = _create_K(βₜ, n)
  Λₜ    = spdiagm(kron(λₜ,ones(T,n)))
  iKΛₜ  = inv(Kₜ*Λₜ)
  invΣₜ = spdiagm(kron(1./σₜ,ones(T,N)))
  gₜ,Gₜ = _create_G(Θ, nᵤ, m, Ts, N)

  M = burnin + nsteps
  for iₘ in 1:M
    _NEB_gibbs!(fₛ, sₛ, iKΛₛ, invΣₛ, iKΛₜ, invΣₜ, W, z, zₜ,
    Gₜ, R, Sₘ, Fₘ, Vₘ, orders, iₘ, burnin)
  end

  fₛ = mean(Fₘ[:,1:end],2)[:]
  sₛ = mean(Sₘ[:,1:end],2)[:]
  vₛ = mean(Vₘ[:,1:end],2)[:]
  Cₜ = cov(Fₘ[:,1:end],2,false)
  Cₛ = cov(Sₘ[:,1:end],2,false)
  Cᵥ = cov(Vₘ[:,1:end],2,false)
  Mₜ = Cₜ + fₛ*fₛ.'
  Mₛ = Cₛ + sₛ*sₛ.'
  Mᵥ = Cᵥ + vₛ*vₛ.'

  v = R*sₛ
  V = zeros(T,N,nᵤ*N)
  for i = 1:nᵤ
    V[:,(i-1)*N+(1:N)] = Toeplitz(v[(i-1)*N+(1:N)],N)
  end
  F = full(Toeplitz(vcat(fₛ[:],zeros(N-n)),N))
  FV = F*V

  Uc,st,Vc = svd(Mᵥ)
  Uₜ       = R₂*Uc
  Uc,ss,Vc = svd(Mₛ)
  Uₛ       = R*Uc

  bₜ = _create_b(zₜ, R₂*vₛ, nᵤ, N)/σᵥ[end]
  bₛ = _create_b(y, v, nᵤ, N)/σᵥ[end-1]

  # update Θ
  df = TwiceDifferentiableFunction(x -> Qₙ(
      x, bₜ, bₛ, σᵥ[end], σᵥ[end-1], Uₜ, st, Uₛ, ss, N, Ts, m, nᵤ))
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
  gₜ, Gₜ = _create_G(Θ, nᵤ, m, Ts, N)
  Fₜ     = full(Toeplitz(vcat(fₛ[:],zeros(N-n)), N))
  W[:]   = vcat(R, Gₜ*R, Fₜ*Gₜ*R)

  Uc,ss,Vc = svd(Cₛ)
  Uₛ       = R*Uc
  Uc,st,Vc = svd(Cᵥ)
  Uₜ       = R₂*Uc

  Eᵢ  = z-W*sₛ
  MM1 = R*Cₛ*R.'
  for i = 1:nᵤ
    idx   = (i-1)*N + (1:N)
    eᵢ    = view(Eᵢ,idx)
    σᵥ[i] = (sumabs2(eᵢ) + trace(view(MM1,idx,idx)))/N
  end
  σᵥ[nᵤ+1] = (sumabs2(Eᵢ[nᵤ*N+(1:N)]) + _quad_cost(Θ, Uₛ, ss, m, N, nᵤ))/N
  σᵥ[end]  = (sumabs2(Eᵢ[end-N+(1:N)]) + _quad_cost(Θ, Uₜ, st, m, N, nᵤ))/N
end

function _NEB_gibbs!{T}(fₛ::AbstractVector{T}, sₛ::AbstractVector{T},
  iKΛₛ::AbstractMatrix{T}, invΣₛ::AbstractMatrix{T}, iKΛₜ::AbstractMatrix{T},
  invΣₜ::AbstractMatrix{T}, W::AbstractMatrix{T}, z::AbstractVector{T},
  zₜ::AbstractVector{T}, Gₜ::AbstractMatrix{T}, R::AbstractMatrix{T},
  Sₘ::AbstractMatrix{T}, Fₘ::AbstractMatrix{T}, Vₘ::AbstractMatrix{T},
  orders::Vector{Int}, iₘ::Int, burnin::Int)
  n,m,nᵤ,nᵣ,N = orders
  nₛ = nᵤ*nᵣ

  ## sample S
  Pₛ, s = _create_Ps(iKΛₛ, invΣₛ, W, z)
  Covₛ  = PDMat(cholfact(Hermitian(Pₛ)))

  Ssampler = MvNormal(s[:], Covₛ)
  sₛ[:]    = rand(Ssampler,1)

  ## sample T
  Wₜ    = full(Toeplitz(Gₜ*R*sₛ[:], n))
  Pₜ, f = _create_Ps(iKΛₜ, invΣₜ, Wₜ, zₜ)
  Covₜ  = PDMat(cholfact(Hermitian(Pₜ)))

  Fsampler = MvNormal(f[:], Covₜ)
  fₛ[:]    = rand(Fsampler,1)

  # update W for next iteration
  Fₜ = full(Toeplitz(vcat(fₛ, zeros(N-n)), N))
  W[end-N+(1:N),:] = Fₜ*Gₜ*R

  if iₘ > burnin
    Fₘ[:,iₘ-burnin] = fₛ
    Sₘ[:,iₘ-burnin] = sₛ
    for i = 0:nₛ-1
      Vₘ[i*(2n-1)+(1:2n-1),iₘ-burnin] = conv(fₛ,sₛ[i*n+(1:n)])
    end
  end
end

# NEBX cost functions
function Qₙ(Θ, bₜ, bₛ, σₜ, σₛ, Uₜ, sₜ, Uₛ, sₛ, N, Ts, m, nᵤ)
  gₜ   = _create_g(Θ, nᵤ, m, Ts, N)
  sums = _quad_cost(Θ, Uₛ, sₛ, m, N, nᵤ)
  sumt = _quad_cost(Θ, Uₜ, sₜ, m, N, nᵤ)

  return sumt/σₜ + sums/σₛ - 2*dot(bₜ,gₜ) - 2*dot(bₛ,gₜ)
end

function Qₜ(Θ, zₜ, zₛ, FV, V, σₜ, σₛ, Uₜ, sₜ, Uₛ, sₛ, N, Ts, m, nᵤ)
  gₜ   = _create_g(Θ, nᵤ, m, Ts, N)
  sumt = _quad_cost(Θ, Uₜ, sₜ, m, N, nᵤ)
  sums = _quad_cost(Θ, Uₛ, sₛ, m, N, nᵤ)

  return sumabs2(zₜ - FV*gₜ)/σₜ + sumabs2(zₛ - V*gₜ)/σₛ +
    sumt/σₜ + sums/σₛ
end
