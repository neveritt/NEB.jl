using GeneralizedSchurAlgorithm
using Distributions
using PDMats
using Optim
λ₀ = 100.0
β₀ = .94
σ₀ = 0.01
Ts = 1.0
N  = 100
b1 = [0,0.7,-0.3]
b2 = [0,0.5,-0.2]
r1 = randn(N,1)
r2 = randn(N,1)
u0 = filt(b1,1,r1) + filt(b2,1,r2)
u = hcat(u0) + 0.1*randn(N,1)
TT = Float64
r = hcat(r1,r2)

b3 = [0, 0.7, -0.3]
a3 = [1, -0.3, 0.1]
y0 = filt(b3,a3,u0)
y  = y0 + 0.1*randn(N,1)
Θ₀ = vcat(b3[2:3], a3[2:3])

m  = 2
n  = 20
nᵤ = 1
nᵣ = 2
nₛ = nᵤ*nᵣ

b4 = [0, 1.3, 0.6, -0.1]
fₛ = impulse(tf(b4,[1.0],Ts,var=:zinv),n)
Xₜ = Toeplitz(vcat(fₛ,zeros(N-n)),N)
y2 = filt(b4,1,y0) + 0.1*randn(N,1)

λₛ, βₛ, σₛ, sₛ, Θ, zₛ = NEB(y,u,r,n,m)


gₜ = impulse(tf(vcat(zeros(1), Θ[1:m]),vcat(ones(1), Θ[m+1:2m]),Ts),N)
g₀ = impulse(tf(vcat(zeros(1), Θ₀[1:m]),vcat(ones(1), Θ₀[m+1:2m]),Ts),N)

sumabs2(gₜ-g₀)/sumabs2(g₀)


## Extension
zₜ = y2[:]
z = vcat(zₛ[:],y2[:])
σₜ = 0.01
λₜ = 100.0
βₜ = 0.94

R  = _create_R(r, nᵤ, nᵣ, nₛ, n, N)
Gₜ = Toeplitz(gₜ, N)
û  = Gₜ*R*sₛ[:]

λₜ, βₜ, σₜ, fₛ = basicEB(zₜ, û, n, λₜ, βₜ, σₜ)

λᵥ = vcat(λₛ,[λₜ])
βᵥ = vcat(βₛ,[βₜ])
σᵥ = vcat(σₛ,[σₜ])
sₛ = sₛ[:]

burnin = 300
nsteps = 700
M = burnin + nsteps

Sₘ = zeros(TT, nₛ*n, nsteps)
Fₘ = zeros(TT, n, nsteps)
Vₘ = zeros(TT, nₛ*(2n-1), nsteps)

function Qₜ(Θ, zₜ, zₛ, F, V, σₜ, σₛ, N, Ts)
  m::Int = round(length(Θ)/2)
  b = Θ[1:m]
  a = Θ[m+1:end]
  gₜ = impulse(tf(vcat(zeros(1), b),vcat(ones(1), a),Ts),N)
  return sumabs2(zₜ - F*V*gₜ)/σₜ^2 + sumabs2(zₛ - V*gₜ)/σₛ^2
end

function _NEB_gibbs!{T}(fₛ::AbstractVector{T}, sₛ::AbstractVector{T},
  KΛₛ::AbstractMatrix{T}, invΣₛ::AbstractMatrix{T}, KΛₜ::AbstractMatrix{T},
  invΣₜ::AbstractMatrix{T}, W::AbstractMatrix{T}, nᵤ::Int,
  Gₜ::AbstractMatrix{T}, R::AbstractMatrix{T}, Sₘ::AbstractMatrix{T}, Fₘ::AbstractMatrix{T},
  Vₘ::AbstractMatrix{T}, iₘ::Int, burnin::Int)

  N  = size(Gₜ, 2)
  n  = size(KΛₜ,2)

  ## sample S
  Pₛ, s = _create_Ps(KΛₛ, invΣₛ, W, z)
  Covₛ  = PDMat(cholfact(Hermitian(Pₛ)))

  Ssampler = MvNormal(s[:], Covₛ)
  sₛ[:]    = rand(Ssampler,1)

  ## sample F
  Sₜ    = Toeplitz(R*sₛ[:], n)
  Wₜ    = Gₜ*Sₜ
  Pₜ, f = _create_Ps(KΛₜ, invΣₜ, Wₜ, zₜ)
  Covₜ  = PDMat(cholfact(Hermitian(Pₜ)))

  Fsampler = MvNormal(f[:], Covₜ)
  fₛ[:]    = rand(Fsampler,1)

  if iₘ > burnin
    Fₘ[:,iₘ-burnin] = fₛ
    Sₘ[:,iₘ-burnin] = sₛ
    for i = 1:2
      Vₘ[(i-1)*(2n-1)+(1:2n-1),iₘ-burnin] = conv(fₛ,sₛ[(i-1)*n+(1:n)])
    end
  end
end

# intialization
σₛ = view(σᵥ, 1:nₛ)
λₛ = view(λᵥ, 1:nₛ)
βₛ = view(βᵥ, 1:nₛ)
σₛ = view(σᵥ, 1:nᵤ+1)
λₜ = view(λᵥ, nₛ+1:nₛ+1)
βₜ = view(βᵥ, nₛ+1:nₛ+1)
σₜ = view(σᵥ, nᵤ+2:nᵤ+2)
R  = _create_R(r, nᵤ, nᵣ, nₛ, n, N)
R₂ = _create_R(r, nᵤ, nᵣ, nₛ, 2n-1, N)

gₜ = impulse(tf(vcat(zeros(1), Θ[1:m]),vcat(ones(1), Θ[m+1:2m]),Ts),N)
Gₜ = Toeplitz(gₜ, N)
Xₜ = Toeplitz(vcat(fₛ[:],zeros(N-n)), N)
W  = vcat(R, Gₜ*R, Xₜ*Gₜ*R)

function Qₛ(Θ, zₜ, zₛ, FV, V, σₜ, σₛ, Aₜ, Aₛ, N, Ts)
  m::Int = round(length(Θ)/2)
  b = Θ[1:m]
  a = Θ[m+1:end]
  gₜ = impulse(tf(vcat(zeros(1), b),vcat(ones(1), a),Ts),N)
  #Gₜ = Toeplitz(gₜ,N)
  return sumabs2(zₜ - FV*gₜ)/σₜ^2 + sumabs2(zₛ - V*gₜ)/σₛ^2 +
    dot(gₜ,Aₜ*gₜ)/σₜ^2 + dot(gₜ,Aₛ*gₜ)/σₛ^2
end

# every iteration
for iter in 1:20
  println(iter)

  Kₛ    = _create_K(βₛ, n)
  Λₛ    = spdiagm(kron(λₛ,ones(TT,n)))
  KΛₛ   = Kₛ*Λₛ
  invΣₛ = spdiagm(kron(1./σᵥ.^2,ones(TT,N)))

  Kₜ    = _create_K(βₜ, n)
  Λₜ    = spdiagm(kron(λₜ,ones(TT,n)))
  KΛₜ   = Kₜ*Λₜ
  invΣₜ = spdiagm(kron(1./σₜ.^2,ones(TT,N)))

  for iₘ in 1:M
    _NEB_gibbs!(fₛ, sₛ, KΛₛ, invΣₛ, KΛₜ, invΣₜ, W, nᵤ,
    Gₜ, R, Sₘ, Fₘ, Vₘ, iₘ, burnin)
  end

  fₛ = mean(Fₘ[:,burnin+1:end],2)[:]
  sₛ = mean(Sₘ[:,burnin+1:end],2)[:]
  vₛ = mean(Vₘ[:,burnin+1:end],2)[:]
  Cₜ = cov(Fₘ[:,burnin+1:end],2,false)
  Cₛ = cov(Sₘ[:,burnin+1:end],2,false)
  Cᵥ = cov(Vₘ[:,burnin+1:end],2,false)
  Mₜ = Cₜ + fₛ*fₛ.'
  Mₛ = Cₛ + sₛ*sₛ.'
  Mᵥ = Cᵥ + vₛ*vₛ.'

  # setup Θ optimization
  uₛ  = R*sₛ
  U   = Toeplitz(uₛ[:],N)
  FV =  Toeplitz(R₂*vₛ,N)

  MM1 = R₂*Cᵥ*R₂.'
  MM2 = R*Cₛ*R.'
  Aₜ = _create_A(MM2,N)
  Aₛ = _create_A(MM1,N)
  #MM1 = X_A_Xt(PDMat(cholfact(Hermitian(Cₛ))),R) + 1e-12*speye(N)
  #MM2 = X_A_Xt(PDMat(Cᵥ),R₂) + 1e-12*speye(N)
  #Aₜ = _create_A(R₂*Mᵥ*R₂.',N)/σᵥ[3]^2
  #Aₛ = _create_A(R*Mₛ*R.',N)/σᵥ[2]^2
  #bₜ = _create_b(zₜ[:], R₂*vₛ, nᵤ, N)/σᵥ[3]^2
  #bₛ = _create_b(zₛ[N+1:2N], R*sₛ, nᵤ, N)/σᵥ[2]^2

  # update Θ
  df = TwiceDifferentiableFunction(x -> Qₛ(
    x, zₜ[:], zₛ[N+1:2N], FV, U, σᵥ[3], σᵥ[2], Aₜ, Aₛ, N, Ts))
  #df = TwiceDifferentiableFunction(x -> Q₀(
  #  x, Aₜ + Aₛ, bₜ + bₛ, N, Ts))
  options  = OptimizationOptions(autodiff = true, g_tol = 1e-32)
  opt = optimize(df, Θ, Newton(), options)
  Θ = opt.minimum

  # update hyper parameters
  for i in 1:nₛ
    idx = (i-1)*n + (1:n)
    λᵥ[i], βᵥ[i] = basicQmin(view(Mₛ,idx,idx))
  end
  λᵥ[end], βᵥ[end] = basicQmin(Mₜ)

  # update noise
  #σᵥ[:] = Eₘ/nsteps/N
  gₜ = impulse(tf(vcat(zeros(1), Θ[1:m]),vcat(ones(1), Θ[m+1:2m]),Ts),N)
  Gₜ = Toeplitz(gₜ, N)
  Xₜ = Toeplitz(vcat(fₛ[:],zeros(N-n)), N)
  W  = vcat(R, Gₜ*R, Xₜ*Gₜ*R)

  Eᵢ = z-W*sₛ
  for i = 1:nᵤ
    idx    = (i-1)*N + (1:N)
    eᵢ     = view(Eᵢ,idx)
    σᵥ[i]  = sqrt((sumabs2(eᵢ) + trace(MM1[idx,idx]))/N)
  end
  σᵥ[nᵤ+1] = sqrt((sumabs2(Eᵢ[nᵤ*N+(1:N)]) + dot(gₜ.',_create_A(MM1,N)*gₜ))/N)
  σᵥ[end]  = sqrt((sumabs2(Eᵢ[end-N+(1:N)]) + dot(gₜ.',_create_A(MM2,N)*gₜ))/N)

  gₜ = impulse(tf(vcat(zeros(1), Θ[1:m]),vcat(ones(1), Θ[m+1:2m]),Ts),N)
  println(sumabs2(gₜ-g₀))
  println(σᵥ)
end

gₜ = impulse(tf(vcat(zeros(1), Θ[1:m]),vcat(ones(1), Θ[m+1:2m]),Ts),N)

sumabs2(gₜ-g₀)
