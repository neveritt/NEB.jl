function basicEB{T}(z::AbstractVector{T},r::AbstractVector{T},
  n::Int=100, λ=100.0, β=.94, σ=0.01;
  maxiter::Int = 100, rtol::Float64=1e-6)
#  maxiter = method.maxiter
#  tol     = method.tol

  N  = length(z)
  n  = n>N ? N : n
  R  = full(Toeplitz(hcat(r),hcat(r[1,1],zeros(1,n-1))))
  TC = T[max(i,j) for i in 1:n, j in 1:n]

  K = Array{T}(n,n)
  W = Array{T}(N,n)
  S = Array{T}(N,N)
  s = Array{T}(N)
  P = Array{T}(N,N)

  η = vcat(λ, β, σ)
  RR = R.'*R
  Rz = R.'*z
  for k = 1:maxiter
    η₀ = η

    K = λ*β.^TC
    S = (K*RR/σ + I)\K
    s = S*Rz/σ
    P = S + s*s.'

    σ   = (sum(abs2,z) - 2*dot(Rz,s) + trace(RR*P))/N
    λ,β = basicQmin(P)

    η = vcat(λ, β, σ)
    if norm(η-η₀)/norm(η₀) < rtol
      return λ, β, σ, s
    end
  end
  return λ, β, σ, s
end
