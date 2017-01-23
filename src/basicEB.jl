function basicEB{T}(z::AbstractArray{T},r::AbstractArray{T},
  n::Int=100, λ=100.0, β=.94, σ=0.01;
  maxiter::Int = 10, tol::Float64=1e-4)
#  maxiter = method.maxiter
#  tol     = method.tol

  N  = size(z,1)
  n  = n>N ? N : n
  R  = full(Toeplitz(hcat(r),hcat(r[1,1],zeros(1,n-1))))
  TC = T[max(i,j) for i in 1:n, j in 1:n]

  K = Array(T,n,n)
  W = Array(T,N,n)
  S = Array(T,N,N)
  s = Array(T,N)
  P = Array(T,N,N)

  η = [λ; β; σ]
  k = 1
  for k = 1:maxiter
    η₀ = η

    K = λ*β.^TC
    W = R*(1/sqrt(σ))
    S = (K*(W.'*W) + eye(n))\K
    s = S*W.'*z/σ
    P = S + s*s.'

    σ   = (sumabs2(z) - 2*dot(z,R*s) + trace(R'*R*P))/N
    λ,β = basicQmin(P)

    η  = [λ; β; σ]
    if norm(η-η₀)/norm(η₀) < tol
      break
    end
  end
  return λ, β, σ, s
end
