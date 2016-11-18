immutable EB <: IdentificationToolbox.OneStepIdMethod
  maxiter::Int
  tol::Float64

  @compat function (::Type{EB})(maxiter::Int, tol::Float64)
    new(maxiter,tol)
  end
end

function EB(; maxiter::Int=1000, tol::Float64=1e-6)
  EB(maxiter,tol)
end

function fitmodel{T<:Real}(data::IdDataObject{T}, n::Vector{Int}, method::FIR; kwargs...)
  basicEB(data, n, method)
end

function basicEB{T}(data::IdDataObject{T}, n, method::EB=EB())
  basicEB(data.y,data.u, n[1])
end

function basicEB{T}(z::AbstractArray{T},r::AbstractArray{T},
  n::Int=100, λ=100.0, β=.94, σ=0.01; method::EB=EB())
  maxiter = method.maxiter
  tol     = method.tol

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
    s = S*W.'*z/sqrt(σ)
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
