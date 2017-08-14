function Toeplitz{T<:Number}(g::Vector{T}, N::Int)
  col = zeros(T,1,N)
  col[1] = g[1]
  Toeplitz(reshape(g,length(g),1), col)
end

function _create_G{T}(Θ::AbstractVector{T}, nᵤ::Int, m::Int, Ts::Float64, N::Int)
  Gₜ = zeros(T,N,N*nᵤ)
  gₜ = zeros(T,N*nᵤ)
  for i in 0:nᵤ-1
    Θᵢ              = Θ[i*2m+(1:2m)]
    gₜ[i*N+(1:N)]   = impulse(vcat(zeros(T,1),Θᵢ[1:m]), vcat(ones(T,1),Θᵢ[m+(1:m)]),Ts,N)
    Gₜ[:,i*N+(1:N)] = Toeplitz(gₜ[i*N+(1:N)],N)
  end
  return gₜ, Gₜ
end

function _create_g{T}(Θ::AbstractVector{T}, nᵤ::Int, m::Int, Ts::Float64, N::Int)
  gₜ = zeros(T,N*nᵤ)
  for i in 0:nᵤ-1
    Θᵢ              = Θ[i*2m+(1:2m)]
    gₜ[i*N+(1:N)]   = impulse(vcat(zeros(T,1),Θᵢ[1:m]), vcat(ones(T,1),Θᵢ[m+(1:m)]),Ts,N)
  end
  gₜ
end

# create R = [R_1...R_nr; 0 … 0;
#             0     ⋱    0 … 0
#             0  …  0   R_1 …  R_nr]
function _create_R{T}(r::AbstractMatrix{T}, nᵤ::Int, nᵣ::Int, nₛ::Int, n::Int, N::Int)
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

function _create_Ps{T}(iKΛ::AbstractMatrix{T}, invΣₑ::AbstractMatrix{T},
  W::AbstractMatrix{T}, z::AbstractVector{T})
  P  = inv(W.'*invΣₑ*W + iKΛ)
  s  = P*(W'*(invΣₑ*z))
  return P, s
end

# create K = [TC_1; 0 … 0;
#             0     ⋱    0 … 0
#             0  …  0   TC_nₛ]
function _create_K{T}(βᵥ::AbstractVector{T}, n::Int)
  TC  = T[max(i,j) for i in 1:n, j in 1:n]
  nₛ = length(βᵥ)
  K  = zeros(Float64,n*nₛ,n*nₛ)
  for i in 0:nₛ-1
    K[i*n+(1:n), i*n+(1:n)] = βᵥ[i+1].^TC
  end
  return K
end

function _quad_cost{T}(U::AbstractMatrix{T}, s::AbstractVector{T})
  sumc = zero(T)
  for i in 1:length(s)
    sumc += s[i]*sum(abs2,U[:,i])
  end
  return sumc
end

function _quad_cost{T}(Θ::AbstractVector{T}, U::AbstractMatrix{T},
  s::AbstractVector{T}, m::Int, N::Int, nᵤ::Int)
  b = Vector{Vector{T}}(nᵤ)
  a = Vector{Vector{T}}(nᵤ)
  for i in 0:nᵤ-1
    Θᵢ = Θ[i*2m+(1:2m)]
    b[i+1]  = vcat(zeros(T,1), Θᵢ[1:m])
    a[i+1]  = vcat(ones(T,1), Θᵢ[m+(1:m)])
  end
  sumc = zero(T)
  for i in 1:length(s)
    vj = zeros(T,N)
    for j in 1:nᵤ
      idxj = (j-1)*N+(1:N)
      vj += filt(b[j], a[j], U[idxj,i])
    end
    sumc += s[i]*sum(abs2, vj)
  end
  return sumc
end
