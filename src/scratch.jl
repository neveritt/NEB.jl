using GeneralizedSchurAlgorithm
using Distributions
using PDMats
using Optim
λ₀ = 100.0
β₀ = .94
σ₀ = 0.01
Ts = 1.0
N  = 60
b1 = [0,0.7,-0.3]
b2 = [0,0.5,-0.2]
r1 = randn(N)
r2 = randn(N)
u0 = filt(b1,1,r1) + filt(b2,1,r2)
u = hcat(u0) + 0.2*randn(N)
TT = Float64
r = hcat(r1,r2)

b3 = [0, 0.7, -0.3]
a3 = [1, -0.3, 0.1]
y0 = filt(b3,a3,u0)
y  = y0 + 0.2*randn(N)
Θ₀ = vcat(b3[2:3], a3[2:3])

m  = 2
n  = 6
nᵤ = 1
nᵣ = 2
nₛ = nᵤ*nᵣ

b4 = [0, 1.3, 0.6, -0.1]
fₛ = impulse(tf(b4,[1.0],Ts,var=:zinv),n)
Xₜ = Toeplitz(vcat(fₛ,zeros(N-n)),N)
y2 = filt(b4,1,y0) + 0.1*randn(N)

λₛ, βₛ, σₛ, sₛ, Θ, zₛ = NEB(y,u,r,n,m)

gₜ = impulse(tf(vcat(zeros(1), Θ[1:m]),vcat(ones(1), Θ[m+1:2m]),Ts),N)
g₀ = impulse(tf(vcat(zeros(1), Θ₀[1:m]),vcat(ones(1), Θ₀[m+1:2m]),Ts),N)

sumabs2(gₜ-g₀)/sumabs2(g₀)


# nebx
λᵥ, βᵥ, σᵥ, sₛ, fₛ, Θ, z = nebx(λₛ, βₛ, σₛ, sₛ[:], Θ, zₛ, y2, n, m, nᵤ, nᵣ, N)

orders = [n, m, nᵤ, nᵣ, N]
for i = 1:2
  λᵥ, βᵥ, σᵥ, sₛ, fₛ, Θ, z = nebx(λᵥ, βᵥ, σᵥ, sₛ, fₛ, Θ, z, orders)
end


#nebx(y,u,r,n,m,y2)

gₜ = impulse(tf(vcat(zeros(1), Θ[1:m]),vcat(ones(1), Θ[m+1:2m]),Ts),N)
g₀ = impulse(tf(vcat(zeros(1), Θ₀[1:m]),vcat(ones(1), Θ₀[m+1:2m]),Ts),N)

sumabs2(gₜ-g₀)/sumabs2(g₀)

λᵥ, βᵥ, σᵥ,



λᵥ, βᵥ, σᵥ,


λᵥ, βᵥ, σᵥ,




λᵥ, βᵥ, σᵥ, sᵥ, Θ, gₜ, Gₜ, R = _initial_NEB(y,u,r,n,m)


S = sᵥ[:]*sᵥ[:].'

R*S*R.'

U,sᵥ,V = svd(S)
Uᵣ = R*U

sums = 0.0
for i in 1:length(sᵥ)
  sums += sᵥ[i]*sumabs2(Uᵣ[:,i])
end
sums
trace(R*S*R.')

sumabs2(Uᵣ[:,1])

filt([1.0],[1.0], Uᵣ)
