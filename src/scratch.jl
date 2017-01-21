using GeneralizedSchurAlgorithm
using Distributions
using PDMats
using Optim

N  = 200
nᵣ = 2
nᵤ = 2
nₛ = nᵤ*nᵣ
using DataFrames
using Polynomials
using Optim
using NetworkEmpiricalBayes
using IdentificationToolbox
import NetworkEmpiricalBayes.impulse

dfr = readtable("networkdatar.dat")
r0 = Matrix{Float64}(200,2)
r0[:,1] = dfr[:,1]
r0[:,2] = dfr[:,2]

dfu = readtable("networkdatau.dat")
u0 = Matrix{Float64}(200,2)
u0[:,1] = dfu[:,1]
u0[:,2] = dfu[:,2]

dfy = readtable("networkdatay.dat")
y0 = Matrix{Float64}(200,2)
y0[:,1] = dfy[:,1]
y0[:,2] = dfy[:,2]

nsru = .01
nsry = .01
Ts = 1.
sqrt(sumabs2(u0.',2)/N*nsru)
r = r0.'
u = u0.' + sqrt(sumabs2(u0.',2)/N*nsru).*randn(2,N)
y = y0.' + sqrt(sumabs2(y0.',2)/N*nsru).*randn(2,N)

z = vcat(y[1:1,:],u)
m = 2
n = 100

data = iddata(z,r)
@time NEBtrace, zₛ = neb(data,n,m)


Θ₀  = [.2, 0.3, 0.4, 0.5, 0.4, 0.5, 0.5, 0.15]
λₛ = last(NEBtrace).λ
βₛ = last(NEBtrace).β
σₛ = last(NEBtrace).σ
sₛ = last(NEBtrace).s
Θ  = last(NEBtrace).Θ

Θᵢ = first(NEBtrace).Θ
σᵢ = first(NEBtrace).σ

# evaluate NEB
gᵢ = impulse(vcat(zeros(1), Θᵢ[1:m]),vcat(ones(1), Θᵢ[m+1:2m]),Ts,N)
gₜ = impulse(vcat(zeros(1), Θ[1:m]),vcat(ones(1), Θ[m+1:2m]),Ts,N)
g₀ = impulse(vcat(zeros(1), Θ₀[1:1]),vcat(ones(1), Θ₀[2:2]),Ts,N)
g2ᵢ = impulse(vcat(zeros(1), Θᵢ[2m+1:3m]),vcat(ones(1), Θᵢ[3m+1:4m]),Ts,N)
g2ₜ = impulse(vcat(zeros(1), Θ[2m+1:3m]),vcat(ones(1), Θ[3m+1:4m]),Ts,N)
g2₀ = impulse(vcat(zeros(1), Θ₀[3:3]),vcat(ones(1), Θ₀[4:4]),Ts,N)

sumabs2(gᵢ-g₀)
sumabs2(gₜ-g₀)

sumabs2(g2ᵢ-g2₀)
sumabs2(g2ₜ-g2₀)

res = Vector{Float64}(length(NEBtrace))
res2 = Vector{Float64}(length(NEBtrace))
for idx in eachindex(NEBtrace)
  Θᵢ = NEBtrace[idx].Θ
  gᵢ = impulse(vcat(zeros(1), Θᵢ[1:m]),vcat(ones(1), Θᵢ[m+1:2m]),Ts,N)
  g2ᵢ = impulse(vcat(zeros(1), Θᵢ[2m+1:3m]),vcat(ones(1), Θᵢ[3m+1:4m]),Ts,N)
  res[idx] = sumabs2(gᵢ-g₀)
  res2[idx] = sumabs2(g2ᵢ-g2₀)
end

using Plots
pyplot()

plot(res)
plot(res2)

orders = [n, m, nᵤ, nᵣ, N]
@time NEBXtrace,z = nebx(y[1:1,:],u,r,y[2:2,:]- r[2:2,:],orders, Ts)

res3 = Vector{Float64}(length(NEBXtrace))
res4 = Vector{Float64}(length(NEBXtrace))
res5 = Vector{Float64}(length(NEBXtrace))
for idx in eachindex(NEBXtrace)
  Θᵢ = NEBXtrace[idx].Θ
  gᵢ = impulse(vcat(zeros(1), Θᵢ[1:m]),vcat(ones(1), Θᵢ[m+1:2m]),Ts,N)
  g2ᵢ = impulse(vcat(zeros(1), Θᵢ[2m+1:3m]),vcat(ones(1), Θᵢ[3m+1:4m]),Ts,N)
  res3[idx] = sumabs2(gᵢ-g₀)
  res4[idx] = sumabs2(g2ᵢ-g2₀)
  res5[idx] = sumabs2(gᵢ-g₀) + sumabs2(g2ᵢ-g2₀)
end

using Plots
pyplot()

plot(vcat(res,res3))
plot(vcat(res2,res4))
plot(vcat(res5))

Θ = NEBXtrace[20].Θ

gₜ = impulse(vcat(zeros(1), Θ[1:m]),vcat(ones(1), Θ[m+1:2m]),Ts,N)
g₀ = impulse(vcat(zeros(1), Θ₀[1:m]),vcat(ones(1), Θ₀[m+1:2m]),Ts,N)
g2ₜ = impulse(vcat(zeros(1), Θ[2m+1:3m]),vcat(ones(1), Θ[3m+1:4m]),Ts,N)
g2₀ = impulse(vcat(zeros(1), Θ₀[2m+1:3m]),vcat(ones(1), Θ₀[3m+1:4m]),Ts,N)

sumabs2(gₜ-g₀)
sumabs2(g2ₜ-g2₀)

fir_m  = 100
orders = [n, m, nᵤ, nᵣ, N]
firmodel = FIR(fir_m*ones(Int,1,nᵣ),ones(Int,1,nᵣ), 1, nᵣ)

options = IdOptions(show_trace=true, iterations = 20, autodiff=true, estimate_initial=false)
oemodel = OE(m*ones(Int,1,nᵤ), m*ones(Int,1,nᵤ), ones(Int,1,nᵤ), 1, nᵤ)

@time xfir, xG, xσ = two_stage(y[1:1,:],u,r,orders, firmodel, oemodel, options)
A,B,F,C,D = IdentificationToolbox._getpolys(oemodel, xG[:])
Θtwo = vcat(coeffs(B[1])[2:1+m], coeffs(F[1])[2:1+m], coeffs(B[2])[2:1+m], coeffs(F[2])[2:1+m])

gₜ = impulse(vcat(zeros(1), Θtwo[1:m]),vcat(ones(1), Θtwo[m+1:2m]),Ts,N)
g₀ = impulse(vcat(zeros(1), Θ₀[1:1]),vcat(ones(1), Θ₀[2:2]),Ts,N)
g2ₜ = impulse(vcat(zeros(1), Θtwo[2m+1:3m]),vcat(ones(1), Θtwo[3m+1:4m]),Ts,N)
g2₀ = impulse(vcat(zeros(1), Θ₀[3:3]),vcat(ones(1), Θ₀[4:4]),Ts,N)

xG[1:m]
Θ₀[1:1]
xG[m+1:2m]
Θ₀[2:2]

sumabs2(gᵢ-g₀)
sumabs2(gₜ-g₀)

sumabs2(g2ᵢ-g2₀)
sumabs2(g2ₜ-g2₀)

function two_stage{T}(y::AbstractMatrix{T},u,r,orders, firmodel, oemodel, options)
  size(y,1) == 1 || throw(DomainError())
  n, m, nᵤ, nᵣ, N = orders
  fir_m = firmodel.orders.nb[1]

  Θ   = _initial_two_stage(y,u,r,orders, firmodel, oemodel, options)
  opt = optimize(x->cost_twostage(y,u,r, x, orders, firmodel, oemodel, options),
        Θ, Newton(), options.OptimizationOptions)

  x = opt.minimizer
  xfir = view(x,1:nₛ*fir_m)
  xG   = view(x,nₛ*fir_m+(1:nᵤ*2m))
  xσ   = view(x,nₛ*fir_m+nᵤ*2m+(1:nᵤ))

  return xfir, xG, xσ
end

function _initial_two_stage{T}(y::AbstractMatrix{T},u,r,orders, firmodel, oemodel, options)
  n, m, nᵤ, nᵣ, N = orders
  nₛ = nᵤ*nᵣ
  fir_m = firmodel.orders.nb[1]

  Θ    = zeros(nₛ*fir_m + nᵤ*2m + nᵤ+1)
  ϴfir = view(Θ,1:nₛ*fir_m)
  ΘG   = view(Θ,nₛ*fir_m+(1:nᵤ*2m))
  ϴσ   = view(Θ,nₛ*fir_m+nᵤ*2m+(1:nᵤ+1))

  # fir
  û  = zeros(T,nᵤ,N)
  for k = 0:nᵤ-1
    zdata     = iddata(u[k+1:k+1,:], r, Ts)
    A,B,F,C,D,info = pem(zdata,firmodel,zeros(T,nᵣ*fir_m),options)
    for j = 0:nᵣ-1
      i = nᵣ*k + j
      ϴfir[i*fir_m+(1:fir_m)] = coeffs(B[j+1])[2:fir_m+1]
    end
    ϴσ[k+1] = info.mse[1]
    û[k+1:k+1,:] += filt(B,F,r)
  end

  # initialization for OE
  options = IdOptions(show_trace=true, iterations = 20, autodiff=true, estimate_initial=false)
  model   = ARX(m,m,ones(Int,nᵤ),1,nᵤ)
  zdata   = iddata(y[1:1,:], û, Ts)
  s       = arx(zdata,model,options)
  for i in 1:nᵤ
    ΘG[(i-1)*m+(1:m)]      = coeffs(s.B[i])[2:m+1]
    ΘG[nᵤ*m+(i-1)*m+(1:m)] = coeffs(s.A[1])[2:m+1]
  end

  # OE model
  oemodel = OE(m*ones(Int,1,nᵤ), m*ones(Int,1,nᵤ), ones(Int,1,nᵤ), 1, nᵤ)
  A,B,F,C,D,info = pem(zdata, oemodel, ΘG[:], options)
  ΘG[:]   = info.opt.minimizer
  ϴσ[end] = info.mse[1]

  return Θ
end

function cost_twostage{T}(y,u,r, x::AbstractVector{T}, orders, firmodel, oemodel, options)
  n, m, nᵤ, nᵣ, N = orders
  nₛ = nᵤ*nᵣ
  fir_m = firmodel.orders.nb[1]

  xfir = view(x,1:nₛ*fir_m)
  xG   = view(x,nₛ*fir_m+(1:nᵤ*2m))
  xσ   = view(x,nₛ*fir_m+nᵤ*2m+(1:nᵤ+1))

  û  = zeros(T,nᵤ,N)
  costsum = zeros(T,1)
  for k = 0:nᵤ-1
    uₖ = u[k+1:k+1,:]
    fdata = iddata(uₖ, r, Ts)
    û[k+1,:] = IdentificationToolbox.predict(fdata, firmodel, xfir[k*nᵣ*fir_m+(1:nᵣ*fir_m)], options)
    costsum[:] += cost(uₖ, û[k+1:k+1,:], N, options)/abs(xσ[k+1])   #ϴσ[k+1]
  end
  zdata = iddata(y[1:1,:], û, Ts)
  ŷ = IdentificationToolbox.predict(zdata, oemodel, xG[:], options)
  costsum[:] += cost(y, ŷ, N, options)/abs(xσ[nᵤ+1])    #ϴσ[nᵤ+1]

  return costsum[1] + log(prod(abs(xσ)))
end


ϴfir  = zeros(T,nₛ*fir_m)

Θ    = zeros(nₛ*fir_m + nᵤ*2m + nᵤ+1)
ϴfir = view(Θ,1:nₛ*fir_m)
ΘG   = view(Θ,nₛ*fir_m+(1:nᵤ*2m))
ϴσ   = view(Θ,nₛ*fir_m+nᵤ*2m+(1:nᵤ+1))

firmodel = FIR(fir_m*ones(Int,1,nᵣ),ones(Int,1,nᵣ), 1, nᵣ)
options = IdOptions(autodiff=true, estimate_initial=false)
û  = zeros(T,nᵤ,N)

for k = 0:nᵤ-1
  zdata     = iddata(u[k+1:k+1,:], r, Ts)
  A,B,F,C,D,info = pem(zdata,firmodel,zeros(T,nᵣ*fir_m),options)
  for j = 0:nᵣ-1
    i = nᵣ*k + j
    ϴfir[i*fir_m+(1:fir_m)] = coeffs(B[j+1])[2:fir_m+1]
  end
  ϴσ[k+1] = info.mse[1]
  û[k+1:k+1,:] += filt(B,F,r)
end

# initialization for OE
options = IdOptions(show_trace=true, iterations = 20, autodiff=true, estimate_initial=false)
model   = ARX(m,m,ones(Int,nᵤ),1,nᵤ)
zdata   = iddata(y[1:1,:], û, Ts)
s       = arx(zdata,model,options)
for i in 1:nᵤ
  ΘG[(i-1)*m+(1:m)]      = coeffs(s.B[i])[2:m+1]
  ΘG[nᵤ*m+(i-1)*m+(1:m)] = coeffs(s.A[1])[2:m+1]
end

# OE model
oemodel = OE(m*ones(Int,1,nᵤ), m*ones(Int,1,nᵤ), ones(Int,1,nᵤ), 1, nᵤ)
A,B,F,C,D,info = pem(zdata, oemodel, ΘG[:], options)
ΘG[:]   = info.opt.minimizer
ϴσ[end] = info.mse[1]
info.opt.minimizer

options = IdOptions(g_tol=1e-32,show_trace=true, iterations = 20, autodiff=true, estimate_initial=false)
opt = optimize(x->cost_twostage(y[1:1,:],u,r, x, orders, firmodel, oemodel, options),
      Θ, Newton(), options.OptimizationOptions)

x = opt.minimizer
xfir = view(x,1:nₛ*fir_m)
xG   = view(x,nₛ*fir_m+(1:nᵤ*2m))
xσ   = view(x,nₛ*fir_m+nᵤ*2m+(1:nᵤ))

A,B,F,C,D = IdentificationToolbox._getpolys(oemodel, xG[:])

Θtwo = vcat(coeffs(B[1])[2:1+m], coeffs(F[1])[2:1+m], coeffs(B[2])[2:1+m], coeffs(F[2])[2:1+m])



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
