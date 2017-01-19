function impulse(G,n)
  u = zeros(n)
  u[1] = 1.0/G.Ts
  lsim(G, u)
end

function impulse(b,a,Ts,n)
  u = zeros(n)
  u[1] = 1.0/Ts
  lsim(b, a, u)
end

# Discrete SISO Transfer Function
function lsim{T1<:Real}(s, u::Array{T1})
  b = numvec(s)
  a = denvec(s)
  lsim(b,a,u)
end

function lsim{T1<:Real}(b, a, u::Array{T1})
  # zeropad b and a if necessary
  lengthb = length(b)
  lengtha = length(a)
  order = max(lengthb,lengtha)
  b = copy!(zeros(eltype(b),order),order-lengthb+1,b,1)
  a = copy!(zeros(eltype(a),order), 1, a, 1)
  filt(b, a, u)
end
