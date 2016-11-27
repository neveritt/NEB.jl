function impulse(G,n)
  u = zeros(n)
  u[1] = 1.0/G.Ts
  lsim(G, u)
end


# Discrete SISO Transfer Function
function lsim{T1<:Real}(s, u::Array{T1})

  # zeropad b and a if necessary
  b = numvec(s)
  a = denvec(s)
  lengthb = length(b)
  lengtha = length(a)
  order = max(lengthb,lengtha)
  b = copy!(zeros(eltype(b),order),order-lengthb+1,b,1)
  a = copy!(zeros(eltype(a),order),order-lengtha+1,a,1)

  filt(b, a, u)
end
