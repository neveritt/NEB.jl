module NetworkEmpiricalBayes

# Import functions for overloading
# import function1

# Export only the useful functions
# export function2

# Polynomials package is needed
using ControlCore
using IdentificationToolbox, Compat
using GeneralizedSchurAlgorithm
using PDMats
using Optim
import ToeplitzMatrices.Toeplitz

function Toeplitz{T<:Real}(g::Vector{T}, N::Int)
  Toeplitz(hcat(g), hcat(g[1], spzeros(T,1,N-1)))
end

# include files
include("basicEB.jl")
include("basicQmin.jl")
include("impulse.jl")
include("neb.jl")
include("nebx.jl")

end # module
