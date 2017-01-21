module NetworkEmpiricalBayes

# Import functions for overloading
# import function1

# Export only the useful functions
# export function2

# Polynomials package is needed
using ControlCore
using IdentificationToolbox, Compat
using GeneralizedSchurAlgorithm
using Polynomials
using Distributions
using PDMats
using Optim
import ToeplitzMatrices.Toeplitz

export  neb,
        nebx,
        basicEB

function Toeplitz{T<:Number}(g::Vector{T}, N::Int)
  col = zeros(T,1,N)
  col[1] = g[1]
  Toeplitz(reshape(g,length(g),1), col)
end

typealias IdDataObject IdentificationToolbox.IdDataObject

# include files
include("basicEB.jl")
include("basicQmin.jl")
include("impulse.jl")
include("neb.jl")
include("nebx.jl")

end # module
