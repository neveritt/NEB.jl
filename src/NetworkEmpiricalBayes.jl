module NetworkEmpiricalBayes

# Import functions for overloading
# import function1

# Export only the useful functions
# export function2

# Polynomials package is needed
using SystemsBase
using IdentificationToolbox, Compat
using GeneralizedSchurAlgorithm
using Polynomials
using Distributions
using PDMats
using Optim
import ToeplitzMatrices.Toeplitz

export  neb, NEBstate,
        nebx, NEBXstate,
        basicEB

const IdDataObject = IdentificationToolbox.IdDataObject

# include files
include("utils.jl")
include("basicEB.jl")
include("basicQmin.jl")
include("impulse.jl")
include("neb.jl")
include("nebx.jl")
include("nebs.jl")

end # module
