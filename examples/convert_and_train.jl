using Pkg
pkg"activate /home/viegorova/.julia/dev/Keras2Flux"

using Flux
using Keras2Flux
using LinearAlgebra

function get_penalty(model::Chain, regs::Array{Any, 1})
    index_model = 1
    index_regs = 1
    penalties = []
    for layer in model
        if layer isa Dense
            println(regs[index_regs](layer.W))   
            penalty(m) = regs[index_regs](m[index_model].W)
            push!(penalties, penalty)
            #println(regs[i])
            index_regs += 1
        end
        index_model += 1
    end
    index_model = 1
    index_regs = 1
    total_penalty(m) = sum([p(m) for p in penalties])
    println(total_penalty)
    println(total_penalty(model))
    return total_penalty
end

model, regs = convert_keras2flux("examples/keras_1_1_0.json")
penalty = get_penalty(model, regs)