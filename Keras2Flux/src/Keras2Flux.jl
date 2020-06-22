module Keras2Flux

import JSON
using LinearAlgebra
using Flux
using Flux: glorot_normal, zeros, crossentropy 

export convert_keras2flux

function get_activation_function(af_name::String)
    if af_name == "relu"
        activation = relu
    elseif af_name == "sigmoid"
        activation = σ
    elseif af_name == "linear"
        activation = identity
    else
        throw("unimplemented")
    end
    return activation
end

function get_init(init_name::String)
    if init_name == "glorot_normal"
        init = glorot_normal
    else
        throw("unimplemented")
    end
    return init
end

function get_regularization(config::Dict{String,Any})
    if !(config["W_regularizer"] === nothing)
        w_l1_reg = config["W_regularizer"]["l1"]
        w_l2_reg = config["W_regularizer"]["l2"]
    else
        w_l1_reg = nothing
        w_l2_reg = nothing
    end

    if !(config["b_regularizer"] === nothing)
        throw("unimplemented")
    end

    l1_reg(params) = sum(w_l1_reg * norm(params, 1))
    l2_reg(params) = sum(w_l2_reg * norm(params, 2))
    regularization(params) = l1_reg(params) + l2_reg(params)

    return regularization
end

function create_dense(config::Dict{String,Any}, prev_out_dim::Int64=-1)
    if !(config["input_dim"] === nothing)
        in = config["input_dim"]
    else
        in = prev_out_dim
    end
    out = config["output_dim"]
    activation = get_activation_function(config["activation"])
    init = get_init(config["init"])
    #TODO: implement weight constraints 
    if !((config["W_constraint"] === nothing) & (config["b_constraint"] === nothing))
        throw("unimplemented")
    end
    dense = Dense(in, out, activation; initW = init, initb = zeros)
    regularization = get_regularization(config)
    return dense, regularization
end

function create_dropout(config::Dict{String,Any})
    p = config["p"]
    dropout = Dropout(p)
    return dropout
end

function build_model(model_config::Array{Any,1})
    layers = []
    regularizations = []
    prev_out_dim = -1
    for layer_config in model_config
        if layer_config["class_name"] == "Dense"
            layer, regularization = create_dense(layer_config["config"], prev_out_dim)
            prev_out_dim = layer_config["config"]["output_dim"]
            push!(regularizations, regularization)
        elseif layer_config["class_name"] == "Dropout"
            layer = create_dropout(layer_config["config"])
        else
            println(layer_config["class_name"])
            throw("unimplemented")
        end
        push!(layers, layer)
    end
    model = Chain(layers...)
    return model, prev_out_dim, regularizations
end

function convert_keras2flux(filename::String)
    jsontxt = ""
    open(filename, "r") do f
        jsontxt = read(f, String)  
    end
    model_params = JSON.parse(jsontxt)  
    if model_params["keras_version"] == "1.1.0"
        model, out_dim, regularizations = build_model(model_params["config"])
        println(typeof(regularizations[1]))
    else
        throw("unimplemented")
    end
    return model, regularizations
end

end
