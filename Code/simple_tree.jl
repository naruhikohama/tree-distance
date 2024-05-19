#!/usr/bin/env julia
# -*- coding: utf-8 -*-
# Author: Tim Henderson
# Email: tim.tadh@gmail.com
# For licensing see the LICENSE file in the top level directory.

mutable struct Node
    label::String
    children::Vector{Node}

    function Node(label::String, children::Vector{Node}=Node[])
        new(label, children)
    end
end

# Funções associadas à estrutura Node

function get_children(node::Node)
    return node.children
end

function get_label(node::Node)
    return node.label
end

function addkid!(node::Node, child::Node; before::Bool=false)
    if before
        insert!(node.children, 1, child)
    else
        push!(node.children, child)
    end
    return node
end

function get(node::Node, label::String)
    if node.label == label
        return node
    end
    for c in node.children
        result = get(c, label)
        if result !== nothing
            return result
        end
    end
    return nothing
end

function iter(node::Node)
    queue = [node]
    return IterTools.flatten((n for n in queue))
end

# Sobrescrita dos operadores e funções de Julia para trabalhar com Node

function Base.in(label::String, node::Node)
    if node.label == label
        return true
    else
        return any(label in c for c in node.children)
    end
end

function Base.in(b::Node, node::Node)
    if node.label == b.label
        return true
    else
        return any(b in c for c in node.children)
    end
end

function Base.:(==)(node::Node, b)
    if b === nothing
        return false
    elseif !(b isa Node)
        throw(error("Must compare against type Node"))
    else
        return node.label == b.label
    end
end

function Base.:(!=)(node::Node, b)
    return !(node == b)
end

function Base.show(io::IO, node::Node)
    print(io, string("Node ", node.label, ">"))
end

function Base.string(node::Node)
    s = "$(length(node.children)):$node.label"
    return join([s; map(string, node.children)], "\n")
end
