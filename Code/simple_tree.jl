mutable struct Node
    label::String
    children::Vector{Node}

    function Node(label::String, children::Vector{Node}=Node[])
        new(label, children)
    end

    function Node(label::String)
        new(label, Node[])
    end

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
            if label in c
                return get(c, label)
            end
        end
        return nothing
    end

    function iterate(node::Node)
        queue = [node]
        return IterTools.flatten((n for n in queue))
    end
end

function Base.in(b::String, node::Node)
    if node.label == b
        return true
    else
        return any(b in c for c in node.children)
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
        throw(error())
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
