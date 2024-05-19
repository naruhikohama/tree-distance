using LinearAlgebra
include("simple_tree.jl")

# Define the AnnotatedTree struct
mutable struct AnnotatedTree
    root::Node
    get_children::Function
    nodes::Vector{Node}
    ids::Vector{Int}
    lmds::Vector{Int}
    keyroots::Vector{Int}

    function AnnotatedTree(root::Node, get_children::Function)
        nodes = Node[]
        ids = Int[]
        lmds = Int[]
        stack = [(root, Int[])]
        pstack = Vector{Tuple{Tuple{Node, Int}, Vector{Int}}}()
        j = 0

        while !isempty(stack)
            n, anc = pop!(stack)
            nid = j
            for c in get_children(n)
                a = copy(anc)
                pushfirst!(a, nid)
                push!(stack, (c, a))
            end
            push!(pstack, ((n, nid), anc))
            j += 1
        end

        lmds_dict = Dict{Int, Int}()
        keyroots_dict = Dict{Int, Int}()
        i = 0
        while !isempty(pstack)
            (n, nid), anc = pop!(pstack)
            push!(nodes, n)
            push!(ids, nid)
            if isempty(get_children(n))
                lmd = i
                for a in anc
                    if !haskey(lmds_dict, a)
                        lmds_dict[a] = i
                    else
                        break
                    end
                end
            else
                lmd = lmds_dict[nid]
            end
            push!(lmds, lmd)
            keyroots_dict[lmd] = i
            i += 1
        end

        keyroots = sort(collect(values(keyroots_dict)))
        new(root, get_children, nodes, ids, lmds, keyroots)
    end
end

# Define the Operation struct
struct Operation
    type::Int
    arg1::Union{Node, Nothing}
    arg2::Union{Node, Nothing}
end

const REMOVE = 0
const INSERT = 1
const UPDATE = 2
const MATCH = 3

function Base.show(io::IO, op::Operation)
    if op.type == REMOVE
        print(io, "<Operation Remove>")
    elseif op.type == INSERT
        print(io, "<Operation Insert>")
    elseif op.type == UPDATE
        print(io, "<Operation Update>")
    else
        print(io, "<Operation Match>")
    end
end

Base.:(==)(op1::Operation, op2::Operation) = op1.type == op2.type && op1.arg1 == op2.arg1 && op1.arg2 == op2.arg2

# Define the string distance function
function strdist(a::String, b::String)::Int
    return a == b ? 0 : 1
end

# Define the simple_distance function
function simple_distance(A::Node, B::Node; get_children::Function=get_children, get_label::Function=get_label, label_dist::Function=strdist, return_operations::Bool=false)
    insert_cost(node::Node) = label_dist("", get_label(node))
    remove_cost(node::Node) = label_dist(get_label(node), "")
    update_cost(a::Node, b::Node) = label_dist(get_label(a), get_label(b))

    return tree_edit_distance(A, B, get_children, insert_cost, remove_cost, update_cost; return_operations=return_operations)
end

# Define the tree_edit_distance function
function tree_edit_distance(A::Node, B::Node, get_children::Function, insert_cost::Function, remove_cost::Function, update_cost::Function; return_operations::Bool=false)
    A_tree = AnnotatedTree(A, get_children)
    B_tree = AnnotatedTree(B, get_children)
    size_a = length(A_tree.nodes)
    size_b = length(B_tree.nodes)
    treedists = zeros(Float64, size_a, size_b)
    operations = [Vector{Operation}(undef, size_b) for _ in 1:size_a]

    function treedist(i, j)
        Al = A_tree.lmds
        Bl = B_tree.lmds
        An = A_tree.nodes
        Bn = B_tree.nodes

        if i < 1 || i > length(Al) || j < 1 || j > length(Bl)
            return
        end

        m = i - Al[i] + 2
        n = j - Bl[j] + 2
        fd = zeros(Float64, m, n)
        partial_ops = [Vector{Operation}(undef, n) for _ in 1:m]

        ioff = Al[i] - 1
        joff = Bl[j] - 1

        for x in 2:m
            if x + ioff <= size(An, 1)
                node = An[x + ioff]
                fd[x, 1] = fd[x - 1, 1] + remove_cost(node)
                push!(partial_ops[x, 1], Operation(REMOVE, node, nothing))
            end
        end
        for y in 2:n
            if y + joff <= size(Bn, 1)
                node = Bn[y + joff]
                fd[1, y] = fd[1, y - 1] + insert_cost(node)
                push!(partial_ops[1, y], Operation(INSERT, nothing, node))
            end
        end

        for x in 2:m
            for y in 2:n
                if x + ioff <= size(An, 1) && y + joff <= size(Bn, 1)
                    node1 = An[x + ioff]
                    node2 = Bn[y + joff]
                    if Al[i] == Al[x + ioff] && Bl[j] == Bl[y + joff]
                        costs = [fd[x - 1, y] + remove_cost(node1),
                                 fd[x, y - 1] + insert_cost(node2),
                                 fd[x - 1, y - 1] + update_cost(node1, node2)]
                        fd[x, y] = minimum(costs)
                        min_index = argmin(costs)

                        if min_index == 1
                            op = Operation(REMOVE, node1, nothing)
                            partial_ops[x, y] = vcat(partial_ops[x - 1, y], [op])
                        elseif min_index == 2
                            op = Operation(INSERT, nothing, node2)
                            partial_ops[x, y] = vcat(partial_ops[x, y - 1], [op])
                        else
                            op_type = fd[x, y] == fd[x - 1, y - 1] ? MATCH : UPDATE
                            op = Operation(op_type, node1, node2)
                            partial_ops[x, y] = vcat(partial_ops[x - 1, y - 1], [op])
                        end

                        operations[x + ioff, y + joff] = partial_ops[x, y]
                        treedists[x + ioff, y + joff] = fd[x, y]
                    else
                        p = Al[x + ioff] - 1 - ioff
                        q = Bl[y + joff] - 1 - joff
                        if p >= 0 && q >= 0 && p + 1 <= size(fd, 1) && q + 1 <= size(fd, 2)
                            costs = [fd[x - 1, y] + remove_cost(node1),
                                     fd[x, y - 1] + insert_cost(node2),
                                     fd[p + 1, q + 1] + treedists[x + ioff, y + joff]]
                            fd[x, y] = minimum(costs)
                            min_index = argmin(costs)
                            if min_index == 1
                                op = Operation(REMOVE, node1, nothing)
                                partial_ops[x, y] = vcat(partial_ops[x - 1, y], [op])
                            elseif min_index == 2
                                op = Operation(INSERT, nothing, node2)
                                partial_ops[x, y] = vcat(partial_ops[x, y - 1], [op])
                            else
                                partial_ops[x, y] = vcat(partial_ops[p + 1, q + 1], operations[x + ioff, y + joff])
                            end
                        end
                    end
                end
            end
        end
    end

    for i in A_tree.keyroots
        for j in B_tree.keyroots
            treedist(i, j)
        end
    end

    if return_operations
        return treedists[end, end], operations[end, end]
    else
        return treedists[end, end]
    end
end
