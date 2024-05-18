using Random
using HDF5
using Distributed
using Base.Iterators: flatten
using ProgressMeter
using JSON
using DataFrames
using CSV
using Reexport
using StatsBase
include("zss.jl")

Random.seed!(42)

function rev_bfs(tree, root_index=0)
    rbfs = Int[]
    queue = [root_index]
    while !isempty(queue)
        current_index = popfirst!(queue)
        unshift!(rbfs, current_index)
        append!(queue, tree[current_index]["children"])
    end
    return rbfs
end

function get_bin(lst, K)
    for i in 1:length(lst)
        if K < lst[i]
            return lst[i-1]
        end
    end
    if K == lst[end]
        return lst[end]
    else
        return nothing
    end
end

function parse_graphviz_tree_zss(gv_tree::String; root_index=0, constant_bins=nothing, args...)
    tree = Dict{Int, Dict}()
    nodes = matchall(r"^(.*?)(?=\s+fillcolor)", gv_tree, RegexOpts("m"))
    for i in 1:length(nodes)
        split_node = split(nodes[i], r" \[label=")
        node_index = parse(Int, split_node[1])
        node_value = replace(split_node[2], r"," => "")
        tree[node_index] = Dict("depth" => 1, "value" => node_value, "children" => Int[])
    end

    children = matchall(r"\d+ -> \d+", gv_tree)
    for i in 1:length(children)
        split_child = split(children[i], r" -> ")
        parent = parse(Int, split_child[1])
        child = parse(Int, split_child[2])
        push!(tree[parent]["children"], child)
        sort!(tree[parent]["children"])
    end

    starting_index = minimum(keys(tree))
    stack = [starting_index]
    tree_height = 0
    while !isempty(stack)
        current_node = popfirst!(stack)
        current_depth = tree[current_node]["depth"]
        tree_height = max(tree_height, current_depth)
        for child in tree[current_node]["children"]
            tree[child]["depth"] = current_depth + 1
            push!(stack, child)
        end
    end

    indexes = rev_bfs(tree)
    zss_list = fill(nothing, length(indexes))

    for i in indexes
        try
            tree[i]["value"] = parse(Float64, tree[i]["value"])
        catch e
            if isa(e, ArgumentError)
                # do nothing
            else
                rethrow(e)
            end
        end

        if constant_bins !== nothing
            tree[i]["value"] = "const_$(get_bin(constant_bins, tree[i]["value"]))"
        end

        if !isempty(tree[i]["children"])
            zss_list[i] = Node(tree[i]["value"], [zss_list[c] for c in tree[i]["children"]])
        else
            zss_list[i] = Node(tree[i]["value"])
        end
    end

    return zss_list[1]
end

function calculate_distances(args)
    graphviz_tree_list, pairs = args
    results = []

    for (i, j) in pairs
        tree1 = parse_graphviz_tree_zss(graphviz_tree_list[i])
        tree2 = parse_graphviz_tree_zss(graphviz_tree_list[j])
        distance = simple_distance(tree1, tree2)
        push!(results, ((i, j), distance))
    end

    return results
end

function write_results_to_hdf5(hdf5_file, results)
    for ((i, j), distance) in results
        row_start_index = i * (i + 1) ÷ 2
        hdf5_file["distance_matrix"][row_start_index + j + 1] = distance
    end
end

function generate_pairs(n)
    return [(i, j) for i in 0:(n-1), j in (i+1):n-1]
end

function calculate_matrix_distance(file, n_workers=nothing)
    if n_workers === nothing
        n_workers = Sys.CPU_THREADS
    end

    graphviz_tree_list = open(file) do f
        JSON.parse(f)
    end

    base_name = splitext(basename(file))[1]
    pattern = r"_FPI_([^_]+)"
    dataset_match = match(pattern, base_name)
    base_match = dataset_match.captures[1]

    output_dir = joinpath("..", "Data", "Distancias", "Edit Tree")
    mkpath(output_dir)

    indices_folder = "../Data/Small Sample Index"
    indices_files = filter(f -> endswith(f, ".csv") && occursin(base_match, splitext(basename(f))[1]), readdir(indices_folder))

    for indices_file in indices_files
        indices = CSV.read(joinpath(indices_folder, indices_file), DataFrame, header=false)[!, 1]
        sample_name = splitext(basename(indices_file))[1]
        parts = split(sample_name, '_')
        sample_number = parts[3]

        sample_trees = filter(i -> i < length(graphviz_tree_list), [graphviz_tree_list[i+1] for i in indices])
        n_trees = length(sample_trees)
        pairs = generate_pairs(n_trees)

        chunk_size = ceil(Int, length(pairs) / n_workers)
        chunks = [pairs[i:min(i+chunk_size-1, end)] for i in 1:chunk_size:length(pairs)]

        tasks = [(sample_trees, chunk) for chunk in chunks]

        output_hdf5_file = joinpath(output_dir, "dist_edit_tree_$base_name_sample_$sample_number.h5")

        h5open(output_hdf5_file, "w") do hdf5_file
            total_size = n_trees * (n_trees + 1) ÷ 2
            hdf5_file["distance_matrix"] = zeros(Float64, total_size)

            println("Inicio cálculo:")
            @showprogress for task in tasks
                results = calculate_distances(task)
                write_results_to_hdf5(hdf5_file, results)
            end
        end

        println("Finished calculation for $base_name sample $sample_number")
    end
end

function main(folder_path)
    files = filter(f -> endswith(f, ".json"), readdir(folder_path))
    for file in files
        println("Calculating distance for file $file")
        calculate_matrix_distance(joinpath(folder_path, file))
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    println("Distância Edição de Árvore")
    folder_path = "../Data/Graphviz Tree/"
    main(folder_path)
end
