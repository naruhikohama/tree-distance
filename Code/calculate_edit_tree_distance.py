import numpy as np
import h5py
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import os
from zss import Node, simple_distance
import re
from tqdm.auto import tqdm
import json
import time
import pandas as pd

np.random.seed(42)

def rev_bfs(tree, root_index=0):
    rbfs = list()

    queue = [root_index]
    while (len(queue) > 0):
        current_index = queue.pop(0)
        rbfs.insert(0, current_index)
        for c in tree[current_index]['children']:
            queue.append(c)

    return rbfs

def get_bin(lst, K):
    for i in range(len(lst)):
        if K < lst[i]:
            return lst[i-1]
    if K == lst[-1]:
        return lst[-1]
    else:
        return None

def parse_graphviz_tree_zss(gv_tree, root_index=0, constant_bins=None, **args):
    tree = dict()
    # nodes = re.findall(r'\d+ \[label=\"\S+', gv_tree)
    nodes = re.findall(r'^(.*?)(?=\s+fillcolor)', gv_tree, re.MULTILINE)

    for i in range(len(nodes)):
        
        nodes[i] = nodes[i].split(' [label=')
        nodes[i][0] = int(nodes[i][0])
        
        nodes[i][1] = nodes[i][1].replace(',', '')
        tree[nodes[i][0]] = {'depth': 1, 'value': nodes[i][1], 'children': list()}
    
    children = re.findall(r'\d+ -> \d+', gv_tree)
    for i in range(len(children)):
        
        children[i] = children[i].split(' -> ')
        children[i][0] = int(children[i][0])
        children[i][1] = int(children[i][1])
        tree[children[i][0]]['children'].append(children[i][1])
        tree[children[i][0]]['children'].sort()
    

    starting_index = min(tree.keys())
    stack = [starting_index]
    tree_height = 0
    while len(stack) > 0:
        current_node = stack.pop(0)
        current_depth = tree[current_node]['depth']
        if (tree_height < current_depth):
            tree_height = current_depth
        for i in tree[current_node]['children']:
            tree[i]['depth'] = current_depth + 1
            stack.append(i)

    indexes = rev_bfs(tree)

    zss_list = [None for i in range(len(indexes))]

    for i in indexes:
        try:
            tree[i]['value'] = float(tree[i]['value'])
        except ValueError:
            pass
        else:
            if (constant_bins is not None):
                tree[i]['value'] = 'const_%.2f' % get_bin(constant_bins, tree[i]['value'])
        if (len(tree[i]['children']) > 0):
            zss_list[i] = Node(tree[i]['value'], [zss_list[c] for c in tree[i]['children']])
        else:
            zss_list[i] = Node(tree[i]['value'])
    # print(zss_list[0])
    return zss_list[0]



def calculate_distances(args):
    graphviz_tree_list, pairs = args
    results = []

    for i, j in pairs:
        # t1 = time.time()
        tree1 = parse_graphviz_tree_zss(graphviz_tree_list[i])
        # print(tree1)
        tree2 = parse_graphviz_tree_zss(graphviz_tree_list[j])
        # t2 = time.time()
        # print('Tempo para cálculo {:.2f} segundos'.format(t2 - t1))
        distance = simple_distance(tree1, tree2)
        # print(f'Distancia entre {i} e {j}: {distance}')
        results.append(((i, j), distance))
    
    return results

def write_results_to_hdf5(hdf5_file, results):
    # t1 = time.time()
    for (i, j), distance in results:
        row_start_index = i * (i + 1) // 2
        hdf5_file["distance_matrix"][row_start_index + j] = distance
    # t2 = time.time()
    # print('Tempo para salvar {:.2f} segundos'.format(t2 - t1))

def generate_pairs(n):
    return [(i, j) for i in range(n) for j in range(i + 1)]


def calculate_matrix_distance(file, n_workers=None):
    if n_workers is None:
        n_workers = os.cpu_count() or 4
  
    with open(file, 'r') as f:
        graphviz_tree_list = json.load(f)#[:20]
    print('Arquivo lido')

    base_name = os.path.splitext(os.path.basename(file))[0]
    pattern = r"_FPI_([^_]+)"
    dataset_match = re.search(pattern, base_name)
    base_match = dataset_match.group(1)


    output_dir = os.path.join("../", "Data", "Distancias", "Edit Tree")
    os.makedirs(output_dir, exist_ok=True)
    
    indices_folder = '../Data/Small Sample Index'
    indices_files = [os.path.join(indices_folder, f) for f in os.listdir(indices_folder)
                 if f.endswith('.csv') and base_match in os.path.splitext(os.path.basename(f))[0]]

    for indices_file in indices_files:
        indices = pd.read_csv(indices_file, header=None).iloc[:, 0]
        sample_name = os.path.splitext(os.path.basename(indices_file))[0]
        parts = sample_name.split('_')
        sample_number = parts[2] 

        # print(indices)

        sample_trees = [graphviz_tree_list[i] for i in indices if i < len(graphviz_tree_list)]
        n_trees = len(sample_trees)
        pairs = generate_pairs(n_trees)

        chunk_size = (len(pairs)  + n_workers - 1) // n_workers
        chunks = [pairs[i:i + chunk_size] for i in range(0, len(pairs), chunk_size)]

        tasks = [(sample_trees, chunk) for chunk in chunks]

        # print(n_trees)

        output_hdf5_file = os.path.join(output_dir, f'dist_edit_tree_{base_name}_sample_{sample_number}.hdf5')
        
        with h5py.File(output_hdf5_file, 'w') as hdf5_file:
            total_size = n_trees * (n_trees + 1) // 2
            hdf5_file.create_dataset("distance_matrix", shape=(total_size,), dtype=np.float64)


            print("Inicio cálculo:")
            t1 = time.time()
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = [executor.submit(calculate_distances, task) for task in tasks]
                with tqdm(total=len(futures), desc="Computing distances") as pbar:
                    for future in as_completed(futures):
                        results = future.result()
                        write_results_to_hdf5(hdf5_file, results)
                        pbar.update(1)
            t2 = time.time()
            print(f'Tempo para calcular distâncias edit tree de {base_name} sample {sample_number}: {t2 - t1:.2f} segundos')



def main(folder_path):
    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.json')]
    
    # with tqdm(total=len(files), desc="Computing distances") as pbar:
    for file in files:
        print(f"Calculating distance for file {file}")
        calculate_matrix_distance(file)
            # pbar.update(1)

if __name__ == "__main__":
    print("Distância Edição de Árvore")
    folder_path = '../Data/Graphviz Tree/'
    main(folder_path)