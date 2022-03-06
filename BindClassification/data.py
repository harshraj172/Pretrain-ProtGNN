# def get_header(pdb_code):
#     """
#     Gets the header of a PDB file
#     """
#     _, header = parsePDB(pdb_code, header=True)
#     return header

# def chainsAB_AG(header):
#     """
#     Gets the chain of Antibody and Antigens from the headr of the PDB file.
#     Antibody Chain(VHH/VH) - Chain whose name has word "heavy".
#     Antigen Chain - Chains whose name has word "spike".
#     """
#     chains_AB, chains_AG = [], []

#     for key in header.keys():
#         if len(key)==1:
#             s = str(header[key]).lower()
#             print(s)
#             if re.search('spike', s) is not None:
#                 chains_AG.append(key)
#             if re.search('heavy', s) is not None:
#                 chains_AB.append(key)
#     return chains_AB, chains_AG

def edge_func(G: nx.Graph) -> nx.Graph:
    """
    Function for creating the edges in a 3D-Structure.
    TODO - Implement the edge features as in "https://arxiv.org/abs/2110.04624"
    """
    return partial(add_k_nn_edges, k=7, long_interaction_threshold=0)
    
def nxgraph(pdb_code, chains):
    """
    creates a NetworkX Graph.
    """
    params = {"granularity": "centroids", 
              "edge_construction_functions": [edge_func]}
    config = ProteinGraphConfig(**params)
    g = construct_graph(config=config, pdb_code=pdb_code, chain_selection=chains)
    return g

def graphAB_AG(pdb_code, chains_AB, chains_AG):
    """
    Returns a tuple of DGL Graph and the chains for
    Antibody and Antigen respectively.
    """
    nxgraph_AB = nxgraph(pdb_code, chains_AB)
    nxgraph_AG = nxgraph(pdb_code, chains_AG)
    
    dglgraph_AB = dgl.from_networkx(nxgraph_AB)
    dglgraph_AG = dgl.from_networkx(nxgraph_AG)

    return dglgraph_AB, dglgraph_AG
  
def create_data(summary_df):
    """
    returns lists of Antibodies, Antigen and their binary labels.
    To create Label 0s select random Antigen(other than the Binding Antigen) 
    for every Antibody. 
    """
    AntibodyGraph_list, AntigenGraph_list, data_list = [], [], []
    data = namedtuple("data", ("Antibody", "Antigen", "Label"))
    for i in range(len(summary_df)):
        chains_AB, chains_AG = summary_df["Hchain"][i].split(' | '), summary_df["antigen_chain"][i].split(' | ')
        dglgraph_AB, dglgraph_AG = graphAB_AG(summary_df["pdb"][i], chains_AB, chains_AG)
        
        AntibodyGraph_list.append(dglgraph_AB) 
        AntigenGraph_list.append(dglgraph_AG)
        data_list.append(data(Antibody=dglgraph_AB, Antigen=dglgraph_AG, Label=1))

    for i in range(len(AntibodyGraph_list)):
        tmp_AntigenGraph_list = AntigenGraph_list.copy()
        tmp_AntigenGraph_list.pop(i)

        AntigenGraph = random.choices(tmp_AntigenGraph_list, k=1)[0]
        
        data_list.append(data(Antibody=AntibodyGraph_list[i], Antigen=AntigenGraph, Label=0))
        random.shuffle(data_list)

    return data_list
  
def get_dataloaders(summary_df, train_batchsize=4, test_batchsize=16):
    data_list = create_data(summary_df)
    train_data, val_data, test_data = data_list[:int(0.8*len(data_list))], data_list[int(0.8*len(data_list)): int(0.9*len(data_list))], data_list[int(0.9*len(data_list)):] 

    train_loader = DataLoader(train_data, batch_size=train_batchsize, shuffle = False, drop_last = False)
    valid_loader = DataLoader(val_data, batch_size=test_batchsize)
    test_loader = DataLoader(test_data, batch_size=test_batchsize)

    return train_loader, valid_loader, test_loader
