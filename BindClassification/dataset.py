DTYPE = np.float32
class _Antibody_Antigen_Dataset(Dataset):
    """Class for Antibody and Antigen data"""
    atom_feature_size, num_bonds = 20, 1
    def __init__(self, meta_file_path, mode='train', transform=None):
        """
        Args:
            meta_file_path (string): Path to meta file
        """
        self.meta_file_path = meta_file_path
        self.mode = mode
        self.load_meta()

    def load_meta(self):
        self.meta_df = pd.read_csv(f"{self.meta_file_path}")
        self.meta_df = self.meta_df[['pdb', 'Hchain', 'antigen_chain']]
        self.meta_df.dropna(subset=["Hchain", "antigen_chain"], inplace=True)
        self.meta_df.reset_index(drop=True, inplace=True)
        self.meta_df['Target'] = 1
        
        # split based on mode(train/val/test)
        self.meta_df = self.split(self.meta_df)

        # create more data
        self.meta_df = self.create_data(self.meta_df)

    def split(self, meta_df):
        if self.mode=='train':
            return meta_df[:int(0.6*len(meta_df))].reset_index(drop=True)
        if self.mode=='valid':
            return meta_df[int(0.6*len(meta_df)):int(0.8*len(meta_df))].reset_index(drop=True)
        if self.mode=='test':
            return meta_df[int(0.8*len(meta_df)):].reset_index(drop=True)
        meta_df

    def create_data(self, meta_df):
        length = len(meta_df)
        print(length)
        for i in tqdm(range(length)):
            idx_list = list(range(length))
            idx_list.pop(i)

            idx2 = random.choices(idx_list, k=1)[0]

            tmp_df = meta_df[meta_df.index==idx2].copy()
            tmp_df['pdb'][idx2] = meta_df['pdb'][i]
            tmp_df['Hchain'][idx2] = meta_df['Hchain'][i]
            tmp_df['Target'][idx2] = 0
            meta_df = pd.concat([meta_df, tmp_df], axis=0).reset_index(drop=True)

        meta_df = meta_df.sample(frac=1).reset_index(drop=True)
        return meta_df

    def get_dglGraph(self, pdb_code, chains):
        ## nxgraph
        params = {"edge_construction_functions": [partial(add_k_nn_edges, k=3, long_interaction_threshold=0)],
                "node_metadata_functions": [amino_acid_one_hot]}
        config = ProteinGraphConfig(**params)
        nxGraph = construct_graph(config=config, pdb_code=pdb_code, chain_selection=chains)
        nxGraph = nxGraph.to_directed()
        
        # get nxgraph node feature
        node_f, node_x = [], []
        for n, char in nxGraph.nodes(data=True):
            node_f.append(char["amino_acid_one_hot"].astype(DTYPE))
            node_x.append(np.asarray(char["coords"]).astype(DTYPE))
        node_f = torch.tensor(node_f).view(len(node_f), len(node_f[0]), 1)
        node_x = torch.tensor(node_x).view(len(node_x), -1)

        # get nxgraph edge feature
        edge_w, edge_d = [], []
        bb = nx.edge_betweenness_centrality(nxGraph, normalized=False)
        nx.set_edge_attributes(nxGraph, bb, "betweenness")    
        for src, dst, char in nxGraph.edges(data=True):
            edge_w.append(char["betweenness"])
            edge_d.append((nxGraph.nodes[src]['coords']-nxGraph.nodes[dst]['coords']).astype(DTYPE))
        edge_w = torch.tensor(edge_w).view(len(edge_w), -1)
        edge_d = torch.tensor(edge_d).view(len(edge_d), -1)

        ## dgl graph
        adjacency = nx.adjacency_matrix(nxGraph)
        adjacency = adjacency.toarray()
        src, dst = np.nonzero(adjacency)
        dglGraph = dgl.graph((src, dst))

        # add node features
        dglGraph.ndata['x'] = node_x
        dglGraph.ndata['f'] = node_f

        # add edge features
        dglGraph.edata['d'] = edge_d
        dglGraph.edata['w'] = edge_w

        return dglGraph
        
    def __len__(self):
        return len(self.meta_df)

    def __getitem__(self, idx):
        pdb_code = self.meta_df["pdb"][idx]
        chains_AB, chains_AG = self.meta_df["Hchain"][idx].split(' | '), self.meta_df["antigen_chain"][idx].split(' | ')
        
        return self.get_dglGraph(pdb_code, chains_AB), self.get_dglGraph(pdb_code, chains_AG), np.asarray(self.meta_df['Target'][idx], dtype=DTYPE)