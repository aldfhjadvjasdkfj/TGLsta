import dgl
import numpy as np
import scipy.sparse as sparse
import sklearn.preprocessing as preprocessing
import torch
import torch.nn.functional as F
from scipy.sparse import linalg

def batcher():  
    def batcher_dev(batch): 
        graph_q, graph_k = zip(*batch)
        return graph_q, graph_k
    return batcher_dev

def copy_graph(graph):
    graph_copy = dgl.DGLGraph()
    all_edges = graph.all_edges()
    src, dst = all_edges
    graph_copy.add_nodes(max(src.max(), dst.max()) + 1)
    graph_copy.add_edges(src, dst)

    return graph_copy

def _rwr_trace_to_dgl_graph(
    g, seed, trace, positional_embedding_size, entire_graph=False
):
    subv = torch.unique(torch.cat(trace)).tolist()  
    try:
        subv.remove(seed)
    except ValueError:
        pass
    subv = [seed] + subv
    if entire_graph:    
        subg = g.subgraph(g.nodes())
    else:
        subg = g.subgraph(subv)
    subg = copy_graph(subg)

    subg = _add_undirected_graph_positional_embedding(subg, positional_embedding_size)  

    subg.ndata["seed"] = torch.zeros(subg.number_of_nodes(), dtype=torch.long)
    if entire_graph:
        subg.ndata["seed"][seed] = 1
    else:
        subg.ndata["seed"][0] = 1
    return subg

def eigen_decomposision(n, k, laplacian, hidden_size, retry):
    if k <= 0:
        return torch.zeros(n, hidden_size)
    laplacian = laplacian.astype("float64")
    ncv = min(n, max(2 * k + 1, 20))
    v0 = np.random.rand(n).astype("float64")
    for i in range(retry):
        try:
            s, u = linalg.eigsh(laplacian, k=k, which="LA", ncv=ncv, v0=v0)
        except sparse.linalg.eigen.arpack.ArpackError:
            ncv = min(ncv * 2, n)
            if i + 1 == retry:
                sparse.save_npz("arpack_error_sparse_matrix.npz", laplacian)
                u = torch.zeros(n, k)
        else:
            break
    x = preprocessing.normalize(u, norm="l2")
    x = torch.from_numpy(x.astype("float32"))
    x = F.pad(x, (0, hidden_size - k), "constant", 0)
    return x

def _add_undirected_graph_positional_embedding(g, hidden_size, retry=10):
    n = g.number_of_nodes()
    adj = g.adjacency_matrix_scipy(transpose=False, return_edge_ids=False).astype(float)   
    norm = sparse.diags(
        dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float
    )
    laplacian = norm * adj * norm
    k = min(n - 2, hidden_size)
    x = eigen_decomposision(n, k, laplacian, hidden_size, retry)
    g.ndata["pos_undirected"] = x.float()
    return g