
import copy
import inspect
from torch_geometric.data import Batch

from augment.submix import SubMix

from augment.graphcl import Identity, NodeDrop, Subgraph, EdgePert, AttrMask

from augment.utils import acronym

class Augment(object):
    def __init__(self, graphs, method=None, **kwargs):
        super().__init__()
        self.graphs = copy.deepcopy(graphs)
        if method is None:
            raise RuntimeError("Augmentation w/o type")
        else:
            self.method = method
            model_class = eval(self.method)
            parameters = inspect.signature(model_class).parameters
            args = {k: v for k, v in kwargs.items() if k in parameters}
            self.model = model_class(self.graphs, **args)

    def augment(self, indices):
        if self.method == "SMixup":
            return self.model(indices)
        return [self.model(i) for i in indices]

    def acr(self):
        return acronym(self.method)

    def __str__(self):
        return self.method

    def __call__(self, indices):
        data = self.augment(indices)
        for graph in data:
            # if not hasattr(graph, 'num_nodes'):
            graph.num_nodes = graph.x.size(0)
            # raise RuntimeError(graph)
        data = Batch().from_data_list(data)
        return data
         