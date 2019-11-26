import unittest
import numpy as np

from GNN import BinarizationGNN 



class TestBinarizationGNN(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)

    def test_defaultparams(self):
        feature_dim = 10
        gnn = BinarizationGNN(feature_dim)
        self.assertEqual(gnn.feature_dim, feature_dim)
        self.assertEqual(gnn.learning_rate, 0.0001)
        self.assertEqual(gnn.eps, 0.001)
        self.assertTrue(all((np.array(gnn.aggregate_weight.shape) == feature_dim).flatten()))
        self.assertEqual(gnn.aggregate_feature.shape[0], feature_dim)
        feature = np.copy(gnn.aggregate_feature)
        feature_defalut = np.zeros(feature_dim)
        feature_defalut[0] = 1.
        self.assertTrue(all((feature == feature_defalut).flatten()))

    def test_aggregate(self):
        feature_dim = 4
        aggregate_weight = np.ones((feature_dim, feature_dim))
        gnn = BinarizationGNN(feature_dim, aggregate_step=1, aggregate_weight=aggregate_weight)
        graph = np.zeros((3, 3), dtype=bool)
        graph[0, 1] = graph[1, 0] = 1
        headout = gnn._aggregate(graph)
        self.assertTrue(all((headout == 2).flatten()))

        feature_dim = 5
        aggregate_weight = np.eye(feature_dim)
        gnn = BinarizationGNN(feature_dim, aggregate_step=2, aggregate_weight=aggregate_weight)
        headout = gnn._aggregate(graph)
        self.assertTrue(all((headout == np.array([2, 0, 0, 0, 0]).flatten())))

        aggregate_weight = - np.ones((feature_dim, feature_dim))
        gnn = BinarizationGNN(feature_dim, aggregate_step=2, aggregate_weight=aggregate_weight)
        headout = gnn._aggregate(graph)
        self.assertTrue(all((headout == 0).flatten()))
    
    def test_optimize(self):
        graphs = np.zeros((1, 10, 10), dtype=bool)
        graphs[0, 1, 0] = graphs[0, 0, 1] = 1
        graphs[0, 1, 5] = graphs[0, 5, 1] = 1
        graphs[0, 4, 7] = graphs[0, 7, 4] = 1
        graphs[0, 8, 1] = graphs[0, 1, 8] = 1
        graphs[0, 2, 7] = graphs[0, 7, 2] = 1
        graphs[0, 3, 4] = graphs[0, 4, 3] = 1
        labels = [1]
        gnn = BinarizationGNN(optimizer='SGD')
        init_loss = gnn._loss_one(graphs[0], labels[0])
        for _ in range(3000):
            gnn._optimize(graphs, labels)
        after_loss = gnn._loss_one(graphs[0], labels[0])
        self.assertTrue(init_loss > after_loss)

        gnn = BinarizationGNN(optimizer='momentum')
        init_loss = gnn._loss_one(graphs[0], labels[0])
        for _ in range(3000):
            gnn._optimize(graphs, labels)
        after_loss = gnn._loss_one(graphs[0], labels[0])
        self.assertTrue(init_loss > after_loss)


if __name__ == '__main__':
    unittest.main()
