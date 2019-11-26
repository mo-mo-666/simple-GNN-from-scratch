import numpy as np



class BinarizationGNN:
    """
    Graph Newral Network to classify graphs into two.

    parameters
    ----------
    feature_dim : int, optional (default = 8)
        Dimension of the feature vectors to each graph node.

    learning_rate : float, optional (default = 0.001)
        Learning rate.

    eps : float, optinal (default = 0.001)
        Used when calculating numerical gradient.

    optimizer : 'SGD' | 'momentun' , optinal (default = 'momentum')
        Optimizing algorithm. Possible values:

        - 'SGD'
            Stochastic Gradient Descent.
        - 'momentum'
            Momentum SGD.
    
    momentum : float, optional (default = 0.9)
        Used when optimizer == 'momentum'.

    batch_size : int, optional (default = 10)
        Batch size.
    
    epoch : int, optional (defalult = 10)
        Epoch.
    
    aggregate_step : int, optional (default = 2)
        Aggregation step T.
    
    aggregate_feature : np.ndarray(feature_dim,) or None (default = None)
        Initial feature vector when aggregating.
        If None, default is np.array([1, 0, 0, 0, ....]).

    aggregate_weight : np.ndarray(feature_dim, feature_dim) or None (default = None)
        Initial weight W in aggregation.
        If None, default is created by using aggregate_weight_param.

    aggregate_weight_param : dict (key:: 'mu', 'sigma') (default = {'mu': 0, 'sigma': 0.4})
        This parameter is used when aggregate_weight is None.
        Initial weight W is initialized with a normal distribution with mean 'mu' and standard deviation 'sigma'. 

    aggregate_activate_func : 'sigmoid' | 'relu' | 'swish' (default = 'relu')
        An Activation function when aggregating.

    feature_vect_each_weight : np.ndarray(feature_dim) or None (default = None)
        Initial weight A when calculating the weighted sum of feature vectors.
    
    feature_vect_add_weight : float (default = 0)
        Initial weight b when calculating the score of feature vectors.
    """

    def __init__(self,
                 feature_dim: int=8,
                 learning_rate: float=0.0001,
                 eps: float=0.001,
                 optimizer :str='momentum',
                 momentum: float=0.9,
                 batch_size: int=10,
                 epoch: int=10,
                 aggregate_step: int=2,
                 aggregate_feature: np.ndarray=None,
                 aggregate_weight: np.ndarray=None,
                 aggregate_weight_param: dict={'mu': 0, 'sigma': 0.4},
                 aggregate_activate_func: str='relu',
                 feature_vect_each_weight: np.ndarray=None,
                 feature_vect_each_weight_param: dict={'mu': 0, 'sigma': 0.4},
                 feature_vect_add_weight: float=0):

        self.feature_dim = feature_dim
        self.learning_rate = learning_rate
        self.eps = eps
        self.optimizer = optimizer
        self.momentum = momentum
        self.batch_size = batch_size
        self.epoch = epoch
        self.aggregate_step = aggregate_step

        if aggregate_feature is not None:
            self.aggregate_feature = np.copy(aggregate_feature)
        else:
            self.aggregate_feature = np.zeros(feature_dim, dtype=np.float32)
            self.aggregate_feature[0] = 1.

        self.aggregate_weight_param = aggregate_weight_param
        if aggregate_weight is not None:
            self.aggregate_weight = np.copy(aggregate_weight)
        else:
            self.aggregate_weight = (np.random.randn(self.feature_dim, self.feature_dim) * self.aggregate_weight_param['sigma'] + self.aggregate_weight_param['mu']).astype(np.float32)
        self.aggregate_activate_func = aggregate_activate_func
        self.feature_vect_each_weight_param = feature_vect_each_weight_param
        if feature_vect_add_weight:
            self.feature_vect_each_weight = np.copy(feature_vect_each_weight)
        else:
            self.feature_vect_each_weight = (np.random.randn(self.feature_dim) * self.feature_vect_each_weight_param['sigma'] + self.feature_vect_each_weight_param['mu']).astype(np.float32)

        self.feature_vect_add_weight = feature_vect_add_weight

        self.learning_params = [self.aggregate_weight, self.feature_vect_each_weight, self.feature_vect_add_weight]

        self.aggregate_weight_d = 0
        self.feature_vect_each_weight_d = 0
        self.feature_vect_add_weight_d = 0
        self.learning_params_d = [self.aggregate_weight_d, self.feature_vect_each_weight_d, self.feature_vect_add_weight_d]


    def _aggregate(self, graph: np.ndarray, learning_params: list=None) -> np.ndarray:
        """
        Aggregate some steps and return head out.

        parameters
        ----------
        graph : np.ndarray
            Single graph.

        learning_params : list, optional (default = None)
            Learing parameters.
            If None, defalut is self.learning_params.
        
        returns
        ----------
        head_out : np.ndarray(feature_dim)
            The head out of the aggregation.
        """
        graph = np.copy(graph).astype(np.float32)
        if not learning_params:
            learning_params = self.learning_params
        weight = np.copy(learning_params[0])
        
        if self.aggregate_activate_func == 'sigmoid':
            f = lambda X: (np.tanh(X / 2.) + 1.) / 2. 
        if self.aggregate_activate_func == 'relu':
            f = lambda X: np.maximum(X, 0)
        if self.aggregate_activate_func == 'swish':
            f = lambda X: X * (np.tanh(X / 2.) + 1.) / 2. 

        n = graph.shape[0] 
        X = np.copy(self.aggregate_feature)
        X = np.tile(X, (n, 1))
        for _ in range(self.aggregate_step):
            A = np.dot(graph, X)
            X = np.dot(A, weight)
            X = f(X)
        # HEADOUT
        head_out = np.sum(X, axis=0) 
        return head_out
    

    def _rawscore_one(self, graph: np.ndarray, learning_params: list=None) -> float:
        """
        Calcurate score after aggregating step.

        parameters
        ----------
        graph : np.ndarray
            Single graph.
        
        learning_params : list, optional (default = None)
            Learing parameters.
            If None, defalut is self.learning_params.
        
        returns
        ----------
        s : float
            The value of the score.
        """
        if not learning_params:
            learning_params = self.learning_params
        h = self._aggregate(graph, learning_params)
        feature_vect_each_weight, feature_vect_add_weight = learning_params[1:]
        s = np.dot(feature_vect_each_weight, h) + feature_vect_add_weight
        return s


    def _predict_one(self, graph: np.ndarray) -> bool:
        """
        Predict a label of a single graph.

        parameters
        ----------
        graph : np.ndarray
            Single graph.
        
        returns
        ----------
        s > 0 : bool
            The predicted label.
        """
        s = self._rawscore_one(graph)
        return s > 0


    def _loss_one(self, graph: np.ndarray, label: bool, learning_params: list=None) -> float:
        """
        Calculate the loss score of a single graph.

        parameters
        ----------
        graph : np.ndarray
            Single graph.

        label : bool
            The correct answer label of the graph.

        learning_params : list, optional (default = None)
            Learing parameters.
            If None, defalut is self.learning_params.

        returns
        ----------
        loss : float
            The value of the loss.
        """
        label = float(label)
        s = self._rawscore_one(graph, learning_params)
        if -100 < s < 100:
            loss = label * np.log(1 + np.exp(-s)) + (1 - label) * np.log(1 + np.exp(s))
        elif s < 0:
            loss = label * s + (1 - label) * np.log(1 + np.exp(s))
        else:
            loss = label * np.log(1 + np.exp(-s)) + (1 - label) * s
        return loss


    def loss(self, graphs: np.ndarray, labels: list) -> float:
        """
        Calculate the avarage loss score of a single graph.

        parameters
        ----------
        graphs : np.ndarray
            Correction of graphs.

        labels : list
            The list of correct answer labels of the graphs.

        learning_params : list, optional (default = None)
            Learing parameters.
            If None, defalut is self.learning_params.

        returns
        ----------
        loss : float
            The value of the loss.
        """
        loss = 0
        n = len(labels)
        for graph, label in zip(graphs, labels):
            loss += self._loss_one(graph, label) / n
        return loss


    def _gradient_one(self, graph: np.ndarray, label: bool) -> list:
        """
        Calculate the gradient of the loss score of a single graph.

        parameters
        ----------
        graph : np.ndarray
            Single graph.

        label : bool
            The correct answer label of the graph.

        learning_params : list, optional (default = None)
            Learing parameters.
            If None, defalut is self.learning_params.

        returns
        ----------
        g_aggregate_weight : np.ndarray
            Gradient of the aggregate_weight parameters.

        g_feature_vect_each_weight : np.ndarray
            Gradient of the feature_vect_each_weight parameters.

        g_feature_vect_add_weight : float
            Gradient of the feature_vect_add_weight parameters.
        """
        loss = self._loss_one(graph, label)
        aggregate_weight, feature_vect_each_weight, feature_vect_add_weight = self.learning_params

        d1, d2 = aggregate_weight.shape
        g_aggregate_weight = np.zeros_like(aggregate_weight)
        for i in range(d1):
            for j in range(d2):
                plus = np.copy(aggregate_weight)
                plus[i, j] += self.eps
                learning_params = [plus, feature_vect_each_weight, feature_vect_add_weight]
                lossplus = self._loss_one(graph, label, learning_params)
                diff = (lossplus - loss) / self.eps
                g_aggregate_weight[i, j] = diff

        d, = feature_vect_each_weight.shape
        g_feature_vect_each_weight = np.zeros_like(feature_vect_each_weight)
        for i in range(d):
            plus = np.copy(feature_vect_each_weight)
            plus[i] += self.eps
            learning_params = [aggregate_weight, plus, feature_vect_add_weight]
            lossplus = self._loss_one(graph, label, learning_params)
            diff = (lossplus - loss) / self.eps
            g_feature_vect_each_weight[i] = diff

        plus = feature_vect_add_weight + self.eps
        learning_params = [aggregate_weight, feature_vect_each_weight, plus]
        lossplus = self._loss_one(graph, label, learning_params)
        diff = (lossplus - loss) / self.eps
        g_feature_vect_add_weight = diff

        return g_aggregate_weight, g_feature_vect_each_weight, g_feature_vect_add_weight


    def _optimize(self, graphs: np.ndarray, labels: list):
        """
        Calculate the gradient of the loss score of a single graph and optimize the learning parameters.

        parameters
        ----------
        graphs : np.ndarray
            Correction of graphs.

        labels : list
            The list of correct answer labels of the graphs.
        """
        n = graphs.shape[0]
        delta_aggregate = 0
        delta_feature_vect_each = 0
        delta_feature_vect_add = 0
        for graph, label in zip(graphs, labels):
            g_aggregate_weight, g_feature_vect_each_weight, g_feature_vect_add_weight = self._gradient_one(graph, label)
            delta_aggregate += g_aggregate_weight / n
            delta_feature_vect_each += g_feature_vect_each_weight / n
            delta_feature_vect_add += g_feature_vect_add_weight / n
        
        if self.optimizer == 'SGD':
            self.aggregate_weight_d = -self.learning_rate * delta_aggregate
            self.feature_vect_each_weight_d = -self.learning_rate * delta_feature_vect_each
            self.feature_vect_add_weight_d = -self.learning_rate * delta_feature_vect_add
            self.aggregate_weight += self.aggregate_weight_d
            self.feature_vect_each_weight += self.feature_vect_each_weight_d
            self.feature_vect_add_weight += self.feature_vect_add_weight_d

        if self.optimizer == 'momentum':
            self.aggregate_weight_d = -self.learning_rate * delta_aggregate + self.momentum * self.aggregate_weight_d
            self.feature_vect_each_weight_d = -self.learning_rate * delta_feature_vect_each + self.momentum * self.feature_vect_each_weight_d
            self.feature_vect_add_weight_d = -self.learning_rate * delta_feature_vect_add + self.momentum * self.feature_vect_add_weight_d
            self.aggregate_weight += self.aggregate_weight_d
            self.feature_vect_each_weight += self.feature_vect_each_weight_d
            self.feature_vect_add_weight += self.feature_vect_add_weight_d

        self.learning_params = [self.aggregate_weight, self.feature_vect_each_weight, self.feature_vect_add_weight]
        self.learning_params_d = [self.aggregate_weight_d, self.feature_vect_each_weight_d, self.feature_vect_add_weight_d]


    def fit(self, graphs: np.ndarray, labels: list):
        """
        Fit the data.
        IF YOU CALL THIS METHOD TWO OR MORE TIMES, YOU CAN FIT ADDITIONAL EPOCH.

        parameters
        ----------
        graphs : np.ndarray
            Correction of graphs.

        labels : list
            The list of correct answer labels of the graphs.
        """
        num = graphs.shape[0]
        for _ in range(self.epoch):
            shuffle_idx = np.random.permutation(np.arange(num))
            shuffle_graphs = graphs[shuffle_idx]
            shuffle_labels = np.array(labels)[shuffle_idx].tolist()
            for i in range(num // self.batch_size):
                batch_graphs = shuffle_graphs[i*self.batch_size:(i+1)*self.batch_size]
                batch_labels = shuffle_labels[i*self.batch_size:(i+1)*self.batch_size]
                self._optimize(batch_graphs, batch_labels)


    def predict(self, graphs: np.ndarray) -> list:
        """
        Predict the labels of the given graphs.

        parameters
        ----------
        graphs : np.ndarray
            Correction of graphs.
        
        returns
        ----------
        labels : list
            List of the labels.
        """
        labels = list()
        for graph in graphs:
            labels.append(self._predict_one(graph))
        return labels


    def predict_prob(self, graphs: np.ndarray, labels: list) -> float:
        """
        Accuracy of the predict.

        parameters
        ----------
        graphs : np.ndarray
            Correction of graphs.
        
        labels : list
            The list of correct answer labels of the graphs.
        
        returns
        ----------
        prob : float
            Correct answer rate.
        """
        predict_labels = self.predict(graphs)
        n = len(labels)
        prob = sum([l == p for l, p in zip(labels, predict_labels)]) / n
        return prob
