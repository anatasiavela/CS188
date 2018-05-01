import numpy as np

import backend
import nn


class Model(object):
    """Base model class for the different applications"""

    def __init__(self):
        self.get_data_and_monitor = None
        self.learning_rate = 0.0

    def run(self, x, y=None):
        raise NotImplementedError("Model.run must be overriden by subclasses")

    def train(self):
        """
        Train the model.

        `get_data_and_monitor` will yield data points one at a time. In between
        yielding data points, it will also monitor performance, draw graphics,
        and assist with automated grading. The model (self) is passed as an
        argument to `get_data_and_monitor`, which allows the monitoring code to
        evaluate the model on examples from the validation set.
        """
        for x, y in self.get_data_and_monitor(self):
            graph = self.run(x, y)
            graph.backprop()
            graph.step(self.learning_rate)


class RegressionModel(Model):
    """
    TODO: Question 4 - [Application] Regression

    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """

    def __init__(self):
        Model.__init__(self)
        self.get_data_and_monitor = backend.get_data_and_monitor_regression

        # Remember to set self.learning_rate!
        # You may use any learning rate that works well for your architecture
        self.learning_rate = 0.09
        self.initialized = False

    def run(self, x, y=None):
        """
        TODO: Question 4 - [Application] Regression

        Runs the model for a batch of examples.

        The correct outputs `y` are known during training, but not at test time.
        If correct outputs `y` are provided, this method must construct and
        return a nn.Graph for computing the training loss. If `y` is None, this
        method must instead return predicted y-values.

        Inputs:
            x: a (batch_size x 1) numpy array
            y: a (batch_size x 1) numpy array, or None
        Output:
            (if y is not None) A nn.Graph instance, where the last added node is
                the loss
            (if y is None) A (batch_size x 1) numpy array of predicted y-values

        Note: DO NOT call backprop() or step() inside this method!

        f(X) = relu(x * W_1 + b_1) * W_2 + b_2
        """
        b = len(x)
        i = x.shape[1]
        h = 36

        if not self.initialized:
            self.W_1 = nn.Variable(i, h)
            self.b_1 = nn.Variable(b, h)
            self.W_2 = nn.Variable(h, i)
            self.b_2 = nn.Variable(b, i)
            self.initialized = True

        g = nn.Graph([self.W_1, self.b_1, self.W_2, self.b_2])

        inX = nn.Input(g, x)
        mul1 = nn.MatrixMultiply(g, inX, self.W_1)
        add1 = nn.MatrixVectorAdd(g, mul1, self.b_1)
        relu = nn.ReLU(g, add1)
        mul2 = nn.MatrixMultiply(g, relu, self.W_2)
        add2 = nn.MatrixVectorAdd(g, mul2, self.b_2)

        if y is not None:
            # At training time, the correct output `y` is known.
            # Here, you should construct a loss node, and return the nn.Graph
            # that the node belongs to. The loss node must be the last node
            # added to the graph.
            inY = nn.Input(g, y)
            loss = nn.SquareLoss(g, add2, inY)
            return g
        else:
            # At test time, the correct output is unknown.
            # You should instead return your model's prediction as a numpy array
            return g.get_output(add2)


class OddRegressionModel(Model):
    """
    TODO: Question 5 - [Application] OddRegression

    A neural network model for approximating a function that maps from real
    numbers to real numbers.

    Unlike RegressionModel, the OddRegressionModel must be structurally
    constrained to represent an odd function, i.e. it must always satisfy the
    property f(x) = -f(-x) at all points during training.
    """

    def __init__(self):
        Model.__init__(self)
        self.get_data_and_monitor = backend.get_data_and_monitor_regression

        # Remember to set self.learning_rate!
        # You may use any learning rate that works well for your architecture
        self.learning_rate = 0.009
        self.initialized = False

    def run(self, x, y=None):
        """
        TODO: Question 5 - [Application] OddRegression

        Runs the model for a batch of examples.

        The correct outputs `y` are known during training, but not at test time.
        If correct outputs `y` are provided, this method must construct and
        return a nn.Graph for computing the training loss. If `y` is None, this
        method must instead return predicted y-values.

        Inputs:
            x: a (batch_size x 1) numpy array
            y: a (batch_size x 1) numpy array, or None
        Output:
            (if y is not None) A nn.Graph instance, where the last added node is
                the loss
            (if y is None) A (batch_size x 1) numpy array of predicted y-values

        Note: DO NOT call backprop() or step() inside this method!
        """
        b = len(x)
        i = x.shape[1]
        h = 50

        if not self.initialized:
            self.W_1 = nn.Variable(i, h)
            self.b_1 = nn.Variable(1, h)
            self.W_2 = nn.Variable(h, i)
            self.b_2 = nn.Variable(1, i)
            self.initialized = True

        g = nn.Graph([self.W_1, self.b_1, self.W_2, self.b_2])

        # positive
        inX = nn.Input(g, x)
        mul1P = nn.MatrixMultiply(g, inX, self.W_1)
        add1P = nn.MatrixVectorAdd(g, mul1P, self.b_1)
        reluP = nn.ReLU(g, add1P)
        mul2P = nn.MatrixMultiply(g, reluP, self.W_2)
        pos = nn.MatrixVectorAdd(g, mul2P, self.b_2)

        # negative
        inN = nn.Input(g, np.matrix([-1.0]))
        inXN = nn.Input(g, -1 * x)
        mul1N = nn.MatrixMultiply(g, inXN, self.W_1)
        add1N = nn.MatrixVectorAdd(g, mul1N, self.b_1)
        reluN = nn.ReLU(g, add1N)
        mul2N = nn.MatrixMultiply(g, reluN, self.W_2)
        add2N = nn.MatrixVectorAdd(g, mul2N, self.b_2)
        neg = nn.MatrixMultiply(g, add2N, inN)

        # combine
        combine = nn.Add(g, pos, neg)

        if y is not None:
            # At training time, the correct output `y` is known.
            # Here, you should construct a loss node, and return the nn.Graph
            # that the node belongs to. The loss node must be the last node
            # added to the graph.
            inY = nn.Input(g, y)
            loss = nn.SquareLoss(g, inY, combine)
            return g
        else:
            # At test time, the correct output is unknown.
            # You should instead return your model's prediction as a numpy array
            return g.get_output(combine)


class DigitClassificationModel(Model):
    """
    TODO: Question 6 - [Application] Digit Classification

    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """

    def __init__(self):
        Model.__init__(self)
        self.get_data_and_monitor = backend.get_data_and_monitor_digit_classification

        # Remember to set self.learning_rate!
        # You may use any learning rate that works well for your architecture
        self.learning_rate = 0.9
        self.initialized = False

    def run(self, x, y=None):
        """
        TODO: Question 6 - [Application] Digit Classification

        Runs the model for a batch of examples.

        The correct labels are known during training, but not at test time.
        When correct labels are available, `y` is a (batch_size x 10) numpy
        array. Each row in the array is a one-hot vector encoding the correct
        class.

        Your model should predict a (batch_size x 10) numpy array of scores,
        where higher scores correspond to greater probability of the image
        belonging to a particular class. You should use `nn.SoftmaxLoss` as your
        training loss.

        Inputs:
            x: a (batch_size x 784) numpy array
            y: a (batch_size x 10) numpy array, or None
        Output:
            (if y is not None) A nn.Graph instance, where the last added node is
                the loss
            (if y is None) A (batch_size x 10) numpy array of scores (aka logits)
        f(X)   = relu(x       * W_1    + b_1)    * W_2   + b_2
        100x10 =     (100x784 * 784x10 + 100x10) * 10x10 + 100x10
        """
        b = len(x) #10
        i = x.shape[1] #784
        h = 200

        if not self.initialized:
            self.W_1 = nn.Variable(i, h)
            self.b_1 = nn.Variable(1, h)
            self.W_2 = nn.Variable(h, 10)
            self.b_2 = nn.Variable(1, 10)
            self.initialized = True

        g = nn.Graph([self.W_1, self.b_1, self.W_2, self.b_2])

        inX = nn.Input(g, x)
        mul1 = nn.MatrixMultiply(g, inX, self.W_1)
        add1 = nn.MatrixVectorAdd(g, mul1, self.b_1)
        relu = nn.ReLU(g, add1)
        mul2 = nn.MatrixMultiply(g, relu, self.W_2)
        add2 = nn.MatrixVectorAdd(g, mul2, self.b_2)
    
        if y is not None:
            inY = nn.Input(g, y)
            loss = nn.SoftmaxLoss(g, add2, inY)
            return g
        else:
            return g.get_output(add2)


class DeepQModel(Model):
    """
    TODO: Question 7 - [Application] Reinforcement Learning

    A model that uses a Deep Q-value Network (DQN) to approximate Q(s,a) as part
    of reinforcement learning.

    (We recommend that you implement the RegressionModel before working on this
    part of the project.)
    """

    def __init__(self):
        Model.__init__(self)
        self.get_data_and_monitor = backend.get_data_and_monitor_rl

        self.num_actions = 2
        self.state_size = 4

        # Remember to set self.learning_rate!
        # You may use any learning rate that works well for your architecture
        self.learning_rate = 0.01
        self.initialized = False

    def run(self, states, Q_target=None):
        """
        TODO: Question 7 - [Application] Reinforcement Learning

        Runs the DQN for a batch of states.

        The DQN takes the state and computes Q-values for all possible actions
        that can be taken. That is, if there are two actions, the network takes
        as input the state s and computes the vector [Q(s, a_1), Q(s, a_2)]

        When Q_target == None, return the matrix of Q-values currently computed
        by the network for the input states.

        When Q_target is passed, it will contain the Q-values which the network
        should be producing for the current states. You must return a nn.Graph
        which computes the training loss between your current Q-value
        predictions and these target values, using nn.SquareLoss.

        Inputs:
            states: a (batch_size x 4) numpy array
            Q_target: a (batch_size x 2) numpy array, or None
        Output:
            (if Q_target is not None) A nn.Graph instance, where the last added
                node is the loss
            (if Q_target is None) A (batch_size x 2) numpy array of Q-value
                scores, for the two actions
        """
        b = len(states) #10
        i = states.shape[1] #784
        h = 330

        if not self.initialized:
            self.W_1 = nn.Variable(i, h)
            self.b_1 = nn.Variable(1, h)
            self.W_2 = nn.Variable(h, 2)
            self.b_2 = nn.Variable(1, 2)
            self.initialized = True

        g = nn.Graph([self.W_1, self.b_1, self.W_2, self.b_2])

        inX = nn.Input(g, states)
        mul1 = nn.MatrixMultiply(g, inX, self.W_1)
        add1 = nn.MatrixVectorAdd(g, mul1, self.b_1)
        relu = nn.ReLU(g, add1)
        mul2 = nn.MatrixMultiply(g, relu, self.W_2)
        add2 = nn.MatrixVectorAdd(g, mul2, self.b_2)

        if Q_target is not None:
            inQ = nn.Input(g, Q_target)
            loss = nn.SquareLoss(g, add2, inQ)
            return g
        else:
            return g.get_output(add2)

    def get_action(self, state, eps):
        """
        Select an action for a single state using epsilon-greedy.

        Inputs:
            state: a (1 x 4) numpy array
            eps: a float, epsilon to use in epsilon greedy
        Output:
            the index of the action to take (either 0 or 1, for 2 actions)
        """
        if np.random.rand() < eps:
            return np.random.choice(self.num_actions)
        else:
            scores = self.run(state)
            return int(np.argmax(scores))


class LanguageIDModel(Model):
    """
    TODO: Question 8 - [Application] Language Identification

    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """

    def __init__(self):
        Model.__init__(self)
        self.get_data_and_monitor = backend.get_data_and_monitor_lang_id

        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Remember to set self.learning_rate!
        # You may use any learning rate that works well for your architecture
        self.learning_rate = 0.01

    def run(self, xs, y=None):
        """
        TODO: Question 8 - [Application] Language Identification

        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        (batch_size x self.num_chars) numpy array, where every row in the array
        is a one-hot vector encoding of a character. For example, if we have a
        batch of 8 three-letter words where the last word is "cat", we will have
        xs[1][7,0] == 1. Here the index 0 reflects the fact that the letter "a"
        is the inital (0th) letter of our combined alphabet for this task.

        The correct labels are known during training, but not at test time.
        When correct labels are available, `y` is a (batch_size x 5) numpy
        array. Each row in the array is a one-hot vector encoding the correct
        class.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node that represents a (batch_size x hidden_size)
        array, for your choice of hidden_size. It should then calculate a
        (batch_size x 5) numpy array of scores, where higher scores correspond
        to greater probability of the word originating from a particular
        language. You should use `nn.SoftmaxLoss` as your training loss.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a (batch_size x self.num_chars) numpy array
            y: a (batch_size x 5) numpy array, or None
        Output:
            (if y is not None) A nn.Graph instance, where the last added node is
                the loss
            (if y is None) A (batch_size x 5) numpy array of scores (aka logits)

        Hint: you may use the batch_size variable in your code
        """

        batch_size = xs[0].shape[0]

        b = batch_size
        i = states.shape[1] #784
        h = 330

        if not self.initialized:
            self.W_1 = nn.Variable(self.num_chars)
            self.initialized = True

        g = nn.Graph([self.W_1])
        self.h = nn.Variable(5, 1)

        for x in xs:
            c = nn.Input(g, x)

        if Q_target is not None:
            inQ = nn.Input(g, Q_target)
            loss = nn.SquareLoss(g, add2, inQ)
            return g
        else:
            return g.get_output(add2)
