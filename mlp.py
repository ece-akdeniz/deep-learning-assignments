import torch

class MLP:
    def __init__(
        self,
        linear_1_in_features,
        linear_1_out_features,
        f_function,
        linear_2_in_features,
        linear_2_out_features,
        g_function
    ):
        """
        Args:
            linear_1_in_features: the in features of first linear layer
            linear_1_out_features: the out features of first linear layer
            linear_2_in_features: the in features of second linear layer
            linear_2_out_features: the out features of second linear layer
            f_function: string for the f function: relu | sigmoid | identity
            g_function: string for the g function: relu | sigmoid | identity
        """
        self.f_function = f_function
        self.g_function = g_function

        self.parameters = dict(
            W1 = torch.randn(linear_1_out_features, linear_1_in_features),
            b1 = torch.randn(linear_1_out_features),
            W2 = torch.randn(linear_2_out_features, linear_2_in_features),
            b2 = torch.randn(linear_2_out_features),
        )
        self.grads = dict(
            dJdW1 = torch.zeros(linear_1_out_features, linear_1_in_features),
            dJdb1 = torch.zeros(linear_1_out_features),
            dJdW2 = torch.zeros(linear_2_out_features, linear_2_in_features),
            dJdb2 = torch.zeros(linear_2_out_features),
        )

        # put all the cache value needed in self.cache
        self.cache = dict()

    def forward(self, x):
        """
        Args:
            x: input tensor of shape (batch_size, linear_1_in_features)
        Return:
            y_hat: output tensor of shape (batch_size, linear_2_out_features)
        """

        # linear layer 1
        s1 = torch.matmul(x, self.parameters['W1'].T) + self.parameters['b1']
        if self.f_function == 'relu':
            z1 = torch.relu(s1)
        elif self.f_function == 'sigmoid':
            z1 = torch.sigmoid(s1)
        else:
            z1 = s1  # identity function

        # linear layer 2
        s2 = torch.matmul(z1, self.parameters['W2'].T) + self.parameters['b2']
        if self.g_function == 'relu':
            y_hat = torch.relu(s2)
        elif self.g_function == 'sigmoid':
            y_hat = torch.sigmoid(s2)
        else:
            y_hat = s2  # identity function

        # save values to cache for use in backward pass
        self.cache['x'] = x
        self.cache['s1'] = s1
        self.cache['z1'] = z1
        self.cache['s2'] = s2

        return y_hat

    def backward(self, dJdy_hat):
        """
        Args:
            dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
        Return:
            dJdx: The gradient tensor of shape (batch_size, linear_1_in_features)
        """

        # get cached values
        x = self.cache['x']
        s1 = self.cache['s1']
        z1 = self.cache['z1']
        s2 = self.cache['s2']

        # compute gradients of loss w.r.t. output of second linear layer
        if self.g_function == 'relu':
            ds2 = dJdy_hat * (s2 > 0).float()
        elif self.g_function == 'sigmoid':
            ds2 = dJdy_hat * torch.sigmoid(s2) * (1 - torch.sigmoid(s2))
        else:
            ds2 = dJdy_hat

        # compute gradients of loss w.r.t. parameters of second linear layer
        self.grads['dJdW2'] = torch.matmul(ds2.T, z1)
        self.grads['dJdb2'] = torch.sum(ds2, dim=0)

        # compute gradients of loss w.r.t. output of first linear layer
        dz1 = torch.matmul(ds2, self.parameters['W2'])
        if self.f_function == 'relu':
            ds1 = dz1 * (s1 > 0).float()
        elif self.f_function == 'sigmoid':
            ds1 = dz1 * torch.sigmoid(s1) * (1 - torch.sigmoid(s1))
        else:
            ds1 = dz1

        # compute gradients of loss w.r.t. parameters of first linear layer
        self.grads['dJdW1'] = torch.matmul(ds1.T, x)
        self.grads['dJdb1'] = torch.sum(ds1, dim=0)

        # return gradients of loss w.r.t. input for use in previous layer
        dJdx = torch.matmul(ds1, self.parameters['W1'])

        return dJdx

    def clear_grad_and_cache(self):
        for grad in self.grads:
            self.grads[grad].zero_()
        self.cache = dict()


def mse_loss(y, y_hat):
    """
    Args:
        y: the label tensor of shape (batch_size, linear_2_out_features)
        y_hat: the prediction tensor of shape (batch_size, linear_2_out_features)

    Return:
        J: scalar loss
        dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
    """

    # calculate the mean squared error loss
    J = torch.mean((y_hat - y) ** 2)

    # calculate the gradient of the loss w.r.t. y_hat
    batch_size, linear_2_out_features = y.shape
    dJdy_hat = 2 * (y_hat - y) / (batch_size * linear_2_out_features)

    return J, dJdy_hat


def bce_loss(y, y_hat):
    """
    Args:
        y: the label tensor of shape (batch_size, linear_2_out_features)
        y_hat: the prediction tensor of shape (batch_size, linear_2_out_features)

    Return:
        J: scalar loss
        dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
    """

    # calculate the binary cross entropy loss
    J = - torch.mean(y * torch.log(y_hat) + (1 - y) * torch.log(1 - y_hat))

    # calculate the gradient of the loss w.r.t. y_hat
    batch_size, linear_2_out_features = y.shape
    dJdy_hat = - (y / y_hat - (1 - y) / (1 - y_hat)) / (batch_size * linear_2_out_features)

    return J, dJdy_hat


def cross_entropy_loss(y, y_hat):
    """
    Args:
        y: the label tensor of shape (batch_size, linear_2_out_features)
        y_hat: the prediction tensor of shape (batch_size, linear_2_out_features)

    Return:
        J: scalar loss
        dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
    """

    # convert y to one-hot encoding
    y_one_hot = torch.zeros_like(y_hat)
    y_one_hot.scatter_(1, y.unsqueeze(1), 1)

    # calculate the cross entropy loss
    J = torch.mean(- torch.sum(y_one_hot * torch.log_softmax(y_hat, dim=1), dim=1))

    # calculate the gradient of the loss w.r.t. y_hat
    batch_size = y_hat.shape[0]
    dJdy_hat = (torch.softmax(y_hat, dim=1) - y_one_hot) / batch_size

    return J, dJdy_hat
