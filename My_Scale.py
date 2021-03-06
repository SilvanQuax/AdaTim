import chainer
from chainer.functions.math import scale
from chainer import link
from chainer.links.connection import bias
from chainer import variable
from chainer.functions import exp
import chainer.functions as F

class Alpha(link.Chain):
    """Broadcasted elementwise product with learnable parameters.

    Computes a elementwise product as :func:`~chainer.functions.scale`
    function does except that its second input is a learnable weight parameter
    :math:`W` the link has.

    Args:
        axis (int): The first axis of the first input of
            :func:`~chainer.functions.scale` function along which its second
            input is applied.
        W_shape (tuple of ints): Shape of learnable weight parameter. If
            ``None``, this link does not have learnable weight parameter so an
            explicit weight needs to be given to its ``__call__`` method's
            second input.
        bias_term (bool): Whether to also learn a bias (equivalent to Scale
            link + Bias link).
        bias_shape (tuple of ints): Shape of learnable bias. If ``W_shape`` is
            ``None``, this should be given to determine the shape. Otherwise,
            the bias has the same shape ``W_shape`` with the weight parameter
            and ``bias_shape`` is ignored.

    .. seealso:: See :func:`~chainer.functions.scale` for details.

    Attributes:
        W (~chainer.Parameter): Weight parameter if ``W_shape`` is given.
            Otherwise, no W attribute.
        bias (~chainer.links.Bias): Bias term if ``bias_term`` is ``True``.
            Otherwise, no bias attribute.

    """

    def __init__(self, axis=1, sigmoid_shape_factor=1.5, W_shape=None, bias_term=False, bias_shape=None):
        super(Alpha, self).__init__()
        self.axis = axis

        with self.init_scope():
            # Add parameter that shapes the sigmoid to bound in (0,1)
            self.sigm_shape = sigmoid_shape_factor
            # Add W parameter and/or bias term.
            if W_shape is not None:
                self.W = variable.Parameter(1, W_shape)
                if bias_term:
                    self.bias = bias.Bias(axis, W_shape)
            else:
                if bias_term:
                    if bias_shape is None:
                        raise ValueError(
                            'bias_shape should be given if W is not '
                            'learnt parameter and bias_term is True.')
                    self.bias = bias.Bias(axis, bias_shape)

    def __call__(self, *xs):
        """Applies broadcasted elementwise product.

        Args:
            xs (list of Variables): Input variables whose length should
                be one if the link has a learnable weight parameter, otherwise
                should be two.
        """
        axis = self.axis

        # Case of only one argument where W is a learnt parameter.
        if hasattr(self, 'W'):
            if chainer.is_debug():
                assert len(xs) == 1
            x, = xs
            self.alpha = 1./(1.+exp(-self.sigm_shape*self.W))
            z = scale.scale(x, self.alpha, axis)
        # Case of two arguments where W is given as an argument.
        else:
            if chainer.is_debug():
                assert len(xs) == 2
            x, y = xs
            z = scale.scale(x, y, axis)

        # Forward propagate bias term if given.
        if hasattr(self, 'bias'):
            return self.bias(z)
        else:
            return z
        
    def initialize_alpha(self, new_alphaS):
        assert self.W.shape == new_alphaS.shape
        self.W.data = (1.00/(self.sigm_shape))*self.inverse_sigmoid(new_alphaS).data
        self.alpha = new_alphaS
    
    def inverse_sigmoid(self, x):
        y = -F.log((1./x) -1. )
        return y