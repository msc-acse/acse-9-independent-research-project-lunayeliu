from scipy.special import roots_legendre
import numpy as np

def _cached_roots_legendre(n):
    """
    Cache roots_legendre results to speed up calls of the fixed_quad
    function.
    """
    if n in _cached_roots_legendre.cache:
        return _cached_roots_legendre.cache[n]

    _cached_roots_legendre.cache[n] = roots_legendre(n)
    return _cached_roots_legendre.cache[n]

_cached_roots_legendre.cache = dict()


class TensorQuad(object):
    """Class to do the multiple integrations in a tensor way

    Input
    func: integrand
    ranges: integraiton limit
    n: number of integraiton points
    w: corrsponding weights

    """
    def __init__(self, func, ranges, n, w):
        self.func = func
        self.ranges = ranges
        self.n = n
        self.w = w
        self.x_list = [] # integration points holder
        self.w_list = [] # corrsponding weigts holder
        self.val_matrix = None
        self.maxdepth = len(ranges)

    def tensor_constructor(self, data, new_vec):
        """Constrcut a tensor-like data structure
        Increase one dimension for the target tensor using a vector
        --------------------------------------------
        data: target tensor
        new_vec: vector data used to increase dimension
        """
        # Use a identity vector to make a duplicate on the new dimension
        dim_lifter = np.ones([len(new_vec)])
        res = np.tensordot(dim_lifter, data, axes=0)

        # Multply the value to calculate the magnitude
        for i,val in enumerate(new_vec):
            res[i] = res[i] * val
        return res


#     def tensor_breaker(self, tensor, *args, depth):
#         """Calculate the value tensor for Gauss Quadrature
#         --------------------------------------------------
#         tensor: a copy of the weight tensor
#         *args: the value for variables (..x1, x2, x3..xn)
#         depth: current layer of the quadrature
#         """
#         for i,val in enumerate(tensor):
#             if depth == 2:
#                 # Start evaluating using values of all variables
#                 size = tensor.shape
#                 #print('final tensor size', size)
#                 for k in range(size[0]):
#                     for j in range(size[1]):
#                         #print('final args', args)
#                         tensor[k, j] = func(self.x_list[0][k], self.x_list[1][j], *args)
#                 return tensor
#             else:
#                 # Get the value of the variable in current layer
#                 #print('depth',depth,' input args', args)
#                 tensor[i] = self.tensor_breaker(tensor[i], self.x_list[depth-1][i], *args, depth=depth-1)
#         return tensor


    def tensor_breaker_vec(self, tensor, *args, depth):
        """Calculate the value tensor for Gauss Quadrature
        --------------------------------------------------
        tensor: a copy of the weight tensor
        *args: the value for variables (..x1, x2, x3..xn)
        depth: current layer of the quadrature
        """
        for i,val in enumerate(tensor):
            if depth == 2:
                # Start evaluating using values of all variables
                size = tensor.shape
                for k in range(size[0]):
                    for j in range(size[1]):
                        if self.val_matrix is not None:
                            self.val_matrix += tensor[k, j] * self.func(self.x_list[0][k], self.x_list[1][j], *args)
                        else:
                            # initialize the value matrix
                            self.val_matrix = tensor[k, j] * self.func(self.x_list[0][k], self.x_list[1][j], *args)
                return
            else:
                # Get the value of the variable in current layer
                self.tensor_breaker_vec(tensor[i], self.x_list[depth-1][i], *args, depth=depth-1)

    def integrate(self, *args, **kwargs):
        """Gauss Quadrature using a tensor way

        Scalar way:
        The mutiple summentions are implemented using
         Weights_Tensor * Value_Tensor after value tensor is calculated,
         do multplication in corrsponding position of each tensor

        Vector way:
        The mutiple summentions are implemented step by step inside the value matrix
        calculation process
        """
        # Calculate the jacobian for the integration limit transformation
        jac = [(rng[1]-rng[0])/2.0 for rng in self.ranges]
        # The product of all Jacobian determinant,
        # could be multplied at the end of all summentions
        jac_prod = np.prod(np.array(jac))

        # Sample and get the integration points and weights
        if type(self.n) == int:
            x, w = _cached_roots_legendre(self.n)
            y = []
            for i,jac_val in enumerate(jac):
                y.append(jac_val*(x+1) + self.ranges[i][0])

            self.x_list = [y[i] for i in range(self.maxdepth)]
            self.w_list = [w for i in range(self.maxdepth)]
        else:
            assert (len(self.n) == self.maxdepth)
            for num in self.n:
                x, w = _cached_roots_legendre(num)
                for i,jac_val in enumerate(jac):
                    y = jac_val*(x+1) + self.ranges[i][0]
                self.x_list.append(y)
                self.w_list.append(w)

        #Calculate the weight tensor
        w = self.w_list[0]
        for w2 in self.w_list[1:]:
            w = self.tensor_constructor(w, w2)
        print("weights tensor shape: ", w.shape)

        #Calculate the value tensor
#         w_copy = w.copy()
#         if not self.vec:
#             val = self.tensor_breaker(w ,depth=self.maxdepth)
#             return jac_prod * np.sum(val * w_copy)
#         else:
#             self.tensor_breaker_vec(w ,depth=self.maxdepth)
#             return jac_prod * self.val_matrix
        #Calculate the value tensor
        self.tensor_breaker_vec(w ,depth=self.maxdepth)
        return jac_prod * self.val_matrix
