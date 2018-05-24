from .base_op import Op


class BasePointwiseOp(Op):
    def __init__(self, name):
        super(BasePointwiseOp, self).__init__(name)
        self._flops_per_element = 1

    def propagateShapes(self):
        assert(len(self._outputs) == 1)
        if len(self._inputs) == 1:
            # Unary operator!
            final_shape = self._inputs[0].shape
        else:
            # Must be binary operator!
            assert(len(self._inputs) == 2)
            # Verify inputs have compatible shape
            in_a_shape = self._inputs[0].shape
            in_b_shape = self._inputs[1].shape
            assert in_a_shape.canBroadcastTogether(in_b_shape), \
                'Op: {}: input shapes cannot broadcast ({}, {})!'.format(
                self._name, in_a_shape, in_b_shape)
            final_shape = in_a_shape.getBroadcastShape(in_b_shape)

        # Set the output dimensions
        self._outputs[0].shape.mergeShape(final_shape)

    def flopsPointwise(self):
        assert(len(self._outputs) == 1)
        out_shape = self._outputs[0].shape
        return out_shape.numElements() * self._flops_per_element

    def calcAlgFlops(self):
        return self.flopsPointwise()


class AddOp(BasePointwiseOp):
    def __init__(self, name):
        super(AddOp, self).__init__(name)


class DivOp(BasePointwiseOp):
    def __init__(self, name):
        super(DivOp, self).__init__(name)


class LogicalNotOp(BasePointwiseOp):
    def __init__(self, name):
        super(LogicalNotOp, self).__init__(name)


class MaximumOp(BasePointwiseOp):
    def __init__(self, name):
        super(MaximumOp, self).__init__(name)


class MulOp(BasePointwiseOp):
    def __init__(self, name):
        super(MulOp, self).__init__(name)


class PowOp(BasePointwiseOp):
    def __init__(self, name):
        super(PowOp, self).__init__(name)


class ExpOp(PowOp):
    def __init__(self, name):
        super(ExpOp, self).__init__(name)


class ReluOp(BasePointwiseOp):
    def __init__(self, name):
        super(ReluOp, self).__init__(name)


class RsqrtOp(BasePointwiseOp):
    def __init__(self, name):
        super(RsqrtOp, self).__init__(name)
        # Assume reciprocal square root is 2 Flops per element
        self._flops_per_element = 2


class SigmoidOp(BasePointwiseOp):
    def __init__(self, name):
        super(SigmoidOp, self).__init__(name)
        # For now, assume sigmoid consists of input negation, exponentiation,
        # addition, and then inversion (i.e., 4 Flops per element)
        self._flops_per_element = 4


class SubOp(BasePointwiseOp):
    def __init__(self, name):
        super(SubOp, self).__init__(name)


class TanhOp(BasePointwiseOp):
    def __init__(self, name):
        super(TanhOp, self).__init__(name)
        # For now, assume tanh consists of input negation, two
        # exponentiations, addition and subtraction, and then inversion
        # (i.e., 6 Flops per element)
        self._flops_per_element = 6


class Conv2DOp(Op):
    def __init__(self, name):
        super(Conv2DOp, self).__init__(name)

    def calcAlgFlops(self):
        raise NotImplementedError('Conv2D alg Flops not implemented yet!')


class MatMulOp(Op):
    def __init__(self, name):
        super(MatMulOp, self).__init__(name)

    def propagateShapes(self):
        # [_] TODO (Joel): Will need to handle transposes here...
        # Verify that shapes can be correctly resolved
        assert(len(self._inputs) == 2)
        tensor_a = self._inputs[0]
        tensor_b = self._inputs[1]
        inner_dim = tensor_a.shape.getDim(1)
        # [_] TODO (Joel): This assert will be too strict at some point
        import sympy
        assert (type(inner_dim) == sympy.Symbol or \
                type(self._inputs[1].shape.getDim(0)) == sympy.Symbol or \
                inner_dim == self._inputs[1].shape.getDim(0)), \
               'Dimension check failed in op {} with inputs {} and {}, '\
               'and output {}'.format(self._name, self._inputs[0].shape,
                                      self._inputs[1].shape,
                                      self._outputs[0].shape)
        if inner_dim != tensor_b.shape.getDim(0):
            print('WARN: MatMulOp: Enough info to resolve dimensions!')
        # Resolve shapes
        tensor_c = self._outputs[0]
        first_dim = tensor_a.shape.getDim(0)
        tensor_c.shape.setDimension(0, first_dim)
        second_dim = tensor_b.shape.getDim(1)
        tensor_c.shape.setDimension(1, second_dim)

    def calcAlgFlops(self):
        # [_] TODO (Joel): Will need to handle transposes here...
        # Get matrix inner dimension
        assert(len(self._inputs) == 2)
        tensor_a = self._inputs[0]
        inner_dim = tensor_a.shape.getDim(1)
        # [_] TODO (Joel): HACK!!!!: FIX ME!!! (Create tensor dimension in
        #                  the tensor_shape code, and add a check... either
        #                  both int and equal, or symbols)
        import sympy
        assert (type(inner_dim) == sympy.Symbol or \
                type(self._inputs[1].shape.getDim(0)) == sympy.Symbol or \
                inner_dim == self._inputs[1].shape.getDim(0)), \
               'Dimension check failed in op {} with inputs {} and {}, '\
               'and output {}'.format(self._name, self._inputs[0].shape,
                                      self._inputs[1].shape,
                                      self._outputs[0].shape)
        # Get output number of elements
        assert(len(self._outputs) == 1)
        out_shape = self._outputs[0].shape
        out_elts = out_shape.numElements()
        return (2 * inner_dim * out_elts)


class ReduceOp(Op):
    def __init__(self, name, reduction_op='sum', axis=0):
        super(ReduceOp, self).__init__(name)
        # TODO (Joel): Extend to handle multiple axis dims (like TF)
        assert isinstance(axis, int)
        self._axis = axis
        self._flops_per_element = 1

    def propagateShapes(self):
        assert(len(self._inputs) == 1)
        out_dim = self._inputs[0].shape.getDim(1 - self._axis)
        assert(len(self._outputs) == 1)
        self._outputs[0].shape.setDimension(0, out_dim)

    def calcAlgFlops(self):
        assert(len(self._inputs) == 1)
        in_dim = self._inputs[0].shape.getDim(self._axis)
        assert(len(self._outputs) == 1)
        out_dim = self._outputs[0].shape.getDim(1 - self._axis)
        return (self._flops_per_element * in_dim * out_dim)


