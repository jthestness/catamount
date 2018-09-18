from .base_op import Op


class PlaceholderOp(Op):
    def __init__(self, name):
        super(PlaceholderOp, self).__init__(name)

    def bindTensorShapeDimension(self, dim_index, dim_name_or_symbol,
                                 make_symbolic=False):
        self.debugAssert(len(self._outputs) == 1)
        self._outputs[0].shape.setDimension(dim_index, dim_name_or_symbol,
                                            make_symbolic=make_symbolic)

    def propagateShapes(self, make_symbolic=False):
        # Placeholders have no inputs to propagate
        pass

    def calcAlgFlops(self):
        # Placeholders have no Flops
        return 0

    def calcAlgBytes(self):
        return self.bytesAccessOutput()

    def calcAlgFootprint(self):
        # Return the size of the output tensor, which must be accessed
        return self.bytesAccessOutput()

