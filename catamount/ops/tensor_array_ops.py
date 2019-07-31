from .base_op import Op
from catamount.tensors.tensor import Tensor


class TensorArray:
    ''' An object to be shared among TensorArray ops to read and write tensor
    handles for while loops. First, the input[0] to the base TensorArray op is
    the number of sequence elements in the array. Second, shape inference on
    downstream ops, there are two ways that the shapes can be determined:
      1) For TensorArrays that have both Scatter and Read downstream ops, the
         shape of the Read's output[0] is the shape of the Scatter's input[2]
         with the sequence dimension removed (0th dimension).
      2) For TensorArrays that have both Gather and Write downstream ops, the
         shape of the Gathers output[0] is the shape of the Write's input[2]
         with an added initial dimension of sequence length.
    '''
    def __init__(self):
        self._parent = None
        # Write ops will need to propagate before we can get the shape
        # of their element tensors for gathers (if there are any)
        self._write_op = None
        self._gather_op = None
        # The tensor reference that holds the sequence length (input to the
        # TensorArrayOp itself)
        self._sequence_tensor = None
        # The tensor reference that holds the full tensor specification (the
        # input to scatter ops, and output from gather ops)
        self._full_tensor = None
        self._element_tensor = None

    def debugString(self):
        return 'TensorArray(name: {}, seq_tens: {}, full_tens: {}, '\
               'elt_tens: {})'.format(self.name, self._sequence_tensor,
                                      self._full_tensor, self._element_tensor)

    def isValid(self):
        array_valid = self._parent is not None
        if not array_valid:
            print('WARN: TensorArray {} is not valid!'.format(self.name))
            print('      parent: {}, write: {}, read: {}'
                  .format(self._parent, self._write, self._read))
        return array_valid

    def associateTensorArrayOp(self, array_op):
        # Only one TensorArrayOp allowed per TensorArray
        assert self._parent is None
        self._parent = array_op

    def associateWriteOp(self, write_op):
        # Only one TensorArrayWriteOp allowed per TensorArray
        assert self._write_op is None
        self._write_op = write_op

    def getWriteOp(self):
        return self._write_op

    def associateGatherOp(self, gather_op):
        # Only one TensorArrayGatherOp allowed per TensorArray
        assert self._gather_op is None
        self._gather_op = gather_op

    def getGatherOp(self):
        return self._gather_op

    def associateSequenceTensor(self, seq_tensor):
        if self._sequence_tensor:
            assert self._sequence_tensor == seq_tensor
        else:
            self._sequence_tensor = seq_tensor

    def associateFullTensor(self, full_tensor):
        # Only one full tensor producer allowed per TensorArray
        assert self._full_tensor is None
        self._full_tensor = full_tensor

    def associateElementTensor(self, element_tensor):
        # Only one full tensor producer allowed per TensorArray
        assert self._element_tensor is None
        self._element_tensor = element_tensor

    def getReadShape(self):
        assert self._sequence_tensor is not None and \
               self._full_tensor is not None
        in_shape = self._full_tensor.shape
        assert self._sequence_tensor.value == in_shape.getDimension(0)
        out_shape = []
        for idx in range(1, in_shape.rank):
            out_shape.append(in_shape.getDimension(idx))
        return out_shape

    def getWriteShape(self):
        in_shape = self._element_tensor.shape
        out_shape = []
        for idx in range(in_shape.rank):
            out_shape.append(in_shape.getDimension(idx))
        return out_shape

    def getSequenceLength(self):
        return self._sequence_tensor.value

    @property
    def name(self):
        return self._parent.name


class BaseArrayOp(Op):
    def __init__(self, name):
        super(BaseArrayOp, self).__init__(name)
        # The array reference to use for pushing and popping
        self._array = None

    def debugString(self):
        to_return = super(BaseArrayOp, self).debugString()
        if self._array is not None:
            to_return += '\n  {}'.format(self._array.debugString())
        return to_return

    def setArray(self, array):
        if self._array is None:
            self._array = array
        else:
            self.debugAssert(self._array == array)

    def getArray(self):
        return self._array

    def calcAlgFlops(self):
        # Array operations have no Flops
        return 0

    def calcAlgBytes(self):
        # Array operations do not perform algorithmic activity,
        # so accessed memory is not algorithmic
        return 0

    def calcAlgFootprint(self):
        # Array operations do not perform algorithmic activity,
        # so accessed memory is not algorithmic
        return 0


class TensorArrayOp(BaseArrayOp):
    def __init__(self, name):
        super(TensorArrayOp, self).__init__(name)
        self._array = TensorArray()
        self._array.associateTensorArrayOp(self)

    def isValid(self):
        return self._array.isValid() and super(TensorArrayOp, self).isValid()

    def propagateShapes(self, make_symbolic=False):
        # First input should be the sequence length
        self.debugAssert(len(self._inputs) == 1)
        # Set the array's sequence tensor so all downstream ops can find it
        self._array.associateSequenceTensor(self._inputs[0])


class TensorArrayReadOp(BaseArrayOp):
    def __init__(self, name):
        super(TensorArrayReadOp, self).__init__(name)

    def propagateShapes(self, make_symbolic=False):
        self.debugAssert(len(self._inputs) == 3)
        self.debugAssert(len(self._outputs) == 1)
        self.debugAssert(self._array is not None)

        # Get the shape from the array
        out_shape = self._array.getReadShape()
        self._outputs[0].mergeShape(out_shape, make_symbolic=make_symbolic)


class TensorArrayWriteOp(BaseArrayOp):
    def __init__(self, name):
        super(TensorArrayWriteOp, self).__init__(name)

    def setArray(self, array):
        super(TensorArrayWriteOp, self).setArray(array)
        self._array.associateWriteOp(self)
        self._array.associateElementTensor(self._inputs[2])

    def propagateShapes(self, make_symbolic=False):
        self.debugAssert(len(self._inputs) == 4)
        # Nothing to do here, since downstream gather op will have access
        # to the element tensor


class TensorArrayGatherOp(BaseArrayOp):
    def __init__(self, name):
        super(TensorArrayGatherOp, self).__init__(name)

    def setArray(self, array):
        super(TensorArrayGatherOp, self).setArray(array)
        self._array.associateGatherOp(self)

    def canVisit(self, visited_ops):
        self.debugAssert(self._array is not None)
        self.debugAssert(self._array._element_tensor is not None)
        write_op = self._array.getWriteOp()
        if write_op not in visited_ops:
            return False
        return super(TensorArrayGatherOp, self).canVisit(visited_ops)

    def propagateShapes(self, make_symbolic=False):
        self.debugAssert(self._array is not None)
        self.debugAssert(self._array._sequence_tensor is not None)
        self.debugAssert(self._array._element_tensor is not None)
        self.debugAssert(len(self._outputs) == 1)
        out_shape = [self._array.getSequenceLength()]
        out_shape.extend(self._array.getWriteShape())
        self._outputs[0].mergeShape(out_shape, make_symbolic=make_symbolic)


class TensorArrayScatterOp(BaseArrayOp):
    def __init__(self, name):
        super(TensorArrayScatterOp, self).__init__(name)

    def propagateShapes(self, make_symbolic=False):
        self.debugAssert(len(self._inputs) == 4)
        self.debugAssert(self._array is not None)
        self._array.associateFullTensor(self._inputs[2])


class TensorArraySizeOp(BaseArrayOp):
    def __init__(self, name):
        super(TensorArraySizeOp, self).__init__(name)

    def propagateShapes(self, make_symbolic=False):
        self.debugAssert(len(self._outputs) == 1)

        out_val = self._array.getSequenceLength()
        self._outputs[0].setValue(out_val)


class TensorArrayGradOp(BaseArrayOp):
    def __init__(self, name):
        super(TensorArrayGradOp, self).__init__(name)

    def propagateShapes(self, make_symbolic=False):
        self.debugAssert(len(self._outputs) == 2)
        # Nothing to do here
