import sympy
import catamount.frameworks.tensorflow
from catamount.api import utils
from catamount.ops import UnknownOp


tf_han_meta = 'tf_han_0/model.ckpt.meta'

def test_tf_han_model():
    graph = catamount.frameworks.tensorflow.import_graph(tf_han_meta)
    assert graph.isValid()

    # correct_init_params = 71065
    correct_init_params = 142146
    init_params = graph.calcModelParameters()
    print('Initial model parameters: {}'.format(init_params))
    assert init_params == correct_init_params

    print('\nInitial graph:\n{}\n'.format(graph))

    base_vocab_size = 21
    base_sequence_length = 93
    vocab_size_symbol = utils.getIntSymbolFromString('vocab_size')
    seq_length_symbol = utils.getIntSymbolFromString('sequence_length')
    batch_size_symbol = utils.getIntSymbolFromString('batch_size')

    # HAXXX: Manually setting TensorArray shapes!
    for op in graph._ops_by_name.values():
        op_name_suffix = op.name.split('/')[-1]
        if 'TensorArrayGather' in op_name_suffix:
            assert isinstance(op, UnknownOp)
            assert len(op._inputs) == 3
            assert len(op._outputs) == 1
            assert op._outputs[0].shape.rank == 3, \
                   '{}'.format(op.name)
            assert op._outputs[0].shape.dims[2] == base_vocab_size
            if op.name == 'training/RMSprop/gradients/bidirectional_1/TensorArrayUnstack_2/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGatherV3' or \
               op.name == 'training/RMSprop/gradients/bidirectional_1/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGatherV3' or \
               op.name == 'bidirectional_1/TensorArrayStack/TensorArrayGatherV3' or \
               op.name == 'bidirectional_1/TensorArrayStack_1/TensorArrayGatherV3':
                zeroth_dim = 1
            else:
                zeroth_dim = seq_length_symbol
            gather_shape = [zeroth_dim,
                            batch_size_symbol,
                            vocab_size_symbol]
            op._outputs[0].mergeShape(gather_shape, make_symbolic=True)
        elif 'TensorArraySize' in op_name_suffix:
            assert isinstance(op, UnknownOp)
            assert len(op._inputs) == 2
            assert len(op._outputs) == 1
            assert op._outputs[0].shape.rank == 0
            # print('FOUND TensorArraySize: {}'.format(op.debugString()))
            op._outputs[0].setValue(seq_length_symbol)
        elif 'TensorArrayRead' in op_name_suffix:
            assert isinstance(op, UnknownOp)
            assert len(op._inputs) == 3
            assert len(op._outputs) == 1
            assert op._outputs[0].shape.isUnknown() or \
                   op._outputs[0].shape.rank == 2, \
                   '{}'.format(op.name)
            if op._outputs[0].shape.dims[1].value == base_vocab_size:
                out_shape = [batch_size_symbol, vocab_size_symbol]
            else:
                assert op._outputs[0].shape.dims[1].value == 1
                out_shape = [batch_size_symbol, 1]
            op._outputs[0].mergeShape(out_shape, make_symbolic=True)

    # Manually remove unused parts of the graph
    all_ops = list(graph.opsByName.values())
    for op in all_ops:
        if 'time_distributed_2' in op.name or \
           'loss_1' in op.name or 'metrics_1' in op.name or \
           'loss_2' in op.name or 'metrics_2' in op.name or \
           op.name.startswith('loss/'):
            graph.removeOp(op)



    const_dict = {
                  'AttnWrapper/Reshape/shape': [-1, vocab_size_symbol],
                  'AttnWrapper/Reshape_2/shape': [-1, seq_length_symbol],
                  'AttnWrapper/Reshape_3/shape': [seq_length_symbol, -1],
                  'AttnWrapper/Reshape_4/shape/1': seq_length_symbol,
                  'AttnWrapper/Reshape_4/shape/2': seq_length_symbol,
                  'AttnWrapper/Reshape_5/shape': [-1, seq_length_symbol],
                  'AttnWrapper/Reshape_6/shape': [seq_length_symbol, -1],
                  'AttnWrapper/Reshape_7/shape/1': seq_length_symbol,
                  'AttnWrapper/Shape_2': [seq_length_symbol, seq_length_symbol],
                  'AttnWrapper/Shape_4': [seq_length_symbol, 1],
                  'AttnWrapper/Tile/multiples': [1, vocab_size_symbol],
                  'AttnWrapper/random_uniform/shape': [seq_length_symbol],
                  'AttnWrapper/random_uniform_1/shape': [seq_length_symbol, seq_length_symbol],
                  'AttnWrapper/random_uniform_2/shape': [vocab_size_symbol, seq_length_symbol],
                  'AttnWrapper/stack': [-1, seq_length_symbol, seq_length_symbol],
                  'AttnWrapper/stack_1': [1, seq_length_symbol, 1],
                  'AttnWrapper/stack_2': [1, seq_length_symbol, 1],
                  'AttnWrapper/while/Reshape/shape': [-1, seq_length_symbol],
                  'AttnWrapper/while/Reshape_1/shape': [seq_length_symbol, -1],
                  'AttnWrapper/while/Reshape_2/shape/1': seq_length_symbol,
                  'AttnWrapper/while/Reshape_2/shape/2': seq_length_symbol,
                  'AttnWrapper/while/Reshape_3/shape': [-1, seq_length_symbol],
                  'AttnWrapper/while/Reshape_4/shape': [seq_length_symbol, -1],
                  'AttnWrapper/while/Reshape_5/shape/1': seq_length_symbol,
                  'AttnWrapper/while/Shape_1': [seq_length_symbol, seq_length_symbol],
                  'AttnWrapper/while/Shape_3': [seq_length_symbol, 1],
                  'AttnWrapper/while/maximum_iterations': seq_length_symbol,
                  'AttnWrapper/while/stack': [1, seq_length_symbol, 1],
                  'AttnWrapper/while/stack_1': [1, seq_length_symbol, 1],
                  'AttnWrapper_1/Reshape/shape': [-1, vocab_size_symbol],
                  'AttnWrapper_1/Tile/multiples': [1, vocab_size_symbol],
                  'AttnWrapper_1/random_uniform_2/shape': [vocab_size_symbol, 1],
                  'BLSTM1/Tile/multiples': [1, vocab_size_symbol],
                  'BLSTM1/Tile_1/multiples': [1, vocab_size_symbol],
                  'BLSTM1/Tile_2/multiples': [1, vocab_size_symbol],
                  'BLSTM1/Tile_3/multiples': [1, vocab_size_symbol],
                  'BLSTM1/backward_lstm_1/random_uniform/shape': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'BLSTM1/backward_lstm_1/strided_slice/stack_1': [0, vocab_size_symbol],
                  'BLSTM1/backward_lstm_1/strided_slice_1/stack': [0, vocab_size_symbol],
                  'BLSTM1/backward_lstm_1/strided_slice_1/stack_1': [0, 2 * vocab_size_symbol],
                  'BLSTM1/backward_lstm_1/strided_slice_10/stack': [2 * vocab_size_symbol],
                  'BLSTM1/backward_lstm_1/strided_slice_2/stack': [0, 2 * vocab_size_symbol],
                  'BLSTM1/backward_lstm_1/strided_slice_4/stack_1': [0, vocab_size_symbol],
                  'BLSTM1/backward_lstm_1/strided_slice_5/stack': [0, vocab_size_symbol],
                  'BLSTM1/backward_lstm_1/strided_slice_5/stack_1': [0, 2 * vocab_size_symbol],
                  'BLSTM1/backward_lstm_1/strided_slice_6/stack': [0, 2 * vocab_size_symbol],
                  'BLSTM1/backward_lstm_1/strided_slice_8/stack_1': [vocab_size_symbol],
                  'BLSTM1/backward_lstm_1/strided_slice_9/stack': [vocab_size_symbol],
                  'BLSTM1/backward_lstm_1/strided_slice_9/stack_1': [2 * vocab_size_symbol],
                  'BLSTM1/forward_lstm_1/random_uniform/shape': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'BLSTM1/forward_lstm_1/strided_slice/stack_1': [0, vocab_size_symbol],
                  'BLSTM1/forward_lstm_1/strided_slice_1/stack': [0, vocab_size_symbol],
                  'BLSTM1/forward_lstm_1/strided_slice_1/stack_1': [0, 2 * vocab_size_symbol],
                  'BLSTM1/forward_lstm_1/strided_slice_10/stack': [2 * vocab_size_symbol],
                  'BLSTM1/forward_lstm_1/strided_slice_2/stack': [0, 2 * vocab_size_symbol],
                  'BLSTM1/forward_lstm_1/strided_slice_4/stack_1': [0, vocab_size_symbol],
                  'BLSTM1/forward_lstm_1/strided_slice_5/stack': [0, vocab_size_symbol],
                  'BLSTM1/forward_lstm_1/strided_slice_5/stack_1': [0, 2 * vocab_size_symbol],
                  'BLSTM1/forward_lstm_1/strided_slice_6/stack': [0, 2 * vocab_size_symbol],
                  'BLSTM1/forward_lstm_1/strided_slice_8/stack_1': [vocab_size_symbol],
                  'BLSTM1/forward_lstm_1/strided_slice_9/stack': [vocab_size_symbol],
                  'BLSTM1/forward_lstm_1/strided_slice_9/stack_1': [2 * vocab_size_symbol],
                  'BLSTM1/while/maximum_iterations': seq_length_symbol,
                  'BLSTM1/while_1/maximum_iterations': seq_length_symbol,
                  'bidirectional_1/Tile/multiples': [1, vocab_size_symbol],
                  'bidirectional_1/Tile_1/multiples': [1, vocab_size_symbol],
                  'bidirectional_1/Tile_2/multiples': [1, vocab_size_symbol],
                  'bidirectional_1/Tile_3/multiples': [1, vocab_size_symbol],
                  'bidirectional_1/backward_lstm_2/random_uniform/shape': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'bidirectional_1/backward_lstm_2/strided_slice/stack_1': [0, vocab_size_symbol],
                  'bidirectional_1/backward_lstm_2/strided_slice_1/stack': [0, vocab_size_symbol],
                  'bidirectional_1/backward_lstm_2/strided_slice_1/stack_1': [0, 2 * vocab_size_symbol],
                  'bidirectional_1/backward_lstm_2/strided_slice_10/stack': [2 * vocab_size_symbol],
                  'bidirectional_1/backward_lstm_2/strided_slice_2/stack': [0, 2 * vocab_size_symbol],
                  'bidirectional_1/backward_lstm_2/strided_slice_4/stack_1': [0, vocab_size_symbol],
                  'bidirectional_1/backward_lstm_2/strided_slice_5/stack': [0, vocab_size_symbol],
                  'bidirectional_1/backward_lstm_2/strided_slice_5/stack_1': [0, 2 * vocab_size_symbol],
                  'bidirectional_1/backward_lstm_2/strided_slice_6/stack': [0, 2 * vocab_size_symbol],
                  'bidirectional_1/backward_lstm_2/strided_slice_8/stack_1': [vocab_size_symbol],
                  'bidirectional_1/backward_lstm_2/strided_slice_9/stack': [vocab_size_symbol],
                  'bidirectional_1/backward_lstm_2/strided_slice_9/stack_1': [2 * vocab_size_symbol],
                  'bidirectional_1/forward_lstm_2/random_uniform/shape': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'bidirectional_1/forward_lstm_2/strided_slice/stack_1': [0, vocab_size_symbol],
                  'bidirectional_1/forward_lstm_2/strided_slice_1/stack': [0, vocab_size_symbol],
                  'bidirectional_1/forward_lstm_2/strided_slice_1/stack_1': [0, 2 * vocab_size_symbol],
                  'bidirectional_1/forward_lstm_2/strided_slice_10/stack': [2 * vocab_size_symbol],
                  'bidirectional_1/forward_lstm_2/strided_slice_2/stack': [0, 2 * vocab_size_symbol],
                  'bidirectional_1/forward_lstm_2/strided_slice_4/stack_1': [0, vocab_size_symbol],
                  'bidirectional_1/forward_lstm_2/strided_slice_5/stack': [0, vocab_size_symbol],
                  'bidirectional_1/forward_lstm_2/strided_slice_5/stack_1': [0, 2 * vocab_size_symbol],
                  'bidirectional_1/forward_lstm_2/strided_slice_6/stack': [0, 2 * vocab_size_symbol],
                  'bidirectional_1/forward_lstm_2/strided_slice_8/stack_1': [vocab_size_symbol],
                  'bidirectional_1/forward_lstm_2/strided_slice_9/stack': [vocab_size_symbol],
                  'bidirectional_1/forward_lstm_2/strided_slice_9/stack_1': [2 * vocab_size_symbol],
                  'embedding_1/random_uniform/shape': [seq_length_symbol, vocab_size_symbol],
                  'seqOut/random_uniform/shape': [vocab_size_symbol, 1],
                  'time_distributed_1/AttnWrapper/Reshape/shape': [-1, vocab_size_symbol],
                  'time_distributed_1/AttnWrapper/Reshape_2/shape': [-1, seq_length_symbol],
                  'time_distributed_1/AttnWrapper/Reshape_3/shape': [seq_length_symbol, -1],
                  'time_distributed_1/AttnWrapper/Reshape_4/shape/1': seq_length_symbol,
                  'time_distributed_1/AttnWrapper/Reshape_4/shape/2': seq_length_symbol,
                  'time_distributed_1/AttnWrapper/Reshape_5/shape': [-1, seq_length_symbol],
                  'time_distributed_1/AttnWrapper/Reshape_6/shape': [seq_length_symbol, -1],
                  'time_distributed_1/AttnWrapper/Reshape_7/shape/1': seq_length_symbol,
                  'time_distributed_1/AttnWrapper/Shape_2': [seq_length_symbol, seq_length_symbol],
                  'time_distributed_1/AttnWrapper/Shape_4': [seq_length_symbol, 1],
                  'time_distributed_1/AttnWrapper/Tile/multiples': [1, vocab_size_symbol],
                  'time_distributed_1/AttnWrapper/stack': [-1, seq_length_symbol, seq_length_symbol],
                  'time_distributed_1/AttnWrapper/stack_1': [1, seq_length_symbol, 1],
                  'time_distributed_1/AttnWrapper/stack_2': [1, seq_length_symbol, 1],
                  'time_distributed_1/AttnWrapper/while/Reshape/shape': [-1, seq_length_symbol],
                  'time_distributed_1/AttnWrapper/while/Reshape_1/shape': [seq_length_symbol, -1],
                  'time_distributed_1/AttnWrapper/while/Reshape_2/shape/1': seq_length_symbol,
                  'time_distributed_1/AttnWrapper/while/Reshape_2/shape/2': seq_length_symbol,
                  'time_distributed_1/AttnWrapper/while/Reshape_3/shape': [-1, seq_length_symbol],
                  'time_distributed_1/AttnWrapper/while/Reshape_4/shape': [seq_length_symbol, -1],
                  'time_distributed_1/AttnWrapper/while/Reshape_5/shape/1': seq_length_symbol,
                  'time_distributed_1/AttnWrapper/while/Shape_1': [seq_length_symbol, seq_length_symbol],
                  'time_distributed_1/AttnWrapper/while/Shape_3': [seq_length_symbol, 1],
                  'time_distributed_1/AttnWrapper/while/stack': [1, seq_length_symbol, 1],
                  'time_distributed_1/AttnWrapper/while/stack_1': [1, seq_length_symbol, 1],
                  'time_distributed_1/AttnWrapper_1/Reshape/shape': [-1, vocab_size_symbol],
                  'time_distributed_1/AttnWrapper_1/Reshape_2/shape': [-1, seq_length_symbol],
                  'time_distributed_1/AttnWrapper_1/Reshape_3/shape': [seq_length_symbol, -1],
                  'time_distributed_1/AttnWrapper_1/Reshape_4/shape/1': seq_length_symbol,
                  'time_distributed_1/AttnWrapper_1/Reshape_4/shape/2': seq_length_symbol,
                  'time_distributed_1/AttnWrapper_1/Reshape_5/shape': [-1, seq_length_symbol],
                  'time_distributed_1/AttnWrapper_1/Reshape_6/shape': [seq_length_symbol, -1],
                  'time_distributed_1/AttnWrapper_1/Reshape_7/shape/1': seq_length_symbol,
                  'time_distributed_1/AttnWrapper_1/Shape_2': [seq_length_symbol, seq_length_symbol],
                  'time_distributed_1/AttnWrapper_1/Shape_4': [seq_length_symbol, 1],
                  'time_distributed_1/AttnWrapper_1/Tile/multiples': [1, vocab_size_symbol],
                  'time_distributed_1/AttnWrapper_1/stack': [-1, seq_length_symbol, seq_length_symbol],
                  'time_distributed_1/AttnWrapper_1/stack_1': [1, seq_length_symbol, 1],
                  'time_distributed_1/AttnWrapper_1/stack_2': [1, seq_length_symbol, 1],
                  'time_distributed_1/AttnWrapper_1/while/Reshape/shape': [-1, seq_length_symbol],
                  'time_distributed_1/AttnWrapper_1/while/Reshape_1/shape': [seq_length_symbol, -1],
                  'time_distributed_1/AttnWrapper_1/while/Reshape_2/shape/1': seq_length_symbol,
                  'time_distributed_1/AttnWrapper_1/while/Reshape_2/shape/2': seq_length_symbol,
                  'time_distributed_1/AttnWrapper_1/while/Reshape_3/shape': [-1, seq_length_symbol],
                  'time_distributed_1/AttnWrapper_1/while/Reshape_4/shape': [seq_length_symbol, -1],
                  'time_distributed_1/AttnWrapper_1/while/Reshape_5/shape/1': seq_length_symbol,
                  'time_distributed_1/AttnWrapper_1/while/Shape_1': [seq_length_symbol, seq_length_symbol],
                  'time_distributed_1/AttnWrapper_1/while/Shape_3': [seq_length_symbol, 1],
                  'time_distributed_1/AttnWrapper_1/while/stack': [1, seq_length_symbol, 1],
                  'time_distributed_1/AttnWrapper_1/while/stack_1': [1, seq_length_symbol, 1],
                  'time_distributed_1/BLSTM1/Tile/multiples': [1, vocab_size_symbol],
                  'time_distributed_1/BLSTM1/Tile_1/multiples': [1, vocab_size_symbol],
                  'time_distributed_1/BLSTM1/Tile_2/multiples': [1, vocab_size_symbol],
                  'time_distributed_1/BLSTM1/Tile_3/multiples': [1, vocab_size_symbol],
                  'time_distributed_1/BLSTM1/while/maximum_iterations': seq_length_symbol,
                  'time_distributed_1/BLSTM1/while_1/maximum_iterations': seq_length_symbol,
                  'time_distributed_1/BLSTM1_1/Tile/multiples': [1, vocab_size_symbol],
                  'time_distributed_1/BLSTM1_1/Tile_1/multiples': [1, vocab_size_symbol],
                  'time_distributed_1/BLSTM1_1/Tile_2/multiples': [1, vocab_size_symbol],
                  'time_distributed_1/BLSTM1_1/Tile_3/multiples': [1, vocab_size_symbol],
                  'time_distributed_1/BLSTM1_1/while/maximum_iterations': seq_length_symbol,
                  'time_distributed_1/BLSTM1_1/while_1/maximum_iterations': seq_length_symbol,
                  'time_distributed_1/Reshape/shape': [-1, seq_length_symbol],
                  'time_distributed_1/Reshape_2/shape': [-1, 1, vocab_size_symbol],
                  'BLSTM1/backward_lstm_1/strided_slice_10/stack_1': [3 * vocab_size_symbol],
                  'BLSTM1/backward_lstm_1/strided_slice_11/stack': [3 * vocab_size_symbol],
                  'BLSTM1/backward_lstm_1/strided_slice_2/stack_1': [0, 3 * vocab_size_symbol],
                  'BLSTM1/backward_lstm_1/strided_slice_3/stack': [0, 3 * vocab_size_symbol],
                  'BLSTM1/backward_lstm_1/strided_slice_6/stack_1': [0, 3 * vocab_size_symbol],
                  'BLSTM1/backward_lstm_1/strided_slice_7/stack': [0, 3 * vocab_size_symbol],
                  'BLSTM1/forward_lstm_1/strided_slice_10/stack_1': [3 * vocab_size_symbol],
                  'BLSTM1/forward_lstm_1/strided_slice_11/stack': [3 * vocab_size_symbol],
                  'BLSTM1/forward_lstm_1/strided_slice_2/stack_1': [0, 3 * vocab_size_symbol],
                  'BLSTM1/forward_lstm_1/strided_slice_3/stack': [0, 3 * vocab_size_symbol],
                  'BLSTM1/forward_lstm_1/strided_slice_6/stack_1': [0, 3 * vocab_size_symbol],
                  'BLSTM1/forward_lstm_1/strided_slice_7/stack': [0, 3 * vocab_size_symbol],
                  'bidirectional_1/backward_lstm_2/strided_slice_10/stack_1': [3 * vocab_size_symbol],
                  'bidirectional_1/backward_lstm_2/strided_slice_11/stack': [3 * vocab_size_symbol],
                  'bidirectional_1/backward_lstm_2/strided_slice_2/stack_1': [0, 3 * vocab_size_symbol],
                  'bidirectional_1/backward_lstm_2/strided_slice_3/stack': [0, 3 * vocab_size_symbol],
                  'bidirectional_1/backward_lstm_2/strided_slice_6/stack_1': [0, 3 * vocab_size_symbol],
                  'bidirectional_1/backward_lstm_2/strided_slice_7/stack': [0, 3 * vocab_size_symbol],
                  'bidirectional_1/forward_lstm_2/strided_slice_10/stack_1': [3 * vocab_size_symbol],
                  'bidirectional_1/forward_lstm_2/strided_slice_11/stack': [3 * vocab_size_symbol],
                  'bidirectional_1/forward_lstm_2/strided_slice_2/stack_1': [0, 3 * vocab_size_symbol],
                  'bidirectional_1/forward_lstm_2/strided_slice_3/stack': [0, 3 * vocab_size_symbol],
                  'bidirectional_1/forward_lstm_2/strided_slice_6/stack_1': [0, 3 * vocab_size_symbol],
                  'bidirectional_1/forward_lstm_2/strided_slice_7/stack': [0, 3 * vocab_size_symbol],
                  'bidirectional_1/forward_lstm_2/strided_slice_10/stack_1': [3 * vocab_size_symbol],
                  'bidirectional_1/forward_lstm_2/strided_slice_10/stack_1': [3 * vocab_size_symbol],
                  'bidirectional_1/backward_lstm_2/strided_slice_6/stack_1': [0, 3 * vocab_size_symbol],
                  'bidirectional_1/backward_lstm_2/strided_slice_6/stack_1': [0, 3 * vocab_size_symbol],
                  'BLSTM1/backward_lstm_1/strided_slice_11/stack': [3 * vocab_size_symbol],
                  'BLSTM1/backward_lstm_1/strided_slice_11/stack': [3 * vocab_size_symbol],
                  'bidirectional_1/forward_lstm_2/strided_slice_11/stack': [3 * vocab_size_symbol],
                  'bidirectional_1/forward_lstm_2/strided_slice_11/stack': [3 * vocab_size_symbol],
                  'BLSTM1/forward_lstm_1/strided_slice_11/stack': [3 * vocab_size_symbol],
                  'BLSTM1/forward_lstm_1/strided_slice_11/stack': [3 * vocab_size_symbol],
                  'bidirectional_1/backward_lstm_2/strided_slice_7/stack': [0, 3 * vocab_size_symbol],
                  'bidirectional_1/backward_lstm_2/strided_slice_7/stack': [0, 3 * vocab_size_symbol],
                  'BLSTM1/forward_lstm_1/strided_slice_7/stack': [0, 3 * vocab_size_symbol],
                  'BLSTM1/forward_lstm_1/strided_slice_7/stack': [0, 3 * vocab_size_symbol],
                  'BLSTM1/backward_lstm_1/strided_slice_10/stack_1': [3 * vocab_size_symbol],
                  'BLSTM1/backward_lstm_1/strided_slice_10/stack_1': [3 * vocab_size_symbol],
                  'BLSTM1/forward_lstm_1/strided_slice_10/stack_1': [3 * vocab_size_symbol],
                  'BLSTM1/forward_lstm_1/strided_slice_10/stack_1': [3 * vocab_size_symbol],
                  'bidirectional_1/backward_lstm_2/strided_slice_10/stack_1': [3 * vocab_size_symbol],
                  'bidirectional_1/backward_lstm_2/strided_slice_10/stack_1': [3 * vocab_size_symbol],
                  'bidirectional_1/backward_lstm_2/strided_slice_11/stack': [3 * vocab_size_symbol],
                  'bidirectional_1/backward_lstm_2/strided_slice_11/stack': [3 * vocab_size_symbol],
                  'BLSTM1/backward_lstm_1/strided_slice_6/stack_1': [0, 3 * vocab_size_symbol],
                  'BLSTM1/backward_lstm_1/strided_slice_6/stack_1': [0, 3 * vocab_size_symbol],
                  'BLSTM1/backward_lstm_1/strided_slice_7/stack': [0, 3 * vocab_size_symbol],
                  'BLSTM1/backward_lstm_1/strided_slice_7/stack': [0, 3 * vocab_size_symbol],
                  'bidirectional_1/forward_lstm_2/strided_slice_2/stack_1': [0, 3 * vocab_size_symbol],
                  'bidirectional_1/forward_lstm_2/strided_slice_2/stack_1': [0, 3 * vocab_size_symbol],
                  'BLSTM1/backward_lstm_1/strided_slice_3/stack': [0, 3 * vocab_size_symbol],
                  'BLSTM1/backward_lstm_1/strided_slice_3/stack': [0, 3 * vocab_size_symbol],
                  'bidirectional_1/forward_lstm_2/strided_slice_3/stack': [0, 3 * vocab_size_symbol],
                  'bidirectional_1/forward_lstm_2/strided_slice_3/stack': [0, 3 * vocab_size_symbol],
                  'BLSTM1/forward_lstm_1/strided_slice_6/stack_1': [0, 3 * vocab_size_symbol],
                  'BLSTM1/forward_lstm_1/strided_slice_6/stack_1': [0, 3 * vocab_size_symbol],
                  'bidirectional_1/backward_lstm_2/strided_slice_2/stack_1': [0, 3 * vocab_size_symbol],
                  'bidirectional_1/backward_lstm_2/strided_slice_2/stack_1': [0, 3 * vocab_size_symbol],
                  'BLSTM1/backward_lstm_1/strided_slice_2/stack_1': [0, 3 * vocab_size_symbol],
                  'BLSTM1/backward_lstm_1/strided_slice_2/stack_1': [0, 3 * vocab_size_symbol],
                  'BLSTM1/forward_lstm_1/strided_slice_2/stack_1': [0, 3 * vocab_size_symbol],
                  'BLSTM1/forward_lstm_1/strided_slice_2/stack_1': [0, 3 * vocab_size_symbol],
                  'bidirectional_1/forward_lstm_2/strided_slice_6/stack_1': [0, 3 * vocab_size_symbol],
                  'bidirectional_1/forward_lstm_2/strided_slice_6/stack_1': [0, 3 * vocab_size_symbol],
                  'BLSTM1/forward_lstm_1/strided_slice_3/stack': [0, 3 * vocab_size_symbol],
                  'BLSTM1/forward_lstm_1/strided_slice_3/stack': [0, 3 * vocab_size_symbol],
                  'bidirectional_1/forward_lstm_2/strided_slice_7/stack': [0, 3 * vocab_size_symbol],
                  'bidirectional_1/forward_lstm_2/strided_slice_7/stack': [0, 3 * vocab_size_symbol],
                  'bidirectional_1/backward_lstm_2/strided_slice_3/stack': [0, 3 * vocab_size_symbol],
                  'bidirectional_1/backward_lstm_2/strided_slice_3/stack': [0, 3 * vocab_size_symbol],
                  'training/RMSprop/gradients/AttnWrapper_1/while/add_14_grad/Shape_1': [vocab_size_symbol],
                  'training/RMSprop/gradients/BLSTM1/backward_lstm_1/strided_slice_10_grad/Shape': [4 * vocab_size_symbol],
                  'training/RMSprop/gradients/BLSTM1/backward_lstm_1/strided_slice_11_grad/Shape': [4 * vocab_size_symbol],
                  'training/RMSprop/gradients/BLSTM1/backward_lstm_1/strided_slice_1_grad/Shape': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/gradients/BLSTM1/backward_lstm_1/strided_slice_2_grad/Shape': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/gradients/BLSTM1/backward_lstm_1/strided_slice_3_grad/Shape': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/gradients/BLSTM1/backward_lstm_1/strided_slice_4_grad/Shape': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/gradients/BLSTM1/backward_lstm_1/strided_slice_5_grad/Shape': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/gradients/BLSTM1/backward_lstm_1/strided_slice_6_grad/Shape': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/gradients/BLSTM1/backward_lstm_1/strided_slice_7_grad/Shape': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/gradients/BLSTM1/backward_lstm_1/strided_slice_8_grad/Shape': [4 * vocab_size_symbol],
                  'training/RMSprop/gradients/BLSTM1/backward_lstm_1/strided_slice_9_grad/Shape': [4 * vocab_size_symbol],
                  'training/RMSprop/gradients/BLSTM1/backward_lstm_1/strided_slice_grad/Shape': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/gradients/BLSTM1/forward_lstm_1/strided_slice_10_grad/Shape': [4 * vocab_size_symbol],
                  'training/RMSprop/gradients/BLSTM1/forward_lstm_1/strided_slice_11_grad/Shape': [4 * vocab_size_symbol],
                  'training/RMSprop/gradients/BLSTM1/forward_lstm_1/strided_slice_1_grad/Shape': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/gradients/BLSTM1/forward_lstm_1/strided_slice_2_grad/Shape': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/gradients/BLSTM1/forward_lstm_1/strided_slice_3_grad/Shape': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/gradients/BLSTM1/forward_lstm_1/strided_slice_4_grad/Shape': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/gradients/BLSTM1/forward_lstm_1/strided_slice_5_grad/Shape': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/gradients/BLSTM1/forward_lstm_1/strided_slice_6_grad/Shape': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/gradients/BLSTM1/forward_lstm_1/strided_slice_7_grad/Shape': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/gradients/BLSTM1/forward_lstm_1/strided_slice_8_grad/Shape': [4 * vocab_size_symbol],
                  'training/RMSprop/gradients/BLSTM1/forward_lstm_1/strided_slice_9_grad/Shape': [4 * vocab_size_symbol],
                  'training/RMSprop/gradients/BLSTM1/forward_lstm_1/strided_slice_grad/Shape': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/gradients/bidirectional_1/backward_lstm_2/strided_slice_10_grad/Shape': [4 * vocab_size_symbol],
                  'training/RMSprop/gradients/bidirectional_1/backward_lstm_2/strided_slice_11_grad/Shape': [4 * vocab_size_symbol],
                  'training/RMSprop/gradients/bidirectional_1/backward_lstm_2/strided_slice_1_grad/Shape': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/gradients/bidirectional_1/backward_lstm_2/strided_slice_2_grad/Shape': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/gradients/bidirectional_1/backward_lstm_2/strided_slice_3_grad/Shape': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/gradients/bidirectional_1/backward_lstm_2/strided_slice_4_grad/Shape': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/gradients/bidirectional_1/backward_lstm_2/strided_slice_5_grad/Shape': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/gradients/bidirectional_1/backward_lstm_2/strided_slice_6_grad/Shape': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/gradients/bidirectional_1/backward_lstm_2/strided_slice_7_grad/Shape': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/gradients/bidirectional_1/backward_lstm_2/strided_slice_8_grad/Shape': [4 * vocab_size_symbol],
                  'training/RMSprop/gradients/bidirectional_1/backward_lstm_2/strided_slice_9_grad/Shape': [4 * vocab_size_symbol],
                  'training/RMSprop/gradients/bidirectional_1/backward_lstm_2/strided_slice_grad/Shape': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/gradients/bidirectional_1/forward_lstm_2/strided_slice_10_grad/Shape': [4 * vocab_size_symbol],
                  'training/RMSprop/gradients/bidirectional_1/forward_lstm_2/strided_slice_11_grad/Shape': [4 * vocab_size_symbol],
                  'training/RMSprop/gradients/bidirectional_1/forward_lstm_2/strided_slice_1_grad/Shape': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/gradients/bidirectional_1/forward_lstm_2/strided_slice_2_grad/Shape': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/gradients/bidirectional_1/forward_lstm_2/strided_slice_3_grad/Shape': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/gradients/bidirectional_1/forward_lstm_2/strided_slice_4_grad/Shape': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/gradients/bidirectional_1/forward_lstm_2/strided_slice_5_grad/Shape': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/gradients/bidirectional_1/forward_lstm_2/strided_slice_6_grad/Shape': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/gradients/bidirectional_1/forward_lstm_2/strided_slice_7_grad/Shape': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/gradients/bidirectional_1/forward_lstm_2/strided_slice_8_grad/Shape': [4 * vocab_size_symbol],
                  'training/RMSprop/gradients/bidirectional_1/forward_lstm_2/strided_slice_9_grad/Shape': [4 * vocab_size_symbol],
                  'training/RMSprop/gradients/bidirectional_1/forward_lstm_2/strided_slice_grad/Shape': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/gradients/time_distributed_1/AttnWrapper/while/ExpandDims_1_grad/Shape': [seq_length_symbol],
                  'training/RMSprop/gradients/time_distributed_1/AttnWrapper/while/Reshape_1_grad/Shape': [seq_length_symbol, seq_length_symbol],
                  'training/RMSprop/gradients/time_distributed_1/AttnWrapper/while/Reshape_4_grad/Shape': [seq_length_symbol,  1],
                  'training/RMSprop/gradients/time_distributed_1/AttnWrapper/while/Tile_1_grad/stack/Const': [1, seq_length_symbol,  1],
                  'training/RMSprop/gradients/time_distributed_1/AttnWrapper/while/Tile_grad/stack/Const': [1, seq_length_symbol,  1],
                  'training/RMSprop/gradients/time_distributed_1/AttnWrapper/while/add_13_grad/Shape_1': [vocab_size_symbol],
                  'training/RMSprop/gradients/time_distributed_1/AttnWrapper/while/add_3_grad/Shape_1': [seq_length_symbol],
                  'training/RMSprop/gradients/time_distributed_1/AttnWrapper/while/add_6_grad/Shape_1': [seq_length_symbol],
                  'training/RMSprop/gradients/time_distributed_1/AttnWrapper/while/add_9_grad/Shape_1': [seq_length_symbol],
                  'training/RMSprop/gradients/time_distributed_1/embedding_1/embedding_lookup_grad/Shape': [seq_length_symbol, vocab_size_symbol],
                  'training/RMSprop/zeros/shape_as_tensor': [seq_length_symbol, vocab_size_symbol],
                  'training/RMSprop/zeros_1/shape_as_tensor': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/zeros_11/shape_as_tensor': [vocab_size_symbol, seq_length_symbol],
                  'training/RMSprop/zeros_12/shape_as_tensor': [seq_length_symbol, seq_length_symbol],
                  'training/RMSprop/zeros_13/shape_as_tensor': [vocab_size_symbol, seq_length_symbol],
                  'training/RMSprop/zeros_15/shape_as_tensor': [vocab_size_symbol, seq_length_symbol],
                  'training/RMSprop/zeros_16/shape_as_tensor': [seq_length_symbol, seq_length_symbol],
                  'training/RMSprop/zeros_17/shape_as_tensor': [vocab_size_symbol, seq_length_symbol],
                  'training/RMSprop/zeros_19/shape_as_tensor': [vocab_size_symbol, seq_length_symbol],
                  'training/RMSprop/zeros_2/shape_as_tensor': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/zeros_20/shape_as_tensor': [seq_length_symbol, seq_length_symbol],
                  'training/RMSprop/zeros_24/shape_as_tensor': [seq_length_symbol, vocab_size_symbol],
                  'training/RMSprop/zeros_27/shape_as_tensor': [vocab_size_symbol, seq_length_symbol],
                  'training/RMSprop/zeros_28/shape_as_tensor': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/zeros_29/shape_as_tensor': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/zeros_31/shape_as_tensor': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/zeros_32/shape_as_tensor': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/zeros_4/shape_as_tensor': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/zeros_5/shape_as_tensor': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/zeros_8/shape_as_tensor': [seq_length_symbol, seq_length_symbol],
                  'training/RMSprop/zeros_9/shape_as_tensor': [vocab_size_symbol, seq_length_symbol],
                  'training/RMSprop/gradients/bidirectional_1/backward_lstm_2/strided_slice_10_grad/Shape': [4 * vocab_size_symbol],
                  'training/RMSprop/gradients/bidirectional_1/backward_lstm_2/strided_slice_10_grad/Shape': [4 * vocab_size_symbol],
                  'training/RMSprop/gradients/bidirectional_1/backward_lstm_2/strided_slice_6_grad/Shape': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/gradients/bidirectional_1/backward_lstm_2/strided_slice_6_grad/Shape': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/zeros_24/shape_as_tensor': [seq_length_symbol, vocab_size_symbol],
                  'training/RMSprop/zeros_24/shape_as_tensor': [seq_length_symbol, vocab_size_symbol],
                  'training/RMSprop/gradients/bidirectional_1/forward_lstm_2/strided_slice_11_grad/Shape': [4 * vocab_size_symbol],
                  'training/RMSprop/gradients/bidirectional_1/forward_lstm_2/strided_slice_11_grad/Shape': [4 * vocab_size_symbol],
                  'training/RMSprop/gradients/bidirectional_1/forward_lstm_2/strided_slice_7_grad/Shape': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/gradients/bidirectional_1/forward_lstm_2/strided_slice_7_grad/Shape': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/gradients/bidirectional_1/backward_lstm_2/strided_slice_2_grad/Shape': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/gradients/bidirectional_1/backward_lstm_2/strided_slice_2_grad/Shape': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/gradients/time_distributed_1/AttnWrapper/while/add_13_grad/Shape_1': [vocab_size_symbol],
                  'training/RMSprop/gradients/time_distributed_1/AttnWrapper/while/add_13_grad/Shape_1': [vocab_size_symbol],
                  'training/RMSprop/gradients/BLSTM1/forward_lstm_1/strided_slice_1_grad/Shape': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/gradients/BLSTM1/forward_lstm_1/strided_slice_1_grad/Shape': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/zeros_16/shape_as_tensor': [seq_length_symbol, seq_length_symbol],
                  'training/RMSprop/zeros_16/shape_as_tensor': [seq_length_symbol, seq_length_symbol],
                  'training/RMSprop/gradients/BLSTM1/backward_lstm_1/strided_slice_grad/Shape': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/gradients/BLSTM1/backward_lstm_1/strided_slice_grad/Shape': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/gradients/bidirectional_1/backward_lstm_2/strided_slice_11_grad/Shape': [4 * vocab_size_symbol],
                  'training/RMSprop/gradients/bidirectional_1/backward_lstm_2/strided_slice_11_grad/Shape': [4 * vocab_size_symbol],
                  'training/RMSprop/gradients/bidirectional_1/backward_lstm_2/strided_slice_7_grad/Shape': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/gradients/bidirectional_1/backward_lstm_2/strided_slice_7_grad/Shape': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/zeros_5/shape_as_tensor': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/zeros_5/shape_as_tensor': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/gradients/BLSTM1/forward_lstm_1/strided_slice_8_grad/Shape': [4 * vocab_size_symbol],
                  'training/RMSprop/gradients/BLSTM1/forward_lstm_1/strided_slice_8_grad/Shape': [4 * vocab_size_symbol],
                  'training/RMSprop/gradients/BLSTM1/forward_lstm_1/strided_slice_4_grad/Shape': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/gradients/BLSTM1/forward_lstm_1/strided_slice_4_grad/Shape': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/zeros_27/shape_as_tensor': [vocab_size_symbol, seq_length_symbol],
                  'training/RMSprop/zeros_27/shape_as_tensor': [vocab_size_symbol, seq_length_symbol],
                  'training/RMSprop/gradients/BLSTM1/forward_lstm_1/strided_slice_grad/Shape': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/gradients/BLSTM1/forward_lstm_1/strided_slice_grad/Shape': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/gradients/bidirectional_1/backward_lstm_2/strided_slice_8_grad/Shape': [4 * vocab_size_symbol],
                  'training/RMSprop/gradients/bidirectional_1/backward_lstm_2/strided_slice_8_grad/Shape': [4 * vocab_size_symbol],
                  'training/RMSprop/gradients/bidirectional_1/backward_lstm_2/strided_slice_4_grad/Shape': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/gradients/bidirectional_1/backward_lstm_2/strided_slice_4_grad/Shape': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/gradients/BLSTM1/forward_lstm_1/strided_slice_3_grad/Shape': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/gradients/BLSTM1/forward_lstm_1/strided_slice_3_grad/Shape': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/zeros_8/shape_as_tensor': [seq_length_symbol, seq_length_symbol],
                  'training/RMSprop/zeros_8/shape_as_tensor': [seq_length_symbol, seq_length_symbol],
                  'training/RMSprop/zeros_17/shape_as_tensor': [vocab_size_symbol, seq_length_symbol],
                  'training/RMSprop/zeros_17/shape_as_tensor': [vocab_size_symbol, seq_length_symbol],
                  'training/RMSprop/gradients/time_distributed_1/AttnWrapper/while/add_6_grad/Shape_1': [seq_length_symbol],
                  'training/RMSprop/gradients/time_distributed_1/AttnWrapper/while/add_6_grad/Shape_1': [seq_length_symbol],
                  'training/RMSprop/gradients/BLSTM1/backward_lstm_1/strided_slice_9_grad/Shape': [4 * vocab_size_symbol],
                  'training/RMSprop/gradients/BLSTM1/backward_lstm_1/strided_slice_9_grad/Shape': [4 * vocab_size_symbol],
                  'training/RMSprop/gradients/BLSTM1/backward_lstm_1/strided_slice_5_grad/Shape': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/gradients/BLSTM1/backward_lstm_1/strided_slice_5_grad/Shape': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/gradients/bidirectional_1/backward_lstm_2/strided_slice_1_grad/Shape': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/gradients/bidirectional_1/backward_lstm_2/strided_slice_1_grad/Shape': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/gradients/time_distributed_1/embedding_1/embedding_lookup_grad/Shape': [seq_length_symbol, vocab_size_symbol],
                  'training/RMSprop/gradients/time_distributed_1/embedding_1/embedding_lookup_grad/Shape': [seq_length_symbol, vocab_size_symbol],
                  'training/RMSprop/gradients/time_distributed_1/AttnWrapper/while/add_9_grad/Shape_1': [seq_length_symbol],
                  'training/RMSprop/gradients/time_distributed_1/AttnWrapper/while/add_9_grad/Shape_1': [seq_length_symbol],
                  'training/RMSprop/gradients/bidirectional_1/forward_lstm_2/strided_slice_9_grad/Shape': [4 * vocab_size_symbol],
                  'training/RMSprop/gradients/bidirectional_1/forward_lstm_2/strided_slice_9_grad/Shape': [4 * vocab_size_symbol],
                  'training/RMSprop/gradients/bidirectional_1/forward_lstm_2/strided_slice_5_grad/Shape': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/gradients/bidirectional_1/forward_lstm_2/strided_slice_5_grad/Shape': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/zeros_28/shape_as_tensor': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/zeros_28/shape_as_tensor': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/gradients/BLSTM1/backward_lstm_1/strided_slice_3_grad/Shape': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/gradients/BLSTM1/backward_lstm_1/strided_slice_3_grad/Shape': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/zeros_29/shape_as_tensor': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/zeros_29/shape_as_tensor': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/gradients/bidirectional_1/forward_lstm_2/strided_slice_1_grad/Shape': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/gradients/bidirectional_1/forward_lstm_2/strided_slice_1_grad/Shape': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/gradients/BLSTM1/forward_lstm_1/strided_slice_10_grad/Shape': [4 * vocab_size_symbol],
                  'training/RMSprop/gradients/BLSTM1/forward_lstm_1/strided_slice_10_grad/Shape': [4 * vocab_size_symbol],
                  'training/RMSprop/gradients/AttnWrapper_1/while/add_14_grad/Shape_1': [vocab_size_symbol],
                  'training/RMSprop/gradients/AttnWrapper_1/while/add_14_grad/Shape_1': [vocab_size_symbol],
                  'training/RMSprop/zeros_9/shape_as_tensor': [vocab_size_symbol, seq_length_symbol],
                  'training/RMSprop/zeros_9/shape_as_tensor': [vocab_size_symbol, seq_length_symbol],
                  'training/RMSprop/gradients/time_distributed_1/AttnWrapper/while/Reshape_1_grad/Shape': [seq_length_symbol, seq_length_symbol],
                  'training/RMSprop/gradients/time_distributed_1/AttnWrapper/while/Reshape_1_grad/Shape': [seq_length_symbol, seq_length_symbol],
                  'training/RMSprop/gradients/BLSTM1/forward_lstm_1/strided_slice_11_grad/Shape': [4 * vocab_size_symbol],
                  'training/RMSprop/gradients/BLSTM1/forward_lstm_1/strided_slice_11_grad/Shape': [4 * vocab_size_symbol],
                  'training/RMSprop/gradients/BLSTM1/forward_lstm_1/strided_slice_7_grad/Shape': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/gradients/BLSTM1/forward_lstm_1/strided_slice_7_grad/Shape': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/gradients/time_distributed_1/AttnWrapper/while/Tile_grad/stack/Const': [1, seq_length_symbol,  1],
                  'training/RMSprop/gradients/time_distributed_1/AttnWrapper/while/Tile_grad/stack/Const': [1, seq_length_symbol,  1],
                  'training/RMSprop/gradients/time_distributed_1/AttnWrapper/while/Tile_1_grad/stack/Const': [1, seq_length_symbol,  1],
                  'training/RMSprop/gradients/time_distributed_1/AttnWrapper/while/Tile_1_grad/stack/Const': [1, seq_length_symbol,  1],
                  'training/RMSprop/zeros_31/shape_as_tensor': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/zeros_31/shape_as_tensor': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/gradients/BLSTM1/forward_lstm_1/strided_slice_6_grad/Shape': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/gradients/BLSTM1/forward_lstm_1/strided_slice_6_grad/Shape': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/zeros_19/shape_as_tensor': [vocab_size_symbol, seq_length_symbol],
                  'training/RMSprop/zeros_19/shape_as_tensor': [vocab_size_symbol, seq_length_symbol],
                  'training/RMSprop/gradients/BLSTM1/backward_lstm_1/strided_slice_10_grad/Shape': [4 * vocab_size_symbol],
                  'training/RMSprop/gradients/BLSTM1/backward_lstm_1/strided_slice_10_grad/Shape': [4 * vocab_size_symbol],
                  'training/RMSprop/zeros_20/shape_as_tensor': [seq_length_symbol, seq_length_symbol],
                  'training/RMSprop/zeros_20/shape_as_tensor': [seq_length_symbol, seq_length_symbol],
                  'training/RMSprop/gradients/BLSTM1/backward_lstm_1/strided_slice_6_grad/Shape': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/gradients/BLSTM1/backward_lstm_1/strided_slice_6_grad/Shape': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/gradients/BLSTM1/forward_lstm_1/strided_slice_2_grad/Shape': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/gradients/BLSTM1/forward_lstm_1/strided_slice_2_grad/Shape': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/gradients/bidirectional_1/backward_lstm_2/strided_slice_grad/Shape': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/gradients/bidirectional_1/backward_lstm_2/strided_slice_grad/Shape': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/gradients/time_distributed_1/AttnWrapper/while/ExpandDims_1_grad/Shape': [seq_length_symbol],
                  'training/RMSprop/gradients/time_distributed_1/AttnWrapper/while/ExpandDims_1_grad/Shape': [seq_length_symbol],
                  'training/RMSprop/gradients/BLSTM1/backward_lstm_1/strided_slice_2_grad/Shape': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/gradients/BLSTM1/backward_lstm_1/strided_slice_2_grad/Shape': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/gradients/bidirectional_1/forward_lstm_2/strided_slice_8_grad/Shape': [4 * vocab_size_symbol],
                  'training/RMSprop/gradients/bidirectional_1/forward_lstm_2/strided_slice_8_grad/Shape': [4 * vocab_size_symbol],
                  'training/RMSprop/zeros/shape_as_tensor': [seq_length_symbol, vocab_size_symbol],
                  'training/RMSprop/zeros/shape_as_tensor': [seq_length_symbol, vocab_size_symbol],
                  'training/RMSprop/zeros_11/shape_as_tensor': [vocab_size_symbol, seq_length_symbol],
                  'training/RMSprop/zeros_11/shape_as_tensor': [vocab_size_symbol, seq_length_symbol],
                  'training/RMSprop/gradients/bidirectional_1/forward_lstm_2/strided_slice_4_grad/Shape': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/gradients/bidirectional_1/forward_lstm_2/strided_slice_4_grad/Shape': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/gradients/bidirectional_1/forward_lstm_2/strided_slice_3_grad/Shape': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/gradients/bidirectional_1/forward_lstm_2/strided_slice_3_grad/Shape': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/zeros_1/shape_as_tensor': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/zeros_1/shape_as_tensor': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/gradients/bidirectional_1/forward_lstm_2/strided_slice_grad/Shape': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/gradients/bidirectional_1/forward_lstm_2/strided_slice_grad/Shape': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/zeros_32/shape_as_tensor': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/zeros_32/shape_as_tensor': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/zeros_12/shape_as_tensor': [seq_length_symbol, seq_length_symbol],
                  'training/RMSprop/zeros_12/shape_as_tensor': [seq_length_symbol, seq_length_symbol],
                  'training/RMSprop/zeros_13/shape_as_tensor': [vocab_size_symbol, seq_length_symbol],
                  'training/RMSprop/zeros_13/shape_as_tensor': [vocab_size_symbol, seq_length_symbol],
                  'training/RMSprop/gradients/BLSTM1/backward_lstm_1/strided_slice_11_grad/Shape': [4 * vocab_size_symbol],
                  'training/RMSprop/gradients/BLSTM1/backward_lstm_1/strided_slice_11_grad/Shape': [4 * vocab_size_symbol],
                  'training/RMSprop/gradients/BLSTM1/backward_lstm_1/strided_slice_7_grad/Shape': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/gradients/BLSTM1/backward_lstm_1/strided_slice_7_grad/Shape': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/gradients/bidirectional_1/forward_lstm_2/strided_slice_2_grad/Shape': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/gradients/bidirectional_1/forward_lstm_2/strided_slice_2_grad/Shape': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/gradients/BLSTM1/forward_lstm_1/strided_slice_9_grad/Shape': [4 * vocab_size_symbol],
                  'training/RMSprop/gradients/BLSTM1/forward_lstm_1/strided_slice_9_grad/Shape': [4 * vocab_size_symbol],
                  'training/RMSprop/gradients/BLSTM1/forward_lstm_1/strided_slice_5_grad/Shape': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/gradients/BLSTM1/forward_lstm_1/strided_slice_5_grad/Shape': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/zeros_2/shape_as_tensor': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/zeros_2/shape_as_tensor': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/gradients/BLSTM1/backward_lstm_1/strided_slice_8_grad/Shape': [4 * vocab_size_symbol],
                  'training/RMSprop/gradients/BLSTM1/backward_lstm_1/strided_slice_8_grad/Shape': [4 * vocab_size_symbol],
                  'training/RMSprop/gradients/bidirectional_1/backward_lstm_2/strided_slice_9_grad/Shape': [4 * vocab_size_symbol],
                  'training/RMSprop/gradients/bidirectional_1/backward_lstm_2/strided_slice_9_grad/Shape': [4 * vocab_size_symbol],
                  'training/RMSprop/gradients/bidirectional_1/backward_lstm_2/strided_slice_3_grad/Shape': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/gradients/bidirectional_1/backward_lstm_2/strided_slice_3_grad/Shape': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/gradients/bidirectional_1/backward_lstm_2/strided_slice_5_grad/Shape': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/gradients/bidirectional_1/backward_lstm_2/strided_slice_5_grad/Shape': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/zeros_15/shape_as_tensor': [vocab_size_symbol, seq_length_symbol],
                  'training/RMSprop/zeros_15/shape_as_tensor': [vocab_size_symbol, seq_length_symbol],
                  'training/RMSprop/gradients/time_distributed_1/AttnWrapper/while/Reshape_4_grad/Shape': [seq_length_symbol,  1],
                  'training/RMSprop/gradients/time_distributed_1/AttnWrapper/while/Reshape_4_grad/Shape': [seq_length_symbol,  1],
                  'training/RMSprop/gradients/bidirectional_1/forward_lstm_2/strided_slice_10_grad/Shape': [4 * vocab_size_symbol],
                  'training/RMSprop/gradients/bidirectional_1/forward_lstm_2/strided_slice_10_grad/Shape': [4 * vocab_size_symbol],
                  'training/RMSprop/gradients/bidirectional_1/forward_lstm_2/strided_slice_6_grad/Shape': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/gradients/bidirectional_1/forward_lstm_2/strided_slice_6_grad/Shape': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/gradients/BLSTM1/backward_lstm_1/strided_slice_4_grad/Shape': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/gradients/BLSTM1/backward_lstm_1/strided_slice_4_grad/Shape': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/gradients/time_distributed_1/AttnWrapper/while/add_3_grad/Shape_1': [seq_length_symbol],
                  'training/RMSprop/gradients/time_distributed_1/AttnWrapper/while/add_3_grad/Shape_1': [seq_length_symbol],
                  'training/RMSprop/zeros_4/shape_as_tensor': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/zeros_4/shape_as_tensor': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/gradients/BLSTM1/backward_lstm_1/strided_slice_1_grad/Shape': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/gradients/BLSTM1/backward_lstm_1/strided_slice_1_grad/Shape': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/gradients/AttnWrapper_1/while/add_14_grad/Shape_1': [vocab_size_symbol],
                  'training/RMSprop/gradients/BLSTM1/backward_lstm_1/strided_slice_10_grad/Shape': [4 * vocab_size_symbol],
                  'training/RMSprop/gradients/BLSTM1/backward_lstm_1/strided_slice_11_grad/Shape': [4 * vocab_size_symbol],
                  'training/RMSprop/gradients/BLSTM1/backward_lstm_1/strided_slice_1_grad/Shape': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/gradients/BLSTM1/backward_lstm_1/strided_slice_2_grad/Shape': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/gradients/BLSTM1/backward_lstm_1/strided_slice_3_grad/Shape': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/gradients/BLSTM1/backward_lstm_1/strided_slice_4_grad/Shape': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/gradients/BLSTM1/backward_lstm_1/strided_slice_5_grad/Shape': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/gradients/BLSTM1/backward_lstm_1/strided_slice_6_grad/Shape': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/gradients/BLSTM1/backward_lstm_1/strided_slice_7_grad/Shape': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/gradients/BLSTM1/backward_lstm_1/strided_slice_8_grad/Shape': [4 * vocab_size_symbol],
                  'training/RMSprop/gradients/BLSTM1/backward_lstm_1/strided_slice_9_grad/Shape': [4 * vocab_size_symbol],
                  'training/RMSprop/gradients/BLSTM1/backward_lstm_1/strided_slice_grad/Shape': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/gradients/BLSTM1/forward_lstm_1/strided_slice_10_grad/Shape': [4 * vocab_size_symbol],
                  'training/RMSprop/gradients/BLSTM1/forward_lstm_1/strided_slice_11_grad/Shape': [4 * vocab_size_symbol],
                  'training/RMSprop/gradients/BLSTM1/forward_lstm_1/strided_slice_1_grad/Shape': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/gradients/BLSTM1/forward_lstm_1/strided_slice_2_grad/Shape': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/gradients/BLSTM1/forward_lstm_1/strided_slice_3_grad/Shape': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/gradients/BLSTM1/forward_lstm_1/strided_slice_4_grad/Shape': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/gradients/BLSTM1/forward_lstm_1/strided_slice_5_grad/Shape': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/gradients/BLSTM1/forward_lstm_1/strided_slice_6_grad/Shape': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/gradients/BLSTM1/forward_lstm_1/strided_slice_7_grad/Shape': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/gradients/BLSTM1/forward_lstm_1/strided_slice_8_grad/Shape': [4 * vocab_size_symbol],
                  'training/RMSprop/gradients/BLSTM1/forward_lstm_1/strided_slice_9_grad/Shape': [4 * vocab_size_symbol],
                  'training/RMSprop/gradients/BLSTM1/forward_lstm_1/strided_slice_grad/Shape': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/gradients/bidirectional_1/backward_lstm_2/strided_slice_10_grad/Shape': [4 * vocab_size_symbol],
                  'training/RMSprop/gradients/bidirectional_1/backward_lstm_2/strided_slice_11_grad/Shape': [4 * vocab_size_symbol],
                  'training/RMSprop/gradients/bidirectional_1/backward_lstm_2/strided_slice_1_grad/Shape': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/gradients/bidirectional_1/backward_lstm_2/strided_slice_2_grad/Shape': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/gradients/bidirectional_1/backward_lstm_2/strided_slice_3_grad/Shape': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/gradients/bidirectional_1/backward_lstm_2/strided_slice_4_grad/Shape': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/gradients/bidirectional_1/backward_lstm_2/strided_slice_5_grad/Shape': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/gradients/bidirectional_1/backward_lstm_2/strided_slice_6_grad/Shape': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/gradients/bidirectional_1/backward_lstm_2/strided_slice_7_grad/Shape': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/gradients/bidirectional_1/backward_lstm_2/strided_slice_8_grad/Shape': [4 * vocab_size_symbol],
                  'training/RMSprop/gradients/bidirectional_1/backward_lstm_2/strided_slice_9_grad/Shape': [4 * vocab_size_symbol],
                  'training/RMSprop/gradients/bidirectional_1/backward_lstm_2/strided_slice_grad/Shape': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/gradients/bidirectional_1/forward_lstm_2/strided_slice_10_grad/Shape': [4 * vocab_size_symbol],
                  'training/RMSprop/gradients/bidirectional_1/forward_lstm_2/strided_slice_11_grad/Shape': [4 * vocab_size_symbol],
                  'training/RMSprop/gradients/bidirectional_1/forward_lstm_2/strided_slice_1_grad/Shape': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/gradients/bidirectional_1/forward_lstm_2/strided_slice_2_grad/Shape': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/gradients/bidirectional_1/forward_lstm_2/strided_slice_3_grad/Shape': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/gradients/bidirectional_1/forward_lstm_2/strided_slice_4_grad/Shape': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/gradients/bidirectional_1/forward_lstm_2/strided_slice_5_grad/Shape': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/gradients/bidirectional_1/forward_lstm_2/strided_slice_6_grad/Shape': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/gradients/bidirectional_1/forward_lstm_2/strided_slice_7_grad/Shape': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/gradients/bidirectional_1/forward_lstm_2/strided_slice_8_grad/Shape': [4 * vocab_size_symbol],
                  'training/RMSprop/gradients/bidirectional_1/forward_lstm_2/strided_slice_9_grad/Shape': [4 * vocab_size_symbol],
                  'training/RMSprop/gradients/bidirectional_1/forward_lstm_2/strided_slice_grad/Shape': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/gradients/time_distributed_1/AttnWrapper/while/ExpandDims_1_grad/Shape': [seq_length_symbol],
                  'training/RMSprop/gradients/time_distributed_1/AttnWrapper/while/Reshape_1_grad/Shape': [seq_length_symbol, seq_length_symbol],
                  'training/RMSprop/gradients/time_distributed_1/AttnWrapper/while/Reshape_4_grad/Shape': [seq_length_symbol,  1],
                  'training/RMSprop/gradients/time_distributed_1/AttnWrapper/while/Tile_1_grad/stack/Const': [1, seq_length_symbol,  1],
                  'training/RMSprop/gradients/time_distributed_1/AttnWrapper/while/Tile_grad/stack/Const': [1, seq_length_symbol,  1],
                  'training/RMSprop/gradients/time_distributed_1/AttnWrapper/while/add_13_grad/Shape_1': [vocab_size_symbol],
                  'training/RMSprop/gradients/time_distributed_1/AttnWrapper/while/add_3_grad/Shape_1': [seq_length_symbol],
                  'training/RMSprop/gradients/time_distributed_1/AttnWrapper/while/add_6_grad/Shape_1': [seq_length_symbol],
                  'training/RMSprop/gradients/time_distributed_1/AttnWrapper/while/add_9_grad/Shape_1': [seq_length_symbol],
                  'training/RMSprop/gradients/time_distributed_1/embedding_1/embedding_lookup_grad/Shape': [seq_length_symbol, vocab_size_symbol],
                  'training/RMSprop/zeros/shape_as_tensor': [seq_length_symbol, vocab_size_symbol],
                  'training/RMSprop/zeros_1/shape_as_tensor': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/zeros_11/shape_as_tensor': [vocab_size_symbol, seq_length_symbol],
                  'training/RMSprop/zeros_12/shape_as_tensor': [seq_length_symbol, seq_length_symbol],
                  'training/RMSprop/zeros_13/shape_as_tensor': [vocab_size_symbol, seq_length_symbol],
                  'training/RMSprop/zeros_15/shape_as_tensor': [vocab_size_symbol, seq_length_symbol],
                  'training/RMSprop/zeros_16/shape_as_tensor': [seq_length_symbol, seq_length_symbol],
                  'training/RMSprop/zeros_17/shape_as_tensor': [vocab_size_symbol, seq_length_symbol],
                  'training/RMSprop/zeros_19/shape_as_tensor': [vocab_size_symbol, seq_length_symbol],
                  'training/RMSprop/zeros_2/shape_as_tensor': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/zeros_20/shape_as_tensor': [seq_length_symbol, seq_length_symbol],
                  'training/RMSprop/zeros_24/shape_as_tensor': [seq_length_symbol, vocab_size_symbol],
                  'training/RMSprop/zeros_27/shape_as_tensor': [vocab_size_symbol, seq_length_symbol],
                  'training/RMSprop/zeros_28/shape_as_tensor': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/zeros_29/shape_as_tensor': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/zeros_31/shape_as_tensor': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/zeros_32/shape_as_tensor': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/zeros_4/shape_as_tensor': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/zeros_5/shape_as_tensor': [vocab_size_symbol, 4 * vocab_size_symbol],
                  'training/RMSprop/zeros_8/shape_as_tensor': [seq_length_symbol, seq_length_symbol],
                  'training/RMSprop/zeros_9/shape_as_tensor': [vocab_size_symbol, seq_length_symbol],
                 }

    graph.bindConstantValues(const_dict)

    bind_dict = {
                  'AttnWrapper_sample_weights': [batch_size_symbol],
                  'AttnWrapper_sample_weights_1': [batch_size_symbol],
                  'AttnWrapper_target': [batch_size_symbol, seq_length_symbol, vocab_size_symbol],
                  'AttnWrapper_target_1': [batch_size_symbol, seq_length_symbol, vocab_size_symbol],
                  'Sum_sample_weights': [batch_size_symbol],
                  'Sum_target': [batch_size_symbol, vocab_size_symbol],
                  'res_input': [batch_size_symbol, seq_length_symbol],
                  'seqOut_sample_weights': [batch_size_symbol],
                  'seqOut_sample_weights_1': [batch_size_symbol],
                  'seqOut_target': [batch_size_symbol, seq_length_symbol], # Second dim might be 1?
                  'seqOut_target_1': [batch_size_symbol, seq_length_symbol], # Second dim might be 1?
                  'sequence_input': [batch_size_symbol, 1, seq_length_symbol],
                  # 'time_distributed_2_sample_weights': [batch_size_symbol],
                  # 'time_distributed_2_target': [batch_size_symbol, None, seq_length_symbol, vocab_size_symbol],
                }

    # Variable and constant sizes are pretty predictable...
    important_ops = [graph.getVariables()]
    important_ops.extend(graph.getConstants())
    for op in graph.getVariables():
        out_shape = []
        include_me = False
        for dim in op.outputs[0].shape.dims:
            if dim.value % base_vocab_size == 0:
                mult_dim = dim.value // base_vocab_size
                out_shape.append(mult_dim * vocab_size_symbol)
                include_me = True
            elif dim.value == base_sequence_length:
                out_shape.append(seq_length_symbol)
                include_me = True
            else:
                out_shape.append(dim.value)
        if include_me:
            bind_dict[op.name] = out_shape

    graph.bindShapesAndPropagate(bind_dict, make_symbolic=True,
                                 warn_if_ill_defined=True)
    # HAX: Manually set a couple ops:
    max_op = graph.opsByName['training/RMSprop/gradients/att2/Sum_grad/Maximum']
    max_op.outputs[0].setValue([batch_size_symbol, 1, vocab_size_symbol])
    max_op = graph.opsByName['training/RMSprop/gradients/AttnWrapper_1/Sum_grad/Maximum']
    max_op.outputs[0].setValue([batch_size_symbol, 1, 1])
    max_op = graph.opsByName['training/RMSprop/gradients/bidirectional_1/Sum_grad/Maximum']
    max_op.outputs[0].setValue([batch_size_symbol, 1, 1])
    max_op = graph.opsByName['training/RMSprop/gradients/bidirectional_1/Sum_1_grad/Maximum']
    max_op.outputs[0].setValue([batch_size_symbol, 1, 1])
    max_op = graph.opsByName['training/RMSprop/gradients/AttnWrapper_1/while/Sum_grad/Maximum']
    max_op.outputs[0].setValue([batch_size_symbol, 1, 1])
    max_op = graph.opsByName['training/RMSprop/gradients/loss_3/seqOut_loss/Mean_grad/Maximum']
    max_op.outputs[0].setValue([batch_size_symbol, 1])
    max_op = graph.opsByName['training/RMSprop/gradients/loss_3/seqOut_loss/Mean_1_grad/Maximum']
    max_op.outputs[0].setValue([batch_size_symbol])
    max_op = graph.opsByName['training/RMSprop/gradients/time_distributed_1/Sum/Sum_grad/Maximum']
    max_op.outputs[0].setValue([batch_size_symbol, 1, vocab_size_symbol])
    max_op = graph.opsByName['training/RMSprop/gradients/time_distributed_1/BLSTM1/Sum_grad/Maximum']
    max_op.outputs[0].setValue([batch_size_symbol, 1, 1])
    max_op = graph.opsByName['training/RMSprop/gradients/time_distributed_1/BLSTM1/Sum_1_grad/Maximum']
    max_op.outputs[0].setValue([batch_size_symbol, 1, 1])
    max_op = graph.opsByName['training/RMSprop/gradients/time_distributed_1/AttnWrapper/Sum_grad/Maximum']
    max_op.outputs[0].setValue([batch_size_symbol, 1, 1])
    max_op = graph.opsByName['training/RMSprop/gradients/time_distributed_1/AttnWrapper/while/Sum_grad/Maximum']
    max_op.outputs[0].setValue([batch_size_symbol, 1, 1])

    sum_op = graph.opsByName['training/RMSprop/gradients/bidirectional_1/add_16_grad/Sum_1']
    dim_to_hack = sum_op.outputs[0].shape.dims[1]
    assert dim_to_hack.symbol is not None
    assert dim_to_hack.value is None
    dim_to_hack._value = 1
    dim_to_hack._symbol = None

    graph.bindShapesAndPropagate(bind_dict, make_symbolic=True, verbose=True,
                                 warn_if_ill_defined=True)

    print('Cleaned graph:\n{}'.format(graph))

    try:
        alg_flops = graph.calcAlgFlops()
        print(alg_flops)
    except Exception as err:
        print("NOT ABLE TO CALCALGFLOPS YET...")

    sym_subs = {
                vocab_size_symbol: 21,
                seq_length_symbol: 381,
                batch_size_symbol: 6000,
               }
    # TODO(joel): Not sure if any of these are correct!
    sym_str_subs = {
                     'graph::iters': 1,
                     'BLSTM1/while/LoopCond_block::iters': seq_length_symbol,
                     'BLSTM1/while_1/LoopCond_block::iters': seq_length_symbol,
                     'time_distributed_1/BLSTM1/while/LoopCond_block::iters': seq_length_symbol,
                     'time_distributed_1/BLSTM1/while_1/LoopCond_block::iters': seq_length_symbol,
                     'time_distributed_1/BLSTM1_1/while/LoopCond_block::iters': seq_length_symbol,
                     'time_distributed_1/BLSTM1_1/while_1/LoopCond_block::iters': seq_length_symbol,
                     'AttnWrapper/while/LoopCond_block::iters': seq_length_symbol,
                     'AttnWrapper_1/while/LoopCond_block::iters': seq_length_symbol,
                     'bidirectional_1/while/LoopCond_block::iters': seq_length_symbol,
                     'bidirectional_1/while_1/LoopCond_block::iters': seq_length_symbol,
                     'time_distributed_1/AttnWrapper/while/LoopCond_block::iters': seq_length_symbol,
                     'time_distributed_1/AttnWrapper_1/while/LoopCond_block::iters': seq_length_symbol,
                     'training/RMSprop/gradients/b_count_10_block::iters': seq_length_symbol,
                     'training/RMSprop/gradients/b_count_18_block::iters': seq_length_symbol,
                     'training/RMSprop/gradients/b_count_22_block::iters': seq_length_symbol,
                     'training/RMSprop/gradients/b_count_6_block::iters': seq_length_symbol,
                     'training/RMSprop/gradients/b_count_14_block::iters': seq_length_symbol,
                     'training/RMSprop/gradients/b_count_2_block::iters': seq_length_symbol,
                    }
    var_refs_table = {}
    for var_name, sub_val in sym_str_subs.items():
        var_ref = utils.getIntSymbolFromString(var_name)
        assert var_name not in sym_subs.keys()
        sym_subs[var_ref] = sub_val
        var_refs_table[var_name] = var_ref

#    print(graph.calcMinimalFootprint(symbol_subs=sym_subs))

    print('\n\nAlgorithmic Bytes:')
    for op in graph.opsByName.values():
        alg_bytes = op.calcAlgBytes()
        if isinstance(alg_bytes, sympy.Expr):
            resolved_alg_bytes = alg_bytes.subs(sym_subs)
        else:
            resolved_alg_bytes = alg_bytes
        print('{}\t{}\t{}'.format(op.name, alg_bytes, resolved_alg_bytes))

    return graph, sym_subs


if __name__ == "__main__":
    test_tf_han_model()

