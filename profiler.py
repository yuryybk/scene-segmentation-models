import tensorflow as tf


def calculate_flops(model_path, shape=None):
    tf.compat.v1.reset_default_graph()
    with tf.compat.v1.Session() as session:
        with tf.compat.v1.gfile.GFile(model_path, 'rb') as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
            session.graph.as_default()

            input_name = graph_def.node[0].name
            input_shape = (1, ) + shape
            if shape is not None:
                tf_input = tf.compat.v1.placeholder(shape=input_shape, dtype='float32', name=input_name)
                tf.import_graph_def(graph_def, name="import", input_map={input_name: tf_input})
            else:
                tf.import_graph_def(graph_def)

            run_meta = tf.compat.v1.RunMetadata()
            flops_stats = tf.compat.v1.profiler.profile(tf.compat.v1.get_default_graph(),
                                                        options=tf.compat.v1.profiler.ProfileOptionBuilder.float_operation(),
                                                        run_meta=run_meta)

            print('FLOPS total = ', flops_stats.total_float_ops)


def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.compat.v1.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.compat.v1.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = tf.compat.v1.graph_util.convert_variables_to_constants(
            session, input_graph_def, output_names, freeze_var_names)
        return frozen_graph
