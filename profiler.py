import tensorflow as tf
import numpy as np

# FlopcoKeras install is broken on pip 20.1 due to use of pip internals in setup.py
# https://www.gitmemory.com/issue/Aloomaio/alooma-python/120/629762153
# from flopco_keras import FlopCoKeras

ProfileOptionBuilder = tf.compat.v1.profiler.ProfileOptionBuilder


def calculate_flops_from_frozen_graph(model_path, save_path, shape=None):
    tf.compat.v1.reset_default_graph()
    with tf.compat.v1.Session() as session:
        with tf.compat.v1.gfile.GFile(model_path, 'rb') as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
            session.graph.as_default()

            input_name = graph_def.node[0].name
            input_shape = (1, ) + shape
            if shape is not None:
                tf_input = tf.compat.v1.placeholder(shape=input_shape, dtype="float32", name=input_name)
                tf.import_graph_def(graph_def, input_map={input_name: tf_input})
            else:
                tf.import_graph_def(graph_def)

            run_meta = tf.compat.v1.RunMetadata()
            opts = build_profile_options(save_path)
            flops_stats = tf.compat.v1.profiler.profile(tf.compat.v1.get_default_graph(),
                                                        options=opts,
                                                        run_meta=run_meta)

            with open(build_profile_flops_file_path(save_path), "a") as f_stat:
                f_stat.write("FLOPS_TOTAL = " + str(flops_stats.total_float_ops) + "\n")

            print('FLOPS total = ', flops_stats.total_float_ops)


# Mentioned here: https://github.com/tensorflow/tensorflow/issues/32809
# But still has issues with Incomplete Shape
def calculate_flops_from_h5(model_h5_path, save_path,):
    session = tf.compat.v1.keras.backend.get_session()
    graph = tf.compat.v1.keras.backend.get_session().graph

    with graph.as_default():
        with session.as_default():
            run_meta = tf.compat.v1.RunMetadata()
            model = tf.compat.v1.keras.models.load_model(model_h5_path)
            opts = build_profile_options(save_path)

            # We use the Keras session graph in the call to the profiler.
            flops = tf.compat.v1.profiler.profile(graph=tf.compat.v1.keras.backend.get_session(),
                                                  run_meta=run_meta,
                                                  options=opts)

            with open(build_profile_flops_file_path(save_path), "a") as f_stat:
                f_stat.write("FLOPS_TOTAL = " + str(flops.total_float_ops) + "\n")

            print('FLOPS total = ', flops.total_float_ops)


# https://github.com/tensorflow/tensorflow/issues/20960
# But still has issues with Incomplete Shape
def calculate_flops_with_session_meta(model_path, save_path, shape=None):
    tf.compat.v1.reset_default_graph()
    with tf.compat.v1.Session(graph=tf.compat.v1.keras.backend.get_session().graph) as session:
        model = tf.compat.v1.keras.models.load_model(model_path, compile=True)
        meta = tf.compat.v1.RunMetadata()
        x = np.random.random([1, shape[0], shape[1], shape[2]])
        session.run(model.output,
                    feed_dict={model.inputs[0]: x},
                    options=tf.compat.v1.RunOptions(trace_level=tf.compat.v1.RunOptions.FULL_TRACE),
                    run_metadata=meta)

        graph = tf.compat.v1.keras.backend.get_session().graph

        opts_variables = build_profile_options(save_path, flops=False)
        tf.compat.v1.profiler.profile(graph, options=opts_variables, run_meta=meta)

        opts_flops = build_profile_options(save_path, flops=True)
        flops_stats = tf.compat.v1.profiler.profile(graph, options=opts_flops, run_meta=meta)

        with open(build_profile_flops_file_path(save_path), "a") as f_stat:
            f_stat.write("FLOPS_TOTAL = " + str(flops_stats.total_float_ops) + "\n")

        print('FLOPS total = ', flops_stats.total_float_ops)


def build_profile_flops_file_path(save_path):
    return save_path + "profiler_flops.txt"


def build_profile_variables_file_path(save_path):
    return save_path + "profiler_variables.txt"


def build_profile_options(save_path, flops=True):
    logs_file = build_profile_variables_file_path(save_path)
    if flops:
        logs_file = build_profile_flops_file_path(save_path)

    select = ['params', 'float_ops', 'input_shapes', 'occurrence', 'op_types', 'tensor_value', 'output_bytes', 'bytes', 'peak_bytes', 'residual_bytes']

    builder = ProfileOptionBuilder.trainable_variables_parameter()
    if flops:
        builder = ProfileOptionBuilder.float_operation()

    return ProfileOptionBuilder(builder) \
        .with_file_output(logs_file) \
        .account_displayed_op_only(False) \
        .order_by("occurrence") \
        .select(select).build()


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

# FlopcoKeras install is broken on pip 20.1 due to use of pip internals in setup.py
# https://www.gitmemory.com/issue/Aloomaio/alooma-python/120/629762153
# def calculate_flops_flopco(model):
#     stats = FlopCoKeras(model)
#     print(f"FLOPs: {stats.total_flops}")
#     print(f"MACs: {stats.total_macs}")
#     print(f"Relative FLOPs: {stats.relative_flops}")
