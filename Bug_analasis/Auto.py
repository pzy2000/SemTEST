import os
import subprocess
import time
from itertools import combinations
import numpy as np
import tensorflow as tf
import sys
import torch
from onnx2torch import convert
import networkx as nx
from tqdm import tqdm

sys.path.append("../")
sys.path.append("../implementations")
from implementations.model2json import model_to_json, union_json, CoverageCalculator
from implementations.scripts.tools import utils
from implementations.scripts.prediction.custom_objects import custom_objects

folder_path = "/data1/pzy/MUTANTS/LEMON/ALL/lstm0-sinewave_origin-1/lstm0-sinewave_origin.h5"
envs = "CUDA_HOME=/usr/local/cuda-10 CUDA_ROOT=/usr/local/cuda-10 LD_LIBRARY_PATH=/usr/local/cuda-10/lib64:$LD_LIBRARY_PATH PATH=/usr/local/cuda-10/bin:$PATH"


def clear_cache():
    if os.path.exists("tmp"):
        subprocess.call("rm -rf tmp", shell=True)
    if os.path.exists("onnx"):
        subprocess.call("rm -rf onnx", shell=True)
    if not os.path.exists("tmp"):
        os.makedirs("tmp")
    if not os.path.exists("onnx"):
        os.makedirs("onnx")


clear_cache()
if os.path.exists("all_layer_info.json"):
    subprocess.call("rm -rf all_layer_info.json", shell=True)
if os.path.exists("output_coverage.xlsx"):
    subprocess.call("rm -rf output_coverage.xlsx", shell=True)
print("sleep 5s")
time.sleep(5)
print("awake")
if not os.path.exists("tmp"):
    os.makedirs("tmp")
if not os.path.exists("py"):
    os.makedirs("py")
if not os.path.exists("onnx"):
    os.makedirs("onnx")
f = open("FAILED.txt", "w")


def _init_input(input_shape):
    input_shape = list(input_shape)
    input_shape[0] = 10
    input_shape = tuple(input_shape)
    input = np.random.rand(*input_shape)
    return input


def check_inconsistency(inconsistency):
    if 10 <= inconsistency:  # 10 is the pre-defined threshold
        print(f"New inconsistency issue found!!!")
        return True
    return False


def generate_metrics_result(res_dict, predict_output, model_name):
    if res_dict is None:
        res_dict = {k: dict() for k in ["D_MAD"]}
    print("Generating Metrics Result")
    accumulative_incons = 0
    backends_pairs_num = 0
    # Compare results pair by pair
    for pair in combinations(predict_output.items(), 2):
        backends_pairs_num += 1
        backend1, backend2 = pair
        bk_name1, prediction1 = backend1
        bk_name2, prediction2 = backend2
        if prediction1.shape != prediction2.shape:
            # If cases happen when the shape of prediction is already inconsistent, return inconsistency as None to raise a warning
            return res_dict, None
        for metrics_name, metrics_result_dict in res_dict.items():
            metrics_func = utils.MetricsUtils.get_metrics_by_name(metrics_name)
            # metrics_results in list type
            if metrics_name == "D_MAD":
                y_test = np.ones_like(prediction1)
                metrics_results = metrics_func(prediction1, prediction2, y_test)
            else:
                metrics_results = metrics_func(prediction1, prediction2)
            print(f"Inconsistency between {bk_name1} and {bk_name2} is {sum(metrics_results)}")
            accumulative_incons += sum(metrics_results)
            for input_idx, delta in enumerate(metrics_results):
                delta_key = "{}_{}_{}_input{}".format(model_name, bk_name1, bk_name2, input_idx)
                metrics_result_dict[delta_key] = delta
    print(f"Accumulative Inconsistency: {accumulative_incons}")
    return res_dict, accumulative_incons


def remove_ndarray_entries(input_dict):
    if isinstance(input_dict, dict):
        keys_to_remove = []
        for key, value in input_dict.items():
            if isinstance(value, np.ndarray):
                # Mark the key for removal
                keys_to_remove.append(key)
            elif isinstance(value, (dict, list)):
                # Recurse into the sub-dictionary or list
                input_dict[key] = remove_ndarray_entries(value)

        for key in keys_to_remove:
            del input_dict[key]

        return input_dict
    elif isinstance(input_dict, list):
        return [remove_ndarray_entries(item) if isinstance(item, (dict, list)) else item for item in input_dict]


def build_graph(module, graph, parent_name=None):
    for name, child in module.named_children():
        layer_name = f"{parent_name}.{name}" if parent_name else name
        graph.add_node(layer_name)
        if parent_name:
            graph.add_edge(parent_name, layer_name)
        build_graph(child, graph, parent_name=layer_name)


def calc_inner_div_torch(model):
    graph = nx.DiGraph()
    build_graph(model, graph)
    # try:
    longest_path = nx.dag_longest_path(graph)
    return len(longest_path) / len(graph)


def calc_inner_div(model):
    graph = nx.DiGraph()
    for layer in model.layers:
        graph.add_node(layer.name)
        for inbound_node in layer._inbound_nodes:
            if inbound_node.inbound_layers:
                for parent_layer in inbound_node.inbound_layers:
                    graph.add_edge(parent_layer.name, layer.name)
    longest_path = nx.dag_longest_path(graph)
    return len(longest_path) / len(graph)


incon_dict = {}
result_dict = {}
# coverage_path_cargo = []
path_list = []
itera = 0
print("start union")
# Using os.walk to RECURSIVELY walk through the subdirectories
if os.path.isfile(folder_path):
    path_list.append((os.path.dirname(folder_path), [], [os.path.basename(folder_path)]))
    model_path = folder_path
    cur_path = model_to_json(model_path)
    union_json(cur_path, os.path.join("all_layer_info.json"))
else:
    for dirpath, dirnames, filenames in os.walk(folder_path):
        path_list.append((dirpath, dirnames, filenames))
        for filename in filenames:
            # print("iteration ", itera)
            if filename.endswith(".h5"):
                model_path = os.path.join(dirpath, filename)
                cur_path = model_to_json(model_path)
                union_json(cur_path, os.path.join("all_layer_info.json"))
print("finished union")
for dirpath, dirnames, filenames in tqdm(path_list):
    for filename in filenames:
        if filename.endswith(".h5"):
            model_path = os.path.join(dirpath, filename)
            cur_path = model_to_json(model_path)
            # union_json(cur_path, os.path.join("all_layer_info.json"))
            # try:
            itera += 1
            model = tf.keras.models.load_model(model_path, custom_objects=custom_objects())
            # model = tf.keras.models.load_model(model_path,)
            model_name = os.path.basename(filename)[:-3]
            model.save(f'tmp/{model_name}', save_format='tf')
            status = subprocess.call(
                f"{envs} python -u -m tf2onnx.convert --saved-model " + f'tmp/{model_name}' + " --output onnx/" + model_name + ".onnx",
                shell=True)
            print("model name", model_name, "status", False if status != 0 else True)
            onnx_model_path = "onnx/" + model_name + ".onnx"
            torch_model = convert(onnx_model_path)
            traced_model = torch.fx.symbolic_trace(torch_model)
            code = traced_model.code
            with open("py/" + model_name + '.py', 'w') as fp:
                fp.write(code)
            inpu = _init_input(model.input_shape)
            tf_input = tf.convert_to_tensor(inpu)
            torch_input = torch.tensor(inpu, dtype=torch.float32)
            tf_output = model(tf_input)
            torch_output = torch_model(torch_input)
            inner_diversity = calc_inner_div_torch(torch_model)
            # print("inner_diversity_torch", inner_diversity)
            # inner_diversity_tf = calc_inner_div(model)
            # print("inner_diversity_tf", inner_diversity_tf)
            _, accumulative_incons = generate_metrics_result(None, {"tf": tf_output.numpy(),
                                                                    "torch": torch_output.detach().numpy()},
                                                             model_name)
            incon_dict[model_name] = accumulative_incons
            result = check_inconsistency(accumulative_incons)
            if result is True:
                result_dict[model_name] = result
                f.write("inconsistency issue found in model: " + model_name + "\n")
            # coverage_path_cargo.append((cur_path, accumulative_incons, model_name, model_path, inner_diversity))

            # except Exception as e:
            # print(e)
            # f.write("model path: " + model_path + " FAILED!!!!\n")
            # f.write(str(e))
            # f.write("\n")
            # continue
            cal_cov = CoverageCalculator("all_layer_info.json")
            cal_cov.load_json(cur_path)
            if 'inner_diversity' not in dir():
                inner_diversity = -1
            if 'accumulative_incons' not in dir():
                accumulative_incons = -1
            print("distance is ", accumulative_incons, "type", type(accumulative_incons))  # np.float64
            cal_cov.cal_coverage2(inner_diversity, model_path, model_name, itera, accumulative_incons)
            clear_cache()
f.close()
# print("coverage_path_cargo", coverage_path_cargo)
# for idx, (path, distance, modelname, model_path, inner_diversity) in enumerate(coverage_path_cargo):
#     print("iteration", idx * 5)
#     cal_cov = CoverageCalculator("all_layer_info.json")
#     cal_cov.load_json(path)
#     cal_cov.cal_coverage2(inner_diversity, model_path, modelname, idx * 5, distance)
print("incon_dict", incon_dict)
print("ISSUES:", result_dict)
