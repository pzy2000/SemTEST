import configparser
import os
import subprocess
import math
import time
from itertools import combinations

import keras
import numpy as np
import tensorflow as tf
import sys
import torch
from PIL import Image
from onnx2torch import convert
import networkx as nx
from tqdm import tqdm

sys.path.append("../")
sys.path.append("../implementations")
from implementations.model2json import model_to_json, union_json, CoverageCalculator
from implementations.scripts.tools import utils
from implementations.scripts.prediction.custom_objects import custom_objects

folder_path = "/data1/zmx/Lemon/resnet50-imagenet/origin"
envs = "CUDA_HOME=/usr/local/cuda-10 CUDA_ROOT=/usr/local/cuda-10 LD_LIBRARY_PATH=/usr/local/cuda-10/lib64:$LD_LIBRARY_PATH PATH=/usr/local/cuda-10/bin:$PATH"


class DataUtils:

    @staticmethod
    def image_resize(x, shape):
        x_return = []
        for x_test in x:
            tmp = np.copy(x_test)
            img = Image.fromarray(tmp.astype('uint8')).convert('RGB')
            img = img.resize(shape, Image.ANTIALIAS)
            x_return.append(np.array(img))
        return np.array(x_return)

    @staticmethod
    def get_data_by_exp(exp):
        import keras
        import keras.backend as K
        K.set_image_data_format("channels_last")

        lemon_cfg = configparser.ConfigParser()
        lemon_cfg.read("./config/demo.conf")
        dataset_dir = lemon_cfg['parameters']['dataset_dir']
        x_test = y_test = []
        if 'fashion-mnist' in exp:
            _, (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
            x_test = DataUtils.get_fashion_mnist_data(x_test)
            y_test = keras.utils.to_categorical(y_test, num_classes=10)
        elif 'fashion2' in exp:
            basedir = os.path.abspath(os.getcwd())
            labels_path = os.path.join(basedir, 'run', 'data', 't10k-labels-idx1-ubyte.gz')
            images_path = os.path.join(basedir, 'run', 'data', 't10k-images-idx3-ubyte.gz')
            import gzip
            with gzip.open(labels_path, 'rb') as lbpath:
                labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                                       offset=8)

            with gzip.open(images_path, 'rb') as imgpath:
                images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                                       offset=16).reshape(len(labels), 28, 28, 1)

            X_test = images.astype('float32') / 255.0
            Y_test = keras.utils.to_categorical(labels, 10)
            x_test, y_test = X_test, Y_test
        elif 'mnist' in exp:
            _, (x_test, y_test) = keras.datasets.mnist.load_data()
            x_test = DataUtils.get_mnist_data(x_test)
            y_test = keras.utils.to_categorical(y_test, num_classes=10)
        elif 'cifar100' in exp:
            from keras.datasets import cifar100
            subtract_pixel_mean = True
            # Load the CIFAR10 data.
            (X_train, Y_train), (X_test, Y_test) = cifar100.load_data()
            # Normalize data.
            X_train = X_train.astype('float32') / 255
            X_test = X_test.astype('float32') / 255
            # If subtract pixel mean is enabled
            if subtract_pixel_mean:
                X_train_mean = np.mean(X_train, axis=0)
                X_train -= X_train_mean
                X_test -= X_train_mean
            Y_test = keras.utils.to_categorical(Y_test, 100)
            x_test, y_test = X_test, Y_test

        elif 'cifar10' in exp:
            _, (x_test, y_test) = keras.datasets.cifar10.load_data()
            x_test = DataUtils.get_cifar10_data(x_test)
            y_test = keras.utils.to_categorical(y_test, num_classes=10)

        elif 'imagenet' in exp:
            input_precessor = DataUtils.imagenet_preprocess_dict()
            input_shapes_dict = DataUtils.imagenet_shape_dict()
            model_name = exp.split("-")[0]
            shape = input_shapes_dict[model_name]
            data_path = os.path.join(dataset_dir, "sampled_imagenet-1500.npz")
            data = np.load(data_path)
            x, y = data['x_test'], data['y_test']
            x_resize = DataUtils.image_resize(np.copy(x), shape)
            x_test = input_precessor[model_name](x_resize)
            y_test = keras.utils.to_categorical(y, num_classes=1000)

        elif 'svhn' in exp:
            import run.SVNH_DatasetUtil
            (_, _), (X_test, Y_test) = run.SVNH_DatasetUtil.load_data()
            x_test, y_test = X_test, Y_test

        elif 'sinewave' in exp:
            """
            see more details in
            https://github.com/StevenZxy/CIS400/tree/f69489c0624157ae86b5d8ddb1fa99c89a927256/code/LSTM-Neural-Network-for-Time-Series-Prediction-master
            """
            import pandas as pd
            dataframe = pd.read_csv(f"{dataset_dir}/sinewave.csv")
            test_size, seq_len = 1500, 50
            data_test = dataframe.get("sinewave").values[-(test_size + 50):]
            data_windows = []
            for i in range(test_size):
                data_windows.append(data_test[i:i + seq_len])
            data_windows = np.array(data_windows).astype(float).reshape((test_size, seq_len, 1))
            data_windows = np.array(data_windows).astype(float)
            x_test = data_windows[:, :-1]
            y_test = data_windows[:, -1, [0]]

        elif 'price' in exp:
            """see more details in https://github.com/omerbsezer/LSTM_RNN_Tutorials_with_Demo/tree/master/StockPricesPredictionProject"""
            x_test, y_test = DataUtils.get_price_data(dataset_dir)

        # TODO: Add your own data preprocessing here
        # Note: The returned inputs should be preprocessed and labels should decoded as one-hot vector which could be directly feed in model.
        # Both of them should be returned in batch, e.g. shape like (1500,28,28,1) and (1500,10)
        # elif 'xxx' in exp:
        #     x_test, y_test = get_your_data(dataset_dir)

        return x_test, y_test

    @staticmethod
    def save_img_from_array(path, array, index, exp):
        im = Image.fromarray(array)
        # path = path.rstrip("/")
        # save_path = "{}/{}_{}.png".format(path,exp,index)
        save_path = os.path.join(path, "{}_{}.png".format(exp, index))
        im.save(save_path)
        return save_path

    @staticmethod
    def shuffled_data(x, y, bs=None):
        ds = x.shape[0]
        all_idx = np.arange(ds)
        np.random.shuffle(all_idx)
        shuffle_idx = all_idx
        # shuffle_idx = all_idx[:bs]
        return x[shuffle_idx], y[shuffle_idx]

    @staticmethod
    def get_mnist_data(x_test):
        x_test = x_test.astype('float32') / 255.0
        x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
        return x_test

    @staticmethod
    def get_fashion_mnist_data(x_test):
        x_test = x_test.astype('float32') / 255.0
        w, h = 28, 28
        x_test = x_test.reshape(x_test.shape[0], w, h, 1)
        return x_test

    @staticmethod
    def get_cifar10_data(x_test):
        x_test = x_test.astype('float32') / 255.0
        w, h = 32, 32
        x_test = x_test.reshape(x_test.shape[0], w, h, 3)
        return x_test

    @staticmethod
    def get_price_data(data_dir):
        import pandas as pd
        from sklearn.preprocessing import MinMaxScaler

        def create_dataset(dataset, look_back=1):
            dataX, dataY = [], []
            for i in range(len(dataset) - look_back - 1):
                a = dataset[i:(i + look_back), 0]
                dataX.append(a)
                dataY.append(dataset[i + look_back, 0])
            return np.array(dataX), np.array(dataY)

        input_file = os.path.join(data_dir, "DIS.csv")
        df = pd.read_csv(input_file, header=None, index_col=None, delimiter=',')
        all_y = df[5].values
        dataset = all_y.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(dataset)

        train_size = int(len(dataset) * 0.5)
        test_size = len(dataset) - train_size
        train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]

        # reshape into X=t and Y=t+1, timestep 240
        look_back = 240
        trainX, trainY = create_dataset(train, look_back)

        # reshape input to be [samples, time steps, features]
        trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
        return trainX, trainY

    @staticmethod
    def imagenet_preprocess_dict():
        import tensorflow.keras as keras
        keras_preprocess_dict = dict()
        keras_preprocess_dict['resnet50'] = keras.applications.resnet50.preprocess_input
        keras_preprocess_dict['densenet121'] = keras.applications.densenet.preprocess_input
        keras_preprocess_dict['mobilenet.1.00.224'] = keras.applications.mobilenet.preprocess_input
        keras_preprocess_dict['vgg16'] = keras.applications.vgg16.preprocess_input
        keras_preprocess_dict['vgg19'] = keras.applications.vgg19.preprocess_input
        keras_preprocess_dict['inception.v3'] = keras.applications.inception_v3.preprocess_input
        keras_preprocess_dict['inception.v2'] = keras.applications.inception_resnet_v2.preprocess_input
        keras_preprocess_dict['xception'] = keras.applications.xception.preprocess_input
        return keras_preprocess_dict

    @staticmethod
    def imagenet_shape_dict():
        image_shapes = dict()
        image_shapes['resnet50'] = (224, 224)
        image_shapes['densenet121'] = (224, 224)
        image_shapes['mobilenet.1.00.224'] = (224, 224)
        image_shapes['vgg16'] = (224, 224)
        image_shapes['vgg19'] = (224, 224)
        image_shapes['inception.v3'] = (299, 299)
        image_shapes['inception.v2'] = (299, 299)
        image_shapes['xception'] = (299, 299)
        return image_shapes


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
# time.sleep(5)
print("awake")
if not os.path.exists("tmp"):
    os.makedirs("tmp")
if not os.path.exists("py"):
    os.makedirs("py")
if not os.path.exists("onnx"):
    os.makedirs("onnx")
f = open("FAILED.txt", "w")
exp = "resnet50"
dataset_dir = "../dataset"
input_precessor = DataUtils.imagenet_preprocess_dict()
input_shapes_dict = DataUtils.imagenet_shape_dict()
model_name = exp.split("-")[0]
shape = input_shapes_dict[model_name]
data_path = os.path.join(dataset_dir, "sampled_imagenet-1500.npz")
data = np.load(data_path)
x, y = data['x_test'], data['y_test']
print("x.shape", x.shape)
print("y.shape", y.shape)
x_resize = DataUtils.image_resize(np.copy(x), shape)
x_test = input_precessor[model_name](x_resize)
num_classes = np.unique(y).shape[0]
print(f"Number of unique classes: {num_classes}")
print("y_shape", y.shape)
y_test = keras.utils.to_categorical(y, num_classes=1000)
print(x_test.shape)
print(y_test.shape)
x_test, y_test = x_test[:10], y_test[:10]
print(x_test.shape)
print(y_test.shape)
print("======================")
# x_test, y_test = x_test[:10], y_test[:10]


def _init_input(input_shape):
    input_shape = list(input_shape)
    input_shape[0] = 10
    input_shape = tuple(input_shape)
    input = np.random.rand(*input_shape)
    return input


def is_nan_or_inf(t):
    if math.isnan(t) or math.isinf(t):
        return True
    else:
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


class MetricsUtils:
    @staticmethod
    def delta(y1_pred, y2_pred, y_true=None):
        y1_pred = np.reshape(y1_pred, [np.shape(y1_pred)[0], -1])
        y2_pred = np.reshape(y2_pred, [np.shape(y2_pred)[0], -1])
        return np.mean(np.abs(y1_pred - y2_pred), axis=1), np.sum(np.abs(y1_pred - y2_pred), axis=1)

    @staticmethod
    def D_MAD_metrics(y1_pred, y2_pred, y_true, epsilon=1e-7):
        theta_y1, sum_y1 = MetricsUtils.delta(y1_pred, y_true)
        theta_y2, sum_y2 = MetricsUtils.delta(y2_pred, y_true)
        return [
            0
            if (sum_y1[i] == 0 and sum_y2[i] == 0)
            else
            np.abs(theta_y1[i] - theta_y2[i]) / (theta_y1[i] + theta_y2[i])
            for i in range(len(y_true))
        ]


incon_dict = {}
result_dict = {}
# coverage_path_cargo = []
path_list = []
itera = 0
d_mader = MetricsUtils()
print("start union")
# Using os.walk to RECURSIVELY walk through the subdirectories
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
            model.summary()
            # exit(0)
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
            # inpu = _init_input(model.input_shape)
            inpu = x_test.copy()
            print("input_shape", inpu.shape)
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
            result = is_nan_or_inf(accumulative_incons)
            d_mad = d_mader.D_MAD_metrics(tf_output, torch_output.detach().numpy(), y_test)
            print("d_mad", d_mad)
            if result is True:
                result_dict[model_name] = result
                f.write("inconsistency issue found in model: " + model_name + "\n")
            # coverage_path_cargo.append((cur_path, accumulative_incons, model_name, model_path, inner_diversity))

            # except Exception as e:
            #     print(e)
            #     f.write("model path: " + model_path + " FAILED!!!!\n")
            #     f.write(str(e))
            #     f.write("\n")
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
print("incon_dict", incon_dict)
print("ISSUES:", result_dict)
