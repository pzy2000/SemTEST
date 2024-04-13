import os
import sys
import tensorflow as tf
import tf2onnx
import pandas as pd
import torch
from onnx2torch import convert
from tqdm import tqdm
from infoplus.TorchInfoPlus import torchinfoplus
sys.path.append("../")
sys.path.append("../implementations")
from implementations.scripts.prediction.custom_objects import custom_objects
envs = "CUDA_HOME=/usr/local/cuda-10 CUDA_ROOT=/usr/local/cuda-10 LD_LIBRARY_PATH=/usr/local/cuda-10/lib64:$LD_LIBRARY_PATH PATH=/usr/local/cuda-10/bin:$PATH"
import onnxruntime as ort
import numpy as np
device = "cuda:1"


def _init_input(input_shape):
    input_shape = list(input_shape)
    input_shape[0] = 1
    input_shape = tuple(input_shape)
    print("input_shape", input_shape)
    input = np.random.rand(*input_shape)
    return input


comet_path = r'/root/LOGS/LEMON_result'
path = comet_path
models = os.listdir(comet_path)
nan_path = []
inconsistency_path = []

for model in models:
    result = pd.read_excel(comet_path + "/" + model + "/output_coverage.xlsx")
    for i in range(result.shape[0]):
        distance = result.iloc[i, :]['distance']
        path = result.iloc[i, :]['path']
        # path = path.replace("/data1/pzy/MUTANTS/LEMON", "G:/mutants/COMET-mutants")
        if np.isnan(distance):
            nan_path.append(path)
        elif distance > 8:
            inconsistency_path.append(path)

# nan_path = ["/data1/pzy/MUTANTS/LEMON/lenet5_fashion/lenet5-fashion-mnist_origin-SpecialI5-MDims8-Edge17-NLAll46-NLAll52-54/lenet5-fashion-mnist_origin-SpecialI5-MDims8-Edge17-NLAll46-NLAll52.h5"]
print(nan_path)
print(inconsistency_path)
# exit(666)

# def clear_cache():
#     if os.path.exists("tmp"):
#         subprocess.call("rm -rf tmp", shell=True)
#     if os.path.exists("onnx"):
#         subprocess.call("rm -rf onnx", shell=True)
#     if not os.path.exists("tmp"):
#         os.makedirs("tmp")
#     if not os.path.exists("onnx"):
#         os.makedirs("onnx")


# clear_cache()
# if os.path.exists("Bug_log.xlsx"):
#     os.remove("Bug_log.xlsx")
# print("sleep 2s")
# time.sleep(2)
# print("awake")
if not os.path.exists("tmp"):
    os.makedirs("tmp")
# if not os.path.exists("py"):
#     os.makedirs("py")
if not os.path.exists("onnx"):
    os.makedirs("onnx")
f = open("BUG_Analysis_FAILED.txt", "w")


def _init_input(input_shape):
    input_shape = list(input_shape)
    input_shape[0] = 10
    input_shape = tuple(input_shape)
    input = np.random.rand(*input_shape)
    print("input_shape", input_shape)
    return input


def load_and_infer_onnx_model(model_path, input_data):
    # 加载ONNX模型
    input_data = input_data.astype(np.float32)
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name

    # 推理
    outputs = session.run(None, {input_name: input_data})

    # 检查推理结果是否含有NaN
    has_nan = np.isnan(outputs).any()

    return has_nan


def info_com(model, np_data, dtypes, verbose=0):
    torch_data = torchinfoplus.np_2_tensor(np_data, dtypes, device)
    result, global_layer_info = torchinfoplus.summary(
        model=model,
        input_data=torch_data,
        # input_size=[(96, 16), (96, 16), (96, 16), (96, 16)],
        dtypes=dtypes,
        col_names=['input_size', 'output_size', 'name'], depth=8,
        verbose=verbose)
    # print("result", result)
    input_datas = torchinfoplus.get_input_datas(global_layer_info)
    output_datas = torchinfoplus.get_output_datas(global_layer_info)
    return input_datas, output_datas


for filenames in tqdm(nan_path):
    # print("1111")
    # for filename in filenames:
    # print("2222")
    filename = filenames
    # print("filename", filename)
    if filename.endswith(".h5"):
        # print("33333")
        model_path = filename
        # cur_path = model_to_json(model_path)
        # union_json(cur_path, os.path.join("all_layer_info.json"))
        # try:
        # print("model_path", model_path)
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects())
        # model = tf.keras.models.load_model(model_path,)
        model_name = os.path.basename(filename)[:-3]
        # print("model_name", model_name)
        # model.save(f'tmp/{model_name}', save_format='tf')
        # status = subprocess.call(
        #     f"{envs} python -u -m tf2onnx.convert --saved-model " + f'tmp/{model_name}' + " --output onnx/" + model_name + ".onnx",
        #     shell=True)
        model_proto, _ = tf2onnx.convert.from_keras(model, opset=13, output_path="onnx/" + model_name + ".onnx")
        # print("model name", model_name)
        onnx_model_path = "onnx/" + model_name + ".onnx"
        print("onnx_model_path", onnx_model_path)
        inpu = _init_input(model.input_shape)
        input_data = inpu.astype(np.float32)
        try:
            torch_model = convert(onnx_model_path)
            torch_input = torch.tensor(inpu, dtype=torch.float32)
            torch_output = torch_model(torch_input)
            nan_check_torch = torch.isnan(torch_output).any()
            if_torch_nan = nan_check_torch.item()
        except Exception as e:
            if_torch_nan = -666
            print(e)
        try:
            tf_input = tf.convert_to_tensor(inpu)
            tf_output = model(tf_input)
            nan_check_tf = tf.reduce_any(tf.math.is_nan(tf_output))
            if_tf_nan = nan_check_tf.numpy()
        except Exception as e:
            if_tf_nan = -666
            print(e)
        # try:
        # session = ort.InferenceSession(model_path)
        # input_name = session.get_inputs()[0].name
        # outputs = session.run(None, {input_name: input_data})
        onnx_has_nan = load_and_infer_onnx_model(onnx_model_path, inpu)
        # except Exception as e:
        #     onnx_has_nan = -666
        #     print(e)
        print('torch推理结果中含有NaN: ', if_torch_nan)
        print(f'tensorflow推理结果中含有NaN: {if_tf_nan}')
        print(f'onnx推理结果中含有NaN: {onnx_has_nan}')

        tf_input = tf.convert_to_tensor(inpu)

        tf_output = model(tf_input)
        nan_check_tf = tf.reduce_any(tf.math.is_nan(tf_output))

        if_tf_nan = nan_check_tf.numpy()
        if 'torch_model' in dir():
            torch_input_list, torch_output_list = info_com(torch_model, [inpu], dtypes=[torch.float32], verbose=0)

            layers = torch_model.named_modules()
            for layer in layers:
                if layer[0] == "model/conv_dw_3/depthwise":
                    print("prev name", type(layer[1]))
                    # np.save("torch_input.npy", torch_input_list[layer[0]][0].detach().cpu().numpy())

            for key in torch_input_list:
                # print("name", key)
                if torch_input_list[key] is None:
                    continue

                inpu_to_be_judge = torch_input_list[key][0]
                nan_check_torch = torch.isnan(inpu_to_be_judge).any()
                if_torch_nan1 = nan_check_torch.item()
                inpu_to_be_judge = torch_output_list[key][0]
                nan_check_torch2 = torch.isnan(inpu_to_be_judge).any()
                if_torch_nan2 = nan_check_torch2.item()
                if not if_torch_nan1 == if_torch_nan2:
                    print("inpu", if_torch_nan1)
                    print("output", if_torch_nan2)
                    print("nan in layer name: ", key)
                # print("fcuk1: ", if_torch_nan1)
                # print("fuck2: ", if_torch_nan2)
                # print()

        del model
        print("tensorflow cache cleared")
        if if_torch_nan != -666:
            del torch_model
            print("torch cache cleared")
f.close()
