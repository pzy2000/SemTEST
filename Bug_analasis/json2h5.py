import keras
import os
import json
import sys
sys.path.append('/root/implementations')


def custom_objects(mode="custom"):
    def no_activation(x):
        return x

    def leakyrelu(x):
        import keras.backend as K
        return K.relu(x, alpha=0.01)

    objects = {}
    objects['no_activation'] = no_activation
    objects['leakyrelu'] = leakyrelu
    if mode == "custom":
        from scripts.generation.custom_layers import CustomPadLayer, CustomCropLayer, CustomDropDimLayer, \
            CustomExpandLayer, CustomCastLayer
        objects['CustomPadLayer'] = CustomPadLayer
        objects['CustomCropLayer'] = CustomCropLayer
        objects['CustomDropDimLayer'] = CustomDropDimLayer
        objects['CustomExpandLayer'] = CustomExpandLayer
        objects['CustomCastLayer'] = CustomCastLayer

    return objects



def load_json(json_path):
    import keras
    with open(json_path, "rb") as file:
        model_json = file.read()
    model = keras.models.model_from_json(model_json, custom_objects=custom_objects())
    return model


def read_json_files_in_folder(folder_path, save_path):
    # 获取文件夹中所有文件的列表
    file_list = os.listdir(folder_path)
    for file_name in file_list:
        print(file_name)
        # 构建文件的完整路径
        file_path = os.path.join(folder_path, file_name)

        # 检查文件是否是JSON文件
        if file_name.endswith('.json'):
            # 打开文件并加载JSON数据
            model = load_json(file_path)
            model_save_path = os.path.join(save_path , file_name[:-5] + '.h5')
            # x = model.inputs
            # y = model.outputs
            # new_model = keras.Model(x, y)  # to avoid some configuration that does not have InputLayer
            model.save(model_save_path)


if __name__ == "__main__":
    # 指定要读取的文件夹路径
    folder_path = 'data/synthesized_models'
    save_path = 'data/h5_models'
    # 调用函数
    read_json_files_in_folder(folder_path, save_path)

