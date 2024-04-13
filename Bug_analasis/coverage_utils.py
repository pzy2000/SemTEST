import keras
# from implementations.scripts.generation.layer_tool_kits import LayerUtils
import os
from implementations.scripts.generation.layer_pools import LAYERLIST
from implementations.scripts.prediction.custom_objects import custom_objects

total_layers = []
for layer_type in LAYERLIST:
    target_layers = LAYERLIST[layer_type]
    total_layers += list(target_layers.available_layers.keys())


class CoverageCalculator:
    def __init__(self, output_dir):
        self.op_num = {}
        self.edge_num = {}
        self.op_type_num = {}
        model_folders = os.listdir(output_dir)
        for model_name in model_folders:
            model_paths = [os.path.join(output_dir, model_name, file) for file in
                           os.listdir(os.path.join(output_dir, model_name)) if file.endswith('.h5')]
            for one_model_path in model_paths:
                layer_dict = {}
                cur_edge_num = 0
                cur_model = keras.models.load_model(one_model_path,custom_objects=custom_objects())
                cur_layers = cur_model.layers
                for layer in cur_layers:
                    layer_name = layer.__class__.__name__
                    layer_dict[layer_name] = layer_dict[layer_name] + 1 if layer_name in layer_dict else 1
                    inbound_nodes = layer._inbound_nodes
                    if inbound_nodes:
                        if isinstance(inbound_nodes[0].inbound_layers, list):
                            cur_edge_num += len(inbound_nodes[0].inbound_layers)
                        else:
                            if inbound_nodes:
                                cur_edge_num += 1
                self.op_type_num[model_name] = len(layer_dict)
                self.op_num[model_name] = sum(layer_dict.values())
                self.edge_num[model_name] = cur_edge_num

        print(self.op_num)
        print(self.edge_num)
        print(self.op_type_num)

    def op_type_cover(self, model_name: str):
        sub_op_num = self.op_type_num[model_name]
        return sub_op_num / len(total_layers)

    def op_num_cover(self, model_name: str):
        return self.op_num[model_name] / max(self.op_num.values())

    def edge_cover(self, model_name):
        return self.edge_num[model_name] / max(self.edge_num.values())


if __name__ == '__main__':
    output_dir = '/root/data/working_dir/COMET/results/models'
    cover_cal = CoverageCalculator(output_dir)
    for model_name in os.listdir(output_dir):
        model_paths = [os.path.join(output_dir, model_name, file) for file in
                       os.listdir(os.path.join(output_dir, model_name)) if file.endswith('.h5')]
        if len(model_paths) == 0:
            continue
        print(f'model_id:{model_name}')
        print(f'op_type_cover:{cover_cal.op_type_cover(model_name)}')
        print(f'op_num_cover:{cover_cal.op_num_cover(model_name)}')
        print(f'edge_cover:{cover_cal.edge_cover(model_name)}')
        print('#########################################')
        print()
