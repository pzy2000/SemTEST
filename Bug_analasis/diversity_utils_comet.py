import os.path
import json
from itertools import combinations_with_replacement
from itertools import product
import inspect
import configparser
import argparse


class CoverageCalculator:

    # init里只能是和具体模型无关的数值的初始化
    def __init__(self, config):
        self.model_structure = {}
        self.input_shapes = {}
        self.output_shapes = {}
        diversity_cfg = configparser.ConfigParser()
        diversity_cfg.read(config)
        parameters = diversity_cfg['parameters']
        self.flags = argparse.Namespace(
            json_path=parameters['json_path'],
            update_json=parameters.getboolean('update_json'),
            SHAPE_SPACE=parameters.getint('SHAPE_SPACE'),
            assert_dim=parameters.getboolean('assert_dim'),
        )

    # load_model完成具体模型信息的读入和处理
    # 在一个模型中某一层的input_shape和output_shape只有一种形状
    def load_model(self, model_dir):
        with open(model_dir, 'r') as json_file:
            model_info = json.load(json_file)
        self.model_structure = model_info['config']['layers']
        self.input_shapes = {}
        self.output_shapes = {}
        for layer_info in self.model_structure:
            if 'Input' not in layer_info['class_name']:
                if layer_info["class_name"] not in self.output_shapes:
                    self.output_shapes[layer_info["type"]] = layer_info['output_shape']
            if len(layer_info["pre_layers"]) != 0:
                if layer_info["type"] not in self.input_shapes:
                    self.input_shapes[layer_info["type"]] = self.model_structure[str(layer_info["pre_layers"][0])][
                        'output_shape']

    def trim_edge(self, edges):
        if not os.path.exists(self.flags.json_path):
            all_layer_info = {}
        else:
            with open(self.flags.json_path, 'r') as json_file:
                all_layer_info = json.load(json_file)

        # 更新输入输出维度
        for layer_type, input_shape in self.input_shapes.items():
            # if layer_name in ['INPUT', 'OUTPUT']:
            if 'input' in layer_type or 'output' in layer_type:
                continue
            layer_info = all_layer_info.setdefault(layer_type, {})
            input_dim = len(input_shape)
            if input_dim not in layer_info.setdefault("input_dim", []):
                layer_info.setdefault("input_dim", []).append(input_dim)
        for layer_type, output_shape in self.output_shapes.items():
            if 'input' in layer_type or 'output' in layer_type:
                continue
            layer_info = all_layer_info.setdefault(layer_type, {})
            output_dim = len(output_shape)
            if output_dim not in layer_info.setdefault("output_dim", []):
                layer_info.setdefault("output_dim", []).append(output_dim)

        # 写回文件
        if self.flags.update_json:
            with open(self.flags.json_path, 'w') as json_file:
                json.dump(all_layer_info, json_file, indent=4)

        trimmed_edge = []
        all_edges = []
        for edge in product(list(all_layer_info.keys()), repeat=2):
            layer_name1, layer_name2 = edge
            # if layer_name1 == "layer1.0.down_sample_layer.1":
            #     print(f"{layer_name1}->{layer_name2}")

            # 是否进行维度检查
            if self.flags.assert_dim is True:
                layer1_class_output_dim = set(all_layer_info[layer_name1]["output_dim"])
                connectable = len(layer1_class_output_dim.intersection(
                    set(all_layer_info[layer_name2]["input_dim"]))) != 0
                if connectable is False:
                    # if layer_name1=='layer1.0.down_sample_layer.1':
                    #     print(all_layer_info[layer_name1]["output_dim"])
                    #     print(layer_name2)
                    #     print(all_layer_info[layer_name2]["input_dim"])
                    continue
            all_edges.append((layer_name1, layer_name2))

        for edge in edges:
            layer_name1, layer_name2 = edge
            # 检查是否是规定范围内的层
            if layer_name1 not in all_layer_info.keys() or layer_name2 not in all_layer_info.keys():
                continue
            if edge not in all_edges:
                continue
            trimmed_edge.append(edge)

        return trimmed_edge, len(all_edges)

    def edge_diversity(self):

        existing_edges = []
        for layer_idx, layer_info in self.model_structure.items():
            layer_type = layer_info['type']
            if 'input' in layer_type:
                continue
            for pre_layer_idx in layer_info["pre_layers"]:
                pre_layer_type = self.model_structure[str(pre_layer_idx)]["type"]
                if 'input' in pre_layer_type:
                    continue
                edge = (pre_layer_type, layer_type)
                if edge not in existing_edges:
                    existing_edges.append(edge)

        print(f"edges before trim: {len(existing_edges)}")
        old = existing_edges
        existing_edges, all_edges_count = self.trim_edge(existing_edges)
        for edge in old:
            if edge not in existing_edges:
                print(f"{edge} has been trimmed")
        print(f"edges after trim: {len(existing_edges)}")
        print(f"all edges num: {all_edges_count}")
        print(f"edge coverage is {len(existing_edges) / all_edges_count}")

    def update_input(self, existing_input):  # update=False):
        """
        existing_input: {"layer_class1": {"input_dim": [], "dtype": [], "shape": []}, ...}
        """
        if not os.path.exists(self.flags.json_path):
            all_layer_info = {}
        else:
            with open(self.flags.json_path, 'r') as json_file:
                all_layer_info = json.load(json_file)

        assert type(all_layer_info) == type(existing_input), print(type(all_layer_info), type(existing_input))
        assert type(all_layer_info) == dict
        new_input = False

        for layer_class in existing_input.keys():
            if layer_class not in all_layer_info.keys():
                new_input = True
                all_layer_info[layer_class] = existing_input[layer_class]
            else:
                new_input_dim = set(existing_input[layer_class]["input_dim"])
                # print(existing_input[layer_class]["dtype"])
                new_dtype = set(existing_input[layer_class]["dtype"])
                # new_shape = set(existing_input[layer_class]["shape"])
                old_input_dim = set(all_layer_info[layer_class]["input_dim"])
                old_dtype = set(all_layer_info[layer_class].setdefault("dtype", []))
                # old_shape = set(all_layer_info[layer_class].setdefault("shape",[]))
                all_layer_info[layer_class]["input_dim"] = list(set.union(old_input_dim, new_input_dim))
                all_layer_info[layer_class]["dtype"] = list(set.union(old_dtype, new_dtype))
                # if len(old_shape) < SHAPE_SPACE:
                #     all_layer_info[layer_class]["shape"] = list(set.union(old_shape, new_shape))
                for shape in existing_input[layer_class]["shape"]:
                    if len(all_layer_info[layer_class].setdefault("shape", [])) >= self.flags.SHAPE_SPACE:
                        break
                    if shape not in all_layer_info[layer_class]["shape"]:
                        all_layer_info[layer_class]["shape"].append(shape)
                # all_layer_info[layer_class]["shape"] = list(set.union(old_shape, new_shape))
        # 写回文件
        if self.flags.update_json:
            with open(self.flags.json_path, 'w') as json_file:
                json.dump(all_layer_info, json_file, indent=4)

        covered_input_dims_num, covered_dtype_num, covered_shape_num = self.covered_input_num(existing_input)
        all_input_dims_num, all_dtype_num, _ = self.covered_input_num(all_layer_info)
        # all_dtype_num = len(all_layer_info.keys())*len()
        all_shape_num = len(all_layer_info.keys()) * self.flags.SHAPE_SPACE
        print(f"The NDims Coverage Is: {covered_input_dims_num}/{all_input_dims_num}")
        print(f"The DType Coverage Is: {covered_dtype_num}/{all_dtype_num}")
        print(f"The Shape Coverage Is: {covered_shape_num}/{all_shape_num}")
        print(
            f"The Input Coverage Is: {covered_input_dims_num + covered_dtype_num + covered_shape_num}/{all_input_dims_num + all_dtype_num + all_shape_num}")

    def input_diversity(self):
        existing_inputs = {}
        for layer_info in self.model_structure.values():
            layer_type = layer_info['type']
            if 'input' in layer_type:
                continue
            if layer_type not in existing_inputs:
                existing_inputs[layer_type] = {"input_dim": [], "dtype": [], "shape": []}
            input_dim = len(self.input_shapes[layer_type])
            dtype = ['float32']
            # unhashable type: 'list'
            shape = str(list(self.input_shapes[layer_type]))
            if input_dim not in existing_inputs[layer_type]['input_dim']:
                existing_inputs[layer_type]['input_dim'].append(input_dim)
            if str(dtype[0]) not in existing_inputs[layer_type]['dtype']:
                existing_inputs[layer_type]['dtype'].append(str(dtype[0]))
            if shape not in existing_inputs[layer_type]['shape']:
                existing_inputs[layer_type]['shape'].append(shape)
        self.update_input(existing_inputs)

    def covered_input_num(self, existing_inputs):
        covered_input_dims_num = 0
        covered_dtype_num = 0
        covered_shape_num = 0
        for layer_class in existing_inputs.keys():
            input_dims_list = existing_inputs[layer_class]["input_dim"]
            covered_input_dims_num += len(input_dims_list)
            dtype_list = existing_inputs[layer_class]["dtype"]
            covered_dtype_num += len(dtype_list)
            # print(f'{layer_class}: {covered_dtype_num}')
            shape_list = existing_inputs[layer_class]["shape"]
            covered_shape_num += min(len(shape_list),
                                     self.flags.SHAPE_SPACE)
            # if the total number of shape is larger that SHAPE_SPACE, we set it as 100%
        return covered_input_dims_num, covered_dtype_num, covered_shape_num

    def node_diversity(self):
        """
        existing_nodes: {"layer_name1": [layer_config1, layer_config2], "layer_name2": [], ...}

        """
        existing_node = {}
        current_config_num = 0
        for layer_info in self.model_structure.values():
            layer_type = layer_info['type']
            if 'input' in layer_type:
                continue
            if layer_type not in existing_node.keys():
                existing_node[layer_type] = []
            # 获取配置
            new_config = layer_info['args'].copy()
            new_config.pop('name', "")
            # Tuple写入到json文件会自动转换为List
            if new_config not in existing_node[layer_type]:
                existing_node[layer_type].append(new_config)

        for layer_class in existing_node:
            current_config_num += len(existing_node[layer_class])

        total_config_num = self.update_nodes(existing_node)
        print(f'parameter coverage is {current_config_num}/{total_config_num}')

    def update_nodes(self, existing_node):
        if not os.path.exists(self.flags.json_path):
            all_layer_info = {}
        else:
            with open(self.flags.json_path, 'r') as json_file:
                all_layer_info = json.load(json_file)
        total_config_num = 0
        for layer_class, config_list in existing_node.items():
            if layer_class not in all_layer_info.keys():
                all_layer_info[layer_class] = {}
            if "config" not in all_layer_info[layer_class].keys():
                all_layer_info[layer_class]['config'] = []
            for config in config_list:
                if config not in all_layer_info[layer_class]['config']:
                    all_layer_info[layer_class]['config'].append(config)
        for layer_class, item in all_layer_info.items():
            total_config_num += len(all_layer_info[layer_class]['config'])
        # 写回文件
        if self.flags.update_json:
            with open(self.flags.json_path, 'w') as json_file:
                json.dump(all_layer_info, json_file, indent=4)
        return total_config_num


if __name__ == '__main__':
    output_dir = 'data/model_json'
    config_path = './diversity.conf'
    # files = os.listdir(output_dir)
    # for file in files:
    #     each_model_dir = os.path.join(output_dir, file, 'models', 'model.json')
    cal_cov = CoverageCalculator(config_path)
    for model_id in os.listdir(output_dir):
        print(f'model_id:{model_id}')
        model_path = os.path.join(output_dir, model_id)
        cal_cov.load_model(model_path)
        print('-------------------------')
        cal_cov.edge_diversity()
        print('-------------------------')
        cal_cov.input_diversity()
        print('-------------------------')
        cal_cov.node_diversity()
        print('########################################')
        print()

