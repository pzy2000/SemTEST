import keras
from scripts.generation.layer_tool_kits import LayerUtils

def calc_coverage(ori_model, sub_model):
    ori_layer = ori_model.layers
    sub_layer = sub_model.layers
    ori_dict, sub_dict = {}, {}
    for layer in ori_layer:
        layer_name = layer.__class__.__name__
        ori_dict[layer_name] = ori_dict[layer_name] + 1 if layer_name in ori_dict else 1
    for layer in sub_layer:
        layer_name = layer.__class__.__name__
        sub_dict[layer_name] = sub_dict[layer_name] + 1 if layer_name in sub_dict else 1
    layerutils = LayerUtils()
    total_op_type = len(layerutils.available_model_level_layers.keys()) + len(
        layerutils.available_source_level_layers.keys())
    op_type_cover = len(sub_dict.keys()) / total_op_type
    # ori_set = set()
    # for layer in list(ori_dict.keys()):
    #     name = layer.__class__.__name__
    #     ori_set.add(name)
    # sub_op_num = 0
    # for layer, cnt in sub_dict.items():
    #     name = layer.__class__.__name__
    #     if name in ori_set:
    #         sub_op_num += cnt
    op_num_cover = sum(sub_dict.values()) / sum(ori_dict.values())
    sub_edge = ori_edge = 0
    for layer in sub_layer:
        inbound_nodes = layer._inbound_nodes
        if inbound_nodes:
            sub_edge += len(inbound_nodes[0].inbound_layers)
    for layer in ori_layer:
        inbound_nodes = layer._inbound_nodes
        if inbound_nodes:
            ori_edge += len(inbound_nodes[0].inbound_layers)
    edge_cover = sub_edge / ori_edge
    return op_type_cover, op_num_cover, edge_cover


model_path1 = ""
model_path2 = ""
sub_model = keras.models.load_model(model_path1)
origin_model = keras.models.load_model(model_path2)
op_type_cover, op_num_cover, edge_cover = calc_coverage(origin_model, sub_model)
