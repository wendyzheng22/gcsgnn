import argparse
import os
import pickle
import random
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from baselines.baseline import get_baseline
from data.data_utils import (
    get_attribute_lists_from_data_list,
    get_dataloaders,
    get_dataset,
)
from data.dataset import Graph, GraphDataset
from model.base_model import BaseModel
from model.explainable_gnn import ExplainGNN, get_cfs_from_output
from model.model_utils import evaluate_gnn, train_gnn
from torch.utils.data import DataLoader
from utils import (
    BEST_DATASET_PARAMETERS,
    DATASET_MAP,
    get_file_paths,
    parse_args,
    update_file_paths,
)


def load_gnn(
    gnn:ExplainGNN, model_path:str, results_path:str, train_loader:DataLoader, val_loader:DataLoader, test_loader:DataLoader, lr:float, wd:float, epochs:int
) -> tuple[BaseModel, GraphDataset]:
    if not os.path.exists(model_path):
        print("---------------------- Training GNN ----------------------")
        train_gnn(gnn, train_loader, val_loader, lr, wd, epochs, results_path, model_path)
        print("---------------------- Testing GNN ----------------------")
        with open(results_path, "a+") as results_file:
            results_file.write("Start testing GNN\n")
            loss, acc_g, acc_cf = evaluate_gnn(gnn, test_loader)
            print(f"test_loss: {loss} | test_acc : {acc_g} | test_acc_cf : {acc_cf}")
            results_file.write(f"test_loss = {loss}, test_acc_g = {acc_g}, test_acc_cf = {acc_cf}\n")
            results_file.write("Completed testing GNN\n")
    gnn.load_state_dict(torch.load(model_path, map_location=args.device))
    if "synthetic" in model_path:
        os.remove(model_path)
    gnn.eval()
    return gnn

def get_cfs(
    gnn: BaseModel, dataset: GraphDataset, undesired_label: int, args:argparse.Namespace, model_path:str, cfs_path: str
) -> tuple[list[Graph], BaseModel]:
    if args.exp_type == "baseline":
        baseline = get_baseline(args.baseline_type)
        if args.baseline_type in ["protgnn", "ses"]:
            cfs, gnn = baseline(dataset, args, model_path) 
        else:
            cfs = baseline(gnn, dataset, args)
    else:
        gnn.eval()
        loader = DataLoader(dataset, args.batch_size)
        cfs = []
        for batch in loader:
            output = gnn.run_one_batch(batch, dataset.node_labels)
            labels, graph_idxs = (
                batch["graph_label"],
                batch["graph_idx"],
            )
            cur_cfs = get_cfs_from_output(
                output, graph_idxs, labels, undesired_label, dataset.node_labels
            )
            cfs.extend(cur_cfs)
            del output
    return cfs, gnn

def get_valid_cfs(
    gnn: BaseModel, cf_list: list[Graph], dataset: GraphDataset, batch_size:int
) -> tuple[float, list[Graph]]:
    true_cf_list = []
    for batch_idx in range(len(cf_list) // batch_size + ((len(cf_list)  % batch_size) != 0)):
        x, adj, edge_attr, graph_idxs, _ = get_attribute_lists_from_data_list(
            cf_list[batch_idx * batch_size : (batch_idx + 1) * batch_size], args.device
        )
        cf_preds = gnn(x, adj, edge_attr)["graph_pred"].argmax(dim=-1)
        true_cf_list.extend([
            cf for i, cf in enumerate(cf_list[batch_idx * batch_size : (batch_idx + 1) * batch_size]) if cf_preds[i] != dataset.undesired_label and dataset[graph_idxs[i]]['graph_label'] == dataset.undesired_label
        ])
        del x, adj, edge_attr, cf_preds
    return len(true_cf_list), true_cf_list


def compute_proximity(true_cf_list: list[Graph], dataset: GraphDataset) -> float:
    if len(true_cf_list) == 0:
        return -1
    dist = 0.0
    for cf in true_cf_list:
        orig_graph = dataset.get_graph_idx(cf["graph_idx"])
        edge_dim = dataset.num_edge_attributes + 1
        node_feat_dim = dataset.num_node_features + 1
        x, x_cf = orig_graph["x"].float(), cf['x'].float()
        if dataset.node_labels:
            x_cf = F.one_hot(x_cf.long().squeeze(-1), num_classes=node_feat_dim).float()
            x = F.one_hot(x.long().squeeze(-1), num_classes=node_feat_dim).float()
        dist += ((orig_graph["adj_matrix"] - cf["adj_matrix"]).pow(2).sum() / 2).sqrt() + (x_cf - x).pow(2).sum().sqrt()
        if "edge_attr_matrix" in cf:
            edge_attr_cf = F.one_hot(cf['edge_attr_matrix'].long().squeeze(-1), num_classes=edge_dim).float()
            edge_attr = F.one_hot(orig_graph["edge_attr_matrix"].long().squeeze(-1),num_classes=edge_dim).float()
            dist += ((edge_attr_cf - edge_attr).pow(2).sum() / 2).sqrt()
    return dist / len(true_cf_list)


def evaluate_cfs(
    cf_list: list[Graph],
    dataset: GraphDataset,
    gnn: BaseModel,
    num_undesired_instances: int,
    batch_size:int
) -> tuple[float, float, list[Graph]]:
    if not cf_list or len(cf_list) == 0:
        return 0, -1, []
    for cf in cf_list:
        cf["adj_matrix"] = (cf["adj_matrix"] >= 0.5).float()
        if dataset.node_labels:
            if len(cf["x"].shape) == 2:
                cf["x"] = cf['x'].argmax(-1).float()
            node_mask = (cf["x"] > 0).float().unsqueeze(-1) * (cf["x"] > 0).float().unsqueeze(-2)
            cf["adj_matrix"] = node_mask * cf["adj_matrix"]
            cf["x"] = (cf["adj_matrix"].sum(-1) > 0).float() * cf["x"]
        else:
            cf["x"] = (cf["adj_matrix"].sum(-1) > 0).float().unsqueeze(-1) * cf["x"]
            node_mask = (cf["x"].sum(-1) > 0).float()
            node_mask = node_mask.unsqueeze(-1) * node_mask.unsqueeze(-2)
            cf["adj_matrix"] = node_mask * cf["adj_matrix"]
        if "edge_attr_matrix" in cf:
            if len(cf["edge_attr_matrix"].shape)== 3:
                cf["edge_attr_matrix"] = cf["edge_attr_matrix"].argmax(-1).float()
            cf["adj_matrix"] = (cf["edge_attr_matrix"] > 0).float() * cf["adj_matrix"]  
            cf["edge_attr_matrix"] = cf["edge_attr_matrix"] * cf["adj_matrix"]
            if dataset.node_labels:
                cf["x"] = (cf["adj_matrix"].sum(-1) > 0).float() * cf["x"]
    num_correct, true_cf_list = get_valid_cfs(gnn, cf_list, dataset, batch_size)
    percent_coverage = num_correct / num_undesired_instances * 100
    if percent_coverage == 0:
        return 0, -1, []
    proximity = compute_proximity(true_cf_list, dataset)
    return percent_coverage, proximity, true_cf_list


def save_exp_results(
    results_path: str,
    cfs_path: str,
    exp_num: int,
    coverage: float,
    proximity: float,
    cfs_list: list[Graph],
    num_undesired: int,
    time_spent: float,
) -> None:
    with open(results_path, "a+") as f:
        if exp_num == 0:
            f.write(
                "Experiment, coverage, proximity, number of valid cfs found / number of undesired instances, time spend\n"
            )
        f.write(
            f"Experiment {exp_num}, {coverage}, {proximity}, {len(cfs_list)} / {num_undesired}, {time_spent}s\n"
        )

    if len(cfs_list) != 0 and "synthetic" not in cfs_path:
        with open(f"{cfs_path}.pkl", "wb") as f:
            pickle.dump(cfs_list, f, pickle.HIGHEST_PROTOCOL)

def main(args) -> None:
    print("---------------------- Start experiments ----------------------")
    start = time.time()
    coverage_list, proximity_list, true_cfs_list, times = [], [], [], []
    if args.exp_type != "param":
        args.gnn_lr = BEST_DATASET_PARAMETERS[args.dataset]['lr']
        args.gnn_wd = BEST_DATASET_PARAMETERS[args.dataset]['wd']
        args.gnn_epochs = BEST_DATASET_PARAMETERS[args.dataset]['epochs']  
    model_path, model_results_path, cfs_path, exp_results_path = get_file_paths(args)
    with open(exp_results_path, "a+") as f:
        f.write(f"{datetime.fromtimestamp(start).strftime('%Y-%m-%d_%H:%M:%S')}\n")
    for i in range(args.num_exp):
        start_exp_time = time.time()
        dataset = get_dataset(args.dataset, 42, args.size)
        cur_model_path, cur_model_results_path, cur_cfs_path = update_file_paths(
            model_path, model_results_path, cfs_path, i
        )
        if args.exp_type == "param":
            gnn = ExplainGNN(
                dataset.num_node_features,
                args.embedding_dim,
                dataset.num_classes,
                dataset.num_edge_attributes,
                dataset.max_num_nodes,
                args.device,
                args.sub_patterns_per_class,
                args.sub_pattern_dim,
                args.encoder_dropout,
                args.decoder_dropout,
                dataset.undesired_label,
            ).to(args.device)
        else:
            gnn = ExplainGNN(
                dataset.num_node_features,
                args.embedding_dim,
                dataset.num_classes,
                dataset.num_edge_attributes,
                dataset.max_num_nodes,
                args.device,
                BEST_DATASET_PARAMETERS[args.dataset]['spc'],
                BEST_DATASET_PARAMETERS[args.dataset]['spd'],
                BEST_DATASET_PARAMETERS[args.dataset]['edo'],
                BEST_DATASET_PARAMETERS[args.dataset]['ddo'],
                dataset.undesired_label,
            ).to(args.device)
        train_loader, val_loader, test_loader = get_dataloaders(dataset, args.batch_size)
        if args.exp_type == "ablation":
            init_state = gnn.state_dict()
            pretrained_model_path = cur_model_path.replace(f"/{args.ablation_type}/", "/pretrained/")
            pretrained_results_path = cur_model_results_path.replace(f"/{args.ablation_type}/", "/pretrained/")
            os.makedirs(os.path.dirname(pretrained_model_path), exist_ok=True)
            gnn.masks.requires_grad = False
            gnn.sub_patterns.requires_grad = False
            pretrained_gnn = load_gnn(gnn, pretrained_model_path, pretrained_results_path, train_loader, val_loader, test_loader, args.gnn_lr, args.gnn_wd, args.gnn_epochs)
            gnn.load_state_dict(init_state)
            gnn.masks.requires_grad = True
            gnn.sub_patterns.requires_grad = True
            if args.ablation_type in ["nd", "nc"]:
                gnn.decoder.load_state_dict(pretrained_gnn.decoder.state_dict())
                for param in gnn.decoder.parameters():
                    param.requires_grad = False
            if args.ablation_type in ["ne", "nc"]:
                gnn.encoder.load_state_dict(pretrained_gnn.encoder.state_dict())
                for param in gnn.encoder.parameters():
                    param.requires_grad = False
            if args.ablation_type in ["np", "nc"]:
                gnn.predictor.load_state_dict(pretrained_gnn.predictor.state_dict())
                for param in gnn.predictor.parameters():
                    param.requires_grad = False
            gnn = load_gnn(gnn, cur_model_path, cur_model_results_path, train_loader, val_loader, test_loader, args.gnn_lr, args.gnn_wd, args.gnn_epochs)
            del init_state, pretrained_gnn
        elif args.exp_type != "baseline" or args.baseline_type not in ['protgnn', 'ses']:
            gnn = load_gnn(gnn, cur_model_path, cur_model_results_path, train_loader, val_loader, test_loader, args.gnn_lr, args.gnn_wd, args.gnn_epochs)
        cfs, gnn = get_cfs(gnn, dataset, dataset.undesired_label, args, cur_model_results_path.replace('training_results_', 'model_')+".pt", cur_cfs_path)
        dataset = get_dataset(args.dataset, 42, args.size)
        coverage, proximity, true_cfs = evaluate_cfs(cfs, dataset, gnn, dataset.num_undesired, args.batch_size)
        exp_time_spent = time.time() - start_exp_time
        save_exp_results(
            exp_results_path,
            cur_cfs_path,
            i,
            coverage,
            proximity,
            true_cfs,
            dataset.num_undesired,
            exp_time_spent,
        )
        coverage_list.append(coverage)
        proximity_list.append(proximity)
        true_cfs_list.append(true_cfs)
        times.append(exp_time_spent)
        print(
            f"Experiment {i} - coverage = {coverage}, proximity = {proximity}, number of valid cfs found = {len(true_cfs)}, time spend = {exp_time_spent}s\n"
        )
        gnn, cfs = 0,0
        with torch.cuda.device(args.device): 
            torch.cuda.empty_cache()

    total_time = time.time() - start
    print(
        f"---------------------- End experiments - time spend: {total_time}s ----------------------"
    )

    proximity_list = [proximity for proximity in proximity_list if proximity != -1]
    with open(exp_results_path, "a") as f:
        if len(proximity_list) > 0:
            f.write(
                f"Summary,{np.mean(coverage_list)} ± {np.std(coverage_list)}, \
                {np.mean(proximity_list)} ± {np.std(proximity_list)}, \
                {np.mean([len(cfs) for cfs in true_cfs_list])}, {total_time}\n"
            )
        else:
            f.write(
                f"Summary,{np.mean(coverage_list)} ± {np.std(coverage_list)}, \
                -1, {np.mean([len(cfs) for cfs in true_cfs_list])}, {total_time}\n"
            )


if __name__ == "__main__":
    args = parse_args()
    args.device = "cuda:" + str(args.device) if torch.cuda.is_available() else "cpu"
    args.dataset = DATASET_MAP[args.dataset]
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    main(args)