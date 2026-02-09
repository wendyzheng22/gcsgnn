import argparse
import random

import torch
import torch.nn.functional as F
from data.data_utils import get_attribute_lists_from_data_list
from data.dataset import Graph, GraphDataset
from model.base_model import BaseModel
from torch.utils.data import DataLoader
from tqdm import tqdm

class RCExplainer(torch.nn.Module):
    def __init__(
        self,
        pred_model: BaseModel,
        num_nodes: int,
        input_dim: int,
        device: str,
        node_labels: bool,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.device = device
        self.node_labels = node_labels
        self.pred_model = pred_model
        self.elayers = torch.nn.Sequential(
            torch.nn.Linear(input_dim * 2, 64), torch.nn.ReLU(), torch.nn.Linear(64, 1)
        ).to(device)
        self.row = torch.arange(num_nodes).repeat_interleave(num_nodes).to(device)
        self.col = torch.arange(num_nodes).repeat(num_nodes).to(device)

    def concrete_sample(self, log_alpha):
        if not self.training:
            return torch.sigmoid(log_alpha)
        random_noise = log_alpha.uniform_(0, 1).to(self.device)
        gate_inputs = torch.log(random_noise) - torch.log(1.0 - random_noise)
        return torch.sigmoid(gate_inputs + log_alpha)

    def forward(self, x, adj, edge):
        x = x.unsqueeze(0).to(self.device)
        adj = adj.unsqueeze(0).to(self.device)
        edge = edge.unsqueeze(0).to(self.device)
        gt_emb = (
            self.pred_model.encoder.forward(x, adj, edge, self.node_labels, pool=False)
            .detach()
            .clone()
            .squeeze()
        )
        h = torch.cat([gt_emb[self.row], gt_emb[self.col]], dim=-1).to(self.device)
        h = self.elayers(h).reshape(-1)
        values = self.concrete_sample(h)
        mask = torch.sparse_coo_tensor(
            indices=torch.cat(
                [self.row.unsqueeze(-1), self.col.unsqueeze(-1)], dim=-1
            ).T,
            values=values,
            size=[self.num_nodes, self.num_nodes],
        ).to(self.device)
        mask = mask.coalesce().to_dense().float()
        sym_mask = (mask + mask.T) / 2
        masked_adj = adj * sym_mask
        inv_masked_adj = (1.0 - sym_mask) * adj
        masked_emb = self.pred_model(x, masked_adj, edge, node_labels=self.node_labels)[
            "graph_emb"
        ]
        inv_emb = self.pred_model(
            x, inv_masked_adj, edge, node_labels=self.node_labels
        )["graph_emb"]
        return masked_adj, gt_emb, masked_emb, inv_emb, mask

    def loss(self, gt_emb, masked_emb, inv_emb, boundaries, mask):
        boundary_loss = 0
        for boundary in boundaries:
            gt_proj = (gt_emb * boundary[:-1]).sum() + boundary[-1]
            ft_proj = (masked_emb * boundary[:-1]).sum() + boundary[-1]
            boundary_loss += F.sigmoid(-1.0 * (gt_proj * ft_proj))
        boundary_loss = 12 * (boundary_loss / len(boundaries))

        min_inv_loss = 1e10
        for boundary in boundaries:
            gt_proj = (gt_emb * boundary[:-1]).sum() + boundary[-1]
            inv_proj = (inv_emb * boundary[:-1]).sum() + boundary[-1]
            inv_loss = F.sigmoid(gt_proj * inv_proj)
            min_inv_loss = min(min_inv_loss, inv_loss)
        inverse_boundary_loss = 12 * min_inv_loss
        net_boundary_loss = boundary_loss + inverse_boundary_loss
        mask = mask * 0.99 + 0.005
        mask_ent_loss = (-mask * mask.log() - (1 - mask) * (1 - mask).log()).mean()
        return net_boundary_loss + mask.sum() * 0.001 + 8.0 * mask_ent_loss


def get_embeddings(dataset: GraphDataset, gnn: BaseModel, batch_size) -> DataLoader:
    (all_x, all_adj, all_edge_attr, all_idx, _) = get_attribute_lists_from_data_list(
        dataset.graphs, gnn.device
    )
    undesired_idxs = []
    for i in range((len(dataset) // batch_size) + ((len(dataset) % batch_size) != 0)):
        outputs = gnn(
            all_x[i * batch_size : (i + 1) * batch_size],
            all_adj[i * batch_size : (i + 1) * batch_size],
            all_edge_attr[i * batch_size : (i + 1) * batch_size],
            node_labels=dataset.node_labels,
        )
        for idx, graph_idx in enumerate(all_idx[i * batch_size : (i + 1) * batch_size]):
            if dataset[graph_idx]["graph_label"] == dataset.undesired_label:
                undesired_idxs.append(graph_idx)
            dataset[graph_idx]["graph_emb"] = outputs["graph_emb"][idx].detach().cpu()
            dataset[graph_idx]["graph_label"] = (
                outputs["graph_pred"][idx].argmax(dim=-1).int().item()
            )
        del outputs
    del all_x, all_adj, all_edge_attr
    torch.cuda.empty_cache()
    return dataset, undesired_idxs


def mine_invariant(match_mat_pos, match_mat_neg):
    buf_pos_idxs = torch.arange(match_mat_pos.shape[0])
    buf_neg_idxs = torch.arange(match_mat_neg.shape[0])
    steps = 0
    inv = torch.zeros(match_mat_pos.shape[1]).bool()
    while True:
        steps += 1
        num = (match_mat_pos[buf_pos_idxs] == 0).sum(dim=0) + 1e-5
        denom = (match_mat_neg[buf_neg_idxs] == 0).sum(dim=0) + 1e-10
        ratio_vec = num / denom
        ratio_vec[inv] = 1e10
        sel_j = ratio_vec.argmin()
        inv[sel_j] = True
        buf_pos_idxs = buf_pos_idxs[
            match_mat_pos[buf_pos_idxs, sel_j].nonzero().squeeze(-1)
        ]
        buf_neg_idxs = buf_neg_idxs[
            match_mat_neg[buf_neg_idxs, sel_j].nonzero().squeeze(-1)
        ]
        if buf_neg_idxs.shape[0] == 0 or steps > 110:
            break

    pos_match = (match_mat_pos[:, inv].sum(dim=-1) == inv.sum()).sum()
    neg_match = (match_mat_neg[:, inv].sum(dim=-1) == inv.sum()).sum()

    for idx in inv.nonzero().squeeze(-1):
        temp = inv.clone()
        temp[idx] = False
        temp_pos = (match_mat_pos[:, temp].sum(dim=-1) == temp.sum()).sum()
        temp_neg = (match_mat_neg[:, temp].sum(dim=-1) == temp.sum()).sum()

        if temp_pos == pos_match and temp_neg == neg_match:
            inv[idx] = False
    return inv


def extract_rules(dataset: GraphDataset, pred_model: BaseModel):
    class_idx, data_embs, labels, idx2rule, rules = {}, [], [], {}, []
    for i, sample in enumerate(dataset):
        if sample["graph_label"] not in class_idx:
            class_idx[sample["graph_label"]] = []
        class_idx[sample["graph_label"]].append(i)
        data_embs.append(sample["graph_emb"])
        labels.append(sample["graph_label"])
    data_embs = torch.stack(data_embs, dim=0).to(pred_model.device)
    labels = torch.tensor(labels)
    for idx in tqdm(random.sample(list(range(len(dataset))), len(dataset))):
        if len(idx2rule) == len(dataset):
            break
        cur_sample, rule_bb, opp_class = dataset[idx], [], []
        neg_idxs = [
            neg_idx
            for key in class_idx
            if key != cur_sample["graph_label"]
            for neg_idx in class_idx[key]
        ]
        for neg_idx in random.sample(neg_idxs, min(100, len(neg_idxs))):
            pos = data_embs[idx].to(pred_model.device).unsqueeze(0)
            neg = data_embs[neg_idx].to(pred_model.device).unsqueeze(0)
            while True:
                boundary_pt = 0.9 * pos + 0.1 * neg
                pred = pred_model.predictor(boundary_pt)
                class2, class1 = pred.argsort().squeeze()[-2:]
                if (
                    pred[0, class1] - pred[0, class2]
                ) ** 2 < 0.00001 and class1 == cur_sample["graph_label"]:
                    break
                if class1 == cur_sample["graph_label"]:
                    pos = boundary_pt
                else:
                    neg = boundary_pt
            opp_class.append(class2)
            basis_list, bias_list = [], []
            boundary_pt.requires_grad = True
            pred_model.eval()
            for i in range(len(class_idx)):
                pred_model.zero_grad()
                pred = pred_model.predictor(boundary_pt).reshape(1, -1)
                pred[:, i].backward()
                basis = boundary_pt.grad.detach().clone()
                basis_list.append(basis.reshape(1, -1).cpu())
                bias = pred[:, i].detach().clone() - (
                    boundary_pt.detach().clone() * basis
                ).sum(dim=1)
                bias_list.append(bias.cpu())
            boundary_pt.requires_grad = False
            basis_list = torch.stack(basis_list, dim=2).squeeze(0)
            bias_list = torch.stack(bias_list, dim=1)
            bb = torch.concat([basis_list, bias_list], dim=0).to(boundary_pt.device)
            exp_boundary_pt = torch.concat(
                [boundary_pt, torch.ones(1, 1).to(boundary_pt.device)], dim=-1
            )
            hash_vals = torch.mm(exp_boundary_pt, bb).squeeze()
            class2, class1 = hash_vals.argsort().squeeze()[-2:]
            rule_bb.append((bb[:, class1] - bb[:, class2]).unsqueeze(-1))
        rule_bb = torch.concat(rule_bb, dim=-1)
        exp_embs = torch.concat(
            [data_embs, torch.ones(data_embs.shape[0], 1).to(data_embs.device)], dim=-1
        )
        hash_vals = torch.mm(exp_embs, rule_bb)
        configs = torch.zeros(hash_vals.shape)
        configs[hash_vals > 0] = 1

        indi_f = labels == cur_sample["graph_label"]
        match_mat_f = configs[indi_f, :] - configs[idx] == 0
        match_mat_g = configs[~indi_f, :] - configs[idx] == 0
        invariant = mine_invariant(match_mat_f, match_mat_g)

        boundaries = []
        for inv_ix in invariant.nonzero().squeeze(-1):
            boundaries.append({"basis": rule_bb[:, inv_ix], "label": opp_class[inv_ix]})
        rules.append({"label": cur_sample["graph_label"], "boundary": boundaries})

        sub_hash = torch.mm(exp_embs, rule_bb[:, invariant])
        subconfigs = torch.zeros(sub_hash.shape)
        subconfigs[sub_hash > 0] = 1
        covered_idx = (subconfigs - configs[idx][invariant] != 0).sum(dim=-1) == 0

        for c_idx in covered_idx.nonzero().squeeze(-1):
            idx2rule[c_idx.item()] = len(rules) - 1
    return {"rules": rules, "idx2rules": idx2rule}


def run_rcexplainer(
    pred_model: BaseModel, dataset: GraphDataset, args: argparse.Namespace
) -> list[Graph]:
    pred_model.eval()
    dataset, undesired_idxs = get_embeddings(dataset, pred_model, args.batch_size)
    print("Extracting rules")
    rules_dict = extract_rules(dataset, pred_model)
    print("Training explainer")
    explainer = RCExplainer(
        pred_model,
        dataset.max_num_nodes,
        args.embedding_dim,
        args.device,
        dataset.node_labels,
    )
    optimizer = torch.optim.Adam(
        [p for n, p in explainer.named_parameters() if "model" not in n], lr=args.gnn_lr
    )
    batch_loss = 0
    for idx, graph in enumerate(tqdm(dataset, desc="Train RCExplainer")):
        rule = rules_dict["rules"][rules_dict["idx2rules"][idx]]
        boundaries = []
        for b_num in range(len(rule["boundary"])):
            boundaries.append(rule["boundary"][b_num]["basis"].to(args.device))
        _, gt_emb, masked_emb, inv_emb, mask = explainer(
            graph["x"], graph["adj_matrix"], graph["edge_attr_matrix"]
        )
        batch_loss += explainer.loss(gt_emb, masked_emb, inv_emb, boundaries, mask)
        if (idx + 1) %  args.batch_size == 0:
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            batch_loss = 0
    cf_examples = []
    for idx in tqdm(undesired_idxs, desc="Get counterfactuals"):
        graph = dataset[idx]
        masked_adj = explainer(
            graph["x"], graph["adj_matrix"], graph["edge_attr_matrix"]
        )[0]
        cf_examples.append(
            Graph(
                x=graph["x"],
                adj_matrix=masked_adj.detach().cpu().squeeze(0),
                edge_attr_matrix=graph["edge_attr_matrix"],
                graph_idx=graph["graph_idx"],
                graph_label=int(not graph["graph_label"]),
            )
        )
    return cf_examples
