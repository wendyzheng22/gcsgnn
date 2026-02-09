import argparse
import math
import random

import numpy as np
import torch
import torch.nn.functional as F
from data.data_utils import get_attribute_lists_from_data_list, get_dataloaders
from data.dataset import Graph, GraphDataset
from model.base_model import BaseModel
from model.model_layer import GATv2Layer
from tqdm import tqdm


class Subgraph:
    def __init__(
        self,
        graph_idx: int,
        adj: torch.Tensor,
        node_feat: torch.Tensor,
        edge_feat: torch.Tensor,
        label: int,
        device: str,
    ):
        self.graph_idx = graph_idx
        self.adj = adj.detach().clone().to(device)
        self.node_feat = node_feat.detach().clone().to(device)
        self.edge_feat = edge_feat.detach().clone().to(device)
        self.label = label
        self.deg = torch.diag(sum(self.adj)).to(device)
        self.device = device


class Player:
    def __init__(
        self, G: Graph, device: str, model: BaseModel, node_labels: bool, beta: float = 0.5, 
    ):
        super().__init__()
        self.G = Subgraph(
            G["graph_idx"],
            G["adj_matrix"],
            G["x"],
            G["edge_attr_matrix"],
            G["graph_label"],
            device,
        )
        self.orig_out = G['graph_label']
        self.beta = beta
        self.model = model
        self.node_labels = node_labels
        self.G_last_out, self.G_cur_out = None, None
        self.reset()

    def getCandidates(self):
        cand_dict, adj = [], self.G_curr.adj
        nodes = adj.sum(-1).nonzero().squeeze()
        cand_dict.extend([(i.item(), j.item(), torch.tensor([0.]).to(self.G.device)) for i,j in adj.nonzero().unbind()])
        cand_dict.extend([(i.item(), j.item(), torch.tensor([1.]).to(self.G.device)) for i,j in (adj != 0).nonzero().unbind() if i in nodes and j in nodes])
        if len(cand_dict) > 100:
            cand_dict = random.sample(cand_dict, 100)
        return cand_dict

    def reset(self):
        self.G_last = Subgraph(
            self.G.graph_idx,
            self.G.adj,
            self.G.node_feat,
            self.G.edge_feat,
            self.G.label,
            self.G.device,
        )
        self.G_curr = Subgraph(
            self.G.graph_idx,
            self.G.adj,
            self.G.node_feat,
            self.G.edge_feat,
            self.G.label,
            self.G.device,
        )
        self.cand_dict = self.getCandidates()
        self.cf_cand = []
        self.G_last_out, self.G_cur_out = None, None

    def loss(self):
        out_last = self.G_last_out
        if out_last is None:
            last_probs = self.model(
                self.G_last.node_feat.unsqueeze(0),
                self.G_last.adj.unsqueeze(0),
                self.G_last.edge_feat.unsqueeze(0),
                self.node_labels,
            )["graph_pred"]
            out_last = last_probs.argmax().unsqueeze(0)
        out_probs_curr = self.model(
            self.G_curr.node_feat.unsqueeze(0),
            self.G_curr.adj.unsqueeze(0),
            self.G_curr.edge_feat.unsqueeze(0),
            self.node_labels,
        )["graph_pred"]
        out_curr = out_probs_curr.argmax()
        self.G_cur_out = out_curr.unsqueeze(0)
        pred_same = (out_curr == out_last).float()
        loss_pred = -F.nll_loss(out_probs_curr, out_last.long())
        loss_graph_dist = (self.G.adj - self.G_curr.adj).abs().sum() / 2
        return pred_same * loss_pred + self.beta * loss_graph_dist

    def perturb(self, action):
        self.G_last = Subgraph(
            self.G_curr.graph_idx,
            self.G_curr.adj,
            self.G_curr.node_feat,
            self.G_curr.edge_feat,
            self.G_curr.label,
            self.G_curr.device,
        )
        self.G_last_out = self.G_cur_out
        u, v, _ = self.cand_dict[action]
        if self.G_curr.adj[u][v].item() == 1:
            p_type = "del"
            self.G_curr.adj[u][v] = torch.tensor(0)
            self.G_curr.adj[v][u] = torch.tensor(0)
            self.G_curr.edge_feat[u][v] = torch.tensor(0)
            self.G_curr.edge_feat[v][u] = torch.tensor(0)
        else:
            p_type = "add"
            self.G_curr.adj[u][v] = torch.tensor(1)
            self.G_curr.adj[v][u] = torch.tensor(1)
            self.G_curr.edge_feat[u][v] = torch.tensor(1)
            self.G_curr.edge_feat[v][u] = torch.tensor(1)
        self.G_curr.deg = torch.diag(sum(self.G_curr.adj)).to(self.G_curr.device)
        self.cf_cand.append((self.cand_dict[action], p_type))
        del self.cand_dict[action]


class Env:
    def __init__(self, num_node_feat: int, node_labels: bool):
        self.num_node_feat = num_node_feat
        self.node_labels = node_labels
        self.statedim = num_node_feat + 1

    def step(self, actions: int, player:Player):
        player.perturb(actions)
        return player.loss()

    def getState(self, player:Player):
        deg = player.G_curr.deg.diagonal()
        node_feats = player.G.node_feat
        if self.node_labels:
            node_feats = F.one_hot(player.G.node_feat.long(), self.num_node_feat)
        features = torch.concat([deg.unsqueeze(1), node_feats], dim=1)
        return (features, player.G_curr.adj, player.G_curr.edge_feat)


class PolicyNetwork(torch.nn.Module):
    def __init__(
        self,
        device: str,
        statedim: int,
        node_labels: bool,
        edge_attr_dim: int,
        hidden_dim: int,
        emb_model_num_layers: int = 1,
        mlp_num_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.device = device
        self.dropout = dropout
        self.node_labels = node_labels
        self.embed_model = torch.nn.ModuleList()
        for i in range(emb_model_num_layers):
            if i == 0:
                self.embed_model.append(GATv2Layer(statedim, hidden_dim, edge_attr_dim))
            else:
                self.embed_model.append(
                    GATv2Layer(hidden_dim, hidden_dim, edge_attr_dim)
                )
            self.embed_model.append(torch.nn.BatchNorm1d(hidden_dim))
        layer_sizes = [2 * hidden_dim + 1] + [hidden_dim // 2] * (mlp_num_layers) + [1]
        self.confidence_model = torch.nn.ModuleList(
            [
                torch.nn.Linear(layer_sizes[i], layer_sizes[i + 1])
                for i in range(len(layer_sizes) - 1)
            ]
        )
    
    def init_parameters(self) -> None:
        for layer in self.confidence_model:
            torch.nn.init.xavier_uniform_(layer.weight)
            bound = 1.0 / math.sqrt(layer.in_features)
            torch.nn.init.uniform_(layer.bias, -bound, bound)

    def get_candidate_embs(self, x, cand_dict):
        embs = []
        for i in range(len(cand_dict)):
            u, v, val = cand_dict[i]
            embs.append(torch.cat((x[u], x[v], val), -1))
        return torch.stack(embs)

    def forward(self, x, adj, edge_attr, cand_dict=None):
        for i, layer in enumerate(self.embed_model):
            if i % 2 == 1:
                x = F.leaky_relu(layer(x.squeeze(0)).unsqueeze(0))  # batch norm
                x = F.dropout(x, self.dropout, training=self.training)
            else:
                x = layer(x, adj, edge_attr, self.node_labels, pool=False)
        if x.isnan().any():
            x = torch.nan_to_num(x, nan=1e-14)
        cand_embs = self.get_candidate_embs(x.squeeze(0), cand_dict)
        for i, linear_tranform in enumerate(self.confidence_model):
            cand_embs = linear_tranform(cand_embs)
            if i != len(self.confidence_model) - 1:
                cand_embs = F.leaky_relu(cand_embs)
        return cand_embs, F.softmax(cand_embs, dim=0)

    def get_action(self, probs, train=True):
        if train:
            m = torch.distributions.Categorical(probs.reshape(1, -1))
            highest_prob_action = m.sample()
            prob = probs[highest_prob_action]
            log_prob = m.log_prob(highest_prob_action).squeeze()
        else:
            highest_prob_action = np.argmax(probs.cpu().detach().numpy())
            prob = probs.squeeze(0)[highest_prob_action]
            log_prob = torch.log(prob).squeeze()
        return highest_prob_action, log_prob, prob


def execute_player(
    player: Player,
    env: Env,
    gnn: BaseModel,
    policy: PolicyNetwork,
    budget: int = 15,
    discount: float = 0.6,
    ent: float = 0.1,
    train:bool =True
):
    player.reset()
    rewards, logp_actions, entropy, b, cf = [], [], 0, 0, None
    orig_out = player.orig_out
    curr_out = orig_out
    while orig_out == curr_out and b < budget and len(player.cand_dict) > 0:
        state = env.getState(player)
        _, logits = policy(
            state[0].unsqueeze(0),
            state[1].unsqueeze(0),
            state[2].unsqueeze(0),
            player.cand_dict,
        )
        action, logp_action, _ = policy.get_action(logits, train=train)
        logp_actions.append(logp_action)
        entropy += -1 * (torch.mean(logits) * torch.log(logits)).sum()
        rewards.append(env.step(action, player))
        curr_out = gnn(
            player.G_curr.node_feat.unsqueeze(0),
            player.G_curr.adj.unsqueeze(0),
            player.G_curr.edge_feat.unsqueeze(0),
            player.node_labels,
        )["graph_pred"]
        curr_out = torch.argmax(curr_out).item()
        b += 1
    if orig_out != curr_out:
        cf_adj = player.G.adj.clone()
        cf_edge_attr = player.G.edge_feat.clone()
        for i in range(len(player.cf_cand)):
            u, v, val = player.cf_cand[i][0]
            cf_adj[u, v] = torch.tensor(val == "add").float()
            cf_edge_attr[u,v] = torch.tensor(val == "add").float()
        cf = Graph(
            x=player.G.node_feat.cpu(),
            adj_matrix=cf_adj.cpu(),
            edge_attr_matrix=cf_edge_attr.cpu(),
            graph_idx=player.G.graph_idx,
            graph_label=player.G.label,
        )
    rewards = torch.stack(rewards).detach()
    log_probs = torch.stack(logp_actions)
    disc_rewards = []
    for t in range(len(rewards)):
        Gt, pw = 0, 0
        for r in rewards[t:]:
            Gt, pw = Gt + discount**pw * (1 / r), pw + 1
        disc_rewards.append(Gt)
    disc_rewards = torch.stack(disc_rewards)
    if disc_rewards.shape[0] > 1:  # normalize discounted rewards
        disc_rewards = (disc_rewards - disc_rewards.mean()) / (
            disc_rewards.std() + 1e-9
        )
    p_grad = (-log_probs * disc_rewards).mean() + ent * entropy
    return p_grad, rewards.sum(), cf

def filter_dataset(dataset: GraphDataset, gnn: BaseModel, batch_size:int) -> GraphDataset:
    x, adj, edge, idxs, labels = get_attribute_lists_from_data_list(dataset.graphs, gnn.device)
    gnn_output = []
    for i in range((len(dataset) // batch_size) + ((len(dataset) % batch_size) != 0)):
        gnn_output.append(
            gnn(
                x[i * batch_size : (i + 1) * batch_size],
                adj[i * batch_size : (i + 1) * batch_size],
                edge[i * batch_size : (i + 1) * batch_size],
                node_labels=dataset.node_labels
            )["graph_pred"].detach().cpu()
        )
    predictions = torch.argmax(torch.concat(gnn_output), dim=1).int()
    undesired_graphs_idxs = [
        idxs[i]
        for i, prediction in enumerate(predictions)
        if prediction == dataset.undesired_label and prediction == labels[i]
    ]
    dataset.filter_dataset(undesired_graphs_idxs)
    for i in range(len(dataset)):
        dataset.graphs[i]["graph_label"] = dataset.undesired_label
    x, adj, edge, idxs = 0,0,0,0 
    torch.cuda.empty_cache()
    return dataset

def run_induce(
    pred_model: BaseModel, dataset: GraphDataset, args: argparse.Namespace
) -> list[Graph]:
    pred_model.eval()
    dataset = filter_dataset(dataset, pred_model, args.batch_size)
    train_loader, _, _ = get_dataloaders(dataset, args.batch_size)
    env = Env(dataset.num_node_features, dataset.node_labels)
    policy = PolicyNetwork(
        args.device,
        env.statedim,
        dataset.node_labels,
        dataset.num_edge_attributes,
        args.embedding_dim,
    ).to(args.device)
    opt = torch.optim.Adam(policy.parameters(), lr=args.gnn_lr)
    min_loss, cfs, best_policy, early_stop = 100000000, [], None, 0
    for episode in range(args.gnn_epochs):
        print(f"Episode: {episode}")
        loss, grad, batch_p_grad = 0, 0, []
        for i in tqdm(range(len(train_loader.dataset))):
            g = train_loader.dataset[random.sample(list(range(len(train_loader.dataset))), 1)[0]]
            player = Player(g, args.device, pred_model, dataset.node_labels)
            p_grad, player_loss, _ = execute_player(player, env, pred_model, policy, train=True)
            batch_p_grad.append(p_grad)
            loss += player_loss.cpu()
            grad += p_grad.detach().cpu()
            if (i % args.batch_size == 0 and i != 0) or i == len(train_loader.dataset) - 1:
                batch_p_grad =  torch.stack(batch_p_grad).mean()
                opt.zero_grad()
                batch_p_grad.backward()
                opt.step()
                batch_p_grad = []
                torch.cuda.empty_cache()
        loss = loss / len(train_loader.dataset)
        grad /= len(train_loader.dataset)
        print(f"Episode loss: {loss} | Episode grad: {grad}")
        if loss < min_loss:
            min_loss = loss
            best_policy = policy.state_dict()
            early_stop = 0
        else:
            early_stop += 1
            if early_stop >= 10:
                break
    policy.load_state_dict(best_policy)
    for g in tqdm(dataset):
        p = Player(g, args.device, pred_model, dataset.node_labels)
        _, _, cf = execute_player(p, env, pred_model, policy, train=False)
        if cf is not None:
            cfs.append(cf)
    return cfs
