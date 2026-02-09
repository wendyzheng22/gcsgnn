import torch
import torch.nn as nn
import torch.nn.functional as F
from data.dataset import Graph
from model.base_model import BaseModel
from model.model_layer import DecodeLayer, GATv2Layer


class ExplainGNN(BaseModel):
    def __init__(
        self,
        num_features: int,  # number of node attributes
        embedding_dim: int,  # size of output embeddings
        num_classes: int,  # number of prediction classes
        num_edge_attr: int,  # number of edge attributes
        max_num_nodes: int,  # max number of nodes a graph in the dataset has
        device: str,
        sub_patterns_per_class: int,  # number of masks / sub patterns per class
        sub_pattern_dim: int,  # length of sub patterns
        encoder_dropout: float, 
        decoder_dropout: float, 
        undesired_label: int,
    ) -> None:
        super().__init__()
        self.device = device
        self.max_num_nodes = max_num_nodes
        self.num_classes = num_classes
        self.sub_patterns_per_class = sub_patterns_per_class
        self.embedding_dim = embedding_dim
        self.sub_pattern_dim = sub_pattern_dim
        self.node_feat_dim = num_features
        self.edge_attr_dim = num_edge_attr
        self.undesired_label = undesired_label

        ## Parameters
        # masks for sub-pattern filtering
        self.masks = nn.Parameter(
            torch.rand(
                sub_patterns_per_class, embedding_dim, sub_pattern_dim, device=device
            )
            * 0.5,
            requires_grad=True,
        )
        # counterfactual sub-patterns to replace masked sub-patterns (sub-patterns are 1d vectors (1 x sub_pattern_dim))
        self.sub_patterns = nn.Parameter(
            torch.rand(sub_patterns_per_class, sub_pattern_dim, device=device) * 0.5,
            requires_grad=True,
        )

        ## Model Components
        self.encoder = GATv2Layer(
            self.node_feat_dim, embedding_dim, self.edge_attr_dim, encoder_dropout
        )
        self.decoder = DecodeLayer(
            embedding_dim,
            self.node_feat_dim,
            self.edge_attr_dim,
            max_num_nodes,
            decoder_dropout,
        )
        self.predictor = nn.Linear(self.embedding_dim, num_classes)
        self.init_parameters()

    def init_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Linear | nn.BatchNorm1d)):
                m.reset_parameters()

    def generate_cf_embeddings(self, embs: torch.Tensor) -> torch.Tensor:
        """
        Global Counterfactual Generation\n
        Applies the matrix masks to filter out the significant inherent sub-patterns
        and substitutes in with the counterfactual sub-patterns
        """
        cf_embs = torch.zeros_like(embs, device=self.device).repeat(
            self.sub_patterns_per_class, 1, 1
        )
        masks = self.masks
        if not self.training:
            masks = (masks >= 0.5).float()
        exp_masks = torch.bmm(self.masks, masks.transpose(1, 2))
        exp_sp = torch.bmm(
            self.sub_patterns.unsqueeze(1), masks.transpose(1, 2)
        ).squeeze(1)
        for i in range(self.sub_patterns_per_class):
            cur_pattern_masked_embs = torch.matmul(
                embs, torch.eye(self.embedding_dim).to(self.device) - exp_masks[i]
            )
            cf_embs[i] = cur_pattern_masked_embs + exp_sp[i]
        return exp_masks, exp_sp, cf_embs

    def decode_embeddings(
        self, graph_emb: torch.Tensor, cf_emb: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        graph_x, graph_adj, graph_edge = self.decoder(graph_emb)
        cf_x = torch.empty_like(graph_x).repeat(self.sub_patterns_per_class, 1, 1, 1)
        cf_adj = torch.empty_like(graph_adj).repeat(
            self.sub_patterns_per_class, 1, 1, 1
        )
        cf_edge = torch.empty_like(graph_edge).repeat(
            self.sub_patterns_per_class, 1, 1, 1, 1
        )
        for i, c_emb in enumerate(cf_emb):
            cf_x[i], cf_adj[i], cf_edge[i] = self.decoder(c_emb)

        return {
            "graph_x": graph_x,
            "graph_adj": graph_adj,
            "graph_edge": graph_edge,
            "cf_x": cf_x,
            "cf_adj": cf_adj,
            "cf_edge": cf_edge,
        }

    def find_best_cfs(
        self,
        model_outputs: dict[str, torch.Tensor],
        x: torch.Tensor,
        adj: torch.Tensor,
        edge: torch.Tensor,
        node_labels: bool = True,
    ) -> torch.Tensor:
        """
        Finds the index of the best cf (has the smallest distance) for each input graph
        """
        cf_x = model_outputs["cf_x"].float()
        if node_labels:
            x = F.one_hot(x.long().squeeze(), num_classes=self.node_feat_dim)
            cf_x = F.one_hot(cf_x.argmax(-1), num_classes=self.node_feat_dim)
        cf_adj = (model_outputs["cf_adj"] >= 0.5).float()
        distance = F.pairwise_distance(x, cf_x).sum(-1) + F.pairwise_distance(
            adj.flatten(1), cf_adj.flatten(2)
        )
        edge_one_hot = F.one_hot(edge.long(), num_classes=self.edge_attr_dim)
        cf_edge = F.one_hot(
            model_outputs["cf_edge"].argmax(-1), num_classes=self.edge_attr_dim
        )
        distance += F.pairwise_distance(edge_one_hot, cf_edge).sum((-1, -2))
        distance[model_outputs["cf_act_preds"] == self.undesired_label] = float("inf")
        return distance.min(-2)[1]

    def pred_loss(
        self, graph_pred: torch.Tensor, cf_preds: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        graph_pred_loss = 1.5 * F.nll_loss(graph_pred, labels)
        undesired_cf_preds = cf_preds[:, labels == self.undesired_label]
        cf_labels = (
            torch.tensor(int(not self.undesired_label))
            .repeat(undesired_cf_preds.shape[1])
            .to(self.device)
        )
        cf_pred_loss = (
            torch.stack(
                [
                    F.nll_loss(undesired_cf_preds[i], cf_labels)
                    for i in range(self.sub_patterns_per_class)
                ]
            )
        ).mean()
        if (labels == self.undesired_label).sum() == 0:
            cf_pred_loss = torch.tensor(0).to(self.device)
        return graph_pred_loss, cf_pred_loss

    def param_loss(
        self, exp_sp: torch.Tensor, graph_emb: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mask_ones = torch.ones(self.sub_patterns_per_class, self.sub_pattern_dim).to(
            self.device
        )
        mask_loss = (
            (self.masks.max(-2)[0] - mask_ones.clone()).abs()
            + (self.masks.abs().sum(-2) - mask_ones.clone()).abs()
        ).sum() / (self.sub_patterns_per_class * self.sub_pattern_dim)
        diagonal = self.sub_patterns.pow(2).sum(-1)
        intra_pattern_loss = (
            (
                torch.mm(exp_sp, exp_sp.T)
                - torch.eye(self.sub_patterns_per_class).to(self.device) * diagonal
            )
            .abs()
            .sum()
        ) / (self.sub_patterns_per_class * self.sub_pattern_dim)
        distance = (
            graph_emb.pow(2).sum(-1, keepdim=True)
            - 2 * torch.mm(graph_emb, exp_sp.T)
            + exp_sp.pow(2).sum(-1)
        )
        sp_dist = distance[labels != self.undesired_label].min(-1)[0].mean()
        if (labels != self.undesired_label).sum() == 0:
            sp_dist = torch.tensor(0).to(self.device)
        return mask_loss, intra_pattern_loss, sp_dist

    def reconst_loss(
        self,
        o_x: torch.Tensor,
        o_adj: torch.Tensor,
        o_edge: torch.Tensor,
        r_x: torch.Tensor,
        r_adj: torch.Tensor,
        r_edge: torch.Tensor,
        node_labels: bool = True,
    ) -> torch.Tensor:
        # weigh each feature based on number of occurances in current batch
        x_count = torch.tensor([(o_x == n).sum() for n in range(self.node_feat_dim)])
        x_weights = torch.zeros_like(x_count)
        x_weights[torch.sort(x_count, descending=True)[1]] = torch.arange(
            0, x_count.shape[0]
        )
        x_weights = (x_weights * 2 + 1).float().to(self.device)
        edge_count = torch.tensor(
            [(o_edge == e).sum() for e in range(self.edge_attr_dim)]
        )
        edge_weights = torch.zeros_like(edge_count)
        edge_weights[torch.sort(edge_count, descending=True)[1]] = torch.arange(
            0, edge_count.shape[0]
        )
        edge_weights = (edge_weights * 2 + 1).float().to(self.device)
        reconst_loss = F.binary_cross_entropy(
            r_adj, o_adj, weight=(o_adj * 2 + 1)
        ) + F.cross_entropy(
            r_edge.permute(0, 3, 1, 2), o_edge.long(), weight=edge_weights
        )
        if node_labels:
            reconst_loss += F.cross_entropy(
                r_x.transpose(1, 2), o_x.long(), weight=x_weights
            )
        else:
            reconst_loss += (
                (r_x - o_x).pow(2).sum((-1, -2)).sqrt().mean()
                / self.node_feat_dim
                / self.max_num_nodes
            )
        return reconst_loss

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        edge_attr: torch.Tensor,
        node_labels: bool = True,
    ) -> dict[str, torch.Tensor]:
        """
        Performs the forward pass on the data passed through
        Params:
            x: list of node features
            adj: list of adjacency matrices
            edge_attr: list of edge attribute matrices
            graph_emb: list of embeddings from encoder
            label: list of graph labels (0 or 1)
        """
        graph_emb = self.encoder.forward(x, adj, edge_attr, node_labels)
        expanded_masks, expanded_sub_patterns, cf_emb = self.generate_cf_embeddings(
            graph_emb
        )
        output = {
            "graph_emb": graph_emb,
            "cf_emb": cf_emb,
            "exp_masks": expanded_masks,
            "exp_sp": expanded_sub_patterns,
        }
        graph_preds = F.log_softmax(self.predictor(graph_emb), dim=-1)
        cf_preds = F.log_softmax(self.predictor(cf_emb), dim=-1)
        output.update({"graph_pred": graph_preds, "cf_pred": cf_preds})
        output.update(self.decode_embeddings(graph_emb, cf_emb))
        return output

    def get_accuracy(
        self,
        model_output: dict[str, torch.Tensor],
        labels: torch.Tensor,
        node_labels: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cf_actual_preds = []
        correct_graph = (model_output["graph_pred"].argmax(dim=1) == labels).sum()
        for i in range(self.sub_patterns_per_class):
            cf_x = model_output["cf_x"][i].float()
            if node_labels:
                cf_x = F.one_hot(
                    cf_x.argmax(-1), num_classes=self.node_feat_dim
                ).float()
            cf_adj = (model_output["cf_adj"][i] >= 0.5).float()
            cf_edge = F.one_hot(
                model_output["cf_edge"][i].argmax(-1), num_classes=self.edge_attr_dim
            ).float()
            emb = self.encoder.forward(cf_x, cf_adj, cf_edge)
            cf_actual_preds.append(
                F.log_softmax(self.predictor(emb), dim=-1).argmax(-1)
            )
        cf_preds = torch.stack(cf_actual_preds)
        correct_cf = ((cf_preds != self.undesired_label).sum(-2) > 0).sum()
        return correct_graph, correct_cf, cf_preds

    def run_one_batch(
        self, batch: dict[str, torch.Tensor], node_labels: bool = True
    ) -> dict[str, torch.Tensor]:
        """
        Does one run with a batch and calculates the loss and number of correct predictions
        Params:
            batch: a dict that contains node features (x), adjacency matrices (adj)
                   edge attribute matrices (edge_attr), and graph labels (label)
        """
        x, adj, edge_attr, labels = (
            batch["x"].to(self.device),
            batch["adj_matrix"].to(self.device),
            batch["edge_attr_matrix"].to(self.device),
            batch["graph_label"].to(self.device),
        )
        outputs = self(x, adj, edge_attr)
        outputs["graph_pred_loss"], outputs["cf_pred_loss"] = self.pred_loss(
            outputs["graph_pred"], outputs["cf_pred"], labels
        )
        (outputs["mask_loss"], outputs["intra_sp_loss"], outputs["sp_dist"]) = (
            self.param_loss(outputs["exp_sp"], outputs["graph_emb"], labels)
        )
        outputs["reconst_loss"] = self.reconst_loss(
            x,
            adj,
            edge_attr,
            outputs["graph_x"],
            outputs["graph_adj"],
            outputs["graph_edge"],
            node_labels=node_labels,
        )

        outputs["loss"] = outputs["graph_pred_loss"] + outputs["reconst_loss"]
        if self.masks.requires_grad:
            outputs["loss"] += (
                outputs["cf_pred_loss"]
                + outputs["mask_loss"]
                + outputs["intra_sp_loss"]
                + outputs["sp_dist"]
            )
        outputs["cor_graph"], outputs["cor_cf"], outputs["cf_act_preds"] = (
            self.get_accuracy(outputs, labels, node_labels)
        )
        outputs["best_cf_indices"] = self.find_best_cfs(
            outputs, x, adj, edge_attr, node_labels
        )
        del x, adj, edge_attr, labels
        return outputs


def get_cfs_from_output(
    model_outputs: dict[str, torch.Tensor],
    graph_idxs: torch.Tensor,
    labels: torch.Tensor,
    undesired_label: int,
    node_labels: bool = True,
) -> list[Graph]:
    B = graph_idxs.shape[0]
    cfs = []
    cf_preds = torch.argmax(
        model_outputs["cf_pred"][model_outputs["best_cf_indices"], range(B), :], dim=-1
    ).cpu()
    is_cf_list = torch.ne(cf_preds, labels)
    for i, is_cf in enumerate(is_cf_list):
        if is_cf and labels[i] == undesired_label:
            cf_idx = model_outputs["best_cf_indices"][i]
            cf_x = (
                model_outputs["cf_x"][cf_idx, i].argmax(-1)
                if node_labels
                else model_outputs["cf_x"][cf_idx, i]
            )
            cfs.append(
                Graph(
                    x=cf_x.detach().float().cpu(),
                    adj_matrix=(model_outputs["cf_adj"][cf_idx, i] >= 0.5)
                    .float()
                    .cpu(),
                    edge_attr_matrix=model_outputs["cf_edge"][cf_idx, i]
                    .argmax(-1)
                    .float()
                    .cpu(),
                    graph_idx=int(graph_idxs[i].item()),
                    graph_label=int(cf_preds[i].item()),
                    cf_idx=cf_idx.item(),
                )
            )
    return cfs
