import math
import time

import torch
from model.base_model import BaseModel
from torch.utils.data import DataLoader


def train_gnn(
    gnn: BaseModel,
    gnn_train_loader: DataLoader,
    gnn_val_loader: DataLoader,
    learning_rate: float,
    weight_decay: float,
    train_epochs: int,
    results_file_path: str,
    model_path: str
) -> BaseModel:
    optimizer = torch.optim.Adam(
        gnn.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    best_eval_loss, best_eval_acc_g, best_eval_acc_cf = 1e10, 0.0, 0.0
    best_train_loss, best_train_acc_g, best_train_acc_cf = 1e10, 0.0, 0.0
    flag = False
    with open(results_file_path, "w+") as results_file:
        results_file.write("Start training GNN\n")
        start_time = time.time()
        for epoch in range(train_epochs):
            gnn.train()
            train_loss, train_acc_g, train_acc_cf = 0.0, 0.0, 0.0
            for batch in gnn_train_loader:
                optimizer.zero_grad()
                output = gnn.run_one_batch(batch, gnn_train_loader.dataset.dataset.node_labels)
                if flag and math.isnan(output["loss"]):
                    import pdb
                    pdb.set_trace()
                output["loss"].backward()
                optimizer.step()

                train_loss += output["loss"].item()
                train_acc_g += output["cor_graph"].item()
                train_acc_cf += output["cor_cf"].item()
                del output

            train_loss /= len(gnn_train_loader)
            train_acc_g /= len(gnn_train_loader.dataset)  # type: ignore
            train_acc_cf /= len(gnn_train_loader.dataset)  # type: ignore
            results_file.write(
                f"epoch {epoch}: train_loss = {train_loss}, train_acc_g = {train_acc_g}, train_acc_cf = {train_acc_cf}\n"
            )

            if (epoch + 1) % 10 == 0:
                results_file.write("Evaluating GNN\n")
                eval_loss, eval_acc_g, eval_acc_cf = evaluate_gnn(gnn, gnn_val_loader)

                if eval_loss <= best_eval_loss:
                    best_eval_loss = eval_loss
                    best_eval_acc_g = eval_acc_g
                    best_eval_acc_cf = eval_acc_cf
                    best_train_loss = train_loss
                    best_train_acc_g = train_acc_g
                    best_train_acc_cf = train_acc_cf
                    torch.save(gnn.state_dict(), model_path)

                time_checkpoint = time.time()
                time_comsumed = time_checkpoint - start_time
                print(
                    f"epoch: {epoch} | train_loss = {train_loss} | train_acc_g = {train_acc_g} | train_acc_cf = {train_acc_cf} | eval_loss: {eval_loss} | eval_acc_g: {eval_acc_g} | eval_acc_cf: {eval_acc_cf} | time_consumed: {time_comsumed}s"
                )
                results_file.write(
                    f"epoch {epoch}: eval_loss = {eval_loss}, eval_acc_g = {eval_acc_g}, eval_acc_cf = {eval_acc_cf}, time consumed = {time_comsumed}s\n"
                )

        end_time = time.time()
        time_comsumed = end_time - start_time
        results_file.write(
            f"best_train_loss = {best_train_loss}, best_train_acc_g = {best_train_acc_g}, best_train_acc_cf = {best_train_acc_cf}, best_eval_loss = {best_eval_loss}, best_eval_acc_g = {best_eval_acc_g}, best_eval_acc_cf = {best_eval_acc_cf}, time consumed = {time_comsumed}s\n"
        )
        results_file.write("Completed training GNN\n")
    del gnn


def evaluate_gnn(gnn: BaseModel, data_loader: DataLoader) -> tuple[float, float, float]:
    gnn.eval()
    eval_loss, eval_acc_g, eval_acc_cf = 0.0, 0.0, 0.0
    with torch.no_grad():
        for batch in data_loader:
            output = gnn.run_one_batch(batch, data_loader.dataset.dataset.node_labels)
            eval_loss += output["loss"].item()
            eval_acc_g += output["cor_graph"].item()
            eval_acc_cf += output["cor_cf"].item()
            del output
    eval_loss /= len(data_loader)
    eval_acc_g /= len(data_loader.dataset)  # type: ignore
    eval_acc_cf /= len(data_loader.dataset)  # type: ignore
    return eval_loss, eval_acc_g, eval_acc_cf
