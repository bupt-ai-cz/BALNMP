import numpy as np
import torch
import pandas as pd
from sklearn import metrics
from torch.nn import functional as F


def disable_dropout(model):
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train(False)


def merge_result(id_list, label_list, score_list, method):
    """Merge predicted results of all bags for each patient"""

    assert method in ["max", "mean"]
    merge_method = np.max if method == "max" else np.mean

    df = pd.DataFrame()
    df["id"] = id_list
    df["label"] = label_list
    df["score"] = score_list
    # https://www.jb51.cc/python/438695.html
    df = df.groupby(by=["id", "label"])["score"].apply(list).reset_index()
    df["bag_num"] = df["score"].apply(len)
    df["score"] = df["score"].apply(merge_method, args=(0,))

    return df["id"].tolist(), df["label"].tolist(), df["score"].tolist(), df["bag_num"].tolist()


def compute_confusion_matrix(label_list, predicted_label_list, num_classes=2):
    label_array = np.array(label_list)
    predicted_label_array = np.array(predicted_label_list)
    confusion_matrix = np.bincount(num_classes * label_array + predicted_label_array, minlength=num_classes**2).reshape((num_classes, num_classes))

    return confusion_matrix


def compute_metrics(label_list, predicted_label_list):
    confusion_matrix = compute_confusion_matrix(label_list, predicted_label_list)
    tn, fp, fn, tp = confusion_matrix.flatten()

    acc = (tn + tp) / (tn + fp + fn + tp)
    sens = tp / (tp + fn)
    spec = tn / (tn + fp)
    ppv = tp / (tp + fp)
    npv = tn / (tn + fn)

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)

    return {"acc": acc, "sens": sens, "spec": spec, "ppv": ppv, "npv": npv, "f1": f1}


def get_best_metrics_and_threshold(label_list, score_list):
    best_metric_dict = {"acc": 0, "sens": 0, "spec": 0, "ppv": 0, "npv": 0, "f1": 0}
    best_threshold = 0

    # search the best metrcis with F1 score (the greater is better)
    for threshold in np.linspace(0, 1, 1000):
        metric_dict, _ = compute_metrics_by_threshold(label_list, score_list, threshold)
        if metric_dict["f1"] > best_metric_dict["f1"]:
            best_metric_dict = metric_dict
            best_threshold = threshold
    best_metric_dict["auc"] = compute_auc(label_list, score_list)

    return best_metric_dict, best_threshold


def compute_metrics_by_threshold(label_list, score_list, threshold):
    # bag will be predicted as the positive (label is 1) when the score is greater than threshold
    predicted_label_list = [1 if score >= threshold else 0 for score in score_list]
    metric_dict = compute_metrics(label_list, predicted_label_list)
    metric_dict["auc"] = compute_auc(label_list, score_list)

    return metric_dict, threshold


def compute_auc(label_list, score_list, multi_class="raise"):
    try:
        # set "multi_class" for computing auc of 2 classes ("raise") and multiple classes ("ovr" or "ovo"), https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
        auc = metrics.roc_auc_score(label_list, score_list, multi_class=multi_class)
    except ValueError:
        auc = 0  # handle error when there is only 1 classes in "label_list"
    return auc


def save_checkpoint(model, save_path):
    torch.save(model.state_dict(), save_path)
    # print(f"save {save_path}")


def train_val_test_binary_class(task_type, epoch, model, data_loader, optimizer, recoder, writer, merge_method):
    total_loss = 0
    label_list = []
    score_list = []  # [score_bag_0, score_bag_1, ..., score_bag_n]
    id_list = []
    patch_path_list = []
    attention_value_list = []  # [attention_00, attention_01, ..., attention_10, attention_11, ..., attention_n0, attention_n1, ...]

    if task_type == "train":
        model.train()
        for index, item in enumerate(data_loader, start=1):
            print(f"\repoch: {epoch}, {task_type}, [{index}/{len(data_loader)}]", end="")
            bag_tensor, label = item["bag_tensor"].cuda(), item["label"].cuda()
            clinical_data = item["clinical_data"][0].cuda() if "clinical_data" in item else None

            optimizer.zero_grad()
            output, attention_value = model(bag_tensor, clinical_data)
            loss = F.cross_entropy(output, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            id_list.append(item["patient_id"][0])
            label_list.append(label.item())
            score = F.softmax(output, dim=-1).squeeze(dim=0)[1].cpu().item()  # use the predicted positive probability as score
            score_list.append(score)
            patch_path_list.extend([p[0] for p in item["patch_paths"]])
            attention_value_list.extend(attention_value[0].cpu().tolist())
    else:
        # model.eavl()
        disable_dropout(model)
        with torch.no_grad():
            for index, item in enumerate(data_loader, start=1):
                print(f"\repoch: {epoch}, {task_type}, [{index}/{len(data_loader)}]", end="")
                bag_tensor, label = item["bag_tensor"].cuda(), item["label"].cuda()
                clinical_data = item["clinical_data"][0].cuda() if "clinical_data" in item else None

                output, attention_value = model(bag_tensor, clinical_data)
                loss = F.cross_entropy(output, label)
                total_loss += loss.item()

                id_list.append(item["patient_id"][0])
                label_list.append(label.item())
                score = F.softmax(output, dim=-1).squeeze(dim=0)[1].cpu().item()  # use the predicted positive probability as score
                score_list.append(score)
                patch_path_list.extend([p[0] for p in item["patch_paths"]])
                attention_value_list.extend(attention_value[0].cpu().tolist())

    recoder.record_attention_value(patch_path_list, attention_value_list, epoch)
    if merge_method != "not_use":
        id_list, label_list, score_list, bag_num_list = merge_result(id_list, label_list, score_list, merge_method)
        recoder.record_score_value(id_list, label_list, bag_num_list, score_list, epoch)

    average_loss = total_loss / len(data_loader)
    metrics_dict, threshold = compute_metrics_by_threshold(label_list, score_list, 0.5)

    print(
        f"\repoch: {epoch}, {task_type}, loss: {average_loss:.3f}, threshold: {threshold}, auc: {metrics_dict['auc']:.3f}, acc: {metrics_dict['acc']:.3f}, sens: {metrics_dict['sens']:.3f}, spec: {metrics_dict['spec']:.3f}, ppv: {metrics_dict['ppv']:.3f}, npv: {metrics_dict['npv']:.3f}, f1: {metrics_dict['f1']:.3f}"
    )

    writer.add_scalars("comparison/loss", {f"{task_type}_loss": average_loss}, epoch)
    writer.add_scalars("comparison/auc", {f"{task_type}_auc": metrics_dict["auc"]}, epoch)
    writer.add_scalars(f"metrics/{task_type}", metrics_dict, epoch)

    return metrics_dict["auc"]


def train_val_test_multi_class(task_type, epoch, model, data_loader, optimizer, recoder, writer, merge_method):
    total_loss = 0
    label_list = []
    score_list = []  # [[score_0_bag_0, score_1_bag_0, score_2_bag_0], [score_0_bag_1, score_1_bag_1, score_2_bag_1], ...]
    id_list = []
    patch_path_list = []
    attention_value_list = []  # [attention_00, attention_01, ..., attention_10, attention_11, ..., attention_n0, attention_n1, ...]

    if task_type == "train":
        model.train()
        for index, item in enumerate(data_loader, start=1):
            print(f"\repoch: {epoch}, {task_type}, [{index}/{len(data_loader)}]", end="")
            bag_tensor, label = item["bag_tensor"].cuda(), item["label"].cuda()
            clinical_data = item["clinical_data"][0].cuda() if "clinical_data" in item else None

            optimizer.zero_grad()
            output, attention_value = model(bag_tensor, clinical_data)
            loss = F.cross_entropy(output, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            id_list.append(item["patient_id"][0])
            label_list.append(label.item())
            score = F.softmax(output, dim=-1).squeeze(dim=0).detach().cpu().numpy()
            score_list.append(score)
            patch_path_list.extend([p[0] for p in item["patch_paths"]])
            attention_value_list.extend(attention_value[0].cpu().tolist())
    else:
        # model.eavl()
        disable_dropout(model)
        with torch.no_grad():
            for index, item in enumerate(data_loader, start=1):
                print(f"\repoch: {epoch}, {task_type}, [{index}/{len(data_loader)}]", end="")
                bag_tensor, label = item["bag_tensor"].cuda(), item["label"].cuda()
                clinical_data = item["clinical_data"][0].cuda() if "clinical_data" in item else None

                output, attention_value = model(bag_tensor, clinical_data)
                loss = F.cross_entropy(output, label)
                total_loss += loss.item()

                id_list.append(item["patient_id"][0])
                label_list.append(label.item())
                score = F.softmax(output, dim=-1).squeeze(dim=0).detach().cpu().numpy()
                score_list.append(score)
                patch_path_list.extend([p[0] for p in item["patch_paths"]])
                attention_value_list.extend(attention_value[0].cpu().tolist())

    recoder.record_attention_value(patch_path_list, attention_value_list, epoch)
    if merge_method != "not_use":
        id_list, label_list, score_list, bag_num_list = merge_result(id_list, label_list, score_list, merge_method)
        recoder.record_score_value(id_list, label_list, bag_num_list, score_list, epoch)

    average_loss = total_loss / len(data_loader)
    # compute AUC of multiple classification with "ovr" setting, https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
    auc = compute_auc(label_list, score_list, multi_class="ovr")
    label_predicted = np.argmax(np.asarray(score_list), axis=-1)
    confusion_matrix = metrics.confusion_matrix(label_list, label_predicted)

    print(f"\repoch: {epoch}, {task_type}, loss: {average_loss:.3f}, auc: {auc:.3f}")
    print(f"confusion matrix: {confusion_matrix}")
    writer.add_scalars("comparison/loss", {f"{task_type}_loss": average_loss}, epoch)
    writer.add_scalars("comparison/auc", {f"{task_type}_auc": auc}, epoch)

    return auc
