import pandas as pd
import os


class Recoder:
    """Record predicted scores of patients and attention value of each patch during training"""

    def __init__(self, save_path, name):
        super().__init__()

        self.score_df = None
        self.score_save_path = os.path.join(save_path, f"{name}-score.xlsx")

        self.attention_df = None
        self.attention_save_path = os.path.join(save_path, f"{name}-attention.xlsx")

    def record_score_value(self, id_list, label_list, bag_num_list, score_list, epoch):
        temp_df = pd.DataFrame()
        temp_df["p_id"] = id_list
        temp_df["label"] = label_list
        temp_df["bag_num"] = bag_num_list
        temp_df[f"score_{epoch}"] = score_list

        self.score_df = self.score_df.merge(temp_df) if self.score_df is not None else temp_df
        self.save_excel(self.score_df, self.score_save_path)

    def record_attention_value(self, patch_path_list, attention_value_list, epoch):
        temp_df = pd.DataFrame()
        temp_df["patch_paths"] = patch_path_list
        temp_df[f"attention_{epoch}"] = attention_value_list

        self.attention_df = self.attention_df.merge(temp_df) if self.attention_df is not None else temp_df
        self.save_excel(self.attention_df, self.attention_save_path)

    def save_excel(self, df, path):
        df.to_excel(path, index=False)
        # print(f"save {path}")
