"""
This module contains the DatasetBuilder class which is used to build datasets for ESM
input.

Classes:
    DatasetBuilder: A class to build and preprocess datasets for ESM models.
"""

import os
import sys
from collections import defaultdict
from logging import Logger

import pandas as pd
from Bio.Data.IUPACData import protein_letters, protein_letters_1to3
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)

sys.path.append(ROOT_DIR)


class DatasetBuilder:
    """build the dataset used for esm input.

    Note:
        With variant data, which contains LOF, GOF label and
        mutated sequence, build dataset.

        variant_df must contains
        labels: str|float, sequence: str, RefSeq: str, GeneName: str, tag:str
        variant_df could be made by rawdataprocessor
    """

    def __init__(self, config: dict, variant_df: pd.DataFrame, logger: Logger):
        self.config = config
        self.variant_df = variant_df
        self.logger = logger

    def build_dataset(self) -> None:
        """Build various dataset for esm model.

        Note:
            data preprocessing to build dataset.
            label encoding and train-test split will be done.

        Examples:
            >>> dataset_builder = DatasetBuilder(
                config = config, variant_df = variant_df, logger = logger
                )
            >>> dataset_builder.build_dataset()
            {"train": defaultdict(list), "test": defaultdict(list)}
        """
        self.logger.info("start to build dataset.")
        self._split_data()

        if self.variant_df["labels"].dtype != float:
            self._one_hot_encoding()
        elif self.config.db_processing == "all_generated":
            self.logger.info("no need to encode labels.")
        else:
            self._scale_dataset()

        return self.return_dataset()

    def _split_data(self) -> None:
        """Depends on self.config.db_processing, do train test split.

        Note:
            possible db_processing are,
            - 5fold
            - stratified_5fold
            - all_test
            - all_generated

        Raises:
            ValueError: if self.config.db_processing is not matched to
            possible processing, raise error.
        """
        if self.config.db_processing == "5fold":
            self._five_fold()
        elif self.config.db_processing == "stratified_5fold":
            self._five_fold(stratify=True)
        elif self.config.db_processing == "all_test":
            self.variant_df["split"] = "test"
        elif self.config.db_processing == "all_generated":
            self._generate_all()
            self.variant_df["split"] = "test"
        else:
            raise ValueError(
                f"not predefined db_processing {self.config.db_processing}"
            )

        self.logger.info(
            "number of train test split are "
            + ", ".join(
                [
                    f"{k}: {v}"
                    for k, v in self.variant_df["split"].value_counts().items()
                ]
            ),
        )

        return

    def _five_fold(self, stratify: bool = False) -> None:
        """Split data into five fold and use self.config.test_fold to
        set train and test.

        Note:
            Use labels to stratify

        Args:
            stratify (bool): is stratify. Defaults to False.

        Raises:
            ValueError: if testfold is not one of 0~4.
            ValueError: if stratify and groups are not allowed combination.
        """
        if self.config.test_fold not in [0, 1, 2, 3, 4]:
            raise ValueError(
                f"test_fold must in range(0,5), {self.config.test_fold} given."
            )

        if stratify:
            skf = StratifiedKFold(
                n_splits=5, shuffle=True, random_state=self.config.random_seed
            )
            splits = skf.split(self.variant_df, self.variant_df["labels"])
            self.logger.info("StratifiedKFold Done.")
        else:
            kf = KFold(n_splits=5, shuffle=True, random_state=self.config.random_seed)
            splits = kf.split(self.variant_df)
            self.logger.info("KFold Done.")

        for fold, (_, test_index) in enumerate(splits):
            if fold == self.config.test_fold:
                self.variant_df["split"] = "train"
                self.variant_df.loc[test_index, "split"] = "test"
                break

        return

    def _generate_all(self) -> None:
        """With reference sequence, generate all possible missense mutation"""
        for_df_dict = defaultdict(list)

        ref_seq = self.variant_df.iloc[0]["RefSeq"]
        gene_name = self.variant_df.iloc[0]["GeneName"]
        refseq_id = self.variant_df.iloc[0]["refseqID"]
        tag = self.variant_df.iloc[0]["tag"]

        for_df_dict["labels"].append(0.0)
        for_df_dict["GeneName"].append(gene_name)
        for_df_dict["refseqID"].append(refseq_id)
        for_df_dict["HGVSp"].append("p.Met1Met")
        for_df_dict["RefSeq"].append(ref_seq)
        for_df_dict["sequence"].append(ref_seq)
        for_df_dict["tag"].append(tag)

        for idx, original_aa in enumerate(ref_seq):
            for aa in protein_letters:
                if aa != original_aa:
                    mutated_seq = ref_seq[:idx] + aa + ref_seq[idx + 1 :]

                    for_df_dict["labels"].append(float(len(for_df_dict["labels"])))
                    for_df_dict["GeneName"].append(gene_name)
                    for_df_dict["refseqID"].append(refseq_id)
                    for_df_dict["HGVSp"].append(
                        f"p.{protein_letters_1to3[original_aa]}"
                        f"{idx}{protein_letters_1to3[aa]}"
                    )
                    for_df_dict["RefSeq"].append(ref_seq)
                    for_df_dict["sequence"].append(mutated_seq)
                    for_df_dict["tag"].append(tag)

        self.variant_df = pd.DataFrame(for_df_dict)

        self.logger.info(f"{len(for_df_dict['labels'])} sequences are generated.")

        return

    def _one_hot_encoding(self) -> None:
        """Encode label of self.variant_df.

        Note:
            Value of label column will be changed.

        Examples:
            >>> self.variant_df["labels"]
            0   GOF
            1   LOF
            ...

            >>> self._one_hot_encoding()

            >>> self.variant_df["labels"]
            0   [1.0, 0.0]
            1   [0.0, 1.0]
            ...
        """
        sorted_labels = sorted(self.variant_df["labels"].unique())
        self.variant_df["labels"] = (
            pd.get_dummies(self.variant_df["labels"], columns=sorted_labels)[
                sorted_labels
            ]
            .astype(float)
            .values.tolist()
        )
        self.logger.info("add_one_hot_encoded_label Done.")
        self.logger.info("----------------------------------------------/")

        return

    def _scale_dataset(self) -> None:
        """Scale label of dataset. The label refers to the score of the
        experiment on the function of the protein.

        Note:
            Fit train_y using MinMaxScaler and transform both train_y and
            test_y.
        """
        train_y = []
        test_y = []

        for i, row in self.variant_df.iterrows():
            if row["split"] == "test":
                test_y.append(row["labels"])
            elif row["split"] == "train":
                train_y.append(row["labels"])

        scaler = MinMaxScaler()
        train_y_scaled = scaler.fit_transform([[y] for y in train_y])

        test_y_scaled = scaler.transform([[y] for y in test_y])

        train_idx = 0
        test_idx = 0

        for i, row in self.variant_df.iterrows():
            if row["split"] == "test":
                self.variant_df.at[i, "labels"] = test_y_scaled[test_idx][0]
                test_idx += 1
            elif row["split"] == "train":
                self.variant_df.at[i, "labels"] = train_y_scaled[train_idx][0]
                train_idx += 1

        self.logger.info(f"len_train: {len(train_y)}, len_test: {len(test_y)}")
        self.logger.info(
            "min max scaling with "
            + f"min: {scaler.data_min_[0]}, max: {scaler.data_max_[0]}",
        )

        return

    def return_dataset(self) -> dict:
        """Return dataset which can used as esm input.

        Returns:
            dict: dictionary of defaultdict(list). it looks like
            {"train": {"labels":[...],...}, "test": {"labels":[...],...}}

        Examples:
            >>> self.variant_df
                labels      ...     RefSeq  sequence  ...   split
            0   [1.0, 0.0]  ...     AAA     AAA       ...   test
            1   [0.0, 1.0]  ...     BBB     BBB       ...   train
            >>> self.return_dataset()
            {"train": {"labels":[...],...},
            "test": {"labels":[[1.0, 0.0],...],"sequence":["AAA",...]
            ,"RefSeq":["AAA",...]}}
        """

        dataset = {"train": defaultdict(list), "test": defaultdict(list)}
        for _, row in self.variant_df.iterrows():
            for col in [
                "labels",
                "sequence",
                "RefSeq",
                "GeneName",
                "refseqID",
                "HGVSp",
            ]:
                if col == "labels" and isinstance(row[col], str) and "[" in row[col]:
                    dataset[row["split"]][col].append(eval(row[col]))
                else:
                    dataset[row["split"]][col].append(row[col])

        return dataset

    def load_variant_data(self, input_path: str) -> None:
        """Load variant tsv file as self.variant_df

        Note:
            input_path must be tsv. it must have labels and sequence.

        Args:
            input_path (str): absolute path for variant tsv file.

        Examples:
            >>> self.load_variant_data({variant data path})
            >>> self.variant_df
            labels	GeneName    trx_id      HGVSp   RefSeq	sequence    tag
            LOF	    AAAS        NM_015.5    p.Q15K	MC...   MC...       hgmd
        """
        self.variant_df = pd.read_csv(input_path, index_col=0)

        self.logger.info(f"{len(self.variant_df)} variants are loaded.")
        self.logger.info("load_variant_data Done.")
        self.logger.info("----------------------------------------------/")

        return

    def write_dataset(self, output_path: str) -> None:
        """Write dataset to train or be inferenced by ESM

        Note:
            write self.variant_df.

        Args:
            output_path (str): absolute path for dataset csv file

        Examples:
            >>> self.variant_df
            labels	GeneName    trx_id      HGVSp   RefSeq	sequence    tag
            LOF	    AAAS        NM_015.5    p.Q15K	MC...   MC...       hgmd
        """
        self.variant_df.to_csv(output_path)

        self.logger.info(f"write train test split dataset to {output_path}.")
        self.logger.info("----------------------------------------------/")
        return
