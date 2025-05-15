"""
This module processes raw data to create a unified format for variant information.

It includes the RawDataProcessor class which handles different databases and formats the
data accordingly.
"""

import os
import re
import sys
from collections import defaultdict
from logging import Logger

import pandas as pd
from Bio.Data.IUPACData import protein_letters_1to3
from omegaconf import DictConfig
from pandas import DataFrame

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)

sys.path.append(ROOT_DIR)

from pipeline.neuron import sequences


class RawDataProcessor:
    """read raw data and make unified data.

    Note:
        raw data contains proteins sequences of reference proteins, and
        variant information with LOF, GOF labels.
        return LOF, GOF label, gene name, transcript id, hgvs code, reference
        sequence, mutated sequence and tag.
        possible db lists are
        1. itanlab
        2. proteingym

    Examples:
        >>> raw_data_processor = RawDataProcessor(**args)
        >>> raw_data_processor.config.db_name
        itanlab
        >>> df = raw_data_processor.process_raw_data()
        >>> df
            labels  GeneName    refseqID    HGVSp   RefSeq  sequence    tag
        0   LOF     geneA       NM_AA       p.      AAA     AAA         hgmd
        1   GOF     geneB       NM_BB       p.      BBB     BBB         clinvar
    """

    def __init__(self, config: DictConfig, logger: Logger):
        self.config = config
        self.logger = logger

        self.gene2seq = defaultdict(set)
        self.trx2seq = defaultdict(set)
        self.prot2seq = defaultdict(set)
        self.uniprot2seq = defaultdict(set)
        self.variant_info_dict = defaultdict(list)

    def process_raw_data(self) -> DataFrame:
        """
        Load raw data and return dataframe with unified format.

        Note:
            - Depends on config.db_name
            - Dataframe contains only one data point per one variant.

        Raises:
            NotImplementedError: If config.db_name is not matched with
            pre-defined options, raise.

        Returns:
            DataFrame: Dataframe contains variant information. ex)

            labels  GeneName    refseqID    HGVSp   RefSeq  sequence    tag
        0   LOF     geneA       NM_AA       p.      AAA     AAA         hgmd
        1   GOF     geneB       NM_BB       p.      BBB     BBB         clinvar
        """
        self.logger.info(f"Start _process_{self.config.db_name}.")

        if self.config.db_name == "itanlab":
            self._process_itanlab()
        elif self.config.db_name == "proteingym":
            self._process_proteingym()
        else:
            raise NotImplementedError(
                f"Processsing {self.config.db_name} is not implemented yet"
            )

        self._remove_too_long_seq()
        self._remove_duplication()

        return self._get_merged_dataframe()

    def _process_itanlab(self) -> None:
        """Process itanlab variant database"""
        self._load_ref_seq()
        self._load_hgmd_variant()
        self._load_clinvar_variant()
        self._make_variant_sequence()

        return

    def _process_proteingym(self) -> None:
        """Process proteingym variant database"""
        self._load_proteingym_variant()

        return

    def _load_ref_seq(self) -> None:
        """Make self.gene2seq {key:gene, value:set of refseq},
           self.trx2seq {key:refseq_id:trx, value:set of refseq},
           self.prot2seq {key:refseq_id:prot, value:set of refseq} and .
           self.uniprot2seq {key:uniprotid, value:set of refseq}

        Note:
            reference sequences from
            self.config.raw_data_path.reviewed_uniprot

            remove the version of refseq_id.
            eg) NM_003405.3 -> NM_003405
            eg) NP_000305.3 -> NP_000305

        Examples:
            >>> self._load_ref_seq()
            >>> self.gene2seq
            {"YWHAH": {"MSSHEGGKK...", ...}, ...}
            >>> self.trx2seq
            {"NM_003405": {"MSSHEGGKK...", ...}, ...}
            >>> self.prot2seq
            {"NP_003405": {"MSSHEGGKK...", ...}, ...}
        """
        self._load_reviewed()
        self.logger.info(
            f"{len(self.gene2seq)} genes, "
            + f"{len(self.trx2seq)} trx, "
            + f"{len(self.prot2seq)} prot, "
            + f"{len(self.uniprot2seq)} uniprot loaded.",
        )
        self.logger.info("_load_ref_seq Done.")
        self.logger.info("----------------------------------------------/")

        return

    def _load_reviewed(self) -> None:
        """
        Read data from self.config.raw_data_path.reviewed_uniprot.
        """
        protein_reference_path = os.path.join(
            ROOT_DIR, self.config.raw_data_path.reviewed_uniprot
        )
        reviewed_df = pd.read_csv(protein_reference_path, sep="\t", low_memory=False)
        for _i, row in reviewed_df.iterrows():
            self.uniprot2seq[row["Entry"].split(".")[0]].add(row["Sequence"])
            if isinstance(row["Gene Names"], str):
                for gene in re.split(r"[;\s]+", row["Gene Names"]):
                    self.gene2seq[gene].add(row["Sequence"])
            if isinstance(row["RefSeq"], str):
                for refseq_id in re.split(r"[;\s]+", row["RefSeq"]):
                    if "NP_" in refseq_id:
                        self.prot2seq[refseq_id.split(".")[0]].add(row["Sequence"])
                    elif "NM_" in refseq_id:
                        self.trx2seq[refseq_id.split(".")[0]].add(row["Sequence"])

        self.logger.info(f"Load {protein_reference_path} done.")

        return

    def _load_hgmd_variant(self) -> None:
        """From HGMD data, get varint info.

        Note:
            information on hgmd_df to use looks like,
            LABEL: LOF or GOF
            GENE: AAAS, ...
            RefSeq: NM_015665.5, ...
            HGVSp: ENSP00000209873.4:p.Phe157CysfsTer16, ...

        Examples:
            >>> self._load_hgmd_variant()
            >>> self.variant_info_dict["var"]
            [("LOF", "AAAS", "NM_015665.5", "p.Phe157CysfsTer16", "hgmd"),...]
        """
        hgmd_df = pd.read_csv(os.path.join(ROOT_DIR, self.config.raw_data_path.hgmd))

        variant_count = 0
        for _i, row in hgmd_df.iterrows():
            label = row["LABEL"]
            gene = row["GENE"]
            refseq_id = row["RefSeq"]
            if ":" in row["HGVSp"]:
                hgvsp = row["HGVSp"].split(":")[1]

                if self._missense_flag(hgvsp):
                    continue

                self.variant_info_dict["var"].append(
                    (label, gene, refseq_id, hgvsp, "hgmd")
                )
                variant_count += 1

        self.logger.info(f"{variant_count} HGMD variants loaded")

        return

    def _missense_flag(self, hgvsp: list[str]) -> bool:
        """
        If self.config.missense_only is True, return True if the variant is not
        missense.
        """
        return self.config.missense_only and any(
            [(x in hgvsp) for x in ["ins", "del", "fs", "Ter", "dup", "?", "WT"]]
        )

    def _load_clinvar_variant(self) -> None:
        """From ClinVar data, get varint info.

        Note:
            information on clinvar_df to use looks like,
            LABEL: LOF or GOF
            GeneSymbol: AAAS, ...
            Name: NM_015665.6(AAAS):c.43C>A (p.Gln15Lys), ...

        Examples:
            >>> self._load_clinvar_variant()
            >>> self.variant_info_dict["var"]
            [("LOF", "AAAS", "NM_015665.5", "p.Gln15Lys", "clinvar"),...]
        """
        clinvar_df = pd.read_csv(
            os.path.join(ROOT_DIR, self.config.raw_data_path.clinvar)
        )

        variant_count = 0
        for _i, row in clinvar_df.iterrows():
            label = row["LABEL"]
            gene = row["GeneSymbol"]
            if "(p." in row["Name"]:
                id_part, variant_part = row["Name"].split(":")
                refseq_id = id_part.split("(")[0]
                hgvsp = variant_part.split("(")[1].replace(")", "")

                if self._missense_flag(hgvsp):
                    continue

                self.variant_info_dict["var"].append(
                    (label, gene, refseq_id, hgvsp, "clinvar")
                )
                variant_count += 1

        self.logger.info(f"{variant_count} ClinVar Variant Loaded")

        return

    def _load_proteingym_variant(self) -> None:
        """From proteingym data, get variant info.

        Note:
            it contains mutated sequences. Do not need to match with
            reference sequences

            mutant,mutated_sequence,DMS_score,DMS_score_bin
            A673C,AAA,-1.01886857127932,1

        Examples:
            >>> self.config.db_option
            A4_HUMAN_Seuma_2022.csv
            >>> self._load_proteingym_variant()
            >>> self.variant_info_dict[("var", "seq")]
            [(0.873783, "A4", "", "p.Gln2Lys", "AQM~", "AKM~","proteingym"),...]
        """
        proteingym_path = os.path.join(
            ROOT_DIR,
            self.config.raw_data_path.proteingym_dir,
            self.config.db_option,
        )
        proteingym_df = pd.read_csv(proteingym_path, low_memory=False)

        gene = self.config.db_option.split("_")[0]

        for i, row in proteingym_df.iterrows():
            if ":" not in row["mutant"]:
                ref_seq, hgvsp = self._restore_ref_from_missense(
                    row["mutated_sequence"],
                    row["mutant"][0],
                    row["mutant"][-1],
                    int(row["mutant"][1:-1]),
                )
                break

        for i, row in proteingym_df.iterrows():
            ref_aa = protein_letters_1to3[row["mutant"][0]]
            var_aa = protein_letters_1to3[row["mutant"][-1]]
            hgvsp = f"p.{ref_aa}{row['mutant'][1]}{var_aa}"
            self.variant_info_dict[("var", "seq")].append(
                (
                    row["DMS_score"],
                    gene,
                    "",
                    hgvsp,
                    ref_seq,
                    row["mutated_sequence"],
                    "proteingym",
                )
            )
        self.logger.info(
            f"{len(self.variant_info_dict[('var', 'seq')])} "
            + "proteingym Variant Loaded",
        )

        return

    def _restore_ref_from_missense(
        self, var_seq: str, ref_1: str, var_1: str, position: int
    ) -> tuple[str, str]:
        """With given missense variant sequence and variant information,
        return reference sequencd and hgvsp

        Args:
            var_seq (str): missense variant sequence
            ref_1 (str): 1 letter aa which is reference
            var_1 (str): 1 letter aa which is variant
            position (int): position of missense

        Raises:
            ValueError: if var_1 and var_seq's aa at position is different,
            it is not right variant sequence and variant information.

        Returns:
            tuple[str, str]: return reference sequence and hgvsp code.

        Examples:
            >>> self._restore_ref_from_missense("AHD", "R","H",2)
            ARD, p.Arg2His
        """
        if var_seq[position - 1] == var_1:
            ref_aa = protein_letters_1to3[ref_1]
            var_aa = protein_letters_1to3[var_1]
            hgvsp = f"p.{ref_aa}{position}{var_aa}"

            return var_seq[: position - 1] + ref_1 + var_seq[position:], hgvsp
        else:
            raise ValueError(f"{var_seq[position - 1]} is not same as {var_1}")

    def _make_variant_sequence(self) -> None:
        """Make variant sequence with variant info and reference sequence.

        Note:
            Use variant sequence if it could be calculated.

        Examples:
            >>> self.variant_info_dict["var"]
            [("LOF", "AAAS", "NM_015665.5", "p.Gln2Lys", "tag"),...]

            >>> self._make_variant_sequence()
            >>> self.variant_info_dict[("var", "seq")]
            [("LOF", "AAAS", "NM_015665.5", "p.Gln2Lys", "AQM~", "AKM~",
              "tag"),...]
        """
        no_avail_variant_cnt = 0

        for variant_info in self.variant_info_dict["var"]:
            label, gene, refseq_id, hgvsp, tag = variant_info

            if "NP_" in refseq_id:
                candidate_seq = self.gene2seq[gene].intersection(
                    self.prot2seq[refseq_id.split(".")[0]]
                )
            elif "NM_" in refseq_id:
                candidate_seq = self.gene2seq[gene].intersection(
                    self.trx2seq[refseq_id.split(".")[0]]
                )
            else:
                candidate_seq = self.uniprot2seq[refseq_id.split(".")[0]]

            variant_make_fail_count = 0
            variant_set = set()
            for ref_seq in candidate_seq:
                try:
                    protein_seq_object = sequences.ProteinSeqObj(
                        reference_seq=ref_seq, hgvs=refseq_id + ":" + hgvsp
                    )
                    variant_set.add((ref_seq, protein_seq_object.mut_seq))
                except:  # noqa: E722
                    variant_make_fail_count += 1
                    continue

            if len(variant_set) == 1:
                ref_seq, var_seq = list(variant_set)[0]
                self.variant_info_dict[("var", "seq")].append(
                    (label, gene, refseq_id, hgvsp, ref_seq, var_seq, tag)
                )
            elif len(variant_set) > 1:
                self.logger.info(f"Warning. {variant_info} has multiple ref.")
            else:
                no_avail_variant_cnt += 1

        variant_count = len(self.variant_info_dict[("var", "seq")])
        self.logger.info(f"{variant_count} variants are merged.")
        self.logger.info(f"{no_avail_variant_cnt} variants could not merged")
        self.logger.info(f"make variant failed {variant_make_fail_count} times")
        self.logger.info("_merge_seq_N_varID Done.")
        self.logger.info("----------------------------------------------/")

        return

    def _remove_too_long_seq(self) -> None:
        """Remove too long protein sequences to train ESM

        Note:
            self.variant_info_dict[("var", "seq")]
        [("GOF", "geneA||geneB", "NM_AA", "WT", "AAA", "AAA", "clinvar"),
        ("LOF", "geneC", "NM_BB||NP_CC", "WT", "CCC", "CCC", "hgmd")]

        Raises:
            ValueError: if no variants are left, raise error
        """
        length_filtered_list = list()
        for variant in self.variant_info_dict[("var", "seq")]:
            if len(variant[4]) <= self.config.max_seq_len:
                length_filtered_list.append(variant)
        self.variant_info_dict[("var", "seq")] = length_filtered_list
        var_num = len(self.variant_info_dict[("var", "seq")])

        self.logger.info(
            "After filtering protein sequences longer than "
            f"{self.config.max_seq_len}, {var_num} variants remained.",
        )
        if var_num == 0:
            raise ValueError("Not enough variants")

        return

    def _remove_duplication(self) -> None:
        """Make only one datapoint per one sequence.

        Note:
            All silent mutations will removed.
            which "=" or "%3D" is presented in mutation code.

            Same mutated sequence with same label will be considered as one.
            e.g. If there are two same mutated sequences,
                 only the first will be recorded.
            {ref_seq:"BAAA", mut_code:"p.B1A", sequence:"AAAA", label:"LOF"}
            {ref_seq:"ABAA", mut_code:"p.B2A", sequence:"AAAA", label:"LOF"}
            Because they have same sequence with same label. Considered as one.

            Both LOF and GOF are labeled on same sequence do not have default.
            In this case, manually determine which label to use.

            if label is float, remove all datapoints, if one sequence has
            multiple labels.

        Examples:
            >>> self.variant_info_dict[("var", "seq")]
            [("LOF", ... , "AAAAA", "tag"), ("LOF", ... , "AAAAA", "tag")]

            >>> self.remove_duplication()
            >>> self.variant_info_dict["done"]
            [("LOF", ... , "AAAAA")]
        """
        dup_check_dict = defaultdict(list)
        label_counter = defaultdict(int)

        for merged_data in self.variant_info_dict[("var", "seq")]:
            hgvs_p = merged_data[3]
            sequence = merged_data[5]
            if "=" in hgvs_p or "%3D" in hgvs_p:
                continue
            dup_check_dict[sequence].append(merged_data)

        for sequence in dup_check_dict:
            label_set = set([x[0] for x in dup_check_dict[sequence]])

            if "itanlab" in self.config.db_name:
                if "LOF" in label_set and "GOF" in label_set:
                    info_list = [x[:4] for x in dup_check_dict[sequence]]
                    self.logger.info(
                        f"Warning. They have same seq, different label {info_list}",
                    )
                elif len(label_set) > 1:
                    dup_check_dict[sequence] = [x for x in dup_check_dict[sequence]]
            else:
                if len(label_set) > 1:
                    continue

            self.variant_info_dict["done"].append(dup_check_dict[sequence][0])

            label_counter[dup_check_dict[sequence][0][0]] += 1

        sequence_count = len(self.variant_info_dict["done"])
        if "itanlab" in self.config.db_name:
            self.logger.info(f"After duplication removed, {label_counter}")
        self.logger.info(f"duplication removed counts are {sequence_count}")
        self.logger.info("_remove_duplication Done.")
        self.logger.info("----------------------------------------------/")

        return

    def _get_merged_dataframe(self) -> DataFrame:
        """
        Return the data after the process as dataframe.

        Note:
            make dataframe with self.variant_info_dict["done"]

        Returns:
            DataFrame: contain below columns.
            "labels","GeneName","refseqID","HGVSp","RefSeq","sequence"

        Examples:
            >>> self.variant_info_dict["done"]
            [("GOF", "geneA", "NM_AA", "WT", "AAA", "AAA"),
            ("LOF", "geneB", "NM_BB", "WT", "BBB", "BBB")]

            >>> self._get_merged_dataframe()
                labels  GeneName    refseqID    HGVSp   RefSeq  sequence
            0   GOF     geneA       NM_AA       WT      AAA     AAA
            1   LOF     geneB       NM_BB       WT      BBB     BBB
        """
        merged_df = pd.DataFrame(
            self.variant_info_dict["done"],
            columns=[
                "labels",
                "GeneName",
                "refseqID",
                "HGVSp",
                "RefSeq",
                "sequence",
                "tag",
            ],
        )

        self.logger.info(f"{merged_df.shape} shaped dataframe returned.")
        self.logger.info("_get_merged_dataframe Done.")
        self.logger.info("----------------------------------------------/")

        return merged_df
