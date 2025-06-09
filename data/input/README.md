# Load Data for ClearVariant
## Repository Structure
After completing all the steps below, the structure of the `data/input/` directory should look like the following.
```
data/input/
├── protein_ref_seq/
│   └── uniprotkb.tsv
├── goflof/
│   ├── goflof_ClinVar_v062021.csv
│   └── goflof_HGMD2019_v032021_allfeat.csv
├── proteingym/
│   ├── DMS_ProteinGym_substitutions.zip
│   └── DMS_ProteinGym_substitutions/
└── main.py
```

## Reference sequence
### Make directory
```bash
mkdir protein_ref_seq
```

### Download reference seq from UniProt
[UniProt link](https://www.uniprot.org/uniprotkb?query=*)

You should download tsv file with below option without compression.
1. Reviewed
2. Human
3. Include Full XRef IDs

Rename tsv file as `uniprotkb.tsv`.


## Gain of Function/Loss of Function
### Make directory
```bash
mkdir goflof
```
### Download ClinVar and HGMD GoF/LoF
[itanlab link](https://itanlab.shinyapps.io/goflof/)
From the `Download` tab at the top right of the website, ClinVar data can be obtained from `Download all ClinVar-based GOF/LOF variants`, and HGMD data can be obtained from `Download all annotated features`.


## ProteinGym
### Make directory
```bash
mkdir proteingym
```

### Download ProteinGym substitution datasets.
```bash
wget https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/DMS_ProteinGym_substitutions.zip
```
### Unzip ProteinGym file.
```bash
unzip DMS_ProteinGym_substitutions.zip
```