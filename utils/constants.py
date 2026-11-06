DATA_ROOT = "../MoleculeNetICL_S3-Prompt"
SAVE_ROOT = "./results"
CKPT_ROOT = "./checkpoints"
ANALYSIS_ROOT = "./analysis_results"

CLASSIFICATION = {"BBBP", "BACE", "HIV", "MUV", "SIDER", "CLINTOX", "TOX21", "TOXCAST"}
REGRESSION = {"ESOL", "FREESOLV", "LIPOPHILICITY"}
POSSIBLE_DATASETS = CLASSIFICATION.union(REGRESSION)
MOLT5_DIM = {"small": 512, "base": 768, "large": 1024}
