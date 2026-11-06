import gc
import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from rdkit.Chem import AllChem, RWMol
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import train_test_split
from transformers import T5Tokenizer, T5ForConditionalGeneration

from utils.helper import log_df, head_print, verbose_print
from utils.constants import DATA_ROOT, MOLT5_DIM, REGRESSION, CLASSIFICATION

RDLogger.DisableLog("rdApp.*")


# same as in `utils/clustering.py`
def safe_MolFromSmiles(smiles, empty=True):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        if empty:
            mol = RWMol()
        else:
            mol = None
    return mol


def get_scaffold_smiles(smiles):
    mol = safe_MolFromSmiles(smiles, empty=False)
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    except Exception as e:
        print(f"smiles:{smiles if smiles else 'none'}")
        return None
    try:
        scaffold_smiles = Chem.MolToSmiles(scaffold, canonical=True)
    except Exception as e:
        print(f"smiles:{smiles}")
        print(f"scaffold:{scaffold if scaffold else 'none'}")
        return None
    return scaffold_smiles


# ===================================
# 1. Morgan Fingerprint Embedding
# ===================================
# Contains:
#   - mol_to_fp
#   - get_fp_embedding
# ===================================
# region: previous
# def mol_to_fp(smiles, idx=None, radius=2, n_bits=1024):
#     mol = Chem.MolFromSmiles(smiles)
#     if mol is None:
#         if idx is not None:
#             print(f"[WARN] Invalid SMILES (raw_ID={idx}): {smiles}")
#         else:
#             print(f"[WARN] Invalid SMILES: {smiles}")
#         return None
#     return AllChem.GetMorganFingerprintAsBitVect(mol, radius, n_bits)
# endregion


def mol_to_fp(smiles, idx=None, radius=2, n_bits=1024):
    """
    Note:
        1. `sanitize=True` (`MolFromSmiles` default) performs full structural checks
            1.  Adjust implicit hydrogens           (SANITIZE_ADJUSTHS)
            2.  Check radicals                      (SANITIZE_FINDRADICALS)
            3.  Initialize RingInfo and aromaticity (SANITIZE_SETAROMATICITY)
            4.  Cleanup general (valence, aromaticity, etc.) (SANITIZE_CLEANUP)
            4.  Cleanup chirality center            (SANITIZE_CLEANUPCHIRALITY)
            5.  Cleanup atropisomers (restricted rotation stereoisomers) (SANITIZE_CLEANUPATROPISOMERS)
            6.  Cleanup organometallics             (SANITIZE_CLEANUP_ORGANOMETALLICS)
            7.  Kekulize (assigns alternating single/double bonds in aromatic bonds)   (SANITIZE_KEKULIZE)
            8.  Set properties                      (SANITIZE_PROPERTIES)
            9.  Set conjugation                     (SANITIZE_SETCONJUGATION)
            10. Set hybridization                   (SANITIZE_SETHYBRIDIZATION)
            11. Symmetrize rings                    (SANITIZE_SYMMRINGS)
        2. Full sanitization can fail (fail to create mol),
            but Mol object may still be usable for fingerprint
        3. Minimal checks needed for safe fingerprint calculation: 1-4

    Procedure:
        1. Create Mol with sanitize=False to avoid any unnecessary sanitization errors
        2. Apply only the four minimal checks before generating the fingerprint
        3. Generate fingerprint
    """

    def msg_index(msg):
        return f"{msg} (raw_ID={idx}): {smiles}" if idx is not None else f"{msg}: {smiles}"

    # 1) Create Mol without full sanitization
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        raise ValueError(msg_index("Mol creation failed"))

    try:
        # 2) Apply minimal sanitization required for fingerprint calculation
        Chem.SanitizeMol(
            mol,
            sanitizeOps=(
                Chem.SanitizeFlags.SANITIZE_ADJUSTHS
                | Chem.SanitizeFlags.SANITIZE_FINDRADICALS
                | Chem.SanitizeFlags.SANITIZE_SETAROMATICITY
                | Chem.SanitizeFlags.SANITIZE_CLEANUPCHIRALITY
            ),
        )
    except Exception as e:
        raise RuntimeError(msg_index("Manual sanitization failed")) from e
    else:
        # 3) Generate fingerprint
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, n_bits)
        return fp


def get_fp_embedding(df: pd.DataFrame, radius: int, n_bits: int):
    df["embed"] = df.apply(lambda row: mol_to_fp(row["smiles"], row["raw_ID"], radius=radius, n_bits=n_bits), axis=1)
    return df


# ===================================
# 2. MolT5 Embedding
# ===================================
# Contains:
#   - get_molt5_model: load MolT5 of given size
#   - mol_to_molt5_batch
#   - get_molt5_embedding
# ===================================
def get_molt5_model(model_size="base", device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = f"laituan245/molt5-{model_size}-smiles2caption"
    tokenizer = T5Tokenizer.from_pretrained(model_name, model_max_length=512, legacy=True)
    model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
    model.eval()

    return tokenizer, model


def mol_to_molt5_batch(smiles_list, tokenizer, model, batch_size=32, device="cuda"):
    embeddings = []
    for i in tqdm(range(0, len(smiles_list), batch_size), leave=False):
        batch_smiles = smiles_list[i : i + batch_size]

        inputs = tokenizer(batch_smiles, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            encoder_outputs = model.encoder(**inputs)
            last_hidden_state = encoder_outputs.last_hidden_state

            # Use embedding as mean pooling of `last_hidden_state`
            input_mask_expanded = inputs["attention_mask"].unsqueeze(-1).expand(last_hidden_state.size()).float()
            sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            batch_embeddings = sum_embeddings / sum_mask  # [batch, hidden_dim]
            embeddings.append(batch_embeddings.cpu())

    embeddings = torch.cat(embeddings, dim=0)  # [num_samples, hidden_dim]

    return embeddings


def get_molt5_embedding(df, model_size="base", batch_size=32, device=None, verbose=True):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    verbose_print("Use", device, verbose=verbose)

    tokenizer, model = get_molt5_model(model_size, device)
    smiles_list = df["smiles"].tolist()
    embeds_tensor = mol_to_molt5_batch(smiles_list, tokenizer, model, batch_size=batch_size, device=device)

    df["embed"] = list(embeds_tensor)

    return df


# ===================================
# 3. Dataset Download & Trimming
# ===================================
# Download datasets, map columns, remove NaNs, and cache
# Contains:
#   - get_url: dataset URL
#   - get_meta: column metadata for the dataset
#   - trim_dataset: load & trim raw dataset
#   - load_or_make_raw_trimmed: load cached or create trimmed dataset
# ===================================
def get_url(dataset_name):
    urls = {
        "BBBP": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv",
        "BACE": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/bace.csv",
        "HIV": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/HIV.csv",
        "ESOL": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv",
        "FREESOLV": "https://deepchemdata.s3.us-west-1.amazonaws.com/datasets/freesolv.csv.gz",
        "LIPOPHILICITY": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/Lipophilicity.csv",
    }
    if dataset_name not in urls:
        raise ValueError(f"Invalid dataset name: {dataset_name}")
    return urls[dataset_name]


def get_meta(dataset_name):
    """
    Return the metadata columns (idx_col, label_col, smiles_col) as a tuple for a dataset.

    Returns (tuple):
        - idx_col (str or None): name of the index column in the raw dataset
            if None, a new index will be added explicitly
        - label_col (str): name of the label column
        - smiles_col (str): name of the column containing molecule SMILES strings
    """

    meta = {
        "BBBP": ("num", "p_np", "smiles"),
        "BACE": ("CID", "Class", "mol"),
        "HIV": (None, "HIV_active", "smiles"),
        "ESOL": ("Compound ID", "measured log solubility in mols per litre", "smiles"),
        "FREESOLV": (None, "y", "smiles"),
        "LIPOPHILICITY": ("CMPD_CHEMBLID", "exp", "smiles"),
    }
    if dataset_name not in meta:
        raise ValueError(f"No metadata for dataset: {dataset_name}")
    return meta[dataset_name]


def trim_dataset(dataset_name, verbose=True):
    """
    ref: https://github.com/deepchem/deepchem/tree/39f29c2741b931f8dfb9cc0259e0a8c5b10ba799/deepchem/molnet/load_function

    Returns:
        - df_raw (pd.DataFrame): original dataset
        - df_trimmed (pd.DataFrame): only ['raw_ID', 'smiles', 'label'] and remove NaNs
    """
    url = get_url(dataset_name)
    idx_col, label_col, smiles_col = get_meta(dataset_name)

    # 1) Read original CSV
    df_raw = pd.read_csv(url)
    log_df(f"Load `df_raw` from url", df=df_raw, verbose=verbose)

    # 2) Add index column if there isn't in `raw.csv`
    if idx_col is None:
        df_raw = df_raw.reset_index()
        idx_col = "index"

    # 3) Keep only required columns
    df = df_raw.rename(columns={idx_col: "raw_ID", label_col: "label", smiles_col: "smiles"})
    df = df[["raw_ID", "smiles", "label"]].copy()
    log_df(f"Keep only `smiles`, `label`, and `raw_ID` columns", df=df, verbose=verbose)

    # SPECIAL CASE
    if dataset_name.upper() == "BBBP":
        # ref: https://link.springer.com/article/10.1023/A:1020792909928, Table 1, molecule #2
        df.loc[df["raw_ID"] == 651, "smiles"] = "s1cc(nc1N=C(N)N)C"
        print("[INFO] Corrected invalid SMILES for BBBP raw_ID=651")

    if dataset_name.upper() == "HIV":
        # drop 2 rows with invalid SMILES - cannot find reference
        invalid_smiles_ids = [30784, 18293]
        df = df[~df["raw_ID"].isin(invalid_smiles_ids)].copy()
        print(f"[INFO] Dropped invalid SMILES for HIV raw_IDs={invalid_smiles_ids}")

    # 4) Drop rows with missing SMILES or label
    df_trimmed = df[df["smiles"].astype(bool)].copy()
    df_trimmed = df_trimmed.dropna(subset=["label"])
    log_df(f"Remove NaNs", df=df_trimmed, verbose=verbose)

    # 5) Drop rows where the scaffold cannot be extracted
    df_trimmed["scaffold_smiles"] = df_trimmed["smiles"].apply(get_scaffold_smiles)
    df_trimmed = df_trimmed.dropna(subset=["scaffold_smiles"])
    log_df(f"Remove NaNs", df=df_trimmed, verbose=verbose)

    # 6) Cast label type
    if dataset_name.upper() in CLASSIFICATION:
        df_trimmed["label"] = df_trimmed["label"].astype(int)
    elif dataset_name.upper() in REGRESSION:
        df_trimmed["label"] = df_trimmed["label"].astype(float)
    log_df(f"Re-type `label` columns", df=df_trimmed, verbose=verbose)

    return df_raw, df_trimmed


def load_or_make_raw_trimmed(dataset_name, save_dir, verbose=False):
    """Load cached dataset if exists; otherwise, download and trim raw dataset."""

    raw_path = os.path.join(save_dir, "raw.csv")
    trimmed_path = os.path.join(save_dir, "trimmed.csv")

    # Load cached preprocessed dataset usage if exists
    if os.path.exists(raw_path) and os.path.exists(trimmed_path):
        df_raw = pd.read_csv(raw_path)
        df = pd.read_csv(trimmed_path)
        log_df(f"Load `df_raw` from `raw_path`: {raw_path}", df=df_raw, verbose=verbose)
        log_df(f"Load `df` from `trimmed_path`: {trimmed_path}", df=df, verbose=verbose)

    # Preprocess raw dataset if not cached
    else:
        df_raw, df = trim_dataset(dataset_name, verbose=verbose)
        df_raw.to_csv(raw_path, index=False)
        df.to_csv(trimmed_path, index=False)
        log_df(f"Save `df_raw` as `raw_path`: {raw_path}", df=df_raw, verbose=verbose)
        log_df(f"Save `df` as `trimmed_path`: {trimmed_path}", df=df, verbose=verbose)

    raw_count = len(df_raw)
    trimmed_count = len(df)
    removed = raw_count - trimmed_count
    log_df(f"Remove {removed} rows due to empty SMILES, label or Scaffold SMILES", verbose=verbose, sep_line=False)
    log_df(f"  >> {trimmed_count}/{raw_count} rows remaining after trimming", verbose=verbose)

    return df_raw, df


# ===================================
# 4. Embedding & Dataset Split
# ===================================
# Contains:
#   - get_embedding
#   - split_dataset: stratified split
# ===================================
def get_embedding(
    df,
    dataset_name,
    embed_model,
    save_dir,
    radius=2,
    n_bits=1024,
    batch_size=32,
    device=None,
    fp_nan_mask=None,
    verbose=False,
):
    model_size = embed_model.lower()

    # Morgan fingerprint
    if model_size == "fp":
        embed_path = os.path.join(save_dir, f"embed_fp_r={radius}_b={n_bits}.pk")
        if os.path.exists(embed_path):
            embed_col = pd.read_pickle(embed_path)
            assert len(embed_col) == len(df), "Length mismatch between df and cached embed column"
            embed_col.index = df["raw_ID"]
            log_df(f"Load `embed` column ({len(embed_col)}) from `embed_path`: {embed_path}", verbose=verbose)
        else:
            log_df("Init Morgan Fingerprint Embedding ...", verbose=verbose, sep_line=False)
            df = get_fp_embedding(df, radius, n_bits)
            log_df(f"Get embedding with FP", df=df, verbose=verbose)
            df["embed"].to_pickle(embed_path)
            embed_col = df["embed"]
            embed_col.index = df["raw_ID"]
            log_df(f"Save `embed_col` as `embed_path`: {embed_path}", df=df, verbose=verbose)
        fp_nan_mask = embed_col.notna()

    # MolT5
    elif model_size in {"small", "base", "large"}:
        assert fp_nan_mask is not None, "`fp_nan_mask` must be provided when getting MolT5 embedding"

        dim = MOLT5_DIM[model_size]
        embed_path = os.path.join(save_dir, f"embed_molt5_{model_size}_d{dim}.pt")
        if os.path.exists(embed_path):
            embeds = torch.load(embed_path)
            embed_col = pd.Series(list(embeds.cpu()), index=df["raw_ID"])  # Keep Tensors; .tolist() would break it
            log_df(
                f"Load `embed` column ({len(embed_col)}) from `embed_path`: {embed_path}",
                verbose=verbose,
                sep_line=False,
            )
        else:
            log_df(f"Init MolT5-{model_size} Embedding ...", verbose=verbose, sep_line=False)
            df = get_molt5_embedding(df, model_size=model_size, batch_size=batch_size, device=device, verbose=verbose)
            log_df(f"Get embedding with MolT5", df=df, verbose=verbose)
            embeds = torch.stack(df["embed"].tolist())
            torch.save(embeds, embed_path)
            embed_col = df["embed"]
            embed_col.index = df["raw_ID"]
            log_df(f"Save `embed_col` as `embed_path`: {embed_path}", df=df, verbose=verbose)

    # Remove NaNs w.r.t FP embedding
    try:
        trimmed_count = len(df)
        if model_size == "fp":
            # [Case A] FP 모델: 여기서 마스크를 "생성"합니다.
            # embed_col.index가 raw_ID이므로, fp_nan_mask도 raw_ID를 인덱스로 가짐
            fp_nan_mask = embed_col.notna()

            # 유효한 raw_ID 목록 추출
            valid_raw_ids = fp_nan_mask[fp_nan_mask].index

        else:
            # [Case B] MolT5 모델: FP에서 만든 마스크를 "사용"만 합니다.
            # fp_nan_mask는 이미 raw_ID를 인덱스로 가지고 있음 (FP 단계에서 넘겨받음)
            valid_raw_ids = fp_nan_mask[fp_nan_mask].index

        # 3. 실제 필터링 적용 (공통)
        # raw_ID를 기준으로 교집합만 남김 (순서가 섞여도 안전함)
        df = df[df["raw_ID"].isin(valid_raw_ids)].copy()
        embed_col = embed_col.loc[valid_raw_ids]

        embed_count = len(df)
    except Exception as e:
        print(model_size)
        raise e

    removed = trimmed_count - embed_count
    log_df(f"Remove {removed} rows due to empty embedding", verbose=verbose, sep_line=False)
    log_df(f"{embed_count}/{trimmed_count} rows remaining after trimming", verbose=verbose)

    return df, embed_col, fp_nan_mask


def split_dataset(df, embed_col, save_dir, dataset_name, verbose=False):
    """
    Updated Split Logic: Train : Valid : Test = 8 : 1 : 1
    """
    train_path = os.path.join(save_dir, "train.csv")
    valid_path = os.path.join(save_dir, "valid.csv")
    test_path = os.path.join(save_dir, "test.csv")

    if os.path.exists(train_path) and os.path.exists(valid_path) and os.path.exists(test_path):
        train_df = pd.read_csv(train_path)
        valid_df = pd.read_csv(valid_path)
        test_df = pd.read_csv(test_path)
        log_df(f"Load split dfs from {save_dir}", df=train_df, verbose=verbose)
    else:
        df = df[["raw_ID", "smiles", "label"]].copy()

        # 1. First Split: Train(80%) vs Temp(20%)
        if dataset_name.upper() in CLASSIFICATION:
            stratify = df["label"]
        else:
            stratify = None

        train_df, temp_df = train_test_split(df, test_size=0.2, stratify=stratify, random_state=42)

        # 2. Second Split: Temp(20%) -> Valid(10%) + Test(10%)
        # Splitting Temp in half results in 10% of total each
        if dataset_name.upper() in CLASSIFICATION:
            stratify_temp = temp_df["label"]
        else:
            stratify_temp = None

        valid_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=stratify_temp, random_state=42)

        train_df.to_csv(train_path, index=False)
        valid_df.to_csv(valid_path, index=False)
        test_df.to_csv(test_path, index=False)

        log_df("Saved Train/Valid/Test (8:1:1) splits", verbose=verbose)

    # Add embed column
    train_df = train_df.join(embed_col.rename("embed"), on="raw_ID")
    valid_df = valid_df.join(embed_col.rename("embed"), on="raw_ID")
    test_df = test_df.join(embed_col.rename("embed"), on="raw_ID")

    # Reset index
    train_df = train_df.reset_index(drop=True)
    valid_df = valid_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    return train_df, valid_df, test_df


# ===================================
# 5. Main Dataset Loading
# ===================================
# Load datasets, preprocess then cache
# Contains:
#   - load_dataset: main loader function
#   - __main__: preprocessing cache workflow
# ===================================
def load_dataset(dataset_name, embed_model, radius=2, n_bits=1024, batch_size=32, device=None, verbose=True):
    save_dir = os.path.join(DATA_ROOT, dataset_name)
    os.makedirs(save_dir, exist_ok=True)

    # 1) Get trimmed dataset
    df_raw, df = load_or_make_raw_trimmed(dataset_name, save_dir, verbose=verbose)

    # 2) FP embeddings (Always needed for clustering/similarity)
    df_fp, embed_col, fp_nan_mask = get_embedding(
        df,
        dataset_name,
        "fp",
        save_dir,
        radius=radius,
        n_bits=n_bits,
        verbose=(embed_model.lower() == "fp" and verbose),
    )

    if embed_model.lower() in {"small", "base", "large"}:
        df, embed_col, _ = get_embedding(
            df,
            dataset_name,
            embed_model,
            save_dir,
            batch_size=batch_size,
            device=device,
            fp_nan_mask=fp_nan_mask,
            verbose=verbose,
        )

    # 4) Split dataset (Returns 3 DataFrames now)
    return split_dataset(df, embed_col, save_dir, dataset_name, verbose=verbose)


if __name__ == "__main__":
    dataset_name = sys.argv[1]
    batch_size = 256 if dataset_name.upper() == "HIV" else 1024

    head_print(f"Preprocessing {dataset_name}")
    print("  1. Download dataset from DeepChem")
    print("  2. Trim columns - keep only SMILES, label, and raw_ID")
    print("  3. Add embedding - ECFP or MolT5-{small, base, large}")
    print("  4. Split dataset - stratified for classification, random for regression")
    head_print("")
    print()

    # for embed_model in ["fp", "small", "base", "large"]:
    for embed_model in ["fp", "small"]:
        if embed_model == "fp":
            head_print("Morgan Fingerprint Embedding")
            kwargs = {"radius": 2, "n_bits": 1024}
        elif embed_model in {"small", "base", "large"}:
            head_print(f"MolT5-{embed_model} Embedding")
            kwargs = {"batch_size": batch_size}

        train_df, valid_df, test_df = load_dataset(dataset_name, embed_model, verbose=True, **kwargs)
        print(f"# train = {len(train_df)}")
        sample = train_df.iloc[0].to_dict()
        print("sample: {")
        for k, v in sample.items():
            if k == "embed":
                v = np.array(v) if embed_model == "fp" else v
                print(embed_model, type(v))
                print(f"  {k}: {v.shape} ")
            else:
                print(f"  {k}: {v}")
        print("}\n")
        print(f"# valid = {len(valid_df)}")
        sample = valid_df.iloc[0].to_dict()
        print("sample: {")
        for k, v in sample.items():
            if k == "embed":
                v = np.array(v) if embed_model == "fp" else v
                print(f"  {k}: {v.shape} ")
            else:
                print(f"  {k}: {v}")
        print("}\n")
        print(f"# test  = {len(test_df)}")
        sample = test_df.iloc[0].to_dict()
        print("sample: {")
        for k, v in sample.items():
            if k == "embed":
                v = np.array(v) if embed_model == "fp" else v
                print(f"  {k}: {v.shape} ")
            else:
                print(f"  {k}: {v}")
        print("}\n")
        head_print("")
        print()

        del train_df, valid_df, test_df
        gc.collect()
        torch.cuda.empty_cache()
