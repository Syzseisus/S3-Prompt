import os
import argparse
import numpy as np
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn.functional as F
from rdkit.Chem import AllChem
from rdkit import Chem, DataStructs

from utils.constants import DATA_ROOT, POSSIBLE_DATASETS
from utils.helper import save_json, head_print, set_all_seeds
from utils.dataset_loader import load_dataset, get_molt5_model, mol_to_molt5_batch


def get_args():
    parser = argparse.ArgumentParser(description="Pre-calculate Exact Top-K shots (KNN)")
    parser.add_argument("--seed", type=int, default=1013)
    parser.add_argument("--dataset", type=str, required=True, choices=POSSIBLE_DATASETS)
    parser.add_argument("--embed_model", type=str, default="fp", choices=["fp", "small", "base", "large"])
    parser.add_argument("--max_k", type=int, default=100, help="Save top-k candidates to support various k later")
    parser.add_argument("--batch_size", type=int, default=32, help="Embedding batch size for MolT5")
    return parser.parse_args()


def get_fingerprints(smiles_list):
    fps = []
    valid_indices = []
    for idx, smi in enumerate(tqdm(smiles_list, desc="Generating Fingerprints")):
        mol = Chem.MolFromSmiles(smi)
        if mol:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
            fps.append(fp)
            valid_indices.append(idx)
    return fps, valid_indices


def get_molt5_embeddings_all(smiles_list, tokenizer, model, batch_size, device):
    all_embeds = []
    for i in tqdm(range(0, len(smiles_list), batch_size), desc="Generating MolT5 Embeddings"):
        batch_smiles = smiles_list[i : i + batch_size]
        # mol_to_molt5_batch returns [B, hidden_dim]
        embeds = mol_to_molt5_batch(batch_smiles, tokenizer, model, batch_size, device)
        all_embeds.append(embeds.cpu())
    return torch.cat(all_embeds, dim=0)


def compute_and_save_topk(query_df, train_df, train_feats, args, tokenizer, model, device, prefix, save_dir):
    """
    Compute similarity between Query and ALL Train data, pick Top-K, and save.
    """
    print(f"\n>>> Processing {prefix} set...")

    query_smiles = query_df["smiles"].tolist()
    train_indices = train_df["raw_ID"].tolist()  # Real ID mapping

    # 1. Query Features
    if args.embed_model == "fp":
        query_feats, query_valid_idx = get_fingerprints(query_smiles)
    else:
        query_feats = get_molt5_embeddings_all(query_smiles, tokenizer, model, args.batch_size, device)
        # Normalize for Cosine Similarity
        query_feats = F.normalize(query_feats, p=2, dim=1)

    result_dict = {}

    # 2. Compute Similarity & Sort
    # FP: Tanimoto (CPU loop)
    k_to_retrieve = args.max_k + 1 if prefix == "train" else args.max_k
    if args.embed_model == "fp":
        # train_feats is list of BitVects
        for i, q_fp in enumerate(tqdm(query_feats, desc="Calculating Similarity")):
            # BulkTanimoto is faster than loop
            sims = DataStructs.BulkTanimotoSimilarity(q_fp, train_feats)
            # Argsort descending
            top_k_idx = np.argsort(sims)[::-1][:k_to_retrieve]

            q_raw_id = str(query_df.iloc[query_valid_idx[i]]["raw_ID"])

            shot_list = []
            for idx in top_k_idx:
                cand_id = str(train_indices[idx])
                # Train 모드이고, 후보 ID가 쿼리 ID와 같다면(자기자신) 건너뜀
                if cand_id == q_raw_id:
                    continue
                shot_list.append(cand_id)
            shot_list = shot_list[: args.max_k]
            result_dict[q_raw_id] = {
                "shot_idx_list": shot_list,
                # "scores": [float(sims[idx]) for idx in top_k_idx] # Optional: save scores
            }

    # MolT5: Matrix Multiplication (GPU)
    else:
        # train_feats is already normalized tensor on CPU
        train_feats = train_feats.to(device)
        query_feats = query_feats.to(device)

        # Split query to avoid OOM if too large
        chunk_size = 100
        for i in tqdm(range(0, len(query_feats), chunk_size), desc="Calculating Similarity (GPU)"):
            q_chunk = query_feats[i : i + chunk_size]
            # [Chunk, Dim] @ [Dim, Train_Size] = [Chunk, Train_Size]
            sim_matrix = torch.mm(q_chunk, train_feats.t())

            # TopK per row
            topk_vals, topk_indices = torch.topk(sim_matrix, k=args.max_k, dim=1)

            topk_indices = topk_indices.cpu().numpy()

            for local_idx, global_train_idxs in enumerate(topk_indices):
                actual_query_idx = i + local_idx
                q_raw_id = str(query_df.iloc[actual_query_idx]["raw_ID"])

                shot_list = []
                for idx in global_train_idxs:
                    cand_id = str(train_indices[idx])

                    if prefix == "train" and cand_id == q_raw_id:
                        continue

                    shot_list.append(cand_id)

                # 정확히 K개만 남김
                shot_list = shot_list[: args.max_k]
                result_dict[q_raw_id] = {"shot_idx_list": shot_list}

        # Cleanup GPU
        train_feats = train_feats.cpu()
        torch.cuda.empty_cache()

    # 3. Save
    save_path = os.path.join(save_dir, f"query_to_shot_{prefix}_topk={args.max_k}.json")
    save_json(result_dict, save_path)
    print(f"Saved to {save_path}")


def main():
    args = get_args()
    set_all_seeds(args.seed)
    head_print(f"Pre-processing Exact Top-K (KNN) with args: {vars(args)}")

    # 1. Load Data
    train_df, valid_df, test_df = load_dataset(args.dataset, args.embed_model, verbose=True)

    # 2. Prepare Train Features (DB)
    print("\n>>> Encoding Training Data...")
    tokenizer, model = None, None
    device = "cpu"

    if args.embed_model == "fp":
        train_feats, _ = get_fingerprints(train_df["smiles"].tolist())
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer, model = get_molt5_model(args.embed_model, device)
        train_feats = get_molt5_embeddings_all(train_df["smiles"].tolist(), tokenizer, model, args.batch_size, device)
        # Normalize Train Features once
        train_feats = F.normalize(train_feats, p=2, dim=1)

    # 3. Setup Save Directory
    # Naming convention: {embed}_topk_all
    embed_tag = f"molt5-{args.embed_model}" if args.embed_model != "fp" else "fp"
    tag = f"topk_{embed_tag}"

    cache_dir = os.path.join(DATA_ROOT, args.dataset, tag)
    os.makedirs(cache_dir, exist_ok=True)
    print(f"[Info] Cache Directory: {cache_dir}")

    # 4. Run for Valid & Test
    compute_and_save_topk(valid_df, train_df, train_feats, args, tokenizer, model, device, "valid", cache_dir)
    compute_and_save_topk(test_df, train_df, train_feats, args, tokenizer, model, device, "test", cache_dir)

    print("\n[Done] All Exact Top-K processing finished.")


if __name__ == "__main__":
    main()
