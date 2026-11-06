import os
import math
import random
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from pprint import pprint
from datetime import datetime
import matplotlib.pyplot as plt
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import Draw, RWMol, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold
from torch.nn.functional import cosine_similarity

from utils.constants import DATA_ROOT, POSSIBLE_DATASETS
from utils.helper import log_df, load_json, save_json, head_print, load_pickle, save_pickle, set_all_seeds
from utils.dataset_loader import (
    mol_to_fp,
    load_dataset,
    get_molt5_model,
    mol_to_molt5_batch,
)

RDLogger.DisableLog("rdApp.*")


# ===================================
# 0. Helper Functions
# ===================================
# Contains:
#   - Molecule handling: safe_MolFromSmiles, get_scaffold_smiles
#   - Embedding: get_molt5_embedding, compute_similarity
#   - Plotting helper: safe_MolToMPL, save_show_close
#   - Bunita merging: UnionFind
# ===================================
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


def get_molt5_embedding(smiles_list, tokenizer, model, batch_size=32):
    embeds_tensor = mol_to_molt5_batch(smiles_list, tokenizer, model, batch_size=batch_size, device=model.device)

    return embeds_tensor


def compute_similarity(query_emb, emb_list, embed_type, aggregate="none", topk=0, batch_size=None):
    """
    Compute similarity between a query embedding and a list of embeddings.

    Modes (`aggregate`):
        1. "none": return all similarities as np.array
        2. "mean": return the mean of similarities as float
        3. "topk": return `(indices, sims)` for top-k similarities
            - `topk` must be specified

    Note:
        When using MolT5 (`embed_type` in {"small", "base", "large"}),
        the `batch_size` argument can be used to reduce memory usage.
    """
    assert aggregate in {"none", "mean", "topk"}, f"Invalid aggregate mode: {aggregate}"

    if embed_type.lower() == "fp":
        sims = np.array(DataStructs.BulkTanimotoSimilarity(query_emb, emb_list))
    elif embed_type.lower() in {"small", "base", "large"}:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        with torch.no_grad():
            query_emb = query_emb.to(torch.float32).squeeze()
            if batch_size:
                sims_list = []
                for i in range(0, len(emb_list), batch_size):
                    batch_emb = torch.stack(emb_list[i : i + batch_size]).to(device)
                    query_emb_expand = query_emb.unsqueeze(0).expand(batch_emb.size(0), -1).to(device)
                    sims_batch = cosine_similarity(query_emb_expand, batch_emb, dim=1)
                    sims_list.append(sims_batch.cpu())
                sims = torch.cat(sims_list, dim=0).numpy()
            else:
                query_emb_expand = query_emb.unsqueeze(0).expand(len(emb_list), -1).to(device)
                emb_stack = torch.stack(emb_list).to(device)
                sims = cosine_similarity(query_emb_expand, emb_stack, dim=1).cpu().numpy()

    if aggregate == "none":
        return sims
    elif aggregate == "mean":
        return np.mean(sims)
    elif aggregate == "topk":
        assert topk > 0, f"`topk` must be specified, but got {topk}"
        sorted_idx = np.argsort(sims)[::-1][:topk]
        return sorted_idx, sims[sorted_idx]


def safe_MolToMPL(mol, ax=None):
    """
    Render and display an RDKit molecule on a Matplotlib axis.
    If rendering fails or the molecule is invalid, display a fallback "Render Failed" message.
    """

    def _ax_text():
        ax.text(0.5, 0.5, "Render\nFailed", ha="center", va="center", fontsize=10, transform=ax.transAxes)

    if mol is None or mol.GetNumAtoms() == 0:
        _ax_text()
        return
    try:
        Draw.MolToMPL(mol, ax=ax)
    except Exception:
        _ax_text()


def save_show_close(fig, save_path=None, save_msg="", show=True, tight=True):
    if tight:
        plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"Saved {save_msg} to {save_path}")
    if show:
        plt.show()
    plt.close(fig)


class UnionFind:
    """
    Disjoint Set (Union-Find) data structure for Bunita merging.
    Each element is a scaffold SMILES.
    UnionFind tracks which scaffolds should be merged into the same cluster.

    Methods:
        find(x): Return the representative of scaffold x's cluster (merge_n)
        union(x, y): Merge the clusters containing scaffolds x and y
        groups(): Return a dictionary `{merge_n: [list of scaffolds in that cluster]}`
    """

    def __init__(self, elements):
        self.parent = {e: e for e in elements}

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        px = self.find(x)
        py = self.find(y)
        if px != py:
            self.parent[py] = px

    def groups(self):
        clusters = {}
        for x in self.parent:
            px = self.find(x)
            if px not in clusters:
                clusters[px] = []
            clusters[px].append(x)
        return clusters


# ===================================
# 1. Scaffold Clustering and Bunita Merging
# ===================================
# Contains: scaffold_bunita_clustering only
# ===================================
def scaffold_bunita_clustering(
    df, embed_model="fp", embed_batch_size=32, eps=0.6, tokenizer=None, model=None, debug=False
):
    """
    Perform scaffold-based clustering with optional Bunita merge of *unique_clusters*

    Procedure:
        1. Extract scaffold SMILES for each molecule
        2. Initialize clusters based on unique scaffolds
        3. Compute scaffold embeddings (fingerprint or MolT5 embedding, depending on `embed_model`)
        4. If `eps > 0`, merge similar clusters using a Union-Find structure among *unique_clusters*
            - "similar" means `similarity >= eps` (computed on scaffold embeddings)
            - *unique_clusters* are clusters containing only a single molecule
            - Non-*unique_clusters* are always preserved as-is (not merged)

    Returns:
        clusters_idx (dict): {scaffold: [list of molecule indices having this scaffold]}
        clusters_scaf (dict): {rep_scaffold: [list of scaffold SMILES in this cluster]}
            - If merged, rep_scaffold is named as `"merge_{id}"`
            - When unique_clusters are merged,
                - their original scaffold keys remain in `clusters_idx`
                - but disappear from `clusters_scaf`
            - Therefore, the number of resulting clusters equals `len(clusters_scaf)` (≤ `len(clusters_idx)`)
        scaf_emb_dict (dict): {scaffold_smiles: embedding vector}
    """

    # 1) Extract scaffolds
    df["scaffold_smiles"] = df["smiles"].apply(lambda x: get_scaffold_smiles(x))
    scaffolds = [s for s in df["scaffold_smiles"].unique() if s is not None]

    # 2) Init clusters
    clusters_idx = {scaf: df.loc[df["scaffold_smiles"] == scaf, "raw_ID"].astype(str).tolist() for scaf in scaffolds}
    clusters_scaf = {scaf: [scaf] for scaf in scaffolds}

    # 3) Compute scaffold embeddings
    if embed_model.lower() == "fp":
        scaf_emb_dict = {scaf: mol_to_fp(scaf, radius=2, n_bits=1024) for scaf in scaffolds}
    elif embed_model.lower() in {"small", "base", "large"}:
        assert tokenizer is not None and model is not None, "tokenizer and model must be provided when using MolT5"
        embeds = get_molt5_embedding(scaffolds, tokenizer, model, embed_batch_size)
        scaf_emb_dict = {scaf: emb.clone() for scaf, emb in zip(scaffolds, embeds)}

    # 4) Bunita merging using Union-Find among unique_clusters
    if eps > 0:
        # 4-1) Identify unique_clusters and init Union-Find
        unique_scafs = [scaf for scaf, idxs in clusters_idx.items() if len(idxs) == 1]
        uf = UnionFind(unique_scafs)
        print(f"Before merge (Unique / Total): {len(unique_scafs)}/{len(clusters_scaf)}")

        # 4-2) Perform Bunita merging
        if debug:
            debug_merges = []

        scaf_list = unique_scafs.copy()
        while scaf_list:
            scaf_i = scaf_list.pop(0)
            embed_i = scaf_emb_dict[scaf_i]
            other_scafs = scaf_list
            if not other_scafs:
                continue

            other_embs = [scaf_emb_dict[s] for s in other_scafs]
            sims = compute_similarity(embed_i, other_embs, embed_model, aggregate="none")
            for s, sim in zip(other_scafs, sims):
                if sim >= eps:
                    uf.union(scaf_i, s)

                    if debug:
                        # 의도: 진짜 Singleton끼리 합쳐지는지, 유사도는 몇인지 확인
                        size_i = len(clusters_idx[scaf_i])
                        size_target = len(clusters_idx[s])

                        # 검증 로직: 둘 중 하나라도 사이즈가 1이 아니면 경고 출력
                        if size_i != 1 or size_target != 1:
                            print(f"[CRITICAL WARNING] Non-singleton merge detected! Sizes: {size_i} <-> {size_target}")

                        # 샘플링하여 일부만 출력 (너무 많을 수 있으므로)
                        if len(debug_merges) < 10:  # 처음 10개 병합 사례만 출력
                            debug_merges.append(
                                f"[MERGE] Scaf1(len={size_i}): {scaf_i} <-> Scaf2(len={size_target}): {s} | Sim: {sim:.4f}"
                            )
        if debug:
            # 병합 로그 출력
            if eps > 0:
                print("\n" + "=" * 40)
                print(f" [DEBUG] Singleton Merge Verification (Top 10 examples)")
                for log in debug_merges:
                    print(log)
                print("=" * 40 + "\n")

        # 4-3) Construct merged clusters from Union-Find groups
        merged_idx = {}
        merged_mols = {}
        merge_counter = 0

        if debug:
            debug_merged_sizes = []  # 병합된 클러스터들의 크기(스캐폴드 개수) 리스트
            debug_total_scafs_accounted = 0  # 처리된 Singleton 스캐폴드 총 개수
        for scaf_list in uf.groups().values():
            if len(scaf_list) > 1:
                merge_name = f"merge_{merge_counter}"
                merge_counter += 1
                merged_mols[merge_name] = scaf_list.copy()
                merged_idx[merge_name] = []
                for s in scaf_list:
                    merged_idx[s] = clusters_idx[s]
                    merged_idx[merge_name] += clusters_idx[s]

                if debug:
                    debug_merged_sizes.append(len(scaf_list))
                    debug_total_scafs_accounted += len(scaf_list)

            else:
                # Keep non-merged clusters
                s = scaf_list[0]
                merged_idx[s] = clusters_idx[s]
                merged_mols[s] = [s]

                if debug:
                    debug_total_scafs_accounted += 1
        if debug:
            # [DEBUG PRINT] 실제 병합 현황판
            print("\n" + "=" * 50)
            print(" [DEBUG] Singleton Merge Statistics Analysis")
            print(f" 1. Total Singletons Input: {len(unique_scafs)} (Should be 484)")
            print(f" 2. Total Scaffolds Output: {debug_total_scafs_accounted} (Must match Input)")
            print(f" 3. Merged Groups Count   : {len(debug_merged_sizes)}")
            print(f" 4. Merged Group Sizes    : {debug_merged_sizes}")

            if len(unique_scafs) != debug_total_scafs_accounted:
                diff = len(unique_scafs) - debug_total_scafs_accounted
                print(f" [CRITICAL ERROR] {diff} singletons have DISAPPEARED! Check logic.")
            else:
                print(" [SUCCESS] No data loss. All singletons accounted for.")
            print("=" * 50 + "\n")

        # 4-4) Add non-unique clusters as-is
        for scaf, idxs in clusters_idx.items():
            if len(idxs) > 1:
                merged_idx[scaf] = idxs
                merged_mols[scaf] = [scaf]

        clusters_idx = merged_idx
        clusters_scaf = merged_mols

        num_unique_after_merge = sum(
            1
            for scaf_list in clusters_scaf.values()
            if len(scaf_list) == 1
            for scaf in scaf_list
            if len(clusters_idx[scaf]) == 1
        )
        print(f"After merge (Unique / Total): {num_unique_after_merge}/{len(clusters_scaf)}")

    return clusters_idx, clusters_scaf, scaf_emb_dict


# ===================================
# 2. Pool Clustering and Its Caching
# ===================================
# Implements the proposed method, **Pool Clustering**.
# Contains:
#   - pool_clustering: selects shots from a single query
#   - retrieve_whole_query: applies pool_clustering to all queries
#   - prepare_query_cluster_data: loads from or saves to cache for retrieve_whole_query
# ===================================
def pool_clustering(
    query_smiles,
    query_scaf,
    clusters_idx,
    clusters_scaf,
    scaf_emb_dict,
    emb_dict,
    k=8,
    sample_strategy="random",
    embed_type="fp",
    tokenizer=None,
    model=None,
    batch_size=32,
    debug=False,
):
    """
    Select k shots based on query molecule .

    Procedure:
        1. Select molecules with the same scaffold as the query scaffold (`query_scaf`)
        2. If insufficient expand to clusters with similar scaffolds
            - For merged clusters, use the average value
        3. As a final fallback, randomly sample molecules from the entire pool
            - Note: ideally, this step is never executed

    Sampling strategies within a cluster (`sample_strategy`):
        - "random": randomly select within the cluster
        - "similarity": select top-k molecules within the cluster based on similarity to the query molecule
        Note:
            - `usage_type` is logged into `step_info` to indicate how a cluster was used:
                - "all": The entire cluster was selected (i.e. regardless of `sample_strategy`)
                - "sampling": Only a subset of molecules was selected
            - This is important because when "all" is frequent, `sample_strategy` doesn't really affect molecule selection

    Returns:
        chosen_idxs (List[int]): indices of selected molecules from the pool
        chosen_clusters (List[str]): cluster representative (`rep_scaf`) each molecule belongs to
            - For merged clusters, this is the 'merge_n' string
        step_info (dict): detailed information per selection step
            - "same_cluster": info for query scaffold cluster
                - rep_scaf (str): representative scaffold
                - cluster_size (int): number of molecules in cluster
                - n_used (int): number of molecules selected from this cluster
                - usage_type (str): "all" if entire cluster used, "sampling" else
            - "similar_cluster": info for similar clusters
                - n_clusters_used (int)
                - clusters_detail (list of dict): for each cluster (same as "same_cluster")
                    - scaf_list (List[str], optional): only present for merged clusters
            - "fallback": info for fallback sampling, `n_used` only
                - n_used (int)
    """

    # Pre-compute query embedding
    if embed_type == "fp":
        query_emb = mol_to_fp(query_smiles, radius=2, n_bits=1024)
    elif embed_type in {"small", "base", "large"}:
        query_emb = get_molt5_embedding([query_smiles], tokenizer, model, 1)  # [1, emb_dim]

    def sample_cluster(idxs, remain):
        # A) If cluster is larger than needed, Sample up to `remain` molecules
        if len(idxs) >= remain:
            if sample_strategy == "random":
                chosen = random.sample(idxs, remain)
            elif sample_strategy == "similarity":
                emb_list = [emb_dict[str(i)] for i in idxs]
                sorted_idx, sims = compute_similarity(
                    query_emb, emb_list, embed_type, aggregate="topk", topk=remain, batch_size=batch_size
                )
                chosen = [idxs[i] for i in sorted_idx]
            else:
                raise NotImplementedError(f"sample_strategy {sample_strategy} not implemented")

            n_used = remain
            usage_type = "sampling"

        # B) If cluster is smaller than needed, Take all molecules in cluster
        else:
            chosen = idxs
            n_used = len(idxs)
            usage_type = "all"

        return chosen, n_used, usage_type

    chosen_idxs = []
    chosen_clusters = []
    step_info = {
        "same_cluster": {"rep_scaf": None, "cluster_size": 0, "n_used": 0, "usage_type": "none"},
        "similar_cluster": {"n_clusters_used": 0, "clusters_detail": []},
        "fallback": {"n_used": 0},
    }

    # 1) Select from the same scaffold molecules
    if query_scaf in clusters_scaf:
        idxs = clusters_idx[query_scaf]
        step_info["same_cluster"]["rep_scaf"] = query_scaf
        step_info["same_cluster"]["cluster_size"] = len(idxs)

        chosen, n_used, usage_type = sample_cluster(idxs, k)
        chosen_idxs += chosen
        chosen_clusters += [query_scaf] * n_used
        step_info["same_cluster"]["n_used"] = n_used
        step_info["same_cluster"]["usage_type"] = usage_type

    # 2) If not enough, expand to clusters with similar scaffolds
    remain = k - len(chosen_idxs)
    if remain > 0:
        # Query scaffold embedding
        if embed_type == "fp":
            query_scaf_emb = mol_to_fp(query_scaf, radius=2, n_bits=1024)
        elif embed_type in {"small", "base", "large"}:
            query_scaf_emb = get_molt5_embedding([query_scaf], tokenizer, model, batch_size=1)[0]

        # 2-1) Compute similarity for each cluster
        cluster_sims = []
        for rep_scaf, scaf_list in clusters_scaf.items():
            if query_scaf in scaf_list:
                continue

            scaf_emb_list = []
            for scaf in scaf_list:
                if scaf in scaf_emb_dict:
                    scaf_emb_list.append(scaf_emb_dict[scaf])
                else:
                    raise AssertionError(f"{scaf} is not in `scaf_emb_dict`. Review code.")
            mean_sim = compute_similarity(
                query_scaf_emb, scaf_emb_list, embed_type, aggregate="mean", batch_size=batch_size
            )
            cluster_sims.append((rep_scaf, mean_sim))

        if debug:
            if random.random() < 0.01:  # 1% 확률로 샘플링 검증
                print(f"\n[QUERY DEBUG] Query Scaf: {query_scaf}")
                print(f"  -> Need {remain} more shots. Checking neighbors...")
                for rank, (r_scaf, score) in enumerate(cluster_sims[:3]):
                    is_merged = "merge" in r_scaf
                    members = clusters_scaf[r_scaf]
                    print(f"     Rank {rank+1}: {r_scaf} (Merged={is_merged}, Size={len(members)}) | Sim: {score:.4f}")
                    if is_merged:
                        print(f"       -> Member Scaffolds: {members[:2]} ...")  # 멤버 확인

        # 2-2) Select from clusters in descending similarity order
        cluster_sims = sorted(cluster_sims, key=lambda x: x[1], reverse=True)
        for rep_scaf, _ in cluster_sims:
            if len(chosen_idxs) >= k:
                break

            # Flatten indices for merged clusters
            if "merge" in rep_scaf:
                idxs = []
                for scaf in clusters_scaf[rep_scaf]:
                    idxs.extend(clusters_idx[scaf])
            else:
                idxs = clusters_idx[rep_scaf]

            step_info["similar_cluster"]["n_clusters_used"] += 1
            remain = k - len(chosen_idxs)

            chosen, n_used, usage_type = sample_cluster(idxs, remain)
            chosen_idxs += chosen
            chosen_clusters += [rep_scaf] * n_used
            detail = {
                "rep_scaf": rep_scaf,
                "cluster_size": len(idxs),
                "n_used": n_used,
                "usage_type": usage_type,
            }
            if "merge" in rep_scaf:
                detail["scaf_list"] = clusters_scaf[rep_scaf]
            step_info["similar_cluster"]["clusters_detail"].append(detail)

    # 3) Fallback: sampling from the entire pool (ideally, never executed)
    remain = k - len(chosen_idxs)
    if remain > 0:
        head_print(" [WARNING] ", sep_line=False)
        print("[WARN] FALLBACK EXECUTED !!")
        pool_idxs = [i for i in emb_dict.keys() if i not in chosen_idxs]
        chosen = random.sample(pool_idxs, remain)

        chosen_idxs += chosen
        chosen_clusters += ["fallback"] * remain
        step_info["fallback"]["n_used"] = len(chosen)

    assert len(chosen_idxs) == k, f"Over/Under-sampled - expected {k} but sampled {len(chosen_idxs)}"

    return chosen_idxs, chosen_clusters, step_info


def retrieve_whole_query(
    test_df,
    clusters_idx,
    clusters_scaf,
    scaf_emb_dict,
    emb_dict,
    k,
    sample_strategy,
    embed_type,
    tokenizer,
    model,
    batch_size=32,
    debug=False,
):
    """
    This function executes `pool_clustering` for all queries in parallel,
    derives `usage_summary` by processing the `step_info` of each query, and
    returns `query_to_shot` and `query_to_step_info`.

    `usage_summary` (dict) logs the usage statistics of shots to verify the effectiveness of `sample_strategy`:
        - `usage_list` (list[str]): usage type per shot, format "{from}_{usage_type}"
            - `from`: how the shot was selected ("same" scaffold, "similar" scaffold, or "fallback")
            - `usage_type`: see `pool_clustering` for details
        - `frac_all` (float): fraction of shots taken as the entire cluster (usage_type="all")
        - `frac_sampling` (float): fraction of shots selected via `sample_strategy` (usage_type="sampling")
        - `num_fallback` (int): number of fallback shots (usage_type="fallback")

    Returns:
        query_to_shot (dict): query_raw_ID -> shot information
            {
                "query_smiles" (str): query molecule smiles,
                "query_scaffold" (str): query scaffold smiles,
                "shot_idx_list" (list[str]): train_df raw_ID list,
                "shot_cluster_idx" (list[int]): `chosen_clusters` from `pool_clustering` function,
                "shot_cluster_size" (list[int]): size of each cluster from which shot is selected,
                "num_unique" (int): number of unique clusters in `shot_cluster_idx`,
                "usage_summary" (dict): described above,
            }
        }
        query_to_step_info (dict): query_raw_ID -> step information
            {
                all key, values are same as `step_info` from `pool_clustering`,
                plus "statistics" (dict), same as `usage_summary`
            }
    """

    def process_query(raw_id, q_smi, q_scaf):
        try:
            topk_idxs, chosen_clusters, step_info = pool_clustering(
                q_smi,
                q_scaf,
                clusters_idx,
                clusters_scaf,
                scaf_emb_dict,
                emb_dict,
                k=k,
                sample_strategy=sample_strategy,
                embed_type=embed_type,
                tokenizer=tokenizer,
                model=model,
                batch_size=batch_size,
                debug=debug,
            )
            cluster_sizes = [len(clusters_idx[c]) for c in chosen_clusters]
            num_unique = cluster_sizes.count(1)

            # Statistics of step info
            usage_list = []
            same_info = step_info["same_cluster"]
            usage_list += [f"same_{same_info['usage_type']}"] * same_info["n_used"]
            for cluster in step_info["similar_cluster"]["clusters_detail"]:
                usage_list += [f"similar_{cluster['usage_type']}"] * cluster["n_used"]
            fallback_n = step_info["fallback"]["n_used"]
            if fallback_n > 0:
                usage_list += ["fallback"] * fallback_n

            # Calculate statistics
            counter = Counter(usage_list)
            total_used = len(usage_list)
            frac_all = (counter["same_all"] + counter["similar_all"]) / total_used if total_used > 0 else 0
            frac_sampling = (
                (counter["same_sampling"] + counter["similar_sampling"]) / total_used if total_used > 0 else 0
            )
            num_fallback = counter.get("fallback", 0)

            step_stat = {
                "usage_list": usage_list,
                "frac_all": frac_all,
                "frac_sampling": frac_sampling,
                "num_fallback": num_fallback,
            }

            shot_info = {
                "query_smiles": q_smi,
                "query_scaffold": q_scaf,
                "shot_idx_list": topk_idxs,
                "shot_cluster_idx": chosen_clusters,
                "shot_cluster_size": cluster_sizes,
                "num_unique": num_unique,
                "usage_summary": step_stat,
            }
            step_info["statistics"] = step_stat
            return raw_id, shot_info, step_info
        except Exception as e:
            print(f"Error processing query {raw_id}: {q_smi}, {q_scaf}")
            raise

    # Process all queries in parallel
    query_to_shot = {}
    query_to_step_info = {}
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(process_query, str(r.raw_ID), r.smiles, r.scaffold_smiles)
            for r in test_df.itertuples(index=False)
        ]
        with tqdm(total=len(futures), desc="Processing queries") as pbar:
            for fut in as_completed(futures):
                try:
                    raw_id, shot_info, step_info = fut.result()
                    raw_id = str(raw_id)
                    query_to_shot[raw_id] = shot_info
                    query_to_step_info[raw_id] = step_info
                    pbar.update(1)
                except Exception as e:
                    raise

    return query_to_shot, query_to_step_info


def prepare_query_cluster_data(
    test_df,
    clusters_idx,
    clusters_scaf,
    scaf_emb_dict,
    emb_dict,
    k=8,
    sample_strategy="random",
    embed_type="fp",
    shot_path=None,
    tokenizer=None,
    model=None,
    batch_size=32,
    debug=False,
):
    """Prepare cluster usage data for queries, either load from cache or run new retrieval."""

    # Load cached cluster usage if exists
    if shot_path and os.path.exists(shot_path):
        query_to_shot = load_json(shot_path)
        print(f"Loaded `query_to_shot` from `{shot_path}`")
        return query_to_shot

    # Run new cluster usage retrieval if not cached
    print("No cached cluster usage found. Running new cluster usage retrieval ...")
    test_df["scaffold_smiles"] = test_df["smiles"].apply(get_scaffold_smiles)
    n_missing = test_df["scaffold_smiles"].isnull().sum()
    if n_missing > 0:
        print(f"[WARNING] {n_missing} queries have invalid scaffold and will be removed.")
        test_df = test_df.dropna(subset=["scaffold_smiles"])

    if embed_type in {"small", "base", "large"}:
        assert tokenizer is not None and model is not None, "tokenizer and model must be provided when using MolT5"

    query_to_shot, query_to_step_info = retrieve_whole_query(
        test_df,
        clusters_idx,
        clusters_scaf,
        scaf_emb_dict,
        emb_dict,
        k,
        sample_strategy,
        embed_type,
        tokenizer,
        model,
        batch_size,
        debug=debug,
    )

    # Save cluster usage if not cached
    if shot_path:
        step_path = shot_path.replace("shot", "step_info")
        save_json(query_to_shot, shot_path)
        save_json(query_to_step_info, step_path)
        print(f"Saved cluster usage per query in {os.path.dirname(shot_path)}")
        print(f"  - {os.path.basename(shot_path)} : shot info")
        print(f"  - {os.path.basename(step_path)} : step info")

    return query_to_shot


# ===================================
# 3. Visualization
# ===================================
# Contains:
#   - plot_cluster_stats: cluster size distribution
#   - plot_unique_cluster_pie: *unique_cluster* usage per query
#   - plot_sample_strategy_hist: `sample_strategy` usage per query
#   - visualize_query_shots: visualize sample query and its selected clusters
# ===================================
def plot_cluster_stats(clusters_idx, save_path=None, show=True):
    """Plot cluster size distribution as a bar chart with log scale."""

    # === [설정] 크기 조절 변수 ===
    DOT_SIZE = 100  # 점 크기 (기존 대비 3~4배 확대)
    LINE_WIDTH = 3  # 추세선 굵기
    fig_width = 15.92 / 2.54  # 약 6.26 inches
    # 3. 전역 폰트 및 사이즈 설정
    rcParams = {
        # [크기 설정]
        "font.size": 18,  # 본문과 동일한 기본 크기 (핵심)
        "axes.labelsize": 18,  # 축 제목 (X, Y Label)
        "axes.titlesize": 20,  # 그래프 제목 (필요 시 약간 크게)
        "xtick.labelsize": 17,  # 눈금 라벨은 본문보다 약간 작게 하는 것이 관례
        "ytick.labelsize": 17,
        "legend.fontsize": 17,  # 범례도 약간 작게
        # [폰트 종류 설정]
        "font.family": "serif",  # 세리프(명조) 계열 사용
        "font.serif": ["Latin Modern Roman"],
        # ^ 1순위: Latin Modern (시스템에 설치 필요), 2순위: Matplotlib 기본, 3순위: 타임즈
        # [그림 크기 및 해상도]
        "figure.figsize": (fig_width, fig_width * 0.75),  # 너비는 고정, 높이는 비율에 맞게 조절
        "figure.dpi": 300,  # 고해상도 설정
        # [LaTeX 수식 렌더링 (선택 사항)]
        # True로 하면 LaTeX 엔진을 직접 써서 렌더링하므로 100% 똑같아지지만,
        # PC에 LaTeX가 설치되어 있어야 하고 속도가 느려집니다.
        "text.usetex": False,
    }

    # ============================

    sizes = [len(v) for v in clusters_idx.values()]
    unique_count = sum([1 for s in sizes if s == 1])
    total_count = len(sizes)

    counter = Counter(sizes)
    x = list(counter.keys())
    y = list(counter.values())

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x, y, log=True)
    ax.set_xlabel("Cluster size")
    ax.set_ylabel("Number of clusters (log scale)")
    ax.set_title(f"Cluster size distribution (Unique/Total: {unique_count}/{total_count})")

    save_show_close(fig, save_path, "cluster size distribution plot", show)

    # log-log
    with plt.rc_context(rcParams):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(x, y, s=DOT_SIZE, alpha=0.6, edgecolors="b", label="Data")

        # [추세선 계산 로직]
        # log(y) = slope * log(x) + intercept 형태의 선형 회귀
        log_x = np.log10(x)
        log_y = np.log10(y)

        # 1차 회귀 수행 (기울기, 절편)
        slope, intercept = np.polyfit(log_x, log_y, 1)

        # R^2 (결정 계수) 계산: 데이터가 추세선에 얼마나 잘 맞는지
        y_pred_log = slope * log_x + intercept
        ss_res = np.sum((log_y - y_pred_log) ** 2)
        ss_tot = np.sum((log_y - np.mean(log_y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)

        # 추세선 그리기 (x 범위에 맞춰 매끄러운 선 생성)
        x_fit = np.logspace(np.log10(min(x)), np.log10(max(x)), 100)  # 로그 스케일 등간격 x
        y_fit = (10**intercept) * (x_fit**slope)  # y = 10^b * x^a

        ax.plot(x_fit, y_fit, "r--", label=f"Trend: $y \\propto x^{{{slope:.2f}}}$ ($R^2={r_squared:.2f}$)")

        # 그래프 스타일 설정
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, which="both", ls="--", alpha=0.3)
        ax.set_xlabel("Cluster size (log scale)")
        ax.set_ylabel("Number of clusters (log scale)")
        ax.set_title(f"Cluster size distribution (log-log scale)")
        ax.legend()

        # 두 번째 그래프 저장
        save_show_close(
            fig,
            save_path.replace("cluster_size_dist", "cluster_size_dist_log-log"),
            "cluster size distribution plot (log-log scale)",
            show,
        )


def plot_unique_cluster_pie(query_to_shot, k=8, save_path=None, show=True):
    """Plot pie chart of unique cluster usage per query."""

    # Count unique clusters per query
    counts = [v["num_unique"] for v in query_to_shot.values()]
    counter = Counter(counts)
    labels, sizes = zip(*[(l, s) for l, s in sorted(counter.items()) if s > 0])

    # Plot pie chart
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(sizes, labels=None, startangle=90, counterclock=False)
    legend_labels = [f"{label}: {size}" for label, size in zip(labels, sizes)]
    ax.legend(
        legend_labels,
        title="Unique clusters used",
        loc="center left",
        bbox_to_anchor=(1, 0.5),
    )
    ax.set_title(f"Unique cluster usage per query (top-{k})")

    save_show_close(fig, save_path, "unique cluster usage pie chart", show)


def plot_sample_strategy_hist(query_to_shot, sample_strategy="random", save_path=None, show=True):
    """Plot histogram of `sample strategy` usage per query."""

    # Calculate statistics
    stats_list = []
    for raw_id, info in query_to_shot.items():
        usage = info["usage_summary"]
        stat = {
            "raw_id": raw_id,
            "frac_all": usage["frac_all"],
            "frac_sampling": usage["frac_sampling"],
            "num_fallback": usage["num_fallback"],
        }
        stats_list.append(stat)
    df_stats = pd.DataFrame(stats_list)

    mean_frac_all = df_stats["frac_all"].mean()
    mean_frac_sampling = df_stats["frac_sampling"].mean()
    num_fallback_total = df_stats["num_fallback"].sum()
    print(f"Mean frac_all: {mean_frac_all:.3f}")
    print(f"Mean frac_sampling: {mean_frac_sampling:.3f}")
    print(f"Total fallback occurrences: {num_fallback_total}")

    # Plot histograms
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(df_stats["frac_all"], bins=10)
    axes[0].set_title("Distribution of frac_all")
    axes[0].set_xlabel("Fraction of shots taken as *all*")
    axes[0].set_ylabel("Number of queries")

    axes[1].hist(df_stats["frac_sampling"], bins=10)
    axes[1].set_title("Distribution of frac_sampling")
    axes[1].set_xlabel(f"Fraction of shots taken as *{str(sample_strategy)}*")
    axes[1].set_ylabel("Number of queries")

    save_show_close(fig, save_path.replace("unique", "sampling"), "usage fraction histograms", show)


def visualize_query_shots(
    query_smiles,
    clusters_idx,
    clusters_scaf,
    scaf_emb_dict,
    emb_dict,
    k=8,
    sample_strategy="random",
    embed_type="fp",
    save_path=None,
    show=True,
    tokenizer=None,
    model=None,
    batch_size=32,
):
    """
    Visualize query and its selected clusters.

    The figure is organized in 3 sections:
        1) Query molecule and its scaffold (centered)
        2) Unmerged clusters' scaffolds (up to 4 per row, left-aligned)
        3) Merged clusters' scaffolds (up to 4 per row, left-aligned)
    """

    # Get query info
    query_scaf = get_scaffold_smiles(query_smiles)
    _, chosen_clusters, _ = pool_clustering(
        query_smiles,
        query_scaf,
        clusters_idx,
        clusters_scaf,
        scaf_emb_dict,
        emb_dict,
        k=k,
        sample_strategy=sample_strategy,
        embed_type=embed_type,
        tokenizer=tokenizer,
        model=model,
        batch_size=batch_size,
    )

    # Prepare query
    q_mol = [safe_MolFromSmiles(s) for s in [query_smiles, query_scaf]]
    q_lgd = ["molecule", "scaffold"]

    # Prepare cluster scaffolds
    umrg_mol = []
    umrg_lgd = []
    mrg_mol = []
    mrg_lgd = []
    added_clusters = set()
    for scaf in chosen_clusters:
        if scaf in added_clusters:
            continue
        if "merge" in scaf:
            scaf_list = set(clusters_scaf[scaf])
            for s in scaf_list:
                mrg_mol.append(safe_MolFromSmiles(s))
                mrg_lgd.append(s)
        else:
            umrg_mol.append(safe_MolFromSmiles(scaf))
            umrg_lgd.append(scaf)
        added_clusters.add(scaf)

    # Figure layout
    N_PER_ROW = 4
    q_rows = 1
    umrg_rows = math.ceil(len(umrg_mol) / N_PER_ROW)
    mrg_rows = math.ceil(len(mrg_mol) / N_PER_ROW)
    fig = plt.figure(figsize=(4 * N_PER_ROW, 4 * (q_rows + umrg_rows + mrg_rows)), constrained_layout=True)
    outer = fig.add_gridspec(3, 1, height_ratios=[1, umrg_rows, mrg_rows], hspace=0.4)

    def add_row(sec_idx, mols, mol_legends, rows, title, col_offset=0):
        ax_section = outer[sec_idx].subgridspec(rows, N_PER_ROW).subplots()
        if rows == 1:
            ax_section = [ax_section]

        for i, (mol, legend) in enumerate(zip(mols, mol_legends)):
            r, c = divmod(i, N_PER_ROW)
            ax = ax_section[r][c + col_offset]
            safe_MolToMPL(mol, ax=ax)
            ax.set_title(legend, fontsize=11)

        fig.text(0.02, 0.95 - 0.1 * sec_idx, title, fontsize=16, fontweight="bold")

    # Add rows to the figure
    idx = 0
    add_row(idx, q_mol, q_lgd, rows=q_rows, title="Query", col_offset=(N_PER_ROW - len(q_mol)) // 2)
    if umrg_rows > 0:
        add_row(idx, umrg_mol, umrg_lgd, rows=umrg_rows, title=f"{len(umrg_mol)} unmerged clusters")
        idx += 1
    if mrg_rows > 0:
        add_row(idx, mrg_mol, mrg_lgd, rows=mrg_rows, title=f"{len(mrg_mol)} merged clusters")

    for ax in fig.axes:
        ax.axis("off")
    save_show_close(fig, save_path, "query & selected clusters", show, tight=False)


# ===================================
# 4. Cluster Caching Management
# ===================================
# Cluster caching functions: saving, loading, and checking clustering data
# File structure (under `cache_dir`):
#   - Common:
#     - clusters_idx.pk: molecule indices per scaffold in each cluster
#     - clusters_scaf.pk: list of scaffolds in each cluster
#     - meta.json: metadata about the clustering
#     - scaf_emb_dict:
#       - *.pk: ECFP (dict)
#       - *.pt: MolT5 (dict, saved via torch.save)
# ===================================
def cluster_paths(cache_dir):
    cluster_idx_path = os.path.join(cache_dir, f"clusters_idx.pk")
    cluster_scaf_path = os.path.join(cache_dir, f"clusters_scaf.pk")
    meta_path = os.path.join(cache_dir, f"meta.json")

    if "fp" in cache_dir:
        cluster_emb_path = os.path.join(cache_dir, f"scaf_emb_dict.pk")
    elif "molt5" in cache_dir:
        cluster_emb_path = os.path.join(cache_dir, f"scaf_emb_dict.pt")

    return cluster_idx_path, cluster_scaf_path, cluster_emb_path, meta_path


def find_cluster(cache_dir):
    cluster_idx_path, cluster_scaf_path, cluster_emb_path, meta_path = cluster_paths(cache_dir)

    if not os.path.exists(cluster_idx_path):
        print(f"Missing cluster index file: {cluster_idx_path}")
        return False
    if not os.path.exists(cluster_scaf_path):
        print(f"Missing cluster scaffold file: {cluster_scaf_path}")
        return False
    if not os.path.exists(cluster_emb_path):
        print(f"Missing cluster embedding file: {cluster_emb_path}")
        return False
    if not os.path.exists(meta_path):
        print(f"Missing cluster metadata file: {meta_path}")
        return False

    return True


def save_cluster(cache_dir, cluster_idx, cluster_scaf, scaf_emb_dict, meta):
    cluster_idx_path, cluster_scaf_path, cluster_emb_path, meta_path = cluster_paths(cache_dir)

    save_pickle(cluster_idx, cluster_idx_path)
    save_pickle(cluster_scaf, cluster_scaf_path)
    print(f"Saved clustering results in {os.path.dirname(cluster_idx_path)}")
    print(f"  - {os.path.basename(cluster_idx_path)} : cluster_idx")
    print(f"  - {os.path.basename(cluster_scaf_path)} : cluster_scaf")

    if cluster_emb_path.endswith(".pk"):
        save_pickle(scaf_emb_dict, cluster_emb_path)
        print(f"Saved ECFP scaf_emb_dict by pickle to {cluster_emb_path}")
    elif cluster_emb_path.endswith(".pt"):
        torch.save(scaf_emb_dict, cluster_emb_path)
        print(f"Saved MolT5 scaf_emb_dict by torch.save to {cluster_emb_path}")

    save_json(meta, meta_path)
    print(f"Saved clustering metadata to {meta_path}")


def load_cluster(cache_dir):
    cluster_idx_path, cluster_scaf_path, cluster_emb_path, meta_path = cluster_paths(cache_dir)

    meta = load_json(meta_path)
    cluster_idx = load_pickle(cluster_idx_path)
    cluster_scaf = load_pickle(cluster_scaf_path)
    if cluster_emb_path.endswith(".pk"):
        scaf_emb_dict = load_pickle(cluster_emb_path)
    elif cluster_emb_path.endswith(".pt"):
        scaf_emb_dict = torch.load(cluster_emb_path, map_location=torch.device("cpu"))

    return cluster_idx, cluster_scaf, scaf_emb_dict, meta


# ===================================
# 5. Pipeline Execution & Visualization
# ===================================
# Orchestrates clustering, query-shot selection, caching, and plotting.
# Contains:
#   - visualize_eps: main pipeline for clustering, query-shot selection, and plotting
#   - get_args: parse command line arguments
#   - __main__: load dataset, prepare embeddings, and run `visualize_eps`
# ===================================
def visualize_eps(
    eps,
    train_df,
    valid_df,
    test_df,
    emb_dict,
    dataset_name,
    embed_model="fp",
    embed_batch_size=32,
    k=8,
    sample_strategy="random",
    show=True,
    debug=False,
):
    # Prepare cache paths and model
    embed_tag = f"molt5-{embed_model}" if embed_model != "fp" else "fp"
    eps_tag = f"eps={eps}" if eps else "no_merge"
    tag = f"{embed_tag}_{eps_tag}_{sample_strategy}-sampling"
    cache_dir = os.path.join(DATA_ROOT, dataset_name, tag)
    plot_dir = os.path.join(cache_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    bar_plot_path = os.path.join(plot_dir, f"cluster_size_dist.png")
    pie_plot_path = os.path.join(plot_dir, f"unique_usage_k={k}.png")
    sampling_plot_path = os.path.join(plot_dir, f"sampling_usage_k={k}.png")
    query_shot_path_valid = os.path.join(cache_dir, f"query_to_shot_valid_k={k}.json")
    query_shot_path_test = os.path.join(cache_dir, f"query_to_shot_test_k={k}.json")
    query_grid_path = os.path.join(plot_dir, f"example_mol_grid_k={k}.png")

    if embed_model in {"small", "base", "large"}:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Use", device)
        tokenizer, model = get_molt5_model(embed_model, device)
    elif embed_model == "fp":
        tokenizer, model = None, None

    # Scaffold clustering, Bunita merging (optional) and visualize cluster size distribution
    msg = "Scaffold Clustering"
    if eps > 0:
        msg += f" and Bunita Merging"
    head_print(msg)
    if find_cluster(cache_dir):
        clusters_idx, clusters_scaf, scaf_emb_dict, meta = load_cluster(cache_dir)
        print(f"Cached clustering results loaded from {cache_dir}")
    else:
        print("No cached clustering results found. Running new clustering ...")
        clusters_idx, clusters_scaf, scaf_emb_dict = scaffold_bunita_clustering(
            train_df,
            embed_model=embed_model,
            embed_batch_size=embed_batch_size,
            eps=eps,
            tokenizer=tokenizer,
            model=model,
            debug=debug,
        )
        meta = {
            "dataset": dataset_name,
            "embed_model": embed_model,
            "eps": eps,
            "sample_strategy": sample_strategy,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "n_train": len(train_df),
            "n_test": len(test_df),
            "n_clusters": len(clusters_scaf),
            "n_unique_clusters": sum(len(v) == 1 for v in clusters_idx.values()),
            "unique_ratio": round(sum(len(v) == 1 for v in clusters_idx.values()) / len(clusters_scaf), 4),
        }
        save_cluster(cache_dir, clusters_idx, clusters_scaf, scaf_emb_dict, meta)

    head_print("Plotting cluster size distribution")
    plot_cluster_stats(clusters_idx, save_path=bar_plot_path, show=show)

    # Caching and visualize cluster usage data for test queries
    head_print("Preparing cluster usage data for VALID queries")
    prepare_query_cluster_data(
        valid_df,  # Valid DF 사용
        clusters_idx,
        clusters_scaf,
        scaf_emb_dict,
        emb_dict,
        k=k,
        sample_strategy=sample_strategy,
        embed_type=embed_model,
        shot_path=query_shot_path_valid,  # Valid용 경로
        tokenizer=tokenizer,
        model=model,
        batch_size=embed_batch_size,
        debug=debug,
    )

    head_print("Preparing cluster usage data for TEST queries")
    query_to_shot_test = prepare_query_cluster_data(
        test_df,
        clusters_idx,
        clusters_scaf,
        scaf_emb_dict,
        emb_dict,
        k=k,
        sample_strategy=sample_strategy,
        embed_type=embed_model,
        shot_path=query_shot_path_test,
        tokenizer=tokenizer,
        model=model,
        batch_size=embed_batch_size,
        debug=debug,
    )

    head_print("Plotting unique cluster usage per query")
    plot_unique_cluster_pie(query_to_shot_test, k=k, save_path=pie_plot_path, show=show)

    head_print("Plotting sample strategy usage per query")
    plot_sample_strategy_hist(
        query_to_shot_test, sample_strategy=sample_strategy, save_path=sampling_plot_path, show=show
    )

    # Example query
    head_print("Visualizing example query and selected clusters")
    example_query = random.choice(test_df["smiles"].tolist())
    visualize_query_shots(
        example_query,
        clusters_idx,
        clusters_scaf,
        scaf_emb_dict,
        emb_dict,
        k=k,
        sample_strategy=sample_strategy,
        embed_type=embed_model,
        save_path=query_grid_path,
        show=show,
        tokenizer=tokenizer,
        model=model,
        batch_size=embed_batch_size,
    )


def get_args():
    # fmt: off
    parser = argparse.ArgumentParser(description="Scaffold-based clustering and query-shot visualization")
    parser.add_argument("--seed", type=int, default=1013, help="Random seed")
    parser.add_argument("--dataset_name", type=str, default="BBBP", help="Dataset name (e.g., BBBP, BACE, etc.)")
    parser.add_argument("--embed_model", type=str, default="fp", choices=["fp", "small", "base", "large"], help="How to embedding molecule.")
    parser.add_argument("--sample_strategy", type=str, default="random", choices=["random", "similarity"], help="How to sample shots in each chosen cluster.")
    parser.add_argument("--eps", type=float, default=0.6, help="Bunita merge similarity threshold (0 < eps <= 1)")
    parser.add_argument("--k", type=int, default=8, help="Number of shots to select per query")
    parser.add_argument("--radius", type=int, default=2, help="Morgan Fingerprint radius.")
    parser.add_argument("--n_bits", type=int, default=1024, help="The number of bits of Morgan Fingerprint.")
    parser.add_argument("--batch_size", type=int, default=256, help="MolT5로 임베딩 할 때 배치로 묶을 건데, 그 배치 사이즈")
    parser.add_argument("--show", action="store_true", default=False, help="Wheter to show the plots or not.")
    parser.add_argument("--debug", action="store_true", default=False, help="Enable debug mode: print extra information")
    # fmt: on

    args = parser.parse_args()

    args.dataset_name = args.dataset_name.upper()
    assert (
        args.dataset_name in POSSIBLE_DATASETS
    ), f"dataset_name must be one of {POSSIBLE_DATASETS} (got {args.dataset_name})"
    assert args.eps >= 0.0, f"eps must be non-negative (got {args.eps})"
    assert args.eps <= 1.0, f"eps must be less than or equal to 1.0 (got {args.eps})"

    return args


if __name__ == "__main__":
    args = get_args()
    head_print(f"Visualize with args:")
    pprint(vars(args))
    set_all_seeds(args.seed)

    # Load and validate dataset
    train_df, valid_df, test_df = load_dataset(args.dataset_name, args.embed_model, verbose=False)
    print(f"Loaded `train_df`, `valid_df` and `test_df` from `./data/{args.dataset_name}`")
    log_df("  - train: train.csv", df=train_df)
    log_df("  - valid: valid.csv", df=valid_df)
    log_df("  - test:  test.csv", df=test_df)
    for c in ["raw_ID", "smiles", "label", "embed"]:
        assert c in train_df.columns, f"{c} not in train_df"
        assert c in valid_df.columns, f"{c} not in valid_df"
        assert c in test_df.columns, f"{c} not in test_df"
        assert train_df[c].isnull().sum() == 0, f"{c} has NaN values in train_df"
        assert valid_df[c].isnull().sum() == 0, f"{c} has NaN values in valid_df"
        assert test_df[c].isnull().sum() == 0, f"{c} has NaN values in test_df"
        if c not in {"label", "embed"}:
            assert train_df[c].duplicated().sum() == 0, f"{c} has duplicated values in train_df"
            assert test_df[c].duplicated().sum() == 0, f"{c} has duplicated values in test_df"

    # Prepare embedding dictionary
    emb_dict = {str(row.raw_ID): row.embed for row in train_df.itertuples()}

    # Main function
    visualize_eps(
        eps=args.eps,
        train_df=train_df,
        valid_df=valid_df,
        test_df=test_df,
        emb_dict=emb_dict,
        dataset_name=args.dataset_name,
        embed_model=args.embed_model,
        embed_batch_size=args.batch_size,
        k=args.k,
        sample_strategy=args.sample_strategy,
        show=args.show,
        debug=args.debug,
    )
