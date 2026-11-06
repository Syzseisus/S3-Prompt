from tqdm import tqdm

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch

from utils.constants import REGRESSION, CLASSIFICATION

# Special Tokens
SCAF_TOKEN = "<|scaffold|>"
SIDE_TOKEN = "<|decoration|>"


class S3PromptDataset(Dataset):
    def __init__(self, query_df, shot_df, query_to_shot, tokenizer, prompter, decomposer, k_shot, is_train=False, is_cls=True, label_stats=None):
        self.query_df = query_df
        # Shot lookup 최적화: 문자열 인덱스 변환
        shot_df["raw_ID"] = shot_df["raw_ID"].astype(str)
        self.shot_df = shot_df.set_index("raw_ID")
        
        self.query_to_shot = query_to_shot
        self.tokenizer = tokenizer
        self.prompter = prompter
        self.decomposer = decomposer
        self.k_shot = k_shot
        self.is_train = is_train
        self.is_cls = is_cls

        # Regression label standardization (z-score using training set stats)
        self.label_stats = label_stats
        self.label_mean = label_stats["mean"] if label_stats else None
        self.label_std = label_stats["std"] if label_stats else None

        # [Optimization 1] 모든 분자(Query + Shot) 미리 분해하여 캐싱
        self.mol_cache = {}
        self._precompute_all_molecules()

        # [Optimization 2] 유효한 샘플만 리스트에 미리 저장 (Pre-build)
        # dict 대신 list를 쓰되, 유효성 검사를 통과한 데이터만 append 하므로 안전합니다.
        self.data_list = []
        print(f"[Info] Pre-building dataset from {len(self.query_df)} queries...")
        self._build_dataset()
        print(f"[Info] Final Dataset Size: {len(self.data_list)} (Skipped: {len(self.query_df) - len(self.data_list)})")

    def _get_safe_graph(self, graph):
        """Null Graph 방지용 더미 그래프 생성"""
        if graph is not None and hasattr(graph, "x") and graph.x.size(0) > 0:
            return graph
        
        # 껍데기만 있는 더미 그래프 (Feature Zero-out)
        dummy = self.decomposer.smiles_to_graph("C")
        if hasattr(dummy, "x") and dummy.x is not None:
            dummy.x.zero_()
        return dummy

    def _precompute_all_molecules(self):
        """
        데이터셋에 존재하는 모든 Unique ID(Query & Shot)에 대해
        RDKit Decomposition 및 Graph 변환을 수행하고 캐시에 저장합니다.
        """
        # 1. 분해해야 할 모든 ID 수집
        all_query_ids = set(self.query_df["raw_ID"].astype(str))
        all_shot_ids = set(self.shot_df.index)
        all_ids = all_query_ids.union(all_shot_ids)

        print(f"[Info] Decomposing {len(all_ids)} unique molecules...")
        
        # 2. DataFrame 통합 (빠른 조회를 위해)
        combined_smiles = {}
        # Shot 우선
        combined_smiles.update(self.shot_df["smiles"].to_dict())
        # Query로 덮어쓰기
        query_dict = dict(zip(self.query_df["raw_ID"].astype(str), self.query_df["smiles"]))
        combined_smiles.update(query_dict)

        for rid in tqdm(all_ids, desc="Graph Gen"):
            smiles = combined_smiles.get(rid)
            if not smiles: continue

            # Decompose
            result = self.decomposer.decompose(smiles)
            
            # Text Parts
            if result is None:
                orig, scaf_s, dec_s = smiles, "None", "None"
                scaf_g, side_gs = None, None
            else:
                orig = result["original_smiles"]
                # SCAF_TOKEN 항상 존재
                scaf_s = SCAF_TOKEN 
                
                # Decoration String Formatting
                if result["decorations"]:
                    dec_list = [SIDE_TOKEN for _ in result["decorations"]]
                    dec_s = "[" + ", ".join(dec_list) + "]"
                else:
                    dec_s = "None"
                
                scaf_g = result["scaf_graphs"]
                side_gs = result["side_graphs"]

            # Safe Graph 처리 미리 적용
            safe_scaf_g = self._get_safe_graph(scaf_g)
            
            # Cache 저장
            self.mol_cache[rid] = {
                "orig": orig,
                "scaf_s": scaf_s,
                "dec_s": dec_s,
                "scaf_g": safe_scaf_g,
                "side_gs": side_gs if side_gs else [] 
            }

    def _build_dataset(self):
        """__getitem__에서 하던 일을 미리 다 해서 self.data_list에 저장"""
        for idx in tqdm(range(len(self.query_df)), desc="Tokenizing"):
            row = self.query_df.iloc[idx]
            raw_id = str(row["raw_ID"])
            q_label = row["label"]
            q_smiles = row["smiles"]

            # 1. Retrieve Shots (유효성 검사)
            if raw_id not in self.query_to_shot:
                continue 
            
            shot_info = self.query_to_shot[raw_id]
            shot_ids = shot_info["shot_idx_list"][: self.k_shot]
            
            # Shot이 부족하거나, Cache에 없는 ID가 있으면 스킵할 수도 있음 (Safety)
            if any(sid not in self.mol_cache for sid in shot_ids):
                continue
            if raw_id not in self.mol_cache:
                continue

            all_scaf_graphs = []
            all_side_graphs = []
            shots_meta_list = []
            
            shots_parts = ["## Examples\n\n"]

            # 2. Build Shots
            for i, sid in enumerate(shot_ids):
                s_data = self.mol_cache[sid]
                s_label = self.shot_df.loc[sid, "label"]
                
                shots_meta_list.append({"smiles": s_data["orig"], "label": s_label})

                if self.is_cls:
                    s_label_str = "Yes" if s_label == 1 else "No"
                else:
                    s_label_val = float(s_label)
                    if self.label_mean is not None and self.label_std is not None:
                        s_label_val = (s_label_val - self.label_mean) / self.label_std
                    s_label_str = f"{s_label_val:.4f}"
                shot_text = (
                    f"### Example {i+1}\n\n"
                    f"- Input: {s_data['orig']}\n"
                    f"[Scaffold]: {s_data['scaf_s']}\n"
                    f"[Decorations]: {s_data['dec_s']}\n"
                    f"- Output:\n"
                    f"[Answer] {s_label_str}\n\n"
                )
                shots_parts.append(shot_text)

                all_scaf_graphs.append(s_data["scaf_g"])
                all_side_graphs.extend(s_data["side_gs"])

            shots_str = "".join(shots_parts)

            # 3. Build Query
            q_data = self.mol_cache[raw_id]
            query_struct_str = (
                f"- Input: {q_data['orig']}\n"
                f"[Scaffold]: {q_data['scaf_s']}\n"
                f"[Decorations]: {q_data['dec_s']}\n"
            )

            all_scaf_graphs.append(q_data["scaf_g"])
            all_side_graphs.extend(q_data["side_gs"])

            # 4. Tokenize
            messages = self.prompter.generate(query_struct_str, shots_str, mode="llama")
            prompt_only = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompt_only += "[Answer]"
            
            full_prompt = prompt_only
            
            # 5. Add Groundtruth if is train
            if self.is_train:
                if self.is_cls:
                    label_str = "Yes" if q_label == 1 else "No"
                else:
                    q_label_val = float(q_label)
                    if self.label_mean is not None and self.label_std is not None:
                        q_label_val = (q_label_val - self.label_mean) / self.label_std
                    label_str = f" {q_label_val:.4f}"
                full_prompt += f"{label_str}{self.tokenizer.eos_token}"

            # 6. Encoding
            enc_full = self.tokenizer(full_prompt, return_tensors="pt", add_special_tokens=False)
            input_ids = enc_full.input_ids.squeeze(0)
            if self.is_train:
                enc_prompt = self.tokenizer(prompt_only, return_tensors="pt", add_special_tokens=False)
                source_len = enc_prompt.input_ids.shape[1]
            else:
                source_len = 0

            # [핵심] 완성된 데이터를 리스트에 추가 (Append)
            # 중간에 continue로 건너뛴 데이터는 리스트에 들어가지 않으므로,
            # 리스트 인덱스 0, 1, 2... 는 항상 유효한 데이터만 가리킵니다.
            self.data_list.append({
                "input_ids": input_ids,
                "source_len": source_len,
                "scaf_graphs": all_scaf_graphs,
                "side_graphs": all_side_graphs,
                "label": q_label,
                "raw_id": raw_id,
                "prompt_text": full_prompt,
                "num_shots": len(shot_ids),
                "metadata": {
                    "smiles": q_smiles,
                    "label": q_label,
                    "shots": shots_meta_list,
                    "mining": shot_info.get("mining"),
                },
            })

    def __len__(self):
        # 유효한 데이터의 개수만 반환
        return len(self.data_list)

    def __getitem__(self, idx):
        # DataLoader가 0 ~ len-1 사이의 idx를 주면 바로 꺼내줌 (O(1))
        return self.data_list[idx]


class S3PromptCollator:
    def __init__(self, decomposer, tokenizer):
        self.decomposer = decomposer
        self.tokenizer = tokenizer
        # Llama usually uses 0 or 2 for padding. Using 0 (unk) is safer than eos (2) for attention masking in some libs
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    def __call__(self, batch):
        input_ids_list = [item["input_ids"] for item in batch]
        max_len = max(len(t) for t in input_ids_list)

        # Left Padding for Generation compatibility (Standard for Llama)
        padded_ids = torch.full((len(batch), max_len), self.pad_token_id, dtype=torch.long)

        # [NEW] Attention Mask 생성 (패딩은 0, 실제 토큰은 1)
        attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)

        batch_scaf_graphs = []
        batch_side_graphs = []
        labels = []
        raw_ids = []
        prompt_texts = []

        batch_metadata = []
        for i, item in enumerate(batch):
            seq_len = len(item["input_ids"])

            # Left Padding 적용
            padded_ids[i, -seq_len:] = item["input_ids"]
            # [NEW] 실제 데이터가 있는 부분만 마스킹 1
            attention_mask[i, -seq_len:] = 1

            batch_scaf_graphs.extend(item["scaf_graphs"])
            batch_side_graphs.extend(item["side_graphs"])
            labels.append(item["label"])
            raw_ids.append(item["raw_id"])
            prompt_texts.append(item["prompt_text"])

            batch_metadata.append(item["metadata"])

        # Empty Batch Handling
        if not batch_scaf_graphs:
            batch_scaf_graphs = [self.decomposer.smiles_to_graph("")]
        if not batch_side_graphs:
            batch_side_graphs = [self.decomposer.smiles_to_graph("")]

        return {
            "input_ids": padded_ids,
            "attention_mask": attention_mask,  # [NEW] 반환값 추가
            "scaf_batch": Batch.from_data_list(batch_scaf_graphs),
            "side_batch": Batch.from_data_list(batch_side_graphs),
            "labels": torch.tensor(labels),
            "raw_ids": raw_ids,
            "prompt_texts": prompt_texts,
            "k_shot": batch[0]["num_shots"],  # Assuming same k for batch
            "metadata": batch_metadata,
        }




class S3PromptCollatorWLabel:
    def __init__(self, decomposer, tokenizer, is_cls=True):
        self.decomposer = decomposer
        self.tokenizer = tokenizer
        self.is_cls = is_cls
        # Llama usually uses 0 or 2 for padding. Using 0 (unk) is safer than eos (2) for attention masking in some libs
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    def __call__(self, batch):
        input_ids_list = [item["input_ids"] for item in batch]
        max_len = max(len(t) for t in input_ids_list)

        # Left Padding for Generation compatibility (Standard for Llama)
        padded_ids = torch.full((len(batch), max_len), self.pad_token_id, dtype=torch.long)

        # [NEW] Attention Mask 생성 (패딩은 0, 실제 토큰은 1)
        attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
        labels = torch.full((len(batch), max_len), -100, dtype=torch.long)

        batch_scaf_graphs = []
        batch_side_graphs = []
        raw_ids = []
        q_labels = []
        prompt_texts = []

        batch_metadata = []
        for i, item in enumerate(batch):
            seq_len = len(item["input_ids"])
            source_len = item.get("source_len", 0)

            # Left Padding 적용
            padded_ids[i, -seq_len:] = item["input_ids"]
            # [NEW] 실제 데이터가 있는 부분만 마스킹 1
            attention_mask[i, -seq_len:] = 1

            labels[i, -seq_len:] = item["input_ids"]
            if source_len > 0:
                labels[i, -seq_len : -seq_len + source_len] = -100

            batch_scaf_graphs.extend(item["scaf_graphs"])
            batch_side_graphs.extend(item["side_graphs"])
            raw_ids.append(item["raw_id"])
            q_labels.append(item["label"])
            prompt_texts.append(item["prompt_text"])

            batch_metadata.append(item["metadata"])

        # Empty Batch Handling
        if not batch_scaf_graphs:
            batch_scaf_graphs = [self.decomposer.smiles_to_graph("")]
        if not batch_side_graphs:
            batch_side_graphs = [self.decomposer.smiles_to_graph("")]

        return {
            "input_ids": padded_ids,
            "attention_mask": attention_mask,  # [NEW] 반환값 추가
            "scaf_batch": Batch.from_data_list(batch_scaf_graphs),
            "side_batch": Batch.from_data_list(batch_side_graphs),
            "labels": labels,
            "q_labels": torch.tensor(q_labels, dtype=torch.long if self.is_cls else torch.float32),
            "raw_ids": raw_ids,
            "prompt_texts": prompt_texts,
            "k_shot": batch[0]["num_shots"],  # Assuming same k for batch
            "metadata": batch_metadata,
        }
