import torch
import torch.nn as nn
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Scaffolds import MurckoScaffold
import matplotlib.pyplot as plt
import io
from PIL import Image
import math
import os

from torch_geometric.data import Data
from ogb.utils.features import atom_to_feature_vector, bond_to_feature_vector


# ==========================================
# 1. Molecule Decomposer (분해 및 텍스트 포맷팅)
# ==========================================
class MoleculeDecomposer:
    def __init__(self):
        pass

    def smiles_to_graph(self, smiles):
        """OGB Feature를 사용하여 SMILES를 PyG Data로 변환"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            # 빈 그래프 반환 (Empty fallback)
            # OGB Atom Feature Length는 보통 9입니다.
            return Data(x=torch.zeros(1, 9, dtype=torch.long), edge_index=torch.empty((2, 0), dtype=torch.long))

        # 1. Node (Atom) Features
        atom_features = []
        for atom in mol.GetAtoms():
            # atom_to_feature_vector returns a list of integers
            atom_features.append(atom_to_feature_vector(atom))

        x = torch.tensor(atom_features, dtype=torch.long)

        # 2. Edge Index & Edge Features
        edge_indices = []
        edge_features = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_indices.append([i, j])
            edge_indices.append([j, i])

            bond_feat = bond_to_feature_vector(bond)
            edge_features.append(bond_feat)
            edge_features.append(bond_feat)  # Undirected graph

        if len(edge_indices) > 0:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_features, dtype=torch.long)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 3), dtype=torch.long)  # OGB bond feat dim is usually 3

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    def decompose(self, smiles: str):
        """
        SMILES를 입력받아 Scaffold와 Decorations(잔가지)로 분해하고,
        연결 부위에 Isotope([1c], [1*] 등)를 태깅하여 반환.
        + [수정] PyG Graph 객체도 생성하여 반환
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # 1. Core(Scaffold) 추출
        core = MurckoScaffold.GetScaffoldForMol(mol)

        # Scaffold가 없는 경우 (전체가 사슬이거나 등등)
        if core.GetNumAtoms() == 0:
            # [수정] 그래프 생성 로직 추가
            # Scaffold가 없으므로 빈 그래프, 전체 분자가 잔가지(Decoration) 취급
            scaf_graph = self.smiles_to_graph("")
            side_graph = self.smiles_to_graph(smiles)

            return {
                "original_smiles": smiles,
                "original_mol": mol,
                "scaffold_smiles": "",
                "scaffold_mol": None,
                "decorations": [{"smiles": smiles, "id": 0, "mol": mol}],
                # [NEW] Dataset에서 요구하는 키 추가
                "scaf_graphs": scaf_graph,
                "side_graphs": [side_graph],
            }

        # 2. Side Chain 분리
        side_chain_mol = Chem.ReplaceCore(mol, core, labelByIndex=True)

        decorations_data = []
        anchor_map = {}

        if side_chain_mol:
            frags = Chem.GetMolFrags(side_chain_mol, asMols=True)
            r_id_counter = 1

            for frag in frags:
                anchor_core_idx = -1
                dummy_idx = -1

                for atom in frag.GetAtoms():
                    if atom.GetAtomicNum() == 0:
                        anchor_core_idx = atom.GetIsotope()
                        dummy_idx = atom.GetIdx()
                        break

                if anchor_core_idx != -1:
                    if anchor_core_idx not in anchor_map:
                        anchor_map[anchor_core_idx] = r_id_counter
                        r_id_counter += 1
                    current_r_id = anchor_map[anchor_core_idx]
                else:
                    current_r_id = 0

                if dummy_idx != -1:
                    frag.GetAtomWithIdx(dummy_idx).SetIsotope(current_r_id)

                frag_smi = Chem.MolToSmiles(frag)
                decorations_data.append({"smiles": frag_smi, "id": current_r_id, "mol": frag})

        # 3. Scaffold 쪽 연결 부위에 Isotope 마킹
        editable_core = Chem.RWMol(core)
        for atom_idx, r_id in anchor_map.items():
            if atom_idx < editable_core.GetNumAtoms():
                editable_core.GetAtomWithIdx(atom_idx).SetIsotope(r_id)

        scaffold_tagged = Chem.MolToSmiles(editable_core)

        # [NEW] 그래프 변환 로직 추가
        # 1) Scaffold Graph
        scaf_graph = self.smiles_to_graph(scaffold_tagged)

        # 2) Decoration Graphs (정렬된 순서대로)
        sorted_decs = sorted(decorations_data, key=lambda x: x["id"])
        side_graphs = [self.smiles_to_graph(item["smiles"]) for item in sorted_decs]

        # 결과 딕셔너리 반환
        return {
            "original_smiles": smiles,
            "original_mol": mol,
            "scaffold_smiles": scaffold_tagged,
            "scaffold_mol": editable_core,
            "decorations": sorted_decs,
            # [NEW] Dataset에서 요구하는 키 추가
            "scaf_graphs": scaf_graph,  # Single Data object
            "side_graphs": side_graphs,  # List of Data objects
        }

    def format_prompt(self, idx: int, smiles: str, label: str) -> str:
        """
        LLM 입력용 Few-shot 텍스트 포맷 생성
        """
        result = self.decompose(smiles)
        if result is None:
            return ""

        dec_strs = [f"`{item['smiles']}`" for item in result["decorations"]]
        decorations_str = ", ".join(dec_strs) if dec_strs else "None"
        scaffold_str = result["scaffold_smiles"] if result["scaffold_smiles"] else "None"

        prompt = (
            f"### Example {idx}\n"
            f"- Input: {result['original_smiles']}\n"
            f"[Scaffold]: `{scaffold_str}`\n"
            f"[Decorations]: [{decorations_str}]\n"
            f"- Output:\n"
            f"[Answer] {label}\n"
        )
        return prompt


# ==========================================
# 2. Soft Prompt Learner (GNN + MLP)
# ==========================================
class SoftPromptLearner(nn.Module):
    def __init__(self, gnn_model: nn.Module, input_dim=768, hidden_dim=256, output_dim=768):
        super().__init__()
        self.gnn = gnn_model
        # 예시: 임베딩을 변환하는 MLP
        self.mlp = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim))

    def freeze_gnn(self, freeze=True):
        """GNN 파라미터 고정 여부 설정"""
        for param in self.gnn.parameters():
            param.requires_grad = not freeze

    def forward(self, scaffold_smiles: str, decoration_smiles_list: list):
        """
        Scaffold와 Decoration SMILES를 받아 GNN -> MLP를 거쳐 Soft Prompt Tensor 반환
        """
        # 1. GNN Forward
        # (주의: 실제 GNN 구현체에 따라 입력 형태(Graph object 등)를 맞춰야 함. 여기선 리스트 가정)

        # Scaffold Embedding
        if scaffold_smiles:
            scaffold_emb = self.gnn([scaffold_smiles])  # shape: [1, dim]
        else:
            scaffold_emb = torch.zeros(1, self.mlp[0].in_features).to(next(self.parameters()).device)

        # Decorations Embedding
        if decoration_smiles_list:
            dec_embs = self.gnn(decoration_smiles_list)  # shape: [N, dim]
        else:
            dec_embs = torch.empty(0, self.mlp[0].in_features).to(next(self.parameters()).device)

        # 2. MLP Projector (학습 가능)
        projected_scaffold = self.mlp(scaffold_emb)
        projected_decs = self.mlp(dec_embs) if len(dec_embs) > 0 else None

        # 3. Return (사용처에 따라 Concat 방식 결정)
        return projected_scaffold, projected_decs


# ==========================================
# 3. Visualization Function (그리기 및 저장)
# ==========================================
def draw_mol_fixed_size(mol, mol_size=(300, 300), bond_length=40):
    """
    개별 분자를 그리는 헬퍼 함수.
    중요: fixedBondLength를 사용하여 분자 크기에 상관없이 '원자/고리의 크기'를 일정하게 유지함.
    """
    try:
        mol = Chem.Mol(mol)  # Copy to avoid modifying original safely
        AllChem.Compute2DCoords(mol)

        d2d = Draw.MolDraw2DCairo(mol_size[0], mol_size[1])
        dopts = d2d.drawOptions()
        dopts.fixedBondLength = bond_length  # 이 설정이 벤젠 고리 크기를 맞추는 핵심

        # 번호(Isotope)가 있으면 표시
        for atom in mol.GetAtoms():
            iso = atom.GetIsotope()
            if iso > 0:
                atom.SetProp("atomLabel", str(iso))  # Isotope를 라벨로 강제 변환하여 시각화

        d2d.DrawMolecule(mol)
        d2d.FinishDrawing()

        # Convert to PIL Image
        png_data = d2d.GetDrawingText()
        return Image.open(io.BytesIO(png_data))
    except Exception as e:
        print(f"Error drawing molecule: {e}")
        return Image.new("RGB", mol_size, (255, 255, 255))


def visualize_decomposition(decomposer_result, save_dir="output", prefix="mol"):
    """
    Decomposer의 결과를 받아 시각화 및 저장
    - 수정사항: 잔가지 개수에 따라 하단 컬럼 수를 동적으로 조절 (LCM 이용)
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    orig_mol = decomposer_result["original_mol"]
    scaf_mol = decomposer_result["scaffold_mol"]
    decs = decomposer_result["decorations"]

    # 1. 개별 이미지 생성 (메모리 상)
    IMG_SIZE = (300, 300)
    BOND_LEN = 45

    img_orig = draw_mol_fixed_size(orig_mol, IMG_SIZE, BOND_LEN)
    img_scaf = draw_mol_fixed_size(scaf_mol, IMG_SIZE, BOND_LEN)

    img_decs = []
    for item in decs:
        img = draw_mol_fixed_size(item["mol"], IMG_SIZE, BOND_LEN)
        img_decs.append(img)

    # 2. 개별 파일 저장
    img_orig.save(f"{save_dir}/{prefix}_original.png")
    img_scaf.save(f"{save_dir}/{prefix}_scaffold.png")
    for idx, img in enumerate(img_decs):
        img.save(f"{save_dir}/{prefix}_dec_{idx+1}.png")

    # 3. 통합 Plot 생성 (LCM 방식 적용 - 동적 컬럼)
    n_decs = len(img_decs)

    # 하단 컬럼(n_cols_dec) 결정
    if n_decs == 0:
        n_cols_dec = 1  # 에러 방지용
    elif n_decs < 5:
        n_cols_dec = n_decs  # 5개 미만이면 잔가지 개수만큼만 컬럼 생성 (꽉 차게 보임)
    else:
        n_cols_dec = 5  # 5개 이상이면 한 줄에 5개

    # GridSpec 경고 방지를 위한 LCM(최소공배수) 계산
    # 상단은 항상 2개(Original, Scaffold) vs 하단은 n_cols_dec개
    common_multiple = (2 * n_cols_dec) // math.gcd(2, n_cols_dec)
    total_cols = common_multiple

    # 각 이미지가 차지할 Span(칸 수) 계산
    top_span = total_cols // 2
    bot_span = total_cols // n_cols_dec

    # 전체 행 수 계산
    n_dec_rows = math.ceil(n_decs / n_cols_dec) if n_decs > 0 else 0
    total_rows = 1 + n_dec_rows

    # Figure 생성 (가로 사이즈도 컬럼 수에 맞춰 조절)
    fig_width = max(10, n_cols_dec * 3)
    fig = plt.figure(figsize=(fig_width, total_rows * 3.5))

    gs = fig.add_gridspec(nrows=total_rows, ncols=total_cols)

    # -- Row 1: Original | Scaffold --
    # 첫 번째 이미지 (0 ~ top_span)
    ax1 = fig.add_subplot(gs[0, 0:top_span])
    ax1.imshow(img_orig)
    ax1.set_title("Original Molecule")
    ax1.axis("off")

    # 두 번째 이미지 (top_span ~ 끝)
    ax2 = fig.add_subplot(gs[0, top_span : 2 * top_span])
    ax2.imshow(img_scaf)
    ax2.set_title("Scaffold (Tagged)")
    ax2.axis("off")

    # -- Row 2+: Decorations --
    for i in range(n_decs):
        # 행 인덱스 (1부터 시작)
        row_idx = 1 + (i // n_cols_dec)
        # 열 인덱스 (0 ~ n_cols_dec-1)
        col_idx_base = i % n_cols_dec

        # 실제 GridSpec 상에서의 시작/끝 위치
        col_start = col_idx_base * bot_span
        col_end = col_start + bot_span

        ax = fig.add_subplot(gs[row_idx, col_start:col_end])
        ax.imshow(img_decs[i])
        ax.set_title(f"Dec {decs[i]['id']}")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(f"{save_dir}/{prefix}_combined.png", dpi=150)
    plt.close()
    print(f"Saved visualization to {save_dir}/{prefix}_combined.png")


# ==========================================
# 실행 예시
# ==========================================
if __name__ == "__main__":
    # 1. 분해 클래스 사용
    decomposer = MoleculeDecomposer()

    # 예시 SMILES
    test_smiles = "Cc1c(O)cccc1"  # 벤젠 + 메틸 + 수산기

    # (1) Prompt Format 확인
    prompt = decomposer.format_prompt(1, test_smiles, "Yes")
    print(">>> Generated Prompt:")
    print(prompt)

    # (2) 분해 결과 구조체 획득
    result = decomposer.decompose(test_smiles)

    # 2. GNN & Soft Prompt 사용
    class MockGNN(nn.Module):
        def forward(self, smiles_list):
            # Mock Output
            return torch.randn(len(smiles_list), 768)

    learner = SoftPromptLearner(MockGNN())
    learner.freeze_gnn(True)  # GNN Freeze

    # 분해된 SMILES 목록 추출
    scaf_smi = result["scaffold_smiles"]
    dec_smis = [d["smiles"] for d in result["decorations"]]

    # 임베딩 생성
    h_scaf, h_decs = learner(scaf_smi, dec_smis)
    print(f">>> Embeddings shape: Scaf={h_scaf.shape}, Decs={h_decs.shape if h_decs is not None else 'None'}")

    # 3. 시각화 (이미지 저장)
    visualize_decomposition(result, save_dir="output_viz", prefix="example_1")
