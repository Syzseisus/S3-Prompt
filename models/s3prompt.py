import torch
import torch.nn as nn

from models.gnn import SharedGNN


class S3PromptModel(nn.Module):
    def __init__(self, llm_model, tokenizer, terminators, gnn_hidden=300):
        super().__init__()

        # 1. Dependency Injection (LLM, Tokenizer)
        self.llm = llm_model
        self.tokenizer = tokenizer
        self.terminators = terminators

        # Freeze LLM
        for p in self.llm.parameters():
            p.requires_grad = False

        llm_dim = self.llm.config.hidden_size

        # 2. Shared GNN (with OGB Encoder)
        self.gnn = SharedGNN(hidden_dim=gnn_hidden)

        # 3. Projectors
        self.scaf_proj = nn.Sequential(
            nn.Linear(gnn_hidden, gnn_hidden * 2), nn.ReLU(), nn.Linear(gnn_hidden * 2, llm_dim)
        )
        self.side_proj = nn.Sequential(
            nn.Linear(gnn_hidden, gnn_hidden * 2), nn.ReLU(), nn.Linear(gnn_hidden * 2, llm_dim)
        )

    def forward(
        self,
        input_ids,
        scaffold_batch,
        sidechain_batch,
        scaf_mask,
        side_mask,
        labels=None,
        attention_mask=None,
        output_hidden_states=False,
    ):

        # Text Embeddings
        inputs_embeds = self.llm.get_input_embeddings()(input_ids).clone()

        # Graph Embeddings
        # Check if batches are not empty
        if scaffold_batch.x.size(0) > 0:
            scaf_feats = self.gnn(scaffold_batch.x, scaffold_batch.edge_index, scaffold_batch.batch)
            scaf_soft_tokens = self.scaf_proj(scaf_feats)
            # Replace
            if scaf_mask.any():
                # Ensure device match and type match
                inputs_embeds[scaf_mask] = scaf_soft_tokens.to(inputs_embeds.dtype)

        if sidechain_batch.x.size(0) > 0:
            side_feats = self.gnn(sidechain_batch.x, sidechain_batch.edge_index, sidechain_batch.batch)
            side_soft_tokens = self.side_proj(side_feats)
            # Replace
            if side_mask.any():
                inputs_embeds[side_mask] = side_soft_tokens.to(inputs_embeds.dtype)

        # Forward
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            labels=labels,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
        )
        return outputs

    def generate(self, input_ids, scaffold_batch, sidechain_batch, scaf_mask, side_mask, attention_mask=None, **kwargs):
        inputs_embeds = self.llm.get_input_embeddings()(input_ids).clone()

        with torch.no_grad():
            if scaffold_batch.x.size(0) > 0:
                scaf_feats = self.gnn(scaffold_batch.x, scaffold_batch.edge_index, scaffold_batch.batch)
                scaf_soft_tokens = self.scaf_proj(scaf_feats)
                if scaf_mask.any():
                    inputs_embeds[scaf_mask] = scaf_soft_tokens.to(inputs_embeds.dtype)

            if sidechain_batch.x.size(0) > 0:
                side_feats = self.gnn(sidechain_batch.x, sidechain_batch.edge_index, sidechain_batch.batch)
                side_soft_tokens = self.side_proj(side_feats)
                if side_mask.any():
                    inputs_embeds[side_mask] = side_soft_tokens.to(inputs_embeds.dtype)

        return self.llm.generate(inputs_embeds=inputs_embeds, attention_mask=attention_mask, **kwargs)
