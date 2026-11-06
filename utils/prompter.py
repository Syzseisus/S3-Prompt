from typing import Dict, List, Tuple, Union

from utils.constants import REGRESSION, CLASSIFICATION

STRICT_CLASSIFICATION = "The answer MUST be one for 'Yes' or 'No'."
STRICT_REGRESSION = (
    "The answer MUST be a single float number (e.g., -3.45). "
    "Output ONLY the numerical value immediately after the [Answer] tag."
)


class Prompter:
    """
    Generates dynamic In-Context Learning (ICL) prompts for different LLMs molecular property prediction.
    Reflects the 'Integrated Prompt Template' defined in the research paper (Figure #N).

    This class handles prompt generation for both
    Gemini (single string) and Llama 3 (messages list) formats,
    dynamically adjusting for classification or regression tasks.

    Attributes:
        dataset (str): Task dataset name (e.g., "Tox21").
        instruction (str): User-provided task instruction.
        strict_answer (str): The string snippet for the answer format.
    """

    def __init__(self, dataset: str, instruction: str):
        self.dataset = dataset
        self.instruction = instruction

        # Determine constraints based on dataset type
        if self.dataset in CLASSIFICATION:
            self.strict_answer = STRICT_CLASSIFICATION
        elif self.dataset in REGRESSION:
            self.strict_answer = STRICT_REGRESSION
        else:
            raise ValueError(f"The dataset `{dataset}` is not in classification nor regression sets.")

    def generate(self, query_formatted_str, shots=None, mode="gemini"):
        """
        Generate the prompt based on the mode.
        Especially for `gemini-2.5-flash-lite` and `meta-llama/Meta-Llama-3-8B-Instruct`.
        The final format format is as follows:

        <System Prompt>
            ## System

            You are an expert AI chemist specializing in Structure-Activity Relationship (SAR) analysis for the {self.dataset} task.
            You will be provided with a molecule's full structure as a **SMILES string**.
            Crucially, the molecule's core [Scaffold] and [Decorations] (side chains) are provided as **learned graph embeddings** (represented by special tokens), not as plain text.
            Your task is to:
            1. Interpret the full SMILES string.
            2. Integrate the deep structural features encoded in the [Scaffold] and [Decorations] embeddings.
            3. Analyze how these structural components influence the target property to predict the label.

            Follow the Task Instruction CAREFULLY.
            Your output MUST contain exactly two sections: [Reasoning] and [Answer].
            No additional text is allowed.

        <User Prompt>
            ## Task Instruction

            ### Domain Context & Objective

            {self.instruction}


            ### Analysis Strategy

            Based on the domain context above, analyze the targe molecule to predict the property.
            Follow these steps strictly:
            1. Global Structure: Interpret the 'Input' SMILES string to understand the overall molecular topology.
            2. Latent Feature Integration: Pay close attention to the `[Scaffold]` and `[Decorations]` fields.
            These are NOT empty text but contain **high-dimensional graph features** that encode key chemical properties.
            3. SAR Synthesis: Combine the biological context (from the Domain Context section) with the structural features (Text + Graph) to infer the label.
            4. Output Format:
                - [Answer] — {self.strict_answer}
                - [Reasoning] — Provide step-by-step reasoning. Keep it concise (ideally under 150 words).

            No additional text before and after these sections is allowed.


            ## Examples

            ### Example # 1

            - Input: {whole SMILES}
            [Scaffold]: {SCAF_TOKEN}
            [Decorations]: [{SIDE_TOKEN}, {SIDE_TOKEN}, ...]
            - Output:
            { [Reasoning] section omitted to avoid parroting }
            [Answer] {label}

            { Repeated for num_shot (K) }

            ## Query

            - Input: {whole SMILES}
            - Output:
            [Answer]
            # AUC 측정의 용이성, 업계 표준 -> CoT 효과는 유지됨


        Returns:
            - If mode=="gemini": Returns single, complete prompt string
            - If mode=="llama": Returns `messages` list formatted for `tokenizer.apply_chat_template`
        """

        # 1. System Prompt
        system_content = (
            "## System\n\n"
            f"You are an expert AI chemist specializing in Structure-Activity Relationship (SAR) analysis for the {self.dataset} task.\n"
            "You will be provided with a molecule's full structure as a **SMILES string**.\n"
            "Crucially, the molecule's core [Scaffold] and [Decorations] (side chains) are provided as **learned graph embeddings** (represented by special tokens), not as plain text.\n"
            "Your task is to:\n"
            "1. Interpret the full SMILES string.\n"
            "2. Integrate the deep structural features encoded in the [Scaffold] and [Decorations] embeddings.\n"
            "3. Analyze how these structural components influence the target property to predict the label.\n"
            "Follow the Task Instruction CAREFULLY.\n"
            "Your output MUST contain exactly two sections: [Answer] and [Reasoning].\n"
            "No additional text is allowed.\n"
        )

        # 2. User Prompt
        # 2-1. Task Instruction
        user_parts = []
        if self.instruction:
            user_parts.append(
                "## Task Instruction\n\n"
                "### Domain Context & Objective\n\n"
                f"{self.instruction}\n\n\n"
                "### Analysis Strategy\n\n"
                "Based on the domain context above, analyze the target molecule to predict the property.\n"  # typo fixed: targe -> target
                "Follow these steps strictly:\n"
                "1. Global Structure: Interpret the 'Input' SMILES string to understand the overall molecular topology.\n"
                "2. Latent Feature Integration: Pay close attention to the `[Scaffold]` and `[Decorations]` fields.\n"
                "These are NOT empty text but contain **high-dimensional graph features** that encode key chemical properties.\n"
                "3. SAR Synthesis: Combine the biological context (from the Domain Context section) with the structural features (Text + Graph) to infer the label.\n"
                "4. Output Format:\n"
                f"    - [Answer] — {self.strict_answer}\n"
                "    - [Reasoning] — Provide step-by-step reasoning. Keep it concise (ideally under 150 words).\n"
                "No additional text before and after these sections is allowed.\n"
            )

        # 2-2. Examples - omitted for zero-shot
        if shots:
            user_parts.append(shots.strip())

        # 2-3. Query
        user_parts.append(f"## Query\n\n{query_formatted_str.strip()}\n- Output:")

        user_content = "\n\n".join(user_parts)

        # 3. Return based on `mode`
        if mode == "debug":
            return (
                f"<System Prompt>\n{system_content}\n\n"
                f"<User Prompt>\n{user_content}\n\n"
                "<Model Output>\n[Answer] ..."
            )

        elif mode == "raw":
            return f"{system_content}\n\n{user_content}[Answer]"

        elif mode == "gemini":
            # Gemini API는 system_instruction을 별도 인자로 받음
            # Returns: (system_instruction, user_content)
            return system_content.strip(), user_content.strip() + "\n[Answer]"

        elif mode == "llama":
            # chat template 형식
            # ALERT: chat template 적용 후 [Answer]을 수동으로 더해줘야 Pre-fill로 제대로 작동함
            return [
                {"role": "system", "content": system_content.strip()},
                {"role": "user", "content": user_content.strip()},
            ]

        else:
            raise ValueError(f"Unsupported mode: {mode}")
