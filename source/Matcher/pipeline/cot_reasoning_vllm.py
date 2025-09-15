# Matcher/pipeline/cot_reasoning_vllm.py
from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Optional

from Matcher.utils.file_utils import read_json_file, write_json_file, write_text_file
from Matcher.utils.logging_config import setup_logging
from tqdm import tqdm
from vllm import LLM, SamplingParams

try:
    # Present in vLLM when LoRA is enabled
    from vllm.lora.request import LoRARequest  # type: ignore
except Exception:  # pragma: no cover
    LoRARequest = None  # type: ignore

logger = setup_logging()


class BatchTrialProcessorVLLM:
    def __init__(
        self,
        llm: LLM,
        tokenizer=None,
        batch_size: int = 16,
        use_cot: bool = True,
        max_new_tokens: int = 5000,
        temperature: float = 0.0,
        top_p: float = 1.0,
        seed: Optional[int] = 1234,
        length_bucket: bool = True,
        lora_request: Optional[LoRARequest] = None,  # type: ignore
    ):
        """
        vLLM-backed trial processor for CoT eligibility evaluation.
        - Keeps long outputs (no custom stop).
        - Uses chat templates if tokenizer supports them.
        - Supports optional LoRA adapter via vLLM's LoRARequest.
        """
        self.llm = llm
        self.tokenizer = tokenizer or getattr(self.llm, "get_tokenizer", lambda: None)()
        self.batch_size = batch_size
        self.use_cot = use_cot
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.seed = seed
        self.length_bucket = length_bucket
        self.lora_request = lora_request  # NEW

        self.sampling_params = SamplingParams(
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            seed=self.seed,
            detokenize=True,
        )

    # ---------------------- I/O helpers ----------------------

    def _load_trial_data(self, nct_id: str, json_folder: str) -> str:
        try:
            path = f"{json_folder}/{nct_id}.json"
            trial_data = read_json_file(path)
            return trial_data.get("eligibility_criteria", "")
        except Exception as e:
            logger.error(f"Error loading {nct_id}: {str(e)}")
            return ""

    # ---------------------- Prompting ----------------------

    def _format_prompt(self, criteria_text: str, patient_profile: str) -> str:
        criteria_text_formatted = (
            f"Eligibility Criteria:\n{criteria_text}"
            if criteria_text
            else "No eligibility criteria provided."
        )

        if self.use_cot:
            system_msg = (
                "You are a medical expert with advanced knowledge in clinical reasoning, diagnostics, and treatment planning. "
                "Answer the following question. Before answering, create a concise chain of thoughts reasoning to ensure a logical and accurate response.\n"
            )
            chat = [
                {"role": "system", "content": system_msg},
                {
                    "role": "user",
                    "content": (
                        "Assess the given patient's eligibility for a clinical trial by evaluating each and every criterion individually.\n\n"
                        "### INCLUSION CRITERIA ASSESSMENT\n"
                        "For each inclusion criterion, classify it as one of:\n"
                        "- **Met:** The patient's data explicitly and unequivocally satisfies the criterion.\n"
                        "- **Not Met:** The patient's data explicitly and unequivocally contradicts or fails to satisfy the criterion.\n"
                        "- **Unclear:** Insufficient or missing patient data to verify.\n"
                        "- **Irrelevant:** The criterion does not apply to the patient's context.\n\n"
                        "### EXCLUSION CRITERIA ASSESSMENT\n"
                        "For each exclusion criterion, classify it as one of:\n"
                        "- **Violated:** The patient's data explicitly and unequivocally violates the criterion.\n"
                        "- **Not Violated:** The patient's data confirms compliance with the criterion.\n"
                        "- **Unclear:** Insufficient or missing patient data to verify.\n"
                        "- **Irrelevant:** The criterion does not apply to the patient's context.\n\n"
                        "### IMPORTANT INSTRUCTIONS\n"
                        "- Ensure all criteria are assessed one-by-one.\n"
                        "- Use **only** the provided patient data; **do not infer, assume, or extrapolate beyond the given information.**\n"
                        "- Justifications must be strictly based on direct evidence from the patient profile.\n"
                        "### RESPONSE FORMAT (STRICTLY FOLLOW)\n"
                        "{\n"
                        '  "Inclusion_Criteria_Evaluation": [\n'
                        '    {"Criterion": "Exact inclusion criterion text", "Classification": "Met | Not Met | Unclear | Irrelevant", "Justification": "Clear, evidence-based rationale using ONLY provided data"}\n'
                        "  ],\n"
                        '  "Exclusion_Criteria_Evaluation": [\n'
                        '    {"Criterion": "Exact exclusion criterion text", "Classification": "Violated | Not Violated | Unclear | Irrelevant", "Justification": "Clear, evidence-based rationale using ONLY provided data"}\n'
                        "  ],\n"
                        '  "Recap": "Concise summary of key qualifying/disqualifying factors",\n'
                        '  "Final Decision": "Eligible | Likely Eligible (leaning toward inclusion) | Likely Ineligible (leaning toward exclusion) | Ineligible"\n'
                        "}\n\n"
                        "### INPUT\n"
                        "---Start of Clinical Trial Criteria---\n"
                        f"{criteria_text_formatted}\n"
                        "---End of Clinical Trial Criteria---\n\n"
                        "----\n"
                        "---Start of Patient Description---\n"
                        f"{patient_profile}\n"
                        "Written informed consent has been obtained from the patient or their legal representative.\n"
                        "---End of Patient Description---\n"
                        "## IMPORTANT REMINDER:\n"
                        "NEVER make assumptions, inferences, or extrapolations beyond the explicitly stated patient information."
                    ),
                },
            ]
        else:
            chat = [
                {
                    "role": "system",
                    "content": (
                        "You are a clinical assistant tasked with assessing the eligibility of a patient for a clinical trial. "
                        "Output only a JSON object evaluating trial eligibility for the patient based only on the provided criteria and patient profile.\n"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "For each criterion, classify:\n"
                        '- If Inclusion Criterion: "Met" | "Not Met" | "Unclear" | "Irrelevant"\n'
                        '- If Exclusion Criterion: "Violated" | "Not Violated" | "Unclear" | "Irrelevant"\n\n'
                        "Provide a justification for each classification based strictly on the provided data. "
                        "Output this JSON schema:\n"
                        "{\n"
                        '  "Inclusion_Criteria_Evaluation": [ {"Criterion": "...", "Classification": "...", "Justification": "..."} ],\n'
                        '  "Exclusion_Criteria_Evaluation": [ {"Criterion": "...", "Classification": "...", "Justification": "..."} ],\n'
                        '  "Final Decision": "Eligible | Likely Eligible | Likely Ineligible | Ineligible"\n'
                        "}\n\n"
                        "---Start of Clinical Trial Criteria---\n"
                        f"{criteria_text_formatted}\n"
                        "---End of Clinical Trial Criteria---\n\n"
                        "---Start of Patient Description---\n"
                        f"{patient_profile}\n"
                        "---End of Patient Description---\n"
                    ),
                },
            ]

        if self.tokenizer is not None and hasattr(
            self.tokenizer, "apply_chat_template"
        ):
            return self.tokenizer.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=True
            )

        system_part = f"{chat[0]['content']}\n\n"
        user_part = f"{chat[1]['content']}\n\n"
        return system_part + user_part + "Answer: "

    # ---------------------- Core batch path (vLLM) ----------------------

    def _process_batch(self, batch: List[Dict], output_folder: str):
        try:
            prompts = [item["prompt"] for item in batch]
            t0 = time.time()
            # Pass LoRARequest if provided (activates adapter)
            results = self.llm.generate(
                prompts,
                self.sampling_params,
                lora_request=self.lora_request,  # NEW
            )
            t1 = time.time()

            decoded_responses: List[str] = []
            in_tok_lens: List[int] = []
            out_tok_lens: List[int] = []

            for r in results:
                text = r.outputs[0].text if r.outputs else ""
                decoded_responses.append(text)
                try:
                    in_tok_lens.append(len(getattr(r, "prompt_token_ids", []) or []))
                except Exception:
                    in_tok_lens.append(0)
                try:
                    out_tok_lens.append(
                        len(getattr(r.outputs[0], "token_ids", []) or [])
                    )
                except Exception:
                    out_tok_lens.append(0)

            for item, response in zip(batch, decoded_responses):
                self._save_outputs(item["nct_id"], response, output_folder)

            total_in = sum(in_tok_lens)
            total_out = sum(out_tok_lens)
            gen_time = max(1e-6, t1 - t0)
            logger.info(
                f"[vLLM] batch={len(batch)} | in_tok≈{total_in} | out_tok≈{total_out} | "
                f"elapsed={gen_time:.2f}s | ~{(total_out / gen_time):.1f} tok/s"
            )

        except Exception as e:
            logger.error(f"Batch processing failed: {str(e)}")
            for item in batch:
                logger.error(f"Failed trial: {item['nct_id']}")

    # ---------------------- Persistence ----------------------

    def _save_outputs(self, nct_id: str, response: str, output_folder: str):
        try:
            os.makedirs(output_folder, exist_ok=True)
            txt_path = f"{output_folder}/{nct_id}.txt"
            write_text_file([response], txt_path)
            try:
                start = response.find("{")
                end = response.rfind("}")
                if start != -1 and end != -1 and end > start:
                    json_str = response[start : end + 1]
                    json_data = json.loads(json_str)
                    write_json_file(json_data, f"{output_folder}/{nct_id}.json")
                    logger.info(f"Processed {nct_id} successfully")
                else:
                    logger.error(f"Invalid JSON boundaries for {nct_id}")
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON response for {nct_id}: {str(e)}")
        except Exception as e:
            logger.error(f"Failed to save {nct_id}: {str(e)}")

    # ---------------------- Public API ----------------------

    def process_trials(
        self,
        nct_ids: List[str],
        json_folder: str,
        output_folder: str,
        patient_profile: List[str],
    ):
        patient_text = " ".join(
            str(line).strip() for line in patient_profile if str(line).strip()
        )

        items: List[Dict] = []
        for nct_id in nct_ids:
            output_path = f"{output_folder}/{nct_id}.json"
            if os.path.exists(output_path):
                logger.info(f"Skipping existing: {nct_id}")
                continue

            criteria_text = self._load_trial_data(nct_id, json_folder)
            prompt = self._format_prompt(criteria_text, patient_text)

            if self.length_bucket and self.tokenizer is not None:
                try:
                    tok_len = len(
                        self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
                    )
                except Exception:
                    tok_len = max(1, len(prompt) // 4)
            else:
                tok_len = max(1, len(prompt) // 4)

            items.append({"nct_id": nct_id, "prompt": prompt, "tok_len": tok_len})

        if not items:
            logger.info("No work to do.")
            return

        if self.length_bucket:
            items.sort(key=lambda x: x["tok_len"])

        for i in tqdm(
            range(0, len(items), self.batch_size), desc="vLLM Processing Trials"
        ):
            batch = items[i : i + self.batch_size]
            self._process_batch(batch, output_folder)
