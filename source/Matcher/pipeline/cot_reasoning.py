import json
import os
from typing import Dict, List

import torch
from Matcher.utils.file_utils import read_json_file, write_json_file, write_text_file
from Matcher.utils.logging_config import setup_logging
from tqdm import tqdm

logger = setup_logging()


class BatchTrialProcessor:
    def __init__(
        self, model, tokenizer, device: int, batch_size: int = 4, use_cot: bool = True
    ):
        self.device = device
        self.batch_size = batch_size
        self.model = model
        self.tokenizer = tokenizer
        self.use_cot = use_cot

        self.model.eval()
        try:
                # Allow TF32 on Ampere+ (gives a free speedup for matmuls with minimal accuracy loss)
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    def _load_trial_data(self, nct_id: str, json_folder: str) -> str:
        try:
            path = f"{json_folder}/{nct_id}.json"
            trial_data = read_json_file(path)
            return trial_data.get("eligibility_criteria", "")
        except Exception as e:
            logger.error(f"Error loading {nct_id}: {str(e)}")
            return ""

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
            # Simple, concise, non-CoT prompt
            chat = [
                {
                    "role": "system",
                    "content": (
                        "You are a clinical assistant tasked with assessing the eligibility of a patient for a clinical trial. Output only a JSON object evaluating trial eligibility for the patient based only on the provided criteria and patient profile. \n\n"
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

        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=True
            )
        system_part = f"{chat[0]['content']}\n\n"
        user_part = f"{chat[1]['content']}\n\n"
        return system_part + user_part + "Answer: "
    
    def _individual_process(self, nct_id: str, prompt: str, output_folder: str):
        """ Process a single trial individually in case of batch failure due to a repetition pitfall of the model."""
        try:
            inputs = self.tokenizer(
                [prompt],
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(f"cuda:{self.device}")
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=5000,
                    do_sample=False,
                    return_dict_in_generate=False,
                    use_cache=True
                    )
            decoded_response = self.tokenizer.batch_decode(
                outputs[:, inputs.input_ids.shape[1] :], skip_special_tokens=True
            )[0]
            self._save_outputs(nct_id, decoded_response, output_folder)
        except Exception as e:
            logger.error(f"Failed individual processing for {nct_id}: {str(e)}")

    def _process_batch(self, batch: List[Dict], output_folder: str):
        try:
            inputs = self.tokenizer(
                [item["prompt"] for item in batch],
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(f"cuda:{self.device}")

            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=5000,
                    do_sample=False,
                    return_dict_in_generate=False,
                    use_cache=True
                    )
            decoded_responses = self.tokenizer.batch_decode(
                outputs[:, inputs.input_ids.shape[1] :], skip_special_tokens=True
            )

            logger.warning(f"Decoded response: {decoded_responses}")
            for item, response in zip(batch, decoded_responses):
                try:
                    self._save_outputs(item["nct_id"], response, output_folder)
                except Exception as e:
                    logger.error(f"Failed to process {item['nct_id']}: {str(e)}. Regenerating individually.")
                    self._individual_process(nct_id=item["nct_id"], prompt=item["prompt"], output_folder=output_folder)
        except Exception as e:
            logger.error(f"Batch processing failed: {str(e)}")
            for item in batch:
                logger.error(f"Failed trial: {item['nct_id']}")

    def _save_outputs(self, nct_id: str, response: str, output_folder: str):
        try:
            txt_path = f"{output_folder}/{nct_id}.txt"
            write_text_file([response], txt_path)
            try:
                json_str = response[response.find("{") : response.rfind("}") + 1]
                json_data = json.loads(json_str)
                write_json_file(json_data, f"{output_folder}/{nct_id}.json")
                logger.info(f"Processed {nct_id} successfully")
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON response for {nct_id}: {str(e)}")
        except Exception as e:
            logger.error(f"Failed to save {nct_id}: {str(e)}")

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
        current_batch = []
        for nct_id in tqdm(nct_ids, desc=f"GPU {self.device} Processing Trials"):
            output_path = f"{output_folder}/{nct_id}.json"
            if os.path.exists(output_path):
                logger.info(f"Skipping existing: {nct_id}")
                continue
            criteria_text = self._load_trial_data(nct_id, json_folder)
            prompt = self._format_prompt(criteria_text, patient_text)
            current_batch.append({"nct_id": nct_id, "prompt": prompt})
            if len(current_batch) >= self.batch_size:
                self._process_batch(current_batch, output_folder)
                current_batch = []
        if current_batch:
            self._process_batch(current_batch, output_folder)
