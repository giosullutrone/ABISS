from typing import cast
import logging

from tqdm import tqdm
from .model import Model
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from vllm.sampling_params import SamplingParams, StructuredOutputsParams
from models import extract_last_json_object
from pydantic import BaseModel
from utils.prompt_utils import model_field_descriptions
import torch

logger = logging.getLogger(__name__)


class ModelVLLM(Model):
    # HF generation_config keys that map to vLLM SamplingParams keys
    _HF_TO_VLLM_KEYS = {
        "temperature": "temperature",
        "top_p": "top_p",
        "top_k": "top_k",
        "max_new_tokens": "max_tokens",
        "repetition_penalty": "repetition_penalty",
    }

    @staticmethod
    def get_default_sampling_kwargs(model_name: str) -> dict:
        """Load the model's recommended sampling parameters from its generation_config.json.

        Returns a dict with vLLM-compatible keys (temperature, top_p, top_k, max_tokens,
        repetition_penalty) for any values defined in the model's HuggingFace generation config.
        Returns an empty dict if the config cannot be loaded.
        """
        from transformers import GenerationConfig
        try:
            gen_config = GenerationConfig.from_pretrained(model_name)
            defaults = {}
            for hf_key, vllm_key in ModelVLLM._HF_TO_VLLM_KEYS.items():
                value = getattr(gen_config, hf_key, None)
                if value is not None:
                    defaults[vllm_key] = value
            return defaults
        except Exception as e:
            logger.info(f"No generation_config found for {model_name}: {e}")
            return {}

    def __init__(self,
                 model_name: str,
                 lora_name: str | None = None,
                 system_prompt: str | None = None,
                 sampling_kwargs: dict | None = None,
                 model_kwargs: dict | None = None,
                 max_batch_with_text_size: int = 10000):
        super().__init__(model_name,
                         lora_name,
                         system_prompt,
                         sampling_kwargs,
                         model_kwargs,
                         max_batch_with_text_size)

    def init(self):
        # Prepare LoRA request if lora_name is provided
        self.lora_request: LoRARequest | None = LoRARequest(lora_name=self.lora_name, 
                                                            lora_int_id=0, 
                                                            lora_path=self.lora_name) if self.lora_name is not None else None

        # Initialize the vLLM model
        self.model: LLM = LLM(model=self.model_name, 
                              trust_remote_code=True, 
                              enable_lora=(self.lora_name is not None),
                              **(self.model_kwargs if self.model_kwargs is not None else {}))

        # Prepare sampling parameters
        self.sampling_params = SamplingParams(**self.sampling_kwargs) if self.sampling_kwargs is not None else None
    
    def get_token_lengths(self, prompts: list[str] | list[list[dict[str, str]]]) -> list[int]:
        prompts_tokens = self.model.preprocess_chat(self.convert_prompt_to_conversation_if_needed(prompts)) # type: ignore        
        return [len(pt.get("prompt_token_ids")) for pt in prompts_tokens]
    
    def _generate_batch(self, prompts: list[list[dict[str, str]]], sampling_params: SamplingParams | None, continue_final_message: bool=False, use_tqdm: bool=True, disable_thinking: bool=False) -> list[str]:
        responses: list[str] = []
        for i in range(0, len(prompts), self.max_batch_with_text_size):
            batch_conversations: list[list[dict[str, str]]] = prompts[i:i + self.max_batch_with_text_size]
            batch_responses = self.model.chat(
                messages=batch_conversations, # type: ignore
                sampling_params=sampling_params,
                lora_request=self.lora_request,
                continue_final_message=continue_final_message,
                add_generation_prompt=not continue_final_message,
                use_tqdm=use_tqdm,
                chat_template_kwargs={"enable_thinking": False} if disable_thinking else None
            )
            for batch_response in batch_responses:
                responses.extend([x.text for x in batch_response.outputs])
        return responses

    def generate_batch(self, prompts: list[str] | list[list[dict[str, str]]]) -> list[str]:
        converted_prompts = self.convert_prompt_to_conversation_if_needed(prompts)
        responses = self._generate_batch(converted_prompts, self.sampling_params)
        return responses

    def generate_batch_with_constraints(self, prompts: list[str] | list[list[dict[str, str]]], constraints: list[type[BaseModel]]) -> list[BaseModel]:
        converted_prompts = self.convert_prompt_to_conversation_if_needed(prompts)
        responses = self._generate_batch_with_constraints(converted_prompts, constraints, raise_exceptions=True)
        # We can cast here since all responses are validated and exceptions are raised otherwise
        return cast(list[BaseModel], responses)

    def generate_batch_with_constraints_unsafe(self, prompts: list[str] | list[list[dict[str, str]]], constraints: list[type[BaseModel]]) -> list[BaseModel | None]:
        converted_prompts = self.convert_prompt_to_conversation_if_needed(prompts)
        validated_responses = self._generate_batch_with_constraints(converted_prompts, constraints, raise_exceptions=False)
        return validated_responses
    
    def _generate_batch_with_constraints(self, prompts: list[list[dict[str, str]]], constraints: list[type[BaseModel]], raise_exceptions: bool) -> list[BaseModel | None]:
        """
        Generate responses for a batch of prompts while enforcing constraints defined by the input Pydantic models.
        Returns n * len(prompts) validated responses (all n responses for each prompt).
        """
        assert len(prompts) == len(constraints), "Number of prompts must match number of constraints."

        # Get the 'n' parameter from sampling_kwargs to know how many responses per prompt
        n = (self.sampling_kwargs or {}).get('n', 1)

        # Generate raw responses (will get n * len(prompts) responses)
        responses = self.generate_batch(prompts)
        assert len(responses) == len(prompts) * n, f"Expected {len(prompts) * n} responses, got {len(responses)}"

        # Validate each response against its corresponding constraint
        # Each response at index i corresponds to constraint at index i // n
        validated_responses: list[BaseModel | None] = []
        for i, response in enumerate(responses):
            constraint_idx = i // n
            validated_response = extract_last_json_object(response, constraints[constraint_idx])
            validated_responses.append(validated_response)

        if all(v is not None for v in validated_responses):
            return validated_responses

        # Single retry: add user message with schema, prime with ```json, continue without thinking
        if not all(v is not None for v in validated_responses):
            logger.info("Retrying invalid responses...")

            conversations_to_regenerate: dict[int, tuple[list[dict[str, str]], type[BaseModel]]] = {}
            for idx, validated_response in enumerate(validated_responses):
                if validated_response is not None:
                    continue

                prompt_idx = idx // n
                constraint = constraints[prompt_idx]
                logger.info(f"Response at index {idx} (prompt {prompt_idx}, variant {idx % n}) was invalid. Regenerating...")

                conversation_to_regenerate = prompts[prompt_idx].copy()
                response_text = responses[idx]

                # Strip thinking tags to avoid chat template incompatibility
                # (Qwen3's template transforms <think> blocks, breaking continue_final_message)
                response_text = response_text.replace("<think>", "").replace("</think>", "")
                conversation_to_regenerate.append({"role": "assistant", "content": response_text})

                # Add user message requesting JSON with schema
                additional_prompt = (
                    "Complete this JSON response properly. Don't produce any text outside the JSON object.\n"
                    f"The complete JSON object must follow this schema:\n{model_field_descriptions(constraint)}"
                )
                conversation_to_regenerate.append({"role": "user", "content": additional_prompt})

                # Prime the response with ```json so the model continues directly with JSON
                conversation_to_regenerate.append({"role": "assistant", "content": "```json\n"})

                conversations_to_regenerate[idx] = (conversation_to_regenerate, constraint)

            # Group conversations by constraint type for batched regeneration
            constraint_groups: dict[type[BaseModel], list[tuple[int, list[dict[str, str]]]]] = {}
            for idx, (conversation, constraint) in conversations_to_regenerate.items():
                if constraint not in constraint_groups:
                    constraint_groups[constraint] = []
                constraint_groups[constraint].append((idx, conversation))

            for constraint, idx_conversation_pairs in tqdm(constraint_groups.items(), desc="Regenerating invalid responses"):
                indices = [idx for idx, _ in idx_conversation_pairs]
                conversations = [conversation for _, conversation in idx_conversation_pairs]

                base_max_tokens = self.sampling_kwargs.get("max_tokens", 2048) if self.sampling_kwargs else 2048
                retry_sampling_params = SamplingParams(
                    **{**(self.sampling_kwargs if self.sampling_kwargs is not None else {}),
                       **{"n": 1, "max_tokens": max(2048, base_max_tokens + 512)}},
                    structured_outputs=StructuredOutputsParams(
                        json=constraint.model_json_schema(),
                        disable_any_whitespace=True
                    )
                )

                regenerated_responses = self._generate_batch(
                    conversations, retry_sampling_params,
                    continue_final_message=True, use_tqdm=False, disable_thinking=True
                )

                for idx, regenerated_response in zip(indices, regenerated_responses):
                    validated_response = extract_last_json_object(regenerated_response, constraint)
                    if validated_response is None:
                        logger.info(f"Retry for index {idx} produced invalid response: '{regenerated_response}'")
                    validated_responses[idx] = validated_response

        # Check if all responses are validated after retry
        if raise_exceptions and not all(v is not None for v in validated_responses):
            failed_indices = [idx for idx, v in enumerate(validated_responses) if v is None]
            raise ValueError(f"Failed to validate responses at indices {failed_indices} after retry. Responses: {[responses[idx] for idx in failed_indices]}")
        
        # At this point, all responses are validated (or raise_exceptions is False)
        return validated_responses

    def close(self):
        del self.model
        del self.lora_request
        del self.sampling_params
        import gc
        gc.collect()
        torch.cuda.empty_cache()