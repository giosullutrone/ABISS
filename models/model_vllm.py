from typing import cast
import logging

from tqdm import tqdm
from .model import Model
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from vllm.sampling_params import SamplingParams, StructuredOutputsParams
from models import extract_last_json_object
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class ModelVLLM(Model):
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
    
    def _generate_batch(self, prompts: list[list[dict[str, str]]], sampling_params: SamplingParams | None, continue_final_message: bool=False, use_tqdm: bool=True) -> list[str]:
        responses: list[str] = []
        for i in range(0, len(prompts), self.max_batch_with_text_size):
            batch_conversations: list[list[dict[str, str]]] = prompts[i:i + self.max_batch_with_text_size]
            batch_responses = self.model.chat(
                messages=batch_conversations, # type: ignore
                sampling_params=sampling_params,
                lora_request=self.lora_request,
                continue_final_message=continue_final_message,
                add_generation_prompt=not continue_final_message,
                use_tqdm=use_tqdm
            )
            for batch_response in batch_responses:
                responses.extend([x.text for x in batch_response.outputs])
        return responses

    def generate_batch(self, prompts: list[str] | list[list[dict[str, str]]]) -> list[str]:
        return self._generate_batch(self.convert_prompt_to_conversation_if_needed(prompts), self.sampling_params)

    def _generate_batch_with_constraints(self, prompts: list[list[dict[str, str]]], constraints: list[type[BaseModel]]) -> list[BaseModel]:
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
            return cast(list[BaseModel], validated_responses)

        # If there are any None values, Re-Generate those specific responses with StructuredOutputsParams
        conversations_to_regenerate: dict[int, tuple[list[dict[str, str]], type[BaseModel]]] = {}
        for idx, validated_response in enumerate(validated_responses):
            if validated_response is not None:
                continue
            
            prompt_idx = idx // n
            constraint = constraints[prompt_idx]
            logger.info(f"Response at index {idx} (prompt {prompt_idx}, variant {idx % n}) was invalid: '{responses[idx]}...'. Regenerating...")

            # We copy the original conversation prompt since it is a list which is mutable
            conversation_to_regenerate = prompts[prompt_idx].copy()
            # We have to add the assistance message with the previous incomplete response
            conversation_to_regenerate.append({"role": "assistant", "content": responses[idx] + "\n"})
            conversations_to_regenerate[idx] = (conversation_to_regenerate, constraint)

        # A batch can't have multiple different StructuredOutputsParams, so we regenerate one by one
        for idx, (conversation, constraint) in tqdm(conversations_to_regenerate.items(), desc="Regenerating invalid responses"):
            structured_sampling_params = SamplingParams(
                **(self.sampling_kwargs if self.sampling_kwargs is not None else {}),
                n=1,  # Only generate 1 response when regenerating with structured outputs
                structured_outputs=StructuredOutputsParams(
                    json=constraint.model_json_schema()
                ),
            )
            regenerated_response = self._generate_batch([conversation], structured_sampling_params, continue_final_message=True, use_tqdm=False)[0]
            validated_response = extract_last_json_object(regenerated_response, constraint)
            if validated_response is None:
                raise ValueError(f"Failed to validate regenerated response at index {idx}: {regenerated_response}.")
            validated_responses[idx] = validated_response

        # At this point, all responses are validated
        return cast(list[BaseModel], validated_responses)

    def generate_batch_with_constraints(self, prompts: list[str] | list[list[dict[str, str]]], constraints: list[type[BaseModel]]) -> list[BaseModel]:
        return self._generate_batch_with_constraints(self.convert_prompt_to_conversation_if_needed(prompts), constraints)

    def close(self):
        del self.model
        del self.lora_request
        del self.sampling_params
