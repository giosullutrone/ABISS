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
from models import reorder_by_prefix_similarity, restore_original_order

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
        converted_prompts = self.convert_prompt_to_conversation_if_needed(prompts)
        converted_prompts_reordered, original_indices, _ = reorder_by_prefix_similarity(converted_prompts)
        responses_reordered = self._generate_batch(converted_prompts_reordered, self.sampling_params)
        return restore_original_order(responses_reordered, original_indices)

    def generate_batch_with_constraints(self, prompts: list[str] | list[list[dict[str, str]]], constraints: list[type[BaseModel]]) -> list[BaseModel]:
        converted_prompts = self.convert_prompt_to_conversation_if_needed(prompts)
        converted_prompts_reordered, original_indices, _constraints_reordered = reorder_by_prefix_similarity(converted_prompts, constraints)
        constraints_reordered = _constraints_reordered[0]
        validated_responses_reordered = self._generate_batch_with_constraints(converted_prompts_reordered, constraints_reordered, raise_exceptions=True)
        return restore_original_order(validated_responses_reordered, original_indices)

    def generate_batch_with_constraints_unsafe(self, prompts: list[str] | list[list[dict[str, str]]], constraints: list[type[BaseModel]]) -> list[BaseModel | None]:
        converted_prompts = self.convert_prompt_to_conversation_if_needed(prompts)
        converted_prompts_reordered, original_indices, _constraints_reordered = reorder_by_prefix_similarity(converted_prompts, constraints)
        constraints_reordered = _constraints_reordered[0]
        validated_responses_reordered = self._generate_batch_with_constraints(converted_prompts_reordered, constraints_reordered, raise_exceptions=False)
        return restore_original_order(validated_responses_reordered, original_indices)

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
            
            # Remove <think> tags if present
            response_text = responses[idx]
            
            response_text = response_text.replace("<think>", "")
            response_text = response_text.replace("</think>", "")
            
            # Add continuation prompt to guide the model
            additional_prompt = (
                "\n\nLet me complete this JSON response properly. "
                f"The complete JSON object must follow this schema:\n{model_field_descriptions(constraint)}\n\n"
                "Here is the complete, valid JSON:\n"
            )
            
            # We have to add the assistance message with the previous incomplete response
            conversation_to_regenerate.append({"role": "assistant", "content": response_text + additional_prompt})
            conversations_to_regenerate[idx] = (conversation_to_regenerate, constraint)

        # Group conversations by constraint type for batched regeneration
        constraint_groups: dict[type[BaseModel], list[tuple[int, list[dict[str, str]]]]] = {}
        for idx, (conversation, constraint) in conversations_to_regenerate.items():
            if constraint not in constraint_groups:
                constraint_groups[constraint] = []
            constraint_groups[constraint].append((idx, conversation))

        # Process each constraint group in batch
        for constraint, idx_conversation_pairs in tqdm(constraint_groups.items(), desc="Regenerating invalid responses by constraint"):
            indices = [idx for idx, _ in idx_conversation_pairs]
            conversations = [conversation for _, conversation in idx_conversation_pairs]
            
            # Create new SamplingParams with StructuredOutputsParams for the constraint
            structured_sampling_params = SamplingParams(
                **{**(self.sampling_kwargs if self.sampling_kwargs is not None else {}), **{"n": 1}},
                structured_outputs=StructuredOutputsParams(
                    json=constraint.model_json_schema(),
                    disable_any_whitespace=True
                ),
            )
            
            regenerated_responses = self._generate_batch(conversations, structured_sampling_params, continue_final_message=True, use_tqdm=False)
            
            for idx, regenerated_response in zip(indices, regenerated_responses):
                validated_response = extract_last_json_object(regenerated_response, constraint)
                if validated_response is None and raise_exceptions:
                    raise ValueError(f"Failed to validate regenerated response at index {idx}: {regenerated_response[:100]}.")
                validated_responses[idx] = validated_response

        # At this point, all responses are validated
        return validated_responses

    def close(self):
        del self.model
        del self.lora_request
        del self.sampling_params
