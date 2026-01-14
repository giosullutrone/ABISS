from typing import cast
from .model import Model
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from vllm.sampling_params import SamplingParams, StructuredOutputsParams
from models import extract_last_json_object
from pydantic import BaseModel


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
    
    def _generate_batch(self, prompts: list[list[dict[str, str]]], sampling_params: SamplingParams | None, continue_final_message: bool=False) -> list[str]:
        responses: list[str] = []
        for i in range(0, len(prompts), self.max_batch_with_text_size):
            batch_conversations: list[list[dict[str, str]]] = prompts[i:i + self.max_batch_with_text_size]
            batch_responses = self.model.chat(
                messages=batch_conversations, # type: ignore
                sampling_params=sampling_params,
                lora_request=self.lora_request,
                continue_final_message=continue_final_message
            )
            for batch_response in batch_responses:
                responses.extend([x.text for x in batch_response.outputs])
        return responses

    def generate_batch(self, prompts: list[str] | list[list[dict[str, str]]]) -> list[str]:
        return self._generate_batch(self.convert_prompt_to_conversation_if_needed(prompts), self.sampling_params)

    def _generate_batch_with_constraints(self, prompts: list[list[dict[str, str]]], constraints: list[type[BaseModel]]) -> list[BaseModel]:
        """
        Generate responses for a batch of prompts while enforcing constraints defined by the input Pydantic models.
        """
        # Generate raw responses
        responses = self.generate_batch(prompts)

        # Parse and validate each response against the corresponding constraint model
        validated_responses: list[BaseModel | None] = []
        for response, constraint in zip(responses, constraints):
            validated_response = extract_last_json_object(response, constraint)
            validated_responses.append(validated_response)

        if all(v is not None for v in validated_responses):
            return cast(list[BaseModel], validated_responses)

        # If there are any None values, Re-Generate those specific responses with StructuredOutputsParams while asking the model to continue from where it left off
        conversations_to_regenerate: dict[int, list[dict[str, str]]] = {}

        for idx, validated_response in enumerate(validated_responses):
            if validated_response is not None:
                continue

            conversation_to_regenerate = prompts[idx]
            # We have to add the assistance message with the previous incomplete response
            conversation_to_regenerate.append({"role": "assistant", "content": responses[idx] + "\n"})
            conversations_to_regenerate[idx] = conversation_to_regenerate

        # A batch can't have multiple different StructuredOutputsParams, so we regenerate one by one
        # We also set add_generation_prompt to False as continuing from the previous message is not compatible with adding a new generation prompt
        for idx, conversation in conversations_to_regenerate.items():
            structured_sampling_params = SamplingParams(
                **{**(self.sampling_kwargs if self.sampling_kwargs is not None else {}), **{"add_generation_prompt": False}},
                structured_outputs=StructuredOutputsParams(
                    json=constraints[idx].model_json_schema()
                ),
            )
            regenerated_response = self._generate_batch([conversation], structured_sampling_params, continue_final_message=True)[0]
            validated_response = extract_last_json_object(regenerated_response, constraints[idx])
            if validated_response is None:
                raise ValueError(f"Failed to validate regenerated response for prompt index {idx}.")
            validated_responses[idx] = validated_response

        # At this point, all responses are validated
        return cast(list[BaseModel], validated_responses)

    def generate_batch_with_constraints(self, prompts: list[str] | list[list[dict[str, str]]], constraints: list[type[BaseModel]]) -> list[BaseModel]:
        return self._generate_batch_with_constraints(self.convert_prompt_to_conversation_if_needed(prompts), constraints)

    def close(self):
        del self.model
        del self.lora_request
        del self.sampling_params
