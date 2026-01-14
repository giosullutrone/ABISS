from typing import cast
from .model import Model
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from vllm.sampling_params import StructuredOutputsParams


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
    
    def generate_batch(self, prompts: list[str] | list[list[dict[str, str]]]) -> list[str]:
        conversations: list[list[dict[str, str]]] = []

        # If prompts is a list of list of dicts, we can assume it's already formatted conversations
        if all(isinstance(prompt, list) for prompt in prompts) and all(isinstance(item, dict) for prompt in prompts for item in prompt):
            conversations = cast(list[list[dict[str, str]]], prompts)
        elif all(isinstance(prompt, str) for prompt in prompts):
            for prompt in prompts:
                conversation: list[dict[str, str]] = []

                if self.system_prompt is not None:
                    conversation.append({"role": "system", "content": self.system_prompt})
                conversation.append({"role": "user", "content": cast(str, prompt)})
                conversations.append(conversation)
        else:
            raise ValueError("Prompts must be either a list of strings or a list of list of dicts.")

        responses: list[str] = []
        for i in range(0, len(conversations), self.max_batch_with_text_size):
            batch_conversations: list[list[dict[str, str]]] = conversations[i:i + self.max_batch_with_text_size]
            batch_responses = self.model.chat(
                messages=batch_conversations, # type: ignore
                sampling_params=self.sampling_params,
                lora_request=self.lora_request
            )
            for batch_response in batch_responses:
                responses.extend([x.text for x in batch_response.outputs])
        return responses

    def generate_batch_with_constraints(self, prompts: list[str] | list[list[dict[str, str]]], constraints: dict) -> list[str]:
        # Save the current sampling params
        original_sampling_params = self.sampling_params
        self.sampling_params = SamplingParams(
            **(self.sampling_kwargs if self.sampling_kwargs is not None else {}),
            structured_outputs=StructuredOutputsParams(json=constraints)
        )
        responses = self.generate_batch(prompts)

        # Restore the original sampling params
        self.sampling_params = original_sampling_params
        return responses

    def close(self):
        del self.model
        del self.lora_request
        del self.sampling_params
