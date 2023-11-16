from typing import Optional, Literal

from dsp.modules.lm import LM

def openai_to_hf(**kwargs):
    hf_kwargs = {}
    for k, v in kwargs.items():
        if k == "n":
            hf_kwargs["num_return_sequences"] = v
        elif k == "frequency_penalty":
            hf_kwargs["repetition_penalty"] = 1.0 - v
        elif k == "presence_penalty":
            hf_kwargs["diversity_penalty"] = v
        elif k == "max_tokens":
            hf_kwargs["max_new_tokens"] = v
        elif k == "model":
            pass
        else:
            hf_kwargs[k] = v

    return hf_kwargs


class CTransformersModel(LM):
    def __init__(self, model: str = "TheBloke/Llama-2-13b-Chat-GGUF", model_file:str="llama-2-13b-chat.Q4_K_M.gguf", model_type: str="llama", gpu_layers: int = 0):
        """wrapper for CTransformers

        Args:
            model (str): HF model identifier to load and use
            gpu_layers (int, optional): how many layers to offload to GPU. Defaults to 0.
        """
        try:
            from ctransformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ModuleNotFoundError(
                "You need to install ctransformers library to use llama.cpp models."
            ) from exc
        super().__init__(model)
        self.provider = "llama.cpp"
        self.model = AutoModelForCausalLM.from_pretrained(
            model,
            model_file=model_file,
            model_type=model_type,
            gpu_layers=gpu_layers,
            context_length=4096,
        )
        self.encoder_decoder_model = False
        self.decoder_only_model = True
        assert self.encoder_decoder_model or self.decoder_only_model, f"Unknown model class: {model}"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model)
        self.rationale = True
        self.drop_prompt_from_output = False
        self.history = []

    def basic_request(self, prompt, **kwargs):
        raw_kwargs = kwargs
        kwargs = {**self.kwargs, **kwargs}
        response = self._generate(prompt, **kwargs)

        history = {
            "prompt": prompt,
            "response": response,
            "kwargs": kwargs,
            "raw_kwargs": raw_kwargs,
        }
        self.history.append(history)

        return response

    def _generate(self, prompt, **kwargs):
        assert not self.is_client
        # TODO: Add caching
        kwargs = {**openai_to_hf(**self.kwargs), **openai_to_hf(**kwargs)}
        # print(prompt)
        if isinstance(prompt, dict):
            try:
                prompt = prompt['messages'][0]['content']
            except (KeyError, IndexError, TypeError):
                print("Failed to extract 'content' from the prompt.")
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # print(kwargs)
        outputs = self.model.generate(**inputs, **kwargs)
        if self.drop_prompt_from_output:
            input_length = inputs.input_ids.shape[1]
            outputs = outputs[:, input_length:]
        completions = [
            {"text": c}
            for c in self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        ]
        response = {
            "prompt": prompt,
            "choices": completions,
        }
        return response

    def __call__(self, prompt, only_completed=True, return_sorted=False, **kwargs):
        assert only_completed, "for now"
        assert return_sorted is False, "for now"

        if kwargs.get("n", 1) > 1 or kwargs.get("temperature", 0.0) > 0.1:
            kwargs["do_sample"] = True

        response = self.request(prompt, **kwargs)
        return [c["text"] for c in response["choices"]]


# @functools.lru_cache(maxsize=None if cache_turn_on else 0)
# @NotebookCacheMemory.cache
# def cached_generate(self, prompt, **kwargs):
#      return self._generate(prompt, **kwargs)
