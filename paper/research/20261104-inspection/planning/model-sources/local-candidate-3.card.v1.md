---
language:
- en
- fr
- de
- es
- it
- pt
- zh
- ja
- ru
- ko
license: apache-2.0
library_name: vllm
base_model: mistralai/Mistral-Small-24B-Instruct-2501
extra_gated_description: If you want to learn more about how we process your personal
  data, please read our <a href="https://mistral.ai/terms/">Privacy Policy</a>.
tags:
- mlx
---

# mlx-community/Mistral-Small-24B-Instruct-2501-4bit

The Model [mlx-community/Mistral-Small-24B-Instruct-2501-4bit](https://huggingface.co/mlx-community/Mistral-Small-24B-Instruct-2501-4bit) was
converted to MLX format from [mistralai/Mistral-Small-24B-Instruct-2501](https://huggingface.co/mistralai/Mistral-Small-24B-Instruct-2501)
using mlx-lm version **0.21.0**.

## Use with mlx

```bash
pip install mlx-lm
```

```python
from mlx_lm import load, generate

model, tokenizer = load("mlx-community/Mistral-Small-24B-Instruct-2501-4bit")

prompt = "hello"

if tokenizer.chat_template is not None:
    messages = [{"role": "user", "content": prompt}]
    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True
    )

response = generate(model, tokenizer, prompt=prompt, verbose=True)
```
