---
language:
- en
- zh
library_name: mlx
license: mit
pipeline_tag: text-generation
tags:
- mlx
base_model: zai-org/GLM-4.7-Flash
---

# mlx-community/GLM-4.7-Flash-4bit

This model [mlx-community/GLM-4.7-Flash-4bit](https://huggingface.co/mlx-community/GLM-4.7-Flash-4bit) was
converted to MLX format from [zai-org/GLM-4.7-Flash](https://huggingface.co/zai-org/GLM-4.7-Flash)
using mlx-lm version **0.30.5**.

## Use with mlx

```bash
pip install mlx-lm
```

```python
from mlx_lm import load, generate

model, tokenizer = load("mlx-community/GLM-4.7-Flash-4bit")

prompt = "hello"

if tokenizer.chat_template is not None:
    messages = [{"role": "user", "content": prompt}]
    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_dict=False,
    )

response = generate(model, tokenizer, prompt=prompt, verbose=True)
```
