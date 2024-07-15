import warnings

import torch
import torch.nn as nn

from models.config.prompt_config import PromptEncoderReparameterizationType, PromptEncoderConfig

"""
The model to encode virtual tokens into prompt embeddings.

Args:
    config ([`PromptTuningConfig`]): The configuration of the prompt embedding.
    word_embeddings (`torch.nn.Module`): The word embeddings of the base transformer model.

**Attributes**:
    - **embedding** (`torch.nn.Embedding`) -- The embedding layer of the prompt embedding.

Example:

```py
>>> from peft import PromptEmbedding, PromptTuningConfig

>>> config = PromptTuningConfig(
...     peft_type="PROMPT_TUNING",
...     task_type="SEQ_2_SEQ_LM",
...     num_virtual_tokens=20,
...     token_dim=768,
...     num_transformer_submodules=1,
...     num_attention_heads=12,
...     num_layers=12,
...     prompt_tuning_init="TEXT",
...     prompt_tuning_init_text="Predict if sentiment of this review is positive, negative or neutral",
...     tokenizer_name_or_path="t5-base",
... )

>>> # t5_model.shared is the word embeddings of the base model
>>> prompt_embedding = PromptEmbedding(config, t5_model.shared)
```

Input Shape: (`batch_size`, `total_virtual_tokens`)

Output Shape: (`batch_size`, `total_virtual_tokens`, `token_dim`)
"""


class PromptEncoder(nn.Module):
    """
    The prompt encoder network that is used to generate the virtual token embeddings for p-tuning.
    """

    def __init__(self, config):
        super().__init__()
        self.token_dim = config.token_dim  # The hidden embedding dimension of the base transformer model.
        self.input_size = self.token_dim  # The input size of the prompt encoder.
        self.output_size = self.token_dim
        self.hidden_size = config.encoder_hidden_size  # The hidden size of the prompt encoder.
        # 可训练的prompt
        self.total_virtual_tokens = config.num_virtual_tokens * config.num_transformer_submodules  # The total number of virtual tokens
        self.encoder_type = config.encoder_reparameterization_type

        # embedding
        self.embedding = torch.nn.Embedding(self.total_virtual_tokens, self.token_dim)
        if not config.inference_mode:
            if self.encoder_type == PromptEncoderReparameterizationType.LSTM:
                lstm_dropout = config.encoder_dropout
                num_layers = config.encoder_num_layers
                # LSTM
                self.lstm_head = torch.nn.LSTM(
                    input_size=self.input_size,
                    hidden_size=self.hidden_size,
                    num_layers=num_layers,
                    dropout=lstm_dropout,
                    bidirectional=True,
                    batch_first=True,
                )
                #  MLP head
                self.mlp_head = torch.nn.Sequential(
                    torch.nn.Linear(self.hidden_size * 2, self.hidden_size * 2),
                    torch.nn.ReLU(),
                    torch.nn.Linear(self.hidden_size * 2, self.output_size),
                )

            elif self.encoder_type == PromptEncoderReparameterizationType.MLP:
                encoder_num_layers_default = PromptEncoderConfig.encoder_num_layers
                if config.encoder_num_layers != encoder_num_layers_default:
                    warnings.warn(
                        f"for {self.encoder_type.value}, the argument `encoder_num_layers` is ignored. "
                        f"Exactly {encoder_num_layers_default} MLP layers are used."
                    )
                layers = [
                    torch.nn.Linear(self.input_size, self.hidden_size),
                    torch.nn.ReLU(),
                    torch.nn.Linear(self.hidden_size, self.hidden_size),
                    torch.nn.ReLU(),
                    torch.nn.Linear(self.hidden_size, self.output_size),
                ]
                # MLP head
                self.mlp_head = torch.nn.Sequential(*layers)

            else:
                raise ValueError("Prompt encoder type not recognized. Please use one of MLP (recommended) or LSTM.")

    def forward(self, indices):
        input_embeds = self.embedding(indices)
        if self.encoder_type == PromptEncoderReparameterizationType.LSTM:
            output_embeds = self.mlp_head(self.lstm_head(input_embeds)[0])
        elif self.encoder_type == PromptEncoderReparameterizationType.MLP:
            output_embeds = self.mlp_head(input_embeds)
        else:
            raise ValueError("Prompt encoder type not recognized. Please use one of MLP (recommended) or LSTM.")

        return output_embeds
