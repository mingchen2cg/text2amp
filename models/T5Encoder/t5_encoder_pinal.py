import torch
import torch.nn as nn
from transformers import T5EncoderModel, AutoTokenizer
from typing import Optional, Union, Dict, Any


class T5Encoder(nn.Module):
    """
    独立封装的 T5EncoderModel
    - forward 默认返回 last_hidden_state: (B, L, H)
    - 可选返回完整 ModelOutput（和 HF T5EncoderModel 一致）
    """
    def __init__(
        self,
        lm_dir: str,
        device: torch.device,
        dtype: torch.dtype = torch.float16,
    ):
        super().__init__()

        self.lm = T5EncoderModel.from_pretrained(
            lm_dir,
            torch_dtype=dtype,
        )
        self.lm.to(device)
        self.device = device
        self.dtype = dtype

        # 默认评估模式
        self.eval()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        return_dict: bool = True,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
    ) -> Union[torch.Tensor, Dict[str, Any]]:
        """
        Args:
            input_ids: (B, L)
            attention_mask: (B, L), 1 for valid tokens, 0 for padding
            return_dict: 是否返回 HF 的 ModelOutput
            output_hidden_states: 是否返回每一层的 hidden states
            output_attentions: 是否返回 attention

        Returns:
            - 若 return_dict=True: transformers.modeling_outputs.BaseModelOutput
            - 否则: tuple，第一项是 last_hidden_state
        """
        outputs = self.lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=return_dict,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )

        if return_dict:
            return outputs
        else:
            # 和 HF 行为一致：outputs[0] 是 last_hidden_state
            return outputs[0]