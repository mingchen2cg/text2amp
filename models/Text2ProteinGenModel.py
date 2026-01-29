import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any

# 引入基础组件
from transformers import T5EncoderModel, T5Config
from models.progen3_module.modeling import ProGen3ForCausalLM, RMSNorm
from models.progen3_module.config import ProGen3Config
from models.pinal_module.abs_model import ABSmodule

# ================================================================= #
#                辅助模块: Cross Attention & Modified Layer          #
# ================================================================= #

class CrossAttentionBlock(nn.Module):
    def __init__(self, hidden_size, encoder_hidden_size, num_heads, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            kdim=encoder_hidden_size,
            vdim=encoder_hidden_size,
            batch_first=True
        )
        self.norm = RMSNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states, encoder_hidden_states, encoder_attention_mask=None):
        # 自动对齐精度
        if encoder_hidden_states.dtype != hidden_states.dtype:
            encoder_hidden_states = encoder_hidden_states.to(hidden_states.dtype)

        residual = hidden_states
        hidden_states = self.norm(hidden_states)

        key_padding_mask = None
        if encoder_attention_mask is not None:
            key_padding_mask = (encoder_attention_mask == 0).bool()

        attn_output, _ = self.attn(
            query=hidden_states,
            key=encoder_hidden_states,
            value=encoder_hidden_states,
            key_padding_mask=key_padding_mask
        )
        return residual + self.dropout(attn_output)

class ProGen3LayerWithCrossAttn(nn.Module):
    """
    这是一个"完全体"的层定义。
    我们不再动态劫持，而是直接定义它包含：Original Layer + Cross Attn
    """
    def __init__(self, original_layer, t5_hidden_dim):
        super().__init__()
        self.original_layer = original_layer # 这是一个 ProGen3Block
        
        # 动态获取参数
        hidden_size = original_layer.hidden_size
        if hasattr(original_layer, "self_attn"):
            num_heads = original_layer.self_attn.num_heads
        elif hasattr(original_layer, "norm_attn_norm"):
            num_heads = original_layer.norm_attn_norm.self_attn.num_heads
        else:
            # Fallback for config
            num_heads = original_layer.num_heads 
        
        self.cross_attn = CrossAttentionBlock(
            hidden_size=hidden_size,
            encoder_hidden_size=t5_hidden_dim,
            num_heads=num_heads
        )

    def forward(self, hidden_states, position_ids, encoder_hidden_states, encoder_attention_mask, 
                past_key_value=None, output_attentions=False, output_router_weights=False, use_cache=False):
        
        # 1. ProGen3 Self-Attention
        if hasattr(self.original_layer, "norm_attn_norm"):
            # Fused
            hidden_states, residual, self_attn_weights, present_key_value = self.original_layer.norm_attn_norm(
                hidden_states=hidden_states, position_ids=position_ids, past_key_value=past_key_value,
                output_attentions=output_attentions, use_cache=use_cache
            )
        else:
            # Unfused
            residual = hidden_states
            hidden_states = self.original_layer.input_layernorm(hidden_states)
            hidden_states, self_attn_weights, present_key_value = self.original_layer.self_attn(
                hidden_states=hidden_states, position_ids=position_ids, past_key_value=past_key_value,
                output_attentions=output_attentions, use_cache=use_cache
            )
            hidden_states = residual + hidden_states

        # 2. Cross-Attention
        hidden_states = self.cross_attn(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask
        )

        # 3. ProGen3 FFN
        residual = hidden_states
        if hasattr(self.original_layer, "post_attention_layernorm"):
            hidden_states = self.original_layer.post_attention_layernorm(hidden_states)
        
        block_sparse_moe = self.original_layer.block_sparse_moe
        if self.original_layer.moe_implementation == "megablocks":
             hidden_states = block_sparse_moe(hidden_states)
             router_weights = None 
        else:
             hidden_states, router_weights = block_sparse_moe(hidden_states)
        
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions: outputs += (self_attn_weights,)
        if use_cache: outputs += (present_key_value,)
        if output_router_weights: outputs += (router_weights,)
        return outputs

# ================================================================= #
#                      主模型类 (最终成品)                           #
# ================================================================= #

class Text2ProteinGenModel(ABSmodule):
    def __init__(self, checkpoint_path):
        super().__init__()
        
        print(f"Loading model configuration from {checkpoint_path}...")
        # 1. 加载 Checkpoint
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        config_dict = checkpoint["config"]
        
        # 2. 从配置恢复结构 (Init from Config, NOT from pretrained path)
        print("Building model architecture...")
        
        # A. 构建 T5 Encoder
        self.lm = T5EncoderModel(config_dict["t5_config"])
        
        # B. 构建 ProGen3 Decoder (带 CrossAttn)
        # 这里我们先构建原始的 ProGen3 结构
        temp_progen = ProGen3ForCausalLM(config_dict["progen_config"])
        
        # C. 替换层结构 (Architecture Reconstruction)
        # 这一步是必须的，因为保存的 state_dict 里包含 cross_attn 权重，
        # 如果模型结构里没有这个层，加载权重会报错。
        self.lm_dim = config_dict["lm_dim"]
        new_layers = nn.ModuleList()
        for layer in temp_progen.model.layers:
            # 包装成带 CrossAttn 的层
            new_layers.append(ProGen3LayerWithCrossAttn(layer, self.lm_dim))
        
        temp_progen.model.layers = new_layers
        self.plm = temp_progen
        
        # 3. 加载权重
        print("Loading state dictionary...")
        state_dict = checkpoint["state_dict"]
        
        # 处理 key 名称 (因为之前的包装器层级差异)
        # 如果你之前的 T5 是用 T5Encoder 包装的，key 可能是 'lm.lm.xxx'
        # 如果是原生 T5EncoderModel，key 是 'lm.xxx'
        # 这里做一个简单的清洗以防万一
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("lm.lm."):
                new_key = k.replace("lm.lm.", "lm.")
                new_state_dict[new_key] = v
            else:
                new_state_dict[k] = v
                
        # 加载所有参数 (strict=True 保证结构完美匹配)
        self.load_state_dict(new_state_dict, strict=True)
        
        # 4. 设置设备兼容性 (Flash Attention 需要 BF16/FP16)
        target_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
        self.plm.to(target_dtype)
        
        print("Model restored successfully!")

    def infer_text(self, batch):
        outputs = self.lm(
            input_ids=batch["text_ids"],
            attention_mask=batch["text_masks"],
            return_dict=True
        )
        return outputs.last_hidden_state, batch["text_masks"]

    def infer(self, batch, text_hidden_states=None, text_attention_mask=None):
        input_ids = batch["protein_ids"] # 使用修正后的字段名
        labels = batch.get("labels", input_ids)
        
        # 数据类型对齐
        target_dtype = self.plm.model.embed_tokens.weight.dtype
        if text_hidden_states is not None:
            text_hidden_states = text_hidden_states.to(target_dtype)

        # Embeddings
        inputs_embeds = self.plm.model.embed_tokens(input_ids)
        sequence_ids = batch.get("sequence_ids", torch.zeros_like(input_ids))
        inputs_embeds += self.plm.model.embed_seq_id(sequence_ids)
        
        hidden_states = inputs_embeds
        position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0).expand(input_ids.shape[0], -1)

        # Forward Layers
        for layer in self.plm.model.layers:
            layer_out = layer(
                hidden_states, position_ids=position_ids,
                encoder_hidden_states=text_hidden_states,
                encoder_attention_mask=text_attention_mask
            )
            hidden_states = layer_out[0]

        # Head
        hidden_states = self.plm.model.norm(hidden_states)
        logits = self.plm.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.plm.model.padding_idx)
            loss = loss_fct(shift_logits.view(-1, self.plm.config.vocab_size), shift_labels.view(-1))

        return {"loss": loss, "logits": logits}

    def forward(self, batch):
        ret = {}
        ret["text_hidden_states"], ret["text_attention_mask"] = self.infer_text(batch)
        output_dict = self.infer(
            batch,
            text_hidden_states=ret["text_hidden_states"],
            text_attention_mask=ret["text_attention_mask"]
        )
        ret.update(output_dict)
        return ret