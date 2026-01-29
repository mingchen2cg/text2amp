# import torch
# import torch.nn as nn
# from typing import Optional, Tuple, Union, Dict, Any
# from copy import deepcopy

# # 导入提供的基础模块
# from models.pinal_module.abs_model import ABSmodule
# from models.T5Encoder.t5_encoder_pinal import T5Encoder
# # 导入 ProGen3 相关组件
# from models.progen3_module.modeling import ProGen3ForCausalLM, RMSNorm
# from models.progen3_module.config import ProGen3Config

# class CrossAttentionBlock(nn.Module):
#     """
#     标准的 Cross-Attention 模块，用于将 Text Embedding 注入到 Protein Embedding 中。
#     使用 PyTorch 原生 MultiheadAttention。
#     """
#     def __init__(self, hidden_size, encoder_hidden_size, num_heads, dropout=0.1):
#         super().__init__()
#         # kdim 和 vdim 设为 encoder_hidden_size (T5 的维度)
#         # embed_dim 设为 hidden_size (ProGen3 的维度)
#         self.attn = nn.MultiheadAttention(
#             embed_dim=hidden_size,
#             num_heads=num_heads,
#             dropout=dropout,
#             kdim=encoder_hidden_size,
#             vdim=encoder_hidden_size,
#             batch_first=True
#         )
#         self.norm = RMSNorm(hidden_size)
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, hidden_states, encoder_hidden_states, encoder_attention_mask=None):
#         """
#         Args:
#             hidden_states: [Batch, SeqLen_Protein, Dim_Protein]
#             encoder_hidden_states: [Batch, SeqLen_Text, Dim_Text]
#             encoder_attention_mask: [Batch, SeqLen_Text] (0 for masked, 1 for valid)
#         """
#         # 确保数据类型一致 (关键修复：防止 FP32 输入导致 BF16/FP16 模型报错)
#         if encoder_hidden_states.dtype != hidden_states.dtype:
#             encoder_hidden_states = encoder_hidden_states.to(hidden_states.dtype)

#         residual = hidden_states
#         hidden_states = self.norm(hidden_states)

#         # 处理 Mask: MultiheadAttention 需要 key_padding_mask (True 表示被 mask)
#         # 输入的 mask 通常 1 是有效，0 是 padding，所以取反
#         key_padding_mask = None
#         if encoder_attention_mask is not None:
#             # 确保 mask 是 bool 类型
#             key_padding_mask = (encoder_attention_mask == 0).bool()

#         # Cross Attention: Query=Protein, Key/Value=Text
#         attn_output, _ = self.attn(
#             query=hidden_states,
#             key=encoder_hidden_states,
#             value=encoder_hidden_states,
#             key_padding_mask=key_padding_mask
#         )
        
#         return residual + self.dropout(attn_output)

# class ProGen3LayerWithCrossAttn(nn.Module):
#     """
#     Wrapper 类：劫持原始 ProGen3 的 DecoderLayer，
#     在 Self-Attention 和 FFN 之间 插入 Cross-Attention。
#     """
#     def __init__(self, original_layer, t5_hidden_dim):
#         super().__init__()
#         self.original_layer = original_layer
        
#         # --- 修复点 1: 直接从 original_layer 获取 hidden_size ---
#         hidden_size = original_layer.hidden_size
        
#         # --- 修复点 2: 健壮地获取 num_heads ---
#         if hasattr(original_layer, "self_attn"):
#             num_heads = original_layer.self_attn.num_heads
#         elif hasattr(original_layer, "norm_attn_norm"):
#             num_heads = original_layer.norm_attn_norm.self_attn.num_heads
#         else:
#             raise AttributeError("Cannot find self_attn or norm_attn_norm in original_layer")
        
#         # 初始化 Cross Attention (随机初始化)
#         self.cross_attn = CrossAttentionBlock(
#             hidden_size=hidden_size,
#             encoder_hidden_size=t5_hidden_dim,
#             num_heads=num_heads
#         )

#     def forward(
#         self,
#         hidden_states: torch.Tensor,
#         position_ids: torch.LongTensor,
#         encoder_hidden_states: torch.Tensor,       # 新增
#         encoder_attention_mask: torch.Tensor,      # 新增
#         past_key_value=None,
#         output_attentions=False,
#         output_router_weights=False,
#         use_cache=False,
#     ) -> Tuple[torch.Tensor, ...]:
        
#         # ================= 1. Original Self-Attention =================
#         if hasattr(self.original_layer, "norm_attn_norm"):
#             # Fused Norm Case
#             hidden_states, residual, self_attn_weights, present_key_value = self.original_layer.norm_attn_norm(
#                 hidden_states=hidden_states,
#                 position_ids=position_ids,
#                 past_key_value=past_key_value,
#                 output_attentions=output_attentions,
#                 use_cache=use_cache,
#             )
#             # 在 Fused 模式下，hidden_states 已经是经过 Norm 后的，可以直接进入下一级
#         else:
#             # Standard Case (Unfused)
#             self_attn = self.original_layer.self_attn
#             input_layernorm = self.original_layer.input_layernorm
            
#             residual = hidden_states
#             hidden_states = input_layernorm(hidden_states)

#             hidden_states, self_attn_weights, present_key_value = self_attn(
#                 hidden_states=hidden_states,
#                 position_ids=position_ids,
#                 past_key_value=past_key_value,
#                 output_attentions=output_attentions,
#                 use_cache=use_cache,
#             )
#             hidden_states = residual + hidden_states

#         # ================= 2. Injected Cross-Attention =================
#         # 这里的 hidden_states 会被传入 CrossAttentionBlock
#         # 注意：CrossAttentionBlock 内部会自己做 Norm，所以可以直接传 Residual Stream
#         hidden_states = self.cross_attn(
#             hidden_states=hidden_states,
#             encoder_hidden_states=encoder_hidden_states,
#             encoder_attention_mask=encoder_attention_mask
#         )

#         # ================= 3. Original FFN (MoE/MLP) =================
#         if hasattr(self.original_layer, "post_attention_layernorm"):
#             # Standard Case
#             residual = hidden_states
#             hidden_states = self.original_layer.post_attention_layernorm(hidden_states)
#         else:
#             # Fused Case: 简单处理，直接透传 (因为很难拆解 fused kernel 内部逻辑)
#             residual = hidden_states 
            
#         block_sparse_moe = self.original_layer.block_sparse_moe

#         # Fully Connected / MoE
#         if self.original_layer.moe_implementation == "megablocks":
#              hidden_states = block_sparse_moe(hidden_states)
#              router_weights = None 
#         else:
#              hidden_states, router_weights = block_sparse_moe(hidden_states)
        
#         hidden_states = residual + hidden_states

#         # 组装输出
#         outputs = (hidden_states,)
#         if output_attentions:
#             outputs += (self_attn_weights,)
#         if use_cache:
#             outputs += (present_key_value,)
#         if output_router_weights:
#             outputs += (router_weights,)
            
#         return outputs

# class Text2AmpModel(ABSmodule):
#     def __init__(self, model_config):
#         super().__init__()
#         self.config = model_config
        
#         # ===================== 1. Text Encoder (T5) ===================== #
#         # T5 默认保持 float32 即可，后面 cross attention 会做 cast
#         print(f"Loading T5 Encoder from {model_config['lm']}...")
#         self.lm = T5Encoder(
#             lm_dir=model_config["lm"],
#             device=torch.device("cpu"), 
#             dtype=torch.float32 
#         )
        
#         self.lm_dim = self.lm.lm.config.d_model
#         print(f"Text Encoder dim: {self.lm_dim}")

#         # ===================== 2. Protein Decoder (ProGen3) ===================== #
#         print(f"Loading ProGen3 from {model_config['plm_type']}...")
        
#         # --- 关键修改：自动选择支持 Flash Attention 的数据类型 ---
#         # 如果 GPU 支持 bfloat16，优先使用 bf16，否则使用 float16。
#         # 绝不要使用 float32，因为 ProGen3 强制开启 FlashAttention。
#         if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
#             target_dtype = torch.bfloat16
#             print("Using bfloat16 for ProGen3 (Flash Attention requirement)")
#         else:
#             target_dtype = torch.float16
#             print("Using float16 for ProGen3 (Flash Attention requirement)")

#         self.plm = ProGen3ForCausalLM.from_pretrained(
#             model_config["plm_type"],
#             torch_dtype=target_dtype, # 强制指定 dtype
#             trust_remote_code=True
#         )
        
#         # ===================== 3. Surgery (Inject Cross-Attention) ===================== #
#         print("Injecting Cross-Attention modules into ProGen3 layers...")
#         self.inject_cross_attention()
        
#         # 更新 hidden size
#         self.config['hidden_size'] = self.plm.config.hidden_size

#         # 将整个模型转到目标类型（主要是把新初始化的层也转过去）
#         self.plm.to(target_dtype)

#     def inject_cross_attention(self):
#         progen_layers = self.plm.model.layers
#         new_layers = nn.ModuleList()
        
#         for i, layer in enumerate(progen_layers):
#             modified_layer = ProGen3LayerWithCrossAttn(
#                 original_layer=layer,
#                 t5_hidden_dim=self.lm_dim
#             )
#             new_layers.append(modified_layer)
        
#         self.plm.model.layers = new_layers
#         print(f"Successfully modified {len(self.plm.model.layers)} layers.")

#     def infer_text(self, batch):
#         text_output = self.lm(
#             input_ids=batch["text_ids"],
#             attention_mask=batch["text_masks"],
#             return_dict=True,
#             output_hidden_states=False
#         )
#         return text_output.last_hidden_state, batch["text_masks"]

#     def infer(
#         self,
#         batch,
#         text_hidden_states=None,
#         text_attention_mask=None,
#         return_dict=True,
#     ):
#         input_ids = batch["structure_token_ids"]
#         labels = batch.get("labels", input_ids)
        
#         # 1. Embedding
#         batch_size, seq_length = input_ids.shape
#         inputs_embeds = self.plm.model.embed_tokens(input_ids)
        
#         if "sequence_ids" in batch:
#             sequence_ids = batch["sequence_ids"]
#         else:
#             sequence_ids = torch.zeros_like(input_ids)
            
#         inputs_embeds = inputs_embeds + self.plm.model.embed_seq_id(sequence_ids)
#         hidden_states = inputs_embeds
        
#         # 2. Position IDs
#         position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
#         position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

#         # 3. Pass through Modified Layers
#         presents = () if return_dict else None
        
#         # 确保 text_hidden_states 类型正确 (T5可能是fp32，这里转为bf16/fp16)
#         if text_hidden_states is not None:
#             text_hidden_states = text_hidden_states.to(hidden_states.dtype)

#         for layer in self.plm.model.layers:
#             layer_outputs = layer(
#                 hidden_states,
#                 position_ids=position_ids,
#                 encoder_hidden_states=text_hidden_states,
#                 encoder_attention_mask=text_attention_mask,
#                 use_cache=False 
#             )
#             hidden_states = layer_outputs[0]
            
#         # 4. Final Norm
#         hidden_states = self.plm.model.norm(hidden_states)
        
#         # 5. LM Head
#         logits = self.plm.lm_head(hidden_states)
        
#         loss = None
#         if labels is not None:
#             shift_logits = logits[..., :-1, :].contiguous()
#             shift_labels = labels[..., 1:].contiguous()
#             loss_fct = nn.CrossEntropyLoss(ignore_index=self.plm.model.padding_idx)
#             # 计算 Loss 时通常还是建议 FP32，但 torch 交叉熵支持混合精度输入
#             loss = loss_fct(shift_logits.view(-1, self.plm.config.vocab_size), shift_labels.view(-1))

#         if return_dict:
#             return {"loss": loss, "logits": logits}
#         return (loss, logits)

#     def forward(self, batch):
#         ret = dict()
#         # 1. Get Text Features
#         ret["text_hidden_states"], ret["text_attention_mask"] = self.infer_text(batch)
        
#         # 2. Generate/Train Protein
#         output_dict = self.infer(
#             batch,
#             text_hidden_states=ret["text_hidden_states"],
#             text_attention_mask=ret["text_attention_mask"],
#         )
#         ret.update(output_dict)
#         return ret