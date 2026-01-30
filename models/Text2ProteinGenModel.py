import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any

# === [新增 Import] 用于清理 MoE 缓存 ===
import megablocks.layers.moe 

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
            # MultiheadAttention 的 key_padding_mask: True 表示被遮蔽(ignore)，False 表示保留
            # T5 tokenizer通常 0 是 padding, 1 是 mask
            # 这里需确保转换逻辑正确，通常 transformer 输出的 mask 1是有效，0是padding
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
    包含 Original ProGen3 Layer + Cross Attention 的复合层
    """
    def __init__(self, original_layer, t5_hidden_dim):
        super().__init__()
        self.original_layer = original_layer 
        
        # 动态获取 hidden_size 和 num_heads
        hidden_size = original_layer.hidden_size
        
        # 兼容 ProGen3 不同的 Attention 定义 (Fused 或 Unfused)
        if hasattr(original_layer, "self_attn"):
            num_heads = original_layer.self_attn.num_heads
        elif hasattr(original_layer, "norm_attn_norm"):
            num_heads = original_layer.norm_attn_norm.self_attn.num_heads
        else:
            # Fallback
            num_heads = original_layer.num_heads 
        
        self.cross_attn = CrossAttentionBlock(
            hidden_size=hidden_size,
            encoder_hidden_size=t5_hidden_dim,
            num_heads=num_heads
        )

    def forward(self, hidden_states, position_ids, encoder_hidden_states, encoder_attention_mask, 
                past_key_value=None, output_attentions=False, output_router_weights=False, use_cache=False):
        
        # 1. ProGen3 Self-Attention (包含 Norm)
        if hasattr(self.original_layer, "norm_attn_norm"):
            # Fused Attention Norm
            hidden_states, residual, self_attn_weights, present_key_value = self.original_layer.norm_attn_norm(
                hidden_states=hidden_states, position_ids=position_ids, past_key_value=past_key_value,
                output_attentions=output_attentions, use_cache=use_cache
            )
        else:
            # Standard Attention
            residual = hidden_states
            hidden_states = self.original_layer.input_layernorm(hidden_states)
            hidden_states, self_attn_weights, present_key_value = self.original_layer.self_attn(
                hidden_states=hidden_states, position_ids=position_ids, past_key_value=past_key_value,
                output_attentions=output_attentions, use_cache=use_cache
            )
            hidden_states = residual + hidden_states

        # 2. Cross-Attention (注入文本信息)
        hidden_states = self.cross_attn(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask
        )

        # 3. ProGen3 FFN / MoE
        residual = hidden_states
        if hasattr(self.original_layer, "post_attention_layernorm"):
            hidden_states = self.original_layer.post_attention_layernorm(hidden_states)
        
        block_sparse_moe = self.original_layer.block_sparse_moe
        
        # 处理不同的 MoE 返回格式
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
#                      主模型类 (最终修复版)                         #
# ================================================================= #

class Text2ProteinGenModel(ABSmodule):
    def __init__(self, checkpoint_path):
        super().__init__()
        
        print(f"Loading model configuration from {checkpoint_path}...")
        # 1. 加载 Checkpoint
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        config_dict = checkpoint["config"]
        progen_conf = config_dict["progen_config"]

        # === [Critical Fix 1] 兼容参数冻结 ===
        # 强制关闭 MegaBlocks 的内存优化，防止反向传播时报错 "Expected all MLP inputs to need grad"
        if hasattr(progen_conf, "moe_memory_optimized"):
            print("Config: Disabling moe_memory_optimized for training stability...")
            progen_conf.moe_memory_optimized = False
        if hasattr(progen_conf, "moe_grouped_gemm"):
            print("Config: Disabling moe_grouped_gemm for training stability...")
            progen_conf.moe_grouped_gemm = False

        # === [Critical Fix 2] 显存优化 ===
        # 训练时强制关闭 use_cache，防止缓存大量历史状态
        progen_conf.use_cache = False 
        
        # 2. 从配置恢复结构
        print("Building model architecture...")
        
        # A. 构建 T5 Encoder
        self.lm = T5EncoderModel(config_dict["t5_config"])
        
        # B. 构建 ProGen3 Decoder
        temp_progen = ProGen3ForCausalLM(progen_conf)
        
        # C. 插入 Cross Attention 层
        self.lm_dim = config_dict["lm_dim"]
        new_layers = nn.ModuleList()
        for layer in temp_progen.model.layers:
            new_layers.append(ProGen3LayerWithCrossAttn(layer, self.lm_dim))
        
        temp_progen.model.layers = new_layers
        self.plm = temp_progen
        
        # 3. 加载权重
        print("Loading state dictionary...")
        state_dict = checkpoint["state_dict"]
        
        # 清洗 Key 名称
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("lm.lm."):
                new_key = k.replace("lm.lm.", "lm.")
                new_state_dict[new_key] = v
            else:
                new_state_dict[k] = v
                
        # 加载所有参数
        self.load_state_dict(new_state_dict, strict=True)
        
        # 4. 设置设备兼容性
        target_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
        self.plm.to(target_dtype)
        
        print("Model restored successfully!")

    def infer_text(self, batch):
        # === [Critical Fix 3] 节省显存 ===
        # T5 只做特征提取，强制 no_grad，避免存储激活值
        with torch.no_grad():
            outputs = self.lm(
                input_ids=batch["text_ids"],
                attention_mask=batch["text_masks"],
                return_dict=True
            )
        return outputs.last_hidden_state, batch["text_masks"]

    def infer(self, batch, text_hidden_states=None, text_attention_mask=None):
        # === [Critical Fix 4] 修复显存泄漏 ===
        # 必须手动清理 MegaBlocks 累积的 load balancing loss
        if hasattr(self.plm.config, "moe_implementation") and self.plm.config.moe_implementation == "megablocks":
             megablocks.layers.moe.clear_load_balancing_loss()

        input_ids = batch["protein_ids"]
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
            # 显式传递 use_cache=False
            layer_out = layer(
                hidden_states, position_ids=position_ids,
                encoder_hidden_states=text_hidden_states,
                encoder_attention_mask=text_attention_mask,
                use_cache=False 
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