import torch
import torch.nn as nn
from transformers import T5Config, T5ForConditionalGeneration
from .StructureTokenPredictionModel import StructureTokenPredictionModel
from .abs_model import ABSmodule

class EndToEndModel(ABSmodule):
    def __init__(self, t2struc_config, prostt5_config_path):
        super().__init__()
        
        # ===================== 1. Structure Prediction Module (T2Struc) ===================== #
        self.t2struc = StructureTokenPredictionModel(t2struc_config)
        
        self.structure_hidden_size = self.t2struc.plm.config.hidden_size # 1280
        self.text_hidden_size = self.t2struc.lm.config.hidden_size       # 1024

        # ===================== 2. Sequence Generation Module (ProstT5) ===================== #
        try:
            prostt5_config = T5Config.from_pretrained(prostt5_config_path, local_files_only=True)
        except:
            print(f"Warning: Could not load config from {prostt5_config_path}, using default T5Config.")
            prostt5_config = T5Config()
            
        self.prostt5 = T5ForConditionalGeneration(prostt5_config)
        self.prostt5_hidden_size = self.prostt5.config.d_model # 1024

        # ===================== 3. Projection Layers ===================== #
        # Structure Projector: 1280 -> 1024
        self.structure_projector = nn.Linear(self.structure_hidden_size, self.prostt5_hidden_size)
        
        # Text Projector: 1024 -> 1024
        if self.text_hidden_size != self.prostt5_hidden_size:
            self.text_projector = nn.Linear(self.text_hidden_size, self.prostt5_hidden_size)
        else:
            self.text_projector = nn.Identity()

        # 初始化投影层
        nn.init.xavier_uniform_(self.structure_projector.weight)
        if self.structure_projector.bias is not None:
            nn.init.constant_(self.structure_projector.bias, 0)

    def infer_text(self, batch, **kwargs):
        return self.t2struc.infer_text(batch)

    def forward(self, batch):
        """
        Training Forward Pass
        """
        text_hidden_states, text_masks = self.t2struc.infer_text(batch)
        
        # 使用 .transformer 获取基础输出
        t2struc_outputs = self.t2struc.plm.transformer(
            input_ids=batch["structure_token_ids"],
            attention_mask=batch["structure_token_masks"],
            encoder_hidden_states=text_hidden_states,
            encoder_attention_mask=text_masks,
            return_dict=True
        )
        
        structure_hidden_states = t2struc_outputs.last_hidden_state
        structure_masks = batch["structure_token_masks"]

        # Projection & Concat
        structure_embeds_proj = self.structure_projector(structure_hidden_states)
        text_embeds_proj = self.text_projector(text_hidden_states)
        
        inputs_embeds = torch.cat([text_embeds_proj, structure_embeds_proj], dim=1)
        attention_mask = torch.cat([text_masks, structure_masks], dim=1)

        prostt5_outputs = self.prostt5(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=batch.get("labels", None),
            return_dict=True
        )

        return {
            "loss": prostt5_outputs.loss,
            "logits": prostt5_outputs.logits
        }

    @torch.no_grad()
    def generate(self, batch, generation_config=None):
        """
        Inference Logic
        """
        # 1. Text Encoding
        text_hidden_states, text_masks = self.t2struc.infer_text(batch)
        
        # ===================== [CRITICAL FIX: Correct IDs for your Tokenizer] =====================
        plm_config = self.t2struc.plm.config
        vocab_size = self.t2struc.plm.transformer.wte.weight.shape[0] # 25
        
        # 你的 Tokenizer 特殊设定: Start=1, EOS=2, Pad=0
        # 逻辑：如果 Config 里的值非法 (>=25)，则强制使用你的正确值
        
        # 1. BOS (Start Token): 你的值是 1
        bos_token_id = plm_config.bos_token_id
        if bos_token_id is None or bos_token_id >= vocab_size:
            bos_token_id = 1  # [Corrected]
            
        # 2. EOS (Stop Token): 你的值是 2
        eos_token_id = plm_config.eos_token_id
        if eos_token_id is None or eos_token_id >= vocab_size:
            eos_token_id = 2  # [Corrected]
            
        # 3. PAD (Padding Token): 你的值是 0
        pad_token_id = plm_config.pad_token_id
        if pad_token_id is None or pad_token_id >= vocab_size:
            pad_token_id = 0  # [Corrected]
        # =========================================================================================

        # 2. Structure Generation
        generated_structure = self.t2struc.plm.generate(
            input_ids=None, 
            encoder_hidden_states=text_hidden_states,
            encoder_attention_mask=text_masks,
            bos_token_id=bos_token_id,  
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            max_length=batch.get("max_structure_len", 512),
            output_hidden_states=True,
            return_dict_in_generate=True,
            **generation_config.get("structure_gen_kwargs", {}) if generation_config else {}
        )
        
        generated_structure_ids = generated_structure.sequences
        
        # 获取 Hidden States (直接调用 .transformer)
        structure_outputs = self.t2struc.plm.transformer(
            input_ids=generated_structure_ids,
            encoder_hidden_states=text_hidden_states,
            encoder_attention_mask=text_masks,
            return_dict=True
        )
        structure_hidden_states = structure_outputs.last_hidden_state
        
        # [CRITICAL] Mask 计算：确保 Pad (0) 被正确 Mask 掉
        structure_masks = (generated_structure_ids != pad_token_id).long()

        # 3. Projection & Concat
        structure_embeds_proj = self.structure_projector(structure_hidden_states)
        text_embeds_proj = self.text_projector(text_hidden_states)
        
        inputs_embeds = torch.cat([text_embeds_proj, structure_embeds_proj], dim=1)
        attention_mask = torch.cat([text_masks, structure_masks], dim=1)
        
        # 4. Sequence Generation (ProstT5)
        generated_sequence = self.prostt5.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_length=batch.get("max_seq_len", 512),
            **generation_config.get("seq_gen_kwargs", {}) if generation_config else {}
        )
        
        return generated_sequence