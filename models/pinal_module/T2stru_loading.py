import torch
import os
from omegaconf import OmegaConf
from transformers import AutoTokenizer, EsmTokenizer
# 假设 StructureTokenPredictionModel 是从本地文件导入的类
from .StructureTokenPredictionModel import StructureTokenPredictionModel 

# 配置日志（保留基础 logger 即可）
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("T2Struc")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_T2Struc(model_path: str, dtype: torch.dtype = torch.bfloat16, device: torch.device = DEVICE):
    """
    从目录或权重文件加载 T2Struc 模型。
    """
    # 1. 自动定位路径
    if os.path.isdir(model_path):
        cfg_path = os.path.join(model_path, "config.yaml")
        weight_path = os.path.join(model_path, "pytorch_model.bin")
    else:
        cfg_path = os.path.join(os.path.dirname(model_path), "config.yaml")
        weight_path = model_path

    if not os.path.isfile(cfg_path) or not os.path.isfile(weight_path):
        raise FileNotFoundError(f"配置文件或权重缺失: {cfg_path} / {weight_path}")

    # 2. 加载配置与模型
    cfg = OmegaConf.load(cfg_path)
    model = StructureTokenPredictionModel(cfg.model).to(dtype=dtype, device=device)

    # 3. 加载权重
    logger.info(f"Loading weights from: {weight_path}")
    state_dict = torch.load(weight_path, map_location=device)
    model.load_state_dict(state_dict)
    
    # 4. 加载分词器
    text_tokenizer = AutoTokenizer.from_pretrained(cfg.model["lm"])
    structure_tokenizer = EsmTokenizer.from_pretrained(cfg.model["tokenizer"])

    return model.eval(), text_tokenizer, structure_tokenizer