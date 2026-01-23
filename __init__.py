from .nodes import (
    Qwen3Loader,
    Qwen3CustomVoice,
    Qwen3VoiceDesign,
    Qwen3VoiceClone,
    Qwen3PromptMaker,
    Qwen3DatasetFromFolder,
    Qwen3DataPrep,
    Qwen3FineTune
)

NODE_CLASS_MAPPINGS = {
    "Qwen3Loader": Qwen3Loader,
    "Qwen3CustomVoice": Qwen3CustomVoice,
    "Qwen3VoiceDesign": Qwen3VoiceDesign,
    "Qwen3VoiceClone": Qwen3VoiceClone,
    "Qwen3PromptMaker": Qwen3PromptMaker,
    "Qwen3DatasetFromFolder": Qwen3DatasetFromFolder,
    "Qwen3DataPrep": Qwen3DataPrep,
    "Qwen3FineTune": Qwen3FineTune
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Qwen3Loader": "Qwen3-TTS Loader",
    "Qwen3CustomVoice": "Qwen3-TTS Custom Voice",
    "Qwen3VoiceDesign": "Qwen3-TTS Voice Design",
    "Qwen3VoiceClone": "Qwen3-TTS Voice Clone",
    "Qwen3PromptMaker": "Qwen3-TTS Prompt Maker",
    "Qwen3DatasetFromFolder": "Qwen3-TTS Dataset Maker",
    "Qwen3DataPrep": "Qwen3-TTS Data Prep",
    "Qwen3FineTune": "Qwen3-TTS Finetune"
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
