"""Real Qwen tokenizer regressions; no model weights or GPU allocation needed."""

import json
import os
from pathlib import Path

import pytest


pytestmark = pytest.mark.skipif(os.environ.get("WHEELARM_MODEL_TEST") != "1",
                                reason="Requires local Qwen tokenizers and XGrammar")


@pytest.fixture(params=["Qwen3-4B", "Qwen3-VL-4B-Instruct"])
def grammar(request):
    from transformers import AutoTokenizer
    import xgrammar as xgr

    root = Path("~/Models/transformers").expanduser() / request.param
    tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True)
    config = json.loads((root / "config.json").read_text())
    vocab_size = config.get("vocab_size") or config["text_config"]["vocab_size"]
    info = xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=vocab_size,
                                              stop_token_ids=[151645, 151643])
    schema = {"type": "object", "required": ["message"], "additionalProperties": False,
              "properties": {"message": {"type": "string"}}}
    compiled = xgr.GrammarCompiler(info).compile_json_schema(json.dumps(schema))
    return tokenizer, compiled


def test_chinese_and_escaped_control_tokens_are_valid(grammar):
    import xgrammar as xgr
    tokenizer, compiled = grammar
    text = json.dumps({"message": "依次导航到厨房，再去卧室。\n允许转义换行。"}, ensure_ascii=False)
    matcher = xgr.GrammarMatcher(compiled)
    for token in tokenizer.encode(text, add_special_tokens=False):
        assert matcher.accept_token(token)
    assert matcher.accept_token(tokenizer.eos_token_id)


@pytest.mark.parametrize("invalid", ['{"message":"导航\t停止"}', '{"message":"导航\n停止"}'])
def test_literal_control_characters_are_rejected(grammar, invalid):
    import xgrammar as xgr
    _, compiled = grammar
    assert not xgr.GrammarMatcher(compiled).accept_string(invalid)


def test_incomplete_json_cannot_end_early(grammar):
    import xgrammar as xgr
    tokenizer, compiled = grammar
    matcher = xgr.GrammarMatcher(compiled)
    assert matcher.accept_string('{"message":"导航')
    assert not matcher.accept_token(tokenizer.eos_token_id)


def test_vision_image_processor_can_convert_numpy_to_torch():
    from PIL import Image
    from transformers import AutoProcessor

    root = Path("~/Models/transformers/Qwen3-VL-4B-Instruct").expanduser()
    processor = AutoProcessor.from_pretrained(root, local_files_only=True)
    messages = [{"role": "user", "content": [
        {"type": "image", "image": Image.new("RGB", (64, 64), "red")},
        {"type": "text", "text": "画面里有什么？"}]}]
    inputs = processor.apply_chat_template(messages, tokenize=True,
        add_generation_prompt=True, return_dict=True, return_tensors="pt")
    assert inputs["pixel_values"].numel() > 0
    assert inputs["input_ids"].shape[0] == 1
