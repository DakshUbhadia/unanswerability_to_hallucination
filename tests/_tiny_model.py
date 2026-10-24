"""CPU fixtures: a tiny random Llama, a byte-level chat tokenizer stub, and matching synthetic directions."""
import torch

from src.config import CFG

SUFFIX = "<as>\n"  # five byte tokens play the role of the post-instruction suffix
N_SPECIAL = 3       # 0 = pad, 1 = eos, 2 = bos


class StubTokenizer:
    padding_side = "left"
    pad_token_id = 0
    eos_token_id = 1
    pad_token = "<pad>"
    eos_token = "<eos>"

    def encode(self, text, add_special_tokens=False):
        return [b + N_SPECIAL for b in str(text).encode("utf-8")]

    def apply_chat_template(self, conv, tokenize=False, add_generation_prompt=True):
        system = conv[0]["content"][:12]
        user = conv[-1]["content"][:48]
        return f"S:{system}|U:{user}{SUFFIX if add_generation_prompt else ''}"

    def __call__(self, texts, return_tensors="pt", padding=True, add_special_tokens=False, truncation=False, **_):
        ids = [self.encode(t) for t in texts]
        width = max(len(x) for x in ids)
        input_ids = torch.full((len(ids), width), self.pad_token_id, dtype=torch.long)
        mask = torch.zeros((len(ids), width), dtype=torch.long)
        for i, row in enumerate(ids):
            if self.padding_side == "left":
                input_ids[i, width - len(row):] = torch.tensor(row)
                mask[i, width - len(row):] = 1
            else:
                input_ids[i, :len(row)] = torch.tensor(row)
                mask[i, :len(row)] = 1
        return {"input_ids": input_ids, "attention_mask": mask}

    def decode(self, ids, skip_special_tokens=True):
        return bytes(int(i) - N_SPECIAL for i in ids if int(i) >= N_SPECIAL).decode("utf-8", errors="ignore")

    def batch_decode(self, rows, skip_special_tokens=True):
        return [self.decode(r, skip_special_tokens) for r in rows]


def tiny_model(n_layers=3, d_model=32, seed=0):
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(seed)
    cfg = LlamaConfig(vocab_size=256 + N_SPECIAL, hidden_size=d_model, intermediate_size=64,
                      num_hidden_layers=n_layers, num_attention_heads=4, num_key_value_heads=2,
                      max_position_embeddings=512, pad_token_id=0, eos_token_id=1, bos_token_id=2)
    model = LlamaForCausalLM(cfg).eval()
    return model


def tiny_contract(tokenizer, n_layers=3, d_model=32):
    unknown = tokenizer.encode("U")[0], tokenizer.encode("u")[0]
    return {"model_name": "tiny-llama", "n_layers": n_layers, "d_model": d_model, "k_suffix": len(SUFFIX),
            "suffix_token_ids": tokenizer.encode(SUFFIX), "is_layer_output_tuple": True,
            "unknown_token_id_nospace": unknown[0], "unknown_token_id_space": unknown[1],
            "unknown_token_ids": list(unknown), "system_prompt_used": CFG.system_prompt, "dtype": "torch.float32"}


def tiny_directions(n_layers=3, k=5, d_model=32, seed=1):
    g = torch.Generator().manual_seed(seed)

    def unit(*shape):
        x = torch.randn(*shape, generator=g)
        return x / x.norm(dim=-1, keepdim=True)

    learned = unit(n_layers, k, d_model)
    norms = torch.rand(n_layers, k, generator=g) * 2 + 0.5
    concept = unit(n_layers, k, d_model)
    concept = concept - (concept * learned).sum(-1, keepdim=True) * learned
    concept = concept / concept.norm(dim=-1, keepdim=True)
    shuf_unit = unit(CFG.n_ctrl_shuffled, n_layers, k, d_model)
    return {"diff_in_means": learned * norms.unsqueeze(-1), "unit_directions": learned, "norms": norms,
            "random_controls_unit": unit(CFG.n_ctrl_random, n_layers, k, d_model),
            "shuffled_controls_unit": shuf_unit, "shuffled_controls_diff": shuf_unit * 0.3,
            "r_concept_unit": concept, "r_concept_ortho_scaled": concept * norms.unsqueeze(-1),
            "cos_pre_ortho": torch.zeros(n_layers, k)}


def tiny_selection(l_star=1, i_star=-1, k=5):
    return {"l_star": l_star, "i_star": i_star, "pos_idx": list(range(-k, 0)).index(i_star),
            "cos_concept_pre_ortho_at_star": 0.1}
