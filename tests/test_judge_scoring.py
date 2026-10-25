import torch
from src.abstention_classifier import LABELS, score_labels
from tests._tiny_model import StubTokenizer, tiny_model


def manual_score(model, tokenizer, prompt_text, label):
    chat=tokenizer.apply_chat_template([{"role":"user","content":prompt_text}],tokenize=False,add_generation_prompt=True)
    prompt_ids=tokenizer.encode(chat,add_special_tokens=False)
    label_ids=tokenizer.encode(label,add_special_tokens=False)
    seq=prompt_ids+label_ids
    with torch.no_grad():
        logits=model(input_ids=torch.tensor([seq]),attention_mask=torch.ones((1,len(seq)),dtype=torch.long)).logits.float()
        logp=logits.log_softmax(-1)
        positions=torch.arange(len(prompt_ids)-1,len(prompt_ids)-1+len(label_ids))
        return float(logp[0,positions,torch.tensor(label_ids)].sum())


def test_teacher_forced_scores_are_finite_argmax_and_repeatable():
    model=tiny_model(seed=19);tokenizer=StubTokenizer();prompt="Classify this reply."
    first=score_labels(model,tokenizer,prompt);second=score_labels(model,tokenizer,prompt)
    assert set(first["judge_logprobs"])==set(LABELS)
    assert all(torch.isfinite(torch.tensor(v)) for v in first["judge_logprobs"].values())
    assert first["judge_label"]==max(LABELS,key=lambda label:first["judge_logprobs"][label])
    assert first==second


def test_variable_label_lengths_match_manual_teacher_forcing():
    model=tiny_model(seed=7);tokenizer=StubTokenizer();prompt="Use the prompt exactly."
    result=score_labels(model,tokenizer,prompt)
    for label in LABELS:
        expected=manual_score(model,tokenizer,prompt,label)
        assert abs(result["judge_logprobs"][label]-expected)<1e-5
