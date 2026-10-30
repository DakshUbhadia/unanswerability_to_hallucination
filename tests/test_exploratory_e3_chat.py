from src.exploratory.chat import discover_suffix, recover_render_date, render_ids, template_date_probe
from src.model_utils import discover_post_instruction_suffix
from src.tokenization_contract import canonical_input_ids, chat_template_sha256, input_ids_sha256
from tests._tiny_model import StubTokenizer


class ChatStub(StubTokenizer):
    chat_template = "date-aware-e3-stub"

    def apply_chat_template(self, conv, tokenize=False, add_generation_prompt=True, date_string=None):
        system, user = conv[0]["content"], conv[-1]["content"]
        date = f"DATE:{date_string}|" if date_string else "DATE:dynamic|"
        suffix = "<as>\n" if add_generation_prompt else ""
        rendered = f"S:{date}{system}|U:{user}{suffix}"
        return self.encode(rendered) if tokenize else rendered


def test_discover_suffix_matches_registered_prompt():
    tok = ChatStub()
    expected = discover_post_instruction_suffix(tok)
    assert discover_suffix(tok, "registered system") == expected


def test_render_without_date_delegates_to_canonical_contract():
    tok = ChatStub()
    suffix = discover_suffix(tok, "neutral")[1]
    assert render_ids(tok, "question", "neutral", suffix) == canonical_input_ids(tok, "question", "neutral", suffix)


def test_date_probe_and_recovery_find_planted_date():
    tok = ChatStub()
    prompt, question, model = "neutral", "question", "stub-model"
    suffix = discover_suffix(tok, prompt)[1]
    probe = template_date_probe(tok, prompt)
    assert probe["dynamic"]
    planted = "17 Oct 2026"
    digest = input_ids_sha256(render_ids(tok, question, prompt, suffix, planted), model,
                              chat_template_sha256(tok))
    rows = [{"question": question, "input_ids_sha256": digest}]
    assert recover_render_date(tok, rows, prompt, model, suffix, "2026-10-15", "2026-10-20") == planted
