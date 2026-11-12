"""TOK for the driver's kernel, built without `transformers` (which the kernel cannot read).

Uses the Rust `tokenizers` library on the checkpoint's tokenizer.json, and renders the chat template with
jinja2 the same way transformers' apply_chat_template does: a sandboxed environment with trim_blocks,
lstrip_blocks and loopcontrols, a `tojson` filter, `raise_exception`, and the special-token variables.
scripts/test_harness.py checks that its output is byte-identical to the harness's own rendering (HF tokenizer).
Everything is read from disk before the kernel is confined.
"""
import json
from datetime import datetime
from pathlib import Path


class KernelTokenizer:
    def __init__(self, path):
        import jinja2
        import jinja2.ext
        import jinja2.sandbox
        from tokenizers import Tokenizer
        p = Path(path)
        self._tk = Tokenizer.from_file(str(p / "tokenizer.json"))
        cfg = json.loads((p / "tokenizer_config.json").read_text())
        tmpl = cfg.get("chat_template")
        if (p / "chat_template.jinja").exists():
            tmpl = (p / "chat_template.jinja").read_text()
        if isinstance(tmpl, list):
            tmpl = {t["name"]: t["template"] for t in tmpl}.get("default")

        def tok_str(v):
            return v.get("content") if isinstance(v, dict) else v

        self.special = {k: tok_str(cfg.get(k)) for k in
                        ("bos_token", "eos_token", "unk_token", "pad_token", "sep_token", "cls_token", "mask_token")}
        for k, v in self.special.items():
            setattr(self, k, v)
        self.chat_template = tmpl

        def raise_exception(msg):
            raise jinja2.exceptions.TemplateError(msg)

        def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
            return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)

        env = jinja2.sandbox.ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                                           extensions=[jinja2.ext.loopcontrols])
        env.filters["tojson"] = tojson
        env.globals["raise_exception"] = raise_exception
        env.globals["strftime_now"] = lambda fmt: datetime.now().strftime(fmt)
        self._template = env.from_string(tmpl) if tmpl else None

    # --- HF-like API -------------------------------------------------------------------------------
    def encode(self, text, add_special_tokens=False):
        return self._tk.encode(text, add_special_tokens=add_special_tokens).ids

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": self.encode(text, add_special_tokens=add_special_tokens)}

    def decode(self, ids, skip_special_tokens=False):
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        return self._tk.decode(list(ids), skip_special_tokens=skip_special_tokens)

    def convert_ids_to_tokens(self, ids):
        if isinstance(ids, int):
            return self._tk.id_to_token(ids)
        return [self._tk.id_to_token(int(i)) for i in ids]

    def convert_tokens_to_ids(self, toks):
        if isinstance(toks, str):
            return self._tk.token_to_id(toks)
        return [self._tk.token_to_id(t) for t in toks]

    def get_vocab(self):
        return self._tk.get_vocab(with_added_tokens=True)

    @property
    def vocab_size(self):
        return self._tk.get_vocab_size(with_added_tokens=True)

    def __len__(self):
        return self.vocab_size

    def apply_chat_template(self, messages, tools=None, tokenize=False, add_generation_prompt=True,
                            enable_thinking=False, **kwargs):
        if self._template is None:
            raise ValueError("checkpoint has no chat template")
        text = self._template.render(messages=messages, tools=tools, add_generation_prompt=add_generation_prompt,
                                     enable_thinking=enable_thinking, **self.special, **kwargs)
        return self.encode(text) if tokenize else text

    def __repr__(self):
        return f"KernelTokenizer(vocab={self.vocab_size}, tokenizers-based; transformers is not available)"
