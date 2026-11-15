"""R.2 teacher-forced engine: Qwen3.5-0.8B in HF transformers on the GPU, resident in the harness server.

Used for teacher-forced passes and for the activations of generation passes. Residual stream index l
(0..24) is the embedding output (l = 0) or the output of decoder block l-1 (before the final norm); MLP
activations of block b (0..23) are the inputs of its down_proj (the SwiGLU product); attention probabilities
come from the full-attention blocks (3, 7, 11, 15, 19, 23). Inputs are right-padded, so causal outputs at
real positions are exact. Edits (name -> float32 array) are copied into the bf16 parameters for a pass and
restored afterwards.
"""
import numpy as np
import torch

from exp_r.r2.common import FULL_ATTN, HIDDEN, MLP_WIDTH, N_LAYERS


class TFEngine:
    def __init__(self, ckpt, batch_tokens=32768):
        from transformers import AutoModelForImageTextToText
        self.model = AutoModelForImageTextToText.from_pretrained(ckpt, dtype=torch.bfloat16, device_map="cuda")
        self.model.eval()
        self.lm = self.model.model.language_model
        self.head = self.model.lm_head
        self.params = dict(self.model.named_parameters())
        self.saved = {}             # name -> original GPU tensor (for restore)
        self.batch_tokens = batch_tokens
        tc = getattr(self.model.config, "text_config", self.model.config)
        self.n_heads = int(tc.num_attention_heads)
        assert len(self.lm.layers) == N_LAYERS

    # ------------------------------------------------------------ edits
    @torch.no_grad()
    def set_edits(self, edits: dict):
        for n, t in self.saved.items():
            self.params[n].data.copy_(t)
        self.saved = {}
        for n, a in edits.items():
            p = self.params[n]
            self.saved[n] = p.data.clone()
            p.data.copy_(torch.from_numpy(a).to(p.dtype))
        if "model.language_model.embed_tokens.weight" in edits:      # never: frozen; tied head guard
            raise RuntimeError("embedding edits are not allowed")

    # ------------------------------------------------------------ core forward with captures
    @torch.no_grad()
    def _forward(self, batch_ids, hidden=(), mlp=(), attn=(), want_logits=False, topk=0):
        """batch_ids: list of lists (right-padded here). Returns dict of CPU tensors per request."""
        B, T = len(batch_ids), max(len(x) for x in batch_ids)
        inp = torch.zeros((B, T), dtype=torch.long)
        for i, x in enumerate(batch_ids):
            inp[i, :len(x)] = torch.tensor(x)
        inp = inp.cuda()
        cap, hooks = {}, []

        def keep(key):
            def h(_m, _i, out):
                cap[key] = (out[0] if isinstance(out, tuple) else out).detach()
            return h

        def keep_in(key):
            def h(_m, args):
                cap[key] = args[0].detach()
            return h

        try:
            if 0 in hidden:
                hooks.append(self.lm.embed_tokens.register_forward_hook(keep("h0")))
            for l in hidden:
                if l >= 1:
                    hooks.append(self.lm.layers[l - 1].register_forward_hook(keep(f"h{l}")))
            for b in mlp:
                hooks.append(self.lm.layers[b].mlp.down_proj.register_forward_pre_hook(keep_in(f"m{b}")))
            kw = {}
            if attn:
                self.model.set_attn_implementation("eager")
                kw["output_attentions"] = True
            out = self.lm(input_ids=inp, use_cache=False, **kw)
        finally:
            for h in hooks:
                h.remove()
            if attn:
                self.model.set_attn_implementation("sdpa")
        res = {k: v for k, v in cap.items()}
        if attn:
            att = out.attentions
            att = list(att or [])
            if len(att) == N_LAYERS:
                by = {b: att[b] for b in FULL_ATTN}
            elif len(att) == len(FULL_ATTN):              # only the full-attention layers report weights
                by = dict(zip(FULL_ATTN, att))
            else:
                by = {b: x for b, x in zip(FULL_ATTN, [x for x in att if x is not None])}
            for b in attn:
                a = by.get(b)
                if a is None:
                    raise RuntimeError(f"no attention probabilities returned for block {b}")
                res[f"a{b}"] = a.detach()
        if want_logits or topk:
            hN = out.last_hidden_state                       # after the final norm
            W = self.head.weight
            lp_tok = torch.zeros((B, T - 1), dtype=torch.float32)
            tk_v = torch.zeros((B, T, topk), dtype=torch.float32) if topk else None
            tk_i = torch.zeros((B, T, topk), dtype=torch.int32) if topk else None
            step = max(1, 4096 // B)                         # B x step x vocab float32 per chunk
            for s in range(0, T, step):
                e = min(T, s + step)
                lg = torch.log_softmax((hN[:, s:e] @ W.T).float(), dim=-1)        # [B, c, V]
                if want_logits:
                    nxt = inp[:, s + 1:e + 1]
                    c = nxt.shape[1]
                    lp_tok[:, s:s + c] = lg[:, :c].gather(-1, nxt.unsqueeze(-1)).squeeze(-1).cpu()
                if topk:
                    v, i = lg.topk(topk, dim=-1)
                    tk_v[:, s:e], tk_i[:, s:e] = v.cpu(), i.int().cpu()
                del lg
            res["lp"], res["tkv"], res["tki"] = lp_tok, tk_v, tk_i
        return res

    def _batches(self, lengths, max_batch_tokens=None):
        cap = max_batch_tokens or self.batch_tokens
        order = sorted(range(len(lengths)), key=lambda i: -lengths[i])
        b, out = [], []
        for i in order:
            if b and (len(b) + 1) * lengths[b[0]] > cap:
                out.append(b)
                b = []
            b.append(i)
        if b:
            out.append(b)
        return out

    # ------------------------------------------------------------ teacher-forced pass
    def teacher_forced(self, ids, hidden=(), mlp=(), attn=(), token_logprobs=False, topk=0):
        """ids: list of token-id lists. Returns numpy arrays right-padded to the longest text."""
        n, T = len(ids), max(len(x) for x in ids)
        lens = [len(x) for x in ids]
        arr = {"input_ids": np.full((n, T), -1, np.int64)}
        for i, x in enumerate(ids):
            arr["input_ids"][i, :len(x)] = x
        if token_logprobs:
            arr["token_logprobs"] = np.zeros((n, T - 1), np.float32)
        if topk:
            arr["topk_logprobs"] = np.zeros((n, T, topk), np.float32)
            arr["topk_ids"] = np.full((n, T, topk), -1, np.int32)
        for l in hidden:
            arr[f"hidden_{l}"] = np.zeros((n, T, HIDDEN), np.float32)
        for b in mlp:
            arr[f"mlp_act_{b}"] = np.zeros((n, T, MLP_WIDTH), np.float32)
        for b in attn:
            arr[f"attn_{b}"] = np.zeros((n, self.n_heads, T, T), np.float32)
        mean_lp = [None] * n
        for bt in self._batches(lens):
            r = self._forward([ids[i] for i in bt], hidden, mlp, attn, token_logprobs or True, topk)
            for j, i in enumerate(bt):
                L = lens[i]
                lp = r["lp"][j, :L - 1].numpy()
                mean_lp[i] = float(lp.mean()) if L > 1 else None
                if token_logprobs:
                    arr["token_logprobs"][i, :L - 1] = lp
                if topk:
                    arr["topk_logprobs"][i, :L] = r["tkv"][j, :L].numpy()
                    arr["topk_ids"][i, :L] = r["tki"][j, :L].numpy()
                for l in hidden:
                    arr[f"hidden_{l}"][i, :L] = r[f"h{l}"][j, :L].float().cpu().numpy()
                for b in mlp:
                    arr[f"mlp_act_{b}"][i, :L] = r[f"m{b}"][j, :L].float().cpu().numpy()
                for b in attn:
                    arr[f"attn_{b}"][i, :, :L, :L] = r[f"a{b}"][j, :, :L, :L].float().cpu().numpy()
            del r
            torch.cuda.empty_cache()
        return arr, lens, mean_lp

    @staticmethod
    def tf_bytes(lens, hidden, mlp, attn, token_logprobs, topk, heads=8):
        n, T = len(lens), max(lens)
        b = n * T * 8 + (n * (T - 1) * 4 if token_logprobs else 0) + n * T * topk * 8
        b += n * T * 4 * (HIDDEN * len(hidden) + MLP_WIDTH * len(mlp)) + n * heads * T * T * 4 * len(attn)
        return b

    # ------------------------------------------------------------ activations of generated answers
    def gen_activations(self, prompt_ids, answer_ids, layers, mlp_on):
        """Residuals (and MLP down_proj inputs of block l-1) at the last prompt token and averaged over the
        answer's first tokens. Returns dict of [N, k, d] float32 arrays."""
        N = len(prompt_ids)
        blocks = [l - 1 for l in layers if l >= 1] if mlp_on else []
        out = {"hidden_last_prompt": np.zeros((N, len(layers), HIDDEN), np.float32),
               "hidden_mean_answer": np.full((N, len(layers), HIDDEN), np.nan, np.float32)}
        if blocks:
            out["mlp_last_prompt"] = np.zeros((N, len(blocks), MLP_WIDTH), np.float32)
            out["mlp_mean_answer"] = np.full((N, len(blocks), MLP_WIDTH), np.nan, np.float32)
        seqs = [p + a for p, a in zip(prompt_ids, answer_ids)]
        for bt in self._batches([len(s) for s in seqs]):
            r = self._forward([seqs[i] for i in bt], hidden=layers, mlp=blocks)
            for j, i in enumerate(bt):
                P, A = len(prompt_ids[i]), len(answer_ids[i])
                for k, l in enumerate(layers):
                    h = r[f"h{l}"][j]
                    out["hidden_last_prompt"][i, k] = h[P - 1].float().cpu().numpy()
                    if A:
                        out["hidden_mean_answer"][i, k] = h[P:P + A].float().mean(0).cpu().numpy()
                for k, b in enumerate(blocks):
                    m = r[f"m{b}"][j]
                    out["mlp_last_prompt"][i, k] = m[P - 1].float().cpu().numpy()
                    if A:
                        out["mlp_mean_answer"][i, k] = m[P:P + A].float().mean(0).cpu().numpy()
            del r
            torch.cuda.empty_cache()
        return out, blocks
