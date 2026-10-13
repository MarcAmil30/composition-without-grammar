"""
Model adapters for the upstream-only benchmark.

Every adapter exposes the same two things:

    load()                         -> pull weights onto the GPU
    continue_(prompts, n, cfg)     -> for each prompt, return cfg.n_samples
                                      sampled continuations and (optionally) one
                                      greedy continuation, each n bp long

Sampling is defined ONCE, in SamplingConfig, and is identical for every model.
Adapters must not override it -- per-model differences belong in framing.py.

Public checkpoints can be resolved through the Hugging Face cache. The fine-tuned
checkpoint is supplied through ``FINETUNED_EVO_CKPT`` (or the legacy
``PROK_CKPT`` environment variable):
    togethercomputer/evo-1-131k-base          Evo 1
    togethercomputer/evo-1-8k-base            (alternate)
    arcinstitute/evo2_1b_base                 Evo 2
    jinyuan22/promogen2-base                  PromoGen2
    LongSafari/hyenadna-medium-450k-seqlen-hf HyenaDNA
    LongSafari/hyenadna-small-32k-seqlen-hf   (alternate)

On an offline compute node, populate the cache before submitting a job and set
``HF_HUB_OFFLINE=1`` so a missing dependency fails loudly.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, asdict, field

# ---------------------------------------------------------------------
# Sampling -- fixed before generation, identical across all five models.
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class SamplingConfig:
    n_samples: int = 20
    temperature: float = 1.0
    top_k: int = 0            # 0 = disabled; full distribution
    top_p: float = 1.0        # 1.0 = disabled
    greedy: bool = True       # one extra deterministic decode per prompt
    seed: int = 0
    batch_size: int = 16

    def as_dict(self):
        return asdict(self)


DEFAULT_SAMPLING = SamplingConfig()

# alphabet the generations are cleaned to before anything is written out
ALPHABET = set("ACGT")


# ---------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------
@dataclass
class ModelSpec:
    key: str
    display: str
    checkpoint: str
    loader: str                       # which Adapter subclass to use
    tasks: tuple = ("generate_promoter", "generate_sd")
    notes: str = ""
    kwargs: dict = field(default_factory=dict)
    seqs_per_forward: int = 256   # 7B models need a much smaller number


REGISTRY = {
    "evo1": ModelSpec(
        "evo1", "Evo 1", "togethercomputer/evo-1-8k-base", "hf_causal",
        notes="StripedHyena 7B, prokaryotic corpus; nucleotide-level. "
              "The 8k-context base rather than evo-1-131k-base: StripedHyena "
              "sizes its KV cache to the model's max context, so the 131k "
              "variant asks for ~60 GiB at any usable batch size. Same model, "
              "same OpenGenome corpus; 8k is ample for 225 bp inputs.",
        kwargs=dict(trust_remote_code=True), seqs_per_forward=8,
    ),
    "evo2": ModelSpec(
        "evo2", "Evo 2", "arcinstitute/evo2_1b_base", "evo2",
        notes="StripedHyena 2. Loaded through the `evo2` package, not plain HF. "
              "Run inside the NGC venv (ngc_evo2_venv / evo2_env).",
    ),
    "prok": ModelSpec(
        "prok", "PROK", os.environ.get(
            "FINETUNED_EVO_CKPT", os.environ.get("PROK_CKPT", "")),
        "hf_causal",
        notes="Evo 1 fine-tuned with unmasked next-token cross-entropy. "
              "Prompt framing is defined in framing.py.",
        kwargs=dict(trust_remote_code=True), seqs_per_forward=32,
    ),
    "hyenadna": ModelSpec(
        "hyenadna", "HyenaDNA", "LongSafari/hyenadna-medium-450k-seqlen-hf", "hf_causal",
        notes="hg38 only, single-character ACGT tokenizer. Used AS-IS, not finetuned. "
              "This is HyenaDNA's top capacity tier -- n_layer 8, d_model 256, "
              "d_inner 1024, identical to large-1m; only max_seq_len differs "
              "(450k vs 1M), which is irrelevant at 220 bp.",
        kwargs=dict(trust_remote_code=True),
    ),
    "promogen2": ModelSpec(
        "promogen2", "PromoGen2", "jinyuan22/promogen2-base", "hf_causal",
        notes="GPT-2, n_positions=256, trained on 160bp windows ending at the start "
              "codon. Reframed to that window by framing.py; 39 Task-1 genes dropped. "
              "Emits EOS at the start codon -- see PROCEDURE.md Task 2 metric 3.",
    ),
}


# ---------------------------------------------------------------------
# Adapters
# ---------------------------------------------------------------------
class Adapter:
    """Base adapter. Subclasses implement _load and _generate_batch."""

    def __init__(self, spec: ModelSpec, device: str = "cuda"):
        self.spec = spec
        self.device = device
        self.model = None
        self.tok = None

    # -- lifecycle ----------------------------------------------------
    def load(self):
        self._load()
        return self

    def _load(self):
        raise NotImplementedError

    # -- generation ---------------------------------------------------
    def continue_(self, prompts, n_bp, cfg: SamplingConfig):
        """
        prompts : list[str], ALL THE SAME LENGTH (generate.py guarantees this)
        returns : list[dict] with keys `greedy` (str | None) and
                  `samples` (list[str]), one entry per prompt, each string
                  exactly the continuation -- prompt stripped, uppercased,
                  non-ACGT removed.

        Every (prompt, decode) pair is one independent sequence, so they are
        tiled into a single flat job list and pushed through in wide forward
        passes. Doing one pass per sample index instead would waste most of the
        GPU on a batch of one whenever a prompt length is unique -- which, for
        task 1, is most of them.
        """
        out = [{"greedy": None, "samples": [None] * cfg.n_samples} for _ in prompts]

        jobs = []                                   # (prompt_idx, sample_idx | -1)
        if cfg.greedy:
            jobs += [(i, -1) for i in range(len(prompts))]
        for s_i in range(cfg.n_samples):
            jobs += [(i, s_i) for i in range(len(prompts))]

        width = max(1, self.spec.seqs_per_forward)
        for k in range(0, len(jobs), width):
            batch = jobs[k:k + width]
            # a forward pass is all-greedy or all-sampled; the job list is
            # ordered so that only one boundary batch can ever be mixed
            for is_greedy in (True, False):
                sel = [(i, s_i) for (i, s_i) in batch
                       if (s_i == -1) == is_greedy]
                if not sel:
                    continue
                gen = self._generate_batch([prompts[i] for i, _ in sel], n_bp, cfg,
                                           greedy=is_greedy,
                                           sample_idx=sel[0][1] if not is_greedy else 0)
                for (i, s_i), g in zip(sel, gen):
                    if s_i == -1:
                        out[i]["greedy"] = g
                    else:
                        out[i]["samples"][s_i] = g

        for o in out:
            o["samples"] = [x for x in o["samples"] if x is not None]
        return out

    def _generate_batch(self, prompts, n_bp, cfg, greedy, sample_idx=0):
        raise NotImplementedError

    # -- helpers ------------------------------------------------------
    @staticmethod
    def _clean(text: str) -> str:
        return "".join(c for c in text.upper() if c in ALPHABET)

    def _decode_positional(self, token_ids) -> str:
        """Decode one token per character, so position i of the output is the
        i-th generated token.

        Deleting non-ACGT characters instead would slide every downstream base
        left, and every metric in this benchmark is a position. PROK in
        particular emits its segment delimiters mid-sequence. Anything that is
        not a single A/C/G/T becomes 'N', which the PWM scanner skips.
        """
        out = []
        for t in token_ids.tolist():
            c = self.tok.decode([t], skip_special_tokens=False).strip().upper()
            out.append(c if len(c) == 1 and c in ALPHABET else "N")
        return "".join(out)


class HFCausalAdapter(Adapter):
    """Anything that loads with AutoModelForCausalLM + AutoTokenizer.

    Covers Evo 1, PROK, HyenaDNA and PromoGen2. Their tokenizers are all
    nucleotide- or byte-level, so one token is one base and `max_new_tokens`
    is a bp count.
    """

    def _load(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        kw = dict(self.spec.kwargs)
        self.tok = AutoTokenizer.from_pretrained(self.spec.checkpoint, **kw)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.spec.checkpoint, torch_dtype=torch.bfloat16, **kw
        ).to(self.device).eval()

        # No padding is ever used -- generate.py batches only equal-length
        # prompts (see `batched_by_length`). That matters for two reasons:
        # Evo's ByteTokenizer has no pad token at all, and right-padding would
        # make a causal model continue *after* the pads for the shorter prompts
        # in a batch, silently corrupting their output.

        # HyenaDNA's remote modeling code does not inherit GenerationMixin, so
        # transformers' .generate is unavailable. Detect it once and use the
        # manual sampling loop below -- same distribution, just written out.
        self._has_generate = hasattr(self.model, "generate")
        if not self._has_generate:
            print(f"  [{self.spec.key}] no .generate -- using the manual sampling loop")

        # one token == one base, or every metric offset is wrong
        probe = self.tok("ACGTACGTAC", add_special_tokens=False)["input_ids"]
        assert len(probe) == 10, (
            f"{self.spec.key}: tokenizer is not single-nucleotide "
            f"(10 bases -> {len(probe)} tokens). Offsets would be meaningless."
        )

    def _generate_batch(self, prompts, n_bp, cfg, greedy, sample_idx=0):
        import torch

        if not self._has_generate:
            return self._manual(prompts, n_bp, cfg, greedy, sample_idx)

        assert len(set(map(len, prompts))) == 1, "batch must be equal-length"
        enc = self.tok(list(prompts), return_tensors="pt",
                       add_special_tokens=False).to(self.device)
        # Evo's ByteTokenizer returns token_type_ids, which StripedHyena's
        # forward does not accept; .generate then rejects it as an unused kwarg.
        enc = {k: v for k, v in enc.items() if k in ("input_ids", "attention_mask")}
        gen_kw = dict(max_new_tokens=n_bp)
        if self.tok.pad_token_id is not None:
            gen_kw["pad_token_id"] = self.tok.pad_token_id
        if greedy:
            gen_kw.update(do_sample=False)
        else:
            gen_kw.update(do_sample=True, temperature=cfg.temperature)
            if cfg.top_k:
                gen_kw["top_k"] = cfg.top_k
            if cfg.top_p < 1.0:
                gen_kw["top_p"] = cfg.top_p
            torch.manual_seed(cfg.seed * 100_003 + sample_idx)

        with torch.inference_mode():
            out = self.model.generate(**enc, **gen_kw)

        # every prompt in the batch is the same length, so this slice is exact
        n_in = enc["input_ids"].shape[1]
        return [self._decode_positional(row[n_in:]) for row in out]


    def _manual(self, prompts, n_bp, cfg, greedy, sample_idx):
        """Plain autoregressive decoding for models whose remote code does not
        expose .generate (HyenaDNA).

        Batched over the whole equal-length group. No KV cache -- a full forward
        per step is cheap for a 6.6M-parameter model and avoids depending on
        cache semantics we cannot verify against this checkpoint.
        """
        import torch

        enc = self.tok(list(prompts), return_tensors="pt",
                       add_special_tokens=False)
        ids = enc["input_ids"].to(self.device)
        n_in = ids.shape[1]
        gen = torch.Generator(device="cpu")
        gen.manual_seed(cfg.seed * 100_003 + sample_idx * 9_176 + n_in)

        with torch.inference_mode():
            for _ in range(n_bp):
                logits = self.model(ids).logits[:, -1, :].float()
                if greedy:
                    nxt = logits.argmax(-1, keepdim=True)
                else:
                    probs = torch.softmax(logits / max(cfg.temperature, 1e-6), dim=-1)
                    nxt = torch.multinomial(probs.cpu(), 1, generator=gen).to(self.device)
                ids = torch.cat([ids, nxt], dim=1)

        return [self._decode_positional(row[n_in:]) for row in ids]


class Evo2Adapter(Adapter):
    """Evo 2 ships its own package with a non-HF generate API.

    Kept separate so its quirks cannot leak into the HF path. Verify against the
    installed `evo2` version on first smoke test -- the API has moved between
    releases.
    """

    def _load(self):
        from evo2 import Evo2
        name = self.spec.checkpoint.split("/")[-1]
        self.model = Evo2(name)

    def _generate_batch(self, prompts, n_bp, cfg, greedy, sample_idx=0):
        # vortex divides the logits by temperature before softmax, so a literal
        # 0.0 produces inf -> NaN -> a device-side assert inside multinomial.
        # top_k=1 with a near-zero temperature is the same argmax, safely.
        out = self.model.generate(
            prompt_seqs=list(prompts),
            n_tokens=n_bp,
            temperature=1e-4 if greedy else cfg.temperature,
            top_k=1 if greedy else (cfg.top_k or 4),
            top_p=cfg.top_p,
        )
        seqs = getattr(out, "sequences", out)
        return [self._clean(s) for s in seqs]


LOADERS = {"hf_causal": HFCausalAdapter, "evo2": Evo2Adapter}


def build(model_key: str, device: str = "cuda") -> Adapter:
    if model_key == "prok" and not REGISTRY[model_key].checkpoint:
        raise ValueError(
            "Set FINETUNED_EVO_CKPT to the fine-tuned Evo checkpoint directory."
        )
    spec = REGISTRY[model_key]
    return LOADERS[spec.loader](spec, device=device)


if __name__ == "__main__":
    print(f"{'key':11s} {'tasks':2s}  checkpoint")
    for k, s in REGISTRY.items():
        print(f"{k:11s} {len(s.tasks)}   {s.checkpoint}")
        print(f"{'':11s}     {s.notes}")
    print("\nsampling:", DEFAULT_SAMPLING.as_dict())
