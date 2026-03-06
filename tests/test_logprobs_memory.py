"""
Test memory-efficient logprobs: gather + logsumexp vs original F.log_softmax.
"""
import torch
import torch.nn.functional as F


def original_get_action_logprobs(logits, input_ids, attention_mask, state_lens):
    """Original: full log_softmax (OOM-prone)."""
    labels = input_ids
    attention_mask = attention_mask.bool()
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    logprobs = F.log_softmax(shift_logits, dim=-1)
    logprobs = logprobs.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
    a_starts = [state_len - 1 for state_len in state_lens]
    logprobs = [
        logprobs[b_idx][attention_mask[b_idx, 1:]][start_idx:].tolist()
        for b_idx, start_idx in enumerate(a_starts)
    ]
    state_action_tokens = [
        input_ids[b_idx][attention_mask[b_idx]].tolist()
        for b_idx, start_idx in enumerate(a_starts)
    ]
    return logprobs, state_action_tokens


def efficient_get_action_logprobs(logits, input_ids, attention_mask, state_lens):
    """New: gather + logsumexp on original dtype, upcast only (B,L) results to fp32."""
    labels = input_ids
    attention_mask = attention_mask.bool()
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    gathered_logits = shift_logits.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
    log_normalizer = shift_logits.logsumexp(dim=-1)
    logprobs = gathered_logits.float() - log_normalizer.float()
    a_starts = [state_len - 1 for state_len in state_lens]
    logprobs = [
        logprobs[b_idx][attention_mask[b_idx, 1:]][start_idx:].tolist()
        for b_idx, start_idx in enumerate(a_starts)
    ]
    state_action_tokens = [
        input_ids[b_idx][attention_mask[b_idx]].tolist()
        for b_idx, start_idx in enumerate(a_starts)
    ]
    return logprobs, state_action_tokens


def test_fp32_exact_match():
    """In fp32, both methods should produce identical results."""
    torch.manual_seed(42)
    B, L, V = 2, 32, 1000
    logits = torch.randn(B, L, V)
    input_ids = torch.randint(0, V, (B, L))
    attention_mask = torch.ones(B, L, dtype=torch.long)
    attention_mask[1, -5:] = 0
    state_lens = [8, 12]

    orig, _ = original_get_action_logprobs(logits, input_ids, attention_mask, state_lens)
    eff, _ = efficient_get_action_logprobs(logits, input_ids, attention_mask, state_lens)

    for b in range(B):
        assert len(orig[b]) == len(eff[b])
        for i, (o, e) in enumerate(zip(orig[b], eff[b])):
            assert abs(o - e) < 1e-5, f"fp32 mismatch at batch {b}, pos {i}: diff={abs(o-e)}"
    print("PASS: test_fp32_exact_match")


def test_fp32_large_vocab():
    """fp32 with Qwen3 vocab size."""
    torch.manual_seed(123)
    B, L, V = 1, 64, 152064
    logits = torch.randn(B, L, V, dtype=torch.float32)
    input_ids = torch.randint(0, V, (B, L))
    attention_mask = torch.ones(B, L, dtype=torch.long)
    state_lens = [10]

    orig, _ = original_get_action_logprobs(logits, input_ids, attention_mask, state_lens)
    eff, _ = efficient_get_action_logprobs(logits, input_ids, attention_mask, state_lens)

    max_diff = max(abs(o - e) for o, e in zip(orig[0], eff[0]))
    assert max_diff < 1e-4, f"fp32 large vocab max diff: {max_diff}"
    print(f"PASS: test_fp32_large_vocab (max_diff={max_diff:.2e})")


def test_bf16_precision():
    """With bf16 inputs, verify our method has acceptable precision.
    logsumexp on bf16 uses max-subtraction for stability, then we upcast
    the (B,L) scalars to fp32 for the final subtraction."""
    torch.manual_seed(123)
    B, L, V = 1, 64, 152064
    logits_bf16 = torch.randn(B, L, V, dtype=torch.bfloat16)
    input_ids = torch.randint(0, V, (B, L))
    attention_mask = torch.ones(B, L, dtype=torch.long)
    state_lens = [10]

    # Ground truth: fp32 throughout
    logits_fp32 = logits_bf16.float()
    gt, _ = original_get_action_logprobs(logits_fp32, input_ids, attention_mask, state_lens)

    orig, _ = original_get_action_logprobs(logits_bf16, input_ids, attention_mask, state_lens)
    eff, _ = efficient_get_action_logprobs(logits_bf16, input_ids, attention_mask, state_lens)

    orig_err = max(abs(g - o) for g, o in zip(gt[0], orig[0]))
    eff_err = max(abs(g - e) for g, e in zip(gt[0], eff[0]))

    # Both have bf16-level error since inputs are bf16; ours should be comparable or better
    assert eff_err < 0.1, f"Efficient error vs ground truth too large: {eff_err}"
    print(f"PASS: test_bf16_precision (orig_err={orig_err:.4f}, eff_err={eff_err:.4f})")


def test_batch_configs():
    """Test various batch sizes, sequence lengths, padding patterns."""
    torch.manual_seed(7)
    configs = [
        (1, 16, 500, [5]),
        (4, 64, 1000, [10, 20, 30, 15]),
        (1, 128, 2000, [64]),
        (3, 32, 800, [5, 10, 8]),
    ]
    for B, L, V, state_lens in configs:
        logits = torch.randn(B, L, V)
        input_ids = torch.randint(0, V, (B, L))
        attention_mask = torch.ones(B, L, dtype=torch.long)
        for b in range(B):
            pad = torch.randint(0, L // 4, (1,)).item()
            attention_mask[b, L - pad:] = 0

        orig, _ = original_get_action_logprobs(logits, input_ids, attention_mask, state_lens)
        eff, _ = efficient_get_action_logprobs(logits, input_ids, attention_mask, state_lens)

        for b in range(B):
            assert len(orig[b]) == len(eff[b])
            for o, e in zip(orig[b], eff[b]):
                assert abs(o - e) < 1e-5
    print(f"PASS: test_batch_configs ({len(configs)} configurations)")


def test_memory_savings():
    """Verify efficient version uses less peak GPU memory."""
    if not torch.cuda.is_available():
        print("SKIP: test_memory_savings (no CUDA)")
        return

    torch.manual_seed(0)
    B, L, V = 1, 2048, 152064
    device = "cuda"

    logits = torch.randn(B, L, V, dtype=torch.bfloat16, device=device)
    input_ids = torch.randint(0, V, (B, L), device=device)
    attention_mask = torch.ones(B, L, dtype=torch.long, device=device)
    state_lens = [64]

    # Efficient first (less memory, won't pollute allocator)
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    eff_logprobs, _ = efficient_get_action_logprobs(logits, input_ids, attention_mask, state_lens)
    eff_peak = torch.cuda.max_memory_allocated() - baseline
    del eff_logprobs
    torch.cuda.empty_cache()

    # Original
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    orig_logprobs, _ = original_get_action_logprobs(logits, input_ids, attention_mask, state_lens)
    orig_peak = torch.cuda.max_memory_allocated() - baseline
    del orig_logprobs
    torch.cuda.empty_cache()

    savings_pct = (1 - eff_peak / orig_peak) * 100
    print(f"PASS: test_memory_savings (orig={orig_peak/1e9:.2f}GB, eff={eff_peak/1e9:.2f}GB, savings={savings_pct:.1f}%)")
    assert eff_peak < orig_peak, f"Expected memory savings: eff={eff_peak}, orig={orig_peak}"


if __name__ == "__main__":
    test_fp32_exact_match()
    test_fp32_large_vocab()
    test_bf16_precision()
    test_batch_configs()
    test_memory_savings()
    print("\nAll tests passed!")
