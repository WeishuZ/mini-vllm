"""Tests for the contiguous-vs-paged capacity analysis."""
from mini_vllm import contiguous_capacity, paged_capacity


def test_paged_fits_more_than_contiguous():
    lens = [50] * 100
    cont = contiguous_capacity(total_slots=10_000, max_seq_len=2048, seq_lens=lens)
    paged = paged_capacity(num_blocks=10_000 // 16, block_size=16, seq_lens=lens)
    assert paged.fit > cont.fit
    assert paged.waste_fraction < cont.waste_fraction
    assert paged.utilization > cont.utilization


def test_contiguous_waste_matches_formula():
    # 4 sequences of length 50 fit when reserving 2048 each into 10k slots.
    cont = contiguous_capacity(total_slots=10_000, max_seq_len=2048, seq_lens=[50] * 100)
    assert cont.fit == 4
    assert cont.reserved_slots == 4 * 2048
    assert cont.used_slots == 4 * 50
    assert cont.wasted_slots == 4 * 2048 - 4 * 50


def test_paged_internal_fragmentation_bounded():
    # Paged waste is at most one block per sequence.
    block_size = 16
    lens = [17, 33, 1, 200, 65]
    paged = paged_capacity(num_blocks=10_000, block_size=block_size, seq_lens=lens)
    assert paged.fit == len(lens)
    assert paged.wasted_slots <= block_size * len(lens)
