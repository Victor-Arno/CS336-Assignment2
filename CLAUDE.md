# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is CS336 Spring 2025 Assignment 2: Systems - a course assignment for implementing optimized Transformer language model components including FlashAttention2, distributed data parallel (DDP) training, and optimizer state sharding.

## Build and Test Commands

```bash
# Run all tests
uv run pytest -v ./tests

# Run specific test files
uv run pytest -v tests/test_attention.py      # FlashAttention2 tests
uv run pytest -v tests/test_ddp.py            # Bucketed DDP tests
uv run pytest -v tests/test_ddp_individual_parameters.py  # Individual parameter DDP tests
uv run pytest -v tests/test_sharded_optimizer.py          # Sharded optimizer tests

# Run a single test
uv run pytest -v tests/test_attention.py::test_flash_forward_pass_pytorch

# Create submission
./test_and_make_submission.sh

# Verify cs336-basics is accessible
uv run python -c "import cs336_basics"
```

## Architecture

### Module Structure

- **cs336-basics/**: Staff implementation of the language model from assignment 1. Contains `cs336_basics` module with:
  - `model.py` - Transformer model implementation
  - `optimizer.py` - Optimizer implementations
  - `data.py` - Data utilities
  - `nn_utils.py` - Neural network utilities

- **cs336_systems/**: Empty module where assignment implementations go. All new code should be added here.

- **tests/**: Test suite with:
  - `adapters.py` - Defines the interface functions that must be implemented
  - `common.py` - Shared test utilities (ToyModel, DDP setup helpers)
  - `conftest.py` - Pytest fixtures including snapshot testing

### Required Implementations (via tests/adapters.py)

1. **FlashAttention2**
   - `get_flashattention_autograd_function_pytorch()` - PyTorch-only implementation
   - `get_flashattention_autograd_function_triton()` - Triton kernel implementation
   - Both must return `torch.autograd.Function` subclasses with forward/backward passes

2. **Distributed Data Parallel**
   - `get_ddp_individual_parameters(module)` - DDP with per-parameter gradient sync
   - `get_ddp_bucketed(module, bucket_size_mb)` - DDP with bucketed gradient sync
   - Associated `on_after_backward` and `on_train_batch_start` hooks

3. **Optimizer State Sharding**
   - `get_sharded_optimizer(params, optimizer_cls, **kwargs)` - ZeRO-style optimizer sharding

### Test Requirements

- FlashAttention implementations must save logsumexp tensor via `ctx.save_for_backward`
- DDP implementations must handle tied weights (see `ToyModelWithTiedWeights`)
- Triton tests require CUDA and are skipped on CPU-only systems
