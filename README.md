# PyTorch 2.0 Implementation & Optimization of SONA
An open-source PyTorch implementation and performance optimization of the SONA (Semantic Outlier via Nuisance Awareness) diffusion outlier generation algorithm based on the paper:
> **Semantic Outlier via Nuisance Awareness**  
> Paper: [arXiv:2408.14841](http://arxiv.org/abs/2408.14841) (Choi et al.)

## Key Features
- **Independent Implementation**: Implemented from scratch in PyTorch & HuggingFace Diffusers.
- **3.22x Speedup Optimization**: Engineered with batched UNet conditioning, prompt caching, `@torch.inference_mode()`, and `torch.compile` (CUDA Graphs).
- **Verified Quality**: 42.14 dB PSNR equivalence.

## License
[MIT License](LICENSE)

*All credit for the original SONA algorithm and formulation belongs to the original paper authors and research labs.*
