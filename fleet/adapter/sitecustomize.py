"""Install storage adapter in every serving worker, only when explicitly enabled."""
import importlib.abc
import importlib.machinery
import os
import sys

class EngramLoader(importlib.abc.Loader):
    def __init__(self, original):
        self.original = original

    def create_module(self, spec):
        return self.original.create_module(spec)

    def exec_module(self, module):
        self.original.exec_module(module)
        if module.__name__ == 'sglang.srt.layers.engram':
            from engram_backend import install
            install(module)
            # batch1 2026-09-25: row lookups on a side stream after the hasher.
            # After engram_backend (it sets _owned_rows/_staging/_works/_store).
            # Gated on DSV41_ENGRAM_PREFETCH.
            from engram_prefetch import install as install_engram_prefetch
            install_engram_prefetch(module)
        elif module.__name__ == 'sglang.srt.layers.quantization.fp8_utils':
            from mxfp8_b12x import install
            install(module)
        elif module.__name__ == 'sglang.srt.layers.quantization.fp8':
            from mxfp8_b12x import install_fp8
            install_fp8(module)
        elif module.__name__ == 'sglang.srt.model_executor.model_runner':
            from prefill_empty_cache import install
            install(module)
        elif module.__name__ == 'sglang.srt.entrypoints.openai.encoding_dsv41':
            from encoding_compat import install_encoder
            install_encoder(module)
        elif module.__name__ == 'sglang.srt.entrypoints.openai.serving_chat':
            from encoding_compat import install_serving_chat
            install_serving_chat(module)
        elif module.__name__ == 'sglang.srt.managers.schedule_batch':
            from loop_abort import install as install_loop_abort
            install_loop_abort(module)
        elif module.__name__ == 'sglang.srt.model_executor.runner.flashinfer_autotune':
            # batch1 2026-09-25: keep the FlashInfer autotune cache across boots under EP
            # (sglang#40320: the stock gate deletes it on every boot, and under EP the
            # per-rank caches can never agree). Gated on DSV41_AUTOTUNE_KEEP.
            if os.environ.get('DSV41_AUTOTUNE_KEEP', '0').strip() not in ('0', 'off', 'false', ''):
                from autotune_keep import install as install_autotune_keep
                install_autotune_keep(module)
        elif module.__name__ == 'sglang.srt.models.deepseek_v4':
            # batch2/batch3 2026-09-25 (upstream tp3-overnight-decode, knapcio TP4 fork).
            if os.environ.get('DSV41_WO_A_W8', '0').strip() not in ('0', 'off', 'false', ''):
                from wo_a_w8 import install_model as install_wo_a_w8
                install_wo_a_w8(module)
            if os.environ.get('DSV41_VERIFY_CAP', '').strip() not in ('', '0', 'off'):
                from verify_cap import install_model as install_verify_cap_model
                install_verify_cap_model(module)
        elif module.__name__ == 'sglang.srt.models.deepseek_v4_dspark':
            # conf:T needs the draft confidence head, which the engine builds only for
            # ragged verify; install_dspark builds it in static mode too.
            if os.environ.get('DSV41_VERIFY_CAP', '').strip().startswith('conf:'):
                from verify_cap import install_dspark as install_verify_cap_dspark
                install_verify_cap_dspark(module)
            if os.environ.get('DSV41_WO_A_W8', '0').strip() not in ('0', 'off', 'false', ''):
                from wo_a_w8 import install_dspark as install_wo_a_w8_dspark
                install_wo_a_w8_dspark(module)
            if os.environ.get('DSV41_DRAFT_HEAD_FP8', '0').strip() not in ('0', 'off', 'false', ''):
                from draft_head_fp8 import install as install_draft_head_fp8
                install_draft_head_fp8(module)
        elif module.__name__ == 'sglang.srt.speculative.dspark_components.dspark_draft_sampler':
            if os.environ.get('DSV41_DRAFT_TAU', '1').strip() not in ('', '1', '1.0'):
                from draft_tau import install as install_draft_tau
                install_draft_tau(module)
        elif module.__name__ == 'sglang.kernels.ops.speculative.dspark.dspark_accept':
            if os.environ.get('DSV41_BLOCK_VERIFY', '0').strip() not in ('0', 'off', 'false', ''):
                from block_verify import install as install_block_verify
                install_block_verify(module)
        elif module.__name__ == 'sglang.srt.speculative.dspark_components.dspark_verify':
            if os.environ.get('DSV41_FOLDED_FENCE', '0').strip() not in ('0', 'off', 'false', ''):
                from folded_result_fence import install as install_folded_fence
                install_folded_fence(module)
            if os.environ.get('DSV41_VERIFY_CAP', '').strip() not in ('', '0', 'off'):
                from verify_cap import install_verify as install_verify_cap_verify
                install_verify_cap_verify(module)
        elif module.__name__ == 'sglang.kernels.ops.moe.moe_fused_gate':
            if os.environ.get('DSV41_VERIFY_CAP', '').strip() not in ('', '0', 'off'):
                from verify_cap import install_gate as install_verify_cap_gate
                install_verify_cap_gate(module)
        elif module.__name__ == 'sglang.srt.speculative.dspark_components.dspark_draft':
            if os.environ.get('DSV41_VERIFY_CAP', '').strip() not in ('', '0', 'off'):
                from verify_cap import install_draft as install_verify_cap_draft
                install_verify_cap_draft(module)
        elif module.__name__ == 'sglang.srt.speculative.dspark_components.dspark_planner':
            if os.environ.get('DSV41_VERIFY_CAP', '').strip() not in ('', '0', 'off'):
                from verify_cap import install_planner as install_verify_cap_planner
                install_verify_cap_planner(module)
        else:
            # V4.1 ratio-1/2 indexers always call the FP4 DeepGEMM kernel.
            # SM120 needs its split-128 planner even when the legacy FP8
            # indexer uses the torch path. The upstream guard misses this case.
            cls = module.PagedIndexerMetadata
            original = cls.__post_init__
            def post_init(self):
                sm12 = bool(getattr(module, '_IS_SM120', False) or
                            getattr(module, '_IS_SM121', False))
                if sm12 and self.compress_ratio in (1, 2):
                    self.force_deep_gemm_metadata = True
                original(self)
            cls.__post_init__ = post_init

class EngramFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname not in ('sglang.srt.layers.engram',
                            'sglang.srt.layers.quantization.fp8_utils',
                            'sglang.srt.layers.quantization.fp8',
                            'sglang.srt.model_executor.model_runner',
                            'sglang.srt.entrypoints.openai.encoding_dsv41',
                            'sglang.srt.entrypoints.openai.serving_chat',
                            'sglang.srt.managers.schedule_batch',
                            'sglang.srt.model_executor.runner.flashinfer_autotune',
                            'sglang.srt.models.deepseek_v4',
                            'sglang.srt.models.deepseek_v4_dspark',
                            'sglang.srt.speculative.dspark_components.dspark_draft_sampler',
                            'sglang.kernels.ops.speculative.dspark.dspark_accept',
                            'sglang.srt.speculative.dspark_components.dspark_verify',
                            'sglang.kernels.ops.moe.moe_fused_gate',
                            'sglang.srt.speculative.dspark_components.dspark_draft',
                            'sglang.srt.speculative.dspark_components.dspark_planner',
                            'sglang.srt.layers.attention.dsv4.metadata'):
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is not None:
            spec.loader = EngramLoader(spec.loader)
        return spec

if os.environ.get('DSV41_SOURCE'):
    sys.meta_path.insert(0, EngramFinder())
    try:
        import tp3_pad
        tp3_pad.install()
    except Exception as exc:
        print(f'DSV41 TP pad not installed: {exc}', flush=True)
