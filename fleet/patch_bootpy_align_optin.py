#!/usr/bin/env python3
"""fq experiment (2026-09-18): make --speculative-dspark-align-verify-tokens-to-graph-tier opt-in.

Why: boot.py adds that flag whenever a profiled SPS table is present. With
SGLANG_RAGGED_VERIFY_MODE=compact (which boot.py also defaults when the table
exists) the target-verify CUDA-graph capture then builds a batch of
bs * DSPARK_BLOCK_SIZE tokens while forward_batch.spec_info.draft_token_num
declares bs * (DSPARK_BLOCK_SIZE + 1), which trips

    engram.py:296  assert num_tokens == bs * block
    AssertionError: engram target-verify expects one equal block per request,
                    got 12 tokens for 4 requests of 4

and kills the engine during boot. static mode boots but ignores the table, so
the flag is the remaining suspect. Set DSPARK_ALIGN_VERIFY_TO_TIER=1 to restore
the upstream behaviour."""
p = 'boot.py'
s = open(p, encoding='utf-8').read()
old = """            args += ['--speculative-dspark-sps-table-path', table]
            # Fills each step's verify window up to the cuda-graph tier the
            # forward is padded to anyway: free verification at the same cost.
            args += ['--speculative-dspark-align-verify-tokens-to-graph-tier']
"""
new = """            args += ['--speculative-dspark-sps-table-path', table]
            # Fills each step's verify window up to the cuda-graph tier the
            # forward is padded to anyway: free verification at the same cost.
            # fq: opt-in only. With compact ragged verify this made the target-verify
            # graph capture build bs*DSPARK_BLOCK_SIZE tokens while draft_token_num
            # declared bs*(block+1), tripping engram.py:296 and killing the engine at
            # boot (see SPS-TABLE-2026-09-18.md). DSPARK_ALIGN_VERIFY_TO_TIER=1 restores it.
            if os.environ.get('DSPARK_ALIGN_VERIFY_TO_TIER', '0').strip().lower() not in (
                '', '0', 'off', 'false', 'no'):
                args += ['--speculative-dspark-align-verify-tokens-to-graph-tier']
"""
if old not in s:
    raise SystemExit('anchor not found -- boot.py already patched?')
s = s.replace(old, new, 1)
open(p, 'w', encoding='utf-8', newline='\n').write(s)
print('patched boot.py: align-verify-to-graph-tier is now opt-in')
