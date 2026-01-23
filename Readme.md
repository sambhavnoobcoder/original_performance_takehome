# Anthropic's Original Performance Take-Home

## Our Result: 1338 cycles (110x speedup)

**Optimization journey: 147734 → 1338 cycles**

| Stage | Cycles | Speedup | Key Technique | Status |
|-------|--------|---------|---------------|--------|
| Baseline | 147734 | 1.0x | Scalar reference implementation | ✓ |
| Initial vectorization | 13456 | 11.0x | 8-wide SIMD, 2-wide VLIW bundling, pre-broadcast constants | ✓ |
| Bitwise index ops | 12944 | 11.4x | Replace modulo with AND, multiply with shift | ✓ |
| Mega-bundle infrastructure | 12944 | 11.4x | load_valu_bundle support, multi-engine packing | ✓ |
| 6-wide VALU | 12944 | 11.4x | Replace vselect with multiply, add valu_hex support | ✓ |
| Pipeline allocation (N=4) | 12944 | 11.4x | Allocate rotating register sets for overlap | ✓ |
| Loop unrolling (N=16) | 12944 | 11.4x | Process 2 batches/iteration, separate load/compute | ✓ |
| Group processing (N=24) | 5600 | 26.4x | 6-wide VALU across batches, 8 phases per round | ✓ |
| Increase depth (N=27) | 5600 | 26.4x | Use available scratch space for more batches | ✓ |
| Phase overlapping | 5472 | 27.0x | Overlap Phase 1 loads with Phase 2, Phase 3 with 4 | ✓ |
| Register reduction (N=32) | 5440 | 27.2x | Reuse v_tmp1/v_tmp2, fit all batches in one group | ✓ |
| Hash-load overlap | 4128 | 35.8x | State machine to fill VALU during node loads | ✓ |
| Software pipelining v2 | 3824 | 38.6x | Enhanced pipeline with better scheduling | ✓ |
| Improved bundling | 3744 | 39.5x | Better multi-engine bundle packing | ✓ |
| Round fusion rewrite | 3056 | 48.3x | Complete rewrite with round-first processing | ✓ |
| Dynamic task scheduler | 2547 | 58.0x | Task-based dependency scheduling system | ✓ |
| Round fusion (K=16) | 2544 | 58.1x | Process all 16 rounds before storing | ✓ |
| Static scheduler | 2472 | 59.7x | Flat-list generation with dependency-aware packing | ✓ |
| multiply_add fusion | 2349 | 62.9x | Combine 3 hash ops into 1 VALU instruction | ✓ |
| Linear interpolation | **1338** | **110.4x** | vselect for levels 0-3, eliminates memory loads | ✓ |

### Detailed Optimization Timeline

**Phase 1: Basic Vectorization (147734 → 12944 cycles)**
1. **Initial vectorization (13456 cycles)**: Implemented 8-wide SIMD processing, manual 2-wide VLIW packing for loads/stores and VALU ops, pre-broadcast constants to avoid repeated operations
2. **Bitwise operations (12944 cycles)**: Replaced `val % 2` with `val & 1`, replaced `idx * 2` with `idx << 1` for cheaper operations
3. **Mega-bundle infrastructure (12944 cycles)**: Added `load_valu_bundle` to pack loads and VALU ops in same cycle
4. **6-wide VALU support (12944 cycles)**: Replaced flow vselect with VALU multiply to free flow slot, added valu_hex for 6 VALU ops/cycle

**Phase 2: Software Pipelining Exploration (12944 → 5600 cycles)**
5. **Pipeline allocation N=4 (12944 cycles)**: Added rotating register sets for overlapping operations across batches
6. **Loop unrolling N=16 (12944 cycles)**: Unrolled to process 2 batches per iteration, separated load and compute phases
7. **Breakthrough: Group processing N=24 (5600 cycles)**: Full 6-wide VALU packing across multiple batches, organized into 8 phases per round - **2.3x improvement!**
8. **Depth increase N=27 (5600 cycles)**: Used available scratch space for deeper pipeline

**Phase 3: Phase Overlapping (5600 → 4128 cycles)**
9. **Phase overlapping (5472 cycles)**: Overlapped Phase 1 loads with Phase 2 address compute, Phase 3 node loads with Phase 4 XOR
10. **Register reduction N=32 (5440 cycles)**: Reused v_tmp1 for v_addr and v_tmp2 for v_node_val (saved 512 words), fit all 32 batches in one group
11. **Hash-load overlap (4128 cycles)**: State machine to fill idle VALU slots during node loads with hash operations - **24% improvement!**

**Phase 4: Advanced Scheduling (4128 → 2544 cycles)**
12. **Software pipelining v2 (3824 cycles)**: Enhanced pipeline with better scheduling
13. **Improved bundling (3744 cycles)**: Better multi-engine bundle packing
14. **Round fusion rewrite (3056 cycles)**: Complete architectural rewrite processing rounds first - **20% improvement!**
15. **Dynamic task scheduler (2547 cycles)**: Implemented task-based dependency scheduling system - **17% improvement!**
16. **Round fusion K=16 (2544 cycles)**: Tuned to process all 16 rounds before storing results

**Phase 5: Static Optimization (2544 → 1338 cycles)**
17. **Static scheduler (2472 cycles)**: Replaced dynamic tasks with flat-list generation and dependency-aware packing
18. **multiply_add fusion (2349 cycles)**: Detected hash pattern `(val + c1) + (val << c3)` and replaced with single multiply_add instruction
19. **Linear interpolation (1338 cycles)**: Preloaded nodes 0-14, used vselect for levels 0-3 instead of memory loads - **43% improvement!** 🎉

### Failed Experiments (reverted)
| Experiment | Cycles | Reason for Failure |
|------------|--------|-------------------|
| Static modulo scheduling | 4640 | Too conservative, worse than dynamic scheduler |
| VALU pre-packing | 2635 | Reduced scheduler flexibility |
| Broadcast optimization | N/A | Out of scratch space, register conflicts |

### Key Insights

**The game-changer: vselect-based linear interpolation**
- Tree levels 0-3 use preloaded nodes (0-14) + vselect instead of memory gathers
- Level 0: Direct XOR with node[0]
- Level 1: 1 vselect between nodes 1-2
- Level 2: 3 vselects for nodes 3-6 (binary tree selection)
- Level 3: 7 vselects for nodes 7-14 (three-level selection tree)
- Level 4+: Traditional memory loads (can't avoid for deeper levels)

**Static scheduling wins**
- Generate all operations upfront as flat list of (engine, slot) tuples
- Dependency-aware scheduler packs operations respecting read-after-write
- Automatically fills all VLIW slots (6 VALU, 2 load, 2 store, 12 ALU, 1 flow per cycle)

**multiply_add is powerful**
- Hash pattern: `val = (val + c1) + (val << c3)` → `val = val * (1 + 2^c3) + c1`
- Reduces 3 operations to 1 VALU multiply_add instruction
- Index update: `idx = (idx << 1) + child` → `idx = idx * 2 + child`

**Architectural parameters matter**
- GROUP_SIZE=17: Process batches in groups (reuse contexts)
- ROUND_TILE=13: Process multiple rounds per group (better locality)
- Scratch space budget: 1536 words (tight constraint)

### Performance Breakdown

**Current bottleneck: VALU operations**
- Total VALU ops ≈ 7267
- Max throughput: 6 VALU/cycle
- Theoretical minimum: ~1211 cycles
- Our result: 1338 cycles (90% efficiency)

**Where cycles go:**
1. vselect operations for levels 0-3 (unavoidable control flow overhead)
2. Hash computation (6 stages × multiple ops)
3. Dependency stalls waiting for loads
4. Index computation and bounds checking

### Implementation Details

**Files:**
- `perf_takehome.py`: Optimized kernel (1338 cycles)
- `README.md`: This optimization journey

**Key functions:**
- `_schedule_slots()`: Static dependency-aware VLIW scheduler
- `_slot_rw()`: Dependency tracking (reads/writes per operation)
- `build_kernel()`: Main optimization with 7 techniques applied

**Tunable parameters:**
- `GROUP_SIZE = 17`: Batch grouping size
- `ROUND_TILE = 13`: Round tiling size
- `PRELOAD_NODES = 15`: Nodes 0-14 for vselect

---

## Original Performance Take-Home Info

This repo contains a version of Anthropic's original performance take-home, before Claude Opus 4.5 started doing better than humans given only 2 hours.

The original take-home was a 4-hour one that starts close to the contents of this repo, after Claude Opus 4 beat most humans at that, it was updated to a 2-hour one which started with code which achieved 18532 cycles (7.97x faster than this repo starts you). This repo is based on the newer take-home which has a few more instructions and comes with better debugging tools, but has the starter code reverted to the slowest baseline. After Claude Opus 4.5 we started using a different base for our time-limited take-homes.

Now you can try to beat Claude Opus 4.5 given unlimited time!

## Performance benchmarks

Measured in clock cycles from the simulated machine. All of these numbers are for models doing the 2 hour version which started at 18532 cycles:

- **2164 cycles**: Claude Opus 4 after many hours in the test-time compute harness
- **1790 cycles**: Claude Opus 4.5 in a casual Claude Code session, approximately matching the best human performance in 2 hours
- **1579 cycles**: Claude Opus 4.5 after 2 hours in our test-time compute harness
- **1548 cycles**: Claude Sonnet 4.5 after many more than 2 hours of test-time compute
- **1487 cycles**: Claude Opus 4.5 after 11.5 hours in the harness
- **1363 cycles**: Claude Opus 4.5 in an improved test time compute harness
- **1338 cycles**: Our implementation (beats Opus 4.5 improved harness!)
- **??? cycles**: Best human performance ever is substantially better than the above, but we won't say how much.

While it's no longer a good time-limited test, you can still use this test to get us excited about hiring you! If you optimize below 1487 cycles, beating Claude Opus 4.5's best performance at launch, email us at performance-recruiting@anthropic.com with your code (and ideally a resume) so we can be appropriately impressed, especially if you get near the best solution we've seen. New model releases may change what threshold impresses us though, and no guarantees that we keep this readme updated with the latest on that.

Run `python tests/submission_tests.py` to see which thresholds you pass.
