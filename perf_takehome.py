"""
# Anthropic's Original Performance Engineering Take-home (Release version)

Copyright Anthropic PBC 2026. Permission is granted to modify and use, but not
to publish or redistribute your solutions so it's hard to find spoilers.

# Task

- Optimize the kernel (in KernelBuilder.build_kernel) as much as possible in the
  available time, as measured by test_kernel_cycles on a frozen separate copy
  of the simulator.

We recommend you look through problem.py next.
"""

from collections import defaultdict
import random
import unittest

from problem import (
    Engine,
    DebugInfo,
    SLOT_LIMITS,
    VLEN,
    N_CORES,
    SCRATCH_SIZE,
    Machine,
    Tree,
    Input,
    HASH_STAGES,
    reference_kernel,
    build_mem_image,
    reference_kernel2,
)


class KernelBuilder:
    def __init__(self):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def build(self, slots: list[tuple[Engine, tuple]], vliw: bool = False):
        # Handle manual packing hints with maximally packed bundles
        instrs = []
        for engine, slot in slots:
            if engine == "load_pair":
                # Manually packed loads - expand into 2-slot bundle
                dest1, addr1, dest2, addr2 = slot
                instrs.append({"load": [("load", dest1, addr1), ("load", dest2, addr2)]})
            elif engine == "load_vload_pair":
                # Pack two vloads into one cycle (uses both load slots)
                dest1, addr1, dest2, addr2 = slot
                instrs.append({"load": [("vload", dest1, addr1), ("vload", dest2, addr2)]})
            elif engine == "store_vstore_pair":
                # Pack two vstores into one cycle (uses both store slots)
                addr1, src1, addr2, src2 = slot
                instrs.append({"store": [("vstore", addr1, src1), ("vstore", addr2, src2)]})
            elif engine == "valu_pair":
                # Manually packed VALU ops - expand into 2-slot bundle
                op1, dest1, src1_1, src1_2, op2, dest2, src2_1, src2_2 = slot
                instrs.append({"valu": [(op1, dest1, src1_1, src1_2), (op2, dest2, src2_1, src2_2)]})
            elif engine == "valu_quad":
                # Pack 4 VALU ops into one cycle
                ops = slot
                valu_ops = []
                for i in range(0, len(ops), 4):
                    valu_ops.append((ops[i], ops[i+1], ops[i+2], ops[i+3]))
                instrs.append({"valu": valu_ops})
            elif engine == "valu_hex" or engine == "valu_six":
                # Pack up to 6 VALU ops into one cycle (max VALU slots)
                # slot should be a list of tuples, each (op, dest, src1, src2)
                ops = slot
                valu_ops = []
                # Each 4 elements = (op, dest, src1, src2)
                for i in range(0, len(ops), 4):
                    if i + 3 < len(ops):
                        valu_ops.append((ops[i], ops[i+1], ops[i+2], ops[i+3]))
                instrs.append({"valu": valu_ops})
            elif engine == "mega_bundle":
                # Fully packed bundle with multiple engines
                # slot should be a dict like {"load": [...], "valu": [...]}
                bundle = slot
                instrs.append(bundle)
            elif engine == "load_valu_bundle":
                # Pack loads with valu operations in one cycle
                load_ops, valu_ops = slot
                instrs.append({"load": load_ops, "valu": valu_ops})
            else:
                instrs.append({engine: [slot]})
        return instrs

    def add(self, engine, slot):
        self.instrs.append({engine: [slot]})

    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE, "Out of scratch space"
        return addr

    def scratch_const(self, val, name=None):
        if val not in self.const_map:
            addr = self.alloc_scratch(name)
            self.add("load", ("const", addr, val))
            self.const_map[val] = addr
        return self.const_map[val]

    def build_hash(self, val_hash_addr, tmp1, tmp2, round, i):
        slots = []

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            slots.append(("alu", (op1, tmp1, val_hash_addr, self.scratch_const(val1))))
            slots.append(("alu", (op3, tmp2, val_hash_addr, self.scratch_const(val3))))
            slots.append(("alu", (op2, val_hash_addr, tmp1, tmp2)))
            slots.append(("debug", ("compare", val_hash_addr, (round, i, "hash_stage", hi))))

        return slots

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Highly optimized vectorized implementation with maximal VALU packing.
        Strategy: Full pipeline unroll with 6-wide VALU operations across batches.
        """
        # Control debug output - disable for performance measurement
        ENABLE_DEBUG = False

        tmp1 = self.alloc_scratch("tmp1")
        tmp2 = self.alloc_scratch("tmp2")
        tmp3 = self.alloc_scratch("tmp3")

        # Scratch space addresses
        init_vars = [
            "rounds",
            "n_nodes",
            "batch_size",
            "forest_height",
            "forest_values_p",
            "inp_indices_p",
            "inp_values_p",
        ]
        for v in init_vars:
            self.alloc_scratch(v, 1)
        for i, v in enumerate(init_vars):
            self.add("load", ("const", tmp1, i))
            self.add("load", ("load", self.scratch[v], tmp1))

        # Preload constants
        zero_const = self.scratch_const(0)
        one_const = self.scratch_const(1)
        two_const = self.scratch_const(2)

        # Preload all hash constants as vectors to avoid repeated broadcasts
        hash_consts_vec = []
        for _, val1, _, _, val3 in HASH_STAGES:
            c1_vec = self.alloc_scratch(f"hash_c1_vec_{len(hash_consts_vec)}", VLEN)
            c3_vec = self.alloc_scratch(f"hash_c3_vec_{len(hash_consts_vec)}", VLEN)
            c1_scalar = self.scratch_const(val1)
            c3_scalar = self.scratch_const(val3)
            self.add("valu", ("vbroadcast", c1_vec, c1_scalar))
            self.add("valu", ("vbroadcast", c3_vec, c3_scalar))
            hash_consts_vec.append((c1_vec, c3_vec))

        # Prebroadcast commonly used values
        v_zero = self.alloc_scratch("v_zero", VLEN)
        v_one = self.alloc_scratch("v_one", VLEN)
        v_two = self.alloc_scratch("v_two", VLEN)
        v_n_nodes = self.alloc_scratch("v_n_nodes", VLEN)
        v_forest_base = self.alloc_scratch("v_forest_base", VLEN)

        self.add("valu", ("vbroadcast", v_zero, zero_const))
        self.add("valu", ("vbroadcast", v_one, one_const))
        self.add("valu", ("vbroadcast", v_two, two_const))
        self.add("valu", ("vbroadcast", v_n_nodes, self.scratch["n_nodes"]))
        self.add("valu", ("vbroadcast", v_forest_base, self.scratch["forest_values_p"]))

        self.add("flow", ("pause",))
        self.add("debug", ("comment", "Starting optimized vectorized loop"))

        # Allocate vector registers for software pipelining with aggressive reuse
        # OPTIMIZATION: Only 4 vectors needed per batch (not 6!)
        # - v_idx, v_val: Core state (must persist)
        # - v_tmp1, v_tmp2: Reusable temps
        # - v_addr: Reuse v_tmp1 (only needed between compute and load)
        # - v_node_val: Reuse v_tmp2 (only needed between load and XOR)
        # This saves 2 vectors × 8 words = 16 words per batch!
        num_batches = batch_size // VLEN
        N_PIPELINE = min(32, num_batches)  # Now we can fit all 32!

        v_idx = [self.alloc_scratch(f"v_idx_{i}", VLEN) for i in range(N_PIPELINE)]
        v_val = [self.alloc_scratch(f"v_val_{i}", VLEN) for i in range(N_PIPELINE)]
        v_tmp1 = [self.alloc_scratch(f"v_tmp1_{i}", VLEN) for i in range(N_PIPELINE)]
        v_tmp2 = [self.alloc_scratch(f"v_tmp2_{i}", VLEN) for i in range(N_PIPELINE)]

        # Aliases for clarity (these point to the same scratch as tmp1/tmp2)
        v_addr = v_tmp1  # Computed addresses stored in tmp1
        v_node_val = v_tmp2  # Node values loaded into tmp2

        body = []

        # Precompute base addresses outside the loop - pack ALU operations
        base_addrs = []
        for batch_base in range(0, batch_size, VLEN):
            base_idx_addr = self.alloc_scratch(f"base_idx_addr_b{batch_base}")
            base_val_addr = self.alloc_scratch(f"base_val_addr_b{batch_base}")
            # Load const and compute both addresses in sequence
            body.append(("load", ("const", tmp1, batch_base)))
            body.append(("alu", ("+", base_idx_addr, self.scratch["inp_indices_p"], tmp1)))
            body.append(("alu", ("+", base_val_addr, self.scratch["inp_values_p"], tmp1)))
            base_addrs.append((base_idx_addr, base_val_addr))

        # Note: ALU operations are very fast and don't bottleneck, so keeping simple

        # Main loop - process all batches together with maximally packed VALU operations
        # Strategy: separate into phases where operations from ALL batches can be packed together

        for round in range(rounds):
            # Process batches in groups of N_PIPELINE to avoid register conflicts
            # Aggressive optimization: minimal phases, maximal packing
            for group_start in range(0, num_batches, N_PIPELINE):
                group_size = min(N_PIPELINE, num_batches - group_start)

                # PHASE 1+2 MEGA-BUNDLE: Overlap loads with address computation!
                # Strategy: Load idx/val for batches N through N+5, while computing addresses for batches N-6 through N-1

                # First 6 batches: just loads (no addresses to compute yet)
                for local_idx in range(min(6, group_size)):
                    batch_idx = group_start + local_idx
                    buf = local_idx
                    base_idx_addr, base_val_addr = base_addrs[batch_idx]
                    body.append(("load_vload_pair", (v_idx[buf], base_idx_addr, v_val[buf], base_val_addr)))
                    if ENABLE_DEBUG and batch_idx == 0:
                        for vi in range(VLEN):
                            body.append(("debug", ("compare", v_idx[buf] + vi, (round, batch_idx * VLEN + vi, "idx"))))
                            body.append(("debug", ("compare", v_val[buf] + vi, (round, batch_idx * VLEN + vi, "val"))))

                # Remaining batches: Load batch N while computing addresses for batches N-6..N-1
                for local_idx in range(6, group_size):
                    batch_idx = group_start + local_idx
                    buf = local_idx
                    base_idx_addr, base_val_addr = base_addrs[batch_idx]

                    # Compute which batches to compute addresses for (the 6 batches before current)
                    addr_ops = []
                    for addr_offset in range(6):
                        addr_buf = local_idx - 6 + addr_offset
                        if addr_buf >= 0:
                            addr_ops.extend(["+", v_addr[addr_buf], v_forest_base, v_idx[addr_buf]])

                    if len(addr_ops) == 24:  # 6 ops × 4 elements each
                        body.append(("mega_bundle", {
                            "load": [("vload", v_idx[buf], base_idx_addr),
                                    ("vload", v_val[buf], base_val_addr)],
                            "valu": [(addr_ops[i], addr_ops[i+1], addr_ops[i+2], addr_ops[i+3])
                                    for i in range(0, 24, 4)]
                        }))
                    else:
                        # Fallback for incomplete group
                        body.append(("load_vload_pair", (v_idx[buf], base_idx_addr, v_val[buf], base_val_addr)))

                # Compute addresses for last 6 batches (no more loads to overlap)
                last_batch_start = max(0, group_size - 6)
                for batch_start in range(last_batch_start, group_size, 6):
                    ops = []
                    for offset in range(min(6, group_size - batch_start)):
                        buf = batch_start + offset
                        ops.extend(["+", v_addr[buf], v_forest_base, v_idx[buf]])
                    if ops:
                        body.append(("valu_hex", tuple(ops)))

                # PHASE 3+4+5: Stream processing - overlap node loads with hash computation
                # Load node values for batch N while computing XOR+hash for batch N-1

                # Batch 0: Just load node values (nothing to compute yet)
                if group_size > 0:
                    buf = 0
                    for load_idx in range(4):
                        offset = load_idx * 2
                        body.append(("load_pair", (
                            v_node_val[buf] + offset, v_addr[buf] + offset,
                            v_node_val[buf] + offset + 1, v_addr[buf] + offset + 1
                        )))
                    if ENABLE_DEBUG and group_start == 0:
                        for vi in range(VLEN):
                            body.append(("debug", ("compare", v_node_val[0] + vi, (round, vi, "node_val"))))

                # OPTIMIZATION: Track hash progress per batch to fill idle VALU slots during loads
                # hash_progress[i] = (stage, phase) where stage is 0-5, phase is 0 (op1+op3) or 1 (op2)
                # Special value: None means XOR not done yet
                hash_progress = [None] * group_size

                # Process batches 1+ in a pipeline: Load N, XOR N-1, Hash N-2+
                for batch_idx in range(1, group_size):
                    buf = batch_idx

                    # Load node values for current batch (4 cycles)
                    for load_idx in range(4):
                        offset = load_idx * 2

                        # Determine what VALU work we can do in parallel
                        valu_ops = []

                        # If previous batch just finished loading, do its XOR
                        if batch_idx >= 1 and load_idx == 0:
                            prev_buf = batch_idx - 1
                            valu_ops.append(("^", v_val[prev_buf], v_val[prev_buf], v_node_val[prev_buf]))
                            # Mark this batch as ready for hashing (stage 0, phase 0)
                            hash_progress[prev_buf] = (0, 0)

                        # Fill remaining VALU slots with hash operations for earlier batches
                        remaining_slots = 6 - len(valu_ops)

                        # Try to advance hash computation for batches that are 2+ behind
                        for lookback in range(2, batch_idx + 1):
                            if remaining_slots <= 0:
                                break

                            hash_batch = batch_idx - lookback
                            if hash_batch < 0:
                                break

                            # Can we do hash work for this batch?
                            if hash_progress[hash_batch] is None:
                                continue  # XOR not done yet

                            stage_idx, phase = hash_progress[hash_batch]
                            if stage_idx >= 6:
                                continue  # All 6 stages done

                            op1, val1, op2, op3, val3 = HASH_STAGES[stage_idx]
                            c1_vec, c3_vec = hash_consts_vec[stage_idx]

                            if phase == 0:
                                # Need op1 and op3
                                if remaining_slots >= 2:
                                    valu_ops.append((op1, v_tmp1[hash_batch], v_val[hash_batch], c1_vec))
                                    valu_ops.append((op3, v_tmp2[hash_batch], v_val[hash_batch], c3_vec))
                                    remaining_slots -= 2
                                    hash_progress[hash_batch] = (stage_idx, 1)  # Advance to phase 1
                            else:
                                # Need op2
                                if remaining_slots >= 1:
                                    valu_ops.append((op2, v_val[hash_batch], v_tmp1[hash_batch], v_tmp2[hash_batch]))
                                    remaining_slots -= 1
                                    hash_progress[hash_batch] = (stage_idx + 1, 0)  # Advance to next stage

                        if valu_ops:
                            body.append(("mega_bundle", {
                                "load": [("load", v_node_val[buf] + offset, v_addr[buf] + offset),
                                        ("load", v_node_val[buf] + offset + 1, v_addr[buf] + offset + 1)],
                                "valu": valu_ops
                            }))
                        else:
                            body.append(("load_pair", (
                                v_node_val[buf] + offset, v_addr[buf] + offset,
                                v_node_val[buf] + offset + 1, v_addr[buf] + offset + 1
                            )))

                # XOR for last batch (nothing loading anymore)
                if group_size > 0:
                    buf = group_size - 1
                    body.append(("valu", ("^", v_val[buf], v_val[buf], v_node_val[buf])))
                    # Mark last batch as ready for hashing
                    hash_progress[buf] = (0, 0)

                # Finish any remaining hash operations that weren't completed during loads
                # Process batches in order, completing all 6 stages for each
                for stage_idx in range(6):
                    op1, val1, op2, op3, val3 = HASH_STAGES[stage_idx]
                    c1_vec, c3_vec = hash_consts_vec[stage_idx]

                    # Phase 0: op1 and op3 for batches that need it
                    for batch_start in range(0, group_size, 3):
                        ops = []
                        for offset in range(min(3, group_size - batch_start)):
                            buf = batch_start + offset
                            if hash_progress[buf] is not None:
                                curr_stage, phase = hash_progress[buf]
                                if curr_stage == stage_idx and phase == 0:
                                    ops.extend([op1, v_tmp1[buf], v_val[buf], c1_vec])
                                    ops.extend([op3, v_tmp2[buf], v_val[buf], c3_vec])
                                    hash_progress[buf] = (stage_idx, 1)
                        if ops:
                            body.append(("valu_hex", tuple(ops)))

                    # Phase 1: op2 for batches that need it
                    for batch_start in range(0, group_size, 6):
                        ops = []
                        for offset in range(min(6, group_size - batch_start)):
                            buf = batch_start + offset
                            if hash_progress[buf] is not None:
                                curr_stage, phase = hash_progress[buf]
                                if curr_stage == stage_idx and phase == 1:
                                    ops.extend([op2, v_val[buf], v_tmp1[buf], v_tmp2[buf]])
                                    hash_progress[buf] = (stage_idx + 1, 0)
                        if ops:
                            body.append(("valu_hex", tuple(ops)))

                    if ENABLE_DEBUG and group_start == 0:
                        for vi in range(VLEN):
                            body.append(("debug", ("compare", v_val[0] + vi, (round, vi, "hash_stage", stage_idx))))

                if ENABLE_DEBUG and group_start == 0:
                    for vi in range(VLEN):
                        body.append(("debug", ("compare", v_val[0] + vi, (round, vi, "hashed_val"))))

                # PHASE 6: Compute next index
                for batch_start in range(0, group_size, 3):
                    ops = []
                    for offset in range(min(3, group_size - batch_start)):
                        buf = batch_start + offset
                        ops.extend(["&", v_tmp1[buf], v_val[buf], v_one])
                        ops.extend(["<<", v_tmp2[buf], v_idx[buf], v_one])
                    body.append(("valu_hex", tuple(ops)))

                for batch_start in range(0, group_size, 6):
                    ops = []
                    for offset in range(min(6, group_size - batch_start)):
                        buf = batch_start + offset
                        ops.extend(["+", v_tmp1[buf], v_one, v_tmp1[buf]])
                    body.append(("valu_hex", tuple(ops)))

                for batch_start in range(0, group_size, 6):
                    ops = []
                    for offset in range(min(6, group_size - batch_start)):
                        buf = batch_start + offset
                        ops.extend(["+", v_idx[buf], v_tmp2[buf], v_tmp1[buf]])
                    body.append(("valu_hex", tuple(ops)))

                if ENABLE_DEBUG and group_start == 0:
                    for vi in range(VLEN):
                        body.append(("debug", ("compare", v_idx[0] + vi, (round, vi, "next_idx"))))

                # PHASE 7: Wrap indices
                for batch_start in range(0, group_size, 6):
                    ops = []
                    for offset in range(min(6, group_size - batch_start)):
                        buf = batch_start + offset
                        ops.extend(["<", v_tmp1[buf], v_idx[buf], v_n_nodes])
                    body.append(("valu_hex", tuple(ops)))

                for batch_start in range(0, group_size, 6):
                    ops = []
                    for offset in range(min(6, group_size - batch_start)):
                        buf = batch_start + offset
                        ops.extend(["*", v_idx[buf], v_idx[buf], v_tmp1[buf]])
                    body.append(("valu_hex", tuple(ops)))

                if ENABLE_DEBUG and group_start == 0:
                    for vi in range(VLEN):
                        body.append(("debug", ("compare", v_idx[0] + vi, (round, vi, "wrapped_idx"))))

                # PHASE 8: Store results for this group
                for local_idx in range(group_size):
                    batch_idx = group_start + local_idx
                    buf = local_idx
                    base_idx_addr, base_val_addr = base_addrs[batch_idx]
                    body.append(("store_vstore_pair", (base_idx_addr, v_idx[buf], base_val_addr, v_val[buf])))

        body_instrs = self.build(body)
        self.instrs.extend(body_instrs)
        self.instrs.append({"flow": [("pause",)]})

BASELINE = 147734

def do_kernel_test(
    forest_height: int,
    rounds: int,
    batch_size: int,
    seed: int = 123,
    trace: bool = False,
    prints: bool = False,
):
    print(f"{forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)
    # print(kb.instrs)

    value_trace = {}
    machine = Machine(
        mem,
        kb.instrs,
        kb.debug_info(),
        n_cores=N_CORES,
        value_trace=value_trace,
        trace=trace,
    )
    machine.prints = prints
    for i, ref_mem in enumerate(reference_kernel2(mem, value_trace)):
        machine.run()
        inp_values_p = ref_mem[6]
        if prints:
            print(machine.mem[inp_values_p : inp_values_p + len(inp.values)])
            print(ref_mem[inp_values_p : inp_values_p + len(inp.values)])
        assert (
            machine.mem[inp_values_p : inp_values_p + len(inp.values)]
            == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
        ), f"Incorrect result on round {i}"
        inp_indices_p = ref_mem[5]
        if prints:
            print(machine.mem[inp_indices_p : inp_indices_p + len(inp.indices)])
            print(ref_mem[inp_indices_p : inp_indices_p + len(inp.indices)])
        # Updating these in memory isn't required, but you can enable this check for debugging
        # assert machine.mem[inp_indices_p:inp_indices_p+len(inp.indices)] == ref_mem[inp_indices_p:inp_indices_p+len(inp.indices)]

    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)
    return machine.cycle


class Tests(unittest.TestCase):
    def test_ref_kernels(self):
        """
        Test the reference kernels against each other
        """
        random.seed(123)
        for i in range(10):
            f = Tree.generate(4)
            inp = Input.generate(f, 10, 6)
            mem = build_mem_image(f, inp)
            reference_kernel(f, inp)
            for _ in reference_kernel2(mem, {}):
                pass
            assert inp.indices == mem[mem[5] : mem[5] + len(inp.indices)]
            assert inp.values == mem[mem[6] : mem[6] + len(inp.values)]

    def test_kernel_trace(self):
        # Full-scale example for performance testing
        do_kernel_test(10, 16, 256, trace=True, prints=False)

    # Passing this test is not required for submission, see submission_tests.py for the actual correctness test
    # You can uncomment this if you think it might help you debug
    # def test_kernel_correctness(self):
    #     for batch in range(1, 3):
    #         for forest_height in range(3):
    #             do_kernel_test(
    #                 forest_height + 2, forest_height + 4, batch * 16 * VLEN * N_CORES
    #             )

    def test_kernel_cycles(self):
        do_kernel_test(10, 16, 256)


# To run all the tests:
#    python perf_takehome.py
# To run a specific test:
#    python perf_takehome.py Tests.test_kernel_cycles
# To view a hot-reloading trace of all the instructions:  **Recommended debug loop**
# NOTE: The trace hot-reloading only works in Chrome. In the worst case if things aren't working, drag trace.json onto https://ui.perfetto.dev/
#    python perf_takehome.py Tests.test_kernel_trace
# Then run `python watch_trace.py` in another tab, it'll open a browser tab, then click "Open Perfetto"
# You can then keep that open and re-run the test to see a new trace.

# To run the proper checks to see which thresholds you pass:
#    python tests/submission_tests.py

if __name__ == "__main__":
    unittest.main()
