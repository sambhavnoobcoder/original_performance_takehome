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
        Highly optimized vectorized implementation.
        """
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

        # Allocate vector registers for software pipelining
        # Process many batches in parallel to overlap loads with computation
        # Each buffer needs 6 vectors × 8 words = 48 words
        # We also have hash constants taking space, so use fewer pipeline stages
        num_batches = batch_size // VLEN
        N_PIPELINE = min(16, num_batches)  # Number of batches in flight
        v_idx = [self.alloc_scratch(f"v_idx_{i}", VLEN) for i in range(N_PIPELINE)]
        v_val = [self.alloc_scratch(f"v_val_{i}", VLEN) for i in range(N_PIPELINE)]
        v_node_val = [self.alloc_scratch(f"v_node_val_{i}", VLEN) for i in range(N_PIPELINE)]
        v_addr = [self.alloc_scratch(f"v_addr_{i}", VLEN) for i in range(N_PIPELINE)]
        v_tmp1 = [self.alloc_scratch(f"v_tmp1_{i}", VLEN) for i in range(N_PIPELINE)]
        v_tmp2 = [self.alloc_scratch(f"v_tmp2_{i}", VLEN) for i in range(N_PIPELINE)]

        body = []

        # Precompute base addresses outside the loop
        base_addrs = []
        for batch_base in range(0, batch_size, VLEN):
            base_idx_addr = self.alloc_scratch(f"base_idx_addr_b{batch_base}")
            base_val_addr = self.alloc_scratch(f"base_val_addr_b{batch_base}")
            body.append(("load", ("const", tmp1, batch_base)))
            body.append(("alu", ("+", base_idx_addr, self.scratch["inp_indices_p"], tmp1)))
            body.append(("alu", ("+", base_val_addr, self.scratch["inp_values_p"], tmp1)))
            base_addrs.append((base_idx_addr, base_val_addr))

        # Main loop with 2x unrolling to hide load latency
        # Process 2 batches at once, interleaving loads and compute

        for round in range(rounds):
            for batch_idx in range(0, num_batches, 2):
                batch_base_0 = batch_idx * VLEN
                buf_0 = batch_idx % N_PIPELINE
                base_idx_addr_0, base_val_addr_0 = base_addrs[batch_idx]

                batch_base_1 = (batch_idx + 1) * VLEN if batch_idx + 1 < num_batches else batch_base_0
                buf_1 = (batch_idx + 1) % N_PIPELINE if batch_idx + 1 < num_batches else buf_0
                base_idx_addr_1, base_val_addr_1 = base_addrs[batch_idx + 1] if batch_idx + 1 < num_batches else base_addrs[batch_idx]

                #=== BATCH 0: Load phase ===
                body.append(("load_vload_pair", (v_idx[buf_0], base_idx_addr_0, v_val[buf_0], base_val_addr_0)))
                if batch_base_0 < VLEN:
                    for vi in range(min(VLEN, batch_size - batch_base_0)):
                        body.append(("debug", ("compare", v_idx[buf_0] + vi, (round, batch_base_0 + vi, "idx"))))
                        body.append(("debug", ("compare", v_val[buf_0] + vi, (round, batch_base_0 + vi, "val"))))

                # Compute addresses while load completes
                body.append(("valu", ("+", v_addr[buf_0], v_forest_base, v_idx[buf_0])))

                #=== BATCH 1: Load phase (if exists) ===
                if batch_idx + 1 < num_batches:
                    body.append(("load_vload_pair", (v_idx[buf_1], base_idx_addr_1, v_val[buf_1], base_val_addr_1)))
                    if batch_base_1 < VLEN:
                        for vi in range(min(VLEN, batch_size - batch_base_1)):
                            body.append(("debug", ("compare", v_idx[buf_1] + vi, (round, batch_base_1 + vi, "idx"))))
                            body.append(("debug", ("compare", v_val[buf_1] + vi, (round, batch_base_1 + vi, "val"))))
                    body.append(("valu", ("+", v_addr[buf_1], v_forest_base, v_idx[buf_1])))

                #=== BATCH 0: Node loads ===
                body.append(("load_pair", (v_node_val[buf_0] + 0, v_addr[buf_0] + 0, v_node_val[buf_0] + 1, v_addr[buf_0] + 1)))
                body.append(("load_pair", (v_node_val[buf_0] + 2, v_addr[buf_0] + 2, v_node_val[buf_0] + 3, v_addr[buf_0] + 3)))
                body.append(("load_pair", (v_node_val[buf_0] + 4, v_addr[buf_0] + 4, v_node_val[buf_0] + 5, v_addr[buf_0] + 5)))
                body.append(("load_pair", (v_node_val[buf_0] + 6, v_addr[buf_0] + 6, v_node_val[buf_0] + 7, v_addr[buf_0] + 7)))
                if batch_base_0 < VLEN:
                    for vi in range(min(VLEN, batch_size - batch_base_0)):
                        body.append(("debug", ("compare", v_node_val[buf_0] + vi, (round, batch_base_0 + vi, "node_val"))))

                #=== BATCH 1: Node loads (if exists) ===
                if batch_idx + 1 < num_batches:
                    body.append(("load_pair", (v_node_val[buf_1] + 0, v_addr[buf_1] + 0, v_node_val[buf_1] + 1, v_addr[buf_1] + 1)))
                    body.append(("load_pair", (v_node_val[buf_1] + 2, v_addr[buf_1] + 2, v_node_val[buf_1] + 3, v_addr[buf_1] + 3)))
                    body.append(("load_pair", (v_node_val[buf_1] + 4, v_addr[buf_1] + 4, v_node_val[buf_1] + 5, v_addr[buf_1] + 5)))
                    body.append(("load_pair", (v_node_val[buf_1] + 6, v_addr[buf_1] + 6, v_node_val[buf_1] + 7, v_addr[buf_1] + 7)))
                    if batch_base_1 < VLEN:
                        for vi in range(min(VLEN, batch_size - batch_base_1)):
                            body.append(("debug", ("compare", v_node_val[buf_1] + vi, (round, batch_base_1 + vi, "node_val"))))

                #=== BATCH 0: Compute phase ===
                body.append(("valu", ("^", v_val[buf_0], v_val[buf_0], v_node_val[buf_0])))
                for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                    c1_vec, c3_vec = hash_consts_vec[hi]
                    body.append(("valu_pair", (op1, v_tmp1[buf_0], v_val[buf_0], c1_vec, op3, v_tmp2[buf_0], v_val[buf_0], c3_vec)))
                    body.append(("valu", (op2, v_val[buf_0], v_tmp1[buf_0], v_tmp2[buf_0])))
                    if batch_base_0 < VLEN:
                        for vi in range(min(VLEN, batch_size - batch_base_0)):
                            body.append(("debug", ("compare", v_val[buf_0] + vi, (round, batch_base_0 + vi, "hash_stage", hi))))
                if batch_base_0 < VLEN:
                    for vi in range(min(VLEN, batch_size - batch_base_0)):
                        body.append(("debug", ("compare", v_val[buf_0] + vi, (round, batch_base_0 + vi, "hashed_val"))))

                body.append(("valu_pair", ("&", v_tmp1[buf_0], v_val[buf_0], v_one, "<<", v_tmp2[buf_0], v_idx[buf_0], v_one)))
                body.append(("valu", ("+", v_tmp1[buf_0], v_one, v_tmp1[buf_0])))
                body.append(("valu", ("+", v_idx[buf_0], v_tmp2[buf_0], v_tmp1[buf_0])))
                if batch_base_0 < VLEN:
                    for vi in range(min(VLEN, batch_size - batch_base_0)):
                        body.append(("debug", ("compare", v_idx[buf_0] + vi, (round, batch_base_0 + vi, "next_idx"))))

                body.append(("valu", ("<", v_tmp1[buf_0], v_idx[buf_0], v_n_nodes)))
                body.append(("valu", ("*", v_idx[buf_0], v_idx[buf_0], v_tmp1[buf_0])))
                if batch_base_0 < VLEN:
                    for vi in range(min(VLEN, batch_size - batch_base_0)):
                        body.append(("debug", ("compare", v_idx[buf_0] + vi, (round, batch_base_0 + vi, "wrapped_idx"))))

                body.append(("store_vstore_pair", (base_idx_addr_0, v_idx[buf_0], base_val_addr_0, v_val[buf_0])))

                #=== BATCH 1: Compute phase (if exists) ===
                if batch_idx + 1 < num_batches:
                    body.append(("valu", ("^", v_val[buf_1], v_val[buf_1], v_node_val[buf_1])))
                    for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                        c1_vec, c3_vec = hash_consts_vec[hi]
                        body.append(("valu_pair", (op1, v_tmp1[buf_1], v_val[buf_1], c1_vec, op3, v_tmp2[buf_1], v_val[buf_1], c3_vec)))
                        body.append(("valu", (op2, v_val[buf_1], v_tmp1[buf_1], v_tmp2[buf_1])))
                        if batch_base_1 < VLEN:
                            for vi in range(min(VLEN, batch_size - batch_base_1)):
                                body.append(("debug", ("compare", v_val[buf_1] + vi, (round, batch_base_1 + vi, "hash_stage", hi))))
                    if batch_base_1 < VLEN:
                        for vi in range(min(VLEN, batch_size - batch_base_1)):
                            body.append(("debug", ("compare", v_val[buf_1] + vi, (round, batch_base_1 + vi, "hashed_val"))))

                    body.append(("valu_pair", ("&", v_tmp1[buf_1], v_val[buf_1], v_one, "<<", v_tmp2[buf_1], v_idx[buf_1], v_one)))
                    body.append(("valu", ("+", v_tmp1[buf_1], v_one, v_tmp1[buf_1])))
                    body.append(("valu", ("+", v_idx[buf_1], v_tmp2[buf_1], v_tmp1[buf_1])))
                    if batch_base_1 < VLEN:
                        for vi in range(min(VLEN, batch_size - batch_base_1)):
                            body.append(("debug", ("compare", v_idx[buf_1] + vi, (round, batch_base_1 + vi, "next_idx"))))

                    body.append(("valu", ("<", v_tmp1[buf_1], v_idx[buf_1], v_n_nodes)))
                    body.append(("valu", ("*", v_idx[buf_1], v_idx[buf_1], v_tmp1[buf_1])))
                    if batch_base_1 < VLEN:
                        for vi in range(min(VLEN, batch_size - batch_base_1)):
                            body.append(("debug", ("compare", v_idx[buf_1] + vi, (round, batch_base_1 + vi, "wrapped_idx"))))

                    body.append(("store_vstore_pair", (base_idx_addr_1, v_idx[buf_1], base_val_addr_1, v_val[buf_1])))

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
