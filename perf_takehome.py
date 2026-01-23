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


def _vec_range(base: int, length: int = VLEN) -> range:
    """Get the range of addresses for a vector."""
    return range(base, base + length)


def _slot_rw(engine: str, slot: tuple) -> tuple[list[int], list[int]]:
    """Get read and write addresses for a slot operation."""
    reads = []
    writes = []

    if engine == "load":
        op = slot[0]
        if op == "const":
            _, dest, _ = slot
            writes.append(dest)
        elif op == "load":
            _, dest, addr = slot
            reads.append(addr)
            writes.append(dest)
        elif op == "vload":
            _, dest, addr = slot
            reads.append(addr)
            writes.extend(_vec_range(dest))
    elif engine == "store":
        op = slot[0]
        if op == "store":
            _, addr, src = slot
            reads.extend([addr, src])
        elif op == "vstore":
            _, addr, src = slot
            reads.append(addr)
            reads.extend(_vec_range(src))
    elif engine == "alu":
        op, dest, *args = slot
        reads.extend(args)
        writes.append(dest)
    elif engine == "valu":
        op = slot[0]
        if op == "vbroadcast":
            _, dest, src = slot
            reads.append(src)
            writes.extend(_vec_range(dest))
        elif op == "multiply_add":
            _, dest, src1, src2, src3 = slot
            reads.extend(_vec_range(src1))
            reads.extend(_vec_range(src2))
            reads.extend(_vec_range(src3))
            writes.extend(_vec_range(dest))
        else:
            _, dest, *args = slot
            for arg in args:
                reads.extend(_vec_range(arg))
            writes.extend(_vec_range(dest))
    elif engine == "flow":
        op = slot[0]
        if op == "add_imm":
            _, dest, src, _ = slot
            reads.append(src)
            writes.append(dest)
        elif op in ["pause", "halt"]:
            pass

    return reads, writes


def _schedule_slots(slots: list[tuple[str, tuple]]) -> list[dict[str, list[tuple]]]:
    """
    Automatically schedule operations into VLIW bundles respecting dependencies.
    """
    cycles = []
    usage = []
    ready_time = defaultdict(int)
    last_write = defaultdict(lambda: -1)
    last_read = defaultdict(lambda: -1)

    def ensure_cycle(cycle: int) -> None:
        while len(cycles) <= cycle:
            cycles.append({})
            usage.append(defaultdict(int))

    def find_cycle(engine: str, earliest: int) -> int:
        cycle = earliest
        limit = SLOT_LIMITS[engine]
        while True:
            ensure_cycle(cycle)
            if usage[cycle][engine] < limit:
                return cycle
            cycle += 1

    for engine, slot in slots:
        reads, writes = _slot_rw(engine, slot)
        earliest = 0
        for addr in reads:
            earliest = max(earliest, ready_time[addr])
        for addr in writes:
            earliest = max(earliest, last_write[addr] + 1, last_read[addr])

        cycle = find_cycle(engine, earliest)
        ensure_cycle(cycle)
        cycles[cycle].setdefault(engine, []).append(slot)
        usage[cycle][engine] += 1

        for addr in reads:
            if last_read[addr] < cycle:
                last_read[addr] = cycle
        for addr in writes:
            last_write[addr] = cycle
            ready_time[addr] = cycle + 1

    return [c for c in cycles if c]


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
        # Also precompute multiply_add constants for hash optimization
        hash_consts_vec = []
        hash_mul_vecs = []
        for op1, val1, op2, op3, val3 in HASH_STAGES:
            c1_vec = self.alloc_scratch(f"hash_c1_vec_{len(hash_consts_vec)}", VLEN)
            c3_vec = self.alloc_scratch(f"hash_c3_vec_{len(hash_consts_vec)}", VLEN)
            c1_scalar = self.scratch_const(val1)
            c3_scalar = self.scratch_const(val3)
            self.add("valu", ("vbroadcast", c1_vec, c1_scalar))
            self.add("valu", ("vbroadcast", c3_vec, c3_scalar))
            hash_consts_vec.append((c1_vec, c3_vec))

            # Optimize: if pattern is val = (val + c1) + (val << c3), use multiply_add
            if op1 == "+" and op2 == "+" and op3 == "<<":
                mul_const = 1 + (1 << val3)
                mul_vec = self.alloc_scratch(f"hash_mul_vec_{len(hash_mul_vecs)}", VLEN)
                mul_scalar = self.scratch_const(mul_const)
                self.add("valu", ("vbroadcast", mul_vec, mul_scalar))
                hash_mul_vecs.append(mul_vec)
            else:
                hash_mul_vecs.append(None)

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

        # Allocate vector registers for software pipelining with aggressive reuse
        # OPTIMIZATION: Only 4 vectors needed per batch (not 6!)
        # - v_idx, v_val: Core state (must persist)
        # - v_tmp1, v_tmp2: Reusable temps
        # - v_addr: Reuse v_tmp1 (only needed between compute and load)
        # - v_node_val: Reuse v_tmp2 (only needed between load and XOR)
        # This saves 2 vectors × 8 words = 16 words per batch!
        num_batches = batch_size // VLEN
        N_PIPELINE = min(32, num_batches)  # All 32 batches fit!

        # BATCH UNROLL FACTOR: Process batches in smaller groups for better ILP
        BATCH_UNROLL = 4  # Process 4 batches' tasks together

        v_idx = [self.alloc_scratch(f"v_idx_{i}", VLEN) for i in range(N_PIPELINE)]
        v_val = [self.alloc_scratch(f"v_val_{i}", VLEN) for i in range(N_PIPELINE)]
        v_tmp1 = [self.alloc_scratch(f"v_tmp1_{i}", VLEN) for i in range(N_PIPELINE)]
        v_tmp2 = [self.alloc_scratch(f"v_tmp2_{i}", VLEN) for i in range(N_PIPELINE)]

        # Aliases for clarity (these point to the same scratch as tmp1/tmp2)
        v_addr = v_tmp1  # Computed addresses stored in tmp1
        v_node_val = v_tmp2  # Node values loaded into tmp2

        # Generate all operations into a flat slots list for static scheduling
        slots = []

        # Precompute base addresses
        base_addrs = []
        for batch_base in range(0, batch_size, VLEN):
            base_idx_addr = self.alloc_scratch(f"base_idx_addr_b{batch_base}")
            base_val_addr = self.alloc_scratch(f"base_val_addr_b{batch_base}")
            slots.append(("load", ("const", tmp1, batch_base)))
            slots.append(("alu", ("+", base_idx_addr, self.scratch["inp_indices_p"], tmp1)))
            slots.append(("alu", ("+", base_val_addr, self.scratch["inp_values_p"], tmp1)))
            base_addrs.append((base_idx_addr, base_val_addr))

        # Main loop with round fusion
        K = rounds  # Process all rounds before storing
        for round_group_start in range(0, rounds, K):
            rounds_in_group = min(K, rounds - round_group_start)

            # Process batches in groups
            for group_start in range(0, num_batches, N_PIPELINE):
                group_size = min(N_PIPELINE, num_batches - group_start)

                for local_idx in range(group_size):
                    buf = local_idx
                    batch_idx = group_start + local_idx
                    base_idx_addr, base_val_addr = base_addrs[batch_idx]

                    # Load initial values
                    slots.append(("load", ("vload", v_idx[buf], base_idx_addr)))
                    slots.append(("load", ("vload", v_val[buf], base_val_addr)))

                    # Process all rounds
                    for round_in_group in range(rounds_in_group):
                        # Address computation
                        slots.append(("valu", ("+", v_addr[buf], v_forest_base, v_idx[buf])))

                        # Scattered loads
                        for lane in range(VLEN):
                            slots.append(("load", ("load", v_node_val[buf] + lane, v_addr[buf] + lane)))

                        # XOR
                        slots.append(("valu", ("^", v_val[buf], v_val[buf], v_node_val[buf])))

                        # Hash
                        for stage_idx in range(6):
                            op1, val1, op2, op3, val3 = HASH_STAGES[stage_idx]
                            c1_vec, c3_vec = hash_consts_vec[stage_idx]
                            mul_vec = hash_mul_vecs[stage_idx]

                            if mul_vec is not None:
                                # Optimized: val = val * mul + c1  (combines 3 ops into 1)
                                slots.append(("valu", ("multiply_add", v_val[buf], v_val[buf], mul_vec, c1_vec)))
                            else:
                                # Standard: 3 separate operations
                                slots.append(("valu", (op1, v_tmp1[buf], v_val[buf], c1_vec)))
                                slots.append(("valu", (op3, v_tmp2[buf], v_val[buf], c3_vec)))
                                slots.append(("valu", (op2, v_val[buf], v_tmp1[buf], v_tmp2[buf])))

                        # Next index computation: idx_new = (idx_old * 2) + ((val & 1) + 1)
                        # Extract bit and add 1 using scalar ALU ops (more efficient)
                        for lane in range(VLEN):
                            slots.append(("alu", ("&", v_tmp1[buf] + lane, v_val[buf] + lane, one_const)))
                            slots.append(("alu", ("+", v_tmp1[buf] + lane, v_tmp1[buf] + lane, one_const)))

                        # Use multiply_add: idx = idx * 2 + child_offset
                        slots.append(("valu", ("multiply_add", v_idx[buf], v_idx[buf], v_two, v_tmp1[buf])))

                        # Wrap to bounds
                        slots.append(("valu", ("<", v_tmp2[buf], v_idx[buf], v_n_nodes)))
                        slots.append(("valu", ("*", v_idx[buf], v_idx[buf], v_tmp2[buf])))

                    # Store results
                    slots.append(("store", ("vstore", base_idx_addr, v_idx[buf])))
                    slots.append(("store", ("vstore", base_val_addr, v_val[buf])))

        # Use static scheduler to pack operations into VLIW bundles
        self.instrs.extend(_schedule_slots(slots))
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
    # You can uncomment this check for debugging
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