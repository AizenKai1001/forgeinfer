"""Tests of tpu/hlo_cost.py (the compiler's cycle estimates in optimized HLO text) on a small hand-written module: the sum over the ENTRY computation only, the grouping by op_name, a stray '}' at column 0 inside the
entry (a long attribute wrapped by the printer), instructions without an estimate, and the clock.

    python test_hlo_cost.py
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hlo_cost as H  # noqa: E402


def cfg(cycles):
    return '{"flag_configs":[],"window_config":{"estimated_cycles":"%d","iteration_bounds":["1","1"]}}' % cycles


HLO = "\n".join([
    "HloModule jit_step_fn, is_scheduled=true",
    "",
    "%fused_computation.1 (p: f32[8,8]) -> f32[8,8] {",
    "  %p = f32[8,8]{1,0} parameter(0)",
    "  ROOT %c = f32[8,8]{1,0} convolution(%p, %p), dim_labels=bf_oi->bf, backend_config=" + cfg(999999999) + "",  # inside a fused computation: must not be counted (the fusion carries the estimate)
    "}",
    "",
    "ENTRY %main (param.1: f32[8,8]) -> f32[8,8] {",
    "  %param.1 = f32[8,8]{1,0} parameter(0)",
    "  %constant.1 = f32[] constant(1), metadata={op_name=\"jit(step_fn)/jvp()/mul\"}",
    "  %fusion.1 = f32[8,8]{1,0} fusion(%param.1), kind=kOutput, calls=%fused_computation.1, metadata={op_name=\"jit(step_fn)/jvp(layer)/dot_general\"}, backend_config=" + cfg(1_500_000),
    "  %fusion.2 = f32[8,8]{1,0} fusion(%fusion.1), kind=kLoop, calls=%fused_computation.1, metadata={op_name=\"jit(step_fn)/transpose(jvp(layer))/mul\"}, backend_config=" + cfg(3_000_000),
    "  %fusion.3 = f32[8,8]{1,0} fusion(%fusion.2), kind=kLoop, metadata={op_name=\"jit(step_fn)/transpose(jvp(layer))/checkpoint/rematted_computation/mul\"}, backend_config=" + cfg(1_500_000),
    "  %fusion.4 = f32[24,8,8]{2,1,0} fusion(%fusion.3), kind=kLoop, metadata={op_name=\"jit(step_fn)/vmap(ik,kj->ij)/dot_general\"}, backend_config=" + cfg(4_500_000),
    "  %fusion.5 = f32[24,8,8]{2,1,0} fusion(%fusion.4), kind=kLoop, metadata={op_name=\"jit(step_fn)/vmap()/add\"}, backend_config=" + cfg(1_500_000),
    "  %reduce.1 = bf16[4,16,8,8]{3,2,1,0} reduce(%fusion.5, %constant.1), dimensions={0}, metadata={op_name=\"splash_mha_dkv_no_residuals/pallas_call/reduce_sum\"}, backend_config=" + cfg(3_000_000),
    "  %custom-call.1 = f32[8,8]{1,0} custom-call(%fusion.1), custom_call_target=\"tpu_custom_call\", metadata={op_name=\"vmap(static_sparsity_splash)/jit(_splash_attention)/splash_mha_fwd_residuals/pallas_call\"}, backend_config=" + cfg(1_500_000),
    "  %copy.1 = f32[8,8]{1,0} copy(%custom-call.1), backend_config=" + cfg(1_500_000),
    "  %copy-start.1 = (f32[8,8]{1,0}, f32[8,8]{1,0}, u32[]) copy-start(%copy.1)",
    "  %sub.1 = f32[8,8]{1,0} fusion(%copy.1), kind=kLoop, metadata={op_name=\"jit(step_fn)/sub\"}, backend_config={\"window_config\":{\"estimated_cycles\":\"1500000\"",
    "}}",  # a wrapped attribute: a '}' at column 0 inside the entry computation
    "  ROOT %tuple.1 = (f32[8,8]{1,0}) tuple(%sub.1)",
    "}",
])


class AnalyzeTest(unittest.TestCase):
    def test_sums_the_entry_computation_only(self):
        r = H.analyze(HLO, ghz=1.5)
        # 1.5e6 + 3e6 + 1.5e6 + 4.5e6 + 1.5e6 + 3e6 + 1.5e6 + 1.5e6 + 1.5e6 = 19.5e6 cycles at 1.5 GHz = 13 ms (the fused computation's 1e9 cycles and the instructions without an estimate do not count)
        self.assertAlmostEqual(r["total_ms"], 13.0, places=6)
        self.assertEqual(r["ops"], 9)

    def test_groups_by_the_place_in_the_program(self):
        r = H.analyze(HLO, ghz=1.5)
        reg = {k: v["ms"] for k, v in r["regions"].items()}
        self.assertAlmostEqual(reg["forward"], 1.0, places=6)
        self.assertAlmostEqual(reg["backward: gradients"], 2.0, places=6)
        self.assertAlmostEqual(reg["backward: recomputation (remat)"], 1.0, places=6)
        self.assertAlmostEqual(reg["muon: Newton-Schulz matmuls"], 3.0, places=6)
        self.assertAlmostEqual(reg["muon: update of the stacked matrices"], 1.0, places=6)
        self.assertAlmostEqual(reg["attention: dK/dV reduction over the head group"], 2.0, places=6)
        self.assertAlmostEqual(reg["attention: kernels"], 1.0, places=6)
        self.assertAlmostEqual(reg["no name (copies, layout changes, constants)"], 1.0, places=6)
        self.assertAlmostEqual(reg["optimizer and the rest (AdamW, clipping, gradient norm)"], 1.0, places=6)
        self.assertAlmostEqual(sum(reg.values()), r["total_ms"], places=6)

    def test_clock_scales_the_milliseconds(self):
        self.assertAlmostEqual(H.analyze(HLO, ghz=3.0)["total_ms"], 6.5, places=6)

    def test_region_rules(self):
        self.assertEqual(H.region("reduce", "splash_mha_dkv_no_residuals/x/reduce_sum"), "attention: dK/dV reduction over the head group")
        self.assertEqual(H.region("custom-call", "splash_mha_dkv_no_residuals/pallas_call"), "attention: backward kernels")
        self.assertEqual(H.region("fusion", "jit(step_fn)/vmap(ik,jk->ij)/dot_general"), "muon: Newton-Schulz matmuls")
        self.assertEqual(H.region("copy", ""), "no name (copies, layout changes, constants)")
        self.assertEqual(H.region("fusion", "jit(step_fn)/transpose(jvp(jvp()))/checkpoint/rematted_computation/mul"), "backward: recomputation (remat)")

    def test_render_mentions_every_region(self):
        text = H.render(H.analyze(HLO), top=2)
        self.assertIn("estimated step: 13.0 ms", text)
        self.assertIn("muon: Newton-Schulz matmuls", text)

    def test_text_without_an_entry_is_an_error(self):
        with self.assertRaises(ValueError):
            H.analyze("HloModule empty\n")


# a module with the matrix-unit work of a training step in miniature: the forward projection (the batch of the einsum is a spatial dimension of the convolution), a weight gradient (the sum over the batch is a window), the
# identity matmul of the lane layout, a batch of Newton-Schulz matmuls in the dilated-window form, a dot one call deeper, and an elementwise fusion; the fusions carry the estimates that the scheduler would have written
HLO_MM = "\n".join([
    "HloModule jit_step_fn, is_scheduled=true",
    "",
    "%convert_fusion.1 (p: f32[1024,1024]) -> bf16[1024,1024,1] {",
    "  %p = f32[1024,1024]{1,0:T(8,128)} parameter(0)",
    "  ROOT %cv = bf16[1024,1024,1]{1,0,2:T(8,128)(2,1)} convert(%p)",
    "}",
    "",
    "%fused_computation.10 (param_0.1: bf16[4,2048,1024], param_1.1: f32[1024,1024]) -> f32[4,2048,1024] {",
    "  %param_0.1 = bf16[4,2048,1024]{2,1,0:T(8,128)(2,1)S(1)} parameter(0)",
    "  %param_1.1 = f32[1024,1024]{1,0:T(8,128)} parameter(1)",
    "  %cv.1 = bf16[1024,1024,1]{1,0,2:T(8,128)(2,1)} fusion(%param_1.1), kind=kLoop, calls=%convert_fusion.1",
    "  ROOT %convolution.1 = f32[4,2048,1024]{1,2,0:T(8,128)} convolution(%param_0.1, %cv.1), window={size=1}, dim_labels=0bf_oi0->0bf, metadata={op_name=\"jit(step_fn)/jvp(oe,bte->bot)/dot_general\"}",
    "}",
    "",
    "%fused_computation.11 (param_0.2: f32[4,2048,1024], param_1.2: bf16[4,2048,4096]) -> f32[1024,4096,1] {",
    "  %param_0.2 = f32[4,2048,1024]{2,1,0:T(8,128)} parameter(0)",
    "  %param_1.2 = bf16[4,2048,4096]{2,1,0:T(8,128)(2,1)} parameter(1)",
    "  ROOT %convolution.2 = f32[1024,4096,1]{1,0,2:T(8,128)} convolution(%param_0.2, %param_1.2), window={size=4}, dim_labels=0fb_0io->bf0, metadata={op_name=\"jit(step_fn)/transpose(jvp(jvp()))/checkpoint/...i,oi->...o/dot_general\"}",
    "}",
    "",
    "%fused_computation.12 (param_0.3: bf16[4,16,64,2048], param_1.3: bf16[64,64,1,1]) -> bf16[4,16,2048,64] {",
    "  %param_0.3 = bf16[4,16,64,2048]{3,2,1,0:T(8,128)(2,1)} parameter(0)",
    "  %param_1.3 = bf16[64,64,1,1]{0,1,3,2:T(8,128)(2,1)} parameter(1)",
    "  ROOT %convolution.3 = bf16[4,16,2048,64]{3,2,1,0:T(8,128)(2,1)} convolution(%param_0.3, %param_1.3), window={size=1x1}, dim_labels=01fb_io01->01bf, operand_precision={highest,highest}, metadata={op_name=\"jit(step_fn)/jvp(bndt,de->bnte)/dot_general\"}",
    "}",
    "",
    "%fused_computation.13 (param_0.4: bf16[6,256,1024], param_1.4: bf16[6,256,1024]) -> f32[6,256,256] {",
    "  %param_0.4 = bf16[6,256,1024]{2,1,0:T(8,128)(2,1)} parameter(0)",
    "  %param_1.4 = bf16[6,256,1024]{2,1,0:T(8,128)(2,1)} parameter(1)",
    "  ROOT %convolution-base-dilated.4 = f32[6,256,256]{2,1,0:T(8,128)} convolution(%param_0.4, %param_1.4), window={size=6 stride=5 lhs_dilate=6}, dim_labels=0bf_0oi->0bf, metadata={op_name=\"jit(step_fn)/shard_map/vmap(ik,jk->ij)/dot_general\"}",
    "}",
    "",
    "%fused_computation.14 (param_0.5: f32[8,16,32], param_1.5: f32[8,32,64]) -> f32[8,16,64] {",
    "  %param_0.5 = f32[8,16,32]{2,1,0} parameter(0)",
    "  %param_1.5 = f32[8,32,64]{2,1,0} parameter(1)",
    "  ROOT %dot.5 = f32[8,16,64]{2,1,0} dot(%param_0.5, %param_1.5), lhs_batch_dims={0}, lhs_contracting_dims={2}, rhs_batch_dims={0}, rhs_contracting_dims={1}",
    "}",
    "",
    "%wrapper.15 (param_0.6: f32[8,16,32], param_1.6: f32[8,32,64]) -> f32[8,16,64] {",
    "  %param_0.6 = f32[8,16,32]{2,1,0} parameter(0)",
    "  %param_1.6 = f32[8,32,64]{2,1,0} parameter(1)",
    "  ROOT %inner = f32[8,16,64]{2,1,0} fusion(%param_0.6, %param_1.6), kind=kOutput, calls=%fused_computation.14",
    "}",
    "",
    "%fused_computation.16 (param_0.7: f32[8,8]) -> f32[8,8] {",
    "  %param_0.7 = f32[8,8]{1,0} parameter(0)",
    "  ROOT %add.7 = f32[8,8]{1,0} add(%param_0.7, %param_0.7)",
    "}",
    "",
    "ENTRY %main (a: bf16[4,2048,1024], w: f32[1024,1024]) -> f32[4,2048,1024] {",
    "  %a = bf16[4,2048,1024]{2,1,0:T(8,128)(2,1)S(1)} parameter(0)",
    "  %w = f32[1024,1024]{1,0:T(8,128)} parameter(1)",
    "  %g = f32[4,2048,1024]{2,1,0:T(8,128)} parameter(2)",
    "  %c = bf16[4,2048,4096]{2,1,0:T(8,128)(2,1)} parameter(3)",
    "  %t = bf16[4,16,64,2048]{3,2,1,0:T(8,128)(2,1)} parameter(4)",
    "  %e = bf16[64,64,1,1]{0,1,3,2:T(8,128)(2,1)} parameter(5)",
    "  %n = bf16[6,256,1024]{2,1,0:T(8,128)(2,1)} parameter(6)",
    "  %d1 = f32[8,16,32]{2,1,0} parameter(7)",
    "  %d2 = f32[8,32,64]{2,1,0} parameter(8)",
    "  %small = f32[8,8]{1,0} parameter(9)",
    "  %fusion.proj = f32[4,2048,1024]{1,2,0:T(8,128)} fusion(%a, %w), kind=kOutput, calls=%fused_computation.10, metadata={op_name=\"jit(step_fn)/jvp(oe,bte->bot)/dot_general\"}, backend_config=" + cfg(130_800),
    "  %fusion.wgrad = f32[1024,4096,1]{1,0,2:T(8,128)} fusion(%g, %c), kind=kOutput, calls=%fused_computation.11, metadata={op_name=\"jit(step_fn)/transpose(jvp(jvp()))/checkpoint/...i,oi->...o/dot_general\"}, backend_config=" + cfg(600_000),
    "  %fusion.ident = bf16[4,16,2048,64]{3,2,1,0:T(8,128)(2,1)} fusion(%t, %e), kind=kOutput, calls=%fused_computation.12, metadata={op_name=\"jit(step_fn)/jvp(bndt,de->bnte)/dot_general\"}, backend_config=" + cfg(300_000),
    "  %fusion.ns = f32[6,256,256]{2,1,0:T(8,128)} fusion(%n, %n), kind=kOutput, calls=%fused_computation.13, metadata={op_name=\"jit(step_fn)/shard_map/vmap(ik,jk->ij)/dot_general\"}, backend_config=" + cfg(15_000),
    "  %fusion.dot = f32[8,16,64]{2,1,0} fusion(%d1, %d2), kind=kOutput, calls=%wrapper.15, metadata={op_name=\"jit(step_fn)/jvp()/dot_general\"}, backend_config=" + cfg(3_000),
    "  %fusion.plain = f32[8,8]{1,0} fusion(%small), kind=kLoop, calls=%fused_computation.16, metadata={op_name=\"jit(step_fn)/jvp()/add\"}, backend_config=" + cfg(9_000_000),
    "  ROOT %tuple.1 = (f32[4,2048,1024]{1,2,0:T(8,128)}) tuple(%fusion.proj)",
    "}",
])


class MatmulTest(unittest.TestCase):
    def setUp(self):
        self.rows = {r["name"]: r for r in H.matmul_fusions(HLO_MM, ghz=1.5, peak=197e12)}

    def test_finds_the_fusions_that_contain_a_matmul_and_only_those(self):
        self.assertEqual(sorted(self.rows), ["fusion.dot", "fusion.ident", "fusion.ns", "fusion.proj", "fusion.wgrad"])  # fusion.plain is an elementwise fusion; the dot is found one call deeper

    def test_flops_of_a_projection_with_the_batch_as_a_spatial_dimension(self):
        r = self.rows["fusion.proj"]
        self.assertEqual(r["flops"], 2 * 4 * 2048 * 1024 * 1024)
        self.assertAlmostEqual(r["ideal_ms"], 2 * 4 * 2048 * 1024 * 1024 / 197e12 * 1e3, places=9)
        self.assertAlmostEqual(r["est_ms"], 130_800 / 1.5e6, places=9)
        self.assertAlmostEqual(r["lost_ms"], r["est_ms"] - r["ideal_ms"], places=9)
        self.assertAlmostEqual(r["eff"], r["ideal_ms"] / r["est_ms"], places=9)
        self.assertEqual(r["conv"][0]["labels"], "0bf_oi0->0bf")
        self.assertEqual(r["conv"][0]["lhs"], "bf16[4,2048,1024]{2,1,0}S1")
        self.assertEqual(r["conv"][0]["rhs"], "bf16[1024,1024,1]{1,0,2}")
        self.assertEqual(r["conv"][0]["out"], "f32[4,2048,1024]{1,2,0}")
        self.assertEqual(r["conv"][0]["k"], 1024)

    def test_weight_gradient_sums_over_the_window(self):
        r = self.rows["fusion.wgrad"]  # 1024 x 4096 outputs, each a sum over 2048 positions of the 4 sequences
        self.assertEqual(r["flops"], 2 * 1024 * 4096 * 2048 * 4)
        self.assertEqual(r["conv"][0]["k"], 2048 * 4)
        self.assertEqual(H.einsum_of(r["op_name"]), ("...i,oi->...o", "bwd"))

    def test_identity_matmul_and_a_dilated_batch_of_matrices(self):
        self.assertEqual(self.rows["fusion.ident"]["flops"], 2 * (4 * 16 * 2048 * 64) * 64)
        # six matrices of 256 x 1024 times 1024 x 256: the stride-5, dilation-6 window of the batch trick has ONE real tap per output position
        self.assertEqual(self.rows["fusion.ns"]["flops"], 6 * 2 * 256 * 256 * 1024)

    def test_dot_inside_a_called_computation(self):
        self.assertEqual(self.rows["fusion.dot"]["flops"], 2 * (8 * 16 * 64) * 32)

    def test_sorted_by_the_time_lost_and_rendered(self):
        rows = H.matmul_fusions(HLO_MM)
        lost = [r["lost_ms"] for r in rows]
        self.assertEqual(lost, sorted(lost, reverse=True))
        self.assertEqual(rows[0]["name"], "fusion.ident")  # 0.2 ms estimated for 1 GFlop: the worst
        text = H.render_matmuls(rows, top=3, fams=2)
        self.assertIn("matmul fusions: 5", text)
        self.assertIn("fusion.ident", text)
        self.assertIn("fwd bndt,de->bnte", text)

    def test_families_group_by_direction_einsum_and_shapes(self):
        fam = H.families(H.matmul_fusions(HLO_MM))
        self.assertEqual(len(fam), 5)
        self.assertAlmostEqual(sum(f[2] for f in fam), sum(r["est_ms"] for r in self.rows.values()), places=9)

    def test_bytes_count_the_padding_of_the_tiles(self):
        r = self.rows["fusion.ident"]  # bf16[4,16,64,2048] + bf16[64,64,1,1] in, bf16[4,16,2048,64] out: the 64-wide minor dimension is padded to 128 lanes
        a = H.parse_array("bf16[4,16,2048,64]{3,2,1,0:T(8,128)(2,1)}")
        self.assertEqual((a["bytes"], a["padded_bytes"]), (4 * 16 * 2048 * 64 * 2, 4 * 16 * 2048 * 128 * 2))
        self.assertGreater(r["bytes"], 2 * a["bytes"])
        self.assertAlmostEqual(r["hbm_ms"], r["bytes"] / 819e9 * 1e3, places=9)

    def test_parse_array(self):
        a = H.parse_array("bf16[64,64]{0,1:T(8,128)(2,1)S(1)}")
        self.assertEqual((a["dtype"], a["dims"], a["m2m"], a["tile"], a["space"]), ("bf16", (64, 64), (0, 1), "T(8,128)(2,1)", "S(1)"))
        self.assertEqual(a["padded_bytes"], 128 * 64 * 2)  # dim 0 is the minor one: 64 -> 128 lanes
        self.assertEqual(H.parse_array("f32[4,2048]{1,0:T(4,128)}")["padded_bytes"], 4 * 2048 * 4)
        self.assertEqual(H.parse_array("f32[]{:T(128)}")["bytes"], 4)
        self.assertEqual(H.short_array("f32[4,2048,1024]{1,2,0:T(8,128)S(1)}"), "f32[4,2048,1024]{1,2,0}S1")
        self.assertIsNone(H.parse_array("token[]"))

    def test_valid_pairs(self):
        self.assertEqual(H.valid_pairs(4, 4, 1), 4)  # a size-1 window: every position reads its own element
        self.assertEqual(H.valid_pairs(4, 1, 4), 4)  # a window as large as the input, one output: the sum over the 4
        self.assertEqual(H.valid_pairs(6, 6, 6, stride=5, lhs_dilate=6), 6)  # the batch trick: one real tap per output
        self.assertEqual(H.valid_pairs(5, 5, 3, pad=(1, 1)), 13)  # a size-3 window with padding 1: 5 outputs, the two edge outputs read 2 elements

    def test_einsum_of(self):
        self.assertEqual(H.einsum_of("jit(step_fn)/jvp(oe,bte->bot)/dot_general"), ("oe,bte->bot", "fwd"))
        self.assertEqual(H.einsum_of("jit(step_fn)/transpose(jvp(jvp()))/checkpoint/rematted_computation/bndt,de->bnte/dot_general"), ("bndt,de->bnte", "remat"))
        self.assertEqual(H.einsum_of("jit(step_fn)/jvp()/mul"), ("", "fwd"))

    def test_bare_convolution_and_dot_in_the_entry_computation(self):
        text = "\n".join([
            "HloModule bare",
            "",
            "ENTRY %main (a: f32[16,32], b: f32[32,8]) -> f32[16,8] {",
            "  %a = f32[16,32]{1,0} parameter(0)",
            "  %b = f32[32,8]{1,0} parameter(1)",
            "  %dot.1 = f32[16,8]{1,0} dot(%a, %b), lhs_contracting_dims={1}, rhs_contracting_dims={0}, metadata={op_name=\"jit(f)/jvp()/dot_general\"}, backend_config=" + cfg(6_000),
            "  ROOT %t = (f32[16,8]{1,0}) tuple(%dot.1)",
            "}",
        ])
        (r,) = H.matmul_fusions(text)
        self.assertEqual((r["name"], r["flops"]), ("dot.1", 2 * 16 * 8 * 32))
        self.assertEqual(r["operands"], ["f32[16,32]{1,0}", "f32[32,8]{1,0}"])

    def test_operands_with_index_comments(self):
        # the printer writes /*index=5*/ in front of every fifth operand: those operands must still be found (their layout and size count)
        text = HLO_MM.replace("fusion(%a, %w), kind=kOutput, calls=%fused_computation.10", "fusion(%a, /*index=1*/%w), kind=kOutput, calls=%fused_computation.10")
        rows = {r["name"]: r for r in H.matmul_fusions(text)}
        self.assertEqual(rows["fusion.proj"]["operands"], ["bf16[4,2048,1024]{2,1,0}S1", "f32[1024,1024]{1,0}"])
        self.assertEqual(rows["fusion.proj"]["operand_bytes"], [4 * 2048 * 1024 * 2, 1024 * 1024 * 4])

    def test_text_without_an_entry_is_an_error(self):
        with self.assertRaises(ValueError):
            H.matmul_fusions("HloModule empty\n")


if __name__ == "__main__":
    unittest.main()
