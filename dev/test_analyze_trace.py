"""Synthetic timing checks for diagnostic attribution (no torch/GPU required)."""
import unittest

from analyze_trace import analyze


def event(name,start,duration,cat="cpu_op",tid=1):
    return {"ph":"X","name":name,"ts":start,"dur":duration,"cat":cat,"pid":1,"tid":tid}


class TraceAnalysisTest(unittest.TestCase):
    def test_stream_union_and_exclusive_cpu_leaf(self):
        events = [event("SKC_TRACE_STEP_11",0,2000,"user_annotation"),
                  event("gpu_a",200,500,"kernel",2),event("gpu_b",400,500,"kernel",3),
                  event("outer",0,2000,"python_function"),
                  event("aten::item",900,900),event("cudaMemcpyAsync",1000,300,"cuda_runtime")]
        r=analyze(events,0.2)
        self.assertAlmostEqual(r["gpu_busy_ms_per_step"],0.7)
        self.assertAlmostEqual(r["all_idle_ms_per_step"],1.3)
        # Initial 0.2 ms gap equals threshold and is deliberately excluded.
        self.assertAlmostEqual(r["gaps_over_threshold_ms_per_step"],1.1)
        totals={x["name"]:x["ms_per_step"] for x in r["top_causes"]}
        self.assertAlmostEqual(totals["cudaMemcpyAsync"],0.3)
        self.assertAlmostEqual(totals["aten::item"],0.6)
        self.assertAlmostEqual(totals["outer"],0.2)

    def test_missing_cpu_is_retained_and_normalized(self):
        r=analyze([event("SKC_TRACE_STEP_11",0,1000,"user_annotation"),
                   event("SKC_TRACE_STEP_12",2000,1000,"user_annotation")])
        self.assertEqual(r["active_steps"],2)
        self.assertEqual(r["top_causes"][0]["kind"],"unattributed")
        self.assertAlmostEqual(r["top_causes"][0]["ms_per_step"],1)

    def test_other_cpu_thread_is_not_attributed(self):
        r=analyze([event("SKC_TRACE_STEP_11",0,1000,"user_annotation"),
                   event("background",0,1000,"cpu_op",9)])
        self.assertEqual(r["top_causes"][0]["kind"],"unattributed")

    def test_gpu_user_annotation_copy_is_not_an_extra_step(self):
        r=analyze([event("SKC_TRACE_STEP_11",0,1000,"user_annotation"),
                   event("SKC_TRACE_STEP_11",100,800,"gpu_user_annotation",7),
                   event("work",100,200,"kernel",7)])
        self.assertEqual(r["active_steps"],1)
        self.assertAlmostEqual(r["gpu_busy_ms_per_step"],0.2)

    def test_duplicate_cpu_labels_are_rejected(self):
        with self.assertRaises(ValueError):
            analyze([event("SKC_TRACE_STEP_11",0,1000,"user_annotation"),
                     event("SKC_TRACE_STEP_11",2000,1000,"user_annotation")])

    def test_autograd_launch_thread_is_included(self):
        r=analyze([event("SKC_TRACE_STEP_11",0,1000,"user_annotation"),
                   event("run_backward",0,1000,"python_function"),
                   event("BackwardFn",0,1000,"cpu_op",8),
                   event("cudaLaunchKernel",400,100,"cuda_runtime",8)])
        self.assertEqual(r["cpu_threads"],["8"])
        totals={x["name"]:x["ms_per_step"] for x in r["top_causes"]}
        self.assertAlmostEqual(totals["BackwardFn"],0.9)
        self.assertAlmostEqual(totals["cudaLaunchKernel"],0.1)


if __name__ == "__main__":
    unittest.main()
