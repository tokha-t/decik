"""Warm-up lifecycle checks without importing torch or creating CUDA contexts."""
import importlib.util
import os
from pathlib import Path
import unittest
from unittest.mock import Mock, patch


def module():
    spec=importlib.util.spec_from_file_location("warmup_test",Path(__file__).resolve().parents[1]/"solution/warmup.py")
    loaded=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


class WarmupProcessTest(unittest.TestCase):
    def test_shards_cover_every_kernel_once(self):
        w=module()
        for shards in (1,2):
            for name in w._KERNELS:
                kernel=Mock();kernel.__name__=name
                for shard in range(shards):
                    w._compile(kernel,42,shard=shard,shards=shards,grid=(1,))
                kernel.warmup.assert_called_once_with(42,grid=(1,))

    def test_worker_failure_retries_in_parent_and_preserves_cache_env(self):
        w=module();proc=Mock();proc.wait.return_value=1;proc.poll.return_value=1
        with patch.object(w.subprocess,"Popen",return_value=proc) as spawn, \
             patch.object(w,"_work",return_value=True) as parent, \
             patch.dict(os.environ,{"TRITON_CACHE_DIR":"/tmp/this-run-cache"}):
            w._process_work(1)
            self.assertEqual(spawn.call_args.kwargs["env"]["TRITON_CACHE_DIR"],"/tmp/this-run-cache")
            self.assertEqual(spawn.call_args.args[0][-2:],["0","1"])
            parent.assert_called_once_with()

    def test_timeout_reaps_child_before_parent_fallback(self):
        w=module();proc=Mock();proc.wait.side_effect=[TimeoutError(),0];proc.poll.return_value=None
        with patch.object(w.subprocess,"Popen",return_value=proc),patch.object(w,"_work") as parent:
            w._process_work(1)
            proc.kill.assert_called_once_with()
            self.assertEqual(proc.wait.call_count,2)
            parent.assert_called_once_with()

    def test_background_sets_done_on_failure(self):
        w=module()
        with patch.object(w,"_process_work",side_effect=RuntimeError("failed")):
            with self.assertRaises(RuntimeError):w._background(2)
        self.assertTrue(w._done.is_set())

    def test_start_is_idempotent_and_rejects_invalid_modes(self):
        for value in ("-1","3","bad"):
            w=module()
            with patch.dict(os.environ,{"SKC_WARMUP_PROCESSES":value}), \
                 patch.object(w.threading,"Thread") as thread, patch.object(w.atexit,"register"):
                w.start();w.start()
                thread.assert_called_once()
                self.assertEqual(thread.call_args.kwargs["args"],(0,))


if __name__=="__main__":unittest.main()
