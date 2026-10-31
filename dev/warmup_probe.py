"""Diagnostic-only warmup/first-step timing wrapper around unchanged harness/_train.py.

Called with the same --config/--metrics/--solution arguments as _train.py. This adds
only a wall-clock timestamp callback; official A/B scores run _train.py directly.
"""
import runpy
from pathlib import Path
import sys

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
import transformers
from solution import warmup


class Probe(transformers.TrainerCallback):
    def on_train_begin(self,args,state,control,**kwargs):
        warmup._mark("train_begin")

    def on_step_begin(self,args,state,control,**kwargs):
        if state.global_step==0:
            warmup._mark("first_step_begin")

    def on_step_end(self,args,state,control,**kwargs):
        if state.global_step==1:
            warmup._mark("first_step_end")


original=transformers.Trainer.__init__
def init(self,*args,**kwargs):
    original(self,*args,**kwargs)
    self.add_callback(Probe())
transformers.Trainer.__init__=init
runpy.run_path(str(REPO/"harness/_train.py"),run_name="__main__")
