import os.path as osp

from mmdet.core import DistEvalHook as _DistEvalHook
from mmdet.core import EvalHook as _EvalHook


class EvalHook(_EvalHook):
    """Please refer to `mmdet.core.evaluation.eval_hooks.py:EvalHook` for
    detailed docstring."""

    def after_train_epoch(self, runner):
        if not self._should_evaluate(runner):
            return
        if hasattr(self.dataloader.dataset,
                   'load_as_video') and self.dataloader.dataset.load_as_video:
            from mmtrack.apis import single_gpu_test
        else:
            from mmdet.apis import single_gpu_test
        results = single_gpu_test(runner.model, self.dataloader, show=False)
        self.evaluate(runner, results)


class DistEvalHook(_DistEvalHook):
    """Please refer to `mmdet.core.evaluation.eval_hooks.py:DistEvalHook` for
    detailed docstring."""

    def after_train_epoch(self, runner):
        if not self._should_evaluate(runner):
            return
        if hasattr(self.dataloader.dataset,
                   'load_as_video') and self.dataloader.dataset.load_as_video:
            from mmtrack.apis import multi_gpu_test
        else:
            from mmdet.apis import multi_gpu_test
        tmpdir = self.tmpdir
        if tmpdir is None:
            tmpdir = osp.join(runner.work_dir, '.eval_hook')
        results = multi_gpu_test(
            runner.model,
            self.dataloader,
            tmpdir=tmpdir,
            gpu_collect=self.gpu_collect)
        if runner.rank == 0:
            print('\n')
            self.evaluate(runner, results)
