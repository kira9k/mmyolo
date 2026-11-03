# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.hooks import CheckpointHook

from mmyolo.registry import HOOKS


@HOOKS.register_module()
class LastEpochCheckpointHook(CheckpointHook):
    """Save every epoch near the end, otherwise use the regular interval.

    Args:
        save_last_epochs (int): Number of final epochs to save individually.
            Uses the runner's actual maximum epochs, including CLI overrides.
            Set to zero to keep only the regular checkpoint schedule.
        save_only_best (bool): Disable ordinary epoch/iteration checkpoints.
            Best weights are still saved after validation via ``save_best``.
        **kwargs: Arguments passed to :class:`CheckpointHook`, including
            retention, best-checkpoint selection and optimizer state saving.
    """

    def __init__(self,
                 save_last_epochs: int = 10,
                 save_only_best: bool = False,
                 **kwargs):
        if (not isinstance(save_last_epochs, int)
                or isinstance(save_last_epochs, bool) or save_last_epochs < 0):
            raise ValueError('save_last_epochs must be a non-negative integer')
        super().__init__(**kwargs)
        self.save_last_epochs = save_last_epochs
        if not isinstance(save_only_best, bool):
            raise ValueError('save_only_best must be a boolean')
        if save_only_best and not self.save_best:
            raise ValueError('save_only_best requires save_best')
        self.save_only_best = save_only_best

    def after_train_epoch(self, runner) -> None:
        if self.save_only_best:
            return
        epoch = runner.epoch + 1
        if (self.by_epoch and self.save_last_epochs > 0
                and epoch > runner.max_epochs - self.save_last_epochs
                and epoch >= self.save_begin):
            runner.logger.info(f'Saving checkpoint at {epoch} epochs')
            self._save_checkpoint(runner)
        else:
            super().after_train_epoch(runner)

    def after_train_iter(self,
                         runner,
                         batch_idx,
                         data_batch=None,
                         outputs=None):
        if not self.save_only_best:
            super().after_train_iter(runner, batch_idx, data_batch, outputs)
