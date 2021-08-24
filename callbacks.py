
from ray.rllib.agents.callbacks import DefaultCallbacks
from evaluation import generate_history, is_t4t

class MA_ist4t(DefaultCallbacks):

    def on_train_result(self, *, trainer, result: dict, **kwargs) -> None:
        """Called at the end of Trainable.train().

        Args:
            trainer (Trainer): Current trainer instance.
            result (dict): Dict of results returned from trainer.train() call.
                You can mutate this object to add additional metrics.
            kwargs: Forward compatibility placeholder.
        """

        if self.legacy_callbacks.get("on_train_result"):
            self.legacy_callbacks["on_train_result"]({
                "trainer": trainer,
                "result": result,
            })
        result['t4t_frac_agent-0'], result['coop_frac_agent-0'] = is_t4t(trainer,n_samples=100, policy_id='agent-0')
        result['t4t_frac_agent-1'], result['coop_frac_agent-1'] = is_t4t(trainer,n_samples=100, policy_id='agent-1')

class SA_ist4t(DefaultCallbacks):

    def on_train_result(self, *, trainer, result: dict, **kwargs) -> None:
        """Called at the end of Trainable.train().

        Args:
            trainer (Trainer): Current trainer instance.
            result (dict): Dict of results returned from trainer.train() call.
                You can mutate this object to add additional metrics.
            kwargs: Forward compatibility placeholder.
        """

        if self.legacy_callbacks.get("on_train_result"):
            self.legacy_callbacks["on_train_result"]({
                "trainer": trainer,
                "result": result,
            })
        result['t4t_frac_agent'], result['coop_frac_agent'] = is_t4t(trainer,n_samples=100)

        