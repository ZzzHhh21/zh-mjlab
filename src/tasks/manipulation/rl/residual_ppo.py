from __future__ import annotations

from typing import Generator

import torch
from rsl_rl.algorithms.ppo import PPO
from rsl_rl.storage import RolloutStorage


class ResidualMaskedPPO(PPO):
  """PPO variant that ignores frozen-base steps during residual training."""

  def __init__(
    self,
    *args,
    residual_std_min: float = 0.05,
    residual_std_max: float = 1.0,
    **kwargs,
  ) -> None:
    super().__init__(*args, **kwargs)
    self.residual_std_min = float(residual_std_min)
    self.residual_std_max = float(residual_std_max)
    self._clamp_actor_std()

  def process_env_step(
    self,
    obs,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    extras: dict[str, torch.Tensor],
  ) -> None:
    train_mask = extras.get("residual_train_mask")
    if train_mask is None:
      train_mask = torch.ones_like(dones, dtype=torch.bool, device=self.device)
    train_mask = train_mask.to(self.device, dtype=torch.bool).view(-1, 1)

    if train_mask.any():
      active_obs = obs[train_mask.squeeze(-1)]
      self.actor.update_normalization(active_obs)
      self.critic.update_normalization(active_obs)
      if self.rnd:
        self.rnd.update_normalization(active_obs)

    self.transition.rewards = rewards.clone()
    self.transition.dones = dones

    if self.rnd:
      self.intrinsic_rewards = self.rnd.get_intrinsic_reward(obs)
      self.transition.rewards += self.intrinsic_rewards

    if "time_outs" in extras:
      self.transition.rewards += self.gamma * torch.squeeze(
        self.transition.values * extras["time_outs"].unsqueeze(1).to(self.device),
        1,
      )

    step_idx = self.storage.step
    self._ensure_train_masks()
    self.storage.add_transition(self.transition)
    self.storage.residual_train_masks[step_idx].copy_(train_mask.float())
    self.transition.clear()
    self.actor.reset(dones)
    self.critic.reset(dones)

  def compute_returns(self, obs) -> None:
    st = self.storage
    self._ensure_train_masks()
    train_masks = st.residual_train_masks

    last_values = self.critic(obs).detach()
    advantage = torch.zeros(st.num_envs, 1, device=self.device)
    for step in reversed(range(st.num_transitions_per_env)):
      next_values = last_values if step == st.num_transitions_per_env - 1 else st.values[step + 1]
      next_is_not_terminal = 1.0 - st.dones[step].float()
      delta = st.rewards[step] + next_is_not_terminal * self.gamma * next_values - st.values[step]
      advantage = delta + next_is_not_terminal * self.gamma * self.lam * advantage
      advantage = advantage * train_masks[step]
      st.returns[step] = st.values[step] + advantage

    st.advantages = st.returns - st.values
    active = train_masks.bool()
    st.advantages[~active] = 0.0
    if not self.normalize_advantage_per_mini_batch and active.any():
      active_advantages = st.advantages[active]
      st.advantages[active] = self._normalize_advantages(active_advantages)

  def update(self) -> dict[str, float]:
    mean_value_loss = 0.0
    mean_surrogate_loss = 0.0
    mean_entropy = 0.0
    mean_rnd_loss = 0.0 if self.rnd else None
    mean_symmetry_loss = 0.0 if self.symmetry else None
    num_updates = 0

    if self.actor.is_recurrent or self.critic.is_recurrent:
      raise NotImplementedError("ResidualMaskedPPO currently supports feedforward policies only.")

    for batch, train_mask in self._masked_mini_batch_generator(
      self.num_mini_batches, self.num_learning_epochs
    ):
      active = train_mask.squeeze(-1).bool()
      if not active.any():
        continue
      batch = self._filter_batch(batch, active)
      original_batch_size = batch.observations.batch_size[0]

      if self.normalize_advantage_per_mini_batch:
        with torch.no_grad():
          batch.advantages = self._normalize_advantages(batch.advantages)

      if self.symmetry and self.symmetry["use_data_augmentation"]:
        data_augmentation_func = self.symmetry["data_augmentation_func"]
        batch.observations, batch.actions = data_augmentation_func(
          env=self.symmetry["_env"],
          obs=batch.observations,
          actions=batch.actions,
        )
        num_aug = int(batch.observations.batch_size[0] / original_batch_size)
        batch.old_actions_log_prob = batch.old_actions_log_prob.repeat(num_aug, 1)
        batch.values = batch.values.repeat(num_aug, 1)
        batch.advantages = batch.advantages.repeat(num_aug, 1)
        batch.returns = batch.returns.repeat(num_aug, 1)

      self.actor(
        batch.observations,
        masks=batch.masks,
        hidden_state=batch.hidden_states[0],
        stochastic_output=True,
      )
      actions_log_prob = self.actor.get_output_log_prob(batch.actions)
      values = self.critic(
        batch.observations,
        masks=batch.masks,
        hidden_state=batch.hidden_states[1],
      )
      distribution_params = tuple(p[:original_batch_size] for p in self.actor.output_distribution_params)
      entropy = self.actor.output_entropy[:original_batch_size]

      if self.desired_kl is not None and self.schedule == "adaptive":
        with torch.inference_mode():
          kl = self.actor.get_kl_divergence(batch.old_distribution_params, distribution_params)
          kl_mean = torch.mean(kl)
          if self.is_multi_gpu:
            torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
            kl_mean /= self.gpu_world_size
          if self.gpu_global_rank == 0:
            if kl_mean > self.desired_kl * 2.0:
              self.learning_rate = max(1e-5, self.learning_rate / 1.5)
            elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
              self.learning_rate = min(1e-2, self.learning_rate * 1.5)
          if self.is_multi_gpu:
            lr_tensor = torch.tensor(self.learning_rate, device=self.device)
            torch.distributed.broadcast(lr_tensor, src=0)
            self.learning_rate = lr_tensor.item()
          for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.learning_rate

      ratio = torch.exp(actions_log_prob - torch.squeeze(batch.old_actions_log_prob))
      advantages = torch.squeeze(batch.advantages)
      surrogate = -advantages * ratio
      surrogate_clipped = -advantages * torch.clamp(
        ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
      )
      surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

      if self.use_clipped_value_loss:
        value_clipped = batch.values + (values - batch.values).clamp(
          -self.clip_param, self.clip_param
        )
        value_losses = (values - batch.returns).pow(2)
        value_losses_clipped = (value_clipped - batch.returns).pow(2)
        value_loss = torch.max(value_losses, value_losses_clipped).mean()
      else:
        value_loss = (batch.returns - values).pow(2).mean()

      loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()

      if self.symmetry:
        if not self.symmetry["use_data_augmentation"]:
          data_augmentation_func = self.symmetry["data_augmentation_func"]
          batch.observations, _ = data_augmentation_func(
            obs=batch.observations,
            actions=None,
            env=self.symmetry["_env"],
          )
        mean_actions = self.actor(batch.observations.detach().clone())
        action_mean_orig = mean_actions[:original_batch_size]
        _, actions_mean_symm = data_augmentation_func(
          obs=None,
          actions=action_mean_orig,
          env=self.symmetry["_env"],
        )
        mse_loss = torch.nn.MSELoss()
        symmetry_loss = mse_loss(
          mean_actions[original_batch_size:],
          actions_mean_symm.detach()[original_batch_size:],
        )
        if self.symmetry["use_mirror_loss"]:
          loss += self.symmetry["mirror_loss_coeff"] * symmetry_loss
        else:
          symmetry_loss = symmetry_loss.detach()

      if self.rnd:
        with torch.no_grad():
          rnd_state = self.rnd.get_rnd_state(batch.observations[:original_batch_size])
          rnd_state = self.rnd.state_normalizer(rnd_state)
        predicted_embedding = self.rnd.predictor(rnd_state)
        target_embedding = self.rnd.target(rnd_state).detach()
        rnd_loss = torch.nn.MSELoss()(predicted_embedding, target_embedding)

      self.optimizer.zero_grad()
      loss.backward()
      if self.rnd:
        self.rnd_optimizer.zero_grad()
        rnd_loss.backward()

      if self.is_multi_gpu:
        self.reduce_parameters()

      torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
      torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
      self.optimizer.step()
      self._clamp_actor_std()
      if self.rnd_optimizer:
        self.rnd_optimizer.step()

      mean_value_loss += value_loss.item()
      mean_surrogate_loss += surrogate_loss.item()
      mean_entropy += entropy.mean().item()
      if mean_rnd_loss is not None:
        mean_rnd_loss += rnd_loss.item()
      if mean_symmetry_loss is not None:
        mean_symmetry_loss += symmetry_loss.item()
      num_updates += 1

    self.storage.clear()

    if num_updates == 0:
      loss_dict = {"value": 0.0, "surrogate": 0.0, "entropy": 0.0}
      if self.rnd:
        loss_dict["rnd"] = 0.0
      if self.symmetry:
        loss_dict["symmetry"] = 0.0
      return loss_dict

    loss_dict = {
      "value": mean_value_loss / num_updates,
      "surrogate": mean_surrogate_loss / num_updates,
      "entropy": mean_entropy / num_updates,
    }
    if self.rnd:
      loss_dict["rnd"] = mean_rnd_loss / num_updates
    if self.symmetry:
      loss_dict["symmetry"] = mean_symmetry_loss / num_updates
    return loss_dict

  def _ensure_train_masks(self) -> None:
    if hasattr(self.storage, "residual_train_masks"):
      return
    self.storage.residual_train_masks = torch.ones(
      self.storage.num_transitions_per_env,
      self.storage.num_envs,
      1,
      device=self.device,
    )

  def _masked_mini_batch_generator(
    self,
    num_mini_batches: int,
    num_epochs: int,
  ) -> Generator[tuple[RolloutStorage.Batch, torch.Tensor], None, None]:
    self._ensure_train_masks()
    st = self.storage
    batch_size = st.num_envs * st.num_transitions_per_env
    mini_batch_size = batch_size // num_mini_batches
    indices = torch.randperm(num_mini_batches * mini_batch_size, device=self.device)

    observations = st.observations.flatten(0, 1)
    actions = st.actions.flatten(0, 1)
    values = st.values.flatten(0, 1)
    returns = st.returns.flatten(0, 1)
    old_actions_log_prob = st.actions_log_prob.flatten(0, 1)
    advantages = st.advantages.flatten(0, 1)
    old_distribution_params = tuple(p.flatten(0, 1) for p in st.distribution_params)
    train_masks = st.residual_train_masks.flatten(0, 1)

    for _ in range(num_epochs):
      for i in range(num_mini_batches):
        start = i * mini_batch_size
        stop = (i + 1) * mini_batch_size
        batch_idx = indices[start:stop]
        yield (
          RolloutStorage.Batch(
            observations=observations[batch_idx],
            actions=actions[batch_idx],
            values=values[batch_idx],
            advantages=advantages[batch_idx],
            returns=returns[batch_idx],
            old_actions_log_prob=old_actions_log_prob[batch_idx],
            old_distribution_params=tuple(p[batch_idx] for p in old_distribution_params),
          ),
          train_masks[batch_idx],
        )

  @staticmethod
  def _filter_batch(
    batch: RolloutStorage.Batch,
    active: torch.Tensor,
  ) -> RolloutStorage.Batch:
    return RolloutStorage.Batch(
      observations=batch.observations[active],
      actions=batch.actions[active],
      values=batch.values[active],
      advantages=batch.advantages[active],
      returns=batch.returns[active],
      old_actions_log_prob=batch.old_actions_log_prob[active],
      old_distribution_params=tuple(p[active] for p in batch.old_distribution_params),
      hidden_states=batch.hidden_states,
      masks=batch.masks,
    )

  @staticmethod
  def _normalize_advantages(advantages: torch.Tensor) -> torch.Tensor:
    if advantages.numel() <= 1:
      return torch.zeros_like(advantages)
    return (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

  def _clamp_actor_std(self) -> None:
    distribution = getattr(self.actor, "distribution", None)
    if distribution is None:
      return
    with torch.no_grad():
      if hasattr(distribution, "std_param"):
        distribution.std_param.clamp_(self.residual_std_min, self.residual_std_max)
      elif hasattr(distribution, "log_std_param"):
        min_log = torch.log(
          torch.tensor(self.residual_std_min, device=distribution.log_std_param.device)
        )
        max_log = torch.log(
          torch.tensor(self.residual_std_max, device=distribution.log_std_param.device)
        )
        distribution.log_std_param.clamp_(min_log, max_log)
