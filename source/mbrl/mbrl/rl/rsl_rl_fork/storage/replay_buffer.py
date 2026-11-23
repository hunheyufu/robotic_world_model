# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch


class ReplayBuffer:
    """Fixed-size buffer to store per-environment transition streams.

    [vendor patch:replay-time-order] The buffer is a ring buffer over a per-environment stream of
    transitions.  Two properties are required for the system-dynamics windows and were not
    guaranteed by the upstream implementation:

    * **Chronological order.**  Once the ring wrapped, the storage order is rotated, so a contiguous
      index slice is *not* a contiguous time span.  Every read goes through
      :meth:`_chronological`, which rotates the entries back into insertion order.
    * **Episode boundaries.**  A window must not span an environment reset: the state right after a
      reset is not produced by the previously applied action.  Resets are recorded from the
      environment's real ``dones`` signal (``terminated | truncated``) and are kept in a separate
      :attr:`reset_buf`, so the learning inputs (state, action, extension, contact, termination)
      keep their documented column meaning.
    """

    def __init__(self, dim, buffer_size, device):
        """Initialize a ReplayBuffer object."""
        self.dim = dim
        self.buffer_size = buffer_size
        self.device = device

        self.replay_buf = None
        self.reset_buf = None
        self.num_envs = None
        self.step = 0
        # number of transitions currently addressable per environment (<= buffer_size)
        self.num_transitions = 0
        # total number of transitions ever inserted (used to detect wrap-around)
        self.num_inserted = 0

    def _initialize_buffer(self, num_envs):
        """Initialize the replay buffer dynamically."""
        self.num_envs = num_envs
        if isinstance(self.dim, list):
            self.replay_buf = [
                torch.zeros(num_envs, self.buffer_size, d, device=self.device)
                if d > 0 else None for d in self.dim
            ]
        else:
            self.replay_buf = torch.zeros(num_envs, self.buffer_size, self.dim, device=self.device)
        self.reset_buf = torch.zeros(num_envs, self.buffer_size, dtype=torch.bool, device=self.device)

    def insert(self, input_buf, reset_flags=None):
        """Insert new transitions in a circular manner.

        Args:
            input_buf: Tensor of shape ``[num_envs, num_inputs, dim]`` (or a list of such tensors for
                the multi-head layout) holding the new transitions.
            reset_flags: Optional per-transition reset signal of shape ``[num_envs, num_inputs]``
                (or ``[num_envs]`` for a single transition).  ``True`` marks a transition after
                which the environment was reset.  Defaults to "no reset".
        """
        if self.replay_buf is None:
            self._initialize_buffer(input_buf[0].shape[0] if isinstance(input_buf, list) else input_buf.shape[0])

        num_inputs = (
            input_buf[0].shape[1] if isinstance(input_buf, list) else input_buf.shape[1]
        )
        if num_inputs > self.buffer_size:
            raise ValueError(
                f"Cannot insert {num_inputs} transitions into a replay buffer of size {self.buffer_size}."
            )
        if reset_flags is None:
            flags = torch.zeros(self.num_envs, num_inputs, dtype=torch.bool, device=self.device)
        else:
            flags = torch.as_tensor(reset_flags, device=self.device).to(torch.bool).reshape(self.num_envs, num_inputs)

        def _insert_into_buffer(r_buf, i_buf):
            end_idx = self.step + num_inputs
            if end_idx > self.buffer_size:
                r_buf[:, self.step:self.buffer_size] = i_buf[:, :self.buffer_size - self.step]
                r_buf[:, :end_idx - self.buffer_size] = i_buf[:, self.buffer_size - self.step:]
            else:
                r_buf[:, self.step:end_idx] = i_buf

        if isinstance(self.replay_buf, list):
            for r_buf, i_buf in zip(self.replay_buf, input_buf):
                if r_buf is not None:
                    _insert_into_buffer(r_buf, i_buf)
        else:
            _insert_into_buffer(self.replay_buf, input_buf)
        _insert_into_buffer(self.reset_buf, flags)

        self.num_inserted += num_inputs
        self.num_transitions = min(self.buffer_size, self.num_inserted)
        self.step = (self.step + num_inputs) % self.buffer_size

    def _chronological(self, buf):
        """Return *buf* rotated into insertion order, truncated to the filled entries."""
        if self.num_transitions == 0:
            return buf[:, :0]
        if self.num_transitions < self.buffer_size:
            # the buffer did not wrap yet: entries occupy [0, num_transitions) in order
            return buf[:, :self.num_transitions]
        if self.step == 0:
            return buf
        # ``self.step`` is the index of the oldest entry once the ring is full
        return torch.cat([buf[:, self.step:], buf[:, :self.step]], dim=1)

    def _chronological_buffers(self):
        """Return the learning inputs in chronological order (list or single tensor)."""
        if isinstance(self.replay_buf, list):
            return [None if buf is None else self._chronological(buf) for buf in self.replay_buf]
        return self._chronological(self.replay_buf)

    def num_valid_sequences(self, sequence_length):
        """Number of ``sequence_length`` windows that do not span an environment reset."""
        if self.replay_buf is None or self.num_transitions < sequence_length:
            return 0
        return int(self._valid_start_indices(sequence_length)[0].numel())

    def _valid_start_indices(self, sequence_length):
        """Return ``(env_indices, start_indices)`` for windows without an episode boundary."""
        reset_flags = self._chronological(self.reset_buf)  # [num_envs, num_transitions]
        # A window [start, start + sequence_length) is usable only when no transition inside it is
        # the first observation of a new episode.
        windows = reset_flags.unfold(1, sequence_length, 1)  # [num_envs, starts, sequence_length]
        valid_mask = ~windows.any(dim=2)
        return torch.where(valid_mask)

    def mini_batch_generator(self, sequence_length, num_mini_batch, mini_batch_size, strict=False):
        """Yield mini-batches of contiguous, episode-consistent sequences.

        Args:
            sequence_length: Number of transitions per sequence.
            num_mini_batch: Number of batches to yield.
            mini_batch_size: Number of sequences per batch (sampled with replacement).
            strict: Raise :class:`RuntimeError` instead of yielding nothing when no valid sequence
                exists.  Evaluation and imagination rollouts use ``strict=True`` so a missing
                window can never be replaced by zero padding; training skips the update instead.
        """
        assert self.replay_buf is not None, "Replay buffer is not initialized."
        sequence_length = int(sequence_length)
        num_valid = self.num_valid_sequences(sequence_length)
        if num_valid == 0:
            message = (
                f"No valid sequence of length {sequence_length} in the system-dynamics replay buffer "
                f"({self.num_transitions} transitions stored, buffer size {self.buffer_size}). "
                "A window must neither be zero-padded nor cross an environment reset."
            )
            if strict:
                raise RuntimeError(message)
            print(f"[WARN] {message} Skipping the system-dynamics update for this iteration.", flush=True)
            return

        replay_buf = self._chronological_buffers()
        env_indices, start_indices = self._valid_start_indices(sequence_length)
        offsets = torch.arange(sequence_length, device=self.device)

        for _ in range(num_mini_batch):
            sampled_idxs = torch.randint(0, num_valid, (mini_batch_size,), device=self.device)
            sampled_envs = env_indices[sampled_idxs]
            sampled_starts = start_indices[sampled_idxs]
            if isinstance(replay_buf, list):
                yield [
                    buf[sampled_envs[:, None], sampled_starts[:, None] + offsets] if buf is not None else None
                    for buf in replay_buf
                ]
            else:
                yield replay_buf[sampled_envs[:, None], sampled_starts[:, None] + offsets]
