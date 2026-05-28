from typing import NamedTuple, Iterator
from pathlib import Path
from PIL import Image
import torch
from jaxtyping import Int
from abc import ABC, abstractmethod
from torch.utils.data import Dataset
from typing import TypeVar, Generic
import random
from torch.utils.data import Sampler, IterableDataset, get_worker_info

# Domain-agnostic type aliases
IdentityId = int
SequenceId = int
FrameId = str

class VideoReIDItem(NamedTuple):
    sample_index: int

    identity_id: IdentityId
    sequence_id: SequenceId
    frame_id: FrameId
    
    frame_path: Path
    frame: Image.Image | None
    frame_tensor: torch.Tensor | None

    segmentation_path: Path | None
    segmentation_tensor: torch.Tensor | None

class VideoReIDBatch(NamedTuple):
    sample_indices: Int[torch.Tensor, "b"]
    
    identity_ids: Int[torch.Tensor, "b"]
    sequence_ids: Int[torch.Tensor, "b"]
    frame_ids: list[FrameId]
    
    frame_paths: list[Path]
    frames: list[Image.Image] | list[None]
    frame_tensors: list[torch.Tensor] | list[None]

    segmentation_paths: list[Path] | list[None]
    segmentations: list[torch.Tensor] | list[None]

def collate_video_reid_ds(batch: list[VideoReIDItem]) -> VideoReIDBatch:
    sample_indices = torch.tensor([item.sample_index for item in batch], dtype=torch.long)

    identity_ids = torch.tensor([item.identity_id for item in batch], dtype=torch.long)
    sequence_ids = torch.tensor([item.sequence_id for item in batch], dtype=torch.long)
    frame_ids = [item.frame_id for item in batch]
    
    frame_paths = [item.frame_path for item in batch]
    frames = [item.frame for item in batch]
    frame_tensors = [item.frame_tensor for item in batch]

    segmentation_paths = [item.segmentation_path for item in batch]
    segmentations = [item.segmentation_tensor for item in batch]
    
    return VideoReIDBatch(
        sample_indices=sample_indices,
        identity_ids=identity_ids, 
        sequence_ids=sequence_ids, 
        frame_ids=frame_ids, 
        frame_paths=frame_paths, 
        frames=frames, 
        frame_tensors=frame_tensors, 
        segmentation_paths=segmentation_paths, 
        segmentations=segmentations
    )

T_co = TypeVar('T_co', covariant=True)

class TypedDataset(Dataset, ABC, Generic[T_co]):
    """
    An Abstract Base Class for PyTorch datasets that enforces strict
    type hinting for the __getitem__ return value across subclasses.
    """
    
    @abstractmethod
    def __getitem__(self, index: int) -> T_co:
        """Must be implemented by subclasses to return type T_co."""
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Must be implemented by subclasses."""
        pass

class AbstractVideoReIDDataset(TypedDataset[VideoReIDItem]):
    """
    The strict contract for ALL Video ReID datasets (Duke, VeRi, Animals, etc.).
    Any subclass that fails to implement these methods will throw a TypeError at instantiation.
    """
    
    @property
    @abstractmethod
    def unique_identities(self) -> list[IdentityId]:
        """Must return a list of all unique identities for sampler logic."""
        pass

    @property
    @abstractmethod
    def sequence_map(self) -> dict[IdentityId, dict[SequenceId, list[int]]]:
        """
        Must return the mapping of identities to their sequences, 
        and sequences to their flat frame indices for the batch sampler.
        """
        pass

    @abstractmethod
    def __getitem__(self, index: int) -> VideoReIDItem:
        """Strictly returns a domain-agnostic VideoReIDItem."""
        pass

class VideoReIDKPFBatchSampler(Sampler):
    """
    For our training process we need to have K identities with P sequences per identity. 
    Within each sequence we sample F frames to form the batch.
    """

    def __init__(
        self, 
        dataset: AbstractVideoReIDDataset,
        batches_per_epoch: int | None,
        num_identities_per_batch: int,
        num_sequences_per_identity: int,
        num_frames_per_sequence: int,
        allow_same_identity_same_sequence: bool = False,
        allow_reduced_sequences_per_identity: bool = False,
        allow_resampling_sample_indices: bool = False,
        epoch_deterministic: bool = False,
        seed: int | None = None,
        verbose: bool = False,
    ):
        self.seed = seed
        self.epoch_deterministic = epoch_deterministic
        self.rng = random.Random(self.seed) if self.seed is not None else random.Random()
        self.verbose = verbose

        self.dataset = dataset
        self.batches_per_epoch = batches_per_epoch
        self.num_identities_per_batch = num_identities_per_batch
        self.num_sequences_per_identity = num_sequences_per_identity
        self.num_frames_per_sequence = num_frames_per_sequence
        self.allow_same_identity_same_sequence = allow_same_identity_same_sequence
        self.allow_reduced_sequences_per_identity = allow_reduced_sequences_per_identity
        self.allow_resampling_sample_indices = allow_resampling_sample_indices

        # Fetch abstract properties directly
        self.unique_identities = dataset.unique_identities
        self.sequence_map = dataset.sequence_map

        # Ensure that no sequence has less than self.num_frames_per_sequence frames. This is absolutely unsupported
        for identity_id in self.unique_identities:
            for sequence_id in self.sequence_map[identity_id]:
                if len(self.sequence_map[identity_id][sequence_id]) < self.num_frames_per_sequence:
                    if not self.allow_resampling_sample_indices:
                        raise ValueError(f"Identity {identity_id} has fewer than {self.num_frames_per_sequence} frames in sequence {sequence_id}. Set allow_resampling_sample_indices to True if you want to allow sampling the same frame more than once in the same sequence if the sequence has fewer frames than num_frames_per_sequence.")
                    else:
                        print(f"WARNING: Identity {identity_id} has fewer than {self.num_frames_per_sequence} ({len(self.sequence_map[identity_id][sequence_id])}) frames in sequence {sequence_id}.")

        assert not(self.allow_reduced_sequences_per_identity and self.allow_same_identity_same_sequence), "Setting both allow_reduced_sequences_per_identity and allow_same_identity_same_sequence is not allowed."
        
        if not self.allow_reduced_sequences_per_identity and not self.allow_same_identity_same_sequence:
            # Check to make sure all identities have at least num_sequences_per_identity sequences.
            for identity_id in self.unique_identities:
                if len(self.sequence_map[identity_id]) < self.num_sequences_per_identity:
                    raise ValueError(f"Identity {identity_id} has fewer than {self.num_sequences_per_identity} sequences. Set allow_reduced_sequences_per_identity to True if you want to allow sampling fewer sequences than num_sequences_per_identity for identities who do not have enough sequences.")
        elif self.allow_reduced_sequences_per_identity:
            print(f"WARNING: allow_reduced_sequences_per_identity is set to True, so identities with fewer than {self.num_sequences_per_identity} sequences will be sampled with fewer sequences. This will create ragged batch sizes.")
        elif self.allow_same_identity_same_sequence:
            print(f"WARNING: allow_same_identity_same_sequence is set to True, so the same sequence may be sampled multiple times for the same identity.")

        assert self.num_identities_per_batch <= len(self.unique_identities), f"Number of identities per batch ({self.num_identities_per_batch}) must be less than or equal to the number of identities in the dataset ({len(self.unique_identities)})."

        self.frames_per_batch = self.num_identities_per_batch * self.num_sequences_per_identity * self.num_frames_per_sequence
        if self.verbose:
            print(f"Created a new {__class__.__name__} with {self.frames_per_batch} frames per batch and epoch length of {self.batches_per_epoch}")
            print(f"This will mean processing {self.frames_per_batch * self.batches_per_epoch} frames per epoch.")

    def sample_sequences(self, available_sequence_ids: list[SequenceId]) -> list[SequenceId]:
        if self.allow_same_identity_same_sequence:
            # Grab all sequences as many times as possible and then sample without replacement for the rest
            k = self.num_sequences_per_identity
            n_avail = len(available_sequence_ids)

            sampled_sequences = available_sequence_ids * (k // n_avail)
            sampled_sequences += self.rng.sample(available_sequence_ids, k % n_avail)
            self.rng.shuffle(sampled_sequences)
            
            return sampled_sequences

        # Otherwise ensure we sample without replacement, accounting for reduced sequences if enabled
        if not self.allow_reduced_sequences_per_identity:
            assert len(available_sequence_ids) >= self.num_sequences_per_identity, f"Not enough sequences available to sample {self.num_sequences_per_identity} sequences."
            true_num_sequences_to_sample = self.num_sequences_per_identity
        else:
            true_num_sequences_to_sample = min(self.num_sequences_per_identity, len(available_sequence_ids))

        return self.rng.sample(available_sequence_ids, k=true_num_sequences_to_sample)

    def sample_batch(self) -> list[int]:
        """
        Returns a list of sample indices from the dataset to be loaded in a single batch.
        The list is of size B = num_identities_per_batch * num_sequences_per_identity * num_frames_per_sequence
        """
        batch_sample_indices: list[int] = []

        # We first sample the identities that will make up the batch
        batch_identity_ids: list[IdentityId] = self.rng.sample(self.unique_identities, self.num_identities_per_batch)

        # For each of these identities, we sample the actual frames
        for identity_id in batch_identity_ids:
            identity_sequence_dict: dict[SequenceId, list[int]] = self.sequence_map[identity_id]
            identity_sequence_ids: list[SequenceId] = list(identity_sequence_dict.keys())

            # Sample sequences for the current identity
            sampled_sequences: list[SequenceId] = self.sample_sequences(identity_sequence_ids)
            
            # Maintain a set of used frames to ensure frames are disjoint between different sequence iterations
            used_sequence_sample_indices: dict[SequenceId, set[int]] = {seq_id: set() for seq_id in set(sampled_sequences)}
            
            for internal_seq_id, sequence_id in enumerate(sampled_sequences):
                sequence_sample_indices: list[int] = identity_sequence_dict[sequence_id]  

                already_used_frame_indices: set[int] = used_sequence_sample_indices[sequence_id]
                
                # Check if we have enough frames to sample
                if len(already_used_frame_indices) + self.num_frames_per_sequence > len(sequence_sample_indices):
                    print(f"Warning: Not enough frames available to sample {self.num_frames_per_sequence} frames from sequence {sequence_id} without replacement. Reusing frames. Sequence has {len(sequence_sample_indices)} total frames")
                    already_used_frame_indices.clear()
                    
                available_frame_indices = [frame_idx for frame_idx in sequence_sample_indices if frame_idx not in already_used_frame_indices]

                # Sample the frames for this sequence without replacement
                if len(available_frame_indices) < self.num_frames_per_sequence:
                    assert self.allow_resampling_sample_indices, f"Not enough frames available to sample {self.num_frames_per_sequence} frames from sequence {sequence_id} for identity {identity_id} without replacement. Only {len(available_frame_indices)} frames are available."
                    selected_sample_indices = self.rng.choices(available_frame_indices, k=self.num_frames_per_sequence)
                else:
                    selected_sample_indices = self.rng.sample(available_frame_indices, self.num_frames_per_sequence)
                    
                # Update the set of used frames for this sequence and append to batch
                used_sequence_sample_indices[sequence_id].update(selected_sample_indices)
                batch_sample_indices.extend(selected_sample_indices)

        return batch_sample_indices

    def __len__(self) -> int:
        if self.batches_per_epoch is None:
            raise TypeError("This BatchSampler is set to infinite mode and has no length.")
        else:
            return self.batches_per_epoch

    def __iter__(self):
        if self.epoch_deterministic:
            assert self.seed is not None, "seed must be set for deterministic mode"
            print(f"Resetting RNG for epoch with seed {self.seed}")
            self.rng = random.Random(self.seed)

        batch_count = 0
        while self.batches_per_epoch is None or batch_count < self.batches_per_epoch:
            yield self.sample_batch()
            batch_count += 1

class VideoReIDKPFBatchIterableDataset(IterableDataset[list[VideoReIDItem]]):
    """
    An IterableDataset wrapper that creates KPF batches directly.
    Because it yields fully constructed lists of VideoReIDItems (unlike a Sampler 
    which only yields indices), it can intercept and modify the sequence_id on 
    the fly to create 'virtual sequences' when the same sequence is oversampled.
    """

    def __init__(
        self, 
        dataset: AbstractVideoReIDDataset,
        batches_per_epoch: int | None,
        num_identities_per_batch: int,
        num_sequences_per_identity: int,
        num_frames_per_sequence: int,
        allow_same_identity_same_sequence: bool = False,
        allow_reduced_sequences_per_identity: bool = False,
        allow_resampling_sample_indices: bool = False,
        epoch_deterministic: bool = False,
        seed: int | None = None,
        verbose: bool = False,
        virtual_sequence_offset: int = 10_000_000,
    ):
        self.dataset = dataset
        self.batches_per_epoch = batches_per_epoch
        self.num_identities_per_batch = num_identities_per_batch
        self.num_sequences_per_identity = num_sequences_per_identity
        self.num_frames_per_sequence = num_frames_per_sequence
        self.allow_same_identity_same_sequence = allow_same_identity_same_sequence
        self.allow_reduced_sequences_per_identity = allow_reduced_sequences_per_identity
        self.allow_resampling_sample_indices = allow_resampling_sample_indices
        
        self.epoch_deterministic = epoch_deterministic
        self.seed = seed
        self.verbose = verbose
        
        # The value added to the sequence_id for every subsequent time 
        # a sequence is repeated within a batch for the same identity.
        self.virtual_sequence_offset = virtual_sequence_offset

        self.unique_identities = dataset.unique_identities
        self.sequence_map = dataset.sequence_map

        # Initial validations
        for identity_id in self.unique_identities:
            for sequence_id in self.sequence_map[identity_id]:
                if len(self.sequence_map[identity_id][sequence_id]) < self.num_frames_per_sequence:
                    if not self.allow_resampling_sample_indices:
                        raise ValueError(f"Identity {identity_id} has fewer than {self.num_frames_per_sequence} frames in sequence {sequence_id}. Set allow_resampling_sample_indices to True if you want to allow sampling the same frame more than once.")
                    elif self.verbose:
                        print(f"WARNING: Identity {identity_id} has fewer than {self.num_frames_per_sequence} ({len(self.sequence_map[identity_id][sequence_id])}) frames in sequence {sequence_id}.")

        assert not(self.allow_reduced_sequences_per_identity and self.allow_same_identity_same_sequence), "Setting both allow_reduced_sequences_per_identity and allow_same_identity_same_sequence is not allowed."
        
        if not self.allow_reduced_sequences_per_identity and not self.allow_same_identity_same_sequence:
            for identity_id in self.unique_identities:
                if len(self.sequence_map[identity_id]) < self.num_sequences_per_identity:
                    raise ValueError(f"Identity {identity_id} has fewer than {self.num_sequences_per_identity} sequences. Set allow_reduced_sequences_per_identity to True if you want to allow sampling fewer sequences.")
        elif self.allow_reduced_sequences_per_identity and self.verbose:
            print(f"WARNING: allow_reduced_sequences_per_identity is set to True, so identities with fewer than {self.num_sequences_per_identity} sequences will be sampled with fewer sequences. This will create ragged batch sizes.")
        elif self.allow_same_identity_same_sequence and self.verbose:
            print(f"WARNING: allow_same_identity_same_sequence is set to True, so the same sequence may be sampled multiple times for the same identity.")

        assert self.num_identities_per_batch <= len(self.unique_identities), f"Number of identities per batch ({self.num_identities_per_batch}) must be less than or equal to the number of identities in the dataset ({len(self.unique_identities)})."

        self.frames_per_batch = self.num_identities_per_batch * self.num_sequences_per_identity * self.num_frames_per_sequence
        if self.verbose:
            print(f"Created a new {__class__.__name__} with max {self.frames_per_batch} frames per batch.")

    def _sample_sequences(self, rng: random.Random, available_sequence_ids: list[SequenceId]) -> list[SequenceId]:
        if self.allow_same_identity_same_sequence:
            k = self.num_sequences_per_identity
            n_avail = len(available_sequence_ids)

            sampled_sequences = available_sequence_ids * (k // n_avail)
            sampled_sequences += rng.sample(available_sequence_ids, k % n_avail)
            rng.shuffle(sampled_sequences)
            return sampled_sequences

        if not self.allow_reduced_sequences_per_identity:
            assert len(available_sequence_ids) >= self.num_sequences_per_identity
            true_num_sequences_to_sample = self.num_sequences_per_identity
        else:
            true_num_sequences_to_sample = min(self.num_sequences_per_identity, len(available_sequence_ids))

        return rng.sample(available_sequence_ids, k=true_num_sequences_to_sample)

    def __iter__(self) -> Iterator[list[VideoReIDItem]]:
        worker_info = get_worker_info()
        
        # 1. Safely handle RNG seeding across multiple PyTorch DataLoader workers
        seed = self.seed
        if seed is not None:
            if worker_info is not None:
                seed += worker_info.id
            if self.epoch_deterministic and self.verbose:
                print(f"Resetting RNG for worker {worker_info.id if worker_info else 0} with seed {seed}")
                
        rng = random.Random(seed) if seed is not None else random.Random()

        # 2. Divide workload across workers if batches_per_epoch is defined
        if self.batches_per_epoch is not None:
            if worker_info is not None:
                per_worker = self.batches_per_epoch // worker_info.num_workers
                worker_id = worker_info.id
                if worker_id < self.batches_per_epoch % worker_info.num_workers:
                    per_worker += 1
                batches_to_yield = per_worker
            else:
                batches_to_yield = self.batches_per_epoch
        else:
            batches_to_yield = float('inf')

        batch_count = 0
        while batch_count < batches_to_yield:
            batch_items: list[VideoReIDItem] = []
            batch_identity_ids = rng.sample(self.unique_identities, self.num_identities_per_batch)

            for identity_id in batch_identity_ids:
                identity_sequence_dict = self.sequence_map[identity_id]
                identity_sequence_ids = list(identity_sequence_dict.keys())

                sampled_sequences = self._sample_sequences(rng, identity_sequence_ids)
                
                used_sequence_sample_indices: dict[SequenceId, set[int]] = {seq_id: set() for seq_id in set(sampled_sequences)}
                
                # Keep track of occurrences to generate distinct virtual sequence IDs
                sequence_occurrences: dict[SequenceId, int] = {}
                
                for sequence_id in sampled_sequences:
                    occurrence_idx = sequence_occurrences.get(sequence_id, 0)
                    sequence_occurrences[sequence_id] = occurrence_idx + 1
                    
                    # Generate the modified virtual sequence ID
                    virtual_sequence_id = sequence_id + (occurrence_idx * self.virtual_sequence_offset)
                    
                    sequence_sample_indices = identity_sequence_dict[sequence_id]  
                    already_used_frame_indices = used_sequence_sample_indices[sequence_id]
                    
                    if len(already_used_frame_indices) + self.num_frames_per_sequence > len(sequence_sample_indices):
                        if self.verbose:
                            print(f"Warning: Not enough frames available to sample {self.num_frames_per_sequence} frames from sequence {sequence_id} in {self.dataset} without replacement. Reusing frames. Only {len(sequence_sample_indices)} frames are available")
                        already_used_frame_indices.clear()
                        
                    available_frame_indices = [idx for idx in sequence_sample_indices if idx not in already_used_frame_indices]

                    if len(available_frame_indices) < self.num_frames_per_sequence:
                        assert self.allow_resampling_sample_indices
                        selected_sample_indices = rng.choices(available_frame_indices, k=self.num_frames_per_sequence)
                    else:
                        selected_sample_indices = rng.sample(available_frame_indices, self.num_frames_per_sequence)
                        
                    used_sequence_sample_indices[sequence_id].update(selected_sample_indices)
                    
                    # Fetch items from the underlying dataset and override sequence_id
                    for sample_idx in selected_sample_indices:
                        item = self.dataset[sample_idx]
                        if occurrence_idx > 0:
                            # _replace is built into Python NamedTuples for immutable updates
                            item = item._replace(sequence_id=virtual_sequence_id)
                        batch_items.append(item)

            yield batch_items
            batch_count += 1

    def __len__(self) -> int:
        if self.batches_per_epoch is None:
            raise TypeError("This IterableDataset is set to infinite mode and has no length.")
        return self.batches_per_epoch