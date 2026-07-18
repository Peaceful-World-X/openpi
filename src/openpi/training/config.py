"""See _CONFIGS for the list of available configs."""

import abc
import json
from collections.abc import Sequence
import dataclasses
import difflib
import logging
import pathlib
import os
import sys
from typing import Any, Literal, Protocol, TypeAlias, Union, List

import etils.epath as epath
import flax.nnx as nnx
import numpy as np
from typing_extensions import override
import tyro

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.models.pi0_fast as pi0_fast
import openpi.models.tokenizer as _tokenizer
import openpi.policies.aloha_policy as aloha_policy
import openpi.policies.droid_policy as droid_policy
import openpi.policies.libero_policy as libero_policy
import openpi.shared.download as _download
import openpi.shared.normalize as _normalize
import openpi.training.droid_rlds_dataset as droid_rlds_dataset
import openpi.training.misc.roboarena_config as roboarena_config
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms

ModelType: TypeAlias = _model.ModelType
# Work around a tyro issue with using nnx.filterlib.Filter directly.
Filter: TypeAlias = nnx.filterlib.Filter

import openpi.policies.ebench_teleop_policy as ebench_teleop_policy
import openpi.policies.robocasa_policy as robocasa_policy
import openpi.policies.ebench_policy as ebench_policy
import openpi.policies.ebench_fixedbase_policy as ebench_fixedbase_policy


@dataclasses.dataclass(frozen=True)
class _LazyDatasetSoup:
    name: str


def _dataset_soup(name: str) -> _LazyDatasetSoup:
    return _LazyDatasetSoup(name)


def _ensure_workspace_robocasa_on_path() -> None:
    workspace_root = pathlib.Path(__file__).resolve().parents[4]
    robocasa_src = workspace_root / "Robocasa" / "robocasa"
    if robocasa_src.is_dir() and str(robocasa_src) not in sys.path:
        sys.path.insert(0, str(robocasa_src))


def _load_robocasa_dataset_soup_registry() -> dict[str, Any]:
    try:
        from robocasa.utils.dataset_registry import DATASET_SOUP_REGISTRY  # type: ignore
    except ModuleNotFoundError:
        _ensure_workspace_robocasa_on_path()
        try:
            from robocasa.utils.dataset_registry import DATASET_SOUP_REGISTRY  # type: ignore
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "robocasa is required for official RoboCasa data-soup configs such as "
                "pi05_pretrain_human300. Install it with `pip install -e "
                "/mnt/pfs/users/wenyao.xue/code/Robocasa/robocasa` or make it importable."
            ) from exc
    return DATASET_SOUP_REGISTRY


def _local_robocasa_pretrain_human300_soup() -> list[dict[str, Any]]:
    root = pathlib.Path("/shared_disk/users/hengtao.li/robocasa_datasets/v1.0/pretrain")
    data_dirs = [
        {"path": str(repo), "filter_key": "100_demos"}
        for pattern in ("atomic/*/*/lerobot", "composite/*/*/lerobot")
        for repo in root.glob(pattern)
        if (repo / "meta" / "info.json").is_file() and (repo / "meta" / "modality.json").is_file()
    ]
    if not data_dirs:
        raise FileNotFoundError(f"No RoboCasa LeRobot repos found under {root}")
    return sorted(data_dirs, key=lambda item: item["path"])


def _resolve_dataset_soup(data_dirs: Any | None) -> Any | None:
    if not isinstance(data_dirs, _LazyDatasetSoup):
        return data_dirs

    if data_dirs.name == "pretrain_human300":
        return _local_robocasa_pretrain_human300_soup()

    registry = _load_robocasa_dataset_soup_registry()
    if data_dirs.name not in registry:
        raise KeyError(f"RoboCasa DATASET_SOUP_REGISTRY[{data_dirs.name!r}] is unavailable.")
    return registry[data_dirs.name]


def _robocasa_data_dirs_to_repo_ids(data_dirs: Any) -> list[str]:
    if isinstance(data_dirs, (str, pathlib.Path)):
        items = [data_dirs]
    else:
        items = list(data_dirs)

    repo_ids: list[str] = []
    missing: list[str] = []
    for item in items:
        path = item.get("path") if isinstance(item, dict) else item
        if path is None:
            continue
        repo = pathlib.Path(path)
        if (repo / "meta" / "info.json").is_file():
            repo_ids.append(str(repo))
        else:
            missing.append(str(repo))

    if missing:
        sample = ", ".join(missing[:5])
        raise FileNotFoundError(f"Missing RoboCasa LeRobot repo metadata for {len(missing)} path(s): {sample}")
    if not repo_ids:
        raise ValueError("Could not resolve any LeRobot repo paths from RoboCasa data_dirs")
    return repo_ids


def _robocasa_stats_entry_to_norm_stats(entry: dict[str, Any]) -> _normalize.NormStats:
    return _normalize.NormStats(
        mean=np.asarray(entry["mean"]),
        std=np.asarray(entry["std"]),
        q01=np.asarray(entry["q01"]) if entry.get("q01") is not None else None,
        q99=np.asarray(entry["q99"]) if entry.get("q99") is not None else None,
    )


def _load_robocasa_repo_norm_stats(repo_ids: Sequence[str]) -> dict[str, _normalize.NormStats] | None:
    for repo_id in repo_ids:
        stats_path = pathlib.Path(repo_id) / "meta" / "stats.json"
        if not stats_path.is_file():
            continue
        stats = json.loads(stats_path.read_text())
        if "observation.state" not in stats or "action" not in stats:
            continue
        logging.info(f"Loaded RoboCasa fallback norm stats from {stats_path}")
        return {
            "state": _robocasa_stats_entry_to_norm_stats(stats["observation.state"]),
            "actions": _robocasa_stats_entry_to_norm_stats(stats["action"]),
        }
    return None


@dataclasses.dataclass(frozen=True)
class AssetsConfig:
    """Determines the location of assets (e.g., norm stats) that will be used to set up the data pipeline.

    These assets will be replicated inside the checkpoint under the `assets/asset_id` directory.

    This can be used to load assets from a different checkpoint (e.g., base model checkpoint) or some other
    centralized location. For example, to load the norm stats for the Trossen robot from the base model checkpoint
    during fine-tuning, use:

    ```
    AssetsConfig(
        assets_dir="/shared_disk/models/projects/openpi/openpi-assets/checkpoints/pi05_base/assets",
        asset_id="trossen",
    )
    ```
    """

    # Assets directory. If not provided, the config assets_dirs will be used. This is useful to load assets from
    # a different checkpoint (e.g., base model checkpoint) or some other centralized location.
    assets_dir: str | None = None

    # Asset id. If not provided, the repo id will be used. This allows users to reference assets that describe
    # different robot platforms.
    asset_id: str | None = None


@dataclasses.dataclass(frozen=True)
class DataConfig:
    # LeRobot repo id. If None, fake data will be created.
    repo_id: Union[str, List[str]] | None = None
    # Directory within the assets directory containing the data assets.
    asset_id: str | None = None
    # Contains precomputed normalization stats. If None, normalization will not be performed.
    norm_stats: dict[str, _transforms.NormStats] | None = None

    # Used to adopt the inputs from a dataset specific format to a common format
    # which is expected by the data transforms.
    repack_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Data transforms, typically include robot specific transformations. Will be applied
    # before the data is normalized. See `model.Observation` and `model.Actions` to learn about the
    # normalized data.
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Model specific transforms. Will be applied after the data is normalized.
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantile_norm: bool = False

    # Names of keys that will be used by the data loader to generate the action sequence. The length of the
    # sequence is defined by the `action_horizon` field in the model config. This should be adjusted if your
    # LeRobot dataset is using different keys to represent the action.
    action_sequence_keys: Sequence[str] = ("actions",)

    # If true, will use the LeRobot dataset task to define the prompt.
    prompt_from_task: bool = False

    # Only used for RLDS data loader (ie currently only used for DROID).
    rlds_data_dir: str | None = None
    # Action space for DROID dataset.
    action_space: droid_rlds_dataset.DroidActionSpace | None = None
    # Path to the data filter file for DROID dataset
    filter_dict_path: str | None = None

    # If true, will disable syncing the dataset from the Hugging Face Hub. Allows training on local-only datasets.
    local_files_only: bool = False

    # Official RoboCasa/OpenPI-style dataset mixture metadata. The local LeRobot
    # data loader still consumes repo_id; this is preserved on DataConfig so
    # RoboCasa configs can mirror the official data_dirs interface.
    data_dirs: Any | None = None
    dataset_weights: list[float] | None = None


class GroupFactory(Protocol):
    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        """Create a group."""


@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(GroupFactory):
    """Creates model transforms for standard pi0 models."""

    # If provided, will determine the default prompt that be used by the model.
    default_prompt: str | None = None

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        match model_config.model_type:
            case _model.ModelType.PI0:
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI05:
                assert isinstance(model_config, pi0_config.Pi0Config)
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                            discrete_state_input=model_config.discrete_state_input,
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI0_FAST:
                tokenizer_cls = (
                    _tokenizer.FASTTokenizer
                    if model_config.fast_model_tokenizer is None
                    else model_config.fast_model_tokenizer
                )
                tokenizer_kwargs = (
                    {} if model_config.fast_model_tokenizer_kwargs is None else model_config.fast_model_tokenizer_kwargs
                )
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizeFASTInputs(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                        ),
                    ],
                    outputs=[
                        _transforms.ExtractFASTActions(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                            action_horizon=model_config.action_horizon,
                            action_dim=model_config.action_dim,
                        )
                    ],
                )


@dataclasses.dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    # The LeRobot repo id.
    repo_id: Union[str, List[str]] = tyro.MISSING
    # Determines how the assets will be loaded.
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    # Base config that will be updated by the factory.
    base_config: tyro.conf.Suppress[DataConfig | None] = None

    @abc.abstractmethod
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """Create a data config."""

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        asset_id = self.assets.asset_id or repo_id
        return dataclasses.replace(
            self.base_config or DataConfig(),
            repo_id=repo_id,
            asset_id=asset_id,
            norm_stats=self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id),
            use_quantile_norm=model_config.model_type != ModelType.PI0,
        )

    def _load_norm_stats(self, assets_dir: epath.Path, asset_id: str | None) -> dict[str, _transforms.NormStats] | None:
        if asset_id is None:
            return None
        try:
            data_assets_dir = str(assets_dir / asset_id)
            norm_stats = _normalize.load(_download.maybe_download(data_assets_dir))
            logging.info(f"Loaded norm stats from {data_assets_dir}")
            return norm_stats
        except FileNotFoundError:
            logging.info(f"Norm stats not found in {data_assets_dir}, skipping.")
        return None


@dataclasses.dataclass(frozen=True)
class FakeDataConfig(DataConfigFactory):
    repo_id: str = "fake"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return DataConfig(repo_id=self.repo_id)


@dataclasses.dataclass(frozen=True)
class SimpleDataConfig(DataConfigFactory):
    # Factory for the data transforms.
    data_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=GroupFactory)
    # Factory for the model transforms.
    model_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=ModelTransformFactory)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            data_transforms=self.data_transforms(model_config),
            model_transforms=self.model_transforms(model_config),
        )


@dataclasses.dataclass(frozen=True)
class LeRobotEBenchFixedBaseDataConfig(DataConfigFactory):
    """
    This config is used to configure transforms that are applied at various parts of the data pipeline.
    For your own dataset, you can copy this class and modify the transforms to match your dataset based on the
    comments below.
    """

    extra_delta_transform: bool = False
    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # The repack transform is *only* applied to the data coming from the dataset,
        # and *not* during inference. We can use it to make inputs from the dataset look
        # as close as possible to those coming from the inference environment (e.g. match the keys).
        # Below, we match the keys in the dataset (which we defined in the data conversion script) to
        # the keys we use in our inference pipeline (defined in the inference script for libero).
        # For your own dataset, first figure out what keys your environment passes to the policy server
        # and then modify the mappings below so your dataset's keys get matched to those target keys.
        # The repack transform simply remaps key names here.
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images/head": "video.overlook_camera_view",
                        "images/hand_left": "video.left_camera_view",
                        "images/hand_right": "video.right_camera_view",
                        "states/joint": "state.joints",
                        "states/gripper": "state.gripper",
                        "actions/joint": "action.joints",
                        "actions/gripper": "action.gripper",
                        "prompt": "prompt",
                        
                    }
                )
            ]
        )

        # The data transforms are applied to the data coming from the dataset *and* during inference.
        # Below, we define the transforms for data going into the model (``inputs``) and the transforms
        # for data coming out of the model (``outputs``) (the latter is only used during inference).
        # We defined these transforms in `libero_policy.py`. You can check the detailed comments there for
        # how to modify the transforms to match your dataset. Once you created your own transforms, you can
        # replace the transforms below with your own.
        data_transforms = _transforms.Group(
            inputs=[ebench_fixedbase_policy.EBenchInputs(model_type=model_config.model_type)],
            outputs=[ebench_fixedbase_policy.EBenchOutputs()],
        )

        # One additional data transform: pi0 models are trained on delta actions (relative to the first
        # state in each action chunk). IF your data has ``absolute`` actions (e.g. target joint angles)
        # you can uncomment the following line to convert the actions to delta actions. The only exception
        # is for the gripper actions which are always absolute.
        # In the example below, we would apply the delta conversion to the first 6 actions (joints) and
        # leave the 7th action (gripper) unchanged, i.e. absolute.
        # In Libero, the raw actions in the dataset are already delta actions, so we *do not* need to
        # apply a separate delta conversion (that's why it's commented out). Choose whether to apply this
        # transform based on whether your dataset uses ``absolute`` or ``delta`` actions out of the box.
        if self.extra_delta_transform:
            delta_action_mask = _transforms.make_bool_mask(12, -4) 
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        # Model transforms include things like tokenizing the prompt and action targets
        # You do not need to change anything here for your own dataset.
        model_transforms = ModelTransformFactory()(model_config)

        # We return all data transforms for training and inference. No need to change anything here.
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )

@dataclasses.dataclass(frozen=True)
class LeRobotEBenchDataConfig(DataConfigFactory):
    """
    This config is used to configure transforms that are applied at various parts of the data pipeline.
    For your own dataset, you can copy this class and modify the transforms to match your dataset based on the
    comments below.
    """

    extra_delta_transform: bool = False
    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # The repack transform is *only* applied to the data coming from the dataset,
        # and *not* during inference. We can use it to make inputs from the dataset look
        # as close as possible to those coming from the inference environment (e.g. match the keys).
        # Below, we match the keys in the dataset (which we defined in the data conversion script) to
        # the keys we use in our inference pipeline (defined in the inference script for libero).
        # For your own dataset, first figure out what keys your environment passes to the policy server
        # and then modify the mappings below so your dataset's keys get matched to those target keys.
        # The repack transform simply remaps key names here.
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images/head": "video.overlook_camera_view",
                        "images/hand_left": "video.left_camera_view",
                        "images/hand_right": "video.right_camera_view",
                        "states/joint": "state.joints",
                        "states/gripper": "state.gripper",
                        "actions/joint": "action.joints",
                        "actions/gripper": "action.gripper",
                        "actions/base": "action.base",
                        "prompt": "prompt",
                        
                    }
                )
            ]
        )


        # The data transforms are applied to the data coming from the dataset *and* during inference.
        # Below, we define the transforms for data going into the model (``inputs``) and the transforms
        # for data coming out of the model (``outputs``) (the latter is only used during inference).
        # We defined these transforms in `libero_policy.py`. You can check the detailed comments there for
        # how to modify the transforms to match your dataset. Once you created your own transforms, you can
        # replace the transforms below with your own.
        data_transforms = _transforms.Group(
            inputs=[ebench_policy.EBenchInputs(model_type=model_config.model_type)],
            outputs=[ebench_policy.EBenchOutputs()],
        )

        # One additional data transform: pi0 models are trained on delta actions (relative to the first
        # state in each action chunk). IF your data has ``absolute`` actions (e.g. target joint angles)
        # you can uncomment the following line to convert the actions to delta actions. The only exception
        # is for the gripper actions which are always absolute.
        # In the example below, we would apply the delta conversion to the first 6 actions (joints) and
        # leave the 7th action (gripper) unchanged, i.e. absolute.
        # In Libero, the raw actions in the dataset are already delta actions, so we *do not* need to
        # apply a separate delta conversion (that's why it's commented out). Choose whether to apply this
        # transform based on whether your dataset uses ``absolute`` or ``delta`` actions out of the box.

       
        if self.extra_delta_transform:
            delta_action_mask = _transforms.make_bool_mask(12, -4) 
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        # Model transforms include things like tokenizing the prompt and action targets
        # You do not need to change anything here for your own dataset.
        model_transforms = ModelTransformFactory()(model_config)

        # We return all data transforms for training and inference. No need to change anything here.
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotAlohaDataConfig(DataConfigFactory):
    # If true, will convert joint dimensions to deltas with respect to the current state before passing to the model.
    # Gripper dimensions will remain in absolute values.
    use_delta_joint_actions: bool = True
    # If provided, will be injected into the input data if the "prompt" key is not present.
    default_prompt: str | None = None
    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model. People who
    # use standard Aloha data should set this to true.
    adapt_to_pi: bool = True
    mask_list: list[int] | None = None  # [7, -1, 7, -1, 1]
    zero_mask_list: list[int] | None = None  # e.g. [-16, 3, -13] to zero dims [16, 17, 18]
    action_horizon: int = 17
    # Repack transforms.
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {"cam_high": "observation.images.top"},
                        "state": "observation.state",
                        "actions": "action",
                    }
                )
            ]
        )
    )
    # Action keys that will be used to read the action sequence from the dataset.
    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        data_transforms = _transforms.Group(
            inputs=[aloha_policy.AlohaInputs(action_dim=model_config.action_dim, adapt_to_pi=self.adapt_to_pi, zero_mask_list=self.zero_mask_list)],
            outputs=[aloha_policy.AlohaOutputs(adapt_to_pi=self.adapt_to_pi, action_horizon=self.action_horizon)],
        )
        if self.use_delta_joint_actions:
            delta_action_mask = _transforms.make_bool_mask(*self.mask_list)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotLiberoDataConfig(DataConfigFactory):
    """
    This config is used to configure transforms that are applied at various parts of the data pipeline.
    For your own dataset, you can copy this class and modify the transforms to match your dataset based on the
    comments below.
    """

    extra_delta_transform: bool = False

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # The repack transform is *only* applied to the data coming from the dataset,
        # and *not* during inference. We can use it to make inputs from the dataset look
        # as close as possible to those coming from the inference environment (e.g. match the keys).
        # Below, we match the keys in the dataset (which we defined in the data conversion script) to
        # the keys we use in our inference pipeline (defined in the inference script for libero).
        # For your own dataset, first figure out what keys your environment passes to the policy server
        # and then modify the mappings below so your dataset's keys get matched to those target keys.
        # The repack transform simply remaps key names here.
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/wrist_image": "wrist_image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        # The data transforms are applied to the data coming from the dataset *and* during inference.
        # Below, we define the transforms for data going into the model (``inputs``) and the transforms
        # for data coming out of the model (``outputs``) (the latter is only used during inference).
        # We defined these transforms in `libero_policy.py`. You can check the detailed comments there for
        # how to modify the transforms to match your dataset. Once you created your own transforms, you can
        # replace the transforms below with your own.
        data_transforms = _transforms.Group(
            inputs=[libero_policy.LiberoInputs(model_type=model_config.model_type)],
            outputs=[libero_policy.LiberoOutputs()],
        )

        # One additional data transform: pi0 models are trained on delta actions (relative to the first
        # state in each action chunk). IF your data has ``absolute`` actions (e.g. target joint angles)
        # you can uncomment the following line to convert the actions to delta actions. The only exception
        # is for the gripper actions which are always absolute.
        # In the example below, we would apply the delta conversion to the first 6 actions (joints) and
        # leave the 7th action (gripper) unchanged, i.e. absolute.
        # In Libero, the raw actions in the dataset are already delta actions, so we *do not* need to
        # apply a separate delta conversion (that's why it's commented out). Choose whether to apply this
        # transform based on whether your dataset uses ``absolute`` or ``delta`` actions out of the box.

        # LIBERO already represents actions as deltas, but we have some old Pi0 checkpoints that are trained with this
        # extra delta transform.
        if self.extra_delta_transform:
            delta_action_mask = _transforms.make_bool_mask(6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        # Model transforms include things like tokenizing the prompt and action targets
        # You do not need to change anything here for your own dataset.
        model_transforms = ModelTransformFactory()(model_config)

        # We return all data transforms for training and inference. No need to change anything here.
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class RLDSDroidDataConfig(DataConfigFactory):
    """
    Config for training on DROID, using RLDS data format (for efficient training on larger datasets).
    """

    rlds_data_dir: str | None = None
    action_space: droid_rlds_dataset.DroidActionSpace | None = None

    # Filtering options. Can pass a path to a dictionary that maps episodes to timestep ranges
    # to tuples denoting ranges of time steps to keep (start, end). Episodes are uniquely identified with
    # f"{recording_folderpath}--{file_path}", both of which are present in the RLDS episode metadata.
    # Path to the filter dictionary file.
    filter_dict_path: str | None = "gs://openpi-assets/droid/droid_sample_ranges_v1_0_1.json"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "observation/image",
                        "observation/wrist_image_left": "observation/wrist_image",
                        "observation/joint_position": "observation/joint_position",
                        "observation/gripper_position": "observation/gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )

        if self.action_space == droid_rlds_dataset.DroidActionSpace.JOINT_POSITION:
            # Data loader returns absolute joint position actions -- convert to delta actions for training.
            delta_action_mask = _transforms.make_bool_mask(7, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory()(model_config)

        assert self.rlds_data_dir is not None, "Need to set rlds data dir for RLDS data loader."

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            rlds_data_dir=self.rlds_data_dir,
            action_space=self.action_space,
            filter_dict_path=self.filter_dict_path,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotDROIDDataConfig(DataConfigFactory):
    """
    Example data config for custom DROID dataset in LeRobot format.
    To convert your custom DROID dataset (<10s of hours) to LeRobot format, see examples/droid/convert_droid_data_to_lerobot.py
    """

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "exterior_image_1_left",
                        "observation/exterior_image_2_left": "exterior_image_2_left",
                        "observation/wrist_image_left": "wrist_image_left",
                        "observation/joint_position": "joint_position",
                        "observation/gripper_position": "gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        # We assume joint *velocity* actions, so we should *not* apply an additional delta transform.
        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )
        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    # Name of the config. Must be unique. Will be used to reference this config.
    name: tyro.conf.Suppress[str]
    # Project name.
    project_name: str = "openpi"
    # Experiment name. Will be used to name the metadata and checkpoint directories.
    exp_name: str = tyro.MISSING

    # Defines the model config. Some attributes (action_dim, action_horizon, and max_token_len) are shared by all models
    # -- see BaseModelConfig. Specific model implementations (e.g., Pi0Config) inherit from BaseModelConfig and may
    # define additional attributes.
    model: _model.BaseModelConfig = dataclasses.field(default_factory=pi0_config.Pi0Config)

    # A weight loader can optionally load (possibly partial) weights from disk after the model is initialized.
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(default_factory=weight_loaders.NoOpWeightLoader)

    # Optional path to a PyTorch checkpoint to load weights from.
    pytorch_weight_path: str | None = None

    # Precision for PyTorch training.
    pytorch_training_precision: Literal["bfloat16", "float32"] = "bfloat16"

    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    ema_decay: float | None = 0.99

    # Specifies which weights should be frozen.
    freeze_filter: tyro.conf.Suppress[Filter] = dataclasses.field(default_factory=nnx.Nothing)

    # Determines the data to be trained on.
    data: DataConfigFactory = dataclasses.field(default_factory=FakeDataConfig)

    # Base directory for config assets (e.g., norm stats).
    assets_base_dir: str = "./assets"
    # Base directory for checkpoints.
    checkpoint_base_dir: str = "/shared_disk/users/can.jin/checkpoint"

    # Random seed that will be used by random generators during training.
    seed: int = 42
    # Global batch size.
    batch_size: int = 64
    # Number of workers to use for the data loader. Increasing this number will speed up data loading but
    # will increase memory and CPU usage.
    num_workers: int = 32
    # Number of train steps (batches) to run.
    num_train_steps: int = 30_000

    # How often (in steps) to log training metrics.
    log_interval: int = 100
    # How often (in steps) to save checkpoints.
    save_interval: int = 1000
    # If set, any existing checkpoints matching step % keep_period == 0 will not be deleted.
    keep_period: int | None = 5000

    # If true, will overwrite the checkpoint directory if it already exists.
    overwrite: bool = False
    # If true, will resume training from the last checkpoint.
    resume: bool = False

    # If true, will enable wandb logging.
    wandb_enabled: bool = True

    # Used to pass metadata to the policy server.
    policy_metadata: dict[str, Any] | None = None

    # If the value is greater than 1, FSDP will be enabled and shard across number of specified devices; overall
    # device memory will be reduced but training could potentially be slower.
    # eg. if total device is 4 and fsdp devices is 2; then the model will shard to 2 devices and run
    # data parallel between 2 groups of devices.
    fsdp_devices: int = 1

    @property
    def assets_dirs(self) -> pathlib.Path:
        """Get the assets directory for this config."""
        return (pathlib.Path(self.assets_base_dir) / self.name).resolve()

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        """Get the checkpoint directory for this config."""
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        """Get the filter for the trainable parameters."""
        return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")


def _discover_lerobot_repo_ids(
    root: str = "/shared_disk/users/wenyao.xue/data/EBench-Dataset",
    subset: str | None = None,
) -> list[str]:
    """Auto-discover local LeRobot dataset directories.

    A valid LeRobot dataset directory contains meta/info.json.
    By default this scans the full EBench root, i.e. all 26 tasks.
    Set EBENCH_DATA_SUBSET=teleop_tasks only when you intentionally want the 7 teleop tasks.
    """
    root_path = pathlib.Path(os.environ.get("EBENCH_DATA_ROOT", root)).resolve()
    subset = os.environ.get("EBENCH_DATA_SUBSET", subset)

    search_root = root_path / subset if subset else root_path
    if not search_root.exists():
        raise FileNotFoundError(f"EBench search root does not exist: {search_root}")

    repo_ids = sorted({str(p.parent.parent.resolve()) for p in search_root.rglob("meta/info.json")})
    if not repo_ids:
        raise FileNotFoundError(f"No LeRobot datasets found under: {search_root}")

    print(f"[INFO] Discovered {len(repo_ids)} EBench LeRobot datasets under {search_root}")
    return repo_ids


@dataclasses.dataclass(frozen=True)
class LeRobotEBenchTeleopDataConfig(DataConfigFactory):
    repo_id: list[str] | str = tyro.MISSING

    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    base_config: tyro.conf.Suppress[DataConfig | None] = None

    default_prompt: str | None = None
    use_delta_joint_actions: bool = True
    use_base: bool = True

    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        if self.repo_id == "auto":
            object.__setattr__(self, "repo_id", _discover_lerobot_repo_ids())

        serving = self.repo_id == "fake"
        repack_transform = _transforms.Group()
        if not serving:
            repack_transform = _transforms.Group(
                inputs=[
                    ebench_teleop_policy.EBenchTeleopInputs(use_base=self.use_base),
                ],
                outputs=[
                    ebench_teleop_policy.EBenchTeleopOutputs(use_base=self.use_base),
                ],
            )

        data_transforms = (
            _transforms.Group(
                inputs=[ebench_teleop_policy.EBenchTeleopInputs(use_base=self.use_base)],
                outputs=[ebench_teleop_policy.EBenchTeleopOutputs(use_base=self.use_base)],
            )
            if serving
            else _transforms.Group()
        )

        if self.use_delta_joint_actions:
            if self.use_base:
                # 19-dim action/state:
                #   0:12  arm joints        absolute target -> delta relative to state
                #   12:16 grippers          keep absolute
                #   16:19 mobile base delta keep as-is because it is already delta
                delta_action_mask = _transforms.make_bool_mask(12, -7)
            else:
                # 16-dim action/state:
                #   0:12  arm joints -> delta
                #   12:16 grippers   -> absolute
                delta_action_mask = _transforms.make_bool_mask(12, -4)

            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        action_sequence_keys = (
            ("action.joints", "action.gripper", "action.base_delta")
            if self.use_base
            else ("action.joints", "action.gripper")
        )

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=action_sequence_keys,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotRobocasaDataConfig(DataConfigFactory):
    """Official RoboCasa/Groot-style data config."""

    repo_id: str | None = None
    data_dirs: Any | None = None
    dataset_weights: list[float] | None = None

    action_dim: int | None = None

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repo_id = self.repo_id
        data_dirs = _resolve_dataset_soup(self.data_dirs)

        if repo_id == "auto":
            data_dirs = _dataset_soup("pretrain_human300")
            data_dirs = _resolve_dataset_soup(data_dirs)
            repo_id = None

        repo_ids = _robocasa_data_dirs_to_repo_ids(data_dirs) if data_dirs else repo_id
        base = self.create_base_config(assets_dirs, model_config)
        repo_id_list = [repo_ids] if isinstance(repo_ids, str) else list(repo_ids or [])

        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "observation.images.robot0_agentview_left",
                        "observation/wrist_image": "observation.images.robot0_eye_in_hand",
                        "observation/right_image": "observation.images.robot0_agentview_right",
                        "observation/state": "observation.state",
                        "actions": "action",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[robocasa_policy.RobocasaInputs(action_dim=model_config.action_dim, model_type=model_config.model_type)],
            outputs=[robocasa_policy.RobocasaOutputs()],
        )

        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            base,
            repo_id=repo_id_list or repo_ids,
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            norm_stats=base.norm_stats or _load_robocasa_repo_norm_stats(repo_id_list),
            data_dirs=data_dirs,
            dataset_weights=self.dataset_weights,
        )


# Use `get_config` if you need to get a config by name in your code.
_CONFIGS = [
    #
    # Inference Aloha configs.
    #
    TrainConfig(
        name="pi0_aloha",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi05_aloha",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi0_aloha_towel",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
            default_prompt="fold the towel",
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi0_aloha_tupperware",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
            default_prompt="open the tupperware and put the food on the plate",
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    #
    # Inference DROID configs.
    #
    TrainConfig(
        name="pi0_droid",
        model=pi0_config.Pi0Config(action_horizon=10),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI0)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    TrainConfig(
        name="pi0_fast_droid",
        model=pi0_fast.Pi0FASTConfig(action_dim=8, action_horizon=10),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI0_FAST)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    TrainConfig(
        name="pi05_droid",
        model=pi0_config.Pi0Config(action_horizon=15, pi05=True),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI05)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    #
    # Fine-tuning Libero configs.
    #
    # These train configs define the hyperparameters for fine-tuning the base model on your own dataset.
    # They are used to define key elements like the dataset you are training on, the base checkpoint you
    # are using, and other hyperparameters like how many training steps to run or what learning rate to use.
    # For your own dataset, you can copy this class and modify the dataset name, and data transforms based on
    # the comments below.
    TrainConfig(
        # Change the name to reflect your model and dataset.
        name="pi0_libero",
        # Here you define the model config -- In this example we use pi0 as the model
        # architecture and perform *full* finetuning. in the examples below we show how to modify
        # this to perform *low-memory* (LORA) finetuning and use pi0-FAST as an alternative architecture.
        model=pi0_config.Pi0Config(),
        # Here you define the dataset you are training on. In this example we use the Libero
        # dataset. For your own dataset, you can change the repo_id to point to your dataset.
        # Also modify the DataConfig to use the new config you made for your dataset above.
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                # This flag determines whether we load the prompt (i.e. the task instruction) from the
                # ``task`` field in the LeRobot dataset. If set to True, the prompt will show up in
                # a field called ``prompt`` in the input dict. The recommended setting is True.
                prompt_from_task=True,
            ),
            extra_delta_transform=True,
        ),
        # Here you define which pre-trained checkpoint you want to load to initialize the model.
        # This should match the model config you chose above -- i.e. in this case we use the pi0 base model.
        weight_loader=weight_loaders.CheckpointWeightLoader("/shared_disk/models/projects/openpi/openpi-assets/checkpoints/pi05_base/params"),
        # Below you can define other hyperparameters like the learning rate, number of training steps, etc.
        # Check the base TrainConfig class for a full list of available hyperparameters.
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_libero_low_mem_finetune",
        # Here is an example of loading a pi0 model for LoRA fine-tuning.
        model=pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("/shared_disk/models/projects/openpi/openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=30_000,
        # The freeze filter defines which parameters should be frozen during training.
        # We have a convenience function in the model config that returns the default freeze filter
        # for the given model config for LoRA finetuning. Just make sure it matches the model config
        # you chose above.
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
    ),
    TrainConfig(
        name="pi0_fast_libero",
        # Here is an example of loading a pi0-FAST model for full finetuning.
        # Modify action_dim and action_horizon to match your dataset (action horizon is equal to
        # the desired action chunk length).
        # The max_token_len is the maximum number of (non-image) tokens the model can handle.
        # This includes the tokenized prompt, proprioceptive state, and (FAST-tokenized) action tokens.
        # Choosing this value too small may chop off tokens at the end of your sequence (the code will throw
        # a warning), while choosing it too large will waste memory (since we pad each batch element to the
        # max_token_len). A good rule of thumb is to use approx 180 for single-arm robots, and approx 250 for
        # two-arm robots. Generally, err on the lower side here first, and potentially increase the value if
        # you see many warnings being thrown during training.
        model=pi0_fast.Pi0FASTConfig(action_dim=7, action_horizon=10, max_token_len=180),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        # Note that we load the pi0-FAST base model checkpoint here.
        # weight_loader=weight_loaders.CheckpointWeightLoader("/shared_disk/models/projects/openpi/openpi-assets/checkpoints/pi0_fast_base/params"),
        weight_loader=weight_loaders.CheckpointWeightLoader("/shared_disk/models/projects/openpi/openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_fast_libero_low_mem_finetune",
        # Here is an example of loading a pi0-FAST model for LoRA finetuning.
        # For setting action_dim, action_horizon, and max_token_len, see the comments above.
        model=pi0_fast.Pi0FASTConfig(
            action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        # weight_loader=weight_loaders.CheckpointWeightLoader("/shared_disk/models/projects/openpi/openpi-assets/checkpoints/pi0_fast_base/params"),
        weight_loader=weight_loaders.CheckpointWeightLoader("/shared_disk/models/projects/openpi/openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=30_000,
        # Again, make sure to match the model config above when extracting the freeze filter
        # that specifies which parameters should be frozen during LoRA finetuning.
        freeze_filter=pi0_fast.Pi0FASTConfig(
            action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
    ),
    TrainConfig(
        name="pi05_libero",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=256,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("/shared_disk/models/projects/openpi/openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path="/path/to/your/pytorch_weight_path",
        num_train_steps=30_000,
    ),
    #
    # Fine-tuning Aloha configs.
    #
    # This is a test config that is used to illustate how train on a custom LeRobot dataset.
    # For instuctions on how to convert and train on your own Aloha dataset see examples/aloha_real/README.md
    TrainConfig(
        name="pi0_aloha_pen_uncap",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            repo_id="physical-intelligence/aloha_pen_uncap_diverse",
            assets=AssetsConfig(
                assets_dir="/shared_disk/models/projects/openpi/openpi-assets/checkpoints/pi05_base/assets",
                asset_id="trossen",
            ),
            default_prompt="uncap the pen",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("/shared_disk/models/projects/openpi/openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=20_000,
    ),
    TrainConfig(
        name="pi05_aloha_pen_uncap",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            repo_id="physical-intelligence/aloha_pen_uncap_diverse",
            assets=AssetsConfig(
                assets_dir="/shared_disk/models/projects/openpi/openpi-assets/checkpoints/pi05_base/assets",
                asset_id="trossen",
            ),
            default_prompt="uncap the pen",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("/shared_disk/models/projects/openpi/openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=20_000,
        batch_size=64,
    ),
    #
    # Personal Config for compute norm.
    #
    # agilex piper config
    TrainConfig(
        # 任务名
        name="pi05_benchmark_open_water_0501",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            # 加载多数据路径
            repo_id='/shared_disk/users/can.jin/dataset/agilex/benchmark_open_water_0501/260501190118_4464',
            assets=AssetsConfig(
                # 加载norm路径
                assets_dir="/mnt/pfs/users/can.jin/public/norm_stats/assets",
                asset_id="benchmark_open_water_0501",
            ),
            base_config=DataConfig(prompt_from_task=True),
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                            "prompt": "prompt",
                        }
                    )
                ]
            ),
            # base_config=DataConfig(
            #     local_files_only=True,  # Set to True for local-only datasets.
            # ),
            use_delta_joint_actions=True,
            action_horizon=14,
            mask_list=[6, -1, 6, -1],
            zero_mask_list=[-14, 18],
            adapt_to_pi=True, # Aloha/Songling
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("/shared_disk/models/projects/openpi/openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=40_000,
        batch_size=128,
        num_workers=64,
    ),
    # agilex aloha config
    TrainConfig(
        # 任务名
        name="pi05_benchmark_open_water_0501_add_move_base",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            # 加载多数据路径
            repo_id='/shared_disk/users/can.jin/dataset/agilex/benchmark_open_water_0501/260501190118_4464',
            assets=AssetsConfig(
                # 加载norm路径
                assets_dir="/mnt/pfs/users/can.jin/public/norm_stats/assets",
                asset_id="benchmark_open_water_0501_add_move_base",
            ),
            base_config=DataConfig(prompt_from_task=True),
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                            "prompt": "prompt",
                        }
                    )
                ]
            ),
            # base_config=DataConfig(
            #     local_files_only=True,  # Set to True for local-only datasets.
            # ),
            use_delta_joint_actions=True,
            action_horizon=16,
            mask_list=[6, -1, 6, -1, 2],
            zero_mask_list=[-16, 16],
            adapt_to_pi=True, # Aloha/Songling
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("/shared_disk/models/projects/openpi/openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=40_000,
        batch_size=128,
        num_workers=64,
    ),
    # g1 arm config
    TrainConfig(
        name="pi05_lerobot_merged_204_333_428_634_n300",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            # 加载数据路径
            repo_id="/shared_disk/users/yang.wang/data-g1/flat_fold_shirt/aligned_data/lerobot_merged_204_333_428_634_n300",
            assets=AssetsConfig(
                # 加载norm路径
                assets_dir="/mnt/pfs/users/can.jin/public/norm_stats/assets/",
                asset_id="lerobot_merged_204_333_428_634_n300",
            ),
            default_prompt="fold the shirt",
            # base_config=DataConfig(prompt_from_task=True),
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                            # "prompt": "prompt",
                        }
                    )
                ]
            ),
            # base_config=DataConfig(
            #     local_files_only=True,  # Set to True for local-only datasets.
            # ),
            use_delta_joint_actions=True,
            action_horizon=16,
            mask_list=[7, -1, 7, -1],
            adapt_to_pi=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("/shared_disk/models/projects/openpi/openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=40_0000,
        # batch_size=8,
        # num_workers=2,
        batch_size=128,
        num_workers=64,
    ),
    # g1 arm + waist(only height) config
    TrainConfig(
        name="pi05_fast_3_data_carry_box_b1_fast_0316_merged",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            # 记载数据路径
            repo_id="/mnt/pfs/users/can.jin/public/datasets/mix_merge/fast_3_data_carry_box_b1_fast_0316_merged",
            assets=AssetsConfig(
                # 加载norm路径 计算时不用管
                assets_dir="/mnt/pfs/users/can.jin/public/norm_stats/assets/",
                asset_id="fast_3_data_carry_box_b1_fast_0316_merged",
            ),
            base_config=DataConfig(prompt_from_task=True),
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                            "prompt": "prompt",

                        }
                    )
                ]
            ),
            adapt_to_pi=False,
            action_horizon=20,
            mask_list=[7, -1, 7, -1, 4],
            zero_mask_list=[-16, 3, -1, 12],
            use_delta_joint_actions=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("/shared_disk/models/projects/openpi/openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=40_000,
        # batch_size=1,
        # num_workers=1,  
        batch_size=128,
        num_workers=64,
        checkpoint_base_dir = "/shared_disk/users/can.jin/model/openpi",
    ),
    # h01 arm config
    TrainConfig(
        name="pi05_build_block_0501_merged",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            # 记载数据路径
            repo_id="/shared_disk/users/can.jin/dataset/h01_robot/h01_benchmark/build_block_0501_merged",
            assets=AssetsConfig(
                # 加载norm路径 计算时不用管
                assets_dir="/mnt/pfs/users/can.jin/public/norm_stats/assets",
                asset_id="build_block_0501_merged",
            ),
            base_config=DataConfig(prompt_from_task=True),
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "state": "observation.state",
                            "actions": "action",
                            "images": {
                                "cam_high": "observation.images.cam_fisheye_front",
                                "cam_left_wrist_up": "observation.images.cam_left_wrist_up",
                                "cam_right_wrist_up": "observation.images.cam_right_wrist_up",
                                "cam_left_wrist_down": "observation.images.cam_left_wrist_down",
                                "cam_right_wrist_down": "observation.images.cam_right_wrist_down",
                            },
                            "prompt": "prompt",
                        }
                    )
                ]
            ),
            adapt_to_pi=False,
            action_horizon=16,
            mask_list=[7, -1, 7, -1],
            zero_mask_list=[-16, 16],
            use_delta_joint_actions=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("/shared_disk/models/projects/openpi/openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=40_000,
        batch_size=128,
        num_workers=64,
        checkpoint_base_dir = "/shared_disk/users/can.jin/model/openpi",
    ),
    # h01 arm + waist config
    TrainConfig(
        # 配置名称
        name="pi05_fast_2_carry_subbox_b1_0509_merged",
        # 用pi0还是pi05
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            # lerobot v2.1数据集路径
            repo_id="/shared_disk/users/can.jin/dataset/h01_robot/merge_fast/fast_2_carry_subbox_b1_0509_merged",
            # 下面两个合起来是norm文件的目录，默认norm文件名为norm_stats.json
            assets=AssetsConfig(
                # 加载norm路径 计算时不用管
                assets_dir="/mnt/pfs/users/can.jin/public/norm_stats/assets",
                asset_id="fast_2_carry_subbox_b1_0509_merged",
            ),
            # 使用数据集中prompt
            base_config=DataConfig(prompt_from_task=True),
            # 映射字段，相机 state action
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "state": "observation.state",
                            "actions": "action",
                            "images": {
                                "cam_high": "observation.images.cam_fisheye_front",
                                "cam_left_wrist_up": "observation.images.cam_left_wrist_up",
                                "cam_right_wrist_up": "observation.images.cam_right_wrist_up",
                                "cam_left_wrist_down": "observation.images.cam_left_wrist_down",
                                "cam_right_wrist_down": "observation.images.cam_right_wrist_down",
                            },
                            "prompt": "prompt",
                        }
                    )
                ]
            ),
            # 如果是songling臂数据，需要设成true，h01和g1设为false
            adapt_to_pi=False,
            # 控制最后输出的action维度【50，20】
            action_horizon=20,
            # deltamask，如果是正数，使用相对值，如果是负数使用绝对值，对于qpos，一般夹爪绝对值，其他用相对值
            mask_list=[7, -1, 7, -1, 4],
            # zeromask，使用腰部数据是，一般不会用全部维度，不用的维度需要置零，避免影响loss计算。计算norm和此处都要置零。
            # 负数表示对应的维度不置零，正数表示置零，有的数据会在action后面加上两个动作维度，不用的话，需要置零。
            zero_mask_list=[-16, 1, -3, 12],
            # 是否采用相对动作，和mask_list对应，为true的时候mask_list生效
            use_delta_joint_actions=True,
        ),
        # 使用的预训练权重
        weight_loader=weight_loaders.CheckpointWeightLoader("/shared_disk/models/projects/openpi/openpi-assets/checkpoints/pi05_base/params"),
        # iter步数
        num_train_steps=40_000,
        batch_size=128,
        num_workers=64,
        # 训练结果存放位置
        checkpoint_base_dir = "/shared_disk/users/can.jin/model/openpi",
    ),

    # ===============================================================================================================================
    # h01 arm + waist config  /shared_disk/users/wenhao.lu/benchmark_0525/h01/benchmark_language_task_merged
    TrainConfig(
        name="pi05_h01_benchmark_language_task_merged",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            # 记载数据路径
            repo_id="/shared_disk/users/wenhao.lu/benchmark_0525/h01/benchmark_language_task_merged",
            assets=AssetsConfig(
                # 加载norm路径 计算时不用管
                assets_dir="/shared_disk/users/wenyao.xue/results/openpi/assets",
                asset_id="pi05_h01_benchmark_language_task_merged",
            ),
            base_config=DataConfig(prompt_from_task=True),
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "state": "observation.state",
                            "actions": "action",
                            "images": {
                                "cam_high": "observation.images.cam_fisheye_front",
                                "cam_left_wrist_up": "observation.images.cam_left_wrist_up",
                                "cam_right_wrist_up": "observation.images.cam_right_wrist_up",
                                "cam_left_wrist_down": "observation.images.cam_left_wrist_down",
                                "cam_right_wrist_down": "observation.images.cam_right_wrist_down",
                            },
                            "prompt": "prompt",
                        }
                    )
                ]
            ),
            adapt_to_pi=False,
            action_horizon=16,
            mask_list=[7, -1, 7, -1],
            zero_mask_list=[-16, 16],
            use_delta_joint_actions=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("/shared_disk/models/projects/openpi/openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=40_000,
        batch_size=128,
        num_workers=64,
        checkpoint_base_dir="/shared_disk/users/wenyao.xue/results/openpi/checkpoints",
    ),
    # h01 arm + waist config  /shared_disk/users/wenhao.lu/benchmark_0525/h01/benchmark_train_post_task_merged
    TrainConfig(
        name="pi05_h01_benchmark_train_post_task_merged",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            # 记载数据路径
            repo_id="/shared_disk/users/wenhao.lu/benchmark_0525/h01/benchmark_train_post_task_merged",
            assets=AssetsConfig(
                # 加载norm路径 计算时不用管
                assets_dir="/shared_disk/users/wenyao.xue/results/openpi/assets",
                asset_id="pi05_h01_benchmark_train_post_task_merged",
            ),
            base_config=DataConfig(prompt_from_task=True),
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "state": "observation.state",
                            "actions": "action",
                            "images": {
                                "cam_high": "observation.images.cam_fisheye_front",
                                "cam_left_wrist_up": "observation.images.cam_left_wrist_up",
                                "cam_right_wrist_up": "observation.images.cam_right_wrist_up",
                                "cam_left_wrist_down": "observation.images.cam_left_wrist_down",
                                "cam_right_wrist_down": "observation.images.cam_right_wrist_down",
                            },
                            "prompt": "prompt",
                        }
                    )
                ]
            ),
            adapt_to_pi=False,
            action_horizon=16,
            mask_list=[7, -1, 7, -1],
            zero_mask_list=[-16, 16],
            use_delta_joint_actions=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("/shared_disk/models/projects/openpi/openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=40_000,
        batch_size=128,
        num_workers=64,
        checkpoint_base_dir="/shared_disk/users/wenyao.xue/results/openpi/checkpoints",
    ),
    # h01_task002_old  /shared_disk/users/wenhao.lu/benchmark_0525/h01/benchmark2_language_color_push_button
    TrainConfig(
        name="pi05_h01_benchmark2_language_color_push_button",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            # 记载数据路径
            repo_id="/shared_disk/users/wenhao.lu/benchmark_0525/h01/raw_datasets/benchmark2_language_color_push_button",
            assets=AssetsConfig(
                # 加载norm路径 计算时不用管
                assets_dir="/shared_disk/users/wenyao.xue/results/openpi/assets",
                asset_id="pi05_h01_benchmark2_language_color_push_button",
            ),
            base_config=DataConfig(prompt_from_task=True),
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "state": "observation.state",
                            "actions": "action",
                            "images": {
                                "cam_high": "observation.images.cam_fisheye_front",
                                "cam_left_wrist_up": "observation.images.cam_left_wrist_up",
                                "cam_right_wrist_up": "observation.images.cam_right_wrist_up",
                                "cam_left_wrist_down": "observation.images.cam_left_wrist_down",
                                "cam_right_wrist_down": "observation.images.cam_right_wrist_down",
                            },
                            "prompt": "prompt",
                        }
                    )
                ]
            ),
            adapt_to_pi=False,
            action_horizon=16,
            mask_list=[7, -1, 7, -1],
            zero_mask_list=[-16, 16],
            use_delta_joint_actions=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("/shared_disk/models/projects/openpi/openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=40_000,
        batch_size=128,
        num_workers=64,
        checkpoint_base_dir="/shared_disk/users/wenyao.xue/results/openpi/checkpoints",
    ),

    # 上面训练的5视角有误 ---------------------------------------------------
    # H01 pound-clay: use the OpenPI model with the same dataset and three cameras as the reference GigaBrain run.
    TrainConfig(
        name="pi05_h01_pound_clay_3cam",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=50),
        data=LeRobotAlohaDataConfig(
            repo_id="/shared_disk/datasets/private_datasets/robot_data/lerobot/COL2607079FE-01_260715060111_5fde16",
            assets=AssetsConfig(
                assets_dir="/shared_disk/users/wenyao.xue/results/openpi/assets",
                asset_id="pi05_h01_pound_clay_3cam",
            ),
            base_config=DataConfig(prompt_from_task=True),
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "state": "observation.state",
                            "actions": "action",
                            "images": {
                                "cam_high": "observation.images.cam_fisheye_front",
                                "cam_left_wrist": "observation.images.cam_left_wrist_up",
                                "cam_right_wrist": "observation.images.cam_right_wrist_up",
                            },
                            "prompt": "prompt",
                        }
                    )
                ]
            ),
            adapt_to_pi=False,
            action_horizon=16,
            mask_list=[7, -1, 7, -1],
            zero_mask_list=[-16, 16],
            use_delta_joint_actions=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("/shared_disk/models/projects/openpi/openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=30_000,
        batch_size=256,
        num_workers=64,
        checkpoint_base_dir="/shared_disk/users/wenyao.xue/results/openpi/checkpoints",
    ),
    # H01 sort-blocks: mirror the three-camera, arm-only OpenPI setup; match the reference batch size and train steps.
    TrainConfig(
        name="pi05_h01_sort_blocks_3cam",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=50),
        data=LeRobotAlohaDataConfig(
            repo_id="/shared_disk/datasets/private_datasets/robot_data/lerobot/COL26070792F-01_260715223632_4dd226",
            assets=AssetsConfig(
                assets_dir="/shared_disk/users/wenyao.xue/results/openpi/assets",
                asset_id="pi05_h01_sort_blocks_3cam",
            ),
            base_config=DataConfig(prompt_from_task=True),
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "state": "observation.state",
                            "actions": "action",
                            "images": {
                                "cam_high": "observation.images.cam_fisheye_front",
                                "cam_left_wrist": "observation.images.cam_left_wrist_up",
                                "cam_right_wrist": "observation.images.cam_right_wrist_up",
                            },
                            "prompt": "prompt",
                        }
                    )
                ]
            ),
            adapt_to_pi=False,
            action_horizon=16,
            mask_list=[7, -1, 7, -1],
            zero_mask_list=[-16, 16],
            use_delta_joint_actions=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("/shared_disk/models/projects/openpi/openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=30_000,
        batch_size=256,
        num_workers=64,
        checkpoint_base_dir="/shared_disk/users/wenyao.xue/results/openpi/checkpoints",
    ),
    # pi05_h01_push_button_beside_plate_3cam
    TrainConfig(
        name="pi05_h01_push_button_beside_plate_3cam",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=50),
        data=LeRobotAlohaDataConfig(
            repo_id=[
                "/shared_disk/datasets/private_datasets/robot_data/lerobot/COL2607096EF-01_260710150250_1975bc",
                "/shared_disk/datasets/private_datasets/robot_data/lerobot/COL260709613-01_260712111702_5c9618",
                "/shared_disk/datasets/private_datasets/robot_data/lerobot/COL2607099C4-01_260712163959_16da02",
            ],
            assets=AssetsConfig(
                assets_dir="/shared_disk/users/wenyao.xue/results/openpi/assets",
                asset_id="pi05_h01_push_button_beside_plate_3cam",
            ),
            base_config=DataConfig(prompt_from_task=False),
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "state": "observation.state",
                            "actions": "action",
                            "images": {
                                "cam_high": "observation.images.cam_fisheye_front",
                                "cam_left_wrist": "observation.images.cam_left_wrist_up",
                                "cam_right_wrist": "observation.images.cam_right_wrist_up",
                            },
                            "prompt": "task",
                        }
                    )
                ]
            ),
            adapt_to_pi=False,
            action_horizon=16,
            mask_list=[7, -1, 7, -1],
            zero_mask_list=[-16, 16],
            use_delta_joint_actions=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("/shared_disk/models/projects/openpi/openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=30_000,
        batch_size=256,
        num_workers=64,
        checkpoint_base_dir="/shared_disk/users/wenyao.xue/results/openpi/checkpoints",
    ),
    # pi05_h01_push_button_beside_plate_3cam_merge
    TrainConfig(
        name="pi05_h01_push_button_beside_plate_3cam_merge",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=50),
        data=LeRobotAlohaDataConfig(
            repo_id="/shared_disk/users/can.jin/dataset/h01_robot/push_button_20260718",
            assets=AssetsConfig(
                assets_dir="/shared_disk/users/wenyao.xue/results/openpi/assets",
                asset_id="pi05_h01_push_button_beside_plate_3cam_merge",
            ),
            base_config=DataConfig(prompt_from_task=True),
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "state": "observation.state",
                            "actions": "action",
                            "images": {
                                "cam_high": "observation.images.cam_fisheye_front",
                                "cam_left_wrist": "observation.images.cam_left_wrist_up",
                                "cam_right_wrist": "observation.images.cam_right_wrist_up",
                            },
                            "prompt": "prompt",
                        }
                    )
                ]
            ),
            adapt_to_pi=False,
            action_horizon=16,
            mask_list=[7, -1, 7, -1],
            zero_mask_list=[-16, 16],
            use_delta_joint_actions=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("/shared_disk/models/projects/openpi/openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=30_000,
        batch_size=256,
        num_workers=64,
        checkpoint_base_dir="/shared_disk/users/wenyao.xue/results/openpi/checkpoints",
    ),
    
    # ---------------------------------------------------------------------
    # h01_task001  TODO /shared_disk/benchmark/post_train_val_data/pick_fork_h01
    TrainConfig(
        name="pi05_h01_task_001_pick_fork_into_basket",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            repo_id="/shared_disk/benchmark/post_train_val_data/pick_fork_h01",
            assets=AssetsConfig(
                assets_dir="/shared_disk/users/wenyao.xue/results/openpi/assets",
                asset_id="pi05_h01_task_001_pick_fork_into_basket",
            ),
            base_config=DataConfig(prompt_from_task=True),
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "state": "observation.state",
                            "actions": "action",
                            "images": {
                                "cam_high": "observation.images.cam_fisheye_front",
                                "cam_left_wrist": "observation.images.cam_left_wrist_up",
                                "cam_right_wrist": "observation.images.cam_right_wrist_up",
                            },
                            "prompt": "prompt",
                        }
                    )
                ]
            ),
            adapt_to_pi=False,
            action_horizon=16,
            mask_list=[7, -1, 7, -1],
            zero_mask_list=[-16, 16],
            use_delta_joint_actions=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("/shared_disk/models/projects/openpi/openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=40_000,
        batch_size=128,
        num_workers=64,
        checkpoint_base_dir="/shared_disk/users/wenyao.xue/results/openpi/checkpoints",
    ),
    # h01_task002  TODO /mnt/pfs/users/wenyao.xue/code/.shared/data/benchmark/post_train_val_data/push_button_h01_17feat_007
    TrainConfig(
        name="pi05_h01_task_002_push_button",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            # 记载数据路径
            repo_id="/mnt/pfs/users/wenyao.xue/code/.shared/data/benchmark/post_train_val_data/push_button_h01_17feat_007",
            assets=AssetsConfig(
                # 加载norm路径 计算时不用管
                assets_dir="/mnt/pfs/users/wenyao.xue/code/.shared/results/gigabrain/assets",
                asset_id="pi05_h01_task_002_push_button",
            ),
            base_config=DataConfig(prompt_from_task=True),
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "state": "observation.state",
                            "actions": "action",
                            "images": {
                                "cam_high": "observation.images.cam_fisheye_front",
                                "cam_left_wrist": "observation.images.cam_left_wrist_up",
                                "cam_right_wrist": "observation.images.cam_right_wrist_up",
                            },
                            "prompt": "prompt",
                        }
                    )
                ]
            ),
            adapt_to_pi=False,
            action_horizon=16,
            mask_list=[7, -1, 7, -1],
            zero_mask_list=[-16, 16],
            use_delta_joint_actions=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("/shared_disk/models/projects/openpi/openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=40_000,
        batch_size=128,
        num_workers=64,
        checkpoint_base_dir="/shared_disk/users/wenyao.xue/results/openpi/checkpoints",
    ),
    # h01_task002_5view  TODO /mnt/pfs/users/wenyao.xue/code/.shared/data/benchmark/post_train_val_data/push_button_h01_17feat_007
    TrainConfig(
        name="pi05_h01_task_002_push_button_5view",
        model=pi0_config.Pi0Config(
            pi05=True,
            image_keys=(
                "base_0_rgb",
                "left_wrist_1_rgb",
                "right_wrist_1_rgb",
                "left_wrist_2_rgb",
                "right_wrist_2_rgb",
            ),
        ),
        data=LeRobotAlohaDataConfig(
            # 记载数据路径
            repo_id="/mnt/pfs/users/wenyao.xue/code/.shared/data/benchmark/post_train_val_data/push_button_h01_17feat_007",
            assets=AssetsConfig(
                # 加载norm路径 计算时不用管
                assets_dir="/mnt/pfs/users/wenyao.xue/code/.shared/results/gigabrain/assets",
                asset_id="pi05_h01_task_002_push_button",
            ),
            base_config=DataConfig(prompt_from_task=True),
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "state": "observation.state",
                            "actions": "action",
                            "images": {
                                "cam_high": "observation.images.cam_fisheye_front",
                                "cam_left_wrist_up": "observation.images.cam_left_wrist_up",
                                "cam_right_wrist_up": "observation.images.cam_right_wrist_up",
                                "cam_left_wrist_down": "observation.images.cam_left_wrist_down",
                                "cam_right_wrist_down": "observation.images.cam_right_wrist_down",
                            },
                            "prompt": "prompt",
                        }
                    )
                ]
            ),
            adapt_to_pi=False,
            action_horizon=16,
            mask_list=[7, -1, 7, -1],
            zero_mask_list=[-16, 16],
            use_delta_joint_actions=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("/shared_disk/models/projects/openpi/openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=40_000,
        batch_size=128,
        num_workers=64,
        checkpoint_base_dir="/shared_disk/users/wenyao.xue/results/openpi/checkpoints",
    ),
    # h01_task004  TODO /mnt/pfs/users/wenyao.xue/code/.shared/data/benchmark/post_train_data/table_setting_h01
    TrainConfig(
        name="pi05_h01_task_004_table_setting",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            # 记载数据路径
            repo_id="/mnt/pfs/users/wenyao.xue/code/.shared/data/benchmark/post_train_data/table_setting_h01",
            assets=AssetsConfig(
                # 加载norm路径 计算时不用管
                assets_dir="/shared_disk/users/wenyao.xue/results/openpi/assets",
                asset_id="pi05_h01_task_004_table_setting",
            ),
            base_config=DataConfig(prompt_from_task=True),
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "state": "observation.state",
                            "actions": "action",
                            "images": {
                                "cam_high": "observation.images.cam_fisheye_front",
                                "cam_left_wrist": "observation.images.cam_left_wrist_up",
                                "cam_right_wrist": "observation.images.cam_right_wrist_up",
                            },
                            "prompt": "prompt",
                        }
                    )
                ]
            ),
            adapt_to_pi=False,
            action_horizon=16,
            mask_list=[7, -1, 7, -1],
            zero_mask_list=[-16, 16],
            use_delta_joint_actions=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("/shared_disk/models/projects/openpi/openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=40_000,
        batch_size=128,
        num_workers=64,
        checkpoint_base_dir="/shared_disk/users/wenyao.xue/results/openpi/checkpoints",
    ),
    # h01_task005  TODO /mnt/pfs/users/wenyao.xue/code/.shared/data/benchmark/post_train_data/brush_table_h01
    TrainConfig(
        name="pi05_h01_task_005_brush_table",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            # 记载数据路径
            repo_id="/mnt/pfs/users/wenyao.xue/code/.shared/data/benchmark/post_train_data/brush_table_h01",
            assets=AssetsConfig(
                # 加载norm路径 计算时不用管
                assets_dir="/shared_disk/users/wenyao.xue/results/openpi/assets",
                asset_id="pi05_h01_task_005_brush_table",
            ),
            base_config=DataConfig(prompt_from_task=True),
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "state": "observation.state",
                            "actions": "action",
                            "images": {
                                "cam_high": "observation.images.cam_fisheye_front",
                                "cam_left_wrist": "observation.images.cam_left_wrist_up",
                                "cam_right_wrist": "observation.images.cam_right_wrist_up",
                            },
                            "prompt": "prompt",
                        }
                    )
                ]
            ),
            adapt_to_pi=False,
            action_horizon=16,
            mask_list=[7, -1, 7, -1],
            zero_mask_list=[-16, 16],
            use_delta_joint_actions=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("/shared_disk/models/projects/openpi/openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=40_000,
        batch_size=128,
        num_workers=64,
        checkpoint_base_dir="/shared_disk/users/wenyao.xue/results/openpi/checkpoints",
    ),


    # agilex aloha config  benchmark_0525/agilex/benchmark_train_post_task_merged
    TrainConfig(
        # 任务名
        name="pi05_benchmark_train_post_task_merged",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            # 加载多数据路径
            repo_id='/shared_disk/users/wenhao.lu/benchmark_0525/agilex/benchmark_train_post_task_merged',
            assets=AssetsConfig(
                # 加载norm路径
                assets_dir="/shared_disk/users/wenyao.xue/results/openpi/assets",
                asset_id="pi05_benchmark_train_post_task_merged",
            ),
            base_config=DataConfig(prompt_from_task=True),
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                            "prompt": "prompt",
                        }
                    )
                ]
            ),
            # base_config=DataConfig(
            #     local_files_only=True,  # Set to True for local-only datasets.
            # ),
            use_delta_joint_actions=True,
            action_horizon=14,
            mask_list=[6, -1, 6, -1],
            zero_mask_list=[-14, 18],
            adapt_to_pi=True, # Aloha/Songling
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("/shared_disk/models/projects/openpi/openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=40_000,
        batch_size=128,
        num_workers=64,
        checkpoint_base_dir="/shared_disk/users/wenyao.xue/results/openpi/checkpoints",
    ),
    # agilex aloha config  benchmark_0525/agilex/benchmark_language_task_merged
    TrainConfig(
        # 任务名
        name="pi05_benchmark_language_task_merged",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            # 加载多数据路径
            repo_id='/shared_disk/users/wenhao.lu/benchmark_0525/agilex/benchmark_language_task_merged',
            assets=AssetsConfig(
                # 加载norm路径
                assets_dir="/shared_disk/users/wenyao.xue/results/openpi/assets",
                asset_id="pi05_benchmark_language_task_merged",
            ),
            base_config=DataConfig(prompt_from_task=True),
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                            "prompt": "prompt",
                        }
                    )
                ]
            ),
            # base_config=DataConfig(
            #     local_files_only=True,  # Set to True for local-only datasets.
            # ),
            use_delta_joint_actions=True,
            action_horizon=14,
            mask_list=[6, -1, 6, -1],
            zero_mask_list=[-14, 18],
            adapt_to_pi=True, # Aloha/Songling
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("/shared_disk/models/projects/openpi/openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=40_000,
        batch_size=128,
        num_workers=64,
        checkpoint_base_dir="/shared_disk/users/wenyao.xue/results/openpi/checkpoints",
    ),
    # agilex_task001  benchmark_0525/agilex/benchmark2_language_object
    TrainConfig(
        # 任务名
        name="pi05_benchmark2_language_object",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            # 加载多数据路径
            repo_id='/shared_disk/users/wenhao.lu/benchmark_0525/agilex/benchmark2_language_object',
            assets=AssetsConfig(
                # 加载norm路径
                assets_dir="/shared_disk/users/wenyao.xue/results/openpi/assets",
                asset_id="pi05_benchmark2_language_object",
            ),
            base_config=DataConfig(prompt_from_task=True),
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                            "prompt": "prompt",
                        }
                    )
                ]
            ),
            # base_config=DataConfig(
            #     local_files_only=True,  # Set to True for local-only datasets.
            # ),
            use_delta_joint_actions=True,
            action_horizon=14,
            mask_list=[6, -1, 6, -1],
            zero_mask_list=[-14, 18],
            adapt_to_pi=True, # Aloha/Songling
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("/shared_disk/models/projects/openpi/openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=40_000,
        batch_size=128,
        num_workers=64,
        checkpoint_base_dir="/shared_disk/users/wenyao.xue/results/openpi/checkpoints",
    ),
    # agilex_task002  /shared_disk/benchmark/post_train_val_data/push_button
    TrainConfig(
        # 任务名
        name="pi05_post_train_val_data_push_button",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            # 加载多数据路径
            repo_id='/shared_disk/benchmark/post_train_val_data/push_button',
            assets=AssetsConfig(
                # 加载norm路径
                assets_dir="/shared_disk/users/wenyao.xue/results/openpi/assets",
                asset_id="pi05_post_train_val_data_push_button",
            ),
            base_config=DataConfig(prompt_from_task=True),
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                            "prompt": "prompt",
                        }
                    )
                ]
            ),
            # base_config=DataConfig(
            #     local_files_only=True,  # Set to True for local-only datasets.
            # ),
            use_delta_joint_actions=True,
            action_horizon=14,
            mask_list=[6, -1, 6, -1],
            zero_mask_list=[-14, 18],
            adapt_to_pi=True, # Aloha/Songling
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("/shared_disk/models/projects/openpi/openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=40_000,
        batch_size=128,
        num_workers=64,
        checkpoint_base_dir="/shared_disk/users/wenyao.xue/results/openpi/checkpoints",
    ),
    # agilex_task003  /shared_disk/benchmark/post_train_val_data/knead_dough_fix
    TrainConfig(
        # 任务名
        name="pi05_post_train_val_data_knead_dough_fix",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            # 加载多数据路径
            repo_id='/shared_disk/benchmark/post_train_val_data/knead_dough_fix',
            assets=AssetsConfig(
                # 加载norm路径
                assets_dir="/shared_disk/users/wenyao.xue/results/openpi/assets",
                asset_id="pi05_post_train_val_data_knead_dough_fix",
            ),
            base_config=DataConfig(prompt_from_task=True),
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                            "prompt": "prompt",
                        }
                    )
                ]
            ),
            use_delta_joint_actions=True,
            action_horizon=14,
            mask_list=[6, -1, 6, -1],
            zero_mask_list=[-14, 18],
            adapt_to_pi=True, # Aloha/Songling
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("/shared_disk/models/projects/openpi/openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=40_000,
        batch_size=128,
        num_workers=64,
        checkpoint_base_dir="/shared_disk/users/wenyao.xue/results/openpi/checkpoints",
    ),
    # agilex_task004  /shared_disk/benchmark/post_train_data/table_setting_agilex
    TrainConfig(
        name="pi05_agilex_task_004_table_setting",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            repo_id='/shared_disk/benchmark/post_train_data/table_setting_agilex',
            assets=AssetsConfig(
                assets_dir="/shared_disk/users/wenyao.xue/results/openpi/assets",
                asset_id="pi05_agilex_task_004_table_setting",
            ),
            base_config=DataConfig(prompt_from_task=True),
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                            "prompt": "prompt",
                        }
                    )
                ]
            ),
            use_delta_joint_actions=True,
            action_horizon=14,
            mask_list=[6, -1, 6, -1],
            zero_mask_list=[-14, 18],
            adapt_to_pi=True, # Aloha/Songling
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("/shared_disk/models/projects/openpi/openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=40_000,
        batch_size=128,
        num_workers=64,
        checkpoint_base_dir="/shared_disk/users/wenyao.xue/results/openpi/checkpoints",
    ),
    # agilex_task005  /shared_disk/benchmark/post_train_data/brush_table_agilex
    TrainConfig(
        # 任务名
        name="pi05_agilex_task_005_brush_table",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            # 加载多数据路径
            repo_id='/shared_disk/benchmark/post_train_data/brush_table_agilex',
            assets=AssetsConfig(
                # 加载norm路径
                assets_dir="/shared_disk/users/wenyao.xue/results/openpi/assets",
                asset_id="pi05_agilex_task_005_brush_table",
            ),
            base_config=DataConfig(prompt_from_task=True),
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                            "prompt": "prompt",
                        }
                    )
                ]
            ),
            use_delta_joint_actions=True,
            action_horizon=14,
            mask_list=[6, -1, 6, -1],
            zero_mask_list=[-14, 18],
            adapt_to_pi=True, # Aloha/Songling
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("/shared_disk/models/projects/openpi/openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=40_000,
        batch_size=128,
        num_workers=64,
        checkpoint_base_dir="/shared_disk/users/wenyao.xue/results/openpi/checkpoints",
    ),

    # ===============================================================================================================================

    #
    # Fine-tuning DROID configs.
    #
    TrainConfig(
        # This config is for fine-tuning pi0-FAST-base on the *full* DROID dataset.
        # We use RLDS data loading to make training on this large dataset tractable.
        # For fine-tuning on your own DROID dataset, see below.
        name="pi0_fast_full_droid_finetune",
        model=pi0_fast.Pi0FASTConfig(
            action_dim=8,
            action_horizon=16,
            max_token_len=180,
        ),
        data=RLDSDroidDataConfig(
            repo_id="droid",
            # Set this to the path to your DROID RLDS dataset (the parent directory of the `droid` directory).
            rlds_data_dir="<path_to_droid_rlds_dataset>",
            action_space=droid_rlds_dataset.DroidActionSpace.JOINT_POSITION,
        ),
        # weight_loader=weight_loaders.CheckpointWeightLoader("/shared_disk/models/projects/openpi/openpi-assets/checkpoints/pi0_fast_base/params"),
        weight_loader=weight_loaders.CheckpointWeightLoader("/shared_disk/models/projects/openpi/openpi-assets/checkpoints/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        num_train_steps=100_000,  # 100k steps should be sufficient, takes ~2 days on 8x H100s
        batch_size=256,
        log_interval=100,
        save_interval=5000,
        keep_period=20_000,
        num_workers=0,  # Important: RLDS DataLoader requires num_workers=0, handles multi-processing internally
    ),
    TrainConfig(
        # This config is for fine-tuning pi05 on the *full* DROID dataset.
        # We use RLDS data loading to make training on this large dataset tractable.
        # For fine-tuning on your own DROID dataset, see below.
        name="pi05_full_droid_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
        ),
        data=RLDSDroidDataConfig(
            repo_id="droid",
            # Set this to the path to your DROID RLDS dataset (the parent directory of the `droid` directory).
            rlds_data_dir="/mnt/pi-data/kevin",
            action_space=droid_rlds_dataset.DroidActionSpace.JOINT_POSITION,
            assets=AssetsConfig(
                assets_dir="/shared_disk/models/projects/openpi/openpi-assets/checkpoints/pi05_base/assets",
                asset_id="droid",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("/shared_disk/models/projects/openpi/openpi-assets/checkpoints/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        num_train_steps=100_000,
        batch_size=256,
        log_interval=100,
        save_interval=5000,
        keep_period=10_000,
        num_workers=0,  # Important: RLDS DataLoader requires num_workers=0, handles multi-processing internally
    ),
    TrainConfig(
        # This config is for fine-tuning pi05-DROID on a custom (smaller) DROID dataset.
        # Here, we use LeRobot data format (like for all other fine-tuning examples)
        # To convert your custom DROID dataset (<10s of hours) to LeRobot format, see examples/droid/convert_droid_data_to_lerobot.py
        name="pi05_droid_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,  # pi05 is trained with 32-dim actions
            action_horizon=16,
        ),
        data=LeRobotDROIDDataConfig(
            # Replace with your custom DROID LeRobot dataset repo id.
            repo_id="your_hf_username/my_droid_dataset",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(
                # Important: reuse the original DROID norm stats during fine-tuning!
                assets_dir="/shared_disk/models/projects/openpi/openpi-assets/checkpoints/pi05_base/assets",
                asset_id="droid",
            ),
        ),
        # weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_droid/params"),
        weight_loader=weight_loaders.CheckpointWeightLoader("/shared_disk/models/projects/openpi/openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=20_000,
        batch_size=32,
    ),

    # ALOHA Sim configs. This config is used to demonstrate how to train on a simple simulated environment.
    TrainConfig(
        name="pi0_aloha_sim",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            repo_id="lerobot/aloha_sim_transfer_cube_human",
            default_prompt="Transfer cube",
            use_delta_joint_actions=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("/shared_disk/models/projects/openpi/openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=20_000,
    ),


    # PI05 EBench task26, 3-view, 19-dim base state/action. ==================================================================================
    TrainConfig(
        name="pi05_ebench_task26",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=20,
            discrete_state_input=False,
        ),
        data=LeRobotEBenchTeleopDataConfig(
            repo_id="auto",
            assets=AssetsConfig(
                assets_dir="/shared_disk/users/wenyao.xue/results/openpi/assets",
                asset_id="pi05_ebench_task26",
            ),
            base_config=DataConfig(prompt_from_task=True),
            use_delta_joint_actions=True,
            use_base=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/shared_disk/models/projects/openpi/openpi-assets/checkpoints/pi05_base/params"
        ),
        checkpoint_base_dir="/shared_disk/users/wenyao.xue/results/openpi/checkpoints",
        num_train_steps=100_000,
        batch_size=128,
        num_workers=64,
        log_interval=100,
        save_interval=5000,
        keep_period=50_000,
    ),
    # Official Inference config for EBench
    TrainConfig(
        name="pi0_ebench_all",
        model=pi0_config.Pi0Config(action_horizon=50),
        data=LeRobotEBenchDataConfig(
            repo_id="your/generalist/repo_id",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
            action_sequence_keys=["action.joints","action.gripper","action.base"],
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        pytorch_weight_path="/path/to/your/pytorch_weight_path",
        num_train_steps=200_000, 
        keep_period=50_000,
        batch_size=128, 
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=200_000,
            decay_lr=5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        num_workers=24,
    ),
    TrainConfig(
        name="pi05_ebench_all",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=50),
        data=LeRobotEBenchDataConfig(
            repo_id="/shared_disk/users/wenyao.xue/data/EBench-Dataset/",
            assets=AssetsConfig(
                assets_dir="/shared_disk/users/wenyao.xue/results/openpi/pi05-ebench-generalist/200000/assets",
                asset_id="ebench",
            ),
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
            action_sequence_keys=["action.joints","action.gripper","action.base"],
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/shared_disk/models/projects/openpi/openpi-assets/checkpoints/pi05_base/params"
        ),
        pytorch_weight_path="/path/to/your/pytorch_weight_path",
        num_train_steps=200_000, 
        batch_size=128, 
        keep_period=50_000,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=200_000,
            decay_lr=5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        num_workers=24,
    ),

    # PI05 RoboCasa, modality.json-based LeRobot datasets. ==================================================================================
    TrainConfig(
        name="pi05_robocasa_task300",
        model=pi0_config.Pi0Config(
            pi05=True,
        ),
        data=LeRobotRobocasaDataConfig(
            repo_id="auto",
            assets=AssetsConfig(
                assets_dir="/shared_disk/users/wenyao.xue/results/openpi/assets",
                asset_id="pi05_robocasa_task300",
            ),
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/shared_disk/models/projects/openpi/openpi-assets/checkpoints/pi05_base/params"
        ),
        checkpoint_base_dir="/shared_disk/users/wenyao.xue/results/openpi/checkpoints",
        num_train_steps=100_000,
        batch_size=128,
        num_workers=32,
        log_interval=100,
        save_interval=5000,
        keep_period=10_000,
    ),
    TrainConfig(
        name="pi05_pretrain_human300",
        model=pi0_config.Pi0Config(
            pi05=True,
            max_token_len=200,
        ),
        data=LeRobotRobocasaDataConfig(
            data_dirs=_dataset_soup("pretrain_human300"),
            assets=AssetsConfig(
                assets_dir="/shared_disk/users/wenyao.xue/results/openpi/assets",
                asset_id="",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/shared_disk/models/projects/openpi/openpi-assets/checkpoints/pi05_base/params"
        ),
        checkpoint_base_dir="/shared_disk/users/wenyao.xue/results/openpi/checkpoints",
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=2.5e-5,
            decay_steps=100_000,
            decay_lr=2.5e-6,
        ),
        num_train_steps=100_000,
        batch_size=64,
        num_workers=4,
        save_interval=5000,
        keep_period=10_000,
    ),

]


if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def cli() -> TrainConfig:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})


def get_config(config_name: str) -> TrainConfig:
    """Get a config by name."""
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")

    return _CONFIGS_DICT[config_name]
