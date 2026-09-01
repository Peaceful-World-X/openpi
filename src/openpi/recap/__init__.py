"""硬件无关的 RECAP 在线采集、标注和迭代编排接口。"""

from openpi.recap.advantage import LabeledEpisode
from openpi.recap.advantage import compute_n_step_advantage
from openpi.recap.advantage import compute_n_step_reward_sums
from openpi.recap.advantage import label_advantages
from openpi.recap.advantage import write_labels
from openpi.recap.collector import InterventionDecision
from openpi.recap.collector import ReCAPEnvironment
from openpi.recap.collector import ReCAPRolloutCollector
from openpi.recap.collector import ReCAPStep
from openpi.recap.episode import ReCAPFrame
from openpi.recap.episode import ReCAPOfflineEpisode
from openpi.recap.episode import load_episodes
from openpi.recap.episode import load_recap_episodes
from openpi.recap.episode import save_episodes
from openpi.recap.episode import save_recap_episodes
from openpi.recap.pipeline import OnlineReCAPRunner
from openpi.recap.pipeline import ReCAPPipelineHooks
from openpi.recap.rewards import ReCAPRewardConfig
from openpi.recap.rewards import build_episode_rewards
from openpi.recap.rewards import compute_episode_returns

__all__ = [
    "InterventionDecision",
    "LabeledEpisode",
    "OnlineReCAPRunner",
    "ReCAPEnvironment",
    "ReCAPFrame",
    "ReCAPOfflineEpisode",
    "ReCAPPipelineHooks",
    "ReCAPRewardConfig",
    "ReCAPRolloutCollector",
    "ReCAPStep",
    "build_episode_rewards",
    "compute_episode_returns",
    "compute_n_step_advantage",
    "compute_n_step_reward_sums",
    "label_advantages",
    "load_episodes",
    "load_recap_episodes",
    "save_episodes",
    "save_recap_episodes",
    "write_labels",
]
