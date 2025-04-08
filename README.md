<br />
<p align="center">
  <h1 align="center">HAZARD Challenge: Embodied Decision Making in Dynamically Changing Environments (ICLR 2024) </h1>
  <p align="center">
Qinhong Zhou*, Sunli Chen*, Yisong Wang, Haozhe Xu, Weihua Du, Hongxin Zhang,
Yilun Du, Joshua B. Tenenbaum, Chuang Gan
  </p>
  <p align="center">
    <a href='https://arxiv.org/abs/2401.12975'>
      <img src='https://img.shields.io/badge/Paper-PDF-red?style=flat&logo=arXiv&logoColor=red' alt='Paper PDF'>
    </a>
    <a href='https://embodied-agi.cs.umass.edu/hazard' style='padding-left: 0.5rem;'>
      <img src='https://img.shields.io/badge/Project-Page-blue?style=flat&logo=Google%20chrome&logoColor=blue' alt='Project Page'>
    </a>
  </p>
  <p align="center">
    <img src="pics/overview.png" alt="Logo" width="95%">
  </p>
</p>

*News* Mar 20: HAZARD challenge will be one of the challenges in CVPR Embodied AI Workshop 2025. The [leaderboard submission](documentation/get_started/submit.md#submit-to-the-leaderboard) is open now, welcome 2025 submissions! (will end in 1 June)

## Detailed documentations

### [Get started](documentation/get_started/overview.md)
* [Overview](documentation/get_started/overview.md)
* [Installation](documentation/get_started/install.md)
* [Create your own agent and submit](documentation/get_started/submit.md)
* [Common utils](documentation/get_started/common_utils.md)

### [Agent documents](documentation/agents/agent.md)

* [Observations](documentation/agents/observations.md)

* [Action space](documentation/agents/action_space.md)

* Default agents
  * [RL agent](documentation/agents/rl_agent.md)
  * [MCTS agent](documentation/agents/MCTS_agent.md)
  * [Random agent](documentation/agents/random_agent.md)
  * [Greedy agent](documentation/agents/greedy_agent.md)
  * [Rule-based agent](documentation/agents/rule_based_agent.md)
  * [LLM-based agent](documentation/agents/LLM-based%20agent.md)
    * Version 1.0
    * Version 2.0

* [Customized agent](documentation/agents/custom_agent.md)

### Baseline results

* Full log files can be found in [this link](https://drive.google.com/file/d/1Z1xtble1nCpq1tyuiwYD4wzynRi1MwVK/view?usp=sharing)
* W/o perception (use ground truth segmentations)
