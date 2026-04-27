# Botlatro Project Plan

## Project Goal

Build an AI agent that can eventually play Balatro at superhuman strength.
The project should progress from reliable local control, to rule-based play,
to statistical evaluation, to search, to learned value and policy models.

The agent should be able to:

1. Play legal Balatro actions reliably.
2. Beat low stakes consistently.
3. Beat high stakes consistently.
4. Discover strong strategies beyond normal human intuition.
5. Evaluate runs statistically across thousands of seeds.
6. Use search plus learned value and policy models to outperform expert human play.

This project is intended for offline/local research using an owned copy of the
game. It should not be used for cheating leaderboards, competitions, or any
online/shared systems. Avoid distributing copyrighted game code or assets.

## Target Architecture

```text
Balatro
  |
  v
BalatroBot / Lua API bridge
  |
  v
Python environment wrapper
  |
  v
Rule-based bot
  |
  v
Simulation + evaluation tools
  |
  v
Search planner
  |
  v
Neural value/policy model
  |
  v
RL / imitation / self-improvement loop
  |
  v
Superhuman agent
```

## Phase 0: Rules and Success Metrics

Before building the AI, define what "superhuman" means and make progress
measurable.

Track these metrics for every bot version:

- Win rate by stake.
- Average ante reached.
- Average final score.
- Average money at death or win.
- Average joker value.
- Average shop decision quality.
- Win rate across a fixed seed set.
- Win rate across random seeds.
- Average runtime per run.

Create fixed benchmark seed pools:

- 100 seeds for fast testing.
- 1,000 seeds for serious evaluation.
- 10,000+ seeds for final benchmarks.

Use the same seeds for each benchmark so bot versions can be compared fairly.

Milestone:

```text
Bot v0.1
Stake: White
Seeds: 100
Win rate: 12%
Average ante: 5.2
Average runtime: 48 seconds/run
```

Without measurement, it will be hard to know whether the AI is improving.

## Phase 1: Local Bot Interface

Do not start with screen reading or image recognition. Balatro is a
state-based strategy game, so the bot should read structured state directly.

Use BalatroBot or a similar local mod bridge to expose game state and legal
actions through an API. The Python side should be responsible for policy,
evaluation, benchmarking, and training.

Recommended stack:

- Python 3.11 or 3.12.
- uv or Poetry for dependency management.
- BalatroBot or equivalent local API bridge.
- Gymnasium.
- PyTorch.
- pytest.
- numpy.
- pandas.
- matplotlib.
- TensorBoard or Weights & Biases.

Gymnasium-style environments are a good target because reinforcement-learning
tools expect a `reset()` / `step()` interface with observations, action spaces,
rewards, and termination flags.

Initial project structure:

```text
balatro-ai/
  README.md
  pyproject.toml

  src/
    balatro_ai/
      api/
        client.py
        actions.py
        state.py

      env/
        balatro_env.py
        observations.py
        rewards.py
        action_space.py

      bots/
        random_bot.py
        greedy_bot.py
        rule_bot.py
        search_bot.py
        neural_bot.py

      eval/
        run_seed.py
        benchmark.py
        compare.py
        metrics.py

      training/
        imitation.py
        ppo.py
        value_model.py

      data/
        replay_logger.py
        replay_loader.py

  tests/
    test_hand_eval.py
    test_action_masks.py
    test_rewards.py
```

First code goal:

1. Start a run.
2. Read the current state.
3. Print hand, jokers, money, blind, and score requirement.
4. Choose a random legal action.
5. Send the action to Balatro.
6. Repeat until the run ends.
7. Log the result.

Milestone:

- The bot can complete 10 full runs without crashing, even if it plays badly.

## Phase 2: Clean State Encoding

This is the most important engineering phase. The AI needs a clean, stable
representation of game state.

State fields to model:

- Current ante.
- Current blind.
- Required score.
- Current score.
- Hands remaining.
- Discards remaining.
- Money.
- Deck size.
- Cards in hand.
- Cards in deck, if visible or known.
- Jokers.
- Consumables.
- Vouchers.
- Shop contents.
- Pack contents.
- Enhanced cards.
- Seals.
- Editions.
- Card ranks and suits.
- Hand levels.
- Current modifiers.

Create two state formats.

### Human-Readable State

Used for debugging, logs, and analysis.

```json
{
  "money": 12,
  "ante": 4,
  "blind": "Big Blind",
  "hand": ["AS", "AH", "AD", "7C", "2D", "9S", "KC", "KD"],
  "jokers": ["Greedy Joker", "Blueprint", "Hologram"]
}
```

### Neural Observation

Used for ML models.

Include:

- Fixed-size numeric tensor.
- Card features.
- Joker features.
- Economy features.
- Blind features.
- Deck-composition features.
- Legal action mask.

The legal action mask is critical. Balatro has many invalid actions depending
on game phase, and neural models should never be allowed to choose impossible
moves.

Milestone:

- Given any game state, the code can produce readable debug text, an ML
  observation vector, and a legal action mask.

## Phase 3: First Baseline Bots

Start simple. Baselines make later progress meaningful.

### Bot 1: Random Legal Bot

Chooses any legal action.

Purpose:

- Test the API.
- Test action masks.
- Test run completion.
- Establish the minimum baseline.

Expected result: terrible, but useful.

### Bot 2: Greedy Scoring Bot

During blind play:

- Enumerate all playable hands from the current hand.
- Score each hand using known scoring rules.
- Play the hand with the highest immediate score.
- If no good hand exists, discard low-value cards.

In shop:

- Buy nothing, or buy only obviously good things with simple rules.

Expected result: can maybe win some easy runs, but not strong.

### Bot 3: Basic Economy Bot

Add rules:

- Keep interest money when possible.
- Prioritize early flat mult and chips.
- Avoid wasting money before interest thresholds.
- Buy strong scaling jokers.
- Buy planets that match current strategy.
- Use tarot cards to improve deck consistency.

Milestone:

- The rule bot beats White Stake sometimes and is clearly better than random.

## Phase 4: Hand and Score Evaluator

The evaluator is the foundation for search and planning.

It should eventually calculate expected score for:

- High Card.
- Pair.
- Two Pair.
- Three of a Kind.
- Straight.
- Flush.
- Full House.
- Four of a Kind.
- Straight Flush.
- Five of a Kind.
- Flush House.
- Flush Five.

Incrementally add support for:

- Joker effects.
- Card enhancements.
- Editions.
- Seals.
- Planet levels.
- Tarot modifications.
- Retriggers.
- Blueprint-style copying.
- Card order effects.
- Held-in-hand effects.
- Glass, steel, gold, and lucky cards.

First evaluator version:

- Poker hand type.
- Base chips.
- Base mult.
- Planet levels.
- Simple +chips.
- Simple +mult.
- Simple xmult.

Later evaluator version:

- Joker ordering.
- Retriggers.
- Conditional jokers.
- Card enhancements.
- Probability-based effects.

Milestone:

- For 100 test states, the evaluator's predicted score matches actual Balatro
  scoring closely enough to trust it.

## Phase 5: Strong Rule-Based Bot

Before deep learning, build a bot that feels like a good human player.

Split decisions by phase.

### Blind Selection Logic

- Take small blind if the reward is worth it.
- Skip blind if the tag is high value.
- Avoid risky skips if the current build is weak.
- Prefer economy and scaling tags early.
- Prefer rare, negative, or polychrome tags when the build can survive.

### Play and Discard Logic

- Estimate the best playable hand.
- Estimate the best discard to improve the next hand.
- Track the current scoring threshold.
- Avoid overcommitting if one small hand wins.
- Save discards when possible.
- Prefer consistency over maximum theoretical score.

### Shop Logic

Give each item a value score:

```text
joker_value = immediate_power + scaling_power + synergy + rarity_bonus - cost_penalty
planet_value = hand_strategy_match + level_scaling - cost
tarot_value = deck_fixing + economy + synergy
voucher_value = long_term_power - cost
pack_value = expected_value - opportunity_cost
reroll_value = money_available * chance_to_improve
```

Milestone:

- The bot has recognizable strategy: builds around good jokers, maintains
  economy, uses tarot cards intelligently, and avoids obvious losing buys.
- Target: consistent White Stake wins.

## Phase 6: Evaluation Harness

Create automated benchmarking commands.

Example commands:

```bash
python -m balatro_ai.eval.benchmark --bot rule_bot --seeds 100 --stake white
python -m balatro_ai.eval.benchmark --bot greedy_bot --seeds 1000 --stake red
python -m balatro_ai.eval.compare --bot-a rule_bot_v1 --bot-b rule_bot_v2
```

Example output:

```text
Bot: rule_bot_v2
Seeds: 1000
Stake: White
Win rate: 38.2%
Average ante: 7.1
Average final money: 16.4
Average runtime: 22 sec/run
Most common death: Ante 6 Boss Blind
```

Save replays for every run.

Replay fields:

- State.
- Legal actions.
- Chosen action.
- Reward.
- Score.
- Money.
- Outcome.
- Bot version.
- Seed.

This replay data becomes training data later.

Milestone:

- Overnight tests can run and clearly show whether a bot version improved.

## Phase 7: Search Planner

Search is the first major strength jump. For each decision, generate legal
candidate actions and estimate their value.

### During Hand Play

For every possible play or discard:

- Calculate immediate score.
- Estimate probability of drawing useful cards next.
- Estimate survival chance.
- Estimate value of saving hands and discards.
- Choose the action with highest expected value.

### During Shop

Evaluate possible purchase paths:

- Buy item A.
- Buy item B.
- Sell joker C.
- Open pack.
- Choose card from pack.
- Reroll.
- Save money.

Then estimate future strength.

Start with beam search:

- Keep the top 10 possible action sequences.
- Expand each sequence.
- Score each resulting state.
- Choose the best first action.

Then add MCTS where randomness matters:

- Selection.
- Expansion.
- Simulation / rollout.
- Backpropagation.

Use MCTS mainly for:

- Discard decisions.
- Pack choices.
- Shop and reroll decisions.
- Risky blind decisions.

Milestone:

- Search bot beats rule bot by a statistically meaningful amount over the
  same seed set.

Example:

```text
Rule bot: 42% White Stake win rate
Search bot: 57% White Stake win rate
```

## Phase 8: First Neural Model

Do not start with end-to-end RL. First train a model to imitate the best
rule/search decisions.

Model inputs:

- Game phase.
- Ante, blind, money, hands, and discards.
- Cards in hand.
- Jokers.
- Consumables.
- Shop contents.
- Deck composition.
- Hand levels.
- Legal action mask.

Model outputs:

- Policy head: probability of each legal action.
- Value head: estimated chance of winning or expected final value.

Training data:

- Rule bot replays.
- Search bot replays.
- Human-labeled decisions, if available.
- Winning runs.
- Failed runs.
- High-score runs.

Losses:

- Policy loss: imitate chosen action.
- Value loss: predict final outcome, ante reached, or run score.

Milestone:

- The neural model can imitate the search bot's choices and play legal full
  runs. It may not be stronger yet.

## Phase 9: Neural-Guided Search

Combine neural models with search in an AlphaZero-style loop adapted for
Balatro.

Use the neural network to make search cheaper and smarter:

- Policy model suggests promising actions.
- Value model evaluates future states.
- Search explores only top actions.
- Bot chooses the best searched action.

Decision flow:

1. Generate legal actions.
2. Rank actions with the policy model.
3. Search the top K actions.
4. Evaluate leaf states with the value model.
5. Choose the best action.

Milestone:

- Neural-guided search beats plain search at the same compute budget.

Example:

```text
Plain search, 1 second/decision: 61% win rate
Neural search, 1 second/decision: 68% win rate
```

## Phase 10: Reinforcement Learning

Use RL only after the environment, baselines, evaluator, and replay pipeline
are solid.

Algorithms to try:

- PPO.
- DQN variant with action masking.
- A2C/A3C.
- IMPALA-style scaling if running many workers.

Expected best development order:

1. Rule bot.
2. Search bot.
3. Imitation learning.
4. Neural-guided search.
5. RL fine-tuning.

Avoid pure RL from scratch at first. It may waste huge compute learning obvious
things like not playing bad hands.

Reward signals:

- Survival.
- Beating a blind.
- Ante reached.
- Win.
- Score efficiency.
- Economy health.
- Deck quality.
- Scaling potential.
- Death penalty.
- Wasted money penalty.
- Illegal action attempt penalty.

Be careful with reward shaping. The final reward should still heavily care
about winning the run, reaching higher stakes, and maximizing expected success.

Milestone:

- RL fine-tuning improves the neural/search bot instead of making it worse.

## Phase 11: Synergy System

Balatro is not only about scoring the most right now. Superhuman play requires
long-term build planning.

Represent strategy archetypes:

- Pair build.
- High Card build.
- Flush build.
- Straight build.
- Full House build.
- Steel card build.
- Lucky card economy build.
- Glass cannon build.
- Retrigger build.
- Joker-scaling build.
- Planet-scaling build.
- Tarot deck-fixing build.

Give every item synergy scores:

- Immediate value.
- Scaling value.
- Economy value.
- Consistency value.
- Synergy with current jokers.
- Synergy with current deck.
- Synergy with current hand levels.
- Risk level.

Milestone:

- The bot can make contextual decisions such as:
  - This joker is weak generally, but excellent for the current build.
  - This planet is not useful because the bot is pivoting away from flushes.
  - This tarot card is worth more than money because it fixes deck consistency.
  - This skip tag is too risky because the next blind is unlikely to be beaten.

## Phase 12: Massive Simulation

Superhuman Balatro likely requires a lot of run data.

Parallelize:

- Multiple Balatro instances.
- Multiple seed workers.
- Replay logging.
- Central results database.

Store results in SQLite or DuckDB.

Suggested tables:

- `runs`.
- `steps`.
- `states`.
- `actions`.
- `items_seen`.
- `jokers_owned`.
- `deaths`.
- `wins`.
- `bot_versions`.

Analyze:

- Which jokers correlate with winning?
- Which early purchases are traps?
- Which skip tags are overrated?
- Which hand archetypes survive high stakes?
- When should the bot reroll?
- When should it save money?

Milestone:

- Strategy is mined from thousands of runs instead of guessed.

## Phase 13: High-Stakes Specialization

A bot that beats White Stake is not necessarily good at high stakes.

Train and evaluate separately for:

- White Stake.
- Red Stake.
- Green Stake.
- Black Stake.
- Blue Stake.
- Purple Stake.
- Orange Stake.
- Gold Stake.

Potential model variants:

- `general_model.pt`.
- `gold_stake_model.pt`.
- `challenge_mode_model.pt`.
- `high_score_model.pt`.

Milestone:

- The bot beats high stakes at a rate that exceeds strong human expectations
  across random seeds.

## Phase 14: Analysis Tools

The bot needs explainability so bad decisions can be debugged.

For every decision, output:

- Top 5 actions considered.
- Estimated value of each.
- Reason chosen.
- Risk estimate.
- Survival chance.
- Expected future money.
- Expected score next blind.

Example:

```text
Decision: Buy Blueprint

Reasons:
+ Copies strongest joker
+ Raises estimated blind survival from 71% to 93%
+ Better than reroll EV
- Costs $10, reducing interest

Final value: +0.184 expected win probability
```

Milestone:

- Bad decisions can be explained and debugged instead of only observed.

## Phase 15: Superhuman Push

Once the full system works, strength improvements should come from:

- Better state encoding.
- Better joker and scoring simulation.
- Bigger replay dataset.
- Better value model.
- Longer search budget.
- Stake-specific training.
- Better action abstraction.
- Better shop planning.
- Better discard probability modeling.

The likely strongest architecture is:

```text
Rule engine for legality and scoring
+
Neural policy for move priors
+
Neural value model for future strength
+
Search planner for tactical decisions
+
Statistical evaluator for shop and risk decisions
```

The final agent should not be "just a neural net." It should combine explicit
rules, accurate simulation, learned priors, learned value estimates, search,
and statistical evaluation.

## Recommended Milestone Order

1. Bot can play legal random runs.
   - Goal: 10 complete runs without crashing.
   - Strength: awful.

2. Greedy bot.
   - Goal: plays best immediate scoring hand.
   - Strength: bad but functional.

3. Rule bot.
   - Goal: makes basic shop and economy decisions.
   - Strength: can win easy runs sometimes.

4. Evaluation harness.
   - Goal: compare bots over 100 to 1,000 seeds.
   - Strength: measurable progress.

5. Search bot.
   - Goal: uses lookahead for hands, discards, and shops.
   - Strength: better than basic human mistakes.

6. Replay dataset.
   - Goal: save every state, action, and outcome.
   - Strength: creates training data.

7. Imitation model.
   - Goal: neural bot copies rule/search bot.
   - Strength: fast, but not yet smarter.

8. Neural-guided search.
   - Goal: search uses model to rank and evaluate actions.
   - Strength: potentially expert-level.

9. RL fine-tuning.
   - Goal: improve from self-generated experience.
   - Strength: starts discovering non-obvious lines.

10. High-stakes specialization.
    - Goal: consistent high-stake performance.
    - Strength: superhuman candidate.

## First Implementation Order

Build the first version in this order:

1. Install BalatroBot and confirm Python can read game state.
2. Create `random_bot.py`.
3. Create `balatro_env.py` with `reset()`, `step()`, and `legal_actions()`.
4. Create a hand evaluator.
5. Create `greedy_bot.py`.
6. Create `benchmark.py`.
7. Create `rule_bot.py`.
8. Add replay logging.
9. Add search for hand and discard decisions.
10. Add shop evaluation.

The first serious target is:

```text
A rule/search bot that can beat White Stake consistently across 100 fixed seeds.
```

After that, move to neural models. Starting with deep reinforcement learning
would likely make the project slower instead of faster.

