# Laya as an RL policy: summary

Full paper: [Laya as an RL policy.md](Laya%20as%20an%20RL%20policy.md) · Code and results: [experiments/laya_policy](../experiments/laya_policy)

## The idea

OpenAI's hide-and-seek agents learned from scratch. They used a small network, PPO, and self-play
against past versions of themselves, over hundreds of millions of games.

We tried a different route: use Laya, a small open decision model, as the agent's brain. Laya has
421M parameters and answers multiple-choice questions about a situation written in English. Each move
becomes a Laya question:

- **State:** "You hold the Queen. You checked, then your opponent bet 1 chip."
- **Options:** fold / call.

PPO with an opponent pool trains only Laya's small decision head (26.5M parameters). The encoder
stays frozen.

## What we tested

- **Kuhn poker.** A tiny poker game where exploitability (how much a perfect opponent could win
  against the policy) can be computed exactly. Transfer test: a 4-card deck described in words but
  never played.
- **Grid hide-and-seek.** A text version on random walled maps, scored against scripted chasers and
  fleers. Transfer test: bigger, unseen maps.
- **Baselines.** A lookup table and a small from-scratch network (the hide-and-seek recipe), compared
  at the same number of games.
- **Hardware.** Everything ran on one 8 GB M1 laptop.

## What we found

| Question | Answer |
|---|---|
| Does Laya learn faster than a from-scratch network? | **Barely in poker, no on the grid.** Poker exploitability (lower is better): Laya 0.155, small network 0.189, lookup table 0.092. Grid hider score: Laya −0.58, small network −0.25 at the same number of games. |
| Does language give transfer to unseen variants? | **No.** On the 4-card deck, trained Laya did worse than untrained Laya. In 2 of 5 runs it folded the Ace, which the text calls "the highest card". |
| Does an opponent pool help, as in hide-and-seek? | **Yes in poker** (Laya 0.208 → 0.155; small network 0.277 → 0.189). On the grid it made no difference. |
| Cost? | Each Laya update costs about 1,000× more compute than the small network's. |

**Why it failed:**

- Untrained, Laya has almost no game sense. In poker it ignores its card; in hide-and-seek it moves
  almost at random.
- RL then learned surface words instead of concepts. In poker it learned the card names but not "the
  highest card", so the unseen Ace confused it.
- On the grid, RL shifted every move toward "north" instead of learning to read where the other
  player and the walls were.

## Verdict

This is a negative result. It's new in a narrow sense: we found no prior work that trains an encoder
decision model like Laya as an RL policy. But it doesn't show an advantage, so it suits a workshop or
a negative-results write-up, not a main conference.

The idea would only pay off where states are naturally text and too varied for hand-made features,
for example rules that change each game. It would also need Laya's prior to be protected during
training, by a KL penalty or a small adapter. Until then, a small network with an opponent pool is
the better choice on a laptop.
