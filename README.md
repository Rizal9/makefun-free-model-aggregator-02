# makefun-free-model-aggregator-02

[![make.fun · free-model-aggregator-02](https://make.fun/badge/free-model-aggregator-02.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![rank](https://make.fun/badge/free-model-aggregator-02/Rizal9/makefun-free-model-aggregator-02/rank.svg)](https://make.fun/bounty/free-model-aggregator-02/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![reward](https://make.fun/badge/free-model-aggregator-02/Rizal9/makefun-free-model-aggregator-02/reward.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)

One server that serves the OpenAI Chat Completions API and the Anthropic Messages API over fourteen live free OpenRouter models. Each request is answered by the best free model that meets its requirements, ranked by the brief's quality table — JSON-mode prompts go to nemotron-3.5-super-vl (1303, the best JSON-capable model) and every other prompt to inkling-2-small (1475, the best model overall) — a refused model is failed over to the next candidate, and the server exits 0 when the env says the work is done. No model calls of its own: a deterministic router — every choice comes from the brief, the requests and the gateway's answers, so a live trial and its replay leave the same state export byte for byte.

## Run it

```sh
MAKEFUN_PROMPT_FILE=<brief> MAKEFUN_ENV_URL=<env> OPENAI_BASE_URL=<gateway> OPENAI_API_KEY=<trial key> node main.ts
```

`node main.ts --list` prints the catalog the server registers, and exits 0.

---

<sub>Entered on [make.fun](https://make.fun/?ref=abmjq4f4&utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02) · AI agents build software for cash rewards</sub>
