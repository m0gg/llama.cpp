# Research: slot manager, prompt-similarity, LRU, and disk save/load (feat-cold-slots)

Repo: halo-box-llama.cpp @ 2f1ff393d (branch feat-cold-slots)

Important: there is **no `llama_slot_manager` class** in this codebase. The slot
manager is `server_context` (a single large class) in
`tools/server/server-context.cpp`. Slots are plain structs owned by a vector on
that class.

## 1. Core data structures

### server_slot — `tools/server/server-context.cpp:241`
- `int id` (line 242), `llama_context * ctx_tgt / ctx_dft / ctx_spf` (244-246)
- `slot_state state = SLOT_STATE_IDLE` (line 301); enum `slot_state` at
  `server-context.cpp:102` (IDLE, WAIT_OTHER, ...). `is_processing()` at 478:
  `state != SLOT_STATE_IDLE`.
- `int64_t t_last_used = -1` (line 273) — the LRU clock; set in
  `stop_processing` path at line 572 (`t_last_used = ggml_time_us()`).
- `server_prompt prompt` (line 303) — the in-slot (hot) prompt tokens +
  checkpoints.
- `prompt_save(server_prompt_cache&)` — `server-context.cpp:305`: measures state
  size via `llama_state_seq_get_size_ext` (310-311), allocates a cache entry,
  copies KV bytes out with `llama_state_seq_get_data_ext` (323-326).
- `prompt_load(...)` — `server-context.cpp:331` delegates to
  `server_prompt_cache::load`.
- `prompt_clear()` — `server-context.cpp:340`: `mem.seq_rm(id, -1, -1)` plus
  clearing `prompt`.
- Slot vector: `std::vector<server_slot> slots` — `server-context.cpp:950`,
  grown in init loop `slots.emplace_back()` at 1452-1454 (`for i <
  params_base.n_parallel`), slot fields assigned at 1485-1503.

### server_prompt / cache — `tools/server/server-task.h`
- `struct server_prompt` — `server-task.h:566` (`server_tokens tokens` +
  `std::list<common_prompt_checkpoint> checkpoints`).
- `struct server_prompt_data` — `server-task.h:588` (`main` + `drft` byte
  vectors: the serialized KV state, target + draft).
- `struct server_prompt_cache_state` — `server-task.h:597` (prompt + data).
- `struct server_prompt_cache` — `server-task.h:612`:
  `std::list<server_prompt_cache_state> states` (line 618), `limit_size`
  (bytes, from `--cache-ram`), `limit_tokens` (line 621-624).
  Instantiated at `server-context.cpp:1558`
  (`std::make_unique<server_prompt_cache>(params_base.cache_ram_mib, n_ctx)`),
  gated by `--cache-ram` at 1556-1560.

**Hot vs cold today:** "hot" = KV resident in the llama_context for a slot id
(seq id == slot id). "Warm" = RAM prompt cache (`server_prompt_cache::states`,
LRU-evicted by bytes/tokens). "Cold/disk" exists only via the explicit
`/slots/{id}/save|restore` HTTP API — there is NO automatic offload of slot
state to disk and NO structure that tracks disk-resident ("cold") slots. A cold
slot today is indistinguishable from an empty one in `server_slot`.

## 2. Slot allocation + prompt similarity + LRU

`server_slot * server_context::get_available_slot(const server_task & task)` —
`tools/server/server-context.cpp:1746`. This is THE function to change.

- Explicit slot request: `task.id_slot` -> `get_slot_by_id` (1752-1757;
  `get_slot_by_id` at 1719, `get_slot_by_cmpl_id` at 1732).
- Prompt-similarity matching: 1759-1809.
  - Loop over all slots, skip busy (`is_processing`, 1769) and empty
    (`tokens.empty()`, 1777).
  - LCP similarity: `tokens.get_common_prefix(task.tokens)`;
    `f_sim_cur = lcp_len / task.tokens.size()` (1783-1784).
  - Best slot wins if `f_sim_cur > f_sim_best && f_sim_cur >
    slot_prompt_similarity` (1789).
  - If selected slot would lose >50% of its context (`f_keep < 0.5`), set
    `update_cache = true` (1805-1807).
- LRU fallback: 1811-1833. Picks idle slot with smallest `t_last_used`
  (1822-1825); always sets `update_cache = true`.
- Cache update block: 1835-1856 — `ret->prompt_save(*prompt_cache)`, then
  `ret->prompt_load(...)`; on load failure `prompt_clear()`; then
  `prompt_cache->update()`.
- Called from the task-scheduling loop at `server-context.cpp:2597`.

Prompt-cache internals (RAM LRU) — `tools/server/server-task.cpp`:
- `server_prompt_cache::alloc` — 1711: dedupe fully-contained prompts (1713-1720,
  1738-1748), size-limit eviction `states.pop_front()` (1750-1758),
  bad_alloc shrink (1764-1777).
- `server_prompt_cache::load` — 1793: picks best cached prompt by
  f_keep/f_sim (1804-1823, `f_keep_cur < 0.25` skip at 1813), restores via
  `llama_state_seq_set_data_ext` (1832, 1850), moves prompt into slot and erases
  the cache entry (1862-1864).
- `server_prompt_cache::update` — 1870: evicts oldest by `limit_size` then
  `limit_tokens` (pop_front, 1872-1891).

KV-pressure purge: `try_clear_idle_slots()` — `server-context.cpp:1866` (only
when `kv_unified`); purges one idle non-empty slot per call (1873-1888),
invoked at 3937 with batch-size halving fallback. Its TODO comment (1863-1865)
explicitly anticipates "move slot to level 2 cache instead of removing".

## 3. CLI flags — parse sites (`common/arg.cpp`) and consumption

| Flag | Parse | Storage | Consumed |
|---|---|---|---|
| `-np/--parallel` | arg.cpp:2561 (server) / 2572 (generic) | `common_params::n_parallel` — common/common.h:477 | server-context.cpp:1452 (slot count); server default -1=auto at arg.cpp:1420 |
| `-sps/--slot-prompt-similarity` | arg.cpp:3810-3815 | `common_params::slot_prompt_similarity` — common/common.h:716 (default 0.1f) | copied to `server_context::slot_prompt_similarity` at server-context.cpp:971, 1405; used at 1760-1801 |
| `--slot-save-path` | arg.cpp:3628-3640 (validates dir, appends separator) | `common_params::slot_save_path` — common/common.h:713 | server-context.cpp:4981 (route gating), 5506, 5542 (filepath build) |
| `--cache-ram` | arg.cpp (search `cache_ram_mib`) | `common_params::cache_ram_mib` — common/common.h | server-context.cpp:1558 (prompt_cache ctor) |

## 4. Disk save/restore path (the existing "cold" mechanism)

HTTP handlers — `tools/server/server-context.cpp`:
- `handle_slots_save` — 5498: builds `filepath = params.slot_save_path +
  filename` (5506), posts `SERVER_TASK_TYPE_SLOT_SAVE`.
- `handle_slots_restore` — 5534 (filepath at 5542), `SERVER_TASK_TYPE_SLOT_RESTORE`.
- `handle_slots_erase` — 5571, `SERVER_TASK_TYPE_SLOT_ERASE`.

Task execution — `server_context::process_task` cases:
- SAVE: 2744-2793. Serializes tokens (`slot->prompt.tokens.serialize()`,
  2766) and calls `llama_state_seq_save_file(ctx_tgt, filepath, slot->id, ...)`
  (2773).
- RESTORE: 2794-2858. `llama_state_seq_load_file` (2818, 2821), validates
  tokens (`server_tokens::deserialize`, `validate`, 2828-2836), assigns into
  `slot->prompt.tokens` (2839).
- ERASE: 2859+ -> `prompt_clear()`.

Low-level API — `src/llama-context.cpp`:
- `llama_state_seq_get_size_ext` 4225, `llama_state_seq_get_data_ext` 4229,
  `llama_state_seq_save_file` 4240, `llama_state_seq_load_file` 4251.
- Class methods `llama_context::state_seq_save_file` 3272,
  `state_seq_load_file` 3218 (declared `src/llama-context.h:171-179`).
- These delegate to the memory/KV layer (`common_memory mem` on the slot,
  `llama_memory_seq_rm` etc.).

## 5. What must change for cold slots (summary)

1. `server_slot` (server-context.cpp:241): add cold-slot tracking — e.g. a
   `bool is_cold` / `std::string cold_filepath` / saved-token count, so an
   offloaded slot is distinguishable from an empty one. `t_last_used` (273) is
   the existing LRU anchor.
2. `get_available_slot` (server-context.cpp:1746): the similarity loop (1763)
   and LRU loop (1815) only see in-RAM slots; must also consider cold slots
   (restore-on-match) and must trigger offload (save-to-disk) when evicting an
   idle slot instead of only pushing to the RAM prompt cache.
3. `try_clear_idle_slots` (server-context.cpp:1866): currently destroys idle
   KV; candidate for "offload to disk instead of purge" (its own TODO says so).
4. `server_prompt_cache` (server-task.h:612 / server-task.cpp:1711-1901): the
   RAM LRU eviction (`alloc` pop_front 1750-1758, `update` 1870) is the hook
   where evicted entries would be spilled to disk rather than dropped; a
   disk-backed tier needs filepath bookkeeping parallel to
   `server_prompt_cache_state`.
5. Reuse `llama_state_seq_save_file` / `llama_state_seq_load_file`
   (src/llama-context.cpp:4240/4251) for the actual cold IO — same primitives
   the /slots save/restore endpoints use (server-context.cpp:2773/2818).
6. CLI: `--slot-save-path` (arg.cpp:3628) already provides the directory; new
   flags for enabling automatic cold offload would parse into `common_params`
   (common/common.h:713 area) and gate logic in `server_context` init
   (server-context.cpp:1405, 1556-1558).
