// The brief's two documents, the free model list, and the catalog.
//
// A trial's brief (the path in MAKEFUN_PROMPT_FILE) is prose with two JSON documents in it, and the
// prose around them does not matter: this file finds each by its `schema` member, under this contest's
// prefix (`free-model-aggregator-02/`).
//
//   - the free model list (`free-model-aggregator-02/models/v1`): one row a model, keyed by the id the
//     gateway routes (`openrouter/<vendor>/<model>:free`), with the model's public rating in
//     milli-points, what it can do (tools, vision, JSON mode, context length) and its free flag. Only a
//     free model may serve a prompt; this baseline never routes a paid one.
//   - the six prompts (`free-model-aggregator-02/prompts/v1`): the prompts the house's client will
//     send, in order, each with its id, its api (`openai` or `anthropic`), whether it streams, its
//     weight (1 or 2) and the requirements a model must meet to serve it. The client sends the bodies;
//     this baseline routes each arrival by its row, which is why the prompt list is read before the
//     first request.
//
// A brief may spell the model list `models` or `ratings` and the prompts `prompts` or `manifest` (the
// sibling -01 contest's names), and one document may carry both; a document is found by the arrays it
// carries, never by a name alone. The brief's own prose is never parsed.
//
// Everything here is deterministic: no clock and no randomness, so `node main.ts --list` and the
// register body build the same document, byte for byte, on a live pass and on its replay.

// ---------------------------------------------------------------------------
// THE FOURTEEN MODELS, EXPLAINED
// ---------------------------------------------------------------------------
// The brief's model list is the whole routing universe: exactly fourteen OpenRouter
// `:free` ids, each routed as `openrouter/<vendor>/<model>:free`. Their sealed
// ratings (milli-points) and capabilities, from the brief:
//   thinkingmachines/inkling-2-small:free             1475  tools yes, vision yes, JSON no
//   nvidia/nemotron-3.5-super-vl:free                 1303  tools yes, vision yes, JSON yes
//   nvidia/nemotron-3-ultra-550b-a55b:free            1285  tools yes, vision no,  JSON no
//   nvidia/nemotron-3.5-lightning:free                1221  tools yes, vision no,  JSON no
//   google/gemma-4-31b-it:free                        1210  tools yes, vision yes, JSON yes
//   cohere/north-mini-code:free                       1200  tools yes, vision no,  JSON no
//   baidu/ernie-5.5-flash-preview:free                1150  tools yes, vision no,  JSON yes
//   dots-studio/dots-3-note-preview:free              1150  tools yes, vision yes, JSON yes
//   google/gemma-4-26b-a4b-it:free                    1150  tools yes, vision yes, JSON yes
//   inclusionai/ling-3.2-flash:free                   1150  tools yes, vision yes, JSON yes
//   liquid/lfm-3-1.7b:free                            1150  tools yes, vision no,  JSON yes
//   nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free 1150 tools yes, vision yes, JSON no
//   particle-ai/particle-1:free                       1150  tools yes, vision yes, JSON yes
//   recursive/flint-v1.0:free                         1150  tools yes, vision no,  JSON yes
// Two facts fall out of this table. First, inkling-2-small is the best model overall
// but cannot do JSON mode, so a JSON prompt's best model is nemotron-3.5-super-vl —
// the routing in main.ts is a consequence of this table, not a choice made there.
// Second, every model has tools, so a tools prompt can be served by any of the
// fourteen; a vision prompt by the nine with vision; a JSON prompt by the ten with
// JSON. The catalog, the model list and the routing all read this one table, so a
// brief that changed a rating would move the routing with it.

import { readFileSync } from 'node:fs';

export const MODELS_SCHEMA = 'free-model-aggregator-02/models/v1';
export const PROMPTS_SCHEMA = 'free-model-aggregator-02/prompts/v1';

/** Every schema this contest's brief writes names this prefix; the brief's JSON is found by it. */
const SCHEMA_PREFIX = 'free-model-aggregator-02/';

/** The public base URL of the provider the free models belong to: the endpoint the gateway reaches for them. */
export const OPENROUTER_PUBLIC_BASE = 'https://openrouter.ai/api/v1';
/** The environment variable an entrant's OpenRouter key is pushed under (an entry secret, kept out of the sandbox). */
export const OPENROUTER_KEY_ENV = 'OPENROUTER_API_KEY';

export type Json = Record<string, unknown>;
export const isRecord = (v: unknown): v is Json => v !== null && typeof v === 'object' && !Array.isArray(v);

/** One free model as the brief lists it. `id` is the id the gateway routes. */
export interface ModelRow {
  id: string;
  /** The provider the model belongs to, as the brief names it (`openrouter`). */
  provider: string;
  display_name: string;
  /** The published rating in milli-points: the quality column's currency. */
  rating_milli: number;
  context_length: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
  free: boolean;
}

/** What one prompt needs: a model without it cannot serve the prompt. */
export interface Requirement {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  min_context?: number;
}

/** One prompt the client will send, as the brief lists it. */
export interface PromptRow {
  id: string;
  api: 'openai' | 'anthropic';
  stream: boolean;
  weight: 1 | 2;
  requires: Requirement;
}

/** The brief as this baseline reads it. `error` is set when a document could not be read. */
export interface Brief {
  models: ModelRow[];
  prompts: PromptRow[];
  error: string | null;
}

/** One model in the catalog document. */
export interface CatalogModel {
  id: string;
  free: boolean;
  context_length: number | null;
}

/** One provider in the catalog document. */
export interface CatalogProvider {
  id: string;
  base_url: string;
  key_env: string;
  wire: 'openai';
  models: CatalogModel[];
}

/** What `--list` prints and what the entry POSTs to /register: the same document, byte for byte. */
export interface CatalogReport {
  providers: CatalogProvider[];
}

/** What a model can do, from the brief's row (null: the brief did not say). */
export interface Capabilities {
  context_length: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
}

/** Every JSON object in `text` that carries a string `schema`, in the order they appear. */
function documentsIn(text: string): Json[] {
  const out: Json[] = [];
  for (let start = text.indexOf('{'); start !== -1; start = text.indexOf('{', start + 1)) {
    const end = objectEnd(text, start);
    if (end === -1) continue;
    try {
      const value: unknown = JSON.parse(text.slice(start, end + 1));
      if (isRecord(value) && typeof value.schema === 'string') out.push(value);
    } catch {
      // Not JSON on its own: look on.
    }
  }
  return out;
}

/** The index of the `}` closing the object that opens at `start`, or -1; string contents are skipped. */
function objectEnd(text: string, start: number): number {
  let depth = 0;
  let inString = false;
  for (let i = start; i < text.length; i++) {
    const ch = text[i];
    if (inString) {
      if (ch === '\\') i += 1;
      else if (ch === '"') inString = false;
    } else if (ch === '"') inString = true;
    else if (ch === '{') depth += 1;
    else if (ch === '}') {
      depth -= 1;
      if (depth === 0) return i;
    }
  }
  return -1;
}

function stringOf(v: unknown): string {
  return typeof v === 'string' ? v : '';
}

function numberOrNull(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
}

function boolOrNull(v: unknown): boolean | null {
  return typeof v === 'boolean' ? v : null;
}

/**
 * The first document under this contest's prefix that carries one of `keys` as an array: the document
 * and the key it was found under. A brief that spells its lists in the sibling's names still reads.
 */
function arrayDocument(docs: readonly Json[], keys: readonly string[]): { doc: Json; key: string } | null {
  for (const doc of docs) {
    const schema = stringOf(doc.schema);
    if (!schema.includes(SCHEMA_PREFIX)) continue;
    for (const key of keys) if (Array.isArray(doc[key])) return { doc, key };
  }
  return null;
}

/**
 * One model row. The id is the row's `id`, or its `model` under the provider that names it: a row that
 * gives the bare OpenRouter id (`vendor/model:free`) and names `openrouter` is routed as the gateway
 * spells it (`openrouter/vendor/model:free`). The brief lists free models, so a row is free unless it
 * says `free: false`; a paid row is never routed.
 */
function modelOf(v: unknown): ModelRow | null {
  if (!isRecord(v)) return null;
  const raw = stringOf(v.id) !== '' ? stringOf(v.id) : stringOf(v.model);
  if (raw === '') return null;
  const named = stringOf(v.provider) !== '' ? stringOf(v.provider) : stringOf(v.wire);
  const id = named === 'openrouter' && !raw.toLowerCase().startsWith('openrouter/') ? `openrouter/${raw}` : raw;
  return {
    id,
    provider: named === '' ? 'openrouter' : named,
    display_name: stringOf(v.display_name) !== '' ? stringOf(v.display_name) : raw,
    rating_milli: numberOrNull(v.rating_milli) ?? 0,
    context_length: numberOrNull(v.context_length),
    tools: boolOrNull(v.tools),
    vision: boolOrNull(v.vision),
    json: boolOrNull(v.json),
    free: v.free !== false,
  };
}

function promptOf(v: unknown): PromptRow | null {
  if (!isRecord(v)) return null;
  const id = stringOf(v.id);
  if (id === '') return null;
  const requires = isRecord(v.requires) ? v.requires : {};
  return {
    id,
    api: v.api === 'anthropic' ? 'anthropic' : 'openai',
    stream: v.stream === true,
    weight: v.weight === 2 ? 2 : 1,
    requires: {
      ...(requires.tools === true ? { tools: true } : {}),
      ...(requires.vision === true ? { vision: true } : {}),
      ...(requires.json === true ? { json: true } : {}),
      ...(typeof requires.min_context === 'number' && Number.isFinite(requires.min_context) ? { min_context: requires.min_context } : {}),
    },
  };
}

/** The brief at `path`, its two documents found under this contest's schema prefix. */
export function readBrief(path: string): Brief {
  if (path === '') return { models: [], prompts: [], error: 'MAKEFUN_PROMPT_FILE is not set' };
  let text: string;
  try {
    text = readFileSync(path, 'utf8');
  } catch (err) {
    return { models: [], prompts: [], error: `cannot read ${path}: ${(err as Error).message}` };
  }
  const docs = documentsIn(text);
  const modelsDoc = arrayDocument(docs, ['models', 'ratings']);
  if (modelsDoc === null) {
    return { models: [], prompts: [], error: `the brief at ${path} holds no ${MODELS_SCHEMA} document` };
  }
  const models = (modelsDoc.doc[modelsDoc.key] as unknown[]).flatMap((v) => {
    const m = modelOf(v);
    return m === null ? [] : [m];
  });
  const promptsDoc = arrayDocument(docs, ['prompts', 'manifest']);
  if (promptsDoc === null) {
    return { models, prompts: [], error: `the brief at ${path} holds no ${PROMPTS_SCHEMA} document` };
  }
  const prompts = (promptsDoc.doc[promptsDoc.key] as unknown[]).flatMap((v) => {
    const p = promptOf(v);
    return p === null ? [] : [p];
  });
  return { models, prompts, error: null };
}

/** The models this entry may serve: the brief's free ones, in the brief's order. */
export function freeModels(brief: Brief): ModelRow[] {
  return brief.models.filter((m) => m.free);
}

/** What a model can do, as the brief's row says it. */
export function capabilitiesOf(model: ModelRow): Capabilities {
  return {
    context_length: model.context_length,
    tools: model.tools,
    vision: model.vision,
    json: model.json,
  };
}

// A requirement is a hard gate, and an unknown capability never passes one: a model
// whose brief row does not say whether it can do JSON is treated as if it cannot,
// because routing a prompt to a model that might not meet it would spend an upstream
// call and risk the quality column. The same conservatism holds for tools, vision
// and context: min_context fails when the context length is unknown or too short.
// Only a requirement the prompt actually states is checked — a prompt with no
// requirements is served by every free model, and the rating sort picks inkling.

/** Whether a model's capabilities meet one prompt's requirements. An unknown capability never meets one. */
export function meetsRequirements(requires: Requirement, caps: Capabilities): boolean {
  if (requires.tools === true && caps.tools !== true) return false;
  if (requires.vision === true && caps.vision !== true) return false;
  if (requires.json === true && caps.json !== true) return false;
  if (requires.min_context !== undefined && (caps.context_length === null || caps.context_length < requires.min_context)) return false;
  return true;
}

// The catalog lists the brief's free models in the brief's own order, deduplicated by
// id: the order is the probe order, never alphabetical, because the scorer compares
// the registered catalog against the replay's byte for byte and a re-sort would break
// the match. The base_url is the provider's public endpoint
// (https://openrouter.ai/api/v1), never the gateway's trial-local address, for the
// same reason: the gateway's origin differs between a live pass and its replay, and
// the catalog must be identical on both. One provider, one wire ("openai"), one
// key_env ("OPENROUTER_API_KEY"): the entry's OpenRouter key is pushed as a secret
// and never appears in the catalog or the repository.

/**
 * The catalog the brief's free model list makes: one provider, the models it serves.
 *
 * The base_url is the provider's public endpoint, never the gateway's trial-local address: the gateway's
 * origin differs between a live pass and its replay, and the catalog is part of the state export a
 * replay must reproduce byte for byte.
 *
 * This baseline lists every free model in the brief — exactly the fourteen dialable ids the catalog
 * column scores — so keep it that way: register this set and keep GET /v1/models (both auth headers)
 * exactly that set, since listing fewer loses the column as surely as listing more.
 */
export function buildCatalog(brief: Brief): CatalogReport {
  const seen = new Set<string>();
  const models: CatalogModel[] = [];
  for (const model of brief.models) {
    if (!model.free || seen.has(model.id)) continue;
    seen.add(model.id);
    models.push({ id: model.id, free: true, context_length: model.context_length });
  }
  if (models.length === 0) return { providers: [] };
  return {
    providers: [
      {
        id: 'openrouter',
        base_url: OPENROUTER_PUBLIC_BASE,
        key_env: OPENROUTER_KEY_ENV,
        wire: 'openai',
        models,
      },
    ],
  };
}
