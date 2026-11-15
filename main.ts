// free-model-aggregator-02: one server, two client wires, fourteen free OpenRouter models.
//
// The trial gives this server a brief and an env, and runs it as `node /bot/main.ts` from the
// repository's root, with Node 22's standard library alone. The brief is at the path in
// $MAKEFUN_PROMPT_FILE and carries two JSON documents: the free model list (fourteen OpenRouter
// `:free` models, each with its gateway id, its published rating in milli-points, what it can
// do — tools, vision, JSON mode — and its context length) and the six prompts (each one's id,
// api, whether it streams, its weight and the requirements a model must meet to serve it).
//
// The server reads the brief, builds one catalog, then:
//   1. listens on 0.0.0.0 at a free port (never 8080: the env itself is there on one layout);
//   2. POSTs $MAKEFUN_ENV_URL/register {"protocol_version": 1, "port": <port>, "catalog": <catalog>};
//   3. serves the client's requests: GET /v1/models (in the shape of the auth header it got),
//      POST /v1/chat/completions and POST /v1/messages, each with a JSON body or with
//      server-sent events;
//   4. polls GET $MAKEFUN_ENV_URL/work every 100 ms and exits 0 when the env says {"done": true}.
//
// HOW A REQUEST IS SERVED
//   Every request is answered by ONE gateway call, always as a plain JSON call (never streamed
//   upstream): the gateway answers with the whole ANSWER text, its usage counts and its stop
//   reason, and lib/wire.ts turns that one answer into whatever wire the client used. The client
//   stream is synthesized here, in the client's event order, from that one answer. An Anthropic
//   request is translated to a Chat Completions body first and the answer translated back, because
//   these ids have no Messages route on the gateway.
//
//   Routing ranks every free model that meets the request's requirements by the brief's
//   rating_milli, best first (ties keep the brief's order, so the order is fixed). The ratings
//   make the first pick optimal on every prompt shape: a JSON-mode prompt goes to
//   nemotron-3.5-super-vl (1303, the best JSON-capable model) and every other prompt to
//   inkling-2-small (1475, the best model overall). Serving every prompt with its best model is
//   what quality 100 needs.
//
//   Failover: a 429, a 5xx or an unreachable gateway is never forwarded. The request is retried
//   on the next candidate in the ranking until one answers 2xx; a 400 or 401 is this server's
//   own fault (a bad request or a bad key) and stops the walk. Every refusal is recorded, and a
//   prompt that is still served after refusals carries the trail in the
//   `x-makefun-upstream-failures` header, so the failover column can see the recovery. A refused
//   call costs no tie-break: only calls the provider answered count.
//
//   `max_tokens` is set on every upstream call (the gateway refuses a call without an output
//   cap): the client's own cap, clamped to the trial's 8192, or 1024 when the client sent none.
//
// DETERMINISM
//   No wall clock, no unseeded randomness: every choice comes from the brief, the requests and
//   the gateway's answers. Response ids and timestamps are fixed constants. The catalog's
//   base_url is OpenRouter's public endpoint, never the gateway's trial-local address, so a live
//   pass and its replay register the same document.
//
// `node main.ts --list` prints the same catalog JSON and exits 0 before any server starts; the
// register body's catalog must be that same document. Build it once, as this file does, and use
// the one object.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  buildCatalog,
  capabilitiesOf,
  freeModels,
  isRecord,
  meetsRequirements,
  readBrief,
  type CatalogReport,
  type Json,
  type ModelRow,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicAnswerFrom,
  anthropicError,
  anthropicModelList,
  anthropicStreamFrom,
  anthropicToOpenAI,
  errorCodeOf,
  errorMessageOf,
  extractChatAnswer,
  openAIAnswerFrom,
  openAIError,
  openAIModelList,
  openAIStreamFrom,
  type ProviderAnswer,
  type ServedModel,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one gateway call may take before it counts as refused, in milliseconds. */
const UPSTREAM_TIMEOUT_MS = 20000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 8 * 1024 * 1024;
/** The trial's output cap: the gateway refuses a call asking for more. */
const MAX_UPSTREAM_TOKENS = 8192;
/** The output cap when a client request carries none. */
const DEFAULT_UPSTREAM_TOKENS = 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';
const GATEWAY_URL = (process.env.OPENAI_BASE_URL ?? '').replace(/\/+$/, '');
/** The trial's dummy key for the entry's gateway lane; the gateway swaps it for the entrant's own key. */
const GATEWAY_KEY = process.env.OPENAI_API_KEY ?? '';

const brief = readBrief(PROMPT_FILE);
const catalog: CatalogReport = buildCatalog(brief);
/** How many client API requests have arrived: the manifest is in arrival order, one request at a time. */
let manifestIndex = 0;

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/**
 * The gateway's Chat Completions route. OPENAI_BASE_URL is the OpenAI-compatible root (`…/v1` on
 * the trial network); a base that does not already end in `/v1` gets one, so both spellings reach
 * the route.
 */
function chatCompletionsUrl(): string {
  if (GATEWAY_URL === '') return '';
  return GATEWAY_URL.endsWith('/v1') ? `${GATEWAY_URL}/chat/completions` : `${GATEWAY_URL}/v1/chat/completions`;
}

/** The free models the brief lists, as GET /v1/models names them. */
function servedModels(): ServedModel[] {
  return freeModels(brief).map((model) => ({ id: model.id, provider: model.provider, display_name: model.display_name }));
}

/**
 * Every free model that can serve one request, best rating first. The sort is stable, so models
 * tied on rating keep the brief's order and the ranking is the same on a live pass and its
 * replay. The first pick is the prompt's best model by the sealed ratings: JSON-mode prompts
 * open with nemotron-3.5-super-vl (1303, the best JSON-capable model) and every other prompt
 * with inkling-2-small (1475, the best model overall).
 */
function candidatesFor(requires: Requirement): ModelRow[] {
  return freeModels(brief)
    .filter((model) => meetsRequirements(requires, capabilitiesOf(model)))
    .sort((a, b) => b.rating_milli - a.rating_milli);
}

/** The requirements of the request now arriving: the brief's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const entry = brief.prompts[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  // The brief's prompt list is the declared source of each request's requirements; this fallback
  // is a guess for a brief that holds none.
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the brief has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** One refused gateway call, as the upstream-failures trail reports it. */
interface UpstreamFailure {
  status: number;
  code: string;
  model: string | null;
}

type CallResult = { ok: true; answer: ProviderAnswer } | { ok: false; failure: UpstreamFailure; message: string };

/**
 * One JSON call to one candidate model on the gateway's Chat Completions route. Upstream is
 * always plain JSON: the client's stream, when it asked for one, is synthesized from this one
 * answer, so no upstream call ever streams. `max_tokens` is always set: the client's own cap
 * clamped to the trial's 8192, or 1024 when the client sent none.
 */
async function callOnce(model: string, clientApi: 'openai' | 'anthropic', body: Json): Promise<CallResult> {
  const upstreamBody: Json = clientApi === 'openai' ? { ...body, model } : anthropicToOpenAI(body, model);
  upstreamBody.stream = false;
  delete upstreamBody.stream_options;
  const cap = typeof upstreamBody.max_tokens === 'number' ? Math.min(Math.max(1, Math.floor(upstreamBody.max_tokens)), MAX_UPSTREAM_TOKENS) : DEFAULT_UPSTREAM_TOKENS;
  upstreamBody.max_tokens = cap;
  let res: Response;
  try {
    res = await fetch(chatCompletionsUrl(), {
      method: 'POST',
      headers: { 'content-type': 'application/json', authorization: `Bearer ${GATEWAY_KEY}` },
      body: JSON.stringify(upstreamBody),
      signal: AbortSignal.timeout(UPSTREAM_TIMEOUT_MS),
    });
  } catch (err) {
    return { ok: false, failure: { status: 0, code: 'fetch_error', model }, message: `the gateway could not be reached: ${(err as Error).message}` };
  }
  const text = await res.text();
  let parsed: unknown = null;
  try {
    parsed = JSON.parse(text);
  } catch {
    // Not JSON: an error body the shape cannot name.
  }
  if (!res.ok) {
    const message = errorMessageOf(parsed);
    return {
      ok: false,
      failure: { status: res.status, code: errorCodeOf(parsed), model },
      message: message !== '' ? message : `the gateway answered HTTP ${res.status}`,
    };
  }
  const answer = extractChatAnswer(isRecord(parsed) ? parsed : {});
  if (answer === null) {
    return { ok: false, failure: { status: 502, code: 'bad_upstream_body', model }, message: 'the gateway answered in a shape this server cannot read' };
  }
  return { ok: true, answer };
}

interface Served {
  status: number;
  contentType: string;
  text: string;
  failures: UpstreamFailure[];
}

/** The refusals behind a served prompt as the `x-makefun-upstream-failures` header, or null when none. */
function failuresHeader(failures: UpstreamFailure[]): string | null {
  if (failures.length === 0) return null;
  const text = JSON.stringify(failures.slice(0, 8).map((f) => ({ status: f.status, code: f.code, model: f.model })));
  return Buffer.byteLength(text) > 8192 ? null : text;
}

/** One client request, answered by the best free model that can serve it, with failover. */
async function serve(clientApi: 'openai' | 'anthropic', body: Json, stream: boolean): Promise<Served> {
  const requires = requirementsFor(body);
  const candidates = candidatesFor(requires);
  const failures: UpstreamFailure[] = [];
  let lastStatus = 503;
  let lastError = 'no free model in the brief meets this request';
  for (const candidate of candidates) {
    const result = await callOnce(candidate.id, clientApi, body);
    if (result.ok) {
      const answer = result.answer;
      if (clientApi === 'openai') {
        if (stream) return { status: 200, contentType: 'text/event-stream', text: openAIStreamFrom(answer, candidate.id), failures };
        return { status: 200, contentType: 'application/json', text: JSON.stringify(openAIAnswerFrom(answer, candidate.id)), failures };
      }
      if (stream) return { status: 200, contentType: 'text/event-stream', text: anthropicStreamFrom(answer, candidate.id), failures };
      return { status: 200, contentType: 'application/json', text: JSON.stringify(anthropicAnswerFrom(answer, candidate.id)), failures };
    }
    failures.push(result.failure);
    lastStatus = result.failure.status;
    lastError = result.message;
    // A 400 or 401 is this server's own fault (a bad request or a bad key), not a failover case.
    if (result.failure.status === 400 || result.failure.status === 401) break;
    // A 429, a 5xx, or an unreachable gateway is a failover case: try the next candidate.
  }
  const status = lastStatus === 503 ? 503 : 502;
  return {
    status,
    contentType: 'application/json',
    text: JSON.stringify(clientApi === 'anthropic' ? anthropicError('api_error', lastError) : openAIError(lastError, 'api_error', null)),
    failures,
  };
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendServed(res: ServerResponse, served: Served): void {
  const headers: Record<string, string> = { 'content-type': served.contentType, 'content-length': String(Buffer.byteLength(served.text)) };
  if (served.contentType === 'text/event-stream') {
    headers['cache-control'] = 'no-cache';
    headers.connection = 'keep-alive';
  }
  const trail = failuresHeader(served.failures);
  if (trail !== null) headers['x-makefun-upstream-failures'] = trail;
  res.writeHead(served.status, headers);
  res.end(served.text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') {
    const text = JSON.stringify(anthropicError(type, message));
    res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
    res.end(text);
  } else {
    const text = JSON.stringify(openAIError(message, type, null));
    res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
    res.end(text);
  }
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') {
    const text = JSON.stringify(anthropicModelList(list));
    res.writeHead(200, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
    res.end(text);
  } else {
    const text = JSON.stringify(openAIModelList(list));
    res.writeHead(200, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
    res.end(text);
  }
}

/** POST /v1/chat/completions: an OpenAI-shaped client. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  sendServed(res, await serve('openai', body, body.stream === true));
}

/** POST /v1/messages: an Anthropic-shaped client. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  sendServed(res, await serve('anthropic', body, body.stream === true));
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the trial's prompt count. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the trial holds ${reply.requests} prompts`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  if (chatCompletionsUrl() === '') throw new Error('OPENAI_BASE_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
