// The client wires, the one upstream wire, and the stream synthesis.
//
// The trial's client calls this server as an OpenAI service (POST /v1/chat/completions) or an
// Anthropic one (POST /v1/messages), with a JSON body or with server-sent events. Every prompt
// goes upstream as ONE plain JSON call to the grader's gateway Chat Completions route — never
// streamed upstream — and this file turns that one answer into whatever wire the client used: a
// JSON body, or a stream synthesized here in the client's event order. Nothing upstream can tell
// a synthesized stream from a forwarded one: the bytes are what the client checks.
//
// The provider's answer is first reduced to one shape (ProviderAnswer): the whole text, the
// usage counts, the stop reason and any tool calls, each as the gateway reported them. The
// renderers below carry those through whole — the `model` field is always the id this server
// really dialled, the finish reason is the upstream's own, and tool calls keep their names,
// because the house scores the model that really answered and checks the wire against the
// client's request.
//
// Everything here is a pure function of its arguments: no clock and no randomness, so the same
// body and the same gateway answer give the same bytes on a live pass and on its replay. The
// model listings use fixed timestamps for the same reason: a `created` that moved between runs
// would be a nondeterministic document.

import { isRecord, type Json } from './providers.ts';

/** One model this server serves, as GET /v1/models lists it. */
export interface ServedModel {
  id: string;
  provider: string;
  display_name: string;
}

/** What one gateway call answered: the whole text, usage, stop reason and tool calls. */
export interface ProviderAnswer {
  text: string;
  input: number;
  output: number;
  /** The upstream finish_reason, as the gateway reported it ('stop', 'length', 'tool_calls', ...). */
  finish: string;
  toolCalls: ToolCall[];
}

/** One tool call the provider made, in OpenAI's shape. */
export interface ToolCall {
  id: string;
  name: string;
  /** The arguments as a JSON string, as the provider sent them. */
  arguments: string;
}

/** A fixed epoch second for every listed model's `created`: nothing in the reply reads the clock. */
const FIXED_OPENAI_CREATED = 1767225600;
/** A fixed instant for every listed model's `created_at`: nothing in the reply reads the clock. */
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';
/** How many characters of answer text one synthesized stream chunk carries. */
const STREAM_PIECE_CHARS = 13;

/** GET /v1/models on an OpenAI-shaped request: the models this server serves, in OpenAI's list shape. */
export function openAIModelList(models: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_OPENAI_CREATED, owned_by: m.provider })),
  };
}

/** GET /v1/models on an Anthropic-shaped request (an x-api-key header): Anthropic's model list shape. */
export function anthropicModelList(models: readonly ServedModel[]): Json {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_ANTHROPIC_CREATED_AT })),
    has_more: false,
  };
}

/** An error in Anthropic's shape (the Messages API's error envelope). */
export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

/** An error in OpenAI's shape (Chat Completions). */
export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

/** The text of a string or of a block array's text blocks; '' when there is none. */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

/** One Anthropic content block as an OpenAI content part: text or an inline image; null for the rest. */
function partOf(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { type: 'text', text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
      return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { type: 'image_url', image_url: { url: source.url } };
  }
  return null;
}

/** A tool_result block's content as text: its text blocks, or its JSON when it holds something else. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** A tool call's arguments as an object: the gateway's JSON string, or {} when it is not one. */
function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

/** OpenAI's finish_reason as Anthropic's stop_reason. */
export function stopReasonOf(finish: unknown): string {
  if (finish === 'length') return 'max_tokens';
  if (finish === 'tool_calls' || finish === 'function_call') return 'tool_use';
  if (finish === 'content_filter') return 'refusal';
  return 'end_turn';
}

/** Split a text into fixed pieces; an empty text is one empty piece, as the providers do. */
function splitText(text: string): string[] {
  const pieces: string[] = [];
  for (let at = 0; at < text.length; at += STREAM_PIECE_CHARS) pieces.push(text.slice(at, at + STREAM_PIECE_CHARS));
  return pieces.length === 0 ? [''] : pieces;
}

/** One server-sent event: `event: <name>` when named, then the JSON data line, then the blank line. */
export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

// ---------------------------------------------------------------------------
// Client -> upstream request translation.
// ---------------------------------------------------------------------------

/**
 * An Anthropic Messages request as an OpenAI Chat Completions request that names `model`: the
 * system prompt becomes a system message, image blocks become image_url data URLs, tool_use
 * blocks become tool_calls, tool_result blocks become tool messages, tools and tool_choice are
 * rewritten, and the output cap, temperature, top_p and stop sequences pass through. Streaming
 * is left to the caller: this server always dials upstream as plain JSON.
 */
export function anthropicToOpenAI(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = textOf(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: assistant ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const toolCalls: Json[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: typeof block.id === 'string' ? block.id : '',
          type: 'function',
          function: {
            name: typeof block.name === 'string' ? block.name : '',
            arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
          },
        });
      }
      const text = textOf(content);
      messages.push({
        role: 'assistant',
        content: text !== '' ? text : null,
        ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
      });
      continue;
    }
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        messages.push({
          role: 'tool',
          tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
          content: toolResultText(block.content),
        });
        continue;
      }
      const part = partOf(block);
      if (part !== null) parts.push(part);
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }

  const out: Json = { model, messages };
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) out.stop = body.stop_sequences;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [
        {
          type: 'function',
          function: {
            name: raw.name,
            description: typeof raw.description === 'string' ? raw.description : '',
            parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
          },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'tool' && typeof choice.name === 'string') out.tool_choice = { type: 'function', function: { name: choice.name } };
  }
  return out;
}

// ---------------------------------------------------------------------------
// Upstream answer -> ProviderAnswer.
// ---------------------------------------------------------------------------

/** A Chat Completions answer as a provider answer, or null when it carries no first choice. */
export function extractChatAnswer(body: Json): ProviderAnswer | null {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : null;
  if (choice === null) return null;
  const message = isRecord(choice.message) ? choice.message : {};
  const text = typeof message.content === 'string' ? message.content : textOf(message.content);
  const usage = isRecord(body.usage) ? body.usage : {};
  const toolCalls: ToolCall[] = [];
  const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  for (const call of calls) {
    if (!isRecord(call) || !isRecord(call.function)) continue;
    const name = typeof call.function.name === 'string' ? call.function.name : '';
    if (name === '') continue;
    toolCalls.push({
      id: typeof call.id === 'string' ? call.id : '',
      name,
      arguments: typeof call.function.arguments === 'string' ? call.function.arguments : '{}',
    });
  }
  return {
    text,
    input: numberOrZero(usage.prompt_tokens),
    output: numberOrZero(usage.completion_tokens),
    finish: typeof choice.finish_reason === 'string' ? choice.finish_reason : 'stop',
    toolCalls,
  };
}

/** The message of a provider error body, in the OpenAI error shape. */
export function errorMessageOf(body: unknown): string {
  if (!isRecord(body)) return '';
  if (isRecord(body.error) && typeof body.error.message === 'string') return body.error.message;
  if (typeof body.message === 'string') return body.message;
  return '';
}

/** The error code of a provider error body, for the upstream-failures trail. */
export function errorCodeOf(body: unknown): string {
  if (!isRecord(body)) return 'upstream_error';
  if (isRecord(body.error)) {
    if (typeof body.error.code === 'string' && body.error.code !== '') return body.error.code;
    if (typeof body.error.type === 'string' && body.error.type !== '') return body.error.type;
  }
  if (typeof body.code === 'string' && body.code !== '') return body.code;
  return 'upstream_error';
}

// ---------------------------------------------------------------------------
// Provider answers -> client wires (JSON and synthesized streams).
// ---------------------------------------------------------------------------

/** A provider answer as an OpenAI Chat Completions object, with the upstream's stop reason and tool calls. */
export function openAIAnswerFrom(answer: ProviderAnswer, model: string): Json {
  const message: Json = { role: 'assistant', content: answer.text };
  if (answer.toolCalls.length > 0) {
    message.tool_calls = answer.toolCalls.map((call) => ({
      id: call.id,
      type: 'function',
      function: { name: call.name, arguments: call.arguments },
    }));
  }
  return {
    id: 'chatcmpl-aggregate',
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, message, finish_reason: answer.finish, logprobs: null }],
    usage: { prompt_tokens: answer.input, completion_tokens: answer.output, total_tokens: answer.input + answer.output },
  };
}

/** A provider answer as an Anthropic Messages object, with the stop reason and tool_use blocks. */
export function anthropicAnswerFrom(answer: ProviderAnswer, model: string): Json {
  const content: Json[] = [];
  if (answer.text !== '') content.push({ type: 'text', text: answer.text });
  for (const call of answer.toolCalls) {
    content.push({ type: 'tool_use', id: call.id !== '' ? call.id : 'toolu_unknown', name: call.name, input: argumentsOf(call.arguments) });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  return {
    id: 'msg_aggregate',
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: stopReasonOf(answer.finish),
    stop_sequence: null,
    usage: { input_tokens: answer.input, output_tokens: answer.output },
  };
}

/**
 * A provider answer as an OpenAI event stream: a role chunk, the text in pieces, one chunk per
 * tool call, one finish chunk, one usage chunk, then [DONE]. No delta follows the finish chunk,
 * and nothing follows [DONE].
 */
export function openAIStreamFrom(answer: ProviderAnswer, model: string): string {
  const chunk = (delta: Json, finish: string | null): Json => ({
    id: 'chatcmpl-aggregate',
    object: 'chat.completion.chunk',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, delta, finish_reason: finish }],
  });
  const frames: string[] = [sseEvent(null, chunk({ role: 'assistant', content: '' }, null))];
  for (const piece of splitText(answer.text)) frames.push(sseEvent(null, chunk({ content: piece }, null)));
  answer.toolCalls.forEach((call, index) => {
    frames.push(
      sseEvent(
        null,
        chunk({ tool_calls: [{ index, id: call.id, type: 'function', function: { name: call.name, arguments: call.arguments } }] }, null),
      ),
    );
  });
  frames.push(sseEvent(null, chunk({}, answer.finish)));
  frames.push(
    sseEvent(null, {
      id: 'chatcmpl-aggregate',
      object: 'chat.completion.chunk',
      created: FIXED_OPENAI_CREATED,
      model,
      choices: [],
      usage: { prompt_tokens: answer.input, completion_tokens: answer.output, total_tokens: answer.input + answer.output },
    }),
  );
  frames.push('data: [DONE]\n\n');
  return frames.join('');
}

/**
 * A provider answer as an Anthropic event stream: message_start, one text block opened, filled
 * and closed (or an empty one when the answer has no text), one tool_use block per tool call,
 * message_delta with the stop reason and the output tokens, then message_stop.
 */
export function anthropicStreamFrom(answer: ProviderAnswer, model: string): string {
  const frames: string[] = [
    sseEvent('message_start', {
      type: 'message_start',
      message: {
        id: 'msg_aggregate',
        type: 'message',
        role: 'assistant',
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: answer.input, output_tokens: 0 },
      },
    }),
  ];
  let nextIndex = 0;
  if (answer.text !== '' || answer.toolCalls.length === 0) {
    frames.push(sseEvent('content_block_start', { type: 'content_block_start', index: nextIndex, content_block: { type: 'text', text: '' } }));
    for (const piece of splitText(answer.text)) {
      frames.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: nextIndex, delta: { type: 'text_delta', text: piece } }));
    }
    frames.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: nextIndex }));
    nextIndex += 1;
  }
  for (const call of answer.toolCalls) {
    const index = nextIndex;
    nextIndex += 1;
    frames.push(
      sseEvent('content_block_start', {
        type: 'content_block_start',
        index,
        content_block: { type: 'tool_use', id: call.id !== '' ? call.id : `toolu_${index}`, name: call.name, input: {} },
      }),
      sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'input_json_delta', partial_json: call.arguments } }),
      sseEvent('content_block_stop', { type: 'content_block_stop', index }),
    );
  }
  frames.push(
    sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: stopReasonOf(answer.finish), stop_sequence: null }, usage: { output_tokens: answer.output } }),
    sseEvent('message_stop', { type: 'message_stop' }),
  );
  return frames.join('');
}
