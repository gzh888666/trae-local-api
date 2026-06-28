/**
 * openai-format.js - Convert Trae SSE events to OpenAI-compatible format
 *
 * Supports native OpenAI function-calling: the upstream Trae API does not
 * support tool_calls natively, so tools are injected as a system-prompt
 * convention asking the model to emit <tool_call>{...}</tool_call> blocks.
 * This module parses those blocks back into structured OpenAI tool_calls so
 * that agent clients (e.g. Codex) can drive their tool loop.
 */
const { v4: uuidv4 } = require('uuid');

const TOOL_CALL_MARKER = '<tool_call>';
const TOOL_CALL_END = '</tool_call>';

/**
 * Extract all <tool_call>{...}</tool_call> blocks from text.
 * @returns {{ cleaned: string, toolCalls: Array }}
 */
function extractToolCalls(text) {
  const toolCalls = [];
  const re = /<tool_call>\s*([\s\S]*?)\s*<\/tool_call>/g;
  let m;
  while ((m = re.exec(text)) !== null) {
    const parsed = safeJSON(m[1]);
    if (parsed && parsed.name) {
      toolCalls.push({
        id: `call_${uuidv4()}`,
        type: 'function',
        function: {
          name: parsed.name,
          arguments: typeof parsed.arguments === 'string'
            ? parsed.arguments
            : JSON.stringify(parsed.arguments ?? {}),
        },
      });
    }
  }
  const cleaned = text.replace(re, '').trim();
  return { cleaned, toolCalls };
}

/**
 * Length of the longest suffix of `text` that is a prefix of the marker.
 * Used to hold back a partial "<tool_cal..." tail so it is never streamed
 * as plain content.
 */
function matchedPrefixLen(text) {
  const max = Math.min(text.length, TOOL_CALL_MARKER.length);
  for (let i = max; i >= 1; i--) {
    if (text.endsWith(TOOL_CALL_MARKER.substring(0, i))) return i;
  }
  return 0;
}

/**
 * Parse Trae SSE response stream and convert to OpenAI format
 * @param {Response} fetchResponse - fetch Response object
 * @param {string} model - model name
 * @param {boolean} stream - streaming mode
 * @param {boolean} agentMode - true when request has tools (agent loop).
 *   In agent mode, hold back the first 200 chars to give abandonment/repetition
 *   detection a chance to fire BEFORE content is streamed out — once a
 *   "giving up" phrase is detected, we can emit just an ERROR + stop instead
 *   of leaking a long useless response that breaks CC-Switch (HTTP 400).
 */
async function handleOpenAIResponse(fetchResponse, model, stream, agentMode = false) {
  if (!stream) {
    return await collectNonStreaming(fetchResponse, model);
  }
  return streamGenerator(fetchResponse, model, agentMode);
}

/**
 * Collect full response for non-streaming mode
 */
async function collectNonStreaming(fetchResponse, model) {
  const text = await fetchResponse.text();
  const events = parseSSE(text);

  let fullContent = '';
  let finishReason = 'stop';
  let reasoningContent = '';

  for (const { event, data } of events) {
    if (event === 'output') {
      const parsed = safeJSON(data);
      if (parsed) {
        if (parsed.reasoning_content) reasoningContent += parsed.reasoning_content;
        if (parsed.response) fullContent += parsed.response;
        if (parsed.finish_reason) finishReason = parsed.finish_reason;
      }
    } else if (event === 'done') {
      const parsed = safeJSON(data);
      if (parsed && parsed.finish_reason) finishReason = parsed.finish_reason;
    }
  }

  let content = '';
  if (reasoningContent) {
    content += `<think>\n${reasoningContent}\n</think>\n\n`;
  }
  content += fullContent;

  // Repetition detection for non-streaming mode
  if (content.length > 3000) {
    const tail = content.slice(-80);
    const occurrences = content.split(tail).length - 1;
    if (occurrences >= 3) {
      content = content.substring(0, 4000) + '\n\n[Response truncated: repetition detected. Please try a different approach or rephrase.]';
    }
  }

  const { cleaned, toolCalls } = extractToolCalls(content);

  if (toolCalls.length > 0) {
    return {
      id: `chatcmpl-${uuidv4()}`,
      object: 'chat.completion',
      created: Math.floor(Date.now() / 1000),
      model,
      choices: [{
        index: 0,
        message: {
          role: 'assistant',
          content: cleaned || null,
          tool_calls: toolCalls,
        },
        finish_reason: 'tool_calls',
      }],
      usage: { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 },
    };
  }

  return {
    id: `chatcmpl-${uuidv4()}`,
    object: 'chat.completion',
    created: Math.floor(Date.now() / 1000),
    model,
    choices: [{
      index: 0,
      message: { role: 'assistant', content },
      finish_reason: finishReason,
    }],
    usage: { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 },
  };
}

/**
 * Generator for streaming mode.
 *
 * Streams content/reasoning incrementally. When a <tool_call> marker is
 * detected, switches to buffering: complete <tool_call>...</tool_call>
 * blocks are parsed and emitted as structured tool_calls deltas (never as
 * plain content). This preserves low latency for normal text responses
 * while correctly supporting function-calling for agent clients.
 *
 * In agent mode (tools present), holds back the first HOLD_THRESHOLD chars
 * so abandonment/repetition detection can fire before content reaches the
 * client — prevents "giving up" responses from leaking through and causing
 * CC-Switch HTTP 400 errors.
 */
async function* streamGenerator(fetchResponse, model, agentMode = false) {
  // In agent mode, hold back first 250 chars. Abandonment phrases like
  // "根据已有的信息" / "由于无法访问" / "让我检查" typically appear in the
  // first 50 chars, and long-text detection triggers at 150 chars.
  // 250 gives enough buffer for detection to fire before any content leaks.
  const HOLD_THRESHOLD = agentMode ? 250 : 0;
  const reader = fetchResponse.body.getReader();
  const decoder = new TextDecoder('utf-8', { fatal: false, ignoreBOM: true });
  let buffer = '';
  let responseContent = '';
  let thinkStarted = false;
  let thinkEnded = false;
  let finishReason = 'stop';
  let emitted = 0;          // chars of responseContent already streamed as content
  let toolMode = false;     // true once <tool_call> detected
  let toolBuf = '';         // accumulates text from first <tool_call> onward
  const toolCalls = [];

  const id = `chatcmpl-${uuidv4()}`;
  const ts = Math.floor(Date.now() / 1000);
  const mk = (delta, fr) => `data: ${JSON.stringify({
    id, object: 'chat.completion.chunk', created: ts, model,
    choices: [{ index: 0, delta, finish_reason: fr }],
  })}\n\n`;

  // Parse complete <tool_call>...</tool_call> blocks currently in toolBuf,
  // emit each as a tool_calls delta, and strip consumed blocks from toolBuf.
  const drainToolBuf = function* () {
    const re = /<tool_call>\s*([\s\S]*?)\s*<\/tool_call>/g;
    let m;
    let lastEnd = 0;
    while ((m = re.exec(toolBuf)) !== null) {
      const parsed = safeJSON(m[1]);
      if (parsed && parsed.name) {
        const args = parsed.arguments;
        const tc = {
          index: toolCalls.length,
          id: `call_${uuidv4()}`,
          type: 'function',
          function: {
            name: parsed.name,
            arguments: typeof args === 'string' ? args : JSON.stringify(args ?? {}),
          },
        };
        toolCalls.push(tc);
        yield mk({ tool_calls: [tc] }, null);
      }
      lastEnd = re.lastIndex;
    }
    if (lastEnd > 0) {
      toolBuf = toolBuf.substring(lastEnd);
    }
  };

  // Emit initial role chunk — standard OpenAI streaming format requires the
  // first chunk to carry delta.role="assistant". Without it, some clients
  // (e.g. CC-Switch converting chat->responses) fail to initialize the
  // response object and report HTTP 400.
  yield mk({ role: 'assistant' }, null);

  // Idle timeout: if no chunk received within 60s during streaming, abort.
  // Trae backend may stall mid-stream on long tasks; without this, the agent
  // loop hangs indefinitely with no error.
  const IDLE_TIMEOUT_MS = 60 * 1000;
  let idleTimer = null;
  const resetIdleTimer = () => {
    if (idleTimer) clearTimeout(idleTimer);
    idleTimer = setTimeout(() => {
      try { reader.cancel().catch(() => {}); } catch (_) {}
    }, IDLE_TIMEOUT_MS);
  };
  resetIdleTimer();

  try {
  while (true) {
    const { done, value } = await reader.read();
    resetIdleTimer();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop() || '';
    let currentEvent = null;

    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed) continue;
      if (trimmed.startsWith('event:')) {
        currentEvent = trimmed.substring(6).trim();
        continue;
      }
      if (!trimmed.startsWith('data:') || !currentEvent) continue;
      const data = trimmed.substring(5).trim();
      const parsed = safeJSON(data);

      if (currentEvent === 'output' && parsed) {
        // reasoning -> <think> stream
        if (parsed.reasoning_content) {
          if (!thinkStarted) {
            thinkStarted = true;
            yield mk({ content: '<think>\n' + parsed.reasoning_content }, null);
          } else {
            yield mk({ content: parsed.reasoning_content }, null);
          }
        }
        if (parsed.response) {
          if (thinkStarted && !thinkEnded) {
            thinkEnded = true;
            yield mk({ content: '\n</think>\n\n' }, null);
          }

          if (toolMode) {
            toolBuf += parsed.response;
            yield* drainToolBuf();
          } else {
            responseContent += parsed.response;

            // Repetition detection: if content is long and recent text repeats,
            // the model is stuck in a loop. Truncate to prevent huge useless output.
            if (responseContent.length > 2000 && !toolMode) {
              // Check multiple window sizes to catch near-repeats
              let isRepetitive = false;
              for (const winSize of [30, 50, 80]) {
                if (responseContent.length > winSize) {
                  const tail = responseContent.slice(-winSize);
                  const occurrences = responseContent.split(tail).length - 1;
                  if (occurrences >= 3) {
                    isRepetitive = true;
                    break;
                  }
                }
              }

              // Hard limit: if content exceeds 6000 chars without tool_call, force stop
              if (!isRepetitive && responseContent.length > 6000) {
                // Check if any 40-char segment appears 3+ times (catches semantic near-repeats)
                for (let s = 0; s < responseContent.length - 40; s += 200) {
                  const seg = responseContent.substring(s, s + 40);
                  if (responseContent.split(seg).length - 1 >= 3) {
                    isRepetitive = true;
                    break;
                  }
                }
                // If still not detected but content is huge, force truncate
                if (!isRepetitive && responseContent.length > 8000) {
                  isRepetitive = true;
                }
              }

              if (isRepetitive) {
                // Repetition detected — emit short error and stop.
                // In agent mode, held-back content (emitted=0) is discarded;
                // in non-agent mode, only already-streamed content is kept.
                yield mk({ content: '[Response truncated: repetition or excessive length detected. Please try a different approach or rephrase.]' }, null);
                yield mk({}, 'stop');
                yield 'data: [DONE]\n\n';
                return;
              }
            }

            // Task abandonment detection: if model outputs phrases indicating it's giving up
            // (e.g. "cannot access", "sandbox restrictions", "I'll provide a suggestion"),
            // truncate immediately — these phrases indicate the model stopped using tools.
            // Covers BOTH English and Chinese — model often switches to Chinese when giving up.
            if (responseContent.length > 50 && !toolMode) {
              const lowerContent = responseContent.toLowerCase();
              const abandonmentPhrases = [
                // English — explicit give-up phrases
                'cannot access the filesystem',
                'sandbox restrictions',
                'i cannot access',
                'due to sandbox',
                'due to restrictions',
                "i'll provide a suggestion",
                'i will provide a suggestion',
                'i am unable to access',
                'i\'m unable to access',
                'cannot access the file system',
                'i do not have access',
                'i don\'t have access',
                'i can suggest',
                'unable to perform',
                'cannot perform this action',
                'without access to',
                'cannot directly access',
                'i don\'t have direct access',
                // Chinese — 中文放弃话术（模型常用）
                '无法访问文件系统',
                '无法访问文件',
                '无法访问项目',
                '无法直接访问',
                '无法获取文件',
                '由于沙盒限制',
                '由于沙箱限制',
                '由于权限限制',
                '由于访问限制',
                '由于无法访问',
                '没有访问权限',
                '没有文件系统访问',
                '无法检查项目结构',
                '无法查看项目',
                '无法读取文件',
                '我将提供建议',
                '我提供建议',
                '为您提供建议',
                '给出一些建议',
                '提供个性化建议',
                '个性化的建议',
                '针对性的建议',
                '基于推断',
                '根据推断',
                '如果用户需要',
                '如果您需要更',
                '需要能够查看',
                '需要访问文件',
                // 新增：分析/建议模式话术（从实际日志提取）
                '根据最近的',
                '我可以分析',
                '分析用户',
                '用户画像',
                '工作模式',
                '线程分析',
                '上下文分析',
                '由于shell',
                'shell工具暂时',
                'shell命令执行遇到',
                '暂时不可用',
                '基于目前掌握',
                '基于我目前',
                '基于已有的信息',
                '根据已有',
                '从最近的',
                '可以看出',
                '这表明',
                '用户正在',
                '用户可能',
                '用户意图',
              ];
              let abandoned = false;
              for (const phrase of abandonmentPhrases) {
                if (lowerContent.includes(phrase) || responseContent.includes(phrase)) {
                  abandoned = true;
                  break;
                }
              }
              if (abandoned) {
                // Abandonment detected — inject a virtual tool_call to keep
                // the agent loop alive. finish_reason=stop would make Codex
                // think the task is complete and halt the loop.
                // In agent mode, held-back content (emitted=0) is discarded.
                const virtualTc = {
                  index: 0,
                  id: `call_${uuidv4()}`,
                  type: 'function',
                  function: {
                    name: 'shell',
                    arguments: JSON.stringify({ command: "echo 'Continue: you have full filesystem access via shell. Do NOT say cannot-access or provide-suggestions. Use <tool_call> to execute the next step.'" }),
                  },
                };
                toolCalls.push(virtualTc);
                yield mk({ tool_calls: [virtualTc] }, null);
                yield mk({}, 'tool_calls');
                yield 'data: [DONE]\n\n';
                return;
              }
            }

            // Action-description detection (agent mode only, short threshold):
            // If model says "让我..." / "let me..." etc. within the first 30 chars,
            // it's describing actions instead of executing. Inject a virtual
            // tool_call to keep the agent loop alive (finish_reason=stop would
            // make Codex think the task is done).
            const actionThreshold = agentMode ? 30 : 500;
            if (responseContent.length > actionThreshold && !toolMode) {
              const lowerContent = responseContent.toLowerCase();
              const actionIndicators = [
                'let me', 'i will', 'i\'ll', 'let\'s', 'i need to',
                '让我', '我来', '我需要', '我将', '接下来',
                // Suggestion-mode indicators (both EN and ZH)
                'i suggest', 'i recommend', 'i can help',
                '建议', '推荐', '我可以帮',
                // 描述动作的话术
                '让我检查', '让我先', '让我查看', '让我确认', '让我搜索', '让我尝试',
                '我来检查', '我来查看', '我先检查', '我可以使用', '我会用',
              ];
              let hasActionVerb = false;
              for (const verb of actionIndicators) {
                if (lowerContent.includes(verb) || responseContent.includes(verb)) { hasActionVerb = true; break; }
              }
              if (hasActionVerb) {
                // Inject a virtual tool_call instead of ERROR+stop.
                // This keeps Codex's agent loop alive — finish_reason=stop would
                // make Codex think the task is complete and halt the loop.
                // The echo output reminds the model to use tools.
                const virtualTc = {
                  index: 0,
                  id: `call_${uuidv4()}`,
                  type: 'function',
                  function: {
                    name: 'shell',
                    arguments: JSON.stringify({ command: "echo 'Continue: use <tool_call> to execute the next step of the task. Do NOT describe actions.'" }),
                  },
                };
                toolCalls.push(virtualTc);
                yield mk({ tool_calls: [virtualTc] }, null);
                yield mk({}, 'tool_calls');
                yield 'data: [DONE]\n\n';
                return;
              }
            }

            // Long text without tool_call: hard length limit.
            // Inject virtual tool_call to keep agent loop alive.
            const hardLimit = agentMode ? 400 : 5000;
            if (responseContent.length > hardLimit && !toolMode) {
              const virtualTc = {
                index: 0,
                id: `call_${uuidv4()}`,
                type: 'function',
                function: {
                  name: 'shell',
                  arguments: JSON.stringify({ command: "echo 'Continue: response was too long without tool_call. Execute the next step now.'" }),
                },
              };
              toolCalls.push(virtualTc);
              yield mk({ tool_calls: [virtualTc] }, null);
              yield mk({}, 'tool_calls');
              yield 'data: [DONE]\n\n';
              return;
            }

            const markerIdx = responseContent.indexOf(TOOL_CALL_MARKER, emitted);
            if (markerIdx !== -1) {
              if (markerIdx > emitted) {
                yield mk({ content: responseContent.substring(emitted, markerIdx) }, null);
              }
              emitted = markerIdx;
              toolMode = true;
              toolBuf = responseContent.substring(markerIdx);
              responseContent = responseContent.substring(0, markerIdx);
              yield* drainToolBuf();
            } else {
              // Hold back any tail that could be a partial <tool_call marker.
              const hold = matchedPrefixLen(responseContent);
              let safeLen = responseContent.length - hold;

              // In agent mode, also hold back the first HOLD_THRESHOLD chars.
              // This gives abandonment/repetition detection (which runs above
              // on the full responseContent) a chance to fire BEFORE any
              // "giving up" text is streamed to the client. Once we pass the
              // threshold, stream normally but keep the marker-prefix hold.
              if (agentMode && responseContent.length < HOLD_THRESHOLD) {
                safeLen = 0; // don't stream yet
              }

              if (safeLen > emitted) {
                yield mk({ content: responseContent.substring(emitted, safeLen) }, null);
                emitted = safeLen;
              }
            }
          }
        }
        if (parsed.finish_reason) finishReason = parsed.finish_reason;
      } else if (currentEvent === 'done' && parsed && parsed.finish_reason) {
        finishReason = parsed.finish_reason;
      }
      currentEvent = null;
    }
  }
  } catch (streamErr) {
    if (idleTimer) clearTimeout(idleTimer);
    // Trae API connection interrupted or read error.
    // Emit a finish chunk so the client gets a well-formed SSE end,
    // then let the caller's catch handle the error logging.
    yield mk({}, finishReason);
    yield 'data: [DONE]\n\n';
    throw streamErr;
  }
  if (idleTimer) clearTimeout(idleTimer);

  // Finalize
  if (toolMode) {
    yield* drainToolBuf();
    // handle an unclosed <tool_call> (stream ended before </tool_call>)
    const openIdx = toolBuf.indexOf(TOOL_CALL_MARKER);
    if (openIdx !== -1) {
      const inner = toolBuf.substring(openIdx + TOOL_CALL_MARKER.length).trim();
      const parsed = safeJSON(inner);
      if (parsed && parsed.name) {
        const args = parsed.arguments;
        const tc = {
          index: toolCalls.length,
          id: `call_${uuidv4()}`,
          type: 'function',
          function: {
            name: parsed.name,
            arguments: typeof args === 'string' ? args : JSON.stringify(args ?? {}),
          },
        };
        toolCalls.push(tc);
        yield mk({ tool_calls: [tc] }, null);
      }
    }
  } else {
    // flush any held-back content
    if (emitted < responseContent.length) {
      yield mk({ content: responseContent.substring(emitted) }, null);
    }
  }

  yield mk({}, toolCalls.length > 0 ? 'tool_calls' : finishReason);
  yield 'data: [DONE]\n\n';
}

function parseSSE(text) {
  const events = [];
  const lines = text.split('\n');
  let currentEvent = null;

  for (const line of lines) {
    const trimmed = line.trim();
    if (trimmed.startsWith('event:')) {
      currentEvent = trimmed.substring(6).trim();
    } else if (trimmed.startsWith('data:') && currentEvent) {
      events.push({
        event: currentEvent,
        data: trimmed.substring(5).trim(),
      });
      currentEvent = null;
    }
  }

  return events;
}

function safeJSON(str) {
  try { return JSON.parse(str); } catch { return null; }
}

module.exports = { handleOpenAIResponse };
