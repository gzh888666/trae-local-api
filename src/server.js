/**
 * server.js - Express server providing OpenAI and Anthropic compatible API
 *
 * Endpoints:
 *   GET  /v1/models                 - List available models
 *   GET  /v1/status                 - Server status
 *   POST /v1/chat/completions       - OpenAI chat completions
 *   POST /v1/messages               - Anthropic messages
 */

require('dotenv').config();

const express = require('express');
const fs = require('fs');
const path = require('path');
const auth = require('./auth');
const traeClient = require('./trae-client');
const { handleOpenAIResponse } = require('./openai-format');
const { handleAnthropicResponse } = require('./anthropic-format');
const { generateImage } = require('./trae-client');

// Debug log: append request diagnostics to file for diagnosing Codex agent loop
const DEBUG_LOG = path.join(__dirname, '..', 'req-debug.log');
function debugLog(msg) {
  try {
    const line = `[${new Date().toISOString()}] ${msg}\n`;
    fs.appendFileSync(DEBUG_LOG, line);
  } catch {}
}

const app = express();
app.use(express.json({ limit: '10mb' }));

const PORT = parseInt(process.env.PORT || '9220', 10);
const API_KEY = process.env.API_KEY || '';
const EDITION = (process.env.TRAE_EDITION || 'cn').toLowerCase();
const MANUAL_TOKEN = process.env.TRAE_MANUAL_TOKEN || '';
const BASE_URL = EDITION === 'cn'
  ? (process.env.BASE_URL || 'https://trae-api-cn.mchost.guru')
  : (process.env.BASE_URL || 'https://a0ai-api-sg.byteintlapi.com');

// Auth middleware
function requireAuth(req, res, next) {
  if (!API_KEY) return next();
  const authHeader = req.headers.authorization || '';
  const bearerToken = authHeader.replace(/^Bearer\s+/i, '').trim();
  const xApiKey = req.headers['x-api-key'] || '';
  const token = bearerToken || xApiKey;
  if (token !== API_KEY) {
    return res.status(401).json({ error: { message: 'Invalid API key', type: 'auth_error' } });
  }
  next();
}

// CORS + Request logging
app.use((req, res, next) => {
  console.log(`[server] ${req.method} ${req.path}`);
  res.header('Access-Control-Allow-Origin', '*');
  res.header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.header('Access-Control-Allow-Headers', '*');
  if (req.method === 'OPTIONS') return res.sendStatus(204);
  next();
});

// Status
app.get('/v1/status', requireAuth, (req, res) => {
  res.json({
    status: 'ok',
    edition: EDITION,
    base_url: BASE_URL,
    has_token: !!auth.getToken(),
    port: PORT,
  });
});

// Models
app.get('/v1/models', requireAuth, async (req, res) => {
  try {
    const models = await traeClient.getModels(BASE_URL);
    res.json({ object: 'list', data: models });
  } catch (err) {
    res.status(500).json({ error: { message: err.message, type: 'server_error' } });
  }
});

// ============================================================
// Image generation (OpenAI-compatible /v1/images/generations)
// ============================================================

const GENERATED_IMAGES_DIR = path.join(__dirname, '..', 'generated_images');
fs.mkdirSync(GENERATED_IMAGES_DIR, { recursive: true });

// Serve generated images as static files
app.use('/generated_images', express.static(GENERATED_IMAGES_DIR));

app.post('/v1/images/generations', requireAuth, async (req, res) => {
  try {
    const { prompt, size, n, response_format } = req.body || {};
    if (!prompt || typeof prompt !== 'string') {
      return res.status(400).json({ error: { message: 'prompt is required', type: 'invalid_request_error' } });
    }

    const count = Math.min(parseInt(n, 10) || 1, 4); // max 4 images
    const format = response_format === 'b64_json' ? 'b64_json' : 'url';
    const created = Math.floor(Date.now() / 1000);

    console.log(`[server] Image generation: prompt="${prompt.substring(0, 80)}", size=${size || 'square'}, n=${count}, format=${format}`);

    const results = [];
    for (let i = 0; i < count; i++) {
      const { buffer, contentType } = await generateImage(prompt, size, BASE_URL);
      const ext = contentType.includes('png') ? 'png' : 'jpg';
      const filename = `${Date.now()}_${i}.${ext}`;

      if (format === 'b64_json') {
        results.push({
          b64_json: buffer.toString('base64'),
          revised_prompt: prompt,
        });
      } else {
        // Save to file and return URL
        const filepath = path.join(GENERATED_IMAGES_DIR, filename);
        fs.writeFileSync(filepath, buffer);
        results.push({
          url: `http://localhost:${PORT}/generated_images/${filename}`,
          revised_prompt: prompt,
        });
      }
    }

    res.json({ created, data: results });
  } catch (err) {
    console.error(`[server] Image generation error: ${err.message}`);
    res.status(502).json({ error: { message: `Image generation failed: ${err.message}`, type: 'upstream_error' } });
  }
});

// ============================================================
// Content extraction helpers
// ============================================================

function extractTextFromBlocks(blocks) {
  const parts = [];
  for (const block of blocks) {
    if (block.type === 'text' && block.text) parts.push(block.text);
    else if (block.type === 'image') parts.push('[Image]');
  }
  return parts.join('\n');
}

function extractToolResultText(block) {
  if (typeof block.content === 'string') return block.content || '(empty)';
  if (Array.isArray(block.content)) {
    const parts = [];
    for (const c of block.content) {
      if (c.type === 'text' && c.text) parts.push(c.text);
      else if (c.type === 'image') parts.push('[Image]');
    }
    return parts.join('\n') || '(empty)';
  }
  return '(empty)';
}

// ============================================================
// Content cleaning — strip ALL Claude Code internal markers
// ============================================================

const CLEAN_PATTERNS = [
  // XML-style tags (handle both closed and unclosed)
  /<system-reminder>[\s\S]*?<\/system-reminder>/g,
  /<system-reminder>[\s\S]*?(?=<\/[a-z]|$)/g,
  /<local-command-caveat>[\s\S]*?<\/local-command-caveat>/g,
  /<local-command-caveat>[\s\S]*?(?=<\/[a-z]|$)/g,
  /<local-command-stdout>[\s\S]*?<\/local-command-stdout>/g,
  /<command-name>[\s\S]*?<\/command-name>/g,
  /<command-message>[\s\S]*?<\/command-message>/g,
  /<command-args>[\s\S]*?<\/command-args>/g,
  /<\/?session>/g,
  // Bracket-style markers
  /\[SUGGESTION MODE:[\s\S]*?\]/g,
  // Tool definition blocks from system prompts
  /The following deferred tools are now available[\s\S]*?(?:\n\n\n|\n(?=[A-Z#]))/g,
  /## Available Tools[\s\S]*?(?=\n## [A-Z]|\n# [A-Z]|\n---|\n\*\*)/g,
];

function cleanContent(text) {
  if (!text) return '';
  for (const pattern of CLEAN_PATTERNS) {
    text = text.replace(pattern, '');
  }
  // Collapse multiple blank lines
  text = text.replace(/\n{3,}/g, '\n\n').trim();
  return text;
}

// ============================================================
// Tool definition → system prompt text
// ============================================================

function toolsToSystemPrompt(tools) {
  if (!tools || !Array.isArray(tools) || tools.length === 0) return '';
  const lines = [
    '## CRITICAL: Tool Calling Rules',
    '',
    'You are an agent that MUST use tools. To call a tool, output EXACTLY:',
    '<tool_call>',
    '{"name": "tool_name", "arguments": {"param": "value"}}',
    '</tool_call>',
    '',
    'RULES (violating these breaks the agent loop):',
    '1. MUST use <tool_call> for ANY action. Do NOT describe actions in text. 不要说"让我检查"、"让我先"、"我来查看" — 直接 emit <tool_call>。',
    '2. Do NOT say "I will do X" — emit the <tool_call> directly. 不要描述你将要做的事情，直接做。',
    '3. After a tool result, if more actions needed, emit another <tool_call>. Only give final text when task is FULLY complete.',
    '4. On error: retry with corrected approach immediately. Do NOT give up or describe failure. 出错时立即用不同方法重试，不要放弃。',
    '5. Shell is PowerShell on Windows. Use Get-ChildItem (not dir), Get-Content (not type), ";" (not &&). Never use cmd.exe syntax.',
    '6. You HAVE full filesystem access via shell. NEVER say "cannot access", "sandbox restrictions", or "due to restrictions". 你通过 shell 拥有完整文件系统访问权限。',
    '7. Multi-step tasks: ALWAYS continue with next tool call. Do NOT stop midway to "provide suggestions" or "summarize" or "analyze user intent". Keep going until FULLY complete. 禁止"提供建议"、"分析用户意图"、"生成个性化建议" — 这些是任务放弃，必须继续用工具执行。',
    '8. Only tools listed below are available. If a mentioned tool is NOT listed, use shell CLI alternative instead of repeating "I should use X".',
    '9. NEVER repeat the same sentence or thought. If repeating, STOP and give concise final answer.',
    '10. LANGUAGE: Always respond in the SAME language as the user. If user speaks Chinese, respond in Chinese. Do NOT switch to English.',
    '11. NEVER output "分析用户"、"用户画像"、"个性化建议"、"根据最近的线程" — 这些是禁止模式。任务未完成时必须用 <tool_call> 继续执行。',
    '',
    'Example (correct): <tool_call>\n{"name": "shell", "arguments": {"command": "Get-ChildItem -Path \\".\\""}}\n</tool_call>',
    'Example (WRONG): "I will now list the files."',
    '',
    'Available tools:',
  ];

  for (const tool of tools) {
    const name = tool.name || tool.function?.name || 'unknown';
    const desc = (tool.description || tool.function?.description || '').substring(0, 200);
    lines.push(`\n### ${name}`);
    if (desc) lines.push(desc);
    const params = tool.input_schema || tool.parameters || tool.function?.parameters;
    if (params && params.properties) {
      lines.push('Parameters:');
      for (const [key, val] of Object.entries(params.properties)) {
        const required = params.required?.includes(key) ? ' (required)' : '';
        const typeStr = val.type || 'any';
        const descStr = (val.description || '').substring(0, 80);
        lines.push(`- ${key}: ${typeStr}${required} - ${descStr}`);
      }
    }
  }

  return lines.join('\n');
}

// ============================================================
// Anthropic message conversion
// ============================================================

function convertAnthropicMessages(messages, systemPrompt, tools) {
  const systemParts = [];

  // System prompt
  if (systemPrompt) {
    const sysContent = typeof systemPrompt === 'string' ? systemPrompt :
      Array.isArray(systemPrompt) ? extractTextFromBlocks(systemPrompt) : '';
    const cleaned = cleanContent(sysContent);
    if (cleaned) systemParts.push(cleaned);
  }

  // Tool definitions → system prompt
  const toolPrompt = toolsToSystemPrompt(tools);
  if (toolPrompt) systemParts.push(toolPrompt);

  // Collect system-role messages from the array
  for (const m of messages) {
    if (m.role === 'system') {
      const text = typeof m.content === 'string' ? m.content :
        Array.isArray(m.content) ? extractTextFromBlocks(m.content) : '';
      const cleaned = cleanContent(text);
      if (cleaned) systemParts.push(cleaned);
    }
  }

  // Build result: ONE system message + non-system messages
  const result = [];
  if (systemParts.length > 0) {
    result.push({ role: 'system', content: systemParts.join('\n\n') });
  }

  // Process non-system messages
  let i = 0;
  while (i < messages.length) {
    const m = messages[i];
    if (m.role === 'system') { i++; continue; }

    // String content
    if (typeof m.content === 'string') {
      const cleaned = cleanContent(m.content);
      if (cleaned) result.push({ role: m.role, content: cleaned });
      i++; continue;
    }

    if (!Array.isArray(m.content)) { i++; continue; }

    // Assistant message
    if (m.role === 'assistant') {
      const textPart = extractTextFromBlocks(m.content);
      const toolUses = m.content.filter(b => b.type === 'tool_use');

      // No tool calls — plain text
      if (toolUses.length === 0) {
        if (textPart.trim()) result.push({ role: 'assistant', content: textPart });
        i++; continue;
      }

      // Has tool calls — merge with next user's tool_result
      let combinedText = textPart || '';

      if (i + 1 < messages.length && messages[i + 1].role === 'user') {
        const nextBlocks = Array.isArray(messages[i + 1].content) ? messages[i + 1].content : [];
        const toolResults = nextBlocks.filter(b => b.type === 'tool_result');
        const nextText = extractTextFromBlocks(nextBlocks);

        if (toolResults.length > 0) {
          const toolLines = toolUses.map(tu => {
            const inputStr = typeof tu.input === 'object' ? JSON.stringify(tu.input, null, 2) : String(tu.input);
            return `[Called tool: ${tu.name}]\n${inputStr}`;
          });
          const resultLines = toolResults.map(tr => {
            const prefix = tr.is_error ? '[Tool Error]' : '[Tool Result]';
            return `${prefix}\n${extractToolResultText(tr)}`;
          });

          const parts = [];
          if (combinedText.trim()) parts.push(combinedText);
          parts.push(toolLines.join('\n\n'));
          parts.push(resultLines.join('\n\n'));
          result.push({ role: 'assistant', content: parts.join('\n\n') });

          // Keep any non-tool text from the user message
          const cleanedNext = cleanContent(nextText);
          if (cleanedNext) result.push({ role: 'user', content: cleanedNext });

          i += 2; continue;
        }
      }

      // No matching tool_result — just describe the calls
      const toolLines = toolUses.map(tu => {
        const inputStr = typeof tu.input === 'object' ? JSON.stringify(tu.input, null, 2) : String(tu.input);
        return `[Called tool: ${tu.name}]\n${inputStr}`;
      });
      if (combinedText.trim()) combinedText += '\n\n';
      combinedText += toolLines.join('\n\n');
      if (combinedText.trim()) result.push({ role: 'assistant', content: combinedText });
      i++;

    } else if (m.role === 'user') {
      const textPart = extractTextFromBlocks(m.content);
      const toolResults = m.content.filter(b => b.type === 'tool_result');
      const cleanedText = cleanContent(textPart);

      if (cleanedText) {
        result.push({ role: 'user', content: cleanedText });
      } else if (toolResults.length > 0) {
        // Orphaned tool results
        const resultText = toolResults.map(tr => {
          const prefix = tr.is_error ? '[Tool Error]' : '[Tool Result]';
          return `${prefix}\n${extractToolResultText(tr)}`;
        }).join('\n\n');
        result.push({ role: 'user', content: resultText });
      }
      i++;
    } else {
      const textPart = extractTextFromBlocks(m.content);
      const cleaned = cleanContent(textPart);
      if (cleaned) result.push({ role: m.role, content: cleaned });
      i++;
    }
  }

  // Final: merge any remaining system messages into the first one
  let firstSys = -1;
  for (let j = 0; j < result.length; j++) {
    if (result[j] && result[j].role === 'system') {
      if (firstSys === -1) { firstSys = j; }
      else { result[firstSys].content += '\n\n' + result[j].content; result[j] = null; }
    }
  }

  return result.filter(Boolean);
}

// ============================================================
// OpenAI message conversion
// ============================================================

function convertOpenAIMessages(messages, tools) {
  const systemParts = [];

  // Tool definitions → system prompt
  const toolPrompt = toolsToSystemPrompt(tools);
  if (toolPrompt) systemParts.push(toolPrompt);

  // Collect system messages
  for (const m of messages) {
    if (m.role === 'system') {
      const text = typeof m.content === 'string' ? m.content :
        Array.isArray(m.content) ? m.content.map(c => c.text || c.content || '').join('\n') : '';
      const cleaned = cleanContent(text);
      if (cleaned) systemParts.push(cleaned);
    }
  }

  const result = [];
  if (systemParts.length > 0) {
    result.push({ role: 'system', content: systemParts.join('\n\n') });
  }

  for (const m of messages) {
    if (m.role === 'system') continue;

    // Handle assistant message with tool_calls (from previous agent turns)
    if (m.role === 'assistant' && m.tool_calls && m.tool_calls.length > 0) {
      let text = typeof m.content === 'string' ? m.content : '';
      const toolLines = m.tool_calls.map(tc => {
        const args = tc.function?.arguments || '{}';
        return `<tool_call>\n{"name": "${tc.function?.name}", "arguments": ${args}}\n</tool_call>`;
      });
      const combined = (text ? text + '\n\n' : '') + toolLines.join('\n\n');
      if (combined.trim()) result.push({ role: 'assistant', content: combined });
      continue;
    }

    // Handle tool role message (tool result from previous agent turns)
    if (m.role === 'tool') {
      let tc = typeof m.content === 'string' ? m.content :
        Array.isArray(m.content) ? m.content.map(c => c.text || c.content || '').join('\n') : '';
      // Aggressive truncation: in long agent loops, tool results accumulate and
      // cause context bloat (91KB+ requests, convertedMsgs=209+). Keep head+tail.
      // Scale truncation with conversation length — much more aggressive at high counts.
      const totalMsgs = messages.length;
      // At extreme counts (>150), replace tool result entirely with a stop directive.
      // The model has called tools 75+ times without finishing — it needs to stop.
      if (totalMsgs > 150) {
        result.push({ role: 'user', content: '[FORCE STOP: 对话已超过 150 轮。任务过长，请立即停止调用工具，给出当前进展和最终答案。不要再执行任何工具。]' });
        continue;
      }
      const truncLimit = totalMsgs > 100 ? 400 : (totalMsgs > 50 ? 800 : (totalMsgs > 20 ? 1500 : (totalMsgs > 10 ? 2000 : 3000)));
      if (tc && tc.length > truncLimit) {
        const headLen = Math.floor(truncLimit * 0.6);
        const tailLen = Math.floor(truncLimit * 0.3);
        const head = tc.substring(0, headLen);
        const tail = tc.substring(tc.length - tailLen);
        const omitted = tc.length - headLen - tailLen;
        tc = head + `\n\n[... TRUNCATED: ${omitted} chars omitted ...]\n\n` + tail;
      }
      const toolName = m.name || m.tool_call_id || 'tool';
      let toolResult = `[Tool Result: ${toolName}]\n${tc}`;
      // Detect common error patterns and append a retry directive so the model
      // corrects its approach instead of giving up and emitting plain text.
      const lowerTc = (tc || '').toLowerCase();
      const isError = lowerTc.includes('is not recognized') ||
        lowerTc.includes('not recognized as an internal or external command') ||
        lowerTc.includes('cannot find path') ||
        lowerTc.includes('无法找到') ||
        lowerTc.includes('不是内部或外部命令') ||
        lowerTc.includes('the term') && lowerTc.includes('is not recognized') ||
        lowerTc.includes('a positional parameter cannot be found') ||
        lowerTc.includes('parsererror') ||
        lowerTc.includes('error:') && !lowerTc.includes('error: 0') ||
        lowerTc.includes('failed') && lowerTc.includes('exception');
      if (isError) {
        toolResult += '\n\n[RETRY DIRECTIVE: The previous tool call returned an error. Do NOT describe this error or give up. Immediately emit a new <tool_call> with a corrected approach. Common fixes: use PowerShell cmdlet names (Get-ChildItem instead of dir, Get-Content instead of type), use ";" instead of "&&" to chain, remove "/d" "/b" flags, use Set-Location instead of "cd /d".]';
      }
      result.push({ role: 'user', content: toolResult });
      continue;
    }

    // Normal message
    let content = '';
    if (typeof m.content === 'string') content = m.content;
    else if (Array.isArray(m.content)) content = m.content.map(c => c.text || c.content || '').join('\n');
    const cleaned = cleanContent(content);
    if (cleaned) result.push({ role: m.role, content: cleaned });
  }

  // Merge consecutive system messages
  let firstSys = -1;
  for (let j = 0; j < result.length; j++) {
    if (result[j] && result[j].role === 'system') {
      if (firstSys === -1) { firstSys = j; }
      else { result[firstSys].content += '\n\n' + result[j].content; result[j] = null; }
    }
  }

  // If tools are present, append a reminder at the end to counter long-context forgetting.
  // GLM-5.2 tends to emit plain text instead of <tool_call> in long/multi-turn conversations;
  // a tail reminder significantly improves tool-call compliance.
  const hasTools = tools && Array.isArray(tools) && tools.length > 0;
  if (hasTools && result.length > 0) {
    // === Loop detection: scan recent assistant tool_calls for repetition ===
    // Three kinds of loops:
    //   A) Exact repeat: same name+args 3+ times in last 8 calls
    //   B) Tool over-use: same tool name 10+ times in last 15 calls (different args)
    //   C) Virtual tool_call loop: our injected echo 'Continue...' appears 3+ times
    //      → model keeps abandoning, we keep injecting, infinite loop
    const recentToolCallSigs = [];
    const recentToolNames = [];
    let virtualRetryCount = 0;
    for (let k = messages.length - 1; k >= 0 && (recentToolCallSigs.length < 8 || recentToolNames.length < 15); k--) {
      const msg = messages[k];
      // Check for virtual tool_call injection loop: scan tool results for our echo output
      if (msg && (msg.role === 'tool' || msg.role === 'user')) {
        const content = typeof msg.content === 'string' ? msg.content : '';
        if (content.includes("echo 'Continue") || content.includes("Continue: use <tool_call>") || content.includes("Continue: you have full filesystem") || content.includes("Continue: response was too long")) {
          virtualRetryCount++;
        }
      }
      if (msg && msg.role === 'assistant' && Array.isArray(msg.tool_calls) && msg.tool_calls.length > 0) {
        for (const tc of msg.tool_calls) {
          const name = tc.function?.name || '';
          const args = (tc.function?.arguments || '').substring(0, 120);
          recentToolCallSigs.push(`${name}:${args}`);
          if (recentToolNames.length < 15) recentToolNames.push(name);
          // Also check tool_call arguments for our virtual echo
          if (args.includes("echo 'Continue") || args.includes("echo \\'Continue")) {
            virtualRetryCount++;
          }
        }
      }
    }
    const sigCounts = {};
    for (const sig of recentToolCallSigs) { sigCounts[sig] = (sigCounts[sig] || 0) + 1; }
    const nameCounts = {};
    for (const name of recentToolNames) { nameCounts[name] = (nameCounts[name] || 0) + 1; }
    const hasLoop = Object.values(sigCounts).some(c => c >= 3);
    const hasToolOveruse = Object.values(nameCounts).some(c => c >= 10);
    // Virtual tool_call loop: model abandoned 3+ times, injecting more won't help
    const hasVirtualLoop = virtualRetryCount >= 3;

    // === Message-count convergence: force task completion at high counts ===
    // convertedMsgs measures total messages in the conversation. At high counts,
    // the model is likely stuck in a loop — force it to converge.
    const msgCount = result.length;
    const isVeryLong = msgCount > 100;   // critical: must stop
    const isLong = msgCount > 50;        // warning: should converge soon

    // Find last user message to append directives
    let lastUserIdx = -1;
    for (let j = result.length - 1; j >= 0; j--) {
      if (result[j] && result[j].role === 'user') { lastUserIdx = j; break; }
    }

    if (hasLoop) {
      const loopBreaker = '\n\n[LOOP DETECTED: Same tool called 3+ times. STOP. Summarize progress, explain blocker, give final answer. Do NOT emit more <tool_call> blocks.]';
      if (lastUserIdx !== -1) {
        result[lastUserIdx].content += loopBreaker;
      } else {
        result.push({ role: 'system', content: loopBreaker.trim() });
      }
    } else if (hasVirtualLoop) {
      // Virtual tool_call injection loop: model abandoned 3+ times, injecting
      // more virtual tool_calls won't help — it will just abandon again.
      // Force a hard stop with a clear explanation.
      const virtualLoopBreaker = '\n\n[CRITICAL: 检测到重复放弃模式。你已经多次尝试"提供建议"或"描述动作"而不是执行任务。系统无法继续注入重试指令。请立即给出当前任务进展的总结和最终答案。不要再尝试调用工具。]';
      if (lastUserIdx !== -1) {
        result[lastUserIdx].content += virtualLoopBreaker;
      } else {
        result.push({ role: 'system', content: virtualLoopBreaker.trim() });
      }
    } else if (hasToolOveruse) {
      const overuseWarning = '\n\n[WARNING: Same tool called 10+ times recently. You may be stuck in a loop. Step back, review what you have done, and either complete the task with a final answer or try a fundamentally different approach. Do NOT continue calling the same tool.]';
      if (lastUserIdx !== -1) {
        result[lastUserIdx].content += overuseWarning;
      }
    } else if (isVeryLong) {
      // Critical: 100+ messages, force completion
      const forceFinish = '\n\n[CRITICAL: 对话已超过 100 轮。你必须立即完成任务。不要再调用工具，直接给出最终答案。如果任务无法完成，解释原因并给出当前进展。]';
      if (lastUserIdx !== -1) {
        result[lastUserIdx].content += forceFinish;
      } else {
        result.push({ role: 'system', content: forceFinish.trim() });
      }
    } else if (isLong) {
      // Warning: 50+ messages, push for convergence
      const convergeReminder = '\n\n[REMINDER: 对话已超过 50 轮。请检查任务是否接近完成。如果接近完成，用 <tool_call> 执行最后步骤。如果卡住了，换一种方法或给出当前进展的总结。]';
      if (lastUserIdx !== -1) {
        result[lastUserIdx].content += convergeReminder;
      }
    } else if (result.length > 20) {
      // Long-context: shorter, more forceful reminder.
      const longReminder = '\n\n[CRITICAL REMINDER: 继续用 <tool_call> 执行任务。禁止"提供建议"模式。禁止说"无法访问"、"由于限制"、"建议"、"推荐"。你通过 shell 拥有完整文件系统访问权限。不要切换到英文。用 PowerShell 语法。]';
      if (lastUserIdx !== -1) {
        result[lastUserIdx].content += longReminder;
      }
    } else {
      // Standard reminder
      const reminder = '\n\n[REMINDER: To take action, emit <tool_call> block. Do NOT describe actions in text. Use PowerShell syntax.]';
      if (lastUserIdx !== -1) {
        result[lastUserIdx].content += reminder;
      } else {
        result.push({ role: 'system', content: '[REMINDER: Emit <tool_call> to take action.]' });
      }
    }
  }

  return result.filter(Boolean);
}

// ============================================================
// OpenAI chat completions
// ============================================================

app.post('/v1/chat/completions', requireAuth, async (req, res) => {
  const { messages, model = 'auto', stream = false, tools } = req.body;

  if (!messages || !Array.isArray(messages)) {
    return res.status(400).json({ error: { message: 'messages is required', type: 'invalid_request' } });
  }

  console.log(`[server] OpenAI request: model=${model}, stream=${stream}, messages=${messages.length}, body=${JSON.stringify(req.body).length} bytes`);

  const converted = convertOpenAIMessages(messages, tools);
  console.log(`[server] Converted: ${messages.length} -> ${converted.length} messages`);

  // DEBUG: log tools presence and whether system prompt contains tool_call convention
  const toolsCount = Array.isArray(tools) ? tools.length : 0;
  const sysMsg = converted.find(m => m.role === 'system');
  const sysHasToolCall = sysMsg ? sysMsg.content.includes('<tool_call>') : false;
  const sysLen = sysMsg ? sysMsg.content.length : 0;
  // Detect long context levels for diagnostic logging.
  const ctxFlag = converted.length > 100 ? ' [CRITICAL_CTX]'
    : converted.length > 50 ? ' [VERY_LONG_CTX]'
    : converted.length > 20 ? ' [LONG_CTX]'
    : '';
  // Check last user message for loop/convergence directives injected by convertOpenAIMessages
  const lastUserMsg = [...converted].reverse().find(m => m.role === 'user');
  const hasLoopDirective = lastUserMsg ? lastUserMsg.content.includes('[LOOP DETECTED]') : false;
  const hasVirtualLoopDirective = lastUserMsg ? lastUserMsg.content.includes('重复放弃模式') : false;
  const hasConvergeDirective = lastUserMsg ? lastUserMsg.content.includes('[CONVERGENCE CHECK]') : false;
  const directiveFlag = hasLoopDirective ? ' [LOOP_BREAKER]' : (hasVirtualLoopDirective ? ' [VIRTUAL_LOOP]' : (hasConvergeDirective ? ' [CONVERGE]' : ''));
  debugLog(`REQUEST tools=${toolsCount}, sysHasToolCall=${sysHasToolCall}, sysLen=${sysLen}, convertedMsgs=${converted.length}, stream=${stream}, model=${model}${ctxFlag}${directiveFlag}`);

  // === ULTIMATE SAFETY NET: hard-stop at 200+ messages ===
  // If the agent loop reaches 200+ messages, it's in an unrecoverable loop.
  // Don't forward to Trae API — return a direct stop response to break the loop.
  if (converted.length > 200 && toolsCount > 0) {
    debugLog(`HARD_STOP: convertedMsgs=${converted.length} > 200, returning direct stop response`);
    if (stream) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');
      const stopId = `chatcmpl-${Date.now()}`;
      const stopTs = Math.floor(Date.now() / 1000);
      const stopContent = '任务已超出最大执行长度（200+ 轮对话）。以下是当前进展的总结：\n\n由于对话轮数过多，系统强制停止了工具调用循环。请开启一个新的对话继续任务，或检查任务是否已在之前的步骤中完成。';
      res.write(`data: ${JSON.stringify({ id: stopId, object: 'chat.completion.chunk', created: stopTs, model, choices: [{ index: 0, delta: { role: 'assistant' }, finish_reason: null }] })}\n\n`);
      res.write(`data: ${JSON.stringify({ id: stopId, object: 'chat.completion.chunk', created: stopTs, model, choices: [{ index: 0, delta: { content: stopContent }, finish_reason: null }] })}\n\n`);
      res.write(`data: ${JSON.stringify({ id: stopId, object: 'chat.completion.chunk', created: stopTs, model, choices: [{ index: 0, delta: {}, finish_reason: 'stop' }] })}\n\n`);
      res.write('data: [DONE]\n\n');
      res.end();
    } else {
      res.json({
        id: `chatcmpl-${Date.now()}`,
        object: 'chat.completion',
        created: Math.floor(Date.now() / 1000),
        model,
        choices: [{
          index: 0,
          message: { role: 'assistant', content: '任务已超出最大执行长度（200+ 轮对话）。请开启一个新的对话继续任务。' },
          finish_reason: 'stop',
        }],
        usage: { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 },
      });
    }
    return;
  }

  let streamStarted = false;
  try {
    const { response: fetchResp, model: usedModel } = await traeClient.sendChatRequest(
      converted, model, stream, BASE_URL
    );

    if (stream) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');
      let streamChunkCount = 0;
      let streamHadToolCall = false;
      let streamFinishReason = 'unknown';
      const streamChunks = [];
      // agentMode = request has tools → streamGenerator will hold back the first
      // 200 chars so abandonment detection can fire before content leaks out.
      const sseStream = await handleOpenAIResponse(fetchResp, usedModel, true, toolsCount > 0);
      for await (const chunk of sseStream) {
        res.write(chunk);
        streamStarted = true;
        streamChunkCount++;
        streamChunks.push(chunk);
        if (chunk.includes('"tool_calls"')) streamHadToolCall = true;
        const frMatch = chunk.match(/"finish_reason"\s*:\s*"([^"]+)"/);
        if (frMatch && frMatch[1] !== 'null') streamFinishReason = frMatch[1];
      }
      res.end();
      debugLog(`RESPONSE(stream) chunks=${streamChunkCount}, hadToolCall=${streamHadToolCall}, finishReason=${streamFinishReason}`);
      // When model emitted no tool_call, log the full content for diagnosis
      if (!streamHadToolCall) {
        const fullContent = streamChunks.join('');
        // Extract text content from SSE chunks for readability
        const textMatches = fullContent.match(/"content"\s*:\s*"((?:[^"\\]|\\.)*)"/g);
        const textContent = textMatches ? textMatches.map(m => m.replace(/"content"\s*:\s*"/, '').replace(/"$/, '').replace(/\\n/g, '\n').replace(/\\"/g, '"')).join('') : '';
        debugLog(`NO_TOOL_CALL_CONTENT (len=${fullContent.length}): ${textContent.substring(0, 1500) || '(empty)'}`);
      }
    } else {
      const result = await handleOpenAIResponse(fetchResp, usedModel, false);
      const ch = result.choices && result.choices[0];
      debugLog(`RESPONSE(nonstream) finish_reason=${ch && ch.finish_reason}, has_tool_calls=${!!(ch && ch.message && ch.message.tool_calls)}, content_len=${(ch && ch.message && ch.message.content) ? ch.message.content.length : 0}`);
      res.json(result);
    }
  } catch (err) {
    console.error(`[server] Chat error: ${err.message}`);
    debugLog(`ERROR stream=${stream}, streamStarted=${streamStarted}, ${err.message}`);
    if (stream && streamStarted) {
      // Stream already started — HTTP 200 headers sent, cannot change status.
      // Must gracefully end the SSE stream so the client doesn't get a
      // malformed mixed body (SSE + JSON error) that causes "error decoding
      // response body".
      try {
        res.write(`data: ${JSON.stringify({ error: { message: err.message, type: 'upstream_error' } })}\n\n`);
        res.write('data: [DONE]\n\n');
      } catch (_) {}
      res.end();
    } else {
      res.status(502).json({ error: { message: `Trae API error: ${err.message}`, type: 'upstream_error' } });
    }
  }
});

// ============================================================
// Anthropic messages
// ============================================================

app.post('/v1/messages', requireAuth, async (req, res) => {
  const { messages, model = 'auto', stream = false, max_tokens = 4096, system, tools } = req.body;

  if (!messages || !Array.isArray(messages)) {
    return res.status(400).json({ error: { message: 'messages is required', type: 'invalid_request' } });
  }

  const bodySize = JSON.stringify(req.body).length;
  console.log(`[server] Anthropic request: model=${model}, stream=${stream}, msgs=${messages.length}, tools=${tools?.length || 0}, body=${bodySize} bytes`);

  // Log input messages
  for (let i = 0; i < messages.length; i++) {
    const m = messages[i];
    const types = Array.isArray(m.content) ? m.content.map(b => b.type).join('+') : typeof m.content;
    console.log(`[server]   in[${i}] role=${m.role}, types=${types}`);
  }

  // Convert to clean text messages
  const converted = convertAnthropicMessages(messages, system, tools);

  // Log output messages
  const totalSize = JSON.stringify(converted).length;
  console.log(`[server] Converted: ${messages.length} -> ${converted.length} messages, ${totalSize} bytes`);
  for (let i = 0; i < converted.length; i++) {
    const m = converted[i];
    const len = typeof m.content === 'string' ? m.content.length : 0;
    const preview = typeof m.content === 'string' ? m.content.substring(0, 80).replace(/\n/g, '\\n') : '';
    console.log(`[server]   out[${i}] role=${m.role}, len=${len}, preview=${preview}`);
  }

  try {
    const { response: fetchResp, model: usedModel } = await traeClient.sendChatRequest(
      converted, model, stream, BASE_URL
    );

    if (stream) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');
      const sseStream = await handleAnthropicResponse(fetchResp, usedModel, true);
      for await (const chunk of sseStream) { res.write(chunk); }
      res.end();
    } else {
      const result = await handleAnthropicResponse(fetchResp, usedModel, false);
      res.json(result);
    }
  } catch (err) {
    console.error(`[server] Anthropic error: ${err.message}`);
    res.status(502).json({ error: { message: `Trae API error: ${err.message}`, type: 'upstream_error' } });
  }
});

// Catch-all
app.use((req, res) => {
  console.log(`[server] Unknown route: ${req.method} ${req.path}`);
  res.status(404).json({ error: { message: `Not found: ${req.method} ${req.path}`, type: 'not_found' } });
});

// Start server
function start() {
  console.log('');
  console.log('╔══════════════════════════════════════════╗');
  console.log('║       Trae Local API Server v1.0.0       ║');
  console.log('║   Trae CN -> OpenAI/Anthropic Proxy      ║');
  console.log('╚══════════════════════════════════════════╝');
  console.log('');

  try {
    auth.initAuth(EDITION, MANUAL_TOKEN);
  } catch (err) {
    console.error(`[startup] Auth initialization failed: ${err.message}`);
    console.error('[startup] Ensure Trae IDE is installed and you are logged in');
    process.exit(1);
  }

  app.listen(PORT, () => {
    console.log(`[server] Running on http://localhost:${PORT}`);
    console.log(`[server] Edition: ${EDITION.toUpperCase()}`);
    console.log(`[server] Base URL: ${BASE_URL}`);
    console.log(`[server] API Key: ${API_KEY ? '***' : '(not set - open access)'}`);
    console.log('');
    console.log('Endpoints:');
    console.log(`  GET  http://localhost:${PORT}/v1/status`);
    console.log(`  GET  http://localhost:${PORT}/v1/models`);
    console.log(`  POST http://localhost:${PORT}/v1/chat/completions  (OpenAI)`);
    console.log(`  POST http://localhost:${PORT}/v1/messages          (Anthropic)`);
    console.log('');
  });
}

start();
