//! Model-agnostic LLM providers using Node 20+ built-in `fetch` (zero external deps).
//! Supports OpenAI-compatible APIs, Google Gemini, Anthropic Messages, and local Ollama.

/**
 * Extracts a JSON object from a model reply (handling markdown code fences).
 */
export function extractJson(text) {
  const raw = String(text || "").trim();
  if (!raw) {
    throw new Error("provider returned an empty response");
  }
  const fence = raw.match(/```(?:json)?\s*([\s\S]*?)```/i);
  const body = fence ? fence[1].trim() : raw;
  try {
    return JSON.parse(body);
  } catch {
    const start = body.indexOf("{");
    const end = body.lastIndexOf("}");
    if (start >= 0 && end > start) {
      return JSON.parse(body.slice(start, end + 1));
    }
    throw new Error(`provider reply did not contain valid JSON: ${body.slice(0, 240)}`);
  }
}

export function createProvider({
  provider = "openai",
  model,
  apiKey,
  baseUrl,
  temperature = 0.1,
} = {}) {
  const name = String(provider).toLowerCase();

  if (name === "openai") {
    const key = apiKey || process.env.OPENAI_API_KEY;
    const url = (baseUrl || process.env.OPENAI_BASE_URL || "https://api.openai.com/v1").replace(/\/$/, "");
    const modelId = model || process.env.OPENAI_MODEL || "gpt-4o-mini";
    return {
      name: `openai:${modelId}`,
      async complete({ system, messages }) {
        if (!key && url.startsWith("https://api.openai.com")) {
          throw new Error("OPENAI_API_KEY is required for OpenAI provider");
        }
        const apiMessages = [
          { role: "system", content: system },
          ...messages.map((m) => ({
            role: m.role,
            content: m.images?.length
              ? [
                  { type: "text", text: m.content },
                  ...m.images.map((png) => ({
                    type: "image_url",
                    image_url: { url: `data:image/png;base64,${png}` },
                  })),
                ]
              : m.content,
          })),
        ];
        const headers = { "content-type": "application/json" };
        if (key) headers.authorization = `Bearer ${key}`;
        const res = await fetch(`${url}/chat/completions`, {
          method: "POST",
          headers,
          body: JSON.stringify({
            model: modelId,
            temperature,
            response_format: { type: "json_object" },
            messages: apiMessages,
          }),
        });
        if (!res.ok) {
          throw new Error(`OpenAI API ${res.status}: ${await res.text()}`);
        }
        const data = await res.json();
        return extractJson(data.choices?.[0]?.message?.content);
      },
    };
  }

  if (name === "gemini") {
    const key = apiKey || process.env.GEMINI_API_KEY;
    const url = (baseUrl || "https://generativelanguage.googleapis.com/v1beta").replace(/\/$/, "");
    const modelId = model || process.env.GEMINI_MODEL || "gemini-2.5-flash";
    return {
      name: `gemini:${modelId}`,
      async complete({ system, messages }) {
        if (!key) {
          throw new Error("GEMINI_API_KEY is required for Gemini provider");
        }
        const contents = messages.map((m) => {
          const parts = [{ text: m.content }];
          for (const png of m.images || []) {
            parts.push({ inline_data: { mime_type: "image/png", data: png } });
          }
          return { role: m.role === "assistant" ? "model" : "user", parts };
        });
        const res = await fetch(
          `${url}/models/${encodeURIComponent(modelId)}:generateContent?key=${encodeURIComponent(key)}`,
          {
            method: "POST",
            headers: { "content-type": "application/json" },
            body: JSON.stringify({
              system_instruction: { parts: [{ text: system }] },
              contents,
              generationConfig: {
                temperature,
                responseMimeType: "application/json",
              },
            }),
          },
        );
        if (!res.ok) {
          throw new Error(`Gemini API ${res.status}: ${await res.text()}`);
        }
        const data = await res.json();
        const text = data.candidates?.[0]?.content?.parts?.map((p) => p.text || "").join("") || "";
        return extractJson(text);
      },
    };
  }

  if (name === "anthropic") {
    const key = apiKey || process.env.ANTHROPIC_API_KEY;
    const url = (baseUrl || "https://api.anthropic.com/v1").replace(/\/$/, "");
    const modelId = model || process.env.ANTHROPIC_MODEL || "claude-sonnet-4-20250514";
    return {
      name: `anthropic:${modelId}`,
      async complete({ system, messages }) {
        if (!key) {
          throw new Error("ANTHROPIC_API_KEY is required for Anthropic provider");
        }
        const apiMessages = messages.map((m) => ({
          role: m.role,
          content: m.images?.length
            ? [
                { type: "text", text: m.content },
                ...m.images.map((png) => ({
                  type: "image",
                  source: { type: "base64", media_type: "image/png", data: png },
                })),
              ]
            : m.content,
        }));
        const res = await fetch(`${url}/messages`, {
          method: "POST",
          headers: {
            "content-type": "application/json",
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
          },
          body: JSON.stringify({
            model: modelId,
            max_tokens: 4096,
            temperature,
            system,
            messages: apiMessages,
          }),
        });
        if (!res.ok) {
          throw new Error(`Anthropic API ${res.status}: ${await res.text()}`);
        }
        const data = await res.json();
        const text = data.content?.filter((b) => b.type === "text").map((b) => b.text).join("") || "";
        return extractJson(text);
      },
    };
  }

  if (name === "ollama") {
    const url = (baseUrl || process.env.OLLAMA_HOST || "http://127.0.0.1:11434").replace(/\/$/, "");
    const modelId = model || process.env.OLLAMA_MODEL || "qwen2.5-coder:14b";
    return {
      name: `ollama:${modelId}`,
      async complete({ system, messages }) {
        const res = await fetch(`${url}/api/chat`, {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({
            model: modelId,
            stream: false,
            format: "json",
            options: { temperature },
            messages: [
              { role: "system", content: system },
              ...messages.map((m) => ({
                role: m.role,
                content: m.content,
                ...(m.images?.length ? { images: m.images } : {}),
              })),
            ],
          }),
        });
        if (!res.ok) {
          throw new Error(`Ollama API ${res.status}: ${await res.text()}`);
        }
        const data = await res.json();
        return extractJson(data.message?.content);
      },
    };
  }

  throw new Error(`unknown provider "${provider}"; use openai, gemini, anthropic, or ollama`);
}
