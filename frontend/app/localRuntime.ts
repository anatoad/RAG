import { ChatModelAdapter } from "@assistant-ui/react";

export const ModelAdapter: ChatModelAdapter = {
  async run({ messages, abortSignal }) {
    // TODO replace with your own API
    const result = await fetch("http://localhost:5000/api/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      // forward the messages in the chat to the API
      body: JSON.stringify({
        messages,
      }),
      // if the user hits the "cancel" button or escape keyboard key, cancel the request
      signal: abortSignal,
    });

    const data = await result.json();
    return {
      content: [
        {
          type: "text",
          text: data.text,
        },
      ],
    };
  },
};
