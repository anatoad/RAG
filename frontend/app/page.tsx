"use client";
import { Thread } from "@/components/assistant-ui/thread";
import { AssistantRuntimeProvider, useLocalRuntime } from "@assistant-ui/react";
import { ThreadList } from "@/components/assistant-ui/thread-list";
import { ModelAdapter } from "@/app/localRuntime";
import { useEffect } from "react";

export default function Home() {
  // const runtime = useChatRuntime({ api: "/api/chat" });
  const runtime = useLocalRuntime(ModelAdapter);

  useEffect(() => {
    fetch("http://localhost:5000/api/restart");
  }, []);

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <main className="h-dvh grid grid-cols-[1fr] gap-x-2 px-4 py-4">
        {/*<ThreadList />*/}
        <Thread />
      </main>
    </AssistantRuntimeProvider>
  );
}
