# interactive_audio_book
interactive_audio_book agent can read book. User can ask to pause and ask question about content, term. Agent can take note etc.

## How to run

```bash
uvicorn main:app --reload
```

# websocket. client react app
https://docs.vocode.dev/open-source/react-quickstart

```
import { useConversation } from "vocode";

const { status, start, stop, error, analyserNode } = useConversation({
  backendUrl: "<YOUR_BACKEND_URL>", // looks like ws://localhost:3000/conversation or wss://asdf1234.ngrok.app/conversation if using ngrok
  audioDeviceConfig: {},
});
```

```
<>
  {status === "idle" && <p>Press me to talk!</p>}
  {status == "error" && error && <p>{error.message}</p>}

  <button
    disabled={["connecting"].includes(status)}
    onClick={status === "connected" ? stop : start}
  ></button>
</>
```