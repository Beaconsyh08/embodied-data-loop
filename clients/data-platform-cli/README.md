# Data Platform CLI

A conversational terminal for your Data Platform account. The platform uses its configured Qwen model;
you do not need a model API key. This client does not install LeRobot, PyTorch or GPU libraries.

Requires Python 3.10+ and access to your own platform server.
Install from this repository using the command below. This public distribution does not
provide a hosted platform, a download bucket or a site-specific CA.

The versioned installer is a deployment template: administrators must configure their own
artifact URL, hashes, CA and server address before distributing it. The update entry must
also point to that administrator's trusted distribution. Do not use placeholder URLs.

## Install from source

The source package requires a platform server with the assistant action APIs:

```bash
pipx install ./clients/data-platform-cli
data-platform --server https://platform.example.com --username your-account
```

For another private CA, obtain the trusted certificate from that platform's administrator:

```bash
data-platform --server https://platform.example.com \
  --ca-bundle /path/to/platform-ca.crt
```

The CLI prompts for your password without echoing it, then lists accessible datasets. Passwords, session
cookies and input history stay in memory; there is no password argument, credential file or TLS bypass.
Exit logs out the session. Conversations are retained by the server under the same policy as the website.
Do not paste credentials into conversation messages.

Operation cards print complete HTTPS result/task URLs immediately. Ordinary follow-ups do not
repeat the previous card; `/history` retains each card with its original message.
Explicit navigation phrases such as `网址在哪儿`, `给我链接`, `链接打不开`, or `where is the link`
act like `/open`: refresh authorized links without a model call, new task or conversation revision.
They explain separate browser login and network requirements; opening an existing result does not
require rebuilding it. Other requests continue to the assistant normally. Markdown replies are
formatted for the terminal, and embedded links use server-approved references and full URLs.

Type a request such as `帮我准备 Viewer`. The assistant displays the target and parameters before doing
anything. Existing results and equivalent running work appear before submission. Use `/open` for authorized result/task links, `/status` to refresh, and `/confirm` for a new operation. Existing results or failed attempts require the explicit `/confirm again` choice; first inspect recovery in Jobs. Describe parameter changes in chat to prepare a revised card. Viewer
accounts can only ask for explanations. Source mutations, deletion, approval and publication remain in
their dedicated web interfaces.

| Input | Behavior |
|---|---|
| Enter / Alt+Enter | Send / insert a new line |
| Up / Down / Tab | Recall input / complete slash commands |
| `/datasets` | Choose another authorized dataset and start a new conversation |
| `/new` | Start another conversation on the selected dataset |
| `/history` | Show the restored server conversation history |
| `/chats` / `/chats next` | List accessible saved conversations, with bounded pagination |
| `/memory` | Explicit account/dataset notes: add, edit, delete, clear, enable/disable and task-update settings |
| `/updates` | Read authorized completion summaries for your confirmed assistant tasks |
| `/preferences` | Show personal preferences and editing commands; shared with the web in this environment |
| `/restore <number or conversation-id>` | Restore your own conversation, pending response and action |
| `/resume` | Recover an interrupted response using its original request ID |
| `/confirm` | Confirm the displayed new operation; retries reuse the same run |
| `/status` | Refresh availability or run status, including before confirmation |
| `/open` | Print fresh authorized result/task links, or website links when no action/result is available |
| `/web` | Print this environment's console and saved conversation links |
| `/confirm again` | Explicitly start a new run after reviewing an existing result or failed attempt |
| `/exit`, `/quit`, `exit`, `quit`, or Ctrl+D on an empty input line | Log out and exit, even with a pending response |

Ctrl+C clears input or interrupts waiting; it does not exit or cancel work already accepted by the server.
To leave after interrupting a response, type `/exit`, `exit`, or `quit`.
If the connection drops, use `/resume`. After restarting the CLI, use the displayed conversation ID with
`/restore`. If a request never reached the server there is nothing to restore. Starting a new conversation
does not cancel the old server request. A new exchange invalidates any older, unconfirmed action card.
Already submitted work can also be followed in the website's Jobs page.

This is an API client for data operations. It cannot read your local files or run shell commands. HTTPS
certificate validation, platform roles, object scopes, input identity checks and CSRF remain enabled.
Replies use terminal Markdown formatting with terminal control sequences removed. Embedded links
are expanded only when they match server-authorized references.

## Personal preferences and history

`/chats` lists your own currently authorized conversations; `/restore 1` opens the first result.
`/history` retains its original meaning: the messages of the current conversation. Context changes
can make an old conversation unavailable; start a new chat instead. History is bounded to six exchanges
per conversation and follows the existing 30-day inactivity/request retention policy.

Preferences are explicit settings shared by web and CLI for the same account and environment:

```text
/preferences language zh
/preferences detail concise
/preferences favorites viewer.prepare,curation.quality
/preferences off
/preferences on
/preferences reset
```

Use `auto` to match message language, `balanced` or `detailed` for other response lengths, and
`/preferences favorites none` to clear favorite operators. At most five business operators can be
selected. Settings affect replies and suggestions on the next message; favorites do not add operations
to a plan. Viewer accounts may manage their own communication preferences but remain read-only for data.
The assistant does not silently infer preferences, train a model, or change operation parameters,
permissions or confirmation requirements. The web Preferences panel can edit or clear the same settings.
Concurrent updates use revisions; reload settings if another client changed them.

Candidate CLI 0.3.34 accompanies the FLORA idle-GPU executor change. Authentication, confirmation and
`/open` contracts are unchanged; publish its immutable artifacts only after platform acceptance.

CLI 0.3.33 accompanies the Agent-local caption dataset selector.

CLI 0.3.32 accompanies the production training-memory Review. CLI 0.3.31 introduced the frozen
training-memory Review. Use the authenticated web Annotation/Review form to prepare and confirm dataset memory jobs. `/memory` below still
manages assistant account notes; it does not generate dataset labels or approve annotations.

## Explicit long-term memory and task updates

```text
/memory
/memory add account Explain terms with short examples
/memory add dataset This project calls a demonstration a trial
/memory edit <memory-id> Updated wording
/memory delete <memory-id>
/memory off
/memory on
/memory clear
/memory followups off
/memory followups on
/updates
```

Use IDs printed by `/memory`. Notes persist on the server until deleted, independently of short chat
history. At most 20 notes of 500 characters are stored; up to eight authorized notes are used per turn,
prioritizing the current dataset. Disabling or deleting notes affects future requests, not existing chat
messages or already sent model calls. Chat never silently saves notes. Notes do not grant permission,
select operations or bypass `/confirm`; dataset notes stop being used after identity or access changes.

New confirmed assistant tasks are followed by the server even after you exit. `/updates` retrieves factual
terminal summaries and authorized HTTPS run/result links; the CLI does not need to stay open or run a
background poller. Summaries wait for final Job/Attempt acknowledgement, require current scope, and remain
for 30 days. Disabling updates mutes pending observations; enabling again follows new confirmations.
Previously generated summaries remain available within retention after re-enabling. Manual retries hide
old summaries; inspect Jobs for retried runs. No automatic follow-on execution, retries or external messages
occur, and direct form/editor tasks and pre-upgrade work are not backfilled.

## Implementation choice

Candidate 0.3.15 retains FLORA Review compatibility and supports explicit memory, Updates, and the server's optional
background parameter/workflow and observed recovery experience. Reuse remains an explicit, confirmed action;
current rules and requested parameters take priority. It requires no Hindsight/mem0 SDK or personal model key. Experience never
confirms a task or changes CLI parameter/permission checks. Publication follows platform acceptance.

[prompt-toolkit](https://python-prompt-toolkit.readthedocs.io/en/stable/pages/asking_for_input.html)
provides terminal editing, in-memory history, completion and key bindings. Rich provides waiting feedback;
Requests uses the platform's existing HTTPS cookie session.
[Qwen-Agent](https://github.com/QwenLM/Qwen-Agent) supports model tools and agents, but is not a dependency:
the platform already owns tool authorization, conversation state and confirmation. Keeping that logic on
the server gives web and CLI users the same behavior.

From this directory, `uv build --wheel` produces an installable client wheel. The server and CLI are
versioned independently and released together: every platform upgrade also publishes a new compatible
CLI version and advances `latest/install.py` after verification, including browser-only platform changes.
The upgrade is incomplete until both platform and CLI acceptance finish. An older server without action APIs cannot confirm operations from this client.

CLI 0.3.17 displays the shared strategy version when the server supplies one. This is planning provenance,
not execution approval; `/confirm` and existing result links are unchanged. Versioned publication and
advancing the stable latest installer require the normal platform compatibility acceptance.

External Codex/Claude Code access: [MCP setup and credential management](../../docs/data_platform_mcp.md).

CLI 0.3.20 accompanies the aligned Temporal Caption review timeline. Authenticated result links,
read-only access and confirmation behavior are unchanged; publish the compatible installer only
after platform acceptance, then advance the stable latest installer.

CLI 0.3.20 updates the installer default to the currently verified platform origin. CLI 0.3.19
versioned objects remain immutable; it was not promoted to the stable latest installer.

CLI 0.3.21 accompanies the prominent Overall and Window score panel. Authentication, result links
and execution confirmation remain compatible; publication follows platform acceptance.

CLI 0.3.22 accompanies the viewport-based Temporal Caption Review layout, with unchanged authentication,
result links and execution confirmation. Publication follows platform acceptance.

CLI 0.3.23 accompanies Review playback shortcuts and automatic caption following. Authentication,
result links and execution confirmation remain compatible; publication follows platform acceptance.


### Learning feedback

`/learning` shows the environment collection state and your independent chat/experience consent.
`/feedback turn|run|job|pipeline <id> resolved|unresolved|modify [reason]` records your assessment using the
server revision. Optional reasons include `wrong_operator`, `wrong_order`, `wrong_input`, `wrong_parameters`,
`missing_review`, `unclear`, `execution_failure`, `infrastructure`, `other`, or `none`.
This command never confirms or retries a task. Manage consent, material withdrawal and preference confirmation
on the platform's `/learning` page. No browser clicks are simulated by the CLI.

CLI 0.3.27 is compatible with dataset-level annotation jobs in Server V0.3.26.
Use the web Jobs view to expand video subtasks; existing task receipts and result links remain valid.

CLI 0.3.29 accompanies Server V0.3.28's native single-view FLORA support and retains
searchable annotated-dataset navigation in Temporal Caption Review. Authentication, result links, confirmation and job execution APIs
remain compatible. Publish the versioned installer/wheel/CA and verify installation before
advancing `cli/latest/install.py` during platform promotion.

CLI 0.3.35 accompanies the mixed-episode Viewer candidate. Authentication, confirmation and result links are unchanged; publication and acceptance are pending.

CLI 0.3.36 accompanies the mixed-episode FLORA candidate; authentication and confirmation contracts remain unchanged.

CLI 0.3.37 accompanies the mixed-episode FLORA pipeline and parallel release validation candidate.

CLI 0.3.38 accompanies the always-visible original instruction in Temporal Caption Review.

CLI 0.3.40 accompanies the UMI processing source candidate. Login, explicit confirmation and authorized result links remain compatible. The wheel/installer must pass installation acceptance before publication; source builds do not advance latest.


## FLORA 人工评测

`data-platform evaluation --server URL --location UUID --web` 打开人工审查入口。去掉 `--web` 查看指标；加 `--export-file pilot.json` 导出标注与结果。`--episode N --oracle-file predictions.json` 导入与冻结参考绑定的固定区间预测。仍使用平台账号登录，不保存密码。
