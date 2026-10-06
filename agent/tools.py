"""MCP-style tool functions: the agent's callable tool surface.

Each function is a thin, documented wrapper around one capability, so the
tool schema can be generated from the docstrings. They take already-wired
collaborators (pipeline, store, actions, loop) — no globals.
"""


def perceive_frame(pipeline, frame):
    """Run the vision pipeline on a raw frame and return its result dict.

    ``pipeline``: object with ``process_frame(frame) -> frame_result``.
    ``frame``: raw frame (e.g. numpy array or bytes).
    Returns the frame result dict ``{"frame_id", "events", "jpeg_bytes",
    "pipeline_ppe", ...}`` with vision evidence attached to each event.
    """
    return pipeline.process_frame(frame)


def query_events(store, filters):
    """Query stored safety events with exact-match filters.

    ``store``: event store (DynamoClient). ``filters``: dict like
    ``{"site_id": "site-a", "type": "NO_HELMET"}`` or None for all events.
    Returns a list of event dicts.
    """
    return store.query_events(filters)


def archive_evidence(actions, jpeg_bytes, event):
    """Archive an evidence JPEG for an event and return its URI.

    ``actions``: ActionExecutor. ``jpeg_bytes``: raw JPEG bytes (may be
    None -> returns None). ``event``: the event dict the frame belongs to.
    """
    return actions.archive_evidence(jpeg_bytes, event)


def request_approval(loop, decision, event):
    """Enqueue a human approval request and return its approval id.

    ``loop``: AgentLoop (uses ``loop.actions``). ``decision``: the policy
    decision dict. ``event``: the triggering event dict.
    """
    return loop.actions.request_approval(decision, event)["approval_id"]
