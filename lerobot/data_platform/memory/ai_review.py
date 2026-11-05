"""Opt-in, evidence-recording AI review; never authenticated human approval."""

import io

from PIL import Image

from lerobot.data_platform.memory.contracts import validate_check
from lerobot.data_platform.memory.source_repair import CAMERAS

DRAFT_GUIDANCE = """
CURRENT VIEW FORMAT: the final THREE images are separate, synchronized, unmodified
224x224 camera panels, ordered main (image), left_wrist_image, right_wrist_image.
Any earlier images are historical three-panel mosaics. Camera names identify viewpoints,
NOT the arm seen in the image. Do not infer gripper handedness from camera name.
A wrist view may show a close-up of fabric while the main view shows its position near
the drum. Absence from another camera's field of view is not conflicting evidence.
Use visible state and relevant retained events; avoid unnecessary color, identity,
contact, empty-container or completed-transfer claims. Do not erase useful history or
invent a generic waiting instruction to obtain approval. If an uncertain attribute is
not needed for the next action, omit that attribute; if the action itself cannot be
grounded, keep needs_review. Review feedback is a hypothesis, never visual evidence.
GEOMETRY: a basket below the drum in the image does not prove cloth is aligned over the
basket. Fabric at/inside the drum opening must be pulled clear before lowering toward
the basket. A fabric-filled wrist view does not prove a secure grip or extraction.
If jaws/contact are occluded, describe visible fabric and its supported location, not
a fabricated grasp event. A grounded next action may secure/pull the visible fabric;
do not prescribe lowering into a basket until the supported geometry permits it.
An empty basket now does not prove that no transfer EVER occurred. Omit such negative
history unless previous memory explicitly establishes it; "not confirmed" is not
"did not occur". Prefer the observable current state when no event is established.
Do not invent a motion axis: for a front-facing drum, "pull clear of the opening"
is preferable to "vertically upward" when the required direction is uncertain.
"""

CHECK_PROMPT = """Independently review the proposed memory and next subtask using ONLY
the task, previous_memory and three current images. Images are main, left wrist, right
wrist in that order. Camera viewpoint does NOT identify the arm visible in it.
First describe what each camera actually shows, without using the proposal as evidence.
Then assess EVERY factual claim in memory and the feasibility of the next subtask.
Different fields of view and occlusion are not contradictions. One clear view can
support a claim; another must visibly contradict it to invalidate that evidence.
Distinguish image-plane height from 3D alignment: cloth near the drum is not necessarily
over the basket just because the basket appears lower in the image. Do not approve
"above the basket" or "lower into basket" unless the cloth is visibly clear of the
drum and aligned for placement. A close-up filled with fabric does not establish grasp
contact or extraction; do not invent gripper jaws hidden by fabric or outside the view.
An empty basket now is not evidence that no transfer ever occurred. "No event confirmed"
does not entail "no event occurred". Reject unsupported negative histories as well as
positive ones. Check that a proposed motion direction is supported by the visible opening.
Do not invent unseen fabric, contact, identity, counts, or basket contents. An unclear
required claim is unsupported. Explicitly visible current state may update stale prior
memory. Previous_memory supports retained history but not a currently contradicted state.
'Garment in basket' is a state, not proof of a release event, its origin or identity.
A prospective action need not already have happened: approach before grasp is possible.
Do not require proof of claims the output does not make. Check all claims actually made.
Return ONLY JSON with exactly these keys:
{"observations":[{"camera":"image","visible":"literal visible state"},
{"camera":"left_wrist_image","visible":"literal visible state"},
{"camera":"right_wrist_image","visible":"literal visible state"}],
"assessments":[{"field":"memory|subtask","claim":"exact assessed claim",
"supported":true,"basis":"current_image|previous_memory|task_and_current_image",
"cameras":["image"],"evidence":"specific supporting or contradictory evidence"}],
"supported":true,"issues":[]}.
Each field must be exactly "memory" or "subtask". Each basis must be exactly one of
"current_image", "previous_memory", "task_and_current_image"; never concatenate choices
or put camera names in basis. Camera names belong only in cameras.
Assess both fields, covering every assertion. If ANY assessment is unsupported, the
overall supported must be false and issues must explain it; otherwise issues must be [].
For prior-memory evidence cameras may be empty; visual evidence must name its camera.
No temporal metadata, earlier clip, or extra fact ledger is available. This is AI
review for a model-only experiment, never human approval or guaranteed ground truth.
"""


def camera_panels(mosaic):
    """Split existing student pixels; never add resolution or extra visual evidence."""
    with Image.open(io.BytesIO(mosaic)) as source:
        if source.size != (672, 224):
            raise ValueError("AI review requires the fixed 672x224 student mosaic")
        panels = []
        for index in range(3):
            output = io.BytesIO()
            source.crop((index * 224, 0, (index + 1) * 224, 224)).save(output, format="PNG")
            panels.append(output.getvalue())
    return panels


def validate_review(value):
    if not isinstance(value, dict) or set(value) != {"observations", "assessments", "supported", "issues"}:
        raise ValueError("Invalid AI review schema")
    observations = value["observations"]
    if not isinstance(observations, list) or len(observations) != 3:
        raise ValueError("AI review must observe all three cameras")
    for item, camera in zip(observations, CAMERAS, strict=True):
        if (
            not isinstance(item, dict)
            or set(item) != {"camera", "visible"}
            or item["camera"] != camera
            or not isinstance(item["visible"], str)
            or not item["visible"].strip()
        ):
            raise ValueError("Invalid AI camera observation")
    assessments = value["assessments"]
    if not isinstance(assessments, list) or not assessments:
        raise ValueError("AI review requires claim assessments")
    for item in assessments:
        if (
            not isinstance(item, dict)
            or set(item) != {"field", "claim", "supported", "basis", "cameras", "evidence"}
            or item["field"] not in {"memory", "subtask"}
            or type(item["supported"]) is not bool
            or item["basis"] not in {"current_image", "previous_memory", "task_and_current_image"}
            or not isinstance(item["cameras"], list)
            or any(c not in CAMERAS for c in item["cameras"])
            or any(not isinstance(item[k], str) or not item[k].strip() for k in ("claim", "evidence"))
            or (item["basis"] != "previous_memory" and not item["cameras"])
        ):
            raise ValueError("Invalid AI claim assessment")
    if {a["field"] for a in assessments} != {"memory", "subtask"}:
        raise ValueError("AI review must assess memory and subtask")
    check = validate_check({k: value[k] for k in ("supported", "issues")})
    if check["supported"] != all(a["supported"] for a in assessments) or (
        check["supported"] and check["issues"]
    ):
        raise ValueError("AI review verdict contradicts its assessments")
    return check
