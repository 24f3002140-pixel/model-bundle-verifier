import hashlib
import json
import math
import re
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()

REQUIRED_FILES = (
    "README.md",
    "training_manifest.json",
    "evaluation.json",
    "inventory.json",
    "adapter_model.safetensors",
    "adapter_config.json",
)

UNSAFE_EXTENSIONS = (".bin", ".pt", ".pth", ".pkl", ".pickle")
HEX40 = re.compile(r"^[0-9a-f]{40}$")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def compact_json(obj: Any) -> bytes:
    return json.dumps(
        obj,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")


def nonempty_string(x: Any) -> bool:
    return isinstance(x, str) and len(x) > 0


def safe_positive_int(x: Any) -> bool:
    return (
        isinstance(x, int)
        and not isinstance(x, bool)
        and x > 0
        and x <= 9007199254740991
    )


def unit_float(x: Any) -> bool:
    return (
        isinstance(x, (int, float))
        and not isinstance(x, bool)
        and math.isfinite(float(x))
        and 0 <= float(x) <= 1
    )


def add(v, code):
    v.append(code)


def load_json(files, name, violations):
    if name not in files:
        return None

    value = files[name]

    if not isinstance(value, str):
        add(violations, f"INVALID_FILE:{name}")
        return None

    try:
        return json.loads(value)
    except Exception:
        add(violations, f"INVALID_JSON:{name}")
        return None


def find_model_card(readme: str):
    """
    Locate literal model-card markers without trying to parse braces
    with a regex. JSON strings containing braces therefore work normally.
    """
    prefix = "<!-- tds-model-card"
    suffix = "-->"

    positions = []
    pos = 0

    while True:
        i = readme.find(prefix, pos)
        if i < 0:
            break
        positions.append(i)
        pos = i + len(prefix)

    if not positions:
        return "missing", None

    if len(positions) > 1:
        return "count", None

    start = positions[0] + len(prefix)
    end = readme.find(suffix, start)

    if end < 0:
        return "invalid", None

    payload = readme[start:end]

    try:
        obj = json.loads(payload)
    except Exception:
        return "invalid", None

    if not isinstance(obj, dict):
        return "invalid", None

    return "valid", obj


def validate_policy(policy, violations):
    if not isinstance(policy, dict):
        add(violations, "INVALID_POLICY")
        return

    slices = policy.get("requiredSlices")

    if not (
        isinstance(slices, list)
        and len(slices) > 0
        and all(nonempty_string(x) for x in slices)
        and len(slices) == len(set(slices))
    ):
        add(violations, "INVALID_POLICY")

    for field in ("license", "intendedUse", "limitations"):
        if not nonempty_string(policy.get(field)):
            add(violations, "INVALID_POLICY")


@app.get("/")
def root():
    return {"status": "ok"}


@app.post("/verify-bundle")
async def verify_bundle(request: Request):
    # Request-level invalid input is the only case returning HTTP 400.
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(
            status_code=400,
            content={"error": "INVALID_INPUT"},
        )

    if not isinstance(body, dict):
        return JSONResponse(
            status_code=400,
            content={"error": "INVALID_INPUT"},
        )

    policy = body.get("policy")
    files = body.get("files")

    if not isinstance(policy, dict) or not isinstance(files, dict):
        return JSONResponse(
            status_code=400,
            content={"error": "INVALID_INPUT"},
        )

    violations = []
    validate_policy(policy, violations)

    # ---------------------------------------------------------
    # 1. Required files and exact UTF-8 representation
    # ---------------------------------------------------------

    for name in REQUIRED_FILES:
        if name not in files:
            add(violations, f"MISSING_FILE:{name}")

    raw = {}

    for name, value in files.items():
        if not isinstance(name, str):
            add(violations, "UNTRACKED_FILE")
            continue

        # Every bundle file is required to be supplied as a UTF-8 string.
        if not isinstance(value, str):
            add(violations, f"INVALID_FILE:{name}")
            continue

        try:
            raw[name] = value.encode("utf-8")
        except UnicodeEncodeError:
            add(violations, f"INVALID_FILE:{name}")

    # ---------------------------------------------------------
    # 2. Extra files and unsafe extensions
    # ---------------------------------------------------------

    required_set = set(REQUIRED_FILES)

    for name in files:
        if not isinstance(name, str):
            continue

        if name not in required_set:
            add(violations, "UNTRACKED_FILE")

        lower = name.lower()
        if any(lower.endswith(ext) for ext in UNSAFE_EXTENSIONS):
            add(violations, "UNSAFE_WEIGHTS")

    # ---------------------------------------------------------
    # 3. Exact inventory
    #
    # Inventory lists every file except inventory.json.
    # Ordering is by UTF-8 filename bytes.
    # Each object must have exact key order:
    # name,bytes,sha256
    # ---------------------------------------------------------

    inventory = load_json(files, "inventory.json", violations)

    expected_entries = []

    inventory_names = [
        name
        for name in files
        if isinstance(name, str) and name != "inventory.json"
    ]

    inventory_names.sort(key=lambda x: x.encode("utf-8"))

    for name in inventory_names:
        if name in raw:
            expected_entries.append({
                "name": name,
                "bytes": len(raw[name]),
                "sha256": sha256(raw[name]),
            })

    recomputed_inventory_bytes = compact_json(expected_entries)
    inventory_digest = sha256(recomputed_inventory_bytes)

    if isinstance(inventory, list):
        inventory_ok = True

        if len(inventory) != len(expected_entries):
            inventory_ok = False
        else:
            for actual, expected in zip(inventory, expected_entries):
                if not isinstance(actual, dict):
                    inventory_ok = False
                    break

                # Exact key order matters.
                if list(actual.keys()) != ["name", "bytes", "sha256"]:
                    inventory_ok = False
                    break

                if actual.get("name") != expected["name"]:
                    inventory_ok = False
                    break

                if actual.get("bytes") != expected["bytes"]:
                    inventory_ok = False
                    break

                if actual.get("sha256") != expected["sha256"]:
                    inventory_ok = False
                    break

        if not inventory_ok:
            add(violations, "INVENTORY_MISMATCH")

    elif "INVALID_JSON:inventory.json" not in violations:
        add(violations, "INVENTORY_MISMATCH")

    # ---------------------------------------------------------
    # 4. Adapter config
    # ---------------------------------------------------------

    config = load_json(files, "adapter_config.json", violations)

    if isinstance(config, dict):
        targets = config.get("target_modules")

        if not (
            safe_positive_int(config.get("r"))
            and isinstance(targets, list)
            and len(targets) > 0
            and all(nonempty_string(x) for x in targets)
            and len(targets) == len(set(targets))
        ):
            add(violations, "INVALID_ADAPTER_CONFIG")

    elif "INVALID_JSON:adapter_config.json" not in violations:
        add(violations, "INVALID_ADAPTER_CONFIG")

    # ---------------------------------------------------------
    # 5. Training manifest / immutable lineage
    # ---------------------------------------------------------

    manifest = load_json(files, "training_manifest.json", violations)

    manifest_fields = (
        "baseRevision",
        "task",
        "datasetDigest",
        "codeDigest",
        "trainingConfigDigest",
        "modelArtifactDigest",
        "evaluationArtifactDigest",
    )

    if isinstance(manifest, dict):
        for field in manifest_fields:
            if field not in manifest or not nonempty_string(manifest[field]):
                add(violations, f"MISSING_MANIFEST_FIELD:{field}")

        base = manifest.get("baseRevision")
        if not (
            isinstance(base, str)
            and HEX40.fullmatch(base) is not None
        ):
            add(violations, "MUTABLE_BASE_REVISION")

    elif "INVALID_JSON:training_manifest.json" not in violations:
        add(violations, "INVALID_TRAINING_MANIFEST")

    # ---------------------------------------------------------
    # 6. Artifact identity
    # ---------------------------------------------------------

    model_digest = None
    evaluation_digest = None

    if "adapter_model.safetensors" in raw:
        model_digest = sha256(raw["adapter_model.safetensors"])

    if "evaluation.json" in raw:
        evaluation_digest = sha256(raw["evaluation.json"])

    if isinstance(manifest, dict):
        if (
            model_digest is not None
            and nonempty_string(manifest.get("modelArtifactDigest"))
            and manifest["modelArtifactDigest"] != model_digest
        ):
            add(violations, "MODEL_ARTIFACT_MISMATCH")

        if (
            evaluation_digest is not None
            and nonempty_string(manifest.get("evaluationArtifactDigest"))
            and manifest["evaluationArtifactDigest"] != evaluation_digest
        ):
            add(violations, "EVALUATION_DIGEST_MISMATCH")

    # ---------------------------------------------------------
    # 7. Evaluation binding
    # ---------------------------------------------------------

    evaluation = load_json(files, "evaluation.json", violations)

    if isinstance(evaluation, dict):
        if (
            model_digest is not None
            and evaluation.get("modelArtifactDigest") != model_digest
        ):
            add(violations, "EVALUATION_ARTIFACT_MISMATCH")

        if not unit_float(evaluation.get("aggregate")):
            add(violations, "INVALID_AGGREGATE")

        required_slices = policy.get("requiredSlices", [])
        slices = evaluation.get("slices")

        if not isinstance(slices, dict):
            for name in required_slices:
                add(violations, f"MISSING_SLICE:{name}")
        else:
            for name in required_slices:
                if name not in slices:
                    add(violations, f"MISSING_SLICE:{name}")
                elif not unit_float(slices[name]):
                    add(violations, f"SLICE_RANGE:{name}")

    elif "INVALID_JSON:evaluation.json" not in violations:
        add(violations, "INVALID_EVALUATION")

    # ---------------------------------------------------------
    # 8. Model card
    # ---------------------------------------------------------

    status = "missing"
    card = None

    if isinstance(files.get("README.md"), str):
        status, card = find_model_card(files["README.md"])

    if status == "missing":
        add(violations, "MODEL_CARD_COUNT")
        add(violations, "MISSING_MODEL_CARD")

    elif status == "count":
        add(violations, "MODEL_CARD_COUNT")

    elif status == "invalid":
        add(violations, "INVALID_MODEL_CARD")

    else:
        expected_card = {
            "task": manifest.get("task") if isinstance(manifest, dict) else None,
            "baseRevision": manifest.get("baseRevision") if isinstance(manifest, dict) else None,
            "datasetDigest": manifest.get("datasetDigest") if isinstance(manifest, dict) else None,
            "modelArtifactDigest": manifest.get("modelArtifactDigest") if isinstance(manifest, dict) else None,
            "license": policy.get("license"),
            "intendedUse": policy.get("intendedUse"),
            "limitations": policy.get("limitations"),
        }

        if any(card.get(k) != v for k, v in expected_card.items()):
            add(violations, "MODEL_CARD_MISMATCH")

    # ---------------------------------------------------------
    # 9. Deterministic serialization
    # ---------------------------------------------------------

    violations = sorted(
        set(violations),
        key=lambda x: x.encode("utf-8"),
    )

    return {
        "decision": "admit" if len(violations) == 0 else "reject",
        "violations": violations,
        "inventoryDigest": inventory_digest,
    }
