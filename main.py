import hashlib
import json
import math
import re
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()

REQUIRED_FILES = [
    "README.md",
    "training_manifest.json",
    "evaluation.json",
    "inventory.json",
    "adapter_model.safetensors",
    "adapter_config.json",
]

UNSAFE_EXTENSIONS = {".bin", ".pt", ".pth", ".pkl", ".pickle"}
HEX40_RE = re.compile(r"^[0-9a-f]{40}$")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def compact_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def is_safe_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and 0 < value <= 9007199254740991


def is_nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and len(value) > 0


def finite_unit(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and 0 <= float(value) <= 1
    )


def add_violation(violations, code):
    violations.append(code)


def parse_json_file(files, name, violations):
    if name not in files:
        return None
    raw = files[name]
    if not isinstance(raw, str):
        add_violation(violations, f"INVALID_FILE:{name}")
        return None
    try:
        return json.loads(raw)
    except Exception:
        add_violation(violations, f"INVALID_JSON:{name}")
        return None


def extract_model_card(readme: str):
    prefix = "<!-- tds-model-card"
    suffix = "-->"
    positions = []
    start = 0
    while True:
        idx = readme.find(prefix, start)
        if idx == -1:
            break
        positions.append(idx)
        start = idx + len(prefix)

    if len(positions) == 0:
        return "missing", None
    if len(positions) > 1:
        return "count", None

    payload_start = positions[0] + len(prefix)
    end = readme.find(suffix, payload_start)
    if end == -1:
        return "invalid", None

    try:
        value = json.loads(readme[payload_start:end])
    except Exception:
        return "invalid", None

    if not isinstance(value, dict):
        return "invalid", None
    return "valid", value


def verify_policy(policy, violations):
    if not isinstance(policy, dict):
        add_violation(violations, "INVALID_POLICY")
        return

    required_slices = policy.get("requiredSlices")
    valid_slices = (
        isinstance(required_slices, list)
        and len(required_slices) > 0
        and all(is_nonempty_string(x) for x in required_slices)
        and len(set(required_slices)) == len(required_slices)
    )
    if not valid_slices or not all(
        is_nonempty_string(policy.get(x))
        for x in ["license", "intendedUse", "limitations"]
    ):
        add_violation(violations, "INVALID_POLICY")


@app.post("/verify-bundle")
async def verify_bundle(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "INVALID_INPUT"})

    if not isinstance(body, dict):
        return JSONResponse(status_code=400, content={"error": "INVALID_INPUT"})

    policy = body.get("policy")
    files = body.get("files")

    if not isinstance(policy, dict) or not isinstance(files, dict):
        return JSONResponse(status_code=400, content={"error": "INVALID_INPUT"})

    violations = []
    verify_policy(policy, violations)

    for name in REQUIRED_FILES:
        if name not in files:
            add_violation(violations, f"MISSING_FILE:{name}")

    raw_bytes = {}
    for name, value in files.items():
        if not isinstance(name, str):
            add_violation(violations, "UNTRACKED_FILE")
            continue
        if name not in REQUIRED_FILES:
            add_violation(violations, "UNTRACKED_FILE")
        if any(name.lower().endswith(ext) for ext in UNSAFE_EXTENSIONS):
            add_violation(violations, "UNSAFE_WEIGHTS")
        if isinstance(value, str):
            try:
                raw_bytes[name] = value.encode("utf-8")
            except Exception:
                add_violation(violations, f"INVALID_FILE:{name}")
        else:
            add_violation(violations, f"INVALID_FILE:{name}")

    inventory = parse_json_file(files, "inventory.json", violations)
    expected_inventory = []
    for name in sorted(
        [n for n in files if isinstance(n, str) and n != "inventory.json"],
        key=lambda x: x.encode("utf-8"),
    ):
        if name in raw_bytes:
            expected_inventory.append({
                "name": name,
                "bytes": len(raw_bytes[name]),
                "sha256": sha256_bytes(raw_bytes[name]),
            })

    inventory_digest = sha256_bytes(compact_json(expected_inventory))

    if isinstance(inventory, list):
        valid = len(inventory) == len(expected_inventory)
        if valid:
            for actual, expected in zip(inventory, expected_inventory):
                if not isinstance(actual, dict) or list(actual.keys()) != ["name", "bytes", "sha256"] or actual != expected:
                    valid = False
                    break
        if not valid:
            add_violation(violations, "INVENTORY_MISMATCH")
    elif "INVALID_JSON:inventory.json" not in violations:
        add_violation(violations, "INVENTORY_MISMATCH")

    config = parse_json_file(files, "adapter_config.json", violations)
    if isinstance(config, dict):
        target_modules = config.get("target_modules")
        if not (
            is_safe_integer(config.get("r"))
            and isinstance(target_modules, list)
            and len(target_modules) > 0
            and all(is_nonempty_string(x) for x in target_modules)
            and len(set(target_modules)) == len(target_modules)
        ):
            add_violation(violations, "INVALID_ADAPTER_CONFIG")
    elif "INVALID_JSON:adapter_config.json" not in violations:
        add_violation(violations, "INVALID_ADAPTER_CONFIG")

    manifest = parse_json_file(files, "training_manifest.json", violations)
    manifest_fields = [
        "baseRevision", "task", "datasetDigest", "codeDigest",
        "trainingConfigDigest", "modelArtifactDigest", "evaluationArtifactDigest"
    ]

    if isinstance(manifest, dict):
        for field in manifest_fields:
            if field not in manifest or not is_nonempty_string(manifest[field]):
                add_violation(violations, f"MISSING_MANIFEST_FIELD:{field}")
        if not (
            isinstance(manifest.get("baseRevision"), str)
            and HEX40_RE.fullmatch(manifest["baseRevision"]) is not None
        ):
            add_violation(violations, "MUTABLE_BASE_REVISION")
    elif "INVALID_JSON:training_manifest.json" not in violations:
        add_violation(violations, "INVALID_TRAINING_MANIFEST")

    model_digest = sha256_bytes(raw_bytes["adapter_model.safetensors"]) if "adapter_model.safetensors" in raw_bytes else None
    evaluation_digest = sha256_bytes(raw_bytes["evaluation.json"]) if "evaluation.json" in raw_bytes else None

    if isinstance(manifest, dict):
        if model_digest is not None and manifest.get("modelArtifactDigest") != model_digest:
            add_violation(violations, "MODEL_ARTIFACT_MISMATCH")
        if evaluation_digest is not None and manifest.get("evaluationArtifactDigest") != evaluation_digest:
            add_violation(violations, "EVALUATION_DIGEST_MISMATCH")

    evaluation = parse_json_file(files, "evaluation.json", violations)
    if isinstance(evaluation, dict):
        if model_digest is not None and evaluation.get("modelArtifactDigest") != model_digest:
            add_violation(violations, "EVALUATION_ARTIFACT_MISMATCH")

        if not finite_unit(evaluation.get("aggregate")):
            add_violation(violations, "INVALID_AGGREGATE")

        slices = evaluation.get("slices")
        required_slices = policy.get("requiredSlices", [])
        if not isinstance(slices, dict):
            for s in required_slices:
                add_violation(violations, f"MISSING_SLICE:{s}")
        else:
            for s in required_slices:
                if s not in slices:
                    add_violation(violations, f"MISSING_SLICE:{s}")
                elif not finite_unit(slices[s]):
                    add_violation(violations, f"SLICE_RANGE:{s}")
    elif "INVALID_JSON:evaluation.json" not in violations:
        add_violation(violations, "INVALID_EVALUATION")

    status, card = ("missing", None)
    if isinstance(files.get("README.md"), str):
        status, card = extract_model_card(files["README.md"])

    if status == "missing":
        add_violation(violations, "MODEL_CARD_COUNT")
        add_violation(violations, "MISSING_MODEL_CARD")
    elif status == "count":
        add_violation(violations, "MODEL_CARD_COUNT")
    elif status == "invalid":
        add_violation(violations, "INVALID_MODEL_CARD")
    else:
        expected = {
            "task": manifest.get("task") if isinstance(manifest, dict) else None,
            "baseRevision": manifest.get("baseRevision") if isinstance(manifest, dict) else None,
            "datasetDigest": manifest.get("datasetDigest") if isinstance(manifest, dict) else None,
            "modelArtifactDigest": manifest.get("modelArtifactDigest") if isinstance(manifest, dict) else None,
            "license": policy.get("license"),
            "intendedUse": policy.get("intendedUse"),
            "limitations": policy.get("limitations"),
        }
        if any(card.get(k) != v for k, v in expected.items()):
            add_violation(violations, "MODEL_CARD_MISMATCH")

    violations = sorted(set(violations), key=lambda x: x.encode("utf-8"))
    return {
        "decision": "admit" if not violations else "reject",
        "violations": violations,
        "inventoryDigest": inventory_digest,
    }
