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

MAX_SAFE_INTEGER = 9007199254740991
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def compact_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")


def is_nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and len(value) > 0


def is_safe_integer(value: Any) -> bool:
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and 0 < value <= MAX_SAFE_INTEGER
    )


def is_unit_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and 0 <= float(value) <= 1
    )


def unique_violation(violations, code):
    violations.append(code)


def reject_duplicate_keys(pairs):
    result = {}

    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value

    return result


def parse_json_file(files, filename, violations):
    if filename not in files:
        return None

    value = files[filename]

    if not isinstance(value, str):
        unique_violation(
            violations,
            f"INVALID_FILE:{filename}",
        )
        return None

    try:
        return json.loads(
            value,
            object_pairs_hook=reject_duplicate_keys,
        )
    except Exception:
        unique_violation(
            violations,
            f"INVALID_JSON:{filename}",
        )
        return None


def validate_policy(policy, violations):
    if not isinstance(policy, dict):
        unique_violation(violations, "INVALID_POLICY")
        return

    required_slices = policy.get("requiredSlices")

    if not (
        isinstance(required_slices, list)
        and len(required_slices) > 0
        and all(is_nonempty_string(x) for x in required_slices)
        and len(required_slices) == len(set(required_slices))
    ):
        unique_violation(violations, "INVALID_POLICY")

    for field in (
        "license",
        "intendedUse",
        "limitations",
    ):
        if not is_nonempty_string(policy.get(field)):
            unique_violation(violations, "INVALID_POLICY")


def parse_model_card(readme):
    prefix = "<!-- tds-model-card"
    suffix = "-->"

    positions = []
    cursor = 0

    while True:
        index = readme.find(prefix, cursor)

        if index == -1:
            break

        positions.append(index)
        cursor = index + len(prefix)

    if len(positions) == 0:
        return "missing", None

    if len(positions) > 1:
        return "count", None

    payload_start = positions[0] + len(prefix)
    payload_end = readme.find(suffix, payload_start)

    if payload_end == -1:
        return "invalid", None

    payload = readme[payload_start:payload_end]

    try:
        card = json.loads(
            payload,
            object_pairs_hook=reject_duplicate_keys,
        )
    except Exception:
        return "invalid", None

    if not isinstance(card, dict):
        return "invalid", None

    return "valid", card


@app.get("/")
def root():
    return {"status": "ok"}


@app.post("/verify-bundle")
async def verify_bundle(request: Request):

    # ============================================================
    # REQUEST VALIDATION
    # ============================================================

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

    # ============================================================
    # REQUIRED FILES
    # ============================================================

    for filename in REQUIRED_FILES:
        if filename not in files:
            unique_violation(
                violations,
                f"MISSING_FILE:{filename}",
            )

    # ============================================================
    # FILE VALUES MUST BE UTF-8 STRINGS
    # ============================================================

    raw_bytes = {}

    for filename, value in files.items():

        if not isinstance(filename, str):
            unique_violation(
                violations,
                "UNTRACKED_FILE",
            )
            continue

        if not isinstance(value, str):
            unique_violation(
                violations,
                f"INVALID_FILE:{filename}",
            )
            continue

        try:
            raw_bytes[filename] = value.encode("utf-8")
        except UnicodeEncodeError:
            unique_violation(
                violations,
                f"INVALID_FILE:{filename}",
            )

    # ============================================================
    # EXTRA FILES + UNSAFE WEIGHTS
    # ============================================================

    required_set = set(REQUIRED_FILES)

    for filename in files:

        if not isinstance(filename, str):
            continue

        if filename not in required_set:
            unique_violation(
                violations,
                "UNTRACKED_FILE",
            )

        lower_name = filename.lower()

        if any(
            lower_name.endswith(extension)
            for extension in UNSAFE_EXTENSIONS
        ):
            unique_violation(
                violations,
                "UNSAFE_WEIGHTS",
            )

    # ============================================================
    # INVENTORY
    # ============================================================

    inventory = parse_json_file(
        files,
        "inventory.json",
        violations,
    )

    # Every supplied file except inventory.json.
    inventory_names = [
        filename
        for filename in files
        if (
            isinstance(filename, str)
            and filename != "inventory.json"
        )
    ]

    # UTF-8 byte ordering, NOT Unicode code-point ordering.
    inventory_names.sort(
        key=lambda filename: filename.encode("utf-8")
    )

    expected_inventory = []

    for filename in inventory_names:

        # A non-string file value cannot have a valid UTF-8 byte
        # representation and therefore cannot produce a valid entry.
        if filename not in raw_bytes:
            continue

        data = raw_bytes[filename]

        expected_inventory.append(
            {
                "name": filename,
                "bytes": len(data),
                "sha256": sha256_bytes(data),
            }
        )

    # inventoryDigest is based on the recomputed compact JSON.
    inventory_digest = sha256_bytes(
        compact_json(expected_inventory)
    )

    if isinstance(inventory, list):

        inventory_valid = True

        # Exact number of entries.
        if len(inventory) != len(expected_inventory):
            inventory_valid = False

        if inventory_valid:

            for actual, expected in zip(
                inventory,
                expected_inventory,
            ):

                # Every entry must be an object.
                if not isinstance(actual, dict):
                    inventory_valid = False
                    break

                # EXACT key order.
                if list(actual.keys()) != [
                    "name",
                    "bytes",
                    "sha256",
                ]:
                    inventory_valid = False
                    break

                # Exact types.
                if not isinstance(
                    actual["name"],
                    str,
                ):
                    inventory_valid = False
                    break

                if not (
                    isinstance(actual["bytes"], int)
                    and not isinstance(
                        actual["bytes"],
                        bool,
                    )
                    and 0 <= actual["bytes"]
                    <= MAX_SAFE_INTEGER
                ):
                    inventory_valid = False
                    break

                if not (
                    isinstance(
                        actual["sha256"],
                        str,
                    )
                    and HEX64.fullmatch(
                        actual["sha256"]
                    )
                    is not None
                ):
                    inventory_valid = False
                    break

                # Lowercase SHA-256 is mandatory.
                if actual["sha256"] != actual["sha256"].lower():
                    inventory_valid = False
                    break

                # Exact values.
                if actual["name"] != expected["name"]:
                    inventory_valid = False
                    break

                if actual["bytes"] != expected["bytes"]:
                    inventory_valid = False
                    break

                if actual["sha256"] != expected["sha256"]:
                    inventory_valid = False
                    break

        if not inventory_valid:
            unique_violation(
                violations,
                "INVENTORY_MISMATCH",
            )

    elif "INVALID_JSON:inventory.json" not in violations:

        unique_violation(
            violations,
            "INVENTORY_MISMATCH",
        )

    # ============================================================
    # ADAPTER CONFIG
    # ============================================================

    adapter_config = parse_json_file(
        files,
        "adapter_config.json",
        violations,
    )

    if isinstance(adapter_config, dict):

        target_modules = adapter_config.get(
            "target_modules"
        )

        valid_config = (
            is_safe_integer(
                adapter_config.get("r")
            )
            and isinstance(
                target_modules,
                list,
            )
            and len(target_modules) > 0
            and all(
                is_nonempty_string(x)
                for x in target_modules
            )
            and len(target_modules)
            == len(set(target_modules))
        )

        if not valid_config:
            unique_violation(
                violations,
                "INVALID_ADAPTER_CONFIG",
            )

    elif "INVALID_JSON:adapter_config.json" not in violations:

        unique_violation(
            violations,
            "INVALID_ADAPTER_CONFIG",
        )

    # ============================================================
    # TRAINING MANIFEST
    # ============================================================

    manifest = parse_json_file(
        files,
        "training_manifest.json",
        violations,
    )

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

            if (
                field not in manifest
                or not is_nonempty_string(
                    manifest[field]
                )
            ):
                unique_violation(
                    violations,
                    f"MISSING_MANIFEST_FIELD:{field}",
                )

        base_revision = manifest.get(
            "baseRevision"
        )

        if not (
            isinstance(base_revision, str)
            and HEX40.fullmatch(
                base_revision
            ) is not None
        ):
            unique_violation(
                violations,
                "MUTABLE_BASE_REVISION",
            )

    elif "INVALID_JSON:training_manifest.json" not in violations:

        unique_violation(
            violations,
            "INVALID_TRAINING_MANIFEST",
        )

    # ============================================================
    # ARTIFACT DIGESTS
    # ============================================================

    model_digest = None
    evaluation_digest = None

    if "adapter_model.safetensors" in raw_bytes:

        model_digest = sha256_bytes(
            raw_bytes[
                "adapter_model.safetensors"
            ]
        )

    if "evaluation.json" in raw_bytes:

        evaluation_digest = sha256_bytes(
            raw_bytes["evaluation.json"]
        )

    # Manifest -> model artifact.
    if (
        isinstance(manifest, dict)
        and model_digest is not None
        and is_nonempty_string(
            manifest.get(
                "modelArtifactDigest"
            )
        )
    ):

        if (
            manifest["modelArtifactDigest"]
            != model_digest
        ):
            unique_violation(
                violations,
                "MODEL_ARTIFACT_MISMATCH",
            )

    # Manifest -> exact evaluation.json bytes.
    if (
        isinstance(manifest, dict)
        and evaluation_digest is not None
        and is_nonempty_string(
            manifest.get(
                "evaluationArtifactDigest"
            )
        )
    ):

        if (
            manifest["evaluationArtifactDigest"]
            != evaluation_digest
        ):
            unique_violation(
                violations,
                "EVALUATION_DIGEST_MISMATCH",
            )

    # ============================================================
    # EVALUATION
    # ============================================================

    evaluation = parse_json_file(
        files,
        "evaluation.json",
        violations,
    )

    if isinstance(evaluation, dict):

        # Evaluation must bind to exact model bytes.
        if (
            model_digest is not None
            and evaluation.get(
                "modelArtifactDigest"
            )
            != model_digest
        ):
            unique_violation(
                violations,
                "EVALUATION_ARTIFACT_MISMATCH",
            )

        # Aggregate.
        if not is_unit_number(
            evaluation.get("aggregate")
        ):
            unique_violation(
                violations,
                "INVALID_AGGREGATE",
            )

        required_slices = policy.get(
            "requiredSlices",
            [],
        )

        slices = evaluation.get("slices")

        if not isinstance(slices, dict):

            for slice_name in required_slices:
                unique_violation(
                    violations,
                    f"MISSING_SLICE:{slice_name}",
                )

        else:

            for slice_name in required_slices:

                if slice_name not in slices:

                    unique_violation(
                        violations,
                        f"MISSING_SLICE:{slice_name}",
                    )

                elif not is_unit_number(
                    slices[slice_name]
                ):

                    unique_violation(
                        violations,
                        f"SLICE_RANGE:{slice_name}",
                    )

    elif "INVALID_JSON:evaluation.json" not in violations:

        unique_violation(
            violations,
            "INVALID_EVALUATION",
        )

    # ============================================================
    # MODEL CARD
    # ============================================================

    if isinstance(
        files.get("README.md"),
        str,
    ):
        card_status, card = parse_model_card(
            files["README.md"]
        )
    else:
        card_status, card = "missing", None

    if card_status == "missing":

        unique_violation(
            violations,
            "MODEL_CARD_COUNT",
        )

        unique_violation(
            violations,
            "MISSING_MODEL_CARD",
        )

    elif card_status == "count":

        unique_violation(
            violations,
            "MODEL_CARD_COUNT",
        )

    elif card_status == "invalid":

        unique_violation(
            violations,
            "INVALID_MODEL_CARD",
        )

    elif card_status == "valid":

        expected_card = {
            "task": (
                manifest.get("task")
                if isinstance(manifest, dict)
                else None
            ),
            "baseRevision": (
                manifest.get("baseRevision")
                if isinstance(manifest, dict)
                else None
            ),
            "datasetDigest": (
                manifest.get("datasetDigest")
                if isinstance(manifest, dict)
                else None
            ),
            "modelArtifactDigest": (
                manifest.get(
                    "modelArtifactDigest"
                )
                if isinstance(manifest, dict)
                else None
            ),
            "license": policy.get("license"),
            "intendedUse": policy.get("intendedUse"),
            "limitations": policy.get("limitations"),
        }

        for field, expected_value in expected_card.items():

            if card.get(field) != expected_value:

                unique_violation(
                    violations,
                    "MODEL_CARD_MISMATCH",
                )
                break

    # ============================================================
    # FINAL DETERMINISTIC SERIALIZATION
    # ============================================================

    violations = sorted(
        set(violations),
        key=lambda code: code.encode("utf-8"),
    )

    return {
        "decision": (
            "admit"
            if len(violations) == 0
            else "reject"
        ),
        "violations": violations,
        "inventoryDigest": inventory_digest,
    }
