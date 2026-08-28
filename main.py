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

UNSAFE_EXTENSIONS = (
    ".bin",
    ".pt",
    ".pth",
    ".pkl",
    ".pickle",
)

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


def nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and len(value) > 0


def safe_integer(value: Any) -> bool:
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and 0 < value <= MAX_SAFE_INTEGER
    )


def finite_unit(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and 0 <= float(value) <= 1
    )


def add_violation(violations, code):
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
        add_violation(
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
        add_violation(
            violations,
            f"INVALID_JSON:{filename}",
        )
        return None


def validate_policy(policy, violations):
    if not isinstance(policy, dict):
        add_violation(
            violations,
            "INVALID_POLICY",
        )
        return

    required_slices = policy.get(
        "requiredSlices"
    )

    if not (
        isinstance(required_slices, list)
        and len(required_slices) > 0
        and all(
            nonempty_string(x)
            for x in required_slices
        )
        and len(required_slices)
        == len(set(required_slices))
    ):
        add_violation(
            violations,
            "INVALID_POLICY",
        )

    for field in (
        "license",
        "intendedUse",
        "limitations",
    ):
        if not nonempty_string(
            policy.get(field)
        ):
            add_violation(
                violations,
                "INVALID_POLICY",
            )


def parse_model_card(readme):
    prefix = "<!-- tds-model-card"
    suffix = "-->"

    positions = []
    cursor = 0

    while True:
        pos = readme.find(prefix, cursor)

        if pos == -1:
            break

        positions.append(pos)
        cursor = pos + len(prefix)

    if len(positions) == 0:
        return "missing", None

    if len(positions) > 1:
        return "count", None

    start = positions[0] + len(prefix)
    end = readme.find(suffix, start)

    if end == -1:
        return "invalid", None

    payload = readme[start:end]

    try:
        value = json.loads(
            payload,
            object_pairs_hook=reject_duplicate_keys,
        )
    except Exception:
        return "invalid", None

    if not isinstance(value, dict):
        return "invalid", None

    return "valid", value


@app.get("/")
def root():
    return {"status": "ok"}


@app.post("/verify-bundle")
async def verify_bundle(request: Request):

    # ==========================================================
    # INPUT
    # ==========================================================

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

    if (
        not isinstance(policy, dict)
        or not isinstance(files, dict)
    ):
        return JSONResponse(
            status_code=400,
            content={"error": "INVALID_INPUT"},
        )

    violations = []

    # ==========================================================
    # POLICY
    # ==========================================================

    validate_policy(
        policy,
        violations,
    )

    # ==========================================================
    # REQUIRED FILES
    # ==========================================================

    for filename in REQUIRED_FILES:
        if filename not in files:
            add_violation(
                violations,
                f"MISSING_FILE:{filename}",
            )

    # ==========================================================
    # UTF-8 FILES
    # ==========================================================

    raw_bytes = {}

    for filename, value in files.items():

        if not isinstance(filename, str):
            add_violation(
                violations,
                "UNTRACKED_FILE",
            )
            continue

        if not isinstance(value, str):
            add_violation(
                violations,
                f"INVALID_FILE:{filename}",
            )
            continue

        try:
            raw_bytes[filename] = value.encode(
                "utf-8"
            )
        except UnicodeEncodeError:
            add_violation(
                violations,
                f"INVALID_FILE:{filename}",
            )

    # ==========================================================
    # EXTRA FILES
    # ==========================================================

    required_set = set(REQUIRED_FILES)

    for filename in files:

        if not isinstance(filename, str):
            continue

        if filename not in required_set:
            add_violation(
                violations,
                "UNTRACKED_FILE",
            )

    # ==========================================================
    # UNSAFE EXTENSIONS
    # ==========================================================

    for filename in files:

        if not isinstance(filename, str):
            continue

        lowered = filename.lower()

        if any(
            lowered.endswith(ext)
            for ext in UNSAFE_EXTENSIONS
        ):
            add_violation(
                violations,
                "UNSAFE_WEIGHTS",
            )

    # ==========================================================
    # INVENTORY
    # ==========================================================

    inventory = parse_json_file(
        files,
        "inventory.json",
        violations,
    )

    inventory_names = [
        filename
        for filename in files
        if (
            isinstance(filename, str)
            and filename != "inventory.json"
        )
    ]

    try:
        inventory_names.sort(
            key=lambda name: name.encode(
                "utf-8"
            )
        )
    except UnicodeEncodeError:
        add_violation(
            violations,
            "INVENTORY_MISMATCH",
        )

    expected_inventory = []

    for filename in inventory_names:

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

    inventory_digest = sha256_bytes(
        compact_json(
            expected_inventory
        )
    )

    inventory_ok = True

    if not isinstance(
        inventory,
        list,
    ):
        inventory_ok = False

    elif len(inventory) != len(
        expected_inventory
    ):
        inventory_ok = False

    else:

        for actual, expected in zip(
            inventory,
            expected_inventory,
        ):

            if not isinstance(
                actual,
                dict,
            ):
                inventory_ok = False
                break

            if list(actual.keys()) != [
                "name",
                "bytes",
                "sha256",
            ]:
                inventory_ok = False
                break

            if not isinstance(
                actual["name"],
                str,
            ):
                inventory_ok = False
                break

            if not (
                isinstance(
                    actual["bytes"],
                    int,
                )
                and not isinstance(
                    actual["bytes"],
                    bool,
                )
                and 0 <= actual["bytes"]
                <= MAX_SAFE_INTEGER
            ):
                inventory_ok = False
                break

            digest = actual["sha256"]

            if not (
                isinstance(
                    digest,
                    str,
                )
                and HEX64.fullmatch(
                    digest
                ) is not None
                and digest == digest.lower()
            ):
                inventory_ok = False
                break

            if (
                actual["name"]
                != expected["name"]
            ):
                inventory_ok = False
                break

            if (
                actual["bytes"]
                != expected["bytes"]
            ):
                inventory_ok = False
                break

            if (
                actual["sha256"]
                != expected["sha256"]
            ):
                inventory_ok = False
                break

    if not inventory_ok:
        add_violation(
            violations,
            "INVENTORY_MISMATCH",
        )

    # ==========================================================
    # ADAPTER CONFIG
    # ==========================================================

    config = parse_json_file(
        files,
        "adapter_config.json",
        violations,
    )

    if isinstance(config, dict):

        target_modules = config.get(
            "target_modules"
        )

        if not (
            safe_integer(
                config.get("r")
            )
            and isinstance(
                target_modules,
                list,
            )
            and len(target_modules) > 0
            and all(
                nonempty_string(x)
                for x in target_modules
            )
            and len(target_modules)
            == len(set(target_modules))
        ):
            add_violation(
                violations,
                "INVALID_ADAPTER_CONFIG",
            )

    elif (
        "INVALID_JSON:adapter_config.json"
        not in violations
    ):
        add_violation(
            violations,
            "INVALID_ADAPTER_CONFIG",
        )

    # ==========================================================
    # TRAINING MANIFEST
    # ==========================================================

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
                or not nonempty_string(
                    manifest[field]
                )
            ):
                add_violation(
                    violations,
                    f"MISSING_MANIFEST_FIELD:{field}",
                )

        base_revision = manifest.get(
            "baseRevision"
        )

        if not (
            isinstance(
                base_revision,
                str,
            )
            and HEX40.fullmatch(
                base_revision
            ) is not None
            and base_revision
            == base_revision.lower()
        ):
            add_violation(
                violations,
                "MUTABLE_BASE_REVISION",
            )

    elif (
        "INVALID_JSON:training_manifest.json"
        not in violations
    ):
        add_violation(
            violations,
            "INVALID_TRAINING_MANIFEST",
        )

    # ==========================================================
    # RECOMPUTED ARTIFACT DIGESTS
    # ==========================================================

    model_digest = None
    evaluation_digest = None

    if (
        "adapter_model.safetensors"
        in raw_bytes
    ):
        model_digest = sha256_bytes(
            raw_bytes[
                "adapter_model.safetensors"
            ]
        )

    if "evaluation.json" in raw_bytes:
        evaluation_digest = sha256_bytes(
            raw_bytes[
                "evaluation.json"
            ]
        )

    # Manifest model artifact binding.
    if (
        isinstance(manifest, dict)
        and model_digest is not None
        and isinstance(
            manifest.get(
                "modelArtifactDigest"
            ),
            str,
        )
        and manifest[
            "modelArtifactDigest"
        ] != model_digest
    ):
        add_violation(
            violations,
            "MODEL_ARTIFACT_MISMATCH",
        )

    # Manifest evaluation artifact binding.
    if (
        isinstance(manifest, dict)
        and evaluation_digest is not None
        and isinstance(
            manifest.get(
                "evaluationArtifactDigest"
            ),
            str,
        )
        and manifest[
            "evaluationArtifactDigest"
        ] != evaluation_digest
    ):
        add_violation(
            violations,
            "EVALUATION_DIGEST_MISMATCH",
        )

    # ==========================================================
    # EVALUATION
    # ==========================================================

    evaluation = parse_json_file(
        files,
        "evaluation.json",
        violations,
    )

    if isinstance(evaluation, dict):

        # First verify model binding.
        evaluation_model_digest = (
            evaluation.get(
                "modelArtifactDigest"
            )
        )

        if not (
            isinstance(
                evaluation_model_digest,
                str,
            )
            and len(
                evaluation_model_digest
            ) > 0
            and model_digest is not None
            and evaluation_model_digest
            == model_digest
        ):
            add_violation(
                violations,
                "EVALUATION_ARTIFACT_MISMATCH",
            )

        # Then aggregate.
        if not finite_unit(
            evaluation.get(
                "aggregate"
            )
        ):
            add_violation(
                violations,
                "INVALID_AGGREGATE",
            )

        # Then required slices.
        required_slices = policy.get(
            "requiredSlices",
            [],
        )

        slices = evaluation.get(
            "slices"
        )

        if not isinstance(
            slices,
            dict,
        ):

            for slice_name in required_slices:
                add_violation(
                    violations,
                    f"MISSING_SLICE:{slice_name}",
                )

        else:

            for slice_name in required_slices:

                if slice_name not in slices:

                    add_violation(
                        violations,
                        f"MISSING_SLICE:{slice_name}",
                    )

                elif not finite_unit(
                    slices[slice_name]
                ):

                    add_violation(
                        violations,
                        f"SLICE_RANGE:{slice_name}",
                    )

    elif (
        "INVALID_JSON:evaluation.json"
        not in violations
    ):
        add_violation(
            violations,
            "INVALID_EVALUATION",
        )

    # ==========================================================
    # MODEL CARD
    # ==========================================================

    if isinstance(
        files.get("README.md"),
        str,
    ):

        card_status, card = parse_model_card(
            files["README.md"]
        )

    else:

        card_status = "missing"
        card = None

    if card_status == "missing":

        add_violation(
            violations,
            "MODEL_CARD_COUNT",
        )

        add_violation(
            violations,
            "MISSING_MODEL_CARD",
        )

    elif card_status == "count":

        add_violation(
            violations,
            "MODEL_CARD_COUNT",
        )

    elif card_status == "invalid":

        add_violation(
            violations,
            "INVALID_MODEL_CARD",
        )

    elif card_status == "valid":

        expected_card = {
            "task": (
                manifest.get("task")
                if isinstance(
                    manifest,
                    dict,
                )
                else None
            ),
            "baseRevision": (
                manifest.get(
                    "baseRevision"
                )
                if isinstance(
                    manifest,
                    dict,
                )
                else None
            ),
            "datasetDigest": (
                manifest.get(
                    "datasetDigest"
                )
                if isinstance(
                    manifest,
                    dict,
                )
                else None
            ),
            "modelArtifactDigest": (
                manifest.get(
                    "modelArtifactDigest"
                )
                if isinstance(
                    manifest,
                    dict,
                )
                else None
            ),
            "license": policy.get(
                "license"
            ),
            "intendedUse": policy.get(
                "intendedUse"
            ),
            "limitations": policy.get(
                "limitations"
            ),
        }

        for field, expected in (
            expected_card.items()
        ):

            if card.get(field) != expected:

                add_violation(
                    violations,
                    "MODEL_CARD_MISMATCH",
                )
                break

    # ==========================================================
    # FINAL SERIALIZATION
    # ==========================================================

    violations = sorted(
        set(violations),
        key=lambda code: code.encode(
            "utf-8"
        ),
    )

    return {
        "decision": (
            "admit"
            if not violations
            else "reject"
        ),
        "violations": violations,
        "inventoryDigest": inventory_digest,
    }
