from pathlib import Path
import json

try:
    from c2pa import Reader
except ImportError:
    Reader = None


AI_TOOL_KEYWORDS = [
    "stable diffusion",
    "midjourney",
    "dall-e",
    "dall e",
    "firefly",
    "adobe firefly",
    "runway",
    "runwayml",
    "gpt",
    "chatgpt",
    "openai",
    "ai generator",
    "ai-generated",
    "ai generated",
    "generative",
    "sora",
    "pika",
    "luma",
    "gen-2",
    "runway gen",
    "photoshop",
    "lightroom",
]

AI_ASSERTION_KEYWORDS = [
    "ai",
    "c2pa.ai",
    "c2pa.generator",
    "cawg.ai_inference",
    "cawg.ai_generative_training",
]

# IPTC official AI source types
IPTC_AI_TYPES = {
    "http://cv.iptc.org/newscodes/digitalsourcetype/algorithmicMedia",
    "http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia",
    "http://cv.iptc.org/newscodes/digitalsourcetype/compositeWithTrainedAlgorithmicMedia",
}


def analyse_c2pa(path: Path) -> dict:
    result = {
        "has_c2pa": False,
        "signature_valid": None,
        "ai_assertions_found": [],
        "tools_detected": [],
        "edit_actions": [],
        "overall_c2pa_score": 0.0,
        "errors": [],
        "raw_manifest": None,
        "digital_source_types": [],  # â† YOU ADDED THIS
        "software_agents": [],  # â† AND THIS
        "claim_generator": None,
        "signer": None,
        "cert_issuer": None,
        "signing_time": None,
        "ingredients": [],
    }

    if Reader is None:
        result["errors"].append("c2pa-python not installed")
        return result

    try:
        with Reader(str(path)) as reader:
            manifest_json = reader.json()
    except Exception as e:
        result["errors"].append(str(e))
        return result

    if not manifest_json:
        return result

    result["has_c2pa"] = True

    try:
        data = json.loads(manifest_json)
    except Exception as e:
        result["errors"].append(f"Failed to parse manifest JSON: {e}")
        return result

    manifests = data.get("manifests", {})
    active_id = data.get("active_manifest")
    active = manifests.get(active_id)

    if not active:
        return result

    result["raw_manifest"] = active
    result["claim_generator"] = active.get("claim_generator") or active.get(
        "claim_generator_info"
    )

    # signature validation
    validation = active.get("validation_status")
    if isinstance(validation, dict):
        errors = validation.get("errors") or []
        result["signature_valid"] = len(errors) == 0
        if not result["cert_issuer"]:
            result["cert_issuer"] = validation.get("issuer")

    signature_info = active.get("signature_info") or {}
    if isinstance(signature_info, dict):
        result["signer"] = signature_info.get("issuer") or signature_info.get("signer")
        result["signing_time"] = signature_info.get("date") or signature_info.get(
            "time"
        )

    ai_assertions = set()
    tools = set()
    actions = set()

    # SCAN ASSERTIONS
    for assertion in active.get("assertions", []):
        label = str(assertion.get("label", "")).lower()
        data_field = assertion.get("data", {}) or {}

        if any(k in label for k in AI_ASSERTION_KEYWORDS):
            ai_assertions.add(assertion.get("label", label))

        # Samsung / modern device structured actions
        if "actions" in data_field:
            for act in data_field["actions"]:
                if not isinstance(act, dict):
                    continue

                # record edit action label
                a = act.get("action")
                if a:
                    actions.add(a)

                # ---- NEW: Save softwareAgent field ----
                software = act.get("softwareAgent")
                if software:
                    result["software_agents"].append(software)

                    # Samsung special detection
                    if software.lower() == "photo assist":
                        tools.add("Samsung Photo Assist")
                        ai_assertions.add("samsung_photo_assist")
                    if any(k in software.lower() for k in AI_TOOL_KEYWORDS):
                        tools.add(software)
                        ai_assertions.add("software_agent_ai_tool")

                # ---- NEW: Save digital source type ----
                dst = act.get("digitalSourceType")
                if dst:
                    result["digital_source_types"].append(dst)

                    if dst in IPTC_AI_TYPES:
                        ai_assertions.add(f"iptc:{dst}")

        software_field = data_field.get("softwareAgent") or data_field.get(
            "software_agent"
        )
        if software_field and isinstance(software_field, str):
            result["software_agents"].append(software_field)
            if any(k in software_field.lower() for k in AI_TOOL_KEYWORDS):
                tools.add(software_field)
                ai_assertions.add("software_agent_ai_tool")

    ingredients = active.get("ingredients") or []
    if isinstance(ingredients, list):
        for ing in ingredients:
            if not isinstance(ing, dict):
                continue
            result["ingredients"].append(ing)
            dst = (
                ing.get("digitalSourceType")
                or ing.get("data", {}).get("digitalSourceType")
                or ing.get("manifest", {}).get("digitalSourceType")
            )
            if dst:
                result["digital_source_types"].append(dst)
                if dst in IPTC_AI_TYPES:
                    ai_assertions.add(f"iptc:{dst}")

            tool = (
                ing.get("softwareAgent")
                or ing.get("data", {}).get("softwareAgent")
                or ing.get("manifest", {}).get("softwareAgent")
            )
            if tool and isinstance(tool, str):
                result["software_agents"].append(tool)
                if any(k in tool.lower() for k in AI_TOOL_KEYWORDS):
                    tools.add(tool)
                    ai_assertions.add("ingredient_ai_tool")

    # write results
    result["ai_assertions_found"] = sorted(ai_assertions)
    result["tools_detected"] = sorted(tools)
    result["edit_actions"] = sorted(actions)

    # --- scoring logic ---
    score = 0.0
    if ai_assertions:
        score = 1.0

    tool_text = " ".join(result["tools_detected"]).lower()
    if any(k in tool_text for k in AI_TOOL_KEYWORDS):
        score = max(score, 0.9)

    for act in actions:
        if act.lower().startswith("generate") or "synth" in act.lower():
            score = max(score, 0.8)

    result["overall_c2pa_score"] = float(score)
    return result

