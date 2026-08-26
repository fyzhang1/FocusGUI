"""
AgentCPM-GUI Evaluation Utilities.

This module provides evaluation functions for GUI Agent tasks,

Supports:
- Action type matching (click, scroll, type, press, stop, long_point)
- Exact match evaluation with bbox tolerance
- Episode-level metrics (success_rate, goal_progress)
- Multi-modal input (image, video, audio)
- Voice instruction input via TTS (Text-to-Speech)
"""

import asyncio
import enum
import hashlib
import json
import math
import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from loguru import logger as eval_logger


# =============================================================================
# TTS (Text-to-Speech) Utilities for Voice Instruction
# =============================================================================

async def _generate_tts_async(text: str, output_path: str, voice: str):
    """Async helper to generate TTS audio using edge-tts."""
    try:
        import edge_tts
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(output_path)
    except ImportError:
        eval_logger.error("edge-tts not installed. Run: pip install edge-tts")
        raise


def generate_tts_audio(text: str, cache_dir: str, voice: str = "zh-CN-XiaoxiaoNeural") -> Optional[str]:
    """
    Generate TTS audio from text using edge-tts with caching.
    
    Args:
        text: The text to convert to speech
        cache_dir: Directory to cache generated audio files
        voice: TTS voice to use (default: zh-CN-XiaoxiaoNeural for Chinese)
        
    Returns:
        Path to the generated audio file, or None if generation failed
    """
    if not text or not text.strip():
        return None
    
    os.makedirs(cache_dir, exist_ok=True)
    
    # Generate cache key based on text and voice
    text_hash = hashlib.md5(f"{text}_{voice}".encode()).hexdigest()[:16]
    cache_path = os.path.join(cache_dir, f"tts_{text_hash}.mp3")
    
    # Return cached file if exists
    if os.path.exists(cache_path):
        eval_logger.debug(f"Using cached TTS audio: {cache_path}")
        return cache_path
    
    try:
        # Run async TTS generation
        asyncio.run(_generate_tts_async(text, cache_path, voice))
        eval_logger.info(f"Generated TTS audio: {cache_path}")
        return cache_path
    except Exception as e:
        eval_logger.warning(f"TTS generation failed for text '{text[:50]}...': {e}")
        return None

try:
    import Levenshtein
    HAS_LEVENSHTEIN = True
except ImportError:
    HAS_LEVENSHTEIN = False
    eval_logger.warning("Levenshtein not installed, text similarity will use basic comparison")


# =============================================================================
# Action Type Enumeration (from AgentCPM-GUI action_type.py)
# =============================================================================

class ActionType(enum.IntEnum):
    """Integer values for each supported action type in AndroidInTheWild."""
    NONE = -1            # No action / empty action
    CLICK = 0            # Single click/tap
    DOUBLE_CLICK = 1     # Double click/tap (tap same position twice)
    LONG_POINT = 2       # Long press
    NO_ACTION = 3        # Legacy no action
    TYPE = 4             # Text input
    DUAL_POINT = 5       # Dual point (tap or swipe)
    PRESS_BACK = 6       # Press back button
    PRESS_HOME = 7       # Press home button
    PRESS_ENTER = 8      # Press enter button
    UNUSED_9 = 9
    UNUSED_10 = 10
    STATUS_TASK_COMPLETE = 11
    STATUS_TASK_IMPOSSIBLE = 12


# =============================================================================
# Constants
# =============================================================================

_TAP_DISTANCE_THRESHOLD = 0.14  # Fraction of the screen
_TAP_DISTANCE_THRESHOLD_AC = 0.04  # For android control
_SWIPE_DISTANCE_THRESHOLD = 0.04  # Interval determining if an action is a tap or a swipe
ANNOTATION_WIDTH_AUGMENT_FRACTION = 1.2
ANNOTATION_HEIGHT_AUGMENT_FRACTION = 1.2
DEFAULT_DURATION = 200

# JSON Schema for output parsing
EXTRACT_SCHEMA = {
    "properties": {
        "duration": {"default": 200},
        "STATUS": {"default": "continue"}
    }
}

# Stop status values
STOP_STATUS = ["finish", "satisfied", "impossible", "interrupt", "need_feedback"]


# =============================================================================
# Helper Functions
# =============================================================================

def _resize_annotation_bounding_boxes(
    annotation_position: Union[List[float], List[List[float]]],
    width_factor: float = 1.2,
    height_factor: float = 1.2,
) -> Union[List[float], List[List[float]]]:
    """Uniformly enlarge bbox(es) by the given factors."""
    def _resize(box: List[float]) -> List[float]:
        y, x, h, w = box
        h_delta = (height_factor - 1) * h
        w_delta = (width_factor - 1) * w
        y = max(0, y - h_delta / 2)
        x = max(0, x - w_delta / 2)
        h = min(1, h + h_delta)
        w = min(1, w + w_delta)
        return [y, x, h, w]

    if not annotation_position:
        return []
    if isinstance(annotation_position[0], list):
        return [_resize(b) for b in annotation_position]
    return _resize(annotation_position)


def is_tap_action(normalized_start_yx: List[float], normalized_end_yx: List[float]) -> bool:
    """Determine if the action is a tap (vs swipe) based on distance."""
    distance = np.linalg.norm(np.array(normalized_start_yx) - np.array(normalized_end_yx))
    return distance <= _SWIPE_DISTANCE_THRESHOLD


def check_inside(x: float, y: float, bbox_list: List[List[float]]) -> Tuple[bool, Optional[np.ndarray]]:
    """Check if point (x, y) is inside any bounding box."""
    if not bbox_list:
        return False, None
    bbox_array = np.array(bbox_list)
    y_min, x_min, height, width = bbox_array[:, 0], bbox_array[:, 1], bbox_array[:, 2], bbox_array[:, 3]
    y_max, x_max = y_min + height, x_min + width

    within_x = (x_min <= x) & (x <= x_max)
    within_y = (y_min <= y) & (y <= y_max)
    within_bbox = within_x & within_y

    if np.any(within_bbox):
        return True, bbox_array[within_bbox]
    return False, None


def obtain_gt_bbox(
    coordinate: Dict[str, float],
    bbox_list: List[List[float]],
    eval_android_control: bool = False
) -> List[List[float]]:
    """Get ground truth bounding boxes containing the coordinate."""
    x, y = coordinate['x'], coordinate['y']
    if len(bbox_list) == 0:
        return []

    if not eval_android_control:
        is_inside, bbox_inside = check_inside(x, y, bbox_list)
        if is_inside:
            return bbox_inside.tolist()
        return []
    else:
        def get_center_distance(box):
            ymin, xmin, h, w = box
            center_y = ymin + h / 2
            center_x = xmin + w / 2
            return ((center_y - y) ** 2 + (center_x - x) ** 2) ** 0.5
        sorted_boxes = sorted(bbox_list, key=get_center_distance)
        return sorted_boxes[:5]


def _get_direction(point1: Dict[str, float], point2: Dict[str, float]) -> str:
    """Calculate swipe direction from two points."""
    try:
        x1, y1 = point1["x"], point1["y"]
        x2, y2 = point2["x"], point2["y"]
        vx, vy = x2 - x1, y2 - y1
    except Exception:
        return "no direction"

    directions = {
        "up": (0, -1),
        "down": (0, 1),
        "left": (-1, 0),
        "right": (1, 0)
    }

    vector_length = math.sqrt(vx ** 2 + vy ** 2)
    if vector_length == 0:
        return "no direction"
    unit_vector = (vx / vector_length, vy / vector_length)

    max_cosine = -float('inf')
    closest_direction = None
    for direction, dir_vector in directions.items():
        dx, dy = dir_vector
        dir_length = math.sqrt(dx ** 2 + dy ** 2)
        cos_theta = (unit_vector[0] * dx + unit_vector[1] * dy) / dir_length
        if cos_theta > max_cosine:
            max_cosine = cos_theta
            closest_direction = direction

    return closest_direction


def get_direction(point: List[float], to: Union[str, List[float]]) -> str:
    """Get direction from point and 'to' parameter."""
    if isinstance(to, str):
        if to in ["up", "down", "left", "right"]:
            return to
        return "no direction"
    elif isinstance(to, list):
        try:
            point1 = {"x": point[0], "y": point[1]}
            point2 = {"x": to[0], "y": to[1]}
            return _get_direction(point1, point2)
        except Exception:
            return "no direction"
    return "no direction"


def text_similarity(pred_text: str, gt_text: str) -> float:
    """Calculate text similarity using Levenshtein ratio."""
    if HAS_LEVENSHTEIN:
        return Levenshtein.ratio(pred_text, gt_text)
    # Fallback: simple containment check
    if pred_text in gt_text or gt_text in pred_text:
        return 1.0
    return 0.0


# =============================================================================
# Output Parsing Functions
# =============================================================================

# Qwen tool_call action name → OmniActionType mapping
# Uses raw integer values to avoid forward reference to OmniActionType enum
# Values: NONE=-1, TAP=0, DOUBLE_TAP=1, LONG_PRESS=2, SWIPE_UP=3, SWIPE_DOWN=4,
#         SWIPE_LEFT=5, SWIPE_RIGHT=6, INPUT=7, BACK=8, HOME=9, 
#         TASK_COMPLETE=10, TASK_IMPOSSIBLE=11
QWEN_TOOL_CALL_ACTION_MAP = {
    "click": 0,           # TAP
    "double_click": 1,    # DOUBLE_TAP
    "long_press": 2,      # LONG_PRESS
    "type": 7,            # INPUT
    "wait": -1,           # NONE
    # "swipe", "system_button", "terminate" need special handling
}


def _convert_tool_call_to_action(tc_data: Dict) -> Optional[Dict]:
    """
    Convert Qwen <tool_call> JSON to unified action dict.
    
    Qwen tool_call format:
        {"name": "mobile_use", "arguments": {"action": "click", "coordinate": [x, y]}}
      or:
        {"name": "click", "arguments": {"coordinate": [540, 1200]}}
      or:
        {"name": "click", "arguments": {"x": 540, "y": 1200}}
    
    Maps to unified OmniActionType (-1 to 11) with coordinates preserved as-is
    (pixel or normalized — evaluation layer handles normalization).
    
    Args:
        tc_data: Parsed JSON from <tool_call> tags
        
    Returns:
        Unified action dict or None if parsing fails
    """
    # Extract action name and arguments
    # Format 1: {"name": "mobile_use", "arguments": {"action": "click", ...}}
    # Format 2: {"name": "click", "arguments": {"coordinate": [x, y]}}
    args = tc_data.get("arguments", tc_data.get("args", {}))
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            return None
    
    action_name = args.get("action", tc_data.get("name", "")).lower().strip()
    
    if not action_name:
        return None
    
    result = {}
    
    # --- Simple mapped actions ---
    if action_name in QWEN_TOOL_CALL_ACTION_MAP:
        result["action_type"] = int(QWEN_TOOL_CALL_ACTION_MAP[action_name])
        
        # Extract coordinate (various key names)
        coord = args.get("coordinate", args.get("coordinates", args.get("position", None)))
        if coord is None and "x" in args and "y" in args:
            coord = [args["x"], args["y"]]
        if coord and isinstance(coord, list) and len(coord) >= 2:
            try:
                result["coordinate"] = [int(float(coord[0])), int(float(coord[1]))]
            except (ValueError, TypeError):
                pass
        
        # Extract text for type action
        if action_name == "type":
            text = args.get("text", args.get("content", args.get("input_text", "")))
            if text:
                result["text"] = str(text)
        
        # Extract duration for wait/long_press
        if "time" in args or "duration" in args:
            try:
                t = float(args.get("time", args.get("duration", 0)))
                result["duration"] = int(t * 1000) if t < 100 else int(t)  # seconds→ms
            except (ValueError, TypeError):
                pass
    
    # --- Swipe: determine direction from coordinates ---
    elif action_name == "swipe":
        coord1 = args.get("coordinate", args.get("start_coordinate", None))
        coord2 = args.get("coordinate2", args.get("end_coordinate", args.get("to", None)))
        
        if coord1 and coord2:
            try:
                x1, y1 = int(float(coord1[0])), int(float(coord1[1]))
                x2, y2 = int(float(coord2[0])), int(float(coord2[1]))
                
                dx, dy = x2 - x1, y2 - y1
                if abs(dx) > abs(dy):
                    action_type = 6 if dx > 0 else 5  # SWIPE_RIGHT / SWIPE_LEFT
                else:
                    action_type = 4 if dy > 0 else 3  # SWIPE_DOWN / SWIPE_UP
                
                result["action_type"] = action_type
                result["coordinate"] = [x1, y1]
                result["start_coordinate"] = [x1, y1]
                result["end_coordinate"] = [x2, y2]
            except (ValueError, TypeError, IndexError):
                return None
        elif coord1:
            # Only start coordinate, try direction from args
            direction = args.get("direction", "").lower()
            dir_map = {"up": 3, "down": 4, "left": 5, "right": 6}
            result["action_type"] = dir_map.get(direction, 3)  # default SWIPE_UP
            try:
                result["coordinate"] = [int(float(coord1[0])), int(float(coord1[1]))]
                result["start_coordinate"] = result["coordinate"]
            except (ValueError, TypeError):
                pass
        else:
            return None
    
    # --- system_button: Back/Home ---
    elif action_name == "system_button":
        button = str(args.get("button", args.get("key", ""))).lower().strip()
        if button in ("back", "return"):
            result["action_type"] = 8   # BACK
        elif button in ("home", "home_screen"):
            result["action_type"] = 9   # HOME
        elif button in ("enter", "return_key"):
            result["action_type"] = 7   # INPUT
        else:
            result["action_type"] = 8   # default to BACK
    
    # --- terminate: success/failure ---
    elif action_name == "terminate":
        status = str(args.get("status", args.get("reason", "success"))).lower().strip()
        if status in ("success", "complete", "done", "finished"):
            result["action_type"] = 10  # TASK_COMPLETE
        else:
            result["action_type"] = 11  # TASK_IMPOSSIBLE
    
    else:
        # Unknown action name — try normalize_action_type_string as fallback
        mapped = normalize_action_type_string(action_name)
        if mapped is not None:
            result["action_type"] = mapped
            # Try extracting coordinate
            coord = args.get("coordinate", args.get("coordinates", None))
            if coord and isinstance(coord, list) and len(coord) >= 2:
                try:
                    result["coordinate"] = [int(float(coord[0])), int(float(coord[1]))]
                except (ValueError, TypeError):
                    pass
        else:
            return None
    
    return result if "action_type" in result else None


def _numbers_from_fragment(fragment: str) -> List[int]:
    """Extract integer-like coordinates from a damaged JSON fragment."""
    numbers = re.findall(r'-?\d+(?:\.\d+)?', fragment or "")
    result = []
    for value in numbers:
        try:
            result.append(int(float(value)))
        except ValueError:
            continue
    return result


def _coordinate_from_damaged_json(text: str, key: str) -> Optional[List[int]]:
    """
    Recover a coordinate value when generation stops inside an array.

    Qwen2.5-Omni often emits fragments such as
    `"coordinate": [168, 137` or `"coordinate": [3(890, 580)]`.
    The recovery is intentionally conservative: it only uses numbers that
    appear immediately after the requested coordinate key.
    """
    pattern = rf'"{re.escape(key)}"\s*:\s*(?:\[[^\]{{}}]*(?:\]|$)|\([^\)]*(?:\)|$)|"[^"]*")'
    match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return None

    values = _numbers_from_fragment(match.group(0).split(":", 1)[-1])
    if len(values) < 2:
        return None

    # Some damaged generations prepend a tiny template/action artifact before
    # the actual coordinate, e.g. [3(890, 580)] or [3 500, 500].
    if len(values) >= 3 and abs(values[0]) <= 11 and max(abs(v) for v in values[1:]) > 50:
        values = values[-2:]

    return [values[0], values[1]]


def parse_damaged_json_action(output_str: str) -> Optional[Dict]:
    """
    Recover a single action dict from malformed JSON-like model output.

    This is a training-free decoding guard, not a semantic heuristic: it only
    recovers fields that the model already emitted near JSON keys.
    """
    if not output_str or "action_type" not in output_str and '"action"' not in output_str:
        return None

    # Keep the first assistant turn. Template bleed-through frequently starts
    # with a literal Human marker after an unfinished JSON object.
    text = re.split(r'\n\s*Human\b|\bHuman\s*\n', output_str, maxsplit=1)[0]
    if "{" not in text:
        text = output_str

    action_type = None
    action_match = re.search(r'"action_type"\s*:\s*("?[-\w ]+"?)', text, flags=re.IGNORECASE)
    if action_match:
        raw_action = action_match.group(1).strip().strip('"')
        try:
            action_type = int(float(raw_action))
        except ValueError:
            action_type = normalize_action_type_string(raw_action)
    else:
        action_match = re.search(r'"action"\s*:\s*"([^"]+)"', text, flags=re.IGNORECASE)
        if action_match:
            action_type = normalize_action_type_string(action_match.group(1))

    if action_type is None or not (-1 <= int(action_type) <= 11):
        return None

    result = {"action_type": int(action_type)}

    for key, out_key in [
        ("coordinate", "coordinate"),
        ("point", "coordinate"),
        ("start_coordinate", "start_coordinate"),
        ("end_coordinate", "end_coordinate"),
        ("coordinate2", "end_coordinate"),
    ]:
        coord = _coordinate_from_damaged_json(text, key)
        if coord:
            result[out_key] = coord
            if out_key == "start_coordinate":
                result.setdefault("coordinate", coord)

    text_match = re.search(r'"(?:text|input_text|content|value)"\s*:\s*"([^"]*)"', text, flags=re.IGNORECASE | re.DOTALL)
    if text_match:
        result["text"] = text_match.group(1)

    return result


def parse_json_output(output_str: str) -> Optional[Dict]:
    """Parse JSON from model output string.
    
    Handles various output formats:
    - Qwen <tool_call> XML format: <tool_call>{"name": "click", "arguments": {...}}</tool_call>
    - Direct JSON: {"action_type": 0, "point": [x, y]}
    - Markdown JSON blocks: ```json {...} ```
    - Prefixed JSON: Output: {"action_type": 0}
    - Function call format: click(x, y), tap(x, y), swipe_up(x, y), etc.
    """
    # Try to find JSON in the output
    output_str = output_str.strip()
    
    # === Priority 1: Tag-based <tool_call> extraction (Qwen format) ===
    # MUST be checked before any generic JSON parsing to avoid <thought> interference
    if "<tool_call>" in output_str and "</tool_call>" in output_str:
        try:
            # Extract content between <tool_call> and </tool_call> tags
            tc_start = output_str.index("<tool_call>") + len("<tool_call>")
            tc_end = output_str.index("</tool_call>", tc_start)
            tc_json_str = output_str[tc_start:tc_end].strip()
            
            # Fix double braces {{ }} → { } (common model artifact from template learning)
            if "{{" in tc_json_str:
                tc_json_str = tc_json_str.replace("{{", "{").replace("}}", "}")
            
            tc_data = json.loads(tc_json_str)
            
            # Convert Qwen tool_call format to unified action dict
            result = _convert_tool_call_to_action(tc_data)
            if result is not None:
                return result
        except (ValueError, json.JSONDecodeError, KeyError) as e:
            eval_logger.debug(f"Failed to parse <tool_call> content: {e}")
            # Fall through to other parsing methods
    
    # Try direct JSON parse
    try:
        return json.loads(output_str)
    except json.JSONDecodeError:
        pass
    
    # Try to find JSON block in markdown
    json_patterns = [
        r'```json\s*(.*?)\s*```',
        r'```\s*(.*?)\s*```',
    ]
    
    for pattern in json_patterns:
        matches = re.findall(pattern, output_str, re.DOTALL)
        for match in matches:
            try:
                # Clean up the match
                clean_match = match.strip()
                if not clean_match.startswith('{'):
                    continue
                return json.loads(clean_match)
            except json.JSONDecodeError:
                continue
    
    # Try to find all JSON-like patterns with balanced braces
    # This handles: Output: {"action_type": -1}, or JSON at end of text
    def find_json_objects(text):
        """Find all balanced JSON object strings in text."""
        results = []
        i = 0
        while i < len(text):
            if text[i] == '{':
                depth = 1
                start = i
                i += 1
                while i < len(text) and depth > 0:
                    if text[i] == '{':
                        depth += 1
                    elif text[i] == '}':
                        depth -= 1
                    i += 1
                if depth == 0:
                    results.append(text[start:i])
            else:
                i += 1
        return results
    
    # Find and try to parse all JSON objects in the text
    json_candidates = find_json_objects(output_str)
    for candidate in json_candidates:
        try:
            parsed = json.loads(candidate)
            # Check if it looks like an action (has action_type or point)
            if isinstance(parsed, dict) and ('action_type' in parsed or 'point' in parsed or 
                                              'coordinate' in parsed or 'action' in parsed):
                return parsed
        except json.JSONDecodeError:
            continue
    
    # Try to parse function call format: action(x, y) or action(x, y, "text")
    # Examples: click(723, 736), tap(100, 200), swipe_up(300, 400), type(100, 200, "hello")
    func_patterns = [
        # action(x, y)
        r'\b(click|tap|double_click|double_tap|long_press|swipe_up|swipe_down|swipe_left|swipe_right|scroll_up|scroll_down|scroll_left|scroll_right|input|type|back|home|none)\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)',
        # action(x, y, "text") for input
        r'\b(input|type)\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*["\']([^"\']*)["\']?\s*\)',
        # action("text") for input without coordinates
        r'\b(input|type)\s*\(\s*["\']([^"\']*)["\']?\s*\)',
        # action() for no-coordinate actions
        r'\b(back|home|none|task_complete|task_impossible|wait)\s*\(\s*\)',
    ]
    
    for pattern in func_patterns:
        match = re.search(pattern, output_str, re.IGNORECASE)
        if match:
            groups = match.groups()
            action_name = groups[0].lower()
            
            # Map function names to action types
            action_map = {
                'click': 0, 'tap': 0,
                'double_click': 1, 'double_tap': 1,
                'long_press': 2,
                'swipe_up': 3, 'scroll_up': 3,
                'swipe_down': 4, 'scroll_down': 4,
                'swipe_left': 5, 'scroll_left': 5,
                'swipe_right': 6, 'scroll_right': 6,
                'input': 7, 'type': 7,
                'back': 8,
                'home': 9,
                'task_complete': 10,
                'task_impossible': 11,
                'none': -1, 'wait': -1,
            }
            
            action_type = action_map.get(action_name)
            if action_type is not None:
                result = {'action_type': action_type}
                
                # Extract coordinates if present
                if len(groups) >= 3 and groups[1] and groups[2]:
                    try:
                        x, y = int(groups[1]), int(groups[2])
                        result['point'] = [x, y]
                        result['coordinate'] = [x, y]
                    except (ValueError, TypeError):
                        pass
                
                # Extract text if present (for input/type actions)
                if len(groups) >= 4 and groups[3]:
                    result['text'] = groups[3]
                elif len(groups) == 2 and action_name in ['input', 'type']:
                    result['text'] = groups[1]
                
                return result

    return parse_damaged_json_action(output_str)


def parse_action_from_json(data: Dict) -> Tuple[Optional[Dict], Optional[Dict], Optional[str]]:
    """Parse action from JSON data according to schema."""
    if data is None:
        return None, None, None
    
    try:
        actions = {}
        parameters = {}
        status = data.get("STATUS", "continue")

        action_keys = ["POINT", "to", "PRESS", "TYPE"]
        for key in action_keys:
            if key in data:
                actions[key] = data[key]

        parameters["duration"] = data.get("duration", DEFAULT_DURATION)
        if "to" in data:
            parameters["to"] = data["to"]

        return actions, parameters, status
    except Exception as e:
        eval_logger.debug(f"Error parsing action: {e}")
        return None, None, None


def action_map(action: Dict, args: Dict, status: str) -> Tuple[Optional[str], Any]:
    """Map parsed action to action type and arguments."""
    duration = args.get('duration', DEFAULT_DURATION) if args else None

    if action is None and args is None and status is None:
        return None, {}
    elif status in STOP_STATUS:
        return "stop", {}
    elif "TYPE" in action:
        return "type", action['TYPE']
    elif "POINT" in action and "to" not in args and duration == DEFAULT_DURATION:
        return "click", action['POINT']
    elif "POINT" in action and "to" in args and duration == DEFAULT_DURATION:
        return "scroll", {"start": action['POINT'], "end": args['to']}
    elif "POINT" in action and "duration" in args and duration > DEFAULT_DURATION:
        return "long_point", {"coordinate": action['POINT'], "duration": args['duration']}
    elif "PRESS" in action:
        return "press", action['PRESS']
    elif "duration" in args:
        return "stop", args['duration']
    else:
        return None, {}


# =============================================================================
# Evaluation Core Functions
# =============================================================================

def parse_prediction(pred_output: str) -> Tuple:
    """Parse model prediction output."""
    pd_action_type, pd_action_yx, pd_action_direction, pd_action_text, pd_action_button, pd_duration = (None,) * 6

    # Parse JSON from output
    parsed = parse_json_output(pred_output)
    if parsed is None:
        return (None,) * 6

    actions, parameters, status = parse_action_from_json(parsed)
    if actions is None:
        return (None,) * 6

    pd_action_type, pd_action_args = action_map(actions, parameters, status)
    if pd_action_type is None:
        return (None,) * 6

    scale = 1000  # Coordinates are in 0-1000 range

    if pd_action_type == "click":
        try:
            pd_action_yx = {"x": pd_action_args[0] / scale, "y": pd_action_args[1] / scale}
        except Exception:
            pd_action_yx = {"x": 0.0, "y": 0.0}
    elif pd_action_type == "long_point":
        try:
            pd_action_yx = {"x": pd_action_args["coordinate"][0] / scale, "y": pd_action_args["coordinate"][1] / scale}
            pd_duration = pd_action_args.get("duration")
        except Exception:
            pd_action_yx = {"x": 0.0, "y": 0.0}
    elif pd_action_type == "scroll":
        pd_action_direction = get_direction(pd_action_args["start"], pd_action_args["end"])
    elif pd_action_type == "type":
        pd_action_text = pd_action_args
    elif pd_action_type == "press":
        pd_action_button = pd_action_args.lower() if isinstance(pd_action_args, str) else None

    return pd_action_type, pd_action_yx, pd_action_direction, pd_action_text, pd_action_button, pd_duration


def parse_ground_truth(doc: Dict) -> Tuple:
    """Parse ground truth from document."""
    gt_action_type, gt_action_yx, gt_cand_nodes = None, None, None
    gt_action_text, gt_action_direction, gt_action_button, gt_duration = None, None, None, None

    result_action_type = doc.get('result_action_type')
    if result_action_type is None:
        return (None,) * 7

    if result_action_type == ActionType.TYPE:
        gt_action_type = "type"
        gt_action_text = doc.get('result_action_text', '')
    elif result_action_type == ActionType.DUAL_POINT:
        touch_yx = doc.get('result_touch_yx', '[0, 0]')
        lift_yx = doc.get('result_lift_yx', '[0, 0]')
        
        if isinstance(touch_yx, str):
            touch_yx = json.loads(touch_yx)
        if isinstance(lift_yx, str):
            lift_yx = json.loads(lift_yx)

        if is_tap_action(touch_yx, lift_yx):
            gt_action_type = "click"
            gt_action_yx = {"y": touch_yx[0], "x": touch_yx[1]}
            ui_positions = doc.get('ui_positions', '[]')
            if isinstance(ui_positions, str):
                gt_cand_nodes = json.loads(ui_positions)
            else:
                gt_cand_nodes = ui_positions
        else:
            gt_action_type = "scroll"
            point1 = {"y": touch_yx[0], "x": touch_yx[1]}
            point2 = {"y": lift_yx[0], "x": lift_yx[1]}
            gt_action_direction = _get_direction(point1, point2)
    elif result_action_type == ActionType.LONG_POINT:
        touch_yx = doc.get('result_touch_yx', '[0, 0]')
        if isinstance(touch_yx, str):
            touch_yx = json.loads(touch_yx)
        gt_action_type = "long_point"
        gt_action_yx = {"y": touch_yx[0], "x": touch_yx[1]}
        ui_positions = doc.get('ui_positions', '[]')
        if isinstance(ui_positions, str):
            gt_cand_nodes = json.loads(ui_positions)
        else:
            gt_cand_nodes = ui_positions
        gt_duration = doc.get('duration')
    elif result_action_type == ActionType.PRESS_BACK:
        gt_action_type = "press"
        gt_action_button = "back"
    elif result_action_type == ActionType.PRESS_HOME:
        gt_action_type = "press"
        gt_action_button = "home"
    elif result_action_type == ActionType.PRESS_ENTER:
        gt_action_type = "press"
        gt_action_button = "enter"
    elif result_action_type in [ActionType.STATUS_TASK_COMPLETE, ActionType.STATUS_TASK_IMPOSSIBLE]:
        gt_action_type = "stop"
    elif result_action_type == ActionType.NO_ACTION:
        gt_action_type = "stop"
        gt_duration = doc.get('duration')

    return gt_action_type, gt_action_yx, gt_cand_nodes, gt_action_text, gt_action_button, gt_action_direction, gt_duration


def evaluate_single_step(
    doc: Dict,
    pred_output: str,
    eval_android_control: bool = False
) -> Dict:
    """Evaluate a single step prediction against ground truth."""
    # Parse ground truth
    gt_action_type, gt_action_yx, gt_cand_nodes, gt_action_text, gt_action_button, gt_action_direction, gt_duration = parse_ground_truth(doc)

    # Parse prediction
    pd_action_type, pd_action_yx, pd_action_direction, pd_action_text, pd_action_button, pd_duration = parse_prediction(pred_output)

    # Compute metrics
    hit_format = pd_action_type is not None
    type_match = pd_action_type is not None and gt_action_type == pd_action_type
    exact_match = False
    pixel_distance = None
    text_dist = None

    if type_match and pd_action_type in ["click", "long_point"]:
        if gt_cand_nodes:
            gt_cand_nodes = _resize_annotation_bounding_boxes(
                gt_cand_nodes,
                ANNOTATION_WIDTH_AUGMENT_FRACTION,
                ANNOTATION_HEIGHT_AUGMENT_FRACTION
            )
        gt_bbox = obtain_gt_bbox(gt_action_yx, gt_cand_nodes or [], eval_android_control)

        if not gt_bbox:
            y_gt, x_gt = gt_action_yx["y"], gt_action_yx["x"]
            y_pd, x_pd = pd_action_yx["y"], pd_action_yx["x"]
            distance = np.linalg.norm(np.array([x_gt, y_gt]) - np.array([x_pd, y_pd]))
            threshold = _TAP_DISTANCE_THRESHOLD_AC if eval_android_control else _TAP_DISTANCE_THRESHOLD
            exact_match = bool(distance <= threshold)
            reference_point = (x_gt, y_gt)
        else:
            reference_point = (gt_action_yx["x"], gt_action_yx["y"])
            for bbox in gt_bbox:
                ymin, xmin, height, width = bbox
                ymax, xmax = ymin + height, xmin + width
                exact_match = (ymin <= pd_action_yx["y"] <= ymax) and (xmin <= pd_action_yx["x"] <= xmax)
                if exact_match:
                    reference_point = ((xmax + xmin) / 2, (ymax + ymin) / 2)
                    break
            if not exact_match:
                y_gt, x_gt = gt_action_yx["y"], gt_action_yx["x"]
                y_pd, x_pd = pd_action_yx["y"], pd_action_yx["x"]
                distance = np.linalg.norm(np.array([x_gt, y_gt]) - np.array([x_pd, y_pd]))
                threshold = _TAP_DISTANCE_THRESHOLD_AC if eval_android_control else _TAP_DISTANCE_THRESHOLD
                exact_match = bool(distance <= threshold)

        # Calculate pixel distance in normalized space [0, 1000]
        if pd_action_yx:
            pixel_distance = np.linalg.norm(
                np.array([pd_action_yx["x"], pd_action_yx["y"]]) * 1000 -
                np.array(reference_point) * 1000
            )

    elif type_match and pd_action_type == "scroll":
        exact_match = pd_action_direction == gt_action_direction

    elif type_match and pd_action_type == "type":
        pd_text_norm = (pd_action_text or "").lower().strip()
        gt_text_norm = (gt_action_text or "").lower().strip()
        text_dist = text_similarity(pd_text_norm, gt_text_norm)
        exact_match = pd_text_norm in gt_text_norm or gt_text_norm in pd_text_norm

    elif type_match and pd_action_type == "press":
        exact_match = pd_action_button == gt_action_button

    elif type_match and pd_action_type == "stop":
        exact_match = True

    return {
        "subset": doc.get("subset", "unknown"),
        "episode_id": doc.get("episode_id", "unknown"),
        "step_id": doc.get("step_id", 0),
        "gt_action_type": gt_action_type,
        "pd_action_type": pd_action_type,
        "type_match": type_match,
        "exact_match": exact_match,
        "hit_format": hit_format,
        "pixel_distance": pixel_distance,
        "text_dist": text_dist,
    }


# =============================================================================
# Interface Functions
# =============================================================================

def agentcpm_doc_to_visual(doc: Dict) -> List:
    """
    Convert document to visual inputs.
    
    Supports:
    - Single image: returns [image]
    - Image sequence: returns list of images
    - Video + Audio: returns [video_path, audio]
    
    Handles:
    - PIL Image objects
    - Image paths (strings)
    - HuggingFace datasets Image type
    """
    from PIL import Image as PILImage
    
    visuals = []
    
    def process_image(img):
        """Process a single image, handling various input types."""
        if img is None:
            return None
        # If it's already a PIL Image
        if hasattr(img, 'convert'):
            return img.convert("RGB")
        # If it's a string path
        if isinstance(img, str):
            if os.path.exists(img):
                return PILImage.open(img).convert("RGB")
            return img  # Return path as-is for video files etc.
        # If it's a dict (HuggingFace datasets Image format)
        if isinstance(img, dict):
            if 'path' in img and img['path']:
                return PILImage.open(img['path']).convert("RGB")
            if 'bytes' in img and img['bytes']:
                import io
                return PILImage.open(io.BytesIO(img['bytes'])).convert("RGB")
        return img
    
    # Handle image field
    if "image" in doc and doc["image"] is not None:
        processed = process_image(doc["image"])
        if processed is not None:
            visuals.append(processed)
    
    # Handle images field (for sequences)
    if "images" in doc and doc["images"] is not None:
        for img in doc["images"]:
            processed = process_image(img)
            if processed is not None:
                visuals.append(processed)
    
    # Handle video field
    if "video" in doc and doc["video"] is not None:
        visuals.append(doc["video"])
    
    # Handle audio field
    if "audio" in doc and doc["audio"] is not None:
        visuals.append(doc["audio"])
    
    return visuals if visuals else []


def agentcpm_doc_to_text(doc: Dict, lmms_eval_specific_kwargs: Optional[Dict] = None) -> str:
    """
    Convert document to text prompt.
    
    Uses lmms_eval_specific_kwargs for prompt customization:
    - pre_prompt: Text before instruction
    - post_prompt: Text after instruction
    - prompt_version: "v1" (default), "v2" (omni with schema)
    - output_schema: JSON schema description for output
    """
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}
    
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
    prompt_version = lmms_eval_specific_kwargs.get("prompt_version", "v1")
    output_schema = lmms_eval_specific_kwargs.get("output_schema", "")
    
    instruction = doc.get("instruction", "")
    
    if prompt_version == "v1":
        # Basic prompt format
        prompt = f"{pre_prompt}{instruction}{post_prompt}"
    elif prompt_version == "v2":
        # Omni benchmark format with schema
        prompt = f"{pre_prompt}{instruction}{post_prompt}"
        if output_schema:
            prompt = f"{prompt}\n\n{output_schema}"
    else:
        prompt = f"{pre_prompt}{instruction}{post_prompt}"
    
    return prompt


def agentcpm_process_results(doc: Dict, results: List[str]) -> Dict:
    """
    Process model results for a single step.
    
    Args:
        doc: Document containing ground truth
        results: List of model outputs (typically one element)
    
    Returns:
        Dictionary with metric values for aggregation
    """
    pred_output = results[0] if results else ""
    
    # Evaluate the step
    eval_result = evaluate_single_step(doc, pred_output)
    
    # Return metrics for aggregation
    return {
        "type_match": eval_result,
        "exact_match": eval_result,
        "success_rate": eval_result,
        "goal_progress": eval_result,
    }


# =============================================================================
# Aggregation Functions
# =============================================================================

def agentcpm_aggregate_type_match(results: List[Dict]) -> float:
    """Aggregate type match accuracy."""
    if not results:
        return 0.0
    
    total = len(results)
    correct = sum(1 for r in results if r.get("type_match", False))
    accuracy = correct / total if total > 0 else 0.0
    
    eval_logger.info(f"Type Match: {correct}/{total} = {accuracy:.4f}")
    return accuracy


def agentcpm_aggregate_exact_match(results: List[Dict]) -> float:
    """Aggregate exact match accuracy."""
    if not results:
        return 0.0
    
    total = len(results)
    correct = sum(1 for r in results if r.get("exact_match", False))
    accuracy = correct / total if total > 0 else 0.0
    
    eval_logger.info(f"Exact Match: {correct}/{total} = {accuracy:.4f}")
    return accuracy


def agentcpm_aggregate_success_rate(results: List[Dict]) -> float:
    """
    Aggregate episode-level success rate.
    An episode is successful only if all steps have exact_match = True.
    """
    if not results:
        return 0.0
    
    # Group results by episode
    episode_results = defaultdict(list)
    for r in results:
        episode_key = f"{r.get('subset', 'unknown')}-{r.get('episode_id', 'unknown')}"
        episode_results[episode_key].append(r)
    
    # Calculate success for each episode
    successes = []
    for episode_key, ep_results in episode_results.items():
        # Sort by step_id
        ep_results.sort(key=lambda x: x.get("step_id", 0))
        # Episode is successful if all steps are exact match
        ep_success = all(r.get("exact_match", False) for r in ep_results)
        successes.append(ep_success)
    
    success_rate = sum(successes) / len(successes) if successes else 0.0
    
    eval_logger.info(f"Success Rate: {sum(successes)}/{len(successes)} = {success_rate:.4f}")
    return success_rate


def agentcpm_aggregate_goal_progress(results: List[Dict]) -> float:
    """
    Aggregate episode-level goal progress.
    Goal progress = average ratio of correct steps before first failure.
    """
    if not results:
        return 0.0
    
    # Group results by episode
    episode_results = defaultdict(list)
    for r in results:
        episode_key = f"{r.get('subset', 'unknown')}-{r.get('episode_id', 'unknown')}"
        episode_results[episode_key].append(r)
    
    # Calculate progress for each episode
    progresses = []
    for episode_key, ep_results in episode_results.items():
        # Sort by step_id
        ep_results.sort(key=lambda x: x.get("step_id", 0))
        
        # Count consecutive correct steps from the beginning
        correct_steps = 0
        for r in ep_results:
            if r.get("exact_match", False):
                correct_steps += 1
            else:
                break
        
        # Progress is ratio of correct steps
        progress = correct_steps / len(ep_results) if ep_results else 0.0
        progresses.append(progress)
    
    goal_progress = sum(progresses) / len(progresses) if progresses else 0.0
    
    eval_logger.info(f"Goal Progress: {goal_progress:.4f}")
    return goal_progress


# =============================================================================
# Additional Utility Functions for Detailed Metrics
# =============================================================================

def compute_atomic_metrics(results: List[Dict]) -> Dict:
    """
    Compute detailed atomic-level metrics by action type.
    
    Returns metrics breakdown for:
    - CLICK, TYPE, SCROLL, PRESS, STOP, LONG_POINT
    """
    recorder = {
        'total': {'count': 0, 'type_match': 0, 'exact_match': 0, 'hit': 0},
        'CLICK': {'count': 0, 'type_match': 0, 'exact_match': 0},
        'TYPE': {'count': 0, 'type_match': 0, 'exact_match': 0, 'text_dist': []},
        'SCROLL': {'count': 0, 'type_match': 0, 'exact_match': 0},
        'PRESS': {'count': 0, 'type_match': 0, 'exact_match': 0},
        'STOP': {'count': 0, 'type_match': 0, 'exact_match': 0},
        'LONG_POINT': {'count': 0, 'type_match': 0, 'exact_match': 0},
    }

    for step in results:
        recorder['total']['count'] += 1
        recorder['total']['hit'] += 1 if step.get('hit_format', False) else 0

        action_type = step.get('gt_action_type', '')
        if isinstance(action_type, str):
            action_type = action_type.upper()
        else:
            action_type = ''

        if action_type in recorder:
            recorder[action_type]['count'] += 1
            recorder[action_type]['type_match'] += 1 if step.get('type_match', False) else 0
            recorder['total']['type_match'] += 1 if step.get('type_match', False) else 0
            recorder[action_type]['exact_match'] += 1 if step.get('exact_match', False) else 0
            recorder['total']['exact_match'] += 1 if step.get('exact_match', False) else 0
            if 'text_dist' in recorder[action_type] and step.get('text_dist') is not None:
                recorder[action_type]['text_dist'].append(step['text_dist'])

    # Calculate scores
    scores = {}
    for metric_key in ['total', 'CLICK', 'LONG_POINT', 'SCROLL', 'PRESS', 'STOP', 'TYPE']:
        count = recorder[metric_key]['count']
        scores[metric_key] = {
            'count': count,
            'type_acc': round(recorder[metric_key]['type_match'] / count, 4) if count > 0 else 0,
            'exact_acc': round(recorder[metric_key]['exact_match'] / count, 4) if count > 0 else 0
        }

    # Hit rate
    if recorder['total']['count'] > 0:
        scores['total']['hit_rate'] = round(recorder['total']['hit'] / recorder['total']['count'], 4)

    # Text distance average
    if recorder['TYPE']['text_dist']:
        scores['TYPE']['text_dist_avg'] = round(
            sum(recorder['TYPE']['text_dist']) / len(recorder['TYPE']['text_dist']), 4
        )

    # Pixel distance statistics
    pixel_distances = [r['pixel_distance'] for r in results if r.get('pixel_distance') is not None]
    if pixel_distances:
        scores['median_pixel_distance'] = round(float(np.median(pixel_distances)), 4)
        filtered = [d for d in pixel_distances if d < 1e15]
        if filtered:
            scores['mean_pixel_distance'] = round(float(np.mean(filtered)), 4)

    return scores


# =============================================================================
# Local Dataset Loading Functions
# =============================================================================

def load_local_cagui_dataset(
    data_dir: str,
    split: str = "test",
    subset: str = "domestic"
) -> List[Dict]:
    """
    Load CAGUI dataset from local directory structure.
    
    Expected structure:
    data_dir/
    └── {split}/
        └── {subset}/
            └── {episode_id}/
                ├── {episode_id}.json
                ├── {episode_id}_0.jpeg
                ├── {episode_id}_1.jpeg
                └── ...
    
    Args:
        data_dir: Root directory of CAGUI dataset
        split: Data split (e.g., "test")
        subset: Subset name (e.g., "domestic")
    
    Returns:
        List of step dictionaries with loaded images
    """
    from PIL import Image as PILImage
    
    subset_dir = os.path.join(data_dir, split, subset)
    if not os.path.exists(subset_dir):
        eval_logger.warning(f"Subset directory not found: {subset_dir}")
        return []
    
    all_steps = []
    episode_dirs = [d for d in os.listdir(subset_dir) if os.path.isdir(os.path.join(subset_dir, d))]
    
    for episode_id in episode_dirs:
        episode_dir = os.path.join(subset_dir, episode_id)
        json_path = os.path.join(episode_dir, f"{episode_id}.json")
        
        if not os.path.exists(json_path):
            eval_logger.warning(f"Episode JSON not found: {json_path}")
            continue
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                episode_data = json.load(f)
            
            # episode_data is a list of steps
            for step in episode_data:
                # Ensure required fields
                step['subset'] = step.get('subset', subset)
                step['episode_id'] = step.get('episode_id', episode_id)
                
                # Load image
                image_path = step.get('image_path', '')
                if image_path:
                    full_image_path = os.path.join(data_dir, split, image_path)
                    if os.path.exists(full_image_path):
                        step['image'] = PILImage.open(full_image_path)
                        step['image_full_path'] = full_image_path
                    else:
                        # Try alternative path
                        alt_path = os.path.join(episode_dir, os.path.basename(image_path))
                        if os.path.exists(alt_path):
                            step['image'] = PILImage.open(alt_path)
                            step['image_full_path'] = alt_path
                        else:
                            eval_logger.warning(f"Image not found: {full_image_path}")
                            step['image'] = None
                
                all_steps.append(step)
                
        except Exception as e:
            eval_logger.error(f"Error loading episode {episode_id}: {e}")
            continue
    
    eval_logger.info(f"Loaded {len(all_steps)} steps from {len(episode_dirs)} episodes")
    return all_steps


def process_docs_cagui(dataset):
    """
    Process documents function.
    
    For HuggingFace datasets that need additional processing.
    """
    # If dataset is already processed, return as-is
    return dataset


# =============================================================================
# Dataset Configuration Helper
# =============================================================================

# Map of dataset names to their configurations
DATASET_CONFIGS = {
    'chinese_app_test': {
        'subset': 'domestic',
        'split': 'test',
    },
    'aitz_test': {
        'subset': ['general', 'install', 'web_shopping', 'google_apps'],
        'split': 'test',
    },
    'gui_odyssey_test': {
        'subset': 'odyssey',
        'split': 'test',
    },
    'android_control_high_test': {
        'subset': 'android_control',
        'split': 'test',
        'eval_android_control': True,
    },
    'android_control_low_test': {
        'subset': 'android_control',
        'split': 'test',
        'eval_android_control': True,
    },
}


def get_dataset_config(data_name: str) -> Dict:
    """Get configuration for a specific dataset."""
    return DATASET_CONFIGS.get(data_name, {})


# =============================================================================
# Omni Benchmark: Multi-Modal Multi-Step GUI Agent Evaluation
# =============================================================================

class OmniActionType(enum.IntEnum):
    """
    Action types for Omni Benchmark (Unified Standard).
    Supports Android GUI operations with video/audio/image inputs.
    
    This is the unified action type standard used across all models.
    Coordinates are always in [x, y] format.
    """
    NONE = -1            # No action / empty action (waiting)
    TAP = 0              # Single tap at coordinate
    DOUBLE_TAP = 1       # Double tap at coordinate (tap same position twice)
    LONG_PRESS = 2       # Long press at coordinate
    SWIPE_UP = 3         # Swipe up gesture
    SWIPE_DOWN = 4       # Swipe down gesture
    SWIPE_LEFT = 5       # Swipe left gesture
    SWIPE_RIGHT = 6      # Swipe right gesture
    INPUT = 7            # Text input (also called TYPE)
    BACK = 8             # Press back button
    HOME = 9             # Press home button
    TASK_COMPLETE = 10   # Task completed successfully
    TASK_IMPOSSIBLE = 11 # Task cannot be completed


# =============================================================================
# Action String to OmniActionType Mapping (For Tolerance Parsing)
# =============================================================================

# Mapping from various string representations to OmniActionType
# This enables tolerance parsing for different model output formats
ACTION_STRING_TO_OMNI_TYPE = {
    # NONE / Wait actions
    "none": OmniActionType.NONE,
    "wait": OmniActionType.NONE,
    "waiting": OmniActionType.NONE,
    "idle": OmniActionType.NONE,
    "no_action": OmniActionType.NONE,
    "no action": OmniActionType.NONE,
    "-1": OmniActionType.NONE,
    
    # TAP / Click actions
    "tap": OmniActionType.TAP,
    "click": OmniActionType.TAP,
    "single_tap": OmniActionType.TAP,
    "single_click": OmniActionType.TAP,
    "touch": OmniActionType.TAP,
    "press": OmniActionType.TAP,
    "0": OmniActionType.TAP,
    
    # DOUBLE_TAP actions
    "double_tap": OmniActionType.DOUBLE_TAP,
    "double_click": OmniActionType.DOUBLE_TAP,
    "doubletap": OmniActionType.DOUBLE_TAP,
    "doubleclick": OmniActionType.DOUBLE_TAP,
    "double tap": OmniActionType.DOUBLE_TAP,
    "double click": OmniActionType.DOUBLE_TAP,
    "1": OmniActionType.DOUBLE_TAP,
    
    # LONG_PRESS actions
    "long_press": OmniActionType.LONG_PRESS,
    "longpress": OmniActionType.LONG_PRESS,
    "long press": OmniActionType.LONG_PRESS,
    "long_click": OmniActionType.LONG_PRESS,
    "longclick": OmniActionType.LONG_PRESS,
    "long click": OmniActionType.LONG_PRESS,
    "hold": OmniActionType.LONG_PRESS,
    "2": OmniActionType.LONG_PRESS,
    
    # SWIPE actions
    "swipe_up": OmniActionType.SWIPE_UP,
    "swipeup": OmniActionType.SWIPE_UP,
    "swipe up": OmniActionType.SWIPE_UP,
    "scroll_up": OmniActionType.SWIPE_UP,
    "scrollup": OmniActionType.SWIPE_UP,
    "scroll up": OmniActionType.SWIPE_UP,
    "3": OmniActionType.SWIPE_UP,
    
    "swipe_down": OmniActionType.SWIPE_DOWN,
    "swipedown": OmniActionType.SWIPE_DOWN,
    "swipe down": OmniActionType.SWIPE_DOWN,
    "scroll_down": OmniActionType.SWIPE_DOWN,
    "scrolldown": OmniActionType.SWIPE_DOWN,
    "scroll down": OmniActionType.SWIPE_DOWN,
    "4": OmniActionType.SWIPE_DOWN,
    
    "swipe_left": OmniActionType.SWIPE_LEFT,
    "swipeleft": OmniActionType.SWIPE_LEFT,
    "swipe left": OmniActionType.SWIPE_LEFT,
    "scroll_left": OmniActionType.SWIPE_LEFT,
    "scrollleft": OmniActionType.SWIPE_LEFT,
    "scroll left": OmniActionType.SWIPE_LEFT,
    "5": OmniActionType.SWIPE_LEFT,
    
    "swipe_right": OmniActionType.SWIPE_RIGHT,
    "swiperight": OmniActionType.SWIPE_RIGHT,
    "swipe right": OmniActionType.SWIPE_RIGHT,
    "scroll_right": OmniActionType.SWIPE_RIGHT,
    "scrollright": OmniActionType.SWIPE_RIGHT,
    "scroll right": OmniActionType.SWIPE_RIGHT,
    "6": OmniActionType.SWIPE_RIGHT,
    
    # Generic swipe (needs coordinate analysis)
    "swipe": None,  # Will be determined by coordinates
    "scroll": None,  # Will be determined by coordinates
    
    # INPUT / Type actions
    "input": OmniActionType.INPUT,
    "type": OmniActionType.INPUT,
    "text": OmniActionType.INPUT,
    "enter_text": OmniActionType.INPUT,
    "input_text": OmniActionType.INPUT,
    "write": OmniActionType.INPUT,
    "7": OmniActionType.INPUT,
    
    # BACK action
    "back": OmniActionType.BACK,
    "go_back": OmniActionType.BACK,
    "press_back": OmniActionType.BACK,
    "navigate_back": OmniActionType.BACK,
    "8": OmniActionType.BACK,
    
    # HOME action
    "home": OmniActionType.HOME,
    "go_home": OmniActionType.HOME,
    "press_home": OmniActionType.HOME,
    "9": OmniActionType.HOME,
    
    # TASK_COMPLETE actions
    "task_complete": OmniActionType.TASK_COMPLETE,
    "complete": OmniActionType.TASK_COMPLETE,
    "done": OmniActionType.TASK_COMPLETE,
    "finish": OmniActionType.TASK_COMPLETE,
    "finished": OmniActionType.TASK_COMPLETE,
    "success": OmniActionType.TASK_COMPLETE,
    "10": OmniActionType.TASK_COMPLETE,
    
    # TASK_IMPOSSIBLE actions
    "task_impossible": OmniActionType.TASK_IMPOSSIBLE,
    "impossible": OmniActionType.TASK_IMPOSSIBLE,
    "cannot_complete": OmniActionType.TASK_IMPOSSIBLE,
    "failed": OmniActionType.TASK_IMPOSSIBLE,
    "failure": OmniActionType.TASK_IMPOSSIBLE,
    "11": OmniActionType.TASK_IMPOSSIBLE,
}


def normalize_action_type_string(action_str: str) -> Optional[int]:
    """
    Normalize action type from string to OmniActionType integer.
    
    This function provides tolerance parsing for various model output formats.
    It handles:
    - Case insensitivity (CLICK, Click, click all map to TAP)
    - Different naming conventions (click vs tap, type vs input)
    - String numbers ("0", "1", etc.)
    - With/without underscores (long_press vs longpress)
    
    Args:
        action_str: The action type string from model output
        
    Returns:
        OmniActionType integer value or None if not recognized
    """
    if action_str is None:
        return None
    
    # Convert to lowercase and strip whitespace
    normalized = str(action_str).lower().strip()
    
    # Direct lookup
    if normalized in ACTION_STRING_TO_OMNI_TYPE:
        return ACTION_STRING_TO_OMNI_TYPE[normalized]
    
    # Try without underscores
    no_underscore = normalized.replace("_", "")
    if no_underscore in ACTION_STRING_TO_OMNI_TYPE:
        return ACTION_STRING_TO_OMNI_TYPE[no_underscore]
    
    # Try with underscores instead of spaces
    with_underscore = normalized.replace(" ", "_")
    if with_underscore in ACTION_STRING_TO_OMNI_TYPE:
        return ACTION_STRING_TO_OMNI_TYPE[with_underscore]
    
    return None


def determine_swipe_direction(
    start_coord: Optional[List[float]], 
    end_coord: Optional[List[float]]
) -> Optional[int]:
    """
    Determine swipe direction from start and end coordinates.
    
    Args:
        start_coord: [x, y] starting coordinate
        end_coord: [x, y] ending coordinate
        
    Returns:
        OmniActionType for swipe direction or None
    """
    if not start_coord or not end_coord:
        return None
    
    try:
        dx = end_coord[0] - start_coord[0]
        dy = end_coord[1] - start_coord[1]
        
        # Determine primary direction based on larger delta
        if abs(dx) > abs(dy):
            # Horizontal swipe
            if dx > 0:
                return OmniActionType.SWIPE_RIGHT
            else:
                return OmniActionType.SWIPE_LEFT
        else:
            # Vertical swipe
            if dy > 0:
                return OmniActionType.SWIPE_DOWN
            else:
                return OmniActionType.SWIPE_UP
    except (TypeError, IndexError):
        return None


# =============================================================================
# Legacy ActionType to OmniActionType Conversion
# =============================================================================

def legacy_to_omni_action_type(
    legacy_type: int,
    touch_xy: Optional[List[float]] = None,
    lift_xy: Optional[List[float]] = None
) -> int:
    """
    Convert legacy ActionType to unified OmniActionType.
    
    Args:
        legacy_type: Legacy ActionType enum value
        touch_xy: Touch coordinate [x, y] (for DUAL_POINT differentiation)
        lift_xy: Lift coordinate [x, y] (for DUAL_POINT differentiation)
        
    Returns:
        OmniActionType enum value
    """
    # Direct mappings
    LEGACY_TO_OMNI = {
        ActionType.NONE: OmniActionType.NONE,
        ActionType.CLICK: OmniActionType.TAP,
        ActionType.DOUBLE_CLICK: OmniActionType.DOUBLE_TAP,
        ActionType.LONG_POINT: OmniActionType.LONG_PRESS,
        ActionType.TYPE: OmniActionType.INPUT,
        ActionType.PRESS_BACK: OmniActionType.BACK,
        ActionType.PRESS_HOME: OmniActionType.HOME,
        ActionType.PRESS_ENTER: OmniActionType.INPUT,  # Map to INPUT as it's similar
        ActionType.STATUS_TASK_COMPLETE: OmniActionType.TASK_COMPLETE,
        ActionType.STATUS_TASK_IMPOSSIBLE: OmniActionType.TASK_IMPOSSIBLE,
        ActionType.NO_ACTION: OmniActionType.TASK_COMPLETE,
    }
    
    if legacy_type in LEGACY_TO_OMNI:
        return LEGACY_TO_OMNI[legacy_type]
    
    if legacy_type == ActionType.DUAL_POINT:
        # Differentiate between TAP and SWIPE based on coordinates
        if touch_xy is None or lift_xy is None:
            return OmniActionType.TAP
        
        # Check if it's a tap (small distance) or swipe
        distance = np.sqrt(
            (lift_xy[0] - touch_xy[0]) ** 2 + 
            (lift_xy[1] - touch_xy[1]) ** 2
        )
        
        if distance <= _SWIPE_DISTANCE_THRESHOLD * 1000:  # Convert threshold to [0, 1000] space
            return OmniActionType.TAP
        
        # Determine swipe direction
        dx = lift_xy[0] - touch_xy[0]
        dy = lift_xy[1] - touch_xy[1]
        
        if abs(dx) > abs(dy):
            return OmniActionType.SWIPE_RIGHT if dx > 0 else OmniActionType.SWIPE_LEFT
        else:
            return OmniActionType.SWIPE_DOWN if dy > 0 else OmniActionType.SWIPE_UP
    
    # Default fallback
    return OmniActionType.TAP


# =============================================================================
# Coordinate Utilities: Auto-detection, Normalization, Rectangle Matching
# =============================================================================

def auto_detect_and_normalize_coordinate(
    coordinate: List[float],
    screen_width: int = 1080,
    screen_height: int = 2400,
    coordinate_type: str = "auto"
) -> List[int]:
    """
    Auto-detect coordinate type and normalize to [0, 1000] range.
    
    Coordinates are always in [x, y] format.
    
    Args:
        coordinate: Input coordinate [x, y]
        screen_width: Screen width in pixels (for pixel coordinate conversion)
        screen_height: Screen height in pixels (for pixel coordinate conversion)
        coordinate_type: "auto" (detect based on value), "pixel", or "normalized"
        
    Returns:
        Normalized coordinate [x, y] in [0, 1000] range
    """
    if not coordinate or len(coordinate) != 2:
        return [-1, -1]
    
    x, y = coordinate[0], coordinate[1]
    
    # Auto-detect mode
    if coordinate_type == "auto":
        # If any coordinate > 1000, treat as pixel coordinate
        if x > 1000 or y > 1000:
            coordinate_type = "pixel"
        else:
            coordinate_type = "normalized"
    
    # Convert pixel to normalized
    if coordinate_type == "pixel":
        norm_x = int(x / screen_width * 1000)
        norm_y = int(y / screen_height * 1000)
        return [norm_x, norm_y]
    
    # Already normalized
    return [int(x), int(y)]


def is_point_in_rect(
    pred_coord: List[float],
    gt_rect: List[List[float]]
) -> bool:
    """
    Check if a point is inside a rectangle.
    
    All coordinates are in [x, y] format.
    
    Args:
        pred_coord: Predicted coordinate [x, y]
        gt_rect: Ground truth rectangle [[x1, y1], [x2, y2]] (top-left, bottom-right)
        
    Returns:
        True if point is inside rectangle (inclusive)
    """
    if not pred_coord or len(pred_coord) != 2:
        return False
    if not gt_rect or len(gt_rect) != 2:
        return False
    if not gt_rect[0] or len(gt_rect[0]) != 2:
        return False
    if not gt_rect[1] or len(gt_rect[1]) != 2:
        return False
    
    pred_x, pred_y = pred_coord[0], pred_coord[1]
    x1, y1 = gt_rect[0][0], gt_rect[0][1]
    x2, y2 = gt_rect[1][0], gt_rect[1][1]
    
    # Ensure min/max are correct (handle reversed coordinates)
    x_min, x_max = min(x1, x2), max(x1, x2)
    y_min, y_max = min(y1, y2), max(y1, y2)
    
    return (x_min <= pred_x <= x_max) and (y_min <= pred_y <= y_max)


def parse_ground_truth_coordinate(doc: Dict) -> Dict:
    """
    Parse ground truth coordinate from document.
    
    Supports two formats:
    1. Single point: [x, y] or "[x, y]" (string)
    2. Rectangle: [[x1, y1], [x2, y2]] or "[[x1, y1], [x2, y2]]" (string)
    
    Uses field: result_touch_xy (preferred) or result_touch_yx (legacy, will swap)
    
    Args:
        doc: Document containing ground truth
        
    Returns:
        Dict with keys:
        - "type": "point" or "rect"
        - "point": [x, y] (for point type)
        - "rect": [[x1, y1], [x2, y2]] (for rect type)
        - "center": [x, y] (center point, for distance calculation)
    """
    # Try new field name first (result_touch_xy)
    coord_data = doc.get('result_touch_xy')
    is_legacy_format = False
    
    # Fallback to legacy field name (result_touch_yx)
    if coord_data is None:
        coord_data = doc.get('result_touch_yx')
        is_legacy_format = True
    
    if coord_data is None:
        return {"type": "none", "point": None, "rect": None, "center": None}
    
    # Parse string to list if needed
    if isinstance(coord_data, str):
        try:
            coord_data = json.loads(coord_data)
        except json.JSONDecodeError:
            return {"type": "none", "point": None, "rect": None, "center": None}
    
    if not isinstance(coord_data, list) or len(coord_data) == 0:
        return {"type": "none", "point": None, "rect": None, "center": None}
    
    # Detect format: point [x, y] vs rectangle [[x1, y1], [x2, y2]]
    if isinstance(coord_data[0], list):
        # Rectangle format [[x1, y1], [x2, y2]]
        if len(coord_data) >= 2 and len(coord_data[0]) == 2 and len(coord_data[1]) == 2:
            if is_legacy_format:
                # Swap y, x to x, y for legacy format
                rect = [
                    [coord_data[0][1], coord_data[0][0]],
                    [coord_data[1][1], coord_data[1][0]]
                ]
            else:
                rect = [coord_data[0], coord_data[1]]
            
            # Calculate center
            center_x = (rect[0][0] + rect[1][0]) / 2
            center_y = (rect[0][1] + rect[1][1]) / 2
            
            return {
                "type": "rect",
                "point": None,
                "rect": rect,
                "center": [center_x, center_y]
            }
    else:
        # Point format [x, y]
        if len(coord_data) == 2:
            if is_legacy_format:
                # Swap y, x to x, y for legacy format
                point = [coord_data[1], coord_data[0]]
            else:
                point = [coord_data[0], coord_data[1]]
            
            return {
                "type": "point",
                "point": point,
                "rect": None,
                "center": point
            }
    
    return {"type": "none", "point": None, "rect": None, "center": None}


def _parse_single_coordinate_field(
    doc: Dict,
    xy_field: str,
    yx_field: str
) -> Dict:
    """
    Parse a single coordinate field from document.
    
    Args:
        doc: Document containing coordinate data
        xy_field: Field name for [x, y] format (preferred)
        yx_field: Field name for [y, x] format (legacy)
        
    Returns:
        Dict with "type", "point", "rect", "center" keys
    """
    coord_data = doc.get(xy_field)
    is_legacy_format = False
    
    if coord_data is None:
        coord_data = doc.get(yx_field)
        is_legacy_format = True
    
    if coord_data is None:
        return {"type": "none", "point": None, "rect": None, "center": None}
    
    if isinstance(coord_data, str):
        # Handle empty string
        if coord_data.strip() == "":
            return {"type": "none", "point": None, "rect": None, "center": None}
        try:
            coord_data = json.loads(coord_data)
        except json.JSONDecodeError:
            return {"type": "none", "point": None, "rect": None, "center": None}
    
    if not isinstance(coord_data, list) or len(coord_data) == 0:
        return {"type": "none", "point": None, "rect": None, "center": None}
    
    # Detect format:
    # 1. Simple point: [x, y] 
    # 2. Single rectangle: [[x1, y1], [x2, y2]]
    # 3. Multiple rectangles: [[[x1,y1],[x2,y2]], [[x3,y3],[x4,y4]]] - use first rect
    
    if isinstance(coord_data[0], list):
        # Check if this is multiple rectangles format: [[[x1,y1],[x2,y2]], [[x3,y3],[x4,y4]]]
        if len(coord_data) >= 1 and isinstance(coord_data[0], list) and len(coord_data[0]) >= 2:
            first_elem = coord_data[0][0]
            if isinstance(first_elem, list):
                # This is multiple rectangles format - use the first rectangle
                first_rect = coord_data[0]
                if len(first_rect) >= 2 and len(first_rect[0]) == 2 and len(first_rect[1]) == 2:
                    if is_legacy_format:
                        rect = [[first_rect[0][1], first_rect[0][0]], [first_rect[1][1], first_rect[1][0]]]
                    else:
                        rect = [first_rect[0], first_rect[1]]
                    center_x = (rect[0][0] + rect[1][0]) / 2
                    center_y = (rect[0][1] + rect[1][1]) / 2
                    return {"type": "rect", "point": None, "rect": rect, "center": [center_x, center_y]}
            else:
                # Single rectangle format: [[x1, y1], [x2, y2]]
                if len(coord_data) >= 2 and len(coord_data[0]) == 2 and len(coord_data[1]) == 2:
                    if is_legacy_format:
                        rect = [[coord_data[0][1], coord_data[0][0]], [coord_data[1][1], coord_data[1][0]]]
                    else:
                        rect = [coord_data[0], coord_data[1]]
                    center_x = (rect[0][0] + rect[1][0]) / 2
                    center_y = (rect[0][1] + rect[1][1]) / 2
                    return {"type": "rect", "point": None, "rect": rect, "center": [center_x, center_y]}
    else:
        # Point format: [x, y]
        if len(coord_data) == 2:
            if is_legacy_format:
                point = [coord_data[1], coord_data[0]]
            else:
                point = [coord_data[0], coord_data[1]]
            return {"type": "point", "point": point, "rect": None, "center": point}
    
    return {"type": "none", "point": None, "rect": None, "center": None}


def parse_ground_truth_swipe_coordinates(doc: Dict) -> Dict:
    """
    Parse both start (touch) and end (lift) coordinates for SWIPE actions.
    
    Args:
        doc: Document containing ground truth with:
            - result_touch_xy / result_touch_yx: Start coordinate
            - result_lift_xy / result_lift_yx: End coordinate
            
    Returns:
        Dict with keys:
        - "start": Coordinate info for start point (from parse_ground_truth_coordinate)
        - "end": Coordinate info for end point
    """
    start_info = _parse_single_coordinate_field(doc, "result_touch_xy", "result_touch_yx")
    end_info = _parse_single_coordinate_field(doc, "result_lift_xy", "result_lift_yx")
    
    return {
        "start": start_info,
        "end": end_info
    }


def evaluate_swipe_coordinate_match(
    pred_start: List[float],
    pred_end: List[float],
    gt_swipe_info: Dict,
    distance_threshold: float = 140,
    screen_width: int = 1080,
    screen_height: int = 2400,
    coordinate_type: str = "auto"
) -> Tuple[bool, Optional[float], Optional[float]]:
    """
    Evaluate if predicted swipe coordinates match ground truth.
    
    Args:
        pred_start: Predicted start coordinate [x, y]
        pred_end: Predicted end coordinate [x, y]
        gt_swipe_info: Ground truth from parse_ground_truth_swipe_coordinates()
        distance_threshold: Distance threshold for matching
        screen_width: Screen width for pixel conversion
        screen_height: Screen height for pixel conversion
        coordinate_type: "auto", "pixel", or "normalized"
        
    Returns:
        Tuple of (is_match, start_distance, end_distance)
        - is_match: True if both start and end match
        - start_distance: Distance from predicted start to GT start center
        - end_distance: Distance from predicted end to GT end center
    """
    start_match, start_dist = evaluate_coordinate_match(
        pred_coord=pred_start,
        gt_info=gt_swipe_info["start"],
        distance_threshold=distance_threshold,
        screen_width=screen_width,
        screen_height=screen_height,
        coordinate_type=coordinate_type
    )
    
    end_match, end_dist = evaluate_coordinate_match(
        pred_coord=pred_end,
        gt_info=gt_swipe_info["end"],
        distance_threshold=distance_threshold,
        screen_width=screen_width,
        screen_height=screen_height,
        coordinate_type=coordinate_type
    )
    
    # Both start and end must match for swipe to be correct
    is_match = start_match and end_match
    
    return is_match, start_dist, end_dist


def evaluate_coordinate_match(
    pred_coord: List[float],
    gt_info: Dict,
    distance_threshold: float = 140,
    screen_width: int = 1080,
    screen_height: int = 2400,
    coordinate_type: str = "auto"
) -> Tuple[bool, Optional[float]]:
    """
    Evaluate if predicted coordinate matches ground truth.
    
    Args:
        pred_coord: Predicted coordinate [x, y] (may be pixel or normalized)
        gt_info: Ground truth info from parse_ground_truth_coordinate()
        distance_threshold: Distance threshold for point matching (in [0, 1000] space)
        screen_width: Screen width for pixel coordinate conversion
        screen_height: Screen height for pixel coordinate conversion
        coordinate_type: "auto", "pixel", or "normalized"
        
    Returns:
        Tuple of (is_match, distance_to_center)
    """
    if gt_info["type"] == "none" or gt_info["center"] is None:
        return False, None
    
    if not pred_coord or len(pred_coord) != 2:
        return False, None
    
    # Normalize prediction coordinate
    norm_pred = auto_detect_and_normalize_coordinate(
        pred_coord,
        screen_width=screen_width,
        screen_height=screen_height,
        coordinate_type=coordinate_type
    )
    
    if norm_pred == [-1, -1]:
        return False, None
    
    # Normalize GT coordinate to [0, 1000] space (GT is always in pixel coordinates)
    center = gt_info["center"]
    norm_center = auto_detect_and_normalize_coordinate(
        center,
        screen_width=screen_width,
        screen_height=screen_height,
        coordinate_type="pixel"  # GT coordinates are always pixel values
    )
    
    # Calculate distance to center in normalized space
    distance = np.sqrt(
        (norm_pred[0] - norm_center[0]) ** 2 +
        (norm_pred[1] - norm_center[1]) ** 2
    )
    
    # Check match based on GT type
    if gt_info["type"] == "rect":
        # Normalize rect corners to [0, 1000] space
        norm_rect = [
            auto_detect_and_normalize_coordinate(
                gt_info["rect"][0],
                screen_width=screen_width,
                screen_height=screen_height,
                coordinate_type="pixel"
            ),
            auto_detect_and_normalize_coordinate(
                gt_info["rect"][1],
                screen_width=screen_width,
                screen_height=screen_height,
                coordinate_type="pixel"
            )
        ]
        # Rectangle: check if point is inside normalized rect
        is_match = is_point_in_rect(norm_pred, norm_rect)
    else:
        # Point: check distance threshold
        is_match = (distance <= distance_threshold)
    
    return is_match, float(distance)


# =============================================================================
# Unified Prompt Template & doc_to_messages (for all models)
# =============================================================================

UNIFIED_SYSTEM_PROMPT = """You are an Android GUI Agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task.

## Output Format
Output ONLY a single JSON object. No markdown, no code blocks, no explanation.
Your ENTIRE response must be valid JSON and nothing else.
You MUST ALWAYS output exactly one JSON object. NEVER output an empty response.

## Action Space

| action_type | Name | Parameters |
|-------------|------|------------|
| -1 | NONE | (none) - Wait/observe without action |
| 0 | TAP | coordinate: [x, y] |
| 1 | DOUBLE_TAP | coordinate: [x, y] |
| 2 | LONG_PRESS | coordinate: [x, y] |
| 3 | SWIPE_UP | start_coordinate: [x, y], end_coordinate: [x, y] |
| 4 | SWIPE_DOWN | start_coordinate: [x, y], end_coordinate: [x, y] |
| 5 | SWIPE_LEFT | start_coordinate: [x, y], end_coordinate: [x, y] |
| 6 | SWIPE_RIGHT | start_coordinate: [x, y], end_coordinate: [x, y] |
| 7 | INPUT | text: "string" |
| 8 | BACK | (none) |
| 9 | HOME | (none) |
| 10 | TASK_COMPLETE | (none) |
| 11 | TASK_IMPOSSIBLE | (none) |

## Coordinate System
- Format: [x, y] where x=horizontal (left→right), y=vertical (top→bottom)
- Range: 0-1000 (normalized). [0, 0] = top-left, [1000, 1000] = bottom-right.

## Rules
- Base actions on the current screenshot (the last image in your input).
- If the task appears COMPLETE or already done, you MUST output {"action_type": 10}.
- If the task requires waiting/observing, output {"action_type": -1}.
- If the task is impossible, output {"action_type": 11}.
- Even when unsure, you MUST still output a valid JSON action. An empty response is NEVER acceptable.

## Examples
- Wait/observe: {"action_type": -1}
- Tap: {"action_type": 0, "coordinate": [500, 500]}
- Swipe down: {"action_type": 4, "start_coordinate": [500, 300], "end_coordinate": [500, 700]}
- Input text: {"action_type": 7, "text": "hello"}
- Back: {"action_type": 8}
- Task done: {"action_type": 10}
"""


def unified_doc_to_messages(doc: Dict, lmms_eval_specific_kwargs: Optional[Dict] = None) -> List[Dict]:
    """
    Unified doc_to_messages for ALL models (Gemini Flash, Gemini Pro, Qwen3 Omni).

    Uses UNIFIED_SYSTEM_PROMPT with JSON output format.
    Reads action_history from dataset_unified.json for GT history injection.

    Input structure:
    1. System prompt
    2. Single user message containing:
       - Action History text (from action_history field)
       - History Screenshot (from history_screenshot_path, step t-2)
       - Current Video (optional)
       - Current Audio (optional)
       - Current Screenshot (primary)
       - Prompt text
    """
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}

    media_dir = lmms_eval_specific_kwargs.get("media_dir", "")

    def resolve_path(path):
        if path and media_dir and not os.path.isabs(path):
            return os.path.join(media_dir, path)
        return path

    messages = []

    # 1. System message
    messages.append({
        "role": "system",
        "content": [{"type": "text", "text": UNIFIED_SYSTEM_PROMPT}]
    })

    task_description = doc.get("task_description", doc.get("instruction", ""))
    action_history = doc.get("action_history", [])
    current_step_index = doc.get("step_index", doc.get("step_id", 0))

    # 2. Build user message content
    user_content = []

    # --- History Screenshot (from step t-2) ---
    history_screenshot = doc.get("history_screenshot_path")
    if history_screenshot:
        user_content.append({"type": "text", "text": "[History Screenshot]:"})
        user_content.append({"type": "image", "url": resolve_path(history_screenshot)})

    # --- Current Video ---
    current_video = doc.get("current_video") or doc.get("video") or doc.get("video_path")
    if current_video:
        user_content.append({"type": "video", "url": resolve_path(current_video)})

    # --- Current Audio ---
    current_audio = doc.get("current_audio") or doc.get("audio") or doc.get("audio_path")
    if current_audio:
        if isinstance(current_audio, str):
            user_content.append({"type": "audio", "url": {"path": resolve_path(current_audio)}})
        else:
            user_content.append({"type": "audio", "url": current_audio})

    # --- Current Screenshot (PRIMARY) ---
    current_image = doc.get("current_image") or doc.get("image") or doc.get("image_path")
    if current_image:
        if isinstance(current_image, str):
            user_content.append({"type": "image", "url": resolve_path(current_image)})
        else:
            user_content.append({"type": "image", "url": current_image})

    # --- Prompt text with action history ---
    history_text = ""
    if action_history:
        history_lines = []
        for ah in action_history:
            history_lines.append(f"Step {ah['step_index']}: {ah['action_text']}")
        history_text = "\n".join(history_lines)

    prompt_parts = [f"Goal: {task_description}", f"Step: {current_step_index}"]
    if history_text:
        prompt_parts.append(f"Action History:\n{history_text}")
    prompt_parts.append("Analyze the current screenshot and output the next action as JSON.")

    user_content.append({"type": "text", "text": "\n\n".join(prompt_parts)})

    messages.append({"role": "user", "content": user_content})

    return messages


# =============================================================================
# V2 Main Experiment: Corrected System Prompt + Language-aware Instruction
# =============================================================================
# Changes from V1 (unified_doc_to_messages):
#   1. System prompt: "with screenshots" → "with screenshots, audio and video"
#   2. English apps use instruction_en, Chinese apps use instruction (Chinese)

ENGLISH_APPS = {"amazon", "instagram", "snapchat", "spotify", "x", "youtube", "tiktok", "duolingo", "googletranslate", "imdb", "ted", "vimeo", "redbulltv", "tasty"}

UNIFIED_V2_SYSTEM_PROMPT = """You are an Android GUI Agent. You are given a task and your action history, with screenshots, audio and video. You need to perform the next action to complete the task.

## Output Format
Output ONLY a single JSON object. No markdown, no code blocks, no explanation.
Your ENTIRE response must be valid JSON and nothing else.
You MUST ALWAYS output exactly one JSON object. NEVER output an empty response.

## Action Space

| action_type | Name | Parameters |
|-------------|------|------------|
| -1 | NONE | (none) - Wait/observe without action |
| 0 | TAP | coordinate: [x, y] |
| 1 | DOUBLE_TAP | coordinate: [x, y] |
| 2 | LONG_PRESS | coordinate: [x, y] |
| 3 | SWIPE_UP | start_coordinate: [x, y], end_coordinate: [x, y] |
| 4 | SWIPE_DOWN | start_coordinate: [x, y], end_coordinate: [x, y] |
| 5 | SWIPE_LEFT | start_coordinate: [x, y], end_coordinate: [x, y] |
| 6 | SWIPE_RIGHT | start_coordinate: [x, y], end_coordinate: [x, y] |
| 7 | INPUT | text: "string" |
| 8 | BACK | (none) |
| 9 | HOME | (none) |
| 10 | TASK_COMPLETE | (none) |
| 11 | TASK_IMPOSSIBLE | (none) |

## Coordinate System
- Format: [x, y] where x=horizontal (left→right), y=vertical (top→bottom)
- Range: 0-1000 (normalized). [0, 0] = top-left, [1000, 1000] = bottom-right.

## Rules
- Base actions on the current screenshot (the last image in your input).
- If the task appears COMPLETE or already done, you MUST output {"action_type": 10}.
- If the task requires waiting/observing, output {"action_type": -1}.
- If the task is impossible, output {"action_type": 11}.
- Even when unsure, you MUST still output a valid JSON action. An empty response is NEVER acceptable.

## Examples
- Wait/observe: {"action_type": -1}
- Tap: {"action_type": 0, "coordinate": [500, 500]}
- Swipe down: {"action_type": 4, "start_coordinate": [500, 300], "end_coordinate": [500, 700]}
- Input text: {"action_type": 7, "text": "hello"}
- Back: {"action_type": 8}
- Task done: {"action_type": 10}
"""


def unified_v2_doc_to_messages(doc: Dict, lmms_eval_specific_kwargs: Optional[Dict] = None) -> List[Dict]:
    """
    V2 Unified doc_to_messages for ALL models.

    Changes from V1:
    1. Uses UNIFIED_V2_SYSTEM_PROMPT (mentions screenshots, audio and video)
    2. English apps use instruction_en, Chinese apps use instruction
    """
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}

    media_dir = lmms_eval_specific_kwargs.get("media_dir", "")

    def resolve_path(path):
        if path and media_dir and not os.path.isabs(path):
            return os.path.join(media_dir, path)
        return path

    messages = []

    # 1. System message
    messages.append({
        "role": "system",
        "content": [{"type": "text", "text": UNIFIED_V2_SYSTEM_PROMPT}]
    })

    # 2. Select instruction based on app language
    app = doc.get("app", "")
    if app in ENGLISH_APPS:
        task_description = doc.get("instruction_en", doc.get("instruction", ""))
    else:
        task_description = doc.get("instruction", "")

    action_history = doc.get("action_history", [])
    current_step_index = doc.get("step_index", doc.get("step_id", 0))

    # 3. Build user message content
    user_content = []

    # --- History Screenshot (from step t-2) ---
    history_screenshot = doc.get("history_screenshot_path")
    if history_screenshot:
        user_content.append({"type": "text", "text": "[History Screenshot]:"})
        user_content.append({"type": "image", "url": resolve_path(history_screenshot)})

    # --- Current Video ---
    current_video = doc.get("current_video") or doc.get("video") or doc.get("video_path")
    if current_video:
        user_content.append({"type": "video", "url": resolve_path(current_video)})

    # --- Current Audio ---
    current_audio = doc.get("current_audio") or doc.get("audio") or doc.get("audio_path")
    if current_audio:
        if isinstance(current_audio, str):
            user_content.append({"type": "audio", "url": {"path": resolve_path(current_audio)}})
        else:
            user_content.append({"type": "audio", "url": current_audio})

    # --- Current Screenshot (PRIMARY) ---
    current_image = doc.get("current_image") or doc.get("image") or doc.get("image_path")
    if current_image:
        if isinstance(current_image, str):
            user_content.append({"type": "image", "url": resolve_path(current_image)})
        else:
            user_content.append({"type": "image", "url": current_image})

    # --- Prompt text with action history ---
    history_text = ""
    if action_history:
        history_lines = []
        for ah in action_history:
            history_lines.append(f"Step {ah['step_index']}: {ah['action_text']}")
        history_text = "\n".join(history_lines)

    prompt_parts = [f"Goal: {task_description}", f"Step: {current_step_index}"]
    if history_text:
        prompt_parts.append(f"Action History:\n{history_text}")
    prompt_parts.append("Analyze the current screenshot and output the next action as JSON.")

    user_content.append({"type": "text", "text": "\n\n".join(prompt_parts)})

    messages.append({"role": "user", "content": user_content})

    return messages


def unified_v2_gui_grounding_doc_to_messages(doc: Dict, lmms_eval_specific_kwargs: Optional[Dict] = None) -> List[Dict]:
    """Unified V2 prompt with training-free screenshot grounding evidence.

    The evidence is precomputed from the screenshot only and stored in
    ``doc["gui_grounding_evidence"]``.  It is deliberately treated as auxiliary
    observation text; the model still has to predict the full OmniGUI action
    from all modalities.
    """
    messages = unified_v2_doc_to_messages(doc, lmms_eval_specific_kwargs)
    evidence = str(doc.get("gui_grounding_evidence", "")).strip()
    if not evidence:
        return messages
    for item in reversed(messages[-1].get("content", [])):
        if isinstance(item, dict) and item.get("type") == "text":
            item["text"] = (
                item.get("text", "")
                + "\n\nScreenshot-only GUI grounding evidence:\n"
                + evidence
                + "\nUse this evidence only when it is relevant; audio, video, and history may still determine the next action."
            )
            break
    return messages


# =============================================================================
# Ablation Experiment: System Prompts and doc_to_messages Variants
# =============================================================================
# Three ablation groups (history info always preserved):
#   1. No Audio: remove audio input, keep video + screenshot
#   2. No Video: remove video input, keep audio + screenshot
#   3. No Audio + No Video: remove both, keep only screenshot

ABLATION_NO_AUDIO_SYSTEM_PROMPT = UNIFIED_SYSTEM_PROMPT.replace(
    "with screenshots.", "with screenshots and video."
)

ABLATION_NO_VIDEO_SYSTEM_PROMPT = UNIFIED_SYSTEM_PROMPT.replace(
    "with screenshots.", "with screenshots and audio."
)

ABLATION_NO_AV_SYSTEM_PROMPT = UNIFIED_SYSTEM_PROMPT  # Already says "with screenshots"


def _ablation_doc_to_messages_impl(doc: Dict, lmms_eval_specific_kwargs: Optional[Dict],
                                    system_prompt: str, include_audio: bool = True,
                                    include_video: bool = True) -> List[Dict]:
    """
    Parameterized doc_to_messages for ablation experiments.
    Same as unified_doc_to_messages but with configurable media skipping.
    History info (action_history + history_screenshot) is always preserved.
    """
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}

    media_dir = lmms_eval_specific_kwargs.get("media_dir", "")

    def resolve_path(path):
        if path and media_dir and not os.path.isabs(path):
            return os.path.join(media_dir, path)
        return path

    messages = []

    # 1. System message
    messages.append({
        "role": "system",
        "content": [{"type": "text", "text": system_prompt}]
    })

    task_description = doc.get("task_description", doc.get("instruction", ""))
    action_history = doc.get("action_history", [])
    current_step_index = doc.get("step_index", doc.get("step_id", 0))

    # 2. Build user message content
    user_content = []

    # --- History Screenshot (from step t-2, always included) ---
    history_screenshot = doc.get("history_screenshot_path")
    if history_screenshot:
        user_content.append({"type": "text", "text": "[History Screenshot]:"})
        user_content.append({"type": "image", "url": resolve_path(history_screenshot)})

    # --- Current Video (conditionally included) ---
    if include_video:
        current_video = doc.get("current_video") or doc.get("video") or doc.get("video_path")
        if current_video:
            user_content.append({"type": "video", "url": resolve_path(current_video)})

    # --- Current Audio (conditionally included) ---
    if include_audio:
        current_audio = doc.get("current_audio") or doc.get("audio") or doc.get("audio_path")
        if current_audio:
            if isinstance(current_audio, str):
                user_content.append({"type": "audio", "url": {"path": resolve_path(current_audio)}})
            else:
                user_content.append({"type": "audio", "url": current_audio})

    # --- Current Screenshot (PRIMARY, always included) ---
    current_image = doc.get("current_image") or doc.get("image") or doc.get("image_path")
    if current_image:
        if isinstance(current_image, str):
            user_content.append({"type": "image", "url": resolve_path(current_image)})
        else:
            user_content.append({"type": "image", "url": current_image})

    # --- Prompt text with action history (always included) ---
    history_text = ""
    if action_history:
        history_lines = []
        for ah in action_history:
            history_lines.append(f"Step {ah['step_index']}: {ah['action_text']}")
        history_text = "\n".join(history_lines)

    prompt_parts = [f"Goal: {task_description}", f"Step: {current_step_index}"]
    if history_text:
        prompt_parts.append(f"Action History:\n{history_text}")
    prompt_parts.append("Analyze the current screenshot and output the next action as JSON.")

    user_content.append({"type": "text", "text": "\n\n".join(prompt_parts)})

    messages.append({"role": "user", "content": user_content})

    return messages


def ablation_no_audio_doc_to_messages(doc: Dict, lmms_eval_specific_kwargs: Optional[Dict] = None) -> List[Dict]:
    """Ablation 1: No audio input. Keep video + screenshot + history."""
    return _ablation_doc_to_messages_impl(
        doc, lmms_eval_specific_kwargs,
        system_prompt=ABLATION_NO_AUDIO_SYSTEM_PROMPT,
        include_audio=False, include_video=True
    )


def ablation_no_video_doc_to_messages(doc: Dict, lmms_eval_specific_kwargs: Optional[Dict] = None) -> List[Dict]:
    """Ablation 2: No video input. Keep audio + screenshot + history."""
    return _ablation_doc_to_messages_impl(
        doc, lmms_eval_specific_kwargs,
        system_prompt=ABLATION_NO_VIDEO_SYSTEM_PROMPT,
        include_audio=True, include_video=False
    )


def ablation_no_av_doc_to_messages(doc: Dict, lmms_eval_specific_kwargs: Optional[Dict] = None) -> List[Dict]:
    """Ablation 3: No audio and no video input. Keep only screenshot + history."""
    return _ablation_doc_to_messages_impl(
        doc, lmms_eval_specific_kwargs,
        system_prompt=ABLATION_NO_AV_SYSTEM_PROMPT,
        include_audio=False, include_video=False
    )


# =============================================================================
# Ablation V2: Language-aware instruction for English apps
# =============================================================================
# Same system prompts and media logic as original ablation, but uses
# instruction_en for English apps instead of instruction (Chinese).

def _ablation_v2_doc_to_messages_impl(doc: Dict, lmms_eval_specific_kwargs: Optional[Dict],
                                       system_prompt: str, include_audio: bool = True,
                                       include_video: bool = True) -> List[Dict]:
    """
    Same as _ablation_doc_to_messages_impl but with language-aware instruction.
    English apps use instruction_en, Chinese apps use instruction.
    """
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}

    media_dir = lmms_eval_specific_kwargs.get("media_dir", "")

    def resolve_path(path):
        if path and media_dir and not os.path.isabs(path):
            return os.path.join(media_dir, path)
        return path

    messages = []

    # 1. System message
    messages.append({
        "role": "system",
        "content": [{"type": "text", "text": system_prompt}]
    })

    # 2. Language-aware instruction selection
    app = doc.get("app", "")
    if app in ENGLISH_APPS:
        task_description = doc.get("instruction_en", doc.get("instruction", ""))
    else:
        task_description = doc.get("instruction", "")

    action_history = doc.get("action_history", [])
    current_step_index = doc.get("step_index", doc.get("step_id", 0))

    # 3. Build user message content
    user_content = []

    # --- History Screenshot (always included) ---
    history_screenshot = doc.get("history_screenshot_path")
    if history_screenshot:
        user_content.append({"type": "text", "text": "[History Screenshot]:"})
        user_content.append({"type": "image", "url": resolve_path(history_screenshot)})

    # --- Current Video (conditionally included) ---
    if include_video:
        current_video = doc.get("current_video") or doc.get("video") or doc.get("video_path")
        if current_video:
            user_content.append({"type": "video", "url": resolve_path(current_video)})

    # --- Current Audio (conditionally included) ---
    if include_audio:
        current_audio = doc.get("current_audio") or doc.get("audio") or doc.get("audio_path")
        if current_audio:
            if isinstance(current_audio, str):
                user_content.append({"type": "audio", "url": {"path": resolve_path(current_audio)}})
            else:
                user_content.append({"type": "audio", "url": current_audio})

    # --- Current Screenshot (PRIMARY, always included) ---
    current_image = doc.get("current_image") or doc.get("image") or doc.get("image_path")
    if current_image:
        if isinstance(current_image, str):
            user_content.append({"type": "image", "url": resolve_path(current_image)})
        else:
            user_content.append({"type": "image", "url": current_image})

    # --- Prompt text with action history ---
    history_text = ""
    if action_history:
        history_lines = []
        for ah in action_history:
            history_lines.append(f"Step {ah['step_index']}: {ah['action_text']}")
        history_text = "\n".join(history_lines)

    prompt_parts = [f"Goal: {task_description}", f"Step: {current_step_index}"]
    if history_text:
        prompt_parts.append(f"Action History:\n{history_text}")
    prompt_parts.append("Analyze the current screenshot and output the next action as JSON.")

    user_content.append({"type": "text", "text": "\n\n".join(prompt_parts)})

    messages.append({"role": "user", "content": user_content})

    return messages


def ablation_v2_no_audio_doc_to_messages(doc: Dict, lmms_eval_specific_kwargs: Optional[Dict] = None) -> List[Dict]:
    """Ablation V2 - No audio: same prompt, but instruction_en for English apps."""
    return _ablation_v2_doc_to_messages_impl(
        doc, lmms_eval_specific_kwargs,
        system_prompt=ABLATION_NO_AUDIO_SYSTEM_PROMPT,
        include_audio=False, include_video=True
    )


def ablation_v2_no_video_doc_to_messages(doc: Dict, lmms_eval_specific_kwargs: Optional[Dict] = None) -> List[Dict]:
    """Ablation V2 - No video: same prompt, but instruction_en for English apps."""
    return _ablation_v2_doc_to_messages_impl(
        doc, lmms_eval_specific_kwargs,
        system_prompt=ABLATION_NO_VIDEO_SYSTEM_PROMPT,
        include_audio=True, include_video=False
    )


def ablation_v2_no_av_doc_to_messages(doc: Dict, lmms_eval_specific_kwargs: Optional[Dict] = None) -> List[Dict]:
    """Ablation V2 - No AV: same prompt, but instruction_en for English apps."""
    return _ablation_v2_doc_to_messages_impl(
        doc, lmms_eval_specific_kwargs,
        system_prompt=ABLATION_NO_AV_SYSTEM_PROMPT,
        include_audio=False, include_video=False
    )


# =============================================================================
# TTS Ablation: Replace text instruction with TTS audio instruction
# =============================================================================
# The task instruction is converted to speech (TTS) and provided as audio
# instead of text. This tests whether models can follow voice instructions.
# All other inputs (screenshot, video, environment audio) remain unchanged.
# Uses pre-generated TTS cache from batch_tts_*.py scripts.
#
# TTS cache format: tts_cache/{md5(instruction_text)}.wav
# English apps: md5(instruction_en), Chinese apps: md5(instruction)

TTS_ABLATION_SYSTEM_PROMPT = """You are an Android GUI Agent. You are given a task via voice instruction, along with your action history, screenshots, audio and video. You need to listen to the voice instruction and perform the next action to complete the task.

## Output Format
Output ONLY a single JSON object. No markdown, no code blocks, no explanation.
Your ENTIRE response must be valid JSON and nothing else.
You MUST ALWAYS output exactly one JSON object. NEVER output an empty response.

## Action Space

| action_type | Name | Parameters |
|-------------|------|------------|
| -1 | NONE | (none) - Wait/observe without action |
| 0 | TAP | coordinate: [x, y] |
| 1 | DOUBLE_TAP | coordinate: [x, y] |
| 2 | LONG_PRESS | coordinate: [x, y] |
| 3 | SWIPE_UP | start_coordinate: [x, y], end_coordinate: [x, y] |
| 4 | SWIPE_DOWN | start_coordinate: [x, y], end_coordinate: [x, y] |
| 5 | SWIPE_LEFT | start_coordinate: [x, y], end_coordinate: [x, y] |
| 6 | SWIPE_RIGHT | start_coordinate: [x, y], end_coordinate: [x, y] |
| 7 | INPUT | text: "string" |
| 8 | BACK | (none) |
| 9 | HOME | (none) |
| 10 | TASK_COMPLETE | (none) |
| 11 | TASK_IMPOSSIBLE | (none) |

## Coordinate System
- Format: [x, y] where x=horizontal (left→right), y=vertical (top→bottom)
- Range: 0-1000 (normalized). [0, 0] = top-left, [1000, 1000] = bottom-right.

## Rules
- Base actions on the current screenshot (the last image in your input).
- Listen to the voice instruction audio for the task description.
- If the task appears COMPLETE or already done, you MUST output {"action_type": 10}.
- If the task requires waiting/observing, output {"action_type": -1}.
- If the task is impossible, output {"action_type": 11}.
- Even when unsure, you MUST still output a valid JSON action. An empty response is NEVER acceptable.

## Examples
- Wait/observe: {"action_type": -1}
- Tap: {"action_type": 0, "coordinate": [500, 500]}
- Swipe down: {"action_type": 4, "start_coordinate": [500, 300], "end_coordinate": [500, 700]}
- Input text: {"action_type": 7, "text": "hello"}
- Back: {"action_type": 8}
- Task done: {"action_type": 10}
"""


def _get_tts_cache_path(text: str, cache_dir: str) -> Optional[str]:
    """
    Get the TTS cache file path for a given instruction text.
    Uses the same hash format as batch_tts_*.py scripts: md5(text).wav
    """
    text_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
    cache_path = os.path.join(cache_dir, f"{text_hash}.wav")
    if os.path.exists(cache_path) and os.path.getsize(cache_path) > 1000:
        return cache_path
    return None


def tts_ablation_doc_to_messages(doc: Dict, lmms_eval_specific_kwargs: Optional[Dict] = None) -> List[Dict]:
    """
    TTS Ablation: Replace text instruction with TTS audio.
    
    Same as unified_v2_doc_to_messages but:
    1. System prompt mentions voice instruction
    2. Task instruction is provided as TTS audio (not text)
    3. Text prompt uses placeholder "[Voice Instruction]" instead of actual instruction
    4. All other media (video, env audio, screenshot, history) preserved
    
    Requires lmms_eval_specific_kwargs:
        - tts_cache_dir: path to TTS cache directory (default: "tts_cache")
        - media_dir: base path for media files
    """
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}

    media_dir = lmms_eval_specific_kwargs.get("media_dir", "")
    tts_cache_dir = lmms_eval_specific_kwargs.get("tts_cache_dir", "tts_cache")

    def resolve_path(path):
        if path and media_dir and not os.path.isabs(path):
            return os.path.join(media_dir, path)
        return path

    messages = []

    # 1. System message
    messages.append({
        "role": "system",
        "content": [{"type": "text", "text": TTS_ABLATION_SYSTEM_PROMPT}]
    })

    # 2. Select instruction based on app language (for TTS cache lookup)
    app = doc.get("app", "")
    if app in ENGLISH_APPS:
        task_description = doc.get("instruction_en", doc.get("instruction", ""))
    else:
        task_description = doc.get("instruction", "")

    action_history = doc.get("action_history", [])
    current_step_index = doc.get("step_index", doc.get("step_id", 0))

    # 3. Build user message content
    user_content = []

    # --- History Screenshot (always included) ---
    history_screenshot = doc.get("history_screenshot_path")
    if history_screenshot:
        user_content.append({"type": "text", "text": "[History Screenshot]:"})
        user_content.append({"type": "image", "url": resolve_path(history_screenshot)})

    # --- Current Video ---
    current_video = doc.get("current_video") or doc.get("video") or doc.get("video_path")
    if current_video:
        user_content.append({"type": "video", "url": resolve_path(current_video)})

    # --- Current Audio (environment audio, not TTS) ---
    current_audio = doc.get("current_audio") or doc.get("audio") or doc.get("audio_path")
    if current_audio:
        if isinstance(current_audio, str):
            user_content.append({"type": "audio", "url": {"path": resolve_path(current_audio)}})
        else:
            user_content.append({"type": "audio", "url": current_audio})

    # --- Current Screenshot (PRIMARY) ---
    current_image = doc.get("current_image") or doc.get("image") or doc.get("image_path")
    if current_image:
        if isinstance(current_image, str):
            user_content.append({"type": "image", "url": resolve_path(current_image)})
        else:
            user_content.append({"type": "image", "url": current_image})

    # --- TTS Voice Instruction Audio ---
    tts_audio_path = _get_tts_cache_path(task_description, tts_cache_dir)
    if tts_audio_path:
        user_content.append({"type": "audio", "url": {"path": tts_audio_path}})
        instruction_display = "[请听语音指令 / Please listen to the voice instruction]"
    else:
        # Fallback to text instruction if TTS not available
        eval_logger.warning(f"TTS cache miss for doc {doc.get('ID', 'unknown')}: {task_description[:50]}...")
        instruction_display = task_description

    # --- Prompt text with action history ---
    history_text = ""
    if action_history:
        history_lines = []
        for ah in action_history:
            history_lines.append(f"Step {ah['step_index']}: {ah['action_text']}")
        history_text = "\n".join(history_lines)

    prompt_parts = [f"Goal: {instruction_display}", f"Step: {current_step_index}"]
    if history_text:
        prompt_parts.append(f"Action History:\n{history_text}")
    prompt_parts.append("Analyze the current screenshot and output the next action as JSON.")

    user_content.append({"type": "text", "text": "\n\n".join(prompt_parts)})

    messages.append({"role": "user", "content": user_content})

    return messages


# Omni Action Type Schema for System Prompt
OMNI_ACTION_SCHEMA = """
## Action Types

| action_type | Name | Parameters |
|-------------|------|------------|
| -1 | NONE | (none) - Wait/observe without action |
| 0 | TAP | coordinate: [x, y] |
| 1 | DOUBLE_TAP | coordinate: [x, y] |
| 2 | LONG_PRESS | coordinate: [x, y] |
| 3 | SWIPE_UP | start_coordinate, end_coordinate |
| 4 | SWIPE_DOWN | start_coordinate, end_coordinate |
| 5 | SWIPE_LEFT | start_coordinate, end_coordinate |
| 6 | SWIPE_RIGHT | start_coordinate, end_coordinate |
| 7 | INPUT | text: str |
| 8 | BACK | (none) |
| 9 | HOME | (none) |
| 10 | TASK_COMPLETE | (none) |
| 11 | TASK_IMPOSSIBLE | (none) |

## When to use NONE (action_type: -1)
Use NONE when:
- The task requires WAITING for something to happen (e.g., waiting for video to play, animation to finish)
- You need to OBSERVE the screen without interacting
- The current moment is NOT the right time to act
- You're monitoring for a specific event or condition

## Coordinates: [x, y] format, range [0, 1000]

## Output: JSON only, no markdown, no explanation
Examples:
- Wait/observe: {"action_type": -1}
- Tap: {"action_type": 0, "coordinate": [300, 500]}
- Input: {"action_type": 7, "text": "hello"}
- Back: {"action_type": 8}
"""

OMNI_SYSTEM_PROMPT = """You are an Android GUI Agent. Analyze the video, audio, and screenshot to decide the next action.

""" + OMNI_ACTION_SCHEMA + """

CRITICAL RULES:
- You MUST ALWAYS output exactly one JSON object. NEVER output an empty response.
- Only output JSON. No markdown, no code blocks, no explanation, no analysis text.
- Your ENTIRE response must be valid JSON and nothing else.
- If the task says "wait", "observe", or requires timing (e.g., "wait for X to happen"), output {"action_type": -1}
- If the task appears to be COMPLETE or already done, you MUST output {"action_type": 10}
- If the task is impossible, output {"action_type": 11}
- Even when unsure, you MUST still output a valid JSON action. An empty response is NEVER acceptable.
"""

OMNI_USER_PROMPT_TEMPLATE = """Task: {task_description}

Step: {step_index}

注意：如果任务要求"等待"、"观察"或"观看"某事发生再行动，当前步骤请输出 {{"action_type": -1}}。
Note: If the task requires waiting/observing before acting, output {{"action_type": -1}} for this step."""

# Voice instruction prompt template (used when TTS is enabled, task description is provided via audio)
OMNI_VOICE_INSTRUCTION_PROMPT_TEMPLATE = """Task: [请听语音指令 / Please listen to the voice instruction]

Step: {step_index}

注意：如果任务要求"等待"、"观察"或"观看"某事发生再行动，当前步骤请输出 {{"action_type": -1}}。
Note: If the task requires waiting/observing before acting, output {{"action_type": -1}} for this step."""

OMNI_HISTORY_USER_TEMPLATE = """Task: {task_description}

Step {step_index}: Analyze the screen and determine the action."""

OMNI_HISTORY_ASSISTANT_TEMPLATE = """{action_json}"""


# =============================================================================
# Gemini Model Prompt Templates
# =============================================================================

# JSON Schema for Gemini response_schema (optional, for structured output)
GEMINI_ACTION_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "action_type": {
            "type": "integer",
            "description": "Action type: -1=NONE, 0=TAP, 1=DOUBLE_TAP, 2=LONG_PRESS, 3=SWIPE_UP, 4=SWIPE_DOWN, 5=SWIPE_LEFT, 6=SWIPE_RIGHT, 7=INPUT, 8=BACK, 9=HOME, 10=TASK_COMPLETE, 11=TASK_IMPOSSIBLE"
        },
        "coordinate": {
            "type": "array",
            "items": {"type": "integer"},
            "description": "Coordinate [x, y] for TAP/LONG_PRESS actions"
        },
        "start_coordinate": {
            "type": "array", 
            "items": {"type": "integer"},
            "description": "Start coordinate [x, y] for SWIPE actions"
        },
        "end_coordinate": {
            "type": "array",
            "items": {"type": "integer"},
            "description": "End coordinate [x, y] for SWIPE actions"
        },
        "text": {
            "type": "string",
            "description": "Text content for INPUT action"
        }
    },
    "required": ["action_type"]
}

GEMINI_ACTION_SCHEMA = """
## Action Types (Unified Standard)

| action_type | Name | Parameters |
|-------------|------|------------|
| -1 | NONE | (none) - Wait/observe without action |
| 0 | TAP | coordinate: [x, y] |
| 1 | DOUBLE_TAP | coordinate: [x, y] |
| 2 | LONG_PRESS | coordinate: [x, y] |
| 3 | SWIPE_UP | start_coordinate: [x, y], end_coordinate: [x, y] |
| 4 | SWIPE_DOWN | start_coordinate: [x, y], end_coordinate: [x, y] |
| 5 | SWIPE_LEFT | start_coordinate: [x, y], end_coordinate: [x, y] |
| 6 | SWIPE_RIGHT | start_coordinate: [x, y], end_coordinate: [x, y] |
| 7 | INPUT | text: "string" |
| 8 | BACK | (none) |
| 9 | HOME | (none) |
| 10 | TASK_COMPLETE | (none) |
| 11 | TASK_IMPOSSIBLE | (none) |

## Coordinate System
- Format: [x, y] where x=horizontal (left→right), y=vertical (top→bottom)
- Range: 0-1000 (normalized). Output coordinates MUST be in [0, 1000] range.
- Origin [0, 0]: top-left corner; [1000, 1000]: bottom-right corner
- IMPORTANT: Format is [x, y], NOT [y, x]. x is horizontal, y is vertical.

## When to use NONE (action_type: -1)
Use NONE when:
- The task requires WAITING for something to happen
- You need to OBSERVE the screen without interacting
- The current moment is NOT the right time to act
"""

GEMINI_SYSTEM_PROMPT = """You are an Android GUI automation agent. Analyze the screen and output the next action.

Your inputs:
1. Screenshot (PNG) - current screen state (PRIMARY)
2. Video (optional) - interaction history  
3. Audio (optional) - device audio

""" + GEMINI_ACTION_SCHEMA + """

CRITICAL RULES:
- You MUST ALWAYS output exactly one JSON object. NEVER output an empty response.
- Output ONLY a single JSON object. No markdown, no code blocks, no explanation, no analysis text.
- Do NOT wrap JSON in ```json``` code blocks.
- Your ENTIRE response must be valid JSON and nothing else.
- Even when you are unsure or the task seems trivial, you MUST still output a valid JSON action.
- An empty response is NEVER acceptable under any circumstances.
- Base actions on the current screenshot
- For SWIPE: provide both start_coordinate and end_coordinate
- Coordinates must be normalized to [0, 1000] range
- Coordinate format is [x, y] where x=horizontal, y=vertical. NOT [y, x].
- If the task appears COMPLETE or already done, you MUST output {"action_type": 10}. This is mandatory.
- If the task requires waiting/observing, output {"action_type": -1}
- If the task is impossible, output {"action_type": 11}

Examples:
- Wait/observe: {"action_type": -1}
- Tap center: {"action_type": 0, "coordinate": [500, 500]}
- Double tap: {"action_type": 1, "coordinate": [300, 400]}
- Input text: {"action_type": 7, "text": "hello"}
- Back: {"action_type": 8}
- Task done: {"action_type": 10}
"""

GEMINI_USER_PROMPT_TEMPLATE = """Goal: {task_description}

Step: {step_index}

Analyze the screenshot and determine the next action."""

GEMINI_HISTORY_USER_TEMPLATE = """Goal: {task_description}

Step {step_index}: Execute next action."""

GEMINI_HISTORY_ASSISTANT_TEMPLATE = """{action_json}"""


# =============================================================================
# Qwen Model Prompt Templates (Tool Call Format with Normalized [0, 1000] Coordinates)
# =============================================================================

QWEN_SYSTEM_PROMPT_TEMPLATE = """You are an Android GUI Agent. You control an Android phone by analyzing screenshots and performing actions.

## Available Actions

You can perform the following actions using the tool format:

1. **click(coordinate)** - Tap at position [x, y]
2. **double_click(coordinate)** - Double tap at position [x, y]
3. **long_press(coordinate, time)** - Long press at [x, y] for `time` seconds (default: 2)
4. **swipe(coordinate, coordinate2)** - Swipe from [x1, y1] to [x2, y2]
5. **type(text)** - Input text string
6. **system_button(button)** - Press system button: "Back" or "Home"
7. **terminate(status)** - End task: "success" (completed) or "failure" (impossible)
8. **wait(time)** - Wait/observe for `time` seconds without action

## Coordinate System
- All coordinates are **normalized values in [0, 1000] range** in [x, y] format
- x: horizontal (0=left edge, 1000=right edge)
- y: vertical (0=top edge, 1000=bottom edge)
- Example: center of screen = [500, 500]

## Response Format
You must output your response in the following format:
1. First, analyze the UI elements and plan your action in a <thought> block.
2. Then, execute the action using the tool format below in a <tool_call> block.

### Examples:

**Tap:**
<thought>
I see the search button near the top-center of the screen. I need to tap it.
</thought>
<tool_call>
{"name": "click", "arguments": {"coordinate": [500, 100]}}
</tool_call>

**Swipe down:**
<thought>
I need to scroll down to see more content. I'll swipe from the middle of the screen downward.
</thought>
<tool_call>
{"name": "swipe", "arguments": {"coordinate": [500, 700], "coordinate2": [500, 300]}}
</tool_call>

**Type text:**
<thought>
The text field is focused. I need to type the search query.
</thought>
<tool_call>
{"name": "type", "arguments": {"text": "hello world"}}
</tool_call>

**Press Back:**
<thought>
I need to go back to the previous screen.
</thought>
<tool_call>
{"name": "system_button", "arguments": {"button": "Back"}}
</tool_call>

**Wait/Observe:**
<thought>
The page is still loading. I should wait for it to finish.
</thought>
<tool_call>
{"name": "wait", "arguments": {"time": 2}}
</tool_call>

**Task Complete:**
<thought>
The task has been completed successfully.
</thought>
<tool_call>
{"name": "terminate", "arguments": {"status": "success"}}
</tool_call>

## CRITICAL RULES
- You MUST ALWAYS output a response containing <thought> and <tool_call> blocks. NEVER output an empty response.
- Always output <thought> first, then <tool_call>
- Coordinates MUST be normalized values in [0, 1000] range
- For swipe: provide both start (coordinate) and end (coordinate2) positions
- If the task requires waiting/observing, use the wait action
- If the task appears COMPLETE or already done, you MUST use terminate with "success". This is mandatory.
- If the task is impossible, use terminate with "failure"
- Even when unsure, you MUST still output a valid action. An empty response is NEVER acceptable.
- Do NOT output anything outside of the <thought> and <tool_call> blocks
"""

QWEN_USER_PROMPT_TEMPLATE = """Task: {task_description}

Step: {step_index}

Analyze the screenshot and determine the next action."""

QWEN_HISTORY_USER_TEMPLATE = """Task: {task_description}

Step {step_index}: Analyze the screen and determine the action."""

QWEN_HISTORY_ASSISTANT_TEMPLATE = """<thought>
Executing step {step_index}.
</thought>
<tool_call>
{action_json}
</tool_call>"""


def _omni_action_to_qwen_tool_call(action_json_str: str) -> str:
    """
    Convert ground truth action JSON (OmniActionType format) to Qwen <tool_call> format.
    
    This is used for building history in qwen_doc_to_messages — the assistant's
    historical responses should be in the same format the model is expected to output.
    
    Both Qwen and GT use [0, 1000] normalized coordinates, so no conversion needed.
    
    Args:
        action_json_str: JSON string with action_type and coordinates in [0, 1000]
        
    Returns:
        Qwen tool_call JSON string
    """
    try:
        if isinstance(action_json_str, str):
            action = json.loads(action_json_str)
        else:
            action = action_json_str
    except (json.JSONDecodeError, TypeError):
        return action_json_str if isinstance(action_json_str, str) else json.dumps(action_json_str)
    
    action_type = action.get("action_type", action.get("type", -1))
    
    # Map OmniActionType to Qwen tool_call
    AT = {-1: "wait", 0: "click", 1: "double_click", 2: "long_press",
          3: "swipe", 4: "swipe", 5: "swipe", 6: "swipe",
          7: "type", 8: "system_button", 9: "system_button",
          10: "terminate", 11: "terminate"}
    
    tc_name = AT.get(action_type, "wait")
    tc_args = {}
    
    # Coordinates are kept in [0, 1000] normalized space (same as GT)
    def to_int_coord(coord):
        """Convert coordinate to integer list, keeping in [0, 1000] space."""
        if coord and isinstance(coord, list) and len(coord) >= 2:
            return [int(float(coord[0])), int(float(coord[1]))]
        return coord
    
    if action_type in (0, 1, 2):  # TAP, DOUBLE_TAP, LONG_PRESS
        coord = action.get("coordinate", action.get("point"))
        if coord:
            tc_args["coordinate"] = to_int_coord(coord)
        if action_type == 2:
            tc_args["time"] = action.get("duration", 2000) / 1000  # ms→s
    
    elif action_type in (3, 4, 5, 6):  # SWIPE_*
        start = action.get("start_coordinate", action.get("coordinate"))
        end = action.get("end_coordinate")
        if start:
            tc_args["coordinate"] = to_int_coord(start)
        if end:
            tc_args["coordinate2"] = to_int_coord(end)
    
    elif action_type == 7:  # INPUT
        tc_args["text"] = action.get("text", "")
    
    elif action_type == 8:  # BACK
        tc_args["button"] = "Back"
    
    elif action_type == 9:  # HOME
        tc_args["button"] = "Home"
    
    elif action_type == 10:  # TASK_COMPLETE
        tc_args["status"] = "success"
    
    elif action_type == 11:  # TASK_IMPOSSIBLE
        tc_args["status"] = "failure"
    
    else:  # NONE / wait
        tc_args["time"] = action.get("duration", 2000) / 1000 if "duration" in action else 2
    
    return json.dumps({"name": tc_name, "arguments": tc_args}, ensure_ascii=False)


def qwen_doc_to_messages(doc: Dict, lmms_eval_specific_kwargs: Optional[Dict] = None) -> List[Dict]:
    """
    Convert document to Qwen-optimized multi-turn messages.
    
    Qwen-specific features:
    - Normalized [0, 1000] coordinates (same as Gemini)
    - <thought> + <tool_call> output format
    - History actions converted to Qwen tool_call format for consistency
    
    Args:
        doc: Document containing task data
        lmms_eval_specific_kwargs: Configuration including:
            - media_dir: Base directory for media files
            
    Returns:
        List of message dicts compatible with ChatMessages protocol
    """
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}
    
    media_dir = lmms_eval_specific_kwargs.get("media_dir", "")
    
    def resolve_path(path):
        """Resolve path with media_dir if relative."""
        if path and media_dir and not os.path.isabs(path):
            return os.path.join(media_dir, path)
        return path
    
    messages = []
    
    # 1. System message (no resolution needed — uses normalized [0,1000] coordinates)
    messages.append({
        "role": "system",
        "content": [{"type": "text", "text": QWEN_SYSTEM_PROMPT_TEMPLATE}]
    })
    
    task_description = doc.get("task_description", doc.get("instruction", ""))
    history = doc.get("history", [])
    
    # 2. Historical steps (user query → assistant tool_call)
    for hist_step in history:
        hist_step_index = hist_step.get("step_index", 0)
        
        # User message with multimodal content
        user_content = []
        
        # Add video if available
        if hist_step.get("video_path"):
            user_content.append({"type": "video", "url": resolve_path(hist_step["video_path"])})
        
        # Add audio if available
        if hist_step.get("audio_path"):
            user_content.append({"type": "audio", "url": {"path": resolve_path(hist_step["audio_path"])}})
        
        # Add image (PNG screenshot)
        if hist_step.get("image_path"):
            user_content.append({"type": "image", "url": resolve_path(hist_step["image_path"])})
        elif hist_step.get("image"):
            user_content.append({"type": "image", "url": resolve_path(hist_step["image"])})
        
        # Add text prompt
        user_text = QWEN_HISTORY_USER_TEMPLATE.format(
            task_description=task_description,
            step_index=hist_step_index
        )
        user_content.append({"type": "text", "text": user_text})
        
        messages.append({"role": "user", "content": user_content})
        
        # Assistant message — convert GT action to Qwen tool_call format
        action_json = hist_step.get("action_json", "{}")
        tc_json = _omni_action_to_qwen_tool_call(action_json)
        
        assistant_text = QWEN_HISTORY_ASSISTANT_TEMPLATE.format(
            step_index=hist_step_index,
            action_json=tc_json
        )
        messages.append({
            "role": "assistant",
            "content": [{"type": "text", "text": assistant_text}]
        })
    
    # 3. Current step — User message requesting prediction
    current_step_index = doc.get("step_index", doc.get("step_id", len(history)))
    current_content = []
    
    # Add current video
    current_video = doc.get("current_video") or doc.get("video") or doc.get("video_path")
    if current_video:
        current_content.append({"type": "video", "url": resolve_path(current_video)})
    
    # Add current audio
    current_audio = doc.get("current_audio") or doc.get("audio") or doc.get("audio_path")
    if current_audio:
        if isinstance(current_audio, str):
            current_content.append({"type": "audio", "url": {"path": resolve_path(current_audio)}})
        else:
            current_content.append({"type": "audio", "url": current_audio})
    
    # Add current image
    current_image = doc.get("current_image") or doc.get("image") or doc.get("image_path")
    if current_image:
        if isinstance(current_image, str):
            current_content.append({"type": "image", "url": resolve_path(current_image)})
        else:
            current_content.append({"type": "image", "url": resolve_path(current_image)})
    
    # Add text prompt
    current_text = QWEN_USER_PROMPT_TEMPLATE.format(
        task_description=task_description,
        step_index=current_step_index
    )
    current_content.append({"type": "text", "text": current_text})
    
    messages.append({"role": "user", "content": current_content})
    
    return messages


def gemini_doc_to_messages(doc: Dict, lmms_eval_specific_kwargs: Optional[Dict] = None) -> List[Dict]:
    """
    Convert document to Gemini-optimized multi-turn messages.
    
    Gemini-specific optimizations:
    - Clear JSON output format with examples
    - Structured action schema
    - Explicit swipe dual-coordinate format
    
    Args:
        doc: Document containing task data
        lmms_eval_specific_kwargs: Configuration including:
            - system_prompt: Custom system prompt (optional)
            - max_history_steps: Number of history steps
            
    Returns:
        List of message dicts compatible with ChatMessages protocol
    """
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}
    
    media_dir = lmms_eval_specific_kwargs.get("media_dir", "")
    
    def resolve_path(path):
        """Resolve path with media_dir if relative."""
        if path and media_dir and not os.path.isabs(path):
            return os.path.join(media_dir, path)
        return path
    
    messages = []
    
    # 1. System message
    system_prompt = lmms_eval_specific_kwargs.get("gemini_system_prompt", GEMINI_SYSTEM_PROMPT)
    messages.append({
        "role": "system",
        "content": [{"type": "text", "text": system_prompt}]
    })
    
    task_description = doc.get("task_description", doc.get("instruction", ""))
    history = doc.get("history", [])
    
    # 2. Historical steps
    for hist_step in history:
        hist_step_index = hist_step.get("step_index", 0)
        
        # User message with multimodal content
        user_content = []
        
        # Add video if available
        if hist_step.get("video_path"):
            user_content.append({
                "type": "video",
                "url": resolve_path(hist_step["video_path"])
            })
        
        # Add audio if available
        if hist_step.get("audio_path"):
            user_content.append({
                "type": "audio",
                "url": {"path": resolve_path(hist_step["audio_path"])}
            })
        
        # Add image (PNG screenshot) - primary input
        if hist_step.get("image_path"):
            user_content.append({
                "type": "image",
                "url": resolve_path(hist_step["image_path"])
            })
        elif hist_step.get("image"):
            user_content.append({
                "type": "image",
                "url": resolve_path(hist_step["image"])
            })
        
        # Add text prompt
        user_text = GEMINI_HISTORY_USER_TEMPLATE.format(
            task_description=task_description,
            step_index=hist_step_index
        )
        user_content.append({"type": "text", "text": user_text})
        
        messages.append({
            "role": "user",
            "content": user_content
        })
        
        # Assistant message with action
        action_json = hist_step.get("action_json", "{}")
        if isinstance(action_json, dict):
            action_json = json.dumps(action_json, ensure_ascii=False)
        
        messages.append({
            "role": "assistant",
            "content": [{"type": "text", "text": action_json}]
        })
    
    # 3. Current step - User message requesting prediction
    current_step_index = doc.get("step_index", doc.get("step_id", len(history)))
    current_content = []
    
    # Add current video
    current_video = doc.get("current_video") or doc.get("video") or doc.get("video_path")
    if current_video:
        current_content.append({
            "type": "video",
            "url": resolve_path(current_video)
        })
    
    # Add current audio
    current_audio = doc.get("current_audio") or doc.get("audio") or doc.get("audio_path")
    if current_audio:
        if isinstance(current_audio, str):
            current_content.append({
                "type": "audio",
                "url": {"path": resolve_path(current_audio)}
            })
        else:
            current_content.append({
                "type": "audio",
                "url": current_audio
            })
    
    # Add current image (PNG screenshot)
    current_image = doc.get("current_image") or doc.get("image") or doc.get("image_path")
    if current_image:
        if isinstance(current_image, str):
            current_content.append({
                "type": "image",
                "url": resolve_path(current_image)
            })
        else:
            current_content.append({
                "type": "image",
                "url": current_image
            })
    
    # Add text prompt
    current_text = GEMINI_USER_PROMPT_TEMPLATE.format(
        task_description=task_description,
        step_index=current_step_index
    )
    current_content.append({"type": "text", "text": current_text})
    
    messages.append({
        "role": "user",
        "content": current_content
    })
    
    return messages


def omni_doc_to_messages(doc: Dict, lmms_eval_specific_kwargs: Optional[Dict] = None) -> List[Dict]:
    """
    Convert Omni Benchmark document to multi-turn messages with full history.
    
    Each step in history includes:
    - Video (path string)
    - Audio (dict with path)
    - Image (PNG path or PIL Image)
    
    Message structure:
    1. System message with OMNI_SYSTEM_PROMPT
    2. For each historical step:
       - User message: video + audio + image + step prompt
       - Assistant message: action JSON
    3. Current step: User message requesting action prediction
    
    Args:
        doc: Document containing:
            - episode_id: Episode identifier
            - step_index: Current step number
            - task_description: Task goal
            - history: List of historical steps, each with:
                - video_path: Path to video file
                - audio_path: Path to audio file (wav/mp3)
                - image_path: Path to PNG screenshot
                - action_json: JSON string of the action taken
            - current_video: Current step video path
            - current_audio: Current step audio path
            - current_image: Current step image path
        lmms_eval_specific_kwargs: Optional kwargs for customization
    
    Returns:
        List of message dicts compatible with ChatMessages protocol
    """
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}
    
    media_dir = lmms_eval_specific_kwargs.get("media_dir", "")
    
    def resolve_path(path):
        """Resolve path with media_dir if relative."""
        if path and media_dir and not os.path.isabs(path):
            return os.path.join(media_dir, path)
        return path
    
    messages = []
    
    # 1. System message
    system_prompt = lmms_eval_specific_kwargs.get("system_prompt", OMNI_SYSTEM_PROMPT)
    messages.append({
        "role": "system",
        "content": [{"type": "text", "text": system_prompt}]
    })
    
    task_description = doc.get("task_description", doc.get("instruction", ""))
    history = doc.get("history", [])
    
    # 2. Historical steps
    for hist_step in history:
        hist_step_index = hist_step.get("step_index", 0)
        
        # User message with multimodal content
        user_content = []
        
        # Add video if available
        if hist_step.get("video_path"):
            user_content.append({
                "type": "video",
                "url": resolve_path(hist_step["video_path"])
            })
        
        # Add audio if available
        if hist_step.get("audio_path"):
            user_content.append({
                "type": "audio",
                "url": {"path": resolve_path(hist_step["audio_path"])}
            })
        
        # Add image (PNG screenshot) - primary input
        if hist_step.get("image_path"):
            user_content.append({
                "type": "image",
                "url": resolve_path(hist_step["image_path"])
            })
        elif hist_step.get("image"):
            # PIL Image object
            user_content.append({
                "type": "image",
                "url": resolve_path(hist_step["image"])
            })
        
        # Add text prompt
        user_text = OMNI_HISTORY_USER_TEMPLATE.format(
            task_description=task_description,
            step_index=hist_step_index
        )
        user_content.append({"type": "text", "text": user_text})
        
        messages.append({
            "role": "user",
            "content": user_content
        })
        
        # Assistant message with action
        action_json = hist_step.get("action_json", "{}")
        if isinstance(action_json, dict):
            action_json = json.dumps(action_json, ensure_ascii=False)
        
        messages.append({
            "role": "assistant",
            "content": [{"type": "text", "text": action_json}]
        })
    
    # 3. Current step - User message requesting prediction
    current_step_index = doc.get("step_index", doc.get("step_id", len(history)))
    current_content = []
    
    # Add current video
    current_video = doc.get("current_video") or doc.get("video") or doc.get("video_path")
    if current_video:
        current_content.append({
            "type": "video",
            "url": resolve_path(current_video)
        })
    
    # Add current audio
    current_audio = doc.get("current_audio") or doc.get("audio") or doc.get("audio_path")
    if current_audio:
        if isinstance(current_audio, str):
            current_content.append({
                "type": "audio",
                "url": {"path": resolve_path(current_audio)}
            })
        else:
            current_content.append({
                "type": "audio",
                "url": current_audio
            })
    
    # Add current image (PNG screenshot)
    current_image = doc.get("current_image") or doc.get("image") or doc.get("image_path")
    if current_image:
        if isinstance(current_image, str):
            current_content.append({
                "type": "image",
                "url": resolve_path(current_image)
            })
        else:
            current_content.append({
                "type": "image",
                "url": current_image
            })
    
    # Add text prompt
    current_text = OMNI_USER_PROMPT_TEMPLATE.format(
        task_description=task_description,
        step_index=current_step_index
    )
    current_content.append({"type": "text", "text": current_text})
    
    messages.append({
        "role": "user",
        "content": current_content
    })
    
    return messages


def omni_doc_to_visual(doc: Dict, lmms_eval_specific_kwargs: Optional[Dict] = None) -> List:
    """
    Extract all visual/audio elements from Omni document.
    
    Returns list of:
    - Video paths (str) for video type
    - Audio dicts ({"path": ...}) for audio type
    - Image paths or PIL Images for image type
    
    Order: All history visuals first, then current step visuals.
    """
    visuals = []
    
    # Get media directory from kwargs
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}
    media_dir = lmms_eval_specific_kwargs.get("media_dir", "")
    
    def resolve_path(path):
        """Resolve path with media_dir if relative."""
        if path and media_dir and not os.path.isabs(path):
            return os.path.join(media_dir, path)
        return path
    
    # Historical steps
    history = doc.get("history", [])
    for hist_step in history:
        # Video
        if hist_step.get("video_path"):
            visuals.append(resolve_path(hist_step["video_path"]))
        # Audio
        if hist_step.get("audio_path"):
            visuals.append({"path": resolve_path(hist_step["audio_path"])})
        # Image
        if hist_step.get("image_path"):
            visuals.append(resolve_path(hist_step["image_path"]))
        elif hist_step.get("image"):
            visuals.append(hist_step["image"])
    
    # Current step - check both formats
    # Format 1: current_video/current_audio/current_image
    # Format 2: video/audio/image
    # Format 3: video_path/audio_path/image_path
    
    current_video = doc.get("current_video") or doc.get("video") or doc.get("video_path")
    if current_video:
        visuals.append(resolve_path(current_video))
    
    current_audio = doc.get("current_audio") or doc.get("audio") or doc.get("audio_path")
    if current_audio:
        if isinstance(current_audio, str):
            visuals.append({"path": resolve_path(current_audio)})
        else:
            visuals.append(current_audio)
    
    current_image = doc.get("current_image") or doc.get("image") or doc.get("image_path")
    if current_image:
        visuals.append(resolve_path(current_image))
    
    return visuals


def omni_doc_to_text(doc: Dict, lmms_eval_specific_kwargs: Optional[Dict] = None) -> str:
    """
    Generate text prompt for Omni Benchmark (fallback for non-message models).
    """
    task_description = doc.get("task_description", doc.get("instruction", ""))
    step_index = doc.get("step_index", doc.get("step_id", 0))
    
    return OMNI_USER_PROMPT_TEMPLATE.format(
        task_description=task_description,
        step_index=step_index
    )


def omni_gui_grounding_doc_to_text(doc: Dict, lmms_eval_specific_kwargs: Optional[Dict] = None) -> str:
    """
    Fallback text prompt with screenshot-grounding evidence for models that do
    not consume doc_to_messages.
    """
    text = omni_doc_to_text(doc, lmms_eval_specific_kwargs)
    evidence = str(doc.get("gui_grounding_evidence", "")).strip()
    if not evidence:
        return text
    return (
        text
        + "\n\nScreenshot-only GUI grounding evidence:\n"
        + evidence
        + "\nUse this evidence only when it is relevant; audio, video, and history may still determine the next action."
    )


def omni_doc_to_target(doc: Dict, lmms_eval_specific_kwargs: Optional[Dict] = None) -> str:
    """
    Generate target string for Omni Benchmark.
    Returns the result_action_json if available, otherwise constructs from components.
    """
    if "result_action_json" in doc:
        return doc["result_action_json"]
    
    # Construct from individual fields
    action_type = doc.get("result_action_type", -1)
    action_text = doc.get("result_action_text", "")
    touch_xy = doc.get("result_touch_xy", "")
    
    target = {
        "action_type": action_type,
        "text": action_text,
        "touch_xy": touch_xy
    }
    return json.dumps(target)


# =============================================================================
# Omni Benchmark: Evaluation Functions
# =============================================================================

def parse_omni_action(output_str: str) -> Optional[Dict]:
    """
    Parse Omni action JSON from model output with tolerance parsing.
    
    This function is designed to handle various model output formats:
    - Standard: {"action_type": 0, "coordinate": [x, y], "text": "..."}
    - String action: {"action": "click", "coordinate": [x, y]}
    - Various key names: action_type, action, type, action_name, etc.
    - Various coordinate names: coordinate, coordinates, position, point, xy, x/y, etc.
    - Swipe variations: start_coordinate/end_coordinate, from/to, etc.
    
    Returns:
        Parsed action dict with normalized keys, or None if parsing fails
    """
    parsed = parse_json_output(output_str)
    if parsed is None:
        eval_logger.debug(f"Failed to parse JSON from output: {output_str[:100]}...")
        return None
    if isinstance(parsed, (int, float)):
        action_type = int(parsed)
        if -1 <= action_type <= 11:
            return {"action_type": action_type}
        eval_logger.debug(f"Parsed bare numeric action outside Omni range: {parsed}")
        return None
    if not isinstance(parsed, dict):
        eval_logger.debug(f"Parsed JSON is not an action object: {parsed}")
        return None
    
    # === Step 1: Extract and normalize action type ===
    action_type = None
    
    # Try various key names for action type
    action_keys = ["action_type", "action", "type", "action_name", "actionType", "actiontype", "operation"]
    for key in action_keys:
        if key in parsed:
            raw_action = parsed[key]
            
            # If it's already an integer
            if isinstance(raw_action, int):
                # Validate range (-1 to 11 for OmniActionType)
                if -1 <= raw_action <= 11:
                    action_type = raw_action
                    break
            
            # If it's a string, use tolerance parsing
            elif isinstance(raw_action, str):
                normalized = normalize_action_type_string(raw_action)
                if normalized is not None:
                    action_type = normalized
                    break
            
            # If it's a float, convert to int
            elif isinstance(raw_action, float):
                int_action = int(raw_action)
                if -1 <= int_action <= 11:
                    action_type = int_action
                    break
    
    if action_type is None:
        # Fallback: If we have a coordinate/point but no action_type, assume click (TAP=0)
        coord_keys = ["coordinate", "coordinates", "position", "point", "xy", "pos", 
                      "location", "touch_point", "click_point"]
        has_coordinate = any(key in parsed for key in coord_keys)
        if has_coordinate:
            action_type = OmniActionType.TAP  # Default to click
            eval_logger.debug(f"Inferred action_type=0 (TAP) from presence of coordinate in: {parsed}")
        else:
            eval_logger.debug(f"Could not extract action_type from parsed JSON: {parsed}")
            return None
    
    result = {"action_type": action_type}
    
    # === Step 2: Extract coordinates with tolerance ===
    
    # Helper function to extract coordinate from various formats
    def extract_coordinate(data: Dict, key_candidates: List[str]) -> Optional[List[int]]:
        for key in key_candidates:
            if key in data:
                coord = data[key]
                
                # Format: [x, y]
                if isinstance(coord, list) and len(coord) >= 2:
                    try:
                        return [int(float(coord[0])), int(float(coord[1]))]
                    except (ValueError, TypeError):
                        continue
                
                # Format: {"x": val, "y": val}
                elif isinstance(coord, dict):
                    if "x" in coord and "y" in coord:
                        try:
                            return [int(float(coord["x"])), int(float(coord["y"]))]
                        except (ValueError, TypeError):
                            continue
                
                # Format: "x,y" or "x y"
                elif isinstance(coord, str):
                    parts = re.split(r'[,\s]+', coord.strip())
                    if len(parts) >= 2:
                        try:
                            return [int(float(parts[0])), int(float(parts[1]))]
                        except (ValueError, TypeError):
                            continue
        
        # Try separate x and y keys
        x_keys = ["x", "X", "pos_x", "posX"]
        y_keys = ["y", "Y", "pos_y", "posY"]
        x_val, y_val = None, None
        for xk in x_keys:
            if xk in data:
                try:
                    x_val = int(float(data[xk]))
                    break
                except (ValueError, TypeError):
                    continue
        for yk in y_keys:
            if yk in data:
                try:
                    y_val = int(float(data[yk]))
                    break
                except (ValueError, TypeError):
                    continue
        if x_val is not None and y_val is not None:
            return [x_val, y_val]
        
        return None
    
    # Check if this is a swipe action
    is_swipe = action_type in [OmniActionType.SWIPE_UP, OmniActionType.SWIPE_DOWN,
                                OmniActionType.SWIPE_LEFT, OmniActionType.SWIPE_RIGHT]
    
    # For generic "swipe" or "scroll" without direction, we need to determine from coordinates
    if action_type is None and parsed.get("action", "").lower() in ["swipe", "scroll"]:
        is_swipe = True
    
    if is_swipe:
        # Extract start coordinate
        start_keys = ["start_coordinate", "startCoordinate", "start", "from", "from_coordinate", 
                      "touch", "touch_point", "coordinate", "coordinates", "position", "point"]
        start_coord = extract_coordinate(parsed, start_keys)
        if start_coord:
            result["start_coordinate"] = start_coord
            result["coordinate"] = start_coord  # backward compatibility
        
        # Extract end coordinate
        end_keys = ["end_coordinate", "endCoordinate", "end", "to", "to_coordinate",
                    "lift", "lift_point", "coordinate2", "target"]
        end_coord = extract_coordinate(parsed, end_keys)
        if end_coord:
            result["end_coordinate"] = end_coord
        
        # If we have a generic swipe and both coordinates, determine direction
        if result["action_type"] is None and start_coord and end_coord:
            direction = determine_swipe_direction(start_coord, end_coord)
            if direction:
                result["action_type"] = direction
    else:
        # Non-swipe actions: extract single coordinate
        coord_keys = ["coordinate", "coordinates", "position", "point", "xy", "pos", 
                      "location", "touch_point", "click_point"]
        coord = extract_coordinate(parsed, coord_keys)
        if coord:
            result["coordinate"] = coord
    
    # === Step 3: Extract text with tolerance ===
    text_keys = ["text", "input_text", "inputText", "content", "value", "string", "message"]
    for key in text_keys:
        if key in parsed:
            text_val = parsed[key]
            if text_val is not None:
                result["text"] = str(text_val)
                break
    
    # === Step 4: Extract additional useful fields ===
    # Duration
    if "duration" in parsed:
        try:
            result["duration"] = int(parsed["duration"])
        except (ValueError, TypeError):
            pass
    
    # Confidence/score
    for key in ["confidence", "score", "probability"]:
        if key in parsed:
            try:
                result["confidence"] = float(parsed[key])
                break
            except (ValueError, TypeError):
                pass
    
    # Reasoning/explanation (useful for debugging)
    for key in ["reasoning", "explanation", "thought", "thinking", "analysis"]:
        if key in parsed:
            result["reasoning"] = str(parsed[key])
            break
    
    eval_logger.debug(f"Parsed action: {result} from input: {output_str[:100]}...")
    return result


def evaluate_omni_step(
    doc: Dict,
    pred_output: str,
    screen_width: int = 1080,
    screen_height: int = 2400,
    coordinate_type: str = "auto"
) -> Dict:
    """
    Evaluate a single Omni step prediction.
    
    Supports:
    - Automatic coordinate type detection (pixel vs normalized)
    - Rectangle ground truth for coordinate matching
    - All OmniActionType actions
    
    Args:
        doc: Document with ground truth
        pred_output: Model's output string
        screen_width: Screen width for pixel coordinate conversion
        screen_height: Screen height for pixel coordinate conversion
        coordinate_type: "auto" (detect), "pixel", or "normalized"
    
    Returns:
        Evaluation result dict
    """
    # Parse prediction
    pred_action = parse_omni_action(pred_output)
    
    # Parse ground truth
    gt_action_type = doc.get("result_action_type")
    gt_text = doc.get("result_action_text", "")
    
    # Parse ground truth coordinate (supports rectangle format)
    gt_coord_info = parse_ground_truth_coordinate(doc)
    
    # Get screen dimensions from doc if available
    if doc.get("image_width"):
        screen_width = doc.get("image_width")
    if doc.get("image_height"):
        screen_height = doc.get("image_height")
    
    # Compute metrics
    hit_format = pred_action is not None
    type_match = False
    exact_match = False
    coordinate_distance = None
    text_match = False
    
    swipe_start_distance = None
    swipe_end_distance = None
    
    if hit_format and gt_action_type is not None:
        pred_type = pred_action.get("action_type")
        type_match = (pred_type == gt_action_type)
        
        if type_match:
            # Check based on action type
            if pred_type in [OmniActionType.TAP, OmniActionType.LONG_PRESS]:
                # Single coordinate actions (TAP, LONG_PRESS)
                pred_coord = pred_action.get("coordinate")
                if pred_coord and gt_coord_info["type"] != "none":
                    exact_match, coordinate_distance = evaluate_coordinate_match(
                        pred_coord=pred_coord,
                        gt_info=gt_coord_info,
                        distance_threshold=140,
                        screen_width=screen_width,
                        screen_height=screen_height,
                        coordinate_type=coordinate_type
                    )
            
            elif pred_type in [OmniActionType.SWIPE_UP, OmniActionType.SWIPE_DOWN,
                              OmniActionType.SWIPE_LEFT, OmniActionType.SWIPE_RIGHT]:
                # Dual coordinate actions (SWIPE)
                pred_start = pred_action.get("start_coordinate", pred_action.get("coordinate"))
                pred_end = pred_action.get("end_coordinate")
                
                # Parse GT swipe coordinates (both start and end)
                gt_swipe_info = parse_ground_truth_swipe_coordinates(doc)
                
                if pred_start and gt_swipe_info["start"]["type"] != "none":
                    if pred_end and gt_swipe_info["end"]["type"] != "none":
                        # Full swipe evaluation with both coordinates
                        exact_match, swipe_start_distance, swipe_end_distance = evaluate_swipe_coordinate_match(
                            pred_start=pred_start,
                            pred_end=pred_end,
                            gt_swipe_info=gt_swipe_info,
                            distance_threshold=140,
                            screen_width=screen_width,
                            screen_height=screen_height,
                            coordinate_type=coordinate_type
                        )
                        coordinate_distance = swipe_start_distance
                    else:
                        # Only start coordinate available, evaluate start only
                        exact_match, coordinate_distance = evaluate_coordinate_match(
                            pred_coord=pred_start,
                            gt_info=gt_swipe_info["start"],
                            distance_threshold=140,
                            screen_width=screen_width,
                            screen_height=screen_height,
                            coordinate_type=coordinate_type
                        )
            
            elif pred_type == OmniActionType.INPUT:
                # Text input action
                pred_text = pred_action.get("text", "")
                if pred_text and gt_text:
                    pred_text_norm = pred_text.lower().strip()
                    gt_text_norm = gt_text.lower().strip()
                    text_match = (pred_text_norm in gt_text_norm or 
                                 gt_text_norm in pred_text_norm)
                    exact_match = text_match
            
            elif pred_type in [OmniActionType.NONE, OmniActionType.BACK, OmniActionType.HOME,
                              OmniActionType.TASK_COMPLETE, OmniActionType.TASK_IMPOSSIBLE]:
                # No-parameter actions
                exact_match = True
    
    return {
        "subset": doc.get("subset", "unknown"),
        "episode_id": doc.get("episode_id", "unknown"),
        "step_id": doc.get("step_index", doc.get("step_id", 0)),
        "gt_action_type": gt_action_type,
        "pd_action_type": pred_action.get("action_type") if pred_action else None,
        "type_match": type_match,
        "exact_match": exact_match,
        "hit_format": hit_format,
        "coordinate_distance": coordinate_distance,
        "swipe_start_distance": swipe_start_distance,
        "swipe_end_distance": swipe_end_distance,
        "text_match": text_match,
        "gt_coord_type": gt_coord_info["type"],  # "point", "rect", or "none"
    }


def omni_process_results(doc: Dict, results: List[str], lmms_eval_specific_kwargs: Optional[Dict] = None) -> Dict:
    """
    Process Omni Benchmark results for a single step.
    
    Args:
        doc: Document with ground truth
        results: List of model outputs
        lmms_eval_specific_kwargs: Configuration including:
            - coordinate_type: "auto", "pixel", or "normalized"
            - screen_width: Screen width for pixel conversion
            - screen_height: Screen height for pixel conversion
    """
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}
    
    pred_output = results[0] if results else ""
    
    # Get configuration
    coordinate_type = lmms_eval_specific_kwargs.get("coordinate_type", "auto")
    screen_width = lmms_eval_specific_kwargs.get("screen_width", 1080)
    screen_height = lmms_eval_specific_kwargs.get("screen_height", 2400)
    
    eval_result = evaluate_omni_step(
        doc, 
        pred_output,
        screen_width=screen_width,
        screen_height=screen_height,
        coordinate_type=coordinate_type
    )
    
    return {
        "type_match": eval_result,
        "exact_match": eval_result,
        "success_rate": eval_result,
        "goal_progress": eval_result,
    }


def omni_cepg_lite_process_results(doc: Dict, results: List[str], lmms_eval_specific_kwargs: Optional[Dict] = None) -> Dict:
    """
    CEPG-lite process hook.

    It keeps the model output fixed, but evaluates it through a training-free
    evidence-prior guard:
    - malformed JSON is recovered by parse_json_output/parse_omni_action;
    - pixel-looking coordinates are auto-normalized instead of assuming the
      model obeyed the normalized-coordinate prompt;
    - no-parameter actions such as NONE are scored consistently.

    Full CEPG would rerank candidate actions with multimodal counterfactual
    likelihoods. This hook is the low-risk executable layer that can be tested
    on cached Qwen2.5-Omni outputs without another full generation pass.
    """
    kwargs = dict(lmms_eval_specific_kwargs or {})
    kwargs["coordinate_type"] = "auto"
    return omni_process_results(doc, results, kwargs)


# Aggregation functions reuse agentcpm versions
omni_aggregate_type_match = agentcpm_aggregate_type_match
omni_aggregate_exact_match = agentcpm_aggregate_exact_match
omni_aggregate_success_rate = agentcpm_aggregate_success_rate
omni_aggregate_goal_progress = agentcpm_aggregate_goal_progress


# =============================================================================
# Omni Local: Functions for local dataset with dynamic history building
# =============================================================================

# Cache for loaded datasets
_omni_local_dataset_cache: Dict[str, List[Dict]] = {}


def _load_omni_local_dataset(dataset_path: str) -> List[Dict]:
    """
    Load and cache local Omni dataset from JSON file.
    
    Args:
        dataset_path: Path to the JSON file containing the dataset
        
    Returns:
        List of document dicts
    """
    global _omni_local_dataset_cache
    
    if dataset_path in _omni_local_dataset_cache:
        return _omni_local_dataset_cache[dataset_path]
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    _omni_local_dataset_cache[dataset_path] = data
    return data


def _build_omni_history(
    full_data: List[Dict],
    episode_id: str,
    current_step_id: int,
    base_dir: str,
    max_history_steps: int = 2,
    include_video: bool = True,
    include_screenshot: bool = True,
    skip_audio: bool = True
) -> List[Dict]:
    """
    Dynamically build history from full dataset.
    
    Args:
        full_data: Complete dataset loaded from JSON
        episode_id: Current episode ID
        current_step_id: Current step index
        base_dir: Base directory for media files
        max_history_steps: Maximum number of history steps to include
        include_video: Whether to include video paths
        include_screenshot: Whether to include screenshot paths
        skip_audio: Whether to skip audio (for models that don't support it)
        
    Returns:
        List of history step dicts with media paths and action_json
    """
    # Filter and sort steps for this episode
    episode_steps = [
        doc for doc in full_data 
        if doc.get("episode_id") == episode_id and doc.get("step_id", 0) < current_step_id
    ]
    episode_steps.sort(key=lambda x: x.get("step_id", 0))
    
    # Take only the most recent history steps
    if max_history_steps > 0 and len(episode_steps) > max_history_steps:
        episode_steps = episode_steps[-max_history_steps:]
    
    history = []
    for step_doc in episode_steps:
        step_id = step_doc.get("step_id", 0)
        file_idx = step_id + 1  # Files are 1-indexed: 1.mp4, 2.mp4, etc.
        
        hist_entry = {
            "step_index": step_id,
        }
        
        # Add video path - prefer doc's video_path if available, fallback to {file_idx}.mp4
        if include_video:
            doc_video = step_doc.get("video_path", "")
            if doc_video:
                video_path = os.path.join(base_dir, doc_video)
            else:
                video_path = os.path.join(base_dir, f"{file_idx}.mp4")
            if os.path.exists(video_path):
                hist_entry["video_path"] = video_path
        
        # Add audio path - prefer doc's audio_path if available
        if not skip_audio:
            doc_audio = step_doc.get("audio_path", "")
            if doc_audio:
                audio_path = os.path.join(base_dir, doc_audio)
            else:
                audio_path = os.path.join(base_dir, f"{file_idx}.wav")
            if os.path.exists(audio_path):
                hist_entry["audio_path"] = audio_path
        
        # Add screenshot path - prefer doc's image_path if available
        if include_screenshot:
            doc_image = step_doc.get("image_path", "")
            if doc_image:
                image_path = os.path.join(base_dir, doc_image)
            else:
                image_path = os.path.join(base_dir, f"{file_idx}.png")
            if os.path.exists(image_path):
                hist_entry["image_path"] = image_path
        
        # Build action_json from ground truth
        action_type = step_doc.get("result_action_type")
        text = step_doc.get("result_action_text", "")
        
        action_dict = {"action_type": action_type}
        
        # Check if this is a swipe action
        is_swipe = action_type in [OmniActionType.SWIPE_UP, OmniActionType.SWIPE_DOWN,
                                    OmniActionType.SWIPE_LEFT, OmniActionType.SWIPE_RIGHT]
        
        if is_swipe:
            # Parse both start and end coordinates for swipe actions
            swipe_info = parse_ground_truth_swipe_coordinates(step_doc)
            
            if swipe_info["start"]["center"]:
                action_dict["start_coordinate"] = [
                    int(swipe_info["start"]["center"][0]),
                    int(swipe_info["start"]["center"][1])
                ]
                # Keep single coordinate for backward compatibility
                action_dict["coordinate"] = action_dict["start_coordinate"]
            
            if swipe_info["end"]["center"]:
                action_dict["end_coordinate"] = [
                    int(swipe_info["end"]["center"][0]),
                    int(swipe_info["end"]["center"][1])
                ]
        else:
            # Parse single coordinate for non-swipe actions
            coord_info = parse_ground_truth_coordinate(step_doc)
            
            # Use center point as coordinate for action_json
            if coord_info["center"] and coord_info["center"] != [-1, -1]:
                # For rect, use center; for point, use the point itself
                action_dict["coordinate"] = [int(coord_info["center"][0]), int(coord_info["center"][1])]
        
        if text and action_type == OmniActionType.INPUT:
            action_dict["text"] = text
        
        hist_entry["action_json"] = json.dumps(action_dict, ensure_ascii=False)
        
        history.append(hist_entry)
    
    return history


def omni_local_doc_to_messages(doc: Dict, lmms_eval_specific_kwargs: Optional[Dict] = None) -> List[Dict]:
    """
    Convert local Omni dataset document to multi-turn messages with dynamic history.
    
    This function dynamically builds history from the full dataset based on
    episode_id and step_id, then constructs the message list.
    
    Args:
        doc: Current document from the dataset
        lmms_eval_specific_kwargs: Configuration parameters including:
            - dataset_path: Path to the full JSON dataset file
            - base_dir: Base directory for media files
            - max_frames_per_video: Frames to extract per video (default: 2)
            - max_history_steps: Number of history steps to include (default: 2)
            - skip_audio: Whether to skip audio content (default: True)
            - include_video: Whether to include video (default: True)
            - include_screenshot: Whether to include screenshots (default: True)
            - system_prompt: Custom system prompt (optional)
            
    Returns:
        List of message dicts compatible with ChatMessages protocol
    """
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}
    
    # Get configuration parameters with defaults
    dataset_path = lmms_eval_specific_kwargs.get("dataset_path", "")
    base_dir = lmms_eval_specific_kwargs.get("base_dir", "")
    max_history_steps = lmms_eval_specific_kwargs.get("max_history_steps", 2)
    skip_audio = lmms_eval_specific_kwargs.get("skip_audio", True)
    include_video = lmms_eval_specific_kwargs.get("include_video", True)
    include_screenshot = lmms_eval_specific_kwargs.get("include_screenshot", True)
    
    # TTS (Voice Instruction) configuration
    tts_enabled = lmms_eval_specific_kwargs.get("tts_enabled", False)
    tts_cache_dir = lmms_eval_specific_kwargs.get("tts_cache_dir", "tts_cache")
    tts_voice = lmms_eval_specific_kwargs.get("tts_voice", "zh-CN-XiaoxiaoNeural")
    
    # If base_dir not specified, derive from dataset_path
    if not base_dir and dataset_path:
        base_dir = os.path.dirname(dataset_path)
    
    messages = []
    
    # 1. System message (handle None value from yaml)
    system_prompt = lmms_eval_specific_kwargs.get("system_prompt") or OMNI_SYSTEM_PROMPT
    messages.append({
        "role": "system",
        "content": [{"type": "text", "text": system_prompt}]
    })
    
    # Get document info
    episode_id = doc.get("episode_id", "")
    current_step_id = doc.get("step_id", 0)
    task_description = doc.get("instruction", doc.get("task_description", ""))
    
    # 2. Build history dynamically
    if dataset_path:
        full_data = _load_omni_local_dataset(dataset_path)
        history = _build_omni_history(
            full_data=full_data,
            episode_id=episode_id,
            current_step_id=current_step_id,
            base_dir=base_dir,
            max_history_steps=max_history_steps,
            include_video=include_video,
            include_screenshot=include_screenshot,
            skip_audio=skip_audio
        )
    else:
        history = []
    
    # 3. Add history messages
    for hist_step in history:
        hist_step_index = hist_step.get("step_index", 0)
        
        user_content = []
        
        # Add video if available
        if hist_step.get("video_path"):
            user_content.append({"type": "video", "url": hist_step["video_path"]})
        
        # Add audio if available (and not skipped)
        if hist_step.get("audio_path"):
            user_content.append({"type": "audio", "url": {"path": hist_step["audio_path"]}})
        
        # Add screenshot
        if hist_step.get("image_path"):
            user_content.append({"type": "image", "url": hist_step["image_path"]})
        
        # Add text prompt
        user_text = OMNI_HISTORY_USER_TEMPLATE.format(
            task_description=task_description,
            step_index=hist_step_index
        )
        user_content.append({"type": "text", "text": user_text})
        
        messages.append({"role": "user", "content": user_content})
        
        # Assistant response
        action_json = hist_step.get("action_json", "{}")
        messages.append({
            "role": "assistant",
            "content": [{"type": "text", "text": action_json}]
        })
    
    # 4. Current step - User message
    current_content = []
    file_idx = current_step_id + 1
    
    # Add current video - prefer doc's video_path if available
    if include_video:
        doc_video = doc.get("video_path", "")
        if doc_video:
            current_video = os.path.join(base_dir, doc_video)
        else:
            current_video = os.path.join(base_dir, f"{file_idx}.mp4")
        if os.path.exists(current_video):
            current_content.append({"type": "video", "url": current_video})
    
    # Add current audio (if not skipped) - prefer doc's audio_path if available
    if not skip_audio:
        doc_audio = doc.get("audio_path", "")
        if doc_audio:
            current_audio = os.path.join(base_dir, doc_audio)
        else:
            current_audio = os.path.join(base_dir, f"{file_idx}.wav")
        if os.path.exists(current_audio):
            current_content.append({"type": "audio", "url": {"path": current_audio}})
    
    # Add current screenshot - prefer doc's image_path if available
    if include_screenshot:
        doc_image = doc.get("image_path", "")
        if doc_image:
            current_image = os.path.join(base_dir, doc_image)
        else:
            current_image = os.path.join(base_dir, f"{file_idx}.png")
        if os.path.exists(current_image):
            current_content.append({"type": "image", "url": current_image})
    
    # Handle voice instruction (TTS) - convert task_description to audio
    if tts_enabled and task_description:
        # Generate TTS audio from task_description
        tts_audio_path = generate_tts_audio(task_description, tts_cache_dir, tts_voice)
        if tts_audio_path and os.path.exists(tts_audio_path):
            # Add voice instruction audio
            current_content.append({"type": "audio", "url": {"path": tts_audio_path}})
            # Use simplified text prompt (task description is in audio)
            current_text = OMNI_VOICE_INSTRUCTION_PROMPT_TEMPLATE.format(
                step_index=current_step_id
            )
        else:
            # TTS failed, fall back to text instruction
            eval_logger.warning(f"TTS failed for task {doc.get('ID', 'unknown')}, using text instruction")
            current_text = OMNI_USER_PROMPT_TEMPLATE.format(
                task_description=task_description,
                step_index=current_step_id
            )
    else:
        # Standard text prompt
        current_text = OMNI_USER_PROMPT_TEMPLATE.format(
            task_description=task_description,
            step_index=current_step_id
        )
    
    current_content.append({"type": "text", "text": current_text})
    
    messages.append({"role": "user", "content": current_content})
    
    return messages


def omni_local_doc_to_visual(doc: Dict, lmms_eval_specific_kwargs: Optional[Dict] = None) -> List:
    """
    Extract all visual elements from local Omni document with dynamic history.
    
    Returns list of media paths/objects for the current step and history.
    """
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}
    
    dataset_path = lmms_eval_specific_kwargs.get("dataset_path", "")
    base_dir = lmms_eval_specific_kwargs.get("base_dir", "")
    max_history_steps = lmms_eval_specific_kwargs.get("max_history_steps", 2)
    skip_audio = lmms_eval_specific_kwargs.get("skip_audio", True)
    include_video = lmms_eval_specific_kwargs.get("include_video", True)
    include_screenshot = lmms_eval_specific_kwargs.get("include_screenshot", True)
    
    if not base_dir and dataset_path:
        base_dir = os.path.dirname(dataset_path)
    
    visuals = []
    
    episode_id = doc.get("episode_id", "")
    current_step_id = doc.get("step_id", 0)
    
    # Build history
    if dataset_path:
        full_data = _load_omni_local_dataset(dataset_path)
        history = _build_omni_history(
            full_data=full_data,
            episode_id=episode_id,
            current_step_id=current_step_id,
            base_dir=base_dir,
            max_history_steps=max_history_steps,
            include_video=include_video,
            include_screenshot=include_screenshot,
            skip_audio=skip_audio
        )
    else:
        history = []
    
    # Add history visuals
    for hist_step in history:
        if hist_step.get("video_path"):
            visuals.append(hist_step["video_path"])
        if hist_step.get("audio_path"):
            visuals.append({"path": hist_step["audio_path"]})
        if hist_step.get("image_path"):
            visuals.append(hist_step["image_path"])
    
    # Add current step visuals
    file_idx = current_step_id + 1
    
    if include_video:
        doc_video = doc.get("video_path", "")
        if doc_video:
            current_video = os.path.join(base_dir, doc_video)
        else:
            current_video = os.path.join(base_dir, f"{file_idx}.mp4")
        if os.path.exists(current_video):
            visuals.append(current_video)
    
    if not skip_audio:
        doc_audio = doc.get("audio_path", "")
        if doc_audio:
            current_audio = os.path.join(base_dir, doc_audio)
        else:
            current_audio = os.path.join(base_dir, f"{file_idx}.wav")
        if os.path.exists(current_audio):
            visuals.append({"path": current_audio})
    
    if include_screenshot:
        doc_image = doc.get("image_path", "")
        if doc_image:
            current_image = os.path.join(base_dir, doc_image)
        else:
            current_image = os.path.join(base_dir, f"{file_idx}.png")
        if os.path.exists(current_image):
            visuals.append(current_image)
    
    return visuals


def omni_local_doc_to_text(doc: Dict, lmms_eval_specific_kwargs: Optional[Dict] = None) -> str:
    """
    Generate text prompt for local Omni document (fallback).
    """
    task_description = doc.get("instruction", doc.get("task_description", ""))
    step_id = doc.get("step_id", 0)
    
    return OMNI_USER_PROMPT_TEMPLATE.format(
        task_description=task_description,
        step_index=step_id
    )


# =============================================================================
# History-Aware Prompt Templates & doc_to_messages Functions
# =============================================================================

# --- Action type name mapping ---
_ACTION_TYPE_NAMES = {
    -1: "NONE (wait/observe)",
    0: "TAP",
    1: "DOUBLE_TAP",
    2: "LONG_PRESS",
    3: "SWIPE_UP",
    4: "SWIPE_DOWN",
    5: "SWIPE_LEFT",
    6: "SWIPE_RIGHT",
    7: "INPUT",
    8: "BACK",
    9: "HOME",
    10: "TASK_COMPLETE",
    11: "TASK_IMPOSSIBLE",
}


def _build_action_history_text(history: list, coordinate_format: str = "normalized",
                               screen_width: int = 1080, screen_height: int = 2400) -> str:
    """
    Build a plain-text summary of action history for the model.
    
    Args:
        history: List of history dicts from dataset
        coordinate_format: "normalized" ([0,1000], used by both Gemini and Qwen) or "pixel" (raw pixel coords)
        screen_width: Screen width for pixel-to-normalized conversion
        screen_height: Screen height for pixel-to-normalized conversion
    
    Returns:
        Formatted string like:
        Step 1: TAP at [92, 920]
        Step 2: SWIPE_DOWN from [500, 800] to [500, 300]
    """
    if not history:
        return ""
    
    lines = []
    for h in history:
        step_idx = h.get("step_index", 0)
        action_type = h.get("result_action_type", -1)
        action_name = _ACTION_TYPE_NAMES.get(action_type, f"UNKNOWN({action_type})")
        
        # Parse coordinates from result_action_json
        action_json_str = h.get("result_action_json", "{}")
        try:
            action_json = json.loads(action_json_str) if isinstance(action_json_str, str) else action_json_str
        except (json.JSONDecodeError, TypeError):
            action_json = {}
        
        # Parse touch_xy: format "[[x1,y1],[x2,y2]]" → center point
        touch_xy_str = h.get("result_touch_xy", "")
        coord_text = ""
        
        if action_type in (0, 1, 2) and touch_xy_str:
            # TAP / DOUBLE_TAP / LONG_PRESS → compute center of bounding box
            try:
                touch_xy = json.loads(touch_xy_str)
                if isinstance(touch_xy, list) and len(touch_xy) >= 2:
                    cx = (touch_xy[0][0] + touch_xy[1][0]) / 2
                    cy = (touch_xy[0][1] + touch_xy[1][1]) / 2
                    if coordinate_format == "normalized":
                        # Already in pixel, need to get actual image dims from data
                        # But we don't have image_width/height here, so normalize 
                        # based on screen dims
                        nx = int(cx / screen_width * 1000)
                        ny = int(cy / screen_height * 1000)
                        coord_text = f" at [{nx}, {ny}]"
                    else:
                        coord_text = f" at [{int(cx)}, {int(cy)}]"
            except (json.JSONDecodeError, TypeError, IndexError):
                pass
        
        elif action_type in (3, 4, 5, 6) and touch_xy_str:
            # SWIPE → start / end coordinates
            lift_xy_str = h.get("result_lift_xy", "")
            try:
                touch_xy = json.loads(touch_xy_str) if touch_xy_str else None
                lift_xy = json.loads(lift_xy_str) if lift_xy_str else None
                if touch_xy and isinstance(touch_xy, list):
                    sx, sy = touch_xy[0][0], touch_xy[0][1]
                    if coordinate_format == "normalized":
                        sx = int(sx / screen_width * 1000)
                        sy = int(sy / screen_height * 1000)
                    start_text = f"[{int(sx)}, {int(sy)}]"
                    
                    if lift_xy and isinstance(lift_xy, list):
                        ex, ey = lift_xy[0][0], lift_xy[0][1]
                        if coordinate_format == "normalized":
                            ex = int(ex / screen_width * 1000)
                            ey = int(ey / screen_height * 1000)
                        coord_text = f" from {start_text} to [{int(ex)}, {int(ey)}]"
                    else:
                        coord_text = f" from {start_text}"
            except (json.JSONDecodeError, TypeError, IndexError):
                pass
        
        elif action_type == 7:
            # INPUT
            text = h.get("result_action_text", "") or action_json.get("text", "")
            if text:
                coord_text = f' text: "{text}"'
        
        lines.append(f"  Step {step_idx + 1}: {action_name}{coord_text}")
    
    return "\n".join(lines)


# --- Gemini History-Aware System Prompt ---
GEMINI_HISTORY_SYSTEM_PROMPT = """You are an Android GUI automation agent. Analyze the screen and output the next action.

Your inputs may include:
1. **Action History** (text) — A summary of actions already performed in previous steps. Use this to understand what has been done and avoid repeating actions.
2. **History Screenshot** (image, optional) — A screenshot from 2 steps ago, showing an earlier screen state for context.
3. **Current Video** (optional) — Screen recording leading up to the current state.
4. **Current Audio** (optional) — Device audio at the current moment.
5. **Current Screenshot** (image) — The current screen state. This is the PRIMARY input for deciding your action.

""" + GEMINI_ACTION_SCHEMA + """

CRITICAL RULES:
- You MUST ALWAYS output exactly one JSON object. NEVER output an empty response.
- Output ONLY valid JSON. No markdown, no code blocks, no explanation, no analysis text.
- Your ENTIRE response must be valid JSON and nothing else.
- Even when unsure or the task seems trivial, you MUST still output a valid JSON action. An empty response is NEVER acceptable.
- Base your action on the **Current Screenshot** (the last image in your input).
- Use the Action History and History Screenshot to understand context and progress.
- For SWIPE: provide both start_coordinate and end_coordinate.
- Coordinates must be normalized to [0, 1000] range.
- Coordinate format is [x, y] where x=horizontal, y=vertical. NOT [y, x].
- If the task appears COMPLETE or already done, you MUST output {"action_type": 10}. This is mandatory.
- If the task requires waiting/observing, output {"action_type": -1}
- If the task is impossible, output {"action_type": 11}

Examples:
- Wait/observe: {"action_type": -1}
- Tap center: {"action_type": 0, "coordinate": [500, 500]}
- Double tap: {"action_type": 1, "coordinate": [300, 400]}
- Input text: {"action_type": 7, "text": "hello"}
- Back: {"action_type": 8}
- Task done: {"action_type": 10}
"""

GEMINI_HISTORY_USER_PROMPT_TEMPLATE = """Goal: {task_description}

Step: {step_index}

{history_section}Analyze the current screenshot and determine the next action."""


# --- Qwen History-Aware System Prompt ---
QWEN_HISTORY_SYSTEM_PROMPT = """You are an Android GUI Agent. You control an Android phone by analyzing screenshots and performing actions.

Your inputs may include:
1. **Action History** (text) — A summary of actions already performed in previous steps. Use this to understand what has been done and avoid repeating actions.
2. **History Screenshot** (image, optional) — A screenshot from 2 steps ago, showing an earlier screen state for context.
3. **Current Video** (optional) — Screen recording leading up to the current state.
4. **Current Audio** (optional) — Device audio at the current moment.
5. **Current Screenshot** (image) — The current screen state. This is the PRIMARY input for deciding your action.

## Available Actions

You can perform the following actions using the tool format:

1. **click(coordinate)** - Tap at position [x, y]
2. **double_click(coordinate)** - Double tap at position [x, y]
3. **long_press(coordinate, time)** - Long press at [x, y] for `time` seconds (default: 2)
4. **swipe(coordinate, coordinate2)** - Swipe from [x1, y1] to [x2, y2]
5. **type(text)** - Input text string
6. **system_button(button)** - Press system button: "Back" or "Home"
7. **terminate(status)** - End task: "success" (completed) or "failure" (impossible)
8. **wait(time)** - Wait/observe for `time` seconds without action

## Coordinate System
- All coordinates are **normalized values in [0, 1000] range** in [x, y] format
- x: horizontal (0=left edge, 1000=right edge)
- y: vertical (0=top edge, 1000=bottom edge)
- Example: center of screen = [500, 500]

## Response Format
You must output your response in the following format:
1. First, analyze the UI elements and plan your action in a <thought> block.
2. Then, execute the action using the tool format below in a <tool_call> block.

### Examples:

**Tap:**
<thought>
I see the search button near the top-center of the screen. I need to tap it.
</thought>
<tool_call>
{"name": "click", "arguments": {"coordinate": [500, 100]}}
</tool_call>

**Swipe down:**
<thought>
I need to scroll down to see more content. I'll swipe from the middle of the screen downward.
</thought>
<tool_call>
{"name": "swipe", "arguments": {"coordinate": [500, 700], "coordinate2": [500, 300]}}
</tool_call>

**Type text:**
<thought>
The text field is focused. I need to type the search query.
</thought>
<tool_call>
{"name": "type", "arguments": {"text": "hello world"}}
</tool_call>

**Press Back:**
<thought>
I need to go back to the previous screen.
</thought>
<tool_call>
{"name": "system_button", "arguments": {"button": "Back"}}
</tool_call>

**Wait/Observe:**
<thought>
The page is still loading. I should wait for it to finish.
</thought>
<tool_call>
{"name": "wait", "arguments": {"time": 2}}
</tool_call>

**Task Complete:**
<thought>
The task has been completed successfully.
</thought>
<tool_call>
{"name": "terminate", "arguments": {"status": "success"}}
</tool_call>

## CRITICAL RULES
- You MUST ALWAYS output a response containing <thought> and <tool_call> blocks. NEVER output an empty response.
- Always output <thought> first, then <tool_call>
- Use Action History and History Screenshot to understand what has already been done
- Base your action on the **Current Screenshot** (the last image)
- Coordinates MUST be normalized values in [0, 1000] range
- For swipe: provide both start (coordinate) and end (coordinate2) positions
- If the task requires waiting/observing, use the wait action
- If the task appears COMPLETE or already done, you MUST use terminate with "success". This is mandatory.
- If the task is impossible, use terminate with "failure"
- Even when unsure, you MUST still output a valid action. An empty response is NEVER acceptable.
- Do NOT output anything outside of the <thought> and <tool_call> blocks
"""

QWEN_HISTORY_USER_PROMPT_TEMPLATE = """Task: {task_description}

Step: {step_index}

{history_section}Analyze the current screenshot and determine the next action."""


# --- Omni History-Aware System Prompt ---
OMNI_HISTORY_SYSTEM_PROMPT = """You are an Android GUI Agent. Analyze the video, audio, and screenshot to decide the next action.

Your inputs may include:
1. **Action History** (text) — A summary of actions already performed in previous steps. Use this to understand what has been done and avoid repeating actions.
2. **History Screenshot** (image, optional) — A screenshot from 2 steps ago, showing an earlier screen state for context.
3. **Current Video** (optional) — Screen recording leading up to the current state.
4. **Current Audio** (optional) — Device audio at the current moment.
5. **Current Screenshot** (image) — The current screen state. This is the PRIMARY input for deciding your action.

""" + OMNI_ACTION_SCHEMA + """

CRITICAL RULES:
- You MUST ALWAYS output exactly one JSON object. NEVER output an empty response.
- Only output JSON. No markdown, no code blocks, no explanation, no analysis text.
- Your ENTIRE response must be valid JSON and nothing else.
- Use Action History and History Screenshot to understand context and progress.
- Base your action on the **Current Screenshot** (the last image in your input).
- If the task says "wait", "observe", or requires timing, output {"action_type": -1}
- If the task appears COMPLETE or already done, you MUST output {"action_type": 10}. This is mandatory.
- If the task is impossible, output {"action_type": 11}
- Even when unsure, you MUST still output a valid JSON action. An empty response is NEVER acceptable.
"""

OMNI_HISTORY_USER_PROMPT_TEMPLATE = """Task: {task_description}

Step: {step_index}

{history_section}注意：如果任务要求"等待"、"观察"或"观看"某事发生再行动，当前步骤请输出 {{"action_type": -1}}。
Note: If the task requires waiting/observing before acting, output {{"action_type": -1}} for this step."""


# =============================================================================
# History-Aware doc_to_messages Functions
# =============================================================================

def gemini_doc_to_messages_with_history(doc: Dict, lmms_eval_specific_kwargs: Optional[Dict] = None) -> List[Dict]:
    """
    Convert document to Gemini messages with GT action history injected as text + history screenshot.
    
    Input structure per step t:
    1. System prompt (with history-aware description)
    2. Single user message containing:
       - Action History text (steps 1 to t-1)
       - History Screenshot (from step t-2, when t >= 3)
       - Current Video
       - Current Audio
       - Current Screenshot
       - Prompt text
    """
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}
    
    media_dir = lmms_eval_specific_kwargs.get("media_dir", "")
    screen_width = lmms_eval_specific_kwargs.get("screen_width", 1080)
    screen_height = lmms_eval_specific_kwargs.get("screen_height", 2400)
    
    def resolve_path(path):
        if path and media_dir and not os.path.isabs(path):
            return os.path.join(media_dir, path)
        return path
    
    messages = []
    
    # 1. System message
    messages.append({
        "role": "system",
        "content": [{"type": "text", "text": GEMINI_HISTORY_SYSTEM_PROMPT}]
    })
    
    task_description = doc.get("task_description", doc.get("instruction", ""))
    history = doc.get("history", [])
    current_step_index = doc.get("step_index", doc.get("step_id", len(history)))
    
    # 2. Build user message content
    current_content = []
    
    # --- Action History text ---
    history_section = ""
    if history:
        action_history_text = _build_action_history_text(
            history, coordinate_format="normalized",
            screen_width=screen_width, screen_height=screen_height
        )
        history_section = f"Action History:\n{action_history_text}\n\n"
    
    # --- History Screenshot (from step t-2) ---
    history_screenshot = doc.get("history_screenshot_path")
    if history_screenshot:
        # Add label text for history screenshot
        current_content.append({"type": "text", "text": "[History Screenshot (from 2 steps ago)]:"})
        current_content.append({"type": "image", "url": resolve_path(history_screenshot)})
    
    # --- Current Video ---
    current_video = doc.get("current_video") or doc.get("video") or doc.get("video_path")
    if current_video:
        current_content.append({"type": "video", "url": resolve_path(current_video)})
    
    # --- Current Audio ---
    current_audio = doc.get("current_audio") or doc.get("audio") or doc.get("audio_path")
    if current_audio:
        if isinstance(current_audio, str):
            current_content.append({"type": "audio", "url": {"path": resolve_path(current_audio)}})
        else:
            current_content.append({"type": "audio", "url": current_audio})
    
    # --- Current Screenshot ---
    current_image = doc.get("current_image") or doc.get("image") or doc.get("image_path")
    if current_image:
        if isinstance(current_image, str):
            current_content.append({"type": "image", "url": resolve_path(current_image)})
        else:
            current_content.append({"type": "image", "url": current_image})
    
    # --- Prompt text (with history section embedded) ---
    current_text = GEMINI_HISTORY_USER_PROMPT_TEMPLATE.format(
        task_description=task_description,
        step_index=current_step_index,
        history_section=history_section
    )
    current_content.append({"type": "text", "text": current_text})
    
    messages.append({"role": "user", "content": current_content})
    
    return messages


def qwen_doc_to_messages_with_history(doc: Dict, lmms_eval_specific_kwargs: Optional[Dict] = None) -> List[Dict]:
    """
    Convert document to Qwen messages with GT action history injected as text + history screenshot.
    
    Same structure as Gemini version but with:
    - Normalized [0, 1000] coordinates (same as Gemini)
    - Qwen-specific system prompt with tool call format
    """
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}
    
    media_dir = lmms_eval_specific_kwargs.get("media_dir", "")
    
    def resolve_path(path):
        if path and media_dir and not os.path.isabs(path):
            return os.path.join(media_dir, path)
        return path
    
    messages = []
    
    # 1. System message (no resolution needed — uses normalized [0,1000] coordinates)
    messages.append({
        "role": "system",
        "content": [{"type": "text", "text": QWEN_HISTORY_SYSTEM_PROMPT}]
    })
    
    task_description = doc.get("task_description", doc.get("instruction", ""))
    history = doc.get("history", [])
    current_step_index = doc.get("step_index", doc.get("step_id", len(history)))
    
    # 2. Build user message content
    current_content = []
    
    # --- Action History text (normalized [0,1000] coords, same as Gemini) ---
    history_section = ""
    if history:
        # Get screen dims from doc for normalizing GT pixel coords in history text
        hist_screen_w = doc.get("image_width") or lmms_eval_specific_kwargs.get("screen_width", 1080)
        hist_screen_h = doc.get("image_height") or lmms_eval_specific_kwargs.get("screen_height", 2400)
        action_history_text = _build_action_history_text(
            history, coordinate_format="normalized",
            screen_width=hist_screen_w, screen_height=hist_screen_h
        )
        history_section = f"Action History:\n{action_history_text}\n\n"
    
    # --- History Screenshot (from step t-2) ---
    history_screenshot = doc.get("history_screenshot_path")
    if history_screenshot:
        current_content.append({"type": "text", "text": "[History Screenshot (from 2 steps ago)]:"})
        current_content.append({"type": "image", "url": resolve_path(history_screenshot)})
    
    # --- Current Video ---
    current_video = doc.get("current_video") or doc.get("video") or doc.get("video_path")
    if current_video:
        current_content.append({"type": "video", "url": resolve_path(current_video)})
    
    # --- Current Audio ---
    current_audio = doc.get("current_audio") or doc.get("audio") or doc.get("audio_path")
    if current_audio:
        if isinstance(current_audio, str):
            current_content.append({"type": "audio", "url": {"path": resolve_path(current_audio)}})
        else:
            current_content.append({"type": "audio", "url": current_audio})
    
    # --- Current Screenshot ---
    current_image = doc.get("current_image") or doc.get("image") or doc.get("image_path")
    if current_image:
        if isinstance(current_image, str):
            current_content.append({"type": "image", "url": resolve_path(current_image)})
        else:
            current_content.append({"type": "image", "url": current_image})
    
    # --- Prompt text ---
    current_text = QWEN_HISTORY_USER_PROMPT_TEMPLATE.format(
        task_description=task_description,
        step_index=current_step_index,
        history_section=history_section
    )
    current_content.append({"type": "text", "text": current_text})
    
    messages.append({"role": "user", "content": current_content})
    
    return messages


def omni_doc_to_messages_with_history(doc: Dict, lmms_eval_specific_kwargs: Optional[Dict] = None) -> List[Dict]:
    """
    Convert document to Omni messages with GT action history injected as text + history screenshot.
    
    Same structure as Gemini version (normalized coords [0, 1000]).
    """
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}
    
    media_dir = lmms_eval_specific_kwargs.get("media_dir", "")
    screen_width = lmms_eval_specific_kwargs.get("screen_width", 1080)
    screen_height = lmms_eval_specific_kwargs.get("screen_height", 2400)
    
    def resolve_path(path):
        if path and media_dir and not os.path.isabs(path):
            return os.path.join(media_dir, path)
        return path
    
    messages = []
    
    # 1. System message
    messages.append({
        "role": "system",
        "content": [{"type": "text", "text": OMNI_HISTORY_SYSTEM_PROMPT}]
    })
    
    task_description = doc.get("task_description", doc.get("instruction", ""))
    history = doc.get("history", [])
    current_step_index = doc.get("step_index", doc.get("step_id", len(history)))
    
    # 2. Build user message content
    current_content = []
    
    # --- Action History text (normalized coords for Omni) ---
    history_section = ""
    if history:
        action_history_text = _build_action_history_text(
            history, coordinate_format="normalized",
            screen_width=screen_width, screen_height=screen_height
        )
        history_section = f"Action History:\n{action_history_text}\n\n"
    
    # --- History Screenshot (from step t-2) ---
    history_screenshot = doc.get("history_screenshot_path")
    if history_screenshot:
        current_content.append({"type": "text", "text": "[History Screenshot (from 2 steps ago)]:"})
        current_content.append({"type": "image", "url": resolve_path(history_screenshot)})
    
    # --- Current Video ---
    current_video = doc.get("current_video") or doc.get("video") or doc.get("video_path")
    if current_video:
        current_content.append({"type": "video", "url": resolve_path(current_video)})
    
    # --- Current Audio ---
    current_audio = doc.get("current_audio") or doc.get("audio") or doc.get("audio_path")
    if current_audio:
        if isinstance(current_audio, str):
            current_content.append({"type": "audio", "url": {"path": resolve_path(current_audio)}})
        else:
            current_content.append({"type": "audio", "url": current_audio})
    
    # --- Current Screenshot ---
    current_image = doc.get("current_image") or doc.get("image") or doc.get("image_path")
    if current_image:
        if isinstance(current_image, str):
            current_content.append({"type": "image", "url": resolve_path(current_image)})
        else:
            current_content.append({"type": "image", "url": current_image})
    
    # --- Prompt text ---
    current_text = OMNI_HISTORY_USER_PROMPT_TEMPLATE.format(
        task_description=task_description,
        step_index=current_step_index,
        history_section=history_section
    )
    current_content.append({"type": "text", "text": current_text})
    
    messages.append({"role": "user", "content": current_content})
    
    return messages
