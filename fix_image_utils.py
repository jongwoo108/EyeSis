#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Add missing functions to image_utils.py"""

# Read existing content
with open('backend/utils/image_utils.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Functions to add
new_functions = '''


def is_diverse_angle(collected_angles, new_angle):
    """Check if new angle is diverse from collected angles"""
    if not collected_angles:
        return True
    max_counts = {
        "left_profile": 50,
        "right_profile": 50,
        "front": 50,
        "left": 50,
        "right": 50,
        "top": 50
    }
    return collected_angles.count(new_angle) < max_counts.get(new_angle, 50)


def is_all_angles_collected(collected_angles):
    """Check if all required angles have been collected"""
    from collections import defaultdict
    required = {"front": 1, "left": 1, "right": 1, "top": 1}
    counts = defaultdict(int)
    for a in collected_angles:
        counts[a] += 1
    return all(counts[a] >= required[a] for a in required)
'''

# Write back with new functions
with open('backend/utils/image_utils.py', 'w', encoding='utf-8') as f:
    f.write(content + new_functions)

print("✅ Functions added successfully!")
