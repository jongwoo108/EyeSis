# Quick script to add match_with_bank_detailed function
with open('backend/services/bank_manager.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Function to add
func = '''

def match_with_bank_detailed(face_emb, gallery):
    """Match face embedding with gallery and return detailed info"""
    if not gallery:
        return "unknown", 0.0, 0.0
    face_emb = l2_normalize(face_emb.astype("float32"))
    sims = []
    for pid, edata in gallery.items():
        if edata.ndim == 2:
            s = float((edata @ face_emb).max())
        else:
            s = float(edata @ face_emb)
        sims.append((pid, s))
    sims.sort(key=lambda x: x[1], reverse=True)
    if not sims:
        return "unknown", 0.0, 0.0
    elif len(sims) == 1:
        return sims[0][0], sims[0][1], 0.0
    else:
        return sims[0][0], sims[0][1], sims[1][1]

'''

# Insert before save_angle_separated_banks
new_content = content.replace('\ndef save_angle_separated_banks', func + 'def save_angle_separated_banks')

with open('backend/services/bank_manager.py', 'w', encoding='utf-8') as f:
    f.write(new_content)

print("Function added successfully!")
