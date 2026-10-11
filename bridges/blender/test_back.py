"""Opens a GLB Entelechy wrote and says what Blender found in it.

    blender -b --python bridges/blender/test_back.py -- --glb /tmp/arm.glb

The loop this closes is the one someone actually works in: rig in
Blender, push or import into Entelechy, and open what Entelechy writes back.
"""

import os
import sys

import bpy

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
glb = argv[argv.index("--glb") + 1] if "--glb" in argv else "/tmp/arm.glb"

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=glb)

armatures = [o for o in bpy.data.objects if o.type == "ARMATURE"]
assert len(armatures) == 1, [(o.name, o.type) for o in bpy.data.objects]
armature = armatures[0]
# Blender's own importer leaves a bone-shape mesh behind (it does the
# same for files Blender exported), so the rig's mesh is the one hanging
# off the armature rather than the only one there is.
meshes = [
    o
    for o in bpy.data.objects
    if o.type == "MESH"
    and (o.parent is armature or any(m.type == "ARMATURE" and m.object is armature for m in o.modifiers))
]
assert len(meshes) == 1, [(o.name, o.type) for o in bpy.data.objects]
mesh = meshes[0]
bones = armature.data.bones
print(f"opened: {len(mesh.data.vertices)} vertices, {len(bones)} bones")

# The mesh hangs off the bones, by the groups that name them.
assert any(m.type == "ARMATURE" and m.object is armature for m in mesh.modifiers), [
    m.type for m in mesh.modifiers
]
names = {group.name for group in mesh.vertex_groups}
assert names, "the mesh has no vertex groups"
weighted = sum(1 for v in mesh.data.vertices if v.groups)
assert weighted == len(mesh.data.vertices), (weighted, len(mesh.data.vertices))
print(f"weighted: every vertex, over {len(names)} group(s)")

# The clips came with it, and they move something.
actions = [a for a in bpy.data.actions]
assert len(actions) >= 2, [a.name for a in actions]
moving = 0
for action in actions:
    curves = list(getattr(action, "fcurves", []) or [])
    if not curves:
        for layer in getattr(action, "layers", []):
            for strip in getattr(layer, "strips", []):
                for bag in getattr(strip, "channelbags", []):
                    curves.extend(bag.fcurves)
    keys = sum(len(curve.keyframe_points) for curve in curves)
    print(f"clip {action.name}: {len(curves)} curve(s), {keys} key(s)")
    if keys:
        moving += 1
assert moving >= 2, "the clips carry no keys"

print("RESULT: ok")
