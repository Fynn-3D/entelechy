"""Pushes a rig out of Blender and exports the same rig as glTF, so the
two ways into Orla can be held against each other.

    blender -b --python bridges/blender/test_rig.py -- --out /tmp/orla-rig

writes `<out>/pushed/` (the bridge's rows) and `<out>/arm.glb` (for
`orla import`). `crates/interface/orla/tests/bridges.rs` poses both and compares the
pictures; run by hand, the lines below say what was written.
"""

import importlib.util
import os
import sys

import bpy

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)


def load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(here, f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
out = argv[argv.index("--out") + 1] if "--out" in argv else "/tmp/orla-rig"

bridge = load("orla_bridge")
maker = load("make_rig")

mesh, armature = maker.build()
print(f"rig: {len(armature.data.bones)} bones, {len(mesh.data.vertices)} vertices")

pushed = os.path.join(out, "pushed")
paths = bridge.push_rig(mesh, pushed)
for path in paths:
    print(f"pushed {os.path.basename(path)}")
names = [os.path.basename(p) for p in paths]
assert any(n.endswith(".skeleton.field") for n in names), "the joints"
assert any(n.endswith(".weights.field") for n in names), "what a vertex hangs off"
assert sum(n.endswith(".clip.field") for n in names) == 2, "a clip per action"

# The pose it stands in, as Live pushes it: one clip of one frame.
live = bridge.push_rig(mesh, os.path.join(out, "live"), clips=False)
assert any(os.path.basename(p) == "Arm.pose.clip.field" for p in live), [
    os.path.basename(p) for p in live
]
print(f"live: {len(live)} rows for the frame it stands on")

# Live: posing the armature — which is what Pose Mode makes active —
# pushes the mesh it deforms, not nothing.
import math

bridge.register()
scene = bpy.context.scene
scene.orla.assets_dir = os.path.join(out, "live")
scene.orla.live = True
bpy.context.view_layer.objects.active = armature
pose_file = os.path.join(out, "live", "Arm.pose.clip.field")
before = open(pose_file, "rb").read()
bridge._last_push = 0.0
armature.pose.bones["upper"].rotation_mode = "XYZ"
armature.pose.bones["upper"].rotation_euler = (math.radians(55.0), 0.0, 0.0)
bpy.context.view_layer.update()
after = open(pose_file, "rb").read()
assert after != before, "posing the armature pushed nothing"
print("live: posing the armature pushed the mesh it deforms")
scene.orla.live = False

glb = os.path.join(out, "arm.glb")
bpy.ops.export_scene.gltf(
    filepath=glb,
    export_format="GLB",
    export_animations=True,
    export_animation_mode="ACTIONS",
    export_skins=True,
    export_apply=False,
)
print(f"exported {os.path.basename(glb)} at {bpy.context.scene.render.fps} frames a second")
print("RESULT: ok")
