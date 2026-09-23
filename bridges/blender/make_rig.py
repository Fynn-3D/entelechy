"""Authors the rigged, animated arm `examples/skin` is built from.

Two bones, a box weighted to them, and two actions to blend between.
Run it headless and import what it writes:

    blender -b --python bridges/blender/make_rig.py -- --out examples/skin/source/arm.glb
    orla import examples/skin/source/arm.glb --out examples/skin/assets

It is here rather than beside the example because it is Blender's
language, not Orla's, and because it is the shortest statement of what a
rig has to be for the bridge to carry it: bones in a chain, a mesh
parented with automatic weights, and actions on the armature.
"""
import bpy, sys, math
from mathutils import Vector

def build():
    """The rig, in an empty file: returns (mesh object, armature)."""
    bpy.ops.wm.read_factory_settings(use_empty=True)

    # Two bones in a chain along +y, each a unit long.
    bpy.ops.object.armature_add(enter_editmode=True, location=(0, 0, 0))
    armature = bpy.context.object
    armature.name = "Rig"
    bones = armature.data.edit_bones
    root = bones[0]
    root.name = "lower"
    root.head = Vector((0, 0, 0))
    root.tail = Vector((0, 1, 0))
    upper = bones.new("upper")
    upper.head = Vector((0, 1, 0))
    upper.tail = Vector((0, 2, 0))
    upper.parent = root
    upper.use_connect = True
    bpy.ops.object.mode_set(mode="OBJECT")

    # A box around the chain, cut across so it can bend.
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, 1, 0))
    mesh = bpy.context.object
    mesh.name = "Arm"
    mesh.scale = (0.22, 2.0, 0.22)
    bpy.ops.object.transform_apply(scale=True)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.subdivide(number_cuts=15)
    bpy.ops.object.mode_set(mode="OBJECT")

    # Weighted to the bones automatically.
    mesh.select_set(True)
    armature.select_set(True)
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.parent_set(type="ARMATURE_AUTO")

    # Two clips, so a scene has something to blend between.
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode="POSE")
    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = 25
    pose = armature.pose.bones
    for bone in ("lower", "upper"):
        pose[bone].rotation_mode = "XYZ"
    armature.animation_data_create()


    def clip(name, keys):
        """One action from (frame, lower, upper) angles, in degrees."""
        action = bpy.data.actions.new(name)
        action.use_fake_user = True
        armature.animation_data.action = action
        if hasattr(armature.animation_data, "action_slot"):
            slot = action.slots.new(id_type="OBJECT", name=armature.name)
            armature.animation_data.action_slot = slot
        for frame, lower, upper in keys:
            scene.frame_set(frame)
            pose["lower"].rotation_euler = (math.radians(lower), 0.0, 0.0)
            pose["upper"].rotation_euler = (math.radians(upper), 0.0, 0.0)
            pose["lower"].keyframe_insert("rotation_euler", frame=frame)
            pose["upper"].keyframe_insert("rotation_euler", frame=frame)
        return action


    clip("swing", ((1, 0.0, 0.0), (13, -20.0, 80.0), (25, 0.0, 0.0)))
    clip("curl", ((1, 0.0, 0.0), (13, 35.0, 110.0), (25, 0.0, 0.0)))
    bpy.ops.object.mode_set(mode="OBJECT")
    scene.frame_set(1)

    return mesh, armature


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    out = argv[argv.index("--out") + 1] if "--out" in argv else "/tmp/arm.glb"
    build()
    bpy.ops.export_scene.gltf(
        filepath=out,
        export_format="GLB",
        export_animations=True,
        export_animation_mode="ACTIONS",
        export_skins=True,
        export_apply=False,
    )
    print(f"RESULT: wrote {out}")
