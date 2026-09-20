# Exercises the bridge headless, against a serving session:
#   orla serve examples/mesh/scene.osn --listen 127.0.0.1:0   (note the port)
#   Blender -b --python bridges/blender/test_bridge.py -- --port <port> --out <dir>
# It pushes a textured cube as the Host's files and reads them back, then
# pulls the torus and the picture from the session and sets a param.
import importlib.util
import os
import struct
import sys

import bpy

args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
port = int(args[args.index("--port") + 1])
out = args[args.index("--out") + 1]

here = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("orla_bridge", os.path.join(here, "orla_bridge.py"))
bridge = importlib.util.module_from_spec(spec)
sys.modules["orla_bridge"] = bridge
spec.loader.exec_module(bridge)
bridge.register()

# A cube with a UV map and a base colour image, moved and scaled.
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.mesh.primitive_cube_add(size=2.0)
cube = bpy.context.active_object
cube.location = (0.0, 0.0, 1.0)
cube.scale = (0.5, 0.5, 0.5)
image = bpy.data.images.new("planks", 8, 8)
image.pixels = [v for _ in range(64) for v in (0.8, 0.5, 0.2, 1.0)]
material = bpy.data.materials.new("planks")
material.use_nodes = True
tex = material.node_tree.nodes.new("ShaderNodeTexImage")
tex.image = image
bsdf = material.node_tree.nodes["Principled BSDF"]
material.node_tree.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
cube.data.materials.append(material)

paths = bridge.push_object(cube, out, bpy.context.evaluated_depsgraph_get())
assert [os.path.basename(p) for p in paths] == ["Cube.vertices.field", "Cube.triangles.field", "Cube.albedo.png"], paths
shape, extent, schema, words = bridge.read_field(paths[0])
assert (shape, extent, schema) == ("Rows", (36, 1, 1), "f,f,f,f,f,f,f,f"), (shape, extent, schema)
floats = struct.unpack("<288f", words)
xs = floats[0::8]
zs = floats[2::8]
assert min(xs) == -0.5 and max(xs) == 0.5, (min(xs), max(xs))
assert min(zs) == 0.5 and max(zs) == 1.5, (min(zs), max(zs))
shape, extent, schema, words = bridge.read_field(paths[1])
assert (shape, extent, schema) == ("Rows", (12, 1, 1), "i,i,i"), (shape, extent, schema)
assert open(paths[2], "rb").read(8) == b"\x89PNG\r\n\x1a\n"
print("push: ok")

client = bridge.OrlaClient("127.0.0.1", port)
status = client.call("status")
assert status["nodes"] >= 3, status
client.call("evaluate", time=0.0)
torus = bridge.pull_mesh(client, "torus", "faces", name="Torus")
assert len(torus.data.vertices) == 1152 and len(torus.data.polygons) == 2304, (len(torus.data.vertices), len(torus.data.polygons))
frame = bridge.pull_image(client)
assert frame.size[0] == 320 and frame.size[1] == 240, frame.size
client.call("param", name="exposure", value=1.5)
client.close()
print("pull: ok")
print("bridge: ok")
