# Orla bridge for Blender: a client of Orla's public contracts and nothing
# more. It writes meshes as the Host's files (which a scene's inputs read
# and the Host watches), and talks to a running `orla-player --serve
# --listen` or `orla serve --listen` over the creation protocol: JSON
# lines on a socket. Nothing here knows how Orla computes.
#
# Install: Edit > Preferences > Add-ons > Install, pick this file. The
# panel is in the 3D view's sidebar (N), tab "Orla".

bl_info = {
    "name": "Orla Bridge",
    "author": "Orla",
    "version": (0, 1, 0),
    "blender": (4, 0, 0),
    "location": "3D View > Sidebar > Orla",
    "description": "Push meshes to an Orla scene as its files, pull rows and pictures back, drive a running window",
    "category": "Import-Export",
}

import json
import os
import socket
import struct
import time

import bpy
import bmesh
from mathutils import Vector


# ---------------------------------------------------------------- Field files

def write_field(path, shape, extent, schema, words):
    """Writes a Field file: `shape` is Rows/Grid/Volume, `extent` (w, h, d),
    `schema` like "f,f,f", `words` the little-endian 32-bit words as bytes."""
    header = f"ORLAFIELD 1\n{shape} {extent[0]} {extent[1]} {extent[2]} {schema}\n".encode()
    with open(path, "wb") as f:
        f.write(header)
        f.write(words)


def read_field(path):
    """Reads a Field file into (shape, extent, schema, words bytes)."""
    with open(path, "rb") as f:
        data = f.read()
    first = data.index(b"\n")
    if data[:first] != b"ORLAFIELD 1":
        raise ValueError(f"{path} is not a Field file")
    second = data.index(b"\n", first + 1)
    shape, w, h, d, schema = data[first + 1:second].decode().split()
    return shape, (int(w), int(h), int(d)), schema, data[second + 1:]


# --------------------------------------------------------------- mesh rows

def mesh_rows(obj, depsgraph=None):
    """The object's mesh as the runtime's rows: one vertex per triangle
    corner, position in world space, split normal, uv with v flipped to
    the top-left convention; and the triangles as indices. Returns
    (vertex words, vertex count, triangle words, triangle count)."""
    evaluated = obj.evaluated_get(depsgraph) if depsgraph else obj
    mesh = evaluated.to_mesh()
    try:
        bm = bmesh.new()
        bm.from_mesh(mesh)
        bmesh.ops.triangulate(bm, faces=bm.faces[:])
        tri = bpy.data.meshes.new("orla_tmp")
        bm.to_mesh(tri)
        bm.free()
        tri.calc_loop_triangles()
        if hasattr(tri, "calc_normals_split"):
            tri.calc_normals_split()
        matrix = obj.matrix_world
        normal_matrix = matrix.to_3x3().inverted_safe().transposed()
        uv_layer = tri.uv_layers.active
        positions = []
        for lt in tri.loop_triangles:
            for k in range(3):
                loop = lt.loops[k]
                vertex = tri.vertices[tri.loops[loop].vertex_index]
                p = matrix @ vertex.co
                n = (normal_matrix @ Vector(lt.split_normals[k])).normalized()
                uv = uv_layer.data[loop].uv if uv_layer else (0.0, 0.0)
                positions.extend([p.x, p.y, p.z, n.x, n.y, n.z, uv[0], 1.0 - uv[1]])
        count = len(tri.loop_triangles) * 3
        vertex_words = struct.pack(f"<{len(positions)}f", *positions)
        triangle_words = struct.pack(f"<{count}i", *range(count))
        bpy.data.meshes.remove(tri)
        return vertex_words, count, triangle_words, count // 3
    finally:
        evaluated.to_mesh_clear()


def base_colour_image(obj):
    """The image wired into the active material's Base Color, if any."""
    material = obj.active_material
    if not material or not material.use_nodes:
        return None
    for node in material.node_tree.nodes:
        if node.type == "BSDF_PRINCIPLED":
            link = next((l for l in material.node_tree.links if l.to_node == node and l.to_socket.name == "Base Color"), None)
            if link and link.from_node.type == "TEX_IMAGE":
                return link.from_node.image
    for node in material.node_tree.nodes:
        if node.type == "TEX_IMAGE" and node.image:
            return node.image
    return None


def push_object(obj, assets_dir, depsgraph=None):
    """Writes <name>.vertices.field, <name>.triangles.field and, when the
    material has one, <name>.albedo.png into assets_dir. Returns the paths."""
    os.makedirs(assets_dir, exist_ok=True)
    name = bpy.path.clean_name(obj.name)
    vertices, count, triangles, faces = mesh_rows(obj, depsgraph)
    paths = []
    path = os.path.join(assets_dir, f"{name}.vertices.field")
    write_field(path, "Rows", (count, 1, 1), "f,f,f,f,f,f,f,f", vertices)
    paths.append(path)
    path = os.path.join(assets_dir, f"{name}.triangles.field")
    write_field(path, "Rows", (faces, 1, 1), "i,i,i", triangles)
    paths.append(path)
    image = base_colour_image(obj)
    if image is not None:
        path = os.path.join(assets_dir, f"{name}.albedo.png")
        settings = bpy.context.scene.render.image_settings
        previous = settings.file_format
        settings.file_format = "PNG"
        try:
            image.save_render(path)
        finally:
            settings.file_format = previous
        paths.append(path)
    return paths


# ---------------------------------------------------------------- protocol

class OrlaClient:
    """One connection to a serving session: JSON lines in, one response
    per request in order."""

    def __init__(self, host, port, timeout=30.0):
        self.sock = socket.create_connection((host, port), timeout=timeout)
        self.buffer = b""
        self.next_id = 0

    def call(self, op, **fields):
        self.next_id += 1
        request = {"id": self.next_id, "op": op, **fields}
        self.sock.sendall((json.dumps(request) + "\n").encode())
        while b"\n" not in self.buffer:
            chunk = self.sock.recv(65536)
            if not chunk:
                raise ConnectionError("Orla closed the connection")
            self.buffer += chunk
        line, self.buffer = self.buffer.split(b"\n", 1)
        response = json.loads(line)
        if "error" in response:
            raise RuntimeError(response["error"].get("message", str(response["error"])))
        return response["ok"]

    def close(self):
        try:
            self.call("quit")
        except Exception:
            pass
        self.sock.close()


def connect(context):
    prefs = context.preferences.addons[__name__].preferences
    return OrlaClient(prefs.host, prefs.port)


def rows_to_mesh(name, vertex_words, vertex_stride, triangle_words):
    """Builds a mesh from vertex rows of `vertex_stride` floats (8: position,
    normal, uv; 9: clip xyzw, normal, uv) and triangle rows of three ints."""
    import base64
    vertices = struct.unpack(f"<{len(vertex_words) // 4}f", vertex_words)
    indices = struct.unpack(f"<{len(triangle_words) // 4}i", triangle_words)
    count = len(vertices) // vertex_stride
    positions = [tuple(vertices[i * vertex_stride:i * vertex_stride + 3]) for i in range(count)]
    faces = [tuple(indices[i:i + 3]) for i in range(0, len(indices), 3)]
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(positions, [], faces)
    uv_at = vertex_stride - 2
    uv_layer = mesh.uv_layers.new(name="UVMap")
    for poly in mesh.polygons:
        for loop_index in poly.loop_indices:
            v = mesh.loops[loop_index].vertex_index
            u, t = vertices[v * vertex_stride + uv_at], vertices[v * vertex_stride + uv_at + 1]
            uv_layer.data[loop_index].uv = (u, 1.0 - t)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    return obj


def pull_mesh(client, vertex_node, triangle_node, name="OrlaMesh"):
    import base64
    vertices = client.call("read", node=vertex_node, port="result")
    triangles = client.call("read", node=triangle_node, port="result")
    stride = len(vertices["schema"].split(","))
    if stride not in (8, 9):
        raise RuntimeError(f"{vertex_node} has {stride} words per row; a mesh has 8 or 9")
    return rows_to_mesh(name, base64.b64decode(vertices["data"]), stride, base64.b64decode(triangles["data"]))


def pull_image(client, name="OrlaFrame"):
    import base64
    frame = client.call("read", format="ppm")
    data = base64.b64decode(frame["data"])
    parts = data.split(b"\n", 3)
    width, height = map(int, parts[1].split())
    pixels = parts[3]
    image = bpy.data.images.get(name)
    if image is None or image.size[0] != width or image.size[1] != height:
        if image is not None:
            bpy.data.images.remove(image)
        image = bpy.data.images.new(name, width, height)
    floats = [0.0] * (width * height * 4)
    for y in range(height):
        row = pixels[y * width * 3:(y + 1) * width * 3]
        base = (height - 1 - y) * width * 4
        for x in range(width):
            floats[base + x * 4] = row[x * 3] / 255.0
            floats[base + x * 4 + 1] = row[x * 3 + 1] / 255.0
            floats[base + x * 4 + 2] = row[x * 3 + 2] / 255.0
            floats[base + x * 4 + 3] = 1.0
    image.pixels = floats
    image.update()
    return image


def parse_value(text):
    text = text.strip()
    if text in ("true", "false"):
        return text == "true"
    if text.startswith("(") or text.startswith("["):
        return [float(v) for v in text.strip("()[]").split(",")]
    try:
        return int(text)
    except ValueError:
        return float(text)


# -------------------------------------------------------------------- UI

class OrlaPreferences(bpy.types.AddonPreferences):
    bl_idname = __name__
    host: bpy.props.StringProperty(name="Host", default="127.0.0.1")
    port: bpy.props.IntProperty(name="Port", default=7788, min=1, max=65535)

    def draw(self, context):
        row = self.layout.row()
        row.prop(self, "host")
        row.prop(self, "port")


class OrlaSettings(bpy.types.PropertyGroup):
    assets_dir: bpy.props.StringProperty(name="Assets", subtype="DIR_PATH", description="The Orla scene's assets directory; inputs there are read and watched")
    live: bpy.props.BoolProperty(name="Live", default=False, description="Push the active object again whenever its geometry changes")
    vertex_node: bpy.props.StringProperty(name="Vertices node", default="torus")
    triangle_node: bpy.props.StringProperty(name="Triangles node", default="faces")
    param_name: bpy.props.StringProperty(name="Param", default="exposure")
    param_value: bpy.props.StringProperty(name="Value", default="1.0")


class ORLA_OT_push(bpy.types.Operator):
    bl_idname = "orla.push"
    bl_label = "Push mesh to Orla"
    bl_description = "Write the active object as vertex and triangle rows (and its base colour as PNG) into the assets directory"

    def execute(self, context):
        obj = context.active_object
        if obj is None or obj.type != "MESH":
            self.report({"ERROR"}, "select a mesh object")
            return {"CANCELLED"}
        settings = context.scene.orla
        if not settings.assets_dir:
            self.report({"ERROR"}, "set the assets directory")
            return {"CANCELLED"}
        paths = push_object(obj, bpy.path.abspath(settings.assets_dir), context.evaluated_depsgraph_get())
        self.report({"INFO"}, "wrote " + ", ".join(os.path.basename(p) for p in paths))
        return {"FINISHED"}


class ORLA_OT_pull_mesh(bpy.types.Operator):
    bl_idname = "orla.pull_mesh"
    bl_label = "Pull mesh from Orla"
    bl_description = "Read two nodes' rows from the running session and build a mesh"

    def execute(self, context):
        settings = context.scene.orla
        try:
            client = connect(context)
            obj = pull_mesh(client, settings.vertex_node, settings.triangle_node)
            client.close()
        except Exception as error:
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}
        self.report({"INFO"}, f"pulled {obj.name}: {len(obj.data.vertices)} vertices")
        return {"FINISHED"}


class ORLA_OT_pull_image(bpy.types.Operator):
    bl_idname = "orla.pull_image"
    bl_label = "Pull picture from Orla"
    bl_description = "Read the session's picture into an Image datablock named OrlaFrame"

    def execute(self, context):
        try:
            client = connect(context)
            image = pull_image(client)
            client.close()
        except Exception as error:
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}
        self.report({"INFO"}, f"pulled {image.name} {image.size[0]}x{image.size[1]}")
        return {"FINISHED"}


class ORLA_OT_param(bpy.types.Operator):
    bl_idname = "orla.param"
    bl_label = "Set param"
    bl_description = "Set a graph parameter in the running session"

    def execute(self, context):
        settings = context.scene.orla
        try:
            client = connect(context)
            client.call("param", name=settings.param_name, value=parse_value(settings.param_value))
            client.close()
        except Exception as error:
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}
        return {"FINISHED"}


class ORLA_OT_publish(bpy.types.Operator):
    bl_idname = "orla.publish"
    bl_label = "Publish text as Kernel"
    bl_description = "Publish the text editor's active text as a Kernel in the running session"

    def execute(self, context):
        text = next((area.spaces.active.text for area in context.screen.areas if area.type == "TEXT_EDITOR" and area.spaces.active.text), None)
        if text is None:
            self.report({"ERROR"}, "open a Kernel in the text editor")
            return {"CANCELLED"}
        try:
            client = connect(context)
            published = client.call("publish", source=text.as_string(), expect="any")
            client.close()
        except Exception as error:
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}
        self.report({"INFO"}, f"published {published['id']} generation {published['generation']}")
        return {"FINISHED"}


class ORLA_PT_panel(bpy.types.Panel):
    bl_label = "Orla"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Orla"

    def draw(self, context):
        settings = context.scene.orla
        layout = self.layout
        box = layout.box()
        box.label(text="Files: the scene's inputs")
        box.prop(settings, "assets_dir")
        box.operator("orla.push")
        box.prop(settings, "live")
        box = layout.box()
        box.label(text="Session: a running window")
        row = box.row()
        row.prop(settings, "vertex_node")
        row.prop(settings, "triangle_node")
        box.operator("orla.pull_mesh")
        box.operator("orla.pull_image")
        row = box.row()
        row.prop(settings, "param_name")
        row.prop(settings, "param_value")
        box.operator("orla.param")
        box.operator("orla.publish")


_last_push = 0.0


def _on_depsgraph(scene, depsgraph):
    global _last_push
    settings = getattr(scene, "orla", None)
    if settings is None or not settings.live or not settings.assets_dir:
        return
    obj = bpy.context.view_layer.objects.active
    if obj is None or obj.type != "MESH":
        return
    changed = any(update.id.original == obj.data or update.id.original == obj for update in depsgraph.updates if update.is_updated_geometry or update.is_updated_transform)
    if not changed or time.monotonic() - _last_push < 0.2:
        return
    _last_push = time.monotonic()
    try:
        push_object(obj, bpy.path.abspath(settings.assets_dir), depsgraph)
    except Exception as error:
        print("orla bridge: push failed:", error)


classes = (OrlaPreferences, OrlaSettings, ORLA_OT_push, ORLA_OT_pull_mesh, ORLA_OT_pull_image, ORLA_OT_param, ORLA_OT_publish, ORLA_PT_panel)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.orla = bpy.props.PointerProperty(type=OrlaSettings)
    if _on_depsgraph not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(_on_depsgraph)


def unregister():
    if _on_depsgraph in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(_on_depsgraph)
    del bpy.types.Scene.orla
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
