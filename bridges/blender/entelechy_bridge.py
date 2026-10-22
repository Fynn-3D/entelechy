# Entelechy bridge for Blender: a client of Entelechy's public contracts and nothing
# more. It writes meshes as the Host's files (which a scene's inputs read
# and the Host watches), and talks to a running `entelechy-player --serve
# --listen` or `entelechy serve --listen` over the creation protocol: JSON
# lines on a socket. Nothing here knows how Entelechy computes.
#
# Install: Edit > Preferences > Add-ons > Install, pick this file. The
# panel is in the 3D view's sidebar (N), tab "Entelechy".

bl_info = {
    "name": "Entelechy Bridge",
    "author": "Entelechy",
    "version": (0, 1, 0),
    "blender": (4, 0, 0),
    "location": "3D View > Sidebar > Entelechy",
    "description": "Push meshes to an Entelechy scene as its files, pull rows and pictures back, drive a running window",
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
    header = f"ENTELECHYFIELD 1\n{shape} {extent[0]} {extent[1]} {extent[2]} {schema}\n".encode()
    with open(path, "wb") as f:
        f.write(header)
        f.write(words)


def read_field(path):
    """Reads a Field file into (shape, extent, schema, words bytes)."""
    with open(path, "rb") as f:
        data = f.read()
    first = data.index(b"\n")
    if data[:first] != b"ENTELECHYFIELD 1":
        raise ValueError(f"{path} is not a Field file")
    second = data.index(b"\n", first + 1)
    shape, w, h, d, schema = data[first + 1:second].decode().split()
    return shape, (int(w), int(h), int(d)), schema, data[second + 1:]


# --------------------------------------------------------------- mesh rows

# Blender is z-up; Entelechy's scenes, like glTF, are y-up. The bridge turns
# axes on the way out and back, so a model stands the same on both sides:
# Blender (x, y, z) is Entelechy (x, z, -y).
AXES_YUP = "YUP"
AXES_NONE = "NONE"


def to_entelechy(v, axes):
    return (v[0], v[2], -v[1]) if axes == AXES_YUP else (v[0], v[1], v[2])


def to_blender(v, axes):
    return (v[0], -v[2], v[1]) if axes == AXES_YUP else (v[0], v[1], v[2])


def mesh_rows(obj, depsgraph=None, axes=AXES_YUP, matrix=None, corners=None):
    """The object's mesh as the runtime's rows: one vertex per triangle
    corner, position in world space, split normal, uv with v flipped to
    the top-left convention; and the triangles as indices. Returns
    (vertex words, vertex count, triangle words, triangle count).

    `matrix` replaces the object's own world matrix, which a rig uses to
    push the mesh in its armature's space. Pass a list as `corners` and
    it is filled with the original vertex each corner came from, which is
    how the weights line up with these rows.

    With no `depsgraph` the mesh comes as it is authored, before its
    modifiers: a rigged mesh is pushed at rest, since its joints pose it.
    """
    evaluated = obj.evaluated_get(depsgraph) if depsgraph else obj
    mesh = evaluated.to_mesh()
    try:
        bm = bmesh.new()
        bm.from_mesh(mesh)
        bmesh.ops.triangulate(bm, faces=bm.faces[:])
        tri = bpy.data.meshes.new("entelechy_tmp")
        bm.to_mesh(tri)
        bm.free()
        tri.calc_loop_triangles()
        if hasattr(tri, "calc_normals_split"):
            tri.calc_normals_split()
        matrix = obj.matrix_world if matrix is None else matrix
        normal_matrix = matrix.to_3x3().inverted_safe().transposed()
        uv_layer = tri.uv_layers.active
        # Smooth or flat as the mesh says: corner normals (4.1 and later),
        # split normals before that.
        corner_normals = getattr(tri, "corner_normals", None)
        positions = []
        for lt in tri.loop_triangles:
            for k in range(3):
                loop = lt.loops[k]
                vertex = tri.vertices[tri.loops[loop].vertex_index]
                p = to_entelechy(matrix @ vertex.co, axes)
                raw = corner_normals[loop].vector if corner_normals is not None else lt.split_normals[k]
                n = to_entelechy((normal_matrix @ Vector(raw)).normalized(), axes)
                uv = uv_layer.data[loop].uv if uv_layer else (0.0, 0.0)
                positions.extend([p[0], p[1], p[2], n[0], n[1], n[2], uv[0], 1.0 - uv[1]])
                if corners is not None:
                    corners.append(tri.loops[loop].vertex_index)
        count = len(tri.loop_triangles) * 3
        vertex_words = struct.pack(f"<{len(positions)}f", *positions)
        triangle_words = struct.pack(f"<{count}i", *range(count))
        bpy.data.meshes.remove(tri)
        return vertex_words, count, triangle_words, count // 3
    finally:
        evaluated.to_mesh_clear()


def base_color_image(obj):
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


def push_object(obj, assets_dir, depsgraph=None, axes=AXES_YUP):
    """Writes <name>.vertices.field, <name>.triangles.field and, when the
    material has one, <name>.albedo.png into assets_dir. Returns the paths."""
    os.makedirs(assets_dir, exist_ok=True)
    name = bpy.path.clean_name(obj.name)
    vertices, count, triangles, faces = mesh_rows(obj, depsgraph, axes)
    paths = []
    path = os.path.join(assets_dir, f"{name}.vertices.field")
    write_field(path, "Rows", (count, 1, 1), "f,f,f,f,f,f,f,f", vertices)
    paths.append(path)
    path = os.path.join(assets_dir, f"{name}.triangles.field")
    write_field(path, "Rows", (faces, 1, 1), "i,i,i", triangles)
    paths.append(path)
    image = base_color_image(obj)
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


# ------------------------------------------------------------------- rigs

# Entelechy's rows are y-up, so a matrix written in Blender's z-up space is
# read in the turned one: `C M C^-1`, with C the same turn `to_entelechy` is.
def axis_change(axes):
    from mathutils import Matrix

    if axes != AXES_YUP:
        return Matrix.Identity(4)
    return Matrix(((1, 0, 0, 0), (0, 0, 1, 0), (0, -1, 0, 0), (0, 0, 0, 1)))


def turned(matrix, axes):
    """A Blender matrix as Entelechy reads it."""
    change = axis_change(axes)
    return change @ matrix @ change.inverted()


def armature_of(obj):
    """The armature that deforms `obj`, if one does."""
    for modifier in obj.modifiers:
        if modifier.type == "ARMATURE" and modifier.object is not None:
            return modifier.object
    return None


def bone_order(armature):
    """The bones as rows, parents before children."""
    order = []

    def walk(bone):
        order.append(bone)
        for child in bone.children:
            walk(child)

    for bone in armature.data.bones:
        if bone.parent is None:
            walk(bone)
    return order


def skeleton_rows(armature, axes=AXES_YUP):
    """Each joint's parent and inverse bind matrix, as the skeleton file
    holds them. Returns (words, joints, [bone names])."""
    bones = bone_order(armature)
    place = {bone.name: index for index, bone in enumerate(bones)}
    words = b""
    for bone in bones:
        parent = place.get(bone.parent.name, -1) if bone.parent else -1
        bind = turned(bone.matrix_local, axes).inverted()
        words += struct.pack("<i", parent)
        # Column-major, as the ABI lays a Mat4 out.
        words += struct.pack("<16f", *[bind[row][column] for column in range(4) for row in range(4)])
    return words, len(bones), [bone.name for bone in bones]


def weight_rows(obj, names, corners):
    """What every corner of the pushed mesh hangs off: the four heaviest
    groups that name a bone, normalized. Returns the words."""
    place = {name: index for index, name in enumerate(names)}
    groups = {group.index: group.name for group in obj.vertex_groups}
    words = b""
    for vertex_index in corners:
        weighted = []
        for element in obj.data.vertices[vertex_index].groups:
            joint = place.get(groups.get(element.group, ""))
            if joint is not None and element.weight > 0.0:
                weighted.append((element.weight, joint))
        weighted.sort(reverse=True)
        weighted = weighted[:4]
        total = sum(weight for weight, _ in weighted)
        joints = [joint for _, joint in weighted] + [0] * (4 - len(weighted))
        amounts = [weight / total for weight, _ in weighted] if total > 0.0 else []
        amounts = amounts + [0.0] * (4 - len(amounts))
        words += struct.pack("<4i", *joints) + struct.pack("<4f", *amounts)
    return words


def pose_rows(armature, names, axes=AXES_YUP):
    """Every joint's local transform at the current frame, as a clip's
    frame holds them: a translation, a rotation (x y z w) and a scale."""
    words = b""
    for name in names:
        bone = armature.pose.bones[name]
        local = bone.matrix if bone.parent is None else bone.parent.matrix.inverted() @ bone.matrix
        at, turn, grow = turned(local, axes).decompose()
        words += struct.pack("<10f", at[0], at[1], at[2], turn[1], turn[2], turn[3], turn[0], grow[0], grow[1], grow[2])
    return words


def action_curves(action):
    """An action's curves, whether it keeps them itself (4.3 and before)
    or in the channel bags of its slots (4.4 and after)."""
    curves = list(getattr(action, "fcurves", []) or [])
    if curves:
        return curves
    for layer in getattr(action, "layers", []):
        for strip in getattr(layer, "strips", []):
            for bag in getattr(strip, "channelbags", []):
                curves.extend(bag.fcurves)
    return curves


def actions_of(armature):
    """The actions that pose this armature: every one with a curve on a
    pose bone, so a rig with several clips writes several."""
    return [
        action
        for action in bpy.data.actions
        if any(curve.data_path.startswith("pose.bones") for curve in action_curves(action))
    ]


def clip_rows(armature, names, action, axes=AXES_YUP):
    """One action as a clip: every frame's joints together, sampled at
    the scene's own rate. Returns (words, frames)."""
    scene = bpy.context.scene
    animation = armature.animation_data or armature.animation_data_create()
    was, frame_was = animation.action, scene.frame_current
    slot_was = getattr(animation, "action_slot", None)
    animation.action = action
    if hasattr(animation, "action_slot") and action.slots:
        animation.action_slot = action.slots[0]
    first, last = (int(round(v)) for v in action.frame_range)
    words = b""
    try:
        for frame in range(first, last + 1):
            scene.frame_set(frame)
            words += pose_rows(armature, names, axes)
    finally:
        animation.action = was
        if slot_was is not None and hasattr(animation, "action_slot"):
            animation.action_slot = slot_was
        scene.frame_set(frame_was)
    return words, last - first + 1


def push_rig(obj, assets_dir, axes=AXES_YUP, clips=True):
    """Writes the rows a posed mesh needs: the mesh at rest in its
    armature's space, what every vertex hangs off, the joints, and one
    clip per action (or just the frame it stands on). Returns the paths."""
    armature = armature_of(obj)
    if armature is None:
        raise ValueError(f"{obj.name} has no armature modifier")
    os.makedirs(assets_dir, exist_ok=True)
    name = bpy.path.clean_name(obj.name)
    # The joints pose the mesh in the armature's own space, so that is
    # the space the rest mesh is written in.
    into_armature = armature.matrix_world.inverted() @ obj.matrix_world
    corners = []
    vertices, count, triangles, faces = mesh_rows(obj, None, axes, into_armature, corners)
    paths = []
    path = os.path.join(assets_dir, f"{name}.vertices.field")
    write_field(path, "Rows", (count, 1, 1), "f,f,f,f,f,f,f,f", vertices)
    paths.append(path)
    path = os.path.join(assets_dir, f"{name}.triangles.field")
    write_field(path, "Rows", (faces, 1, 1), "i,i,i", triangles)
    paths.append(path)
    skeleton, joints, names = skeleton_rows(armature, axes)
    path = os.path.join(assets_dir, f"{name}.skeleton.field")
    write_field(path, "Rows", (joints, 1, 1), "i," + ",".join(["f"] * 16), skeleton)
    paths.append(path)
    path = os.path.join(assets_dir, f"{name}.weights.field")
    write_field(path, "Rows", (count, 1, 1), "i,i,i,i,f,f,f,f", weight_rows(obj, names, corners))
    paths.append(path)
    pose = "f," * 9 + "f"
    if clips:
        for action in actions_of(armature):
            words, frames = clip_rows(armature, names, action, axes)
            clip_name = bpy.path.clean_name(action.name).lower()
            path = os.path.join(assets_dir, f"{name}.{clip_name}.clip.field")
            write_field(path, "Rows", (frames * joints, 1, 1), pose, words)
            paths.append(path)
    else:
        path = os.path.join(assets_dir, f"{name}.pose.clip.field")
        write_field(path, "Rows", (joints, 1, 1), pose, pose_rows(armature, names, axes))
        paths.append(path)
    return paths


# ---------------------------------------------------------------- protocol

class EntelechyClient:
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
                raise ConnectionError("Entelechy closed the connection")
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
    # Installed as an add-on, the host and port are preferences; run as a
    # script, they are the defaults.
    addon = context.preferences.addons.get(__name__)
    if addon is not None:
        return EntelechyClient(addon.preferences.host, addon.preferences.port)
    return EntelechyClient("127.0.0.1", 7788)


def rows_to_mesh(name, vertex_words, vertex_stride, triangle_words, weld=True, into=None, axes=AXES_YUP):
    """Builds a mesh from vertex rows of `vertex_stride` floats (8: position,
    normal, uv; 9: clip xyzw, normal, uv) and triangle rows of three ints.
    The rows hold a vertex per triangle corner; `weld` merges corners that
    share a position, keeping each corner's uv. With `into` an existing
    object, its mesh is replaced and the object keeps its place."""
    vertices = struct.unpack(f"<{len(vertex_words) // 4}f", vertex_words)
    indices = struct.unpack(f"<{len(triangle_words) // 4}i", triangle_words)
    count = len(vertices) // vertex_stride
    positions = [to_blender(vertices[i * vertex_stride:i * vertex_stride + 3], axes) for i in range(count)]
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
    if weld and count:
        bm = bmesh.new()
        bm.from_mesh(mesh)
        bmesh.ops.remove_doubles(bm, verts=bm.verts[:], dist=1e-6)
        bm.to_mesh(mesh)
        bm.free()
    for poly in mesh.polygons:
        poly.use_smooth = True
    mesh.update()
    if into is not None:
        previous = into.data
        into.data = mesh
        if previous.users == 0:
            bpy.data.meshes.remove(previous)
        return into
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    return obj


def read_rows(client, vertex_node, triangles, evaluate=True):
    """The rows behind a mesh: a node's vertex rows and a node's or a
    `.field` file's triangle rows, as bytes with the vertex stride. The
    nodes are pinned first, so fusion keeps them materialized, and the
    session is evaluated so they are current."""
    import base64
    client.call("pin", node=vertex_node)
    from_file = triangles.endswith(".field")
    if not from_file:
        client.call("pin", node=triangles)
    if evaluate:
        client.call("evaluate", time=time.monotonic() - _started)
    vertices = client.call("read", node=vertex_node, port="result", format="words")
    stride = len(vertices["schema"].split(","))
    if stride not in (8, 9):
        raise RuntimeError(f"{vertex_node} has {stride} words per row; a mesh has 8 or 9")
    if from_file:
        _, _, _, triangle_words = read_field(bpy.path.abspath(triangles))
    else:
        triangle_words = base64.b64decode(client.call("read", node=triangles, port="result", format="words")["data"])
    return base64.b64decode(vertices["data"]), stride, triangle_words


_started = time.monotonic()


def pull_mesh(client, vertex_node, triangles, name="EntelechyMesh", weld=True, into=None, axes=AXES_YUP):
    """Vertices come from a node's rows; the triangles from a node's rows,
    or from a `.field` file when the scene reads them as an input."""
    vertex_words, stride, triangle_words = read_rows(client, vertex_node, triangles)
    return rows_to_mesh(name, vertex_words, stride, triangle_words, weld=weld, into=into, axes=axes)


# ------------------------------------------------------------------ watch

_watch = {"client": None, "digest": None, "object": None}


def watch_once(context):
    """One tick of the watch: reads the rows, and rebuilds the watched
    object only when they changed. Returns "built", "unchanged", or an
    error message. The connection is kept between ticks."""
    import hashlib
    settings = context.scene.entelechy
    try:
        if _watch["client"] is None:
            _watch["client"] = connect(context)
        client = _watch["client"]
        vertex_words, stride, triangle_words = read_rows(client, settings.vertex_node, settings.triangle_node)
    except Exception as error:
        _watch["client"] = None
        return f"watch: {error}"
    digest = hashlib.blake2b(vertex_words + triangle_words, digest_size=16).digest()
    obj = _watch["object"]
    if obj is not None and obj.name not in bpy.data.objects:
        obj = None
    if digest == _watch["digest"] and obj is not None:
        return "unchanged"
    _watch["digest"] = digest
    _watch["object"] = rows_to_mesh(settings.watch_name, vertex_words, stride, triangle_words, weld=settings.weld, into=obj, axes=settings.axes)
    return "built"


def _watch_tick():
    settings = getattr(bpy.context.scene, "entelechy", None)
    if settings is None or not settings.watching:
        if _watch["client"] is not None:
            _watch["client"].close()
            _watch["client"] = None
        return None
    outcome = watch_once(bpy.context)
    if outcome.startswith("watch:"):
        print("entelechy bridge:", outcome)
    return settings.watch_interval


def pull_image(client, name="EntelechyFrame"):
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

class EntelechyPreferences(bpy.types.AddonPreferences):
    bl_idname = __name__
    host: bpy.props.StringProperty(name="Host", default="127.0.0.1")
    port: bpy.props.IntProperty(name="Port", default=7788, min=1, max=65535)

    def draw(self, context):
        row = self.layout.row()
        row.prop(self, "host")
        row.prop(self, "port")


class EntelechySettings(bpy.types.PropertyGroup):
    assets_dir: bpy.props.StringProperty(name="Assets", subtype="DIR_PATH", description="The Entelechy scene's assets directory; inputs there are read and watched")
    axes: bpy.props.EnumProperty(
        name="Axes",
        items=[
            (AXES_YUP, "Y up", "Blender's Z becomes Entelechy's Y, as glTF does; a model stands the same on both sides"),
            (AXES_NONE, "As is", "Coordinates cross unchanged"),
        ],
        default=AXES_YUP,
        description="How axes turn between Blender (z up) and Entelechy (y up), on the way out and back",
    )
    live: bpy.props.BoolProperty(name="Live", default=False, description="Push the active object again whenever its geometry changes")
    vertex_node: bpy.props.StringProperty(name="Vertices node", default="torus")
    triangle_node: bpy.props.StringProperty(name="Triangles", default="faces", description="A node, or a .field file when the scene reads triangles as an input")
    weld: bpy.props.BoolProperty(name="Weld", default=True, description="Merge the triangle corners that share a position; uvs stay per corner")
    watching: bpy.props.BoolProperty(name="Watching", default=False)
    watch_name: bpy.props.StringProperty(name="Object", default="EntelechyWatch", description="The object the watch keeps current")
    watch_interval: bpy.props.FloatProperty(name="Every", default=0.5, min=0.1, max=10.0, subtype="TIME_ABSOLUTE", description="Seconds between reads")
    param_name: bpy.props.StringProperty(name="Param", default="exposure")
    param_value: bpy.props.StringProperty(name="Value", default="1.0")


class ENTELECHY_OT_push(bpy.types.Operator):
    bl_idname = "entelechy.push"
    bl_label = "Push mesh to Entelechy"
    bl_description = "Write the active object as vertex and triangle rows (and its base color as PNG) into the assets directory"

    def execute(self, context):
        obj = context.active_object
        if obj is None or obj.type != "MESH":
            self.report({"ERROR"}, "select a mesh object")
            return {"CANCELED"}
        settings = context.scene.entelechy
        if not settings.assets_dir:
            self.report({"ERROR"}, "set the assets directory")
            return {"CANCELED"}
        paths = push_object(obj, bpy.path.abspath(settings.assets_dir), context.evaluated_depsgraph_get(), settings.axes)
        self.report({"INFO"}, "wrote " + ", ".join(os.path.basename(p) for p in paths))
        return {"FINISHED"}


class ENTELECHY_OT_push_rig(bpy.types.Operator):
    bl_idname = "entelechy.push_rig"
    bl_label = "Push rig to Entelechy"
    bl_description = "Write the active object and its armature as the rows a scene poses: the mesh at rest, what every vertex hangs off, the joints, and one clip per action"

    def execute(self, context):
        obj = context.active_object
        if obj is None or obj.type != "MESH":
            self.report({"ERROR"}, "select a mesh object")
            return {"CANCELED"}
        settings = context.scene.entelechy
        if not settings.assets_dir:
            self.report({"ERROR"}, "set the assets directory")
            return {"CANCELED"}
        try:
            paths = push_rig(obj, bpy.path.abspath(settings.assets_dir), settings.axes)
        except ValueError as error:
            self.report({"ERROR"}, str(error))
            return {"CANCELED"}
        self.report({"INFO"}, "wrote " + ", ".join(os.path.basename(p) for p in paths))
        return {"FINISHED"}


class ENTELECHY_OT_pull_mesh(bpy.types.Operator):
    bl_idname = "entelechy.pull_mesh"
    bl_label = "Pull mesh from Entelechy"
    bl_description = "Read two nodes' rows from the running session and build a mesh"

    def execute(self, context):
        settings = context.scene.entelechy
        try:
            client = connect(context)
            obj = pull_mesh(client, settings.vertex_node, settings.triangle_node, weld=settings.weld, axes=settings.axes)
            client.close()
        except Exception as error:
            self.report({"ERROR"}, str(error))
            return {"CANCELED"}
        self.report({"INFO"}, f"pulled {obj.name}: {len(obj.data.vertices)} vertices")
        return {"FINISHED"}


class ENTELECHY_OT_watch(bpy.types.Operator):
    bl_idname = "entelechy.watch"
    bl_label = "Watch"
    bl_description = "Keep an object current with two nodes' rows: read them every interval and rebuild when they change"

    def execute(self, context):
        settings = context.scene.entelechy
        settings.watching = not settings.watching
        if settings.watching:
            _watch["digest"] = None
            if not bpy.app.timers.is_registered(_watch_tick):
                bpy.app.timers.register(_watch_tick, first_interval=0.0)
            self.report({"INFO"}, f"watching {settings.vertex_node} into {settings.watch_name}")
        else:
            self.report({"INFO"}, "watch off")
        return {"FINISHED"}


class ENTELECHY_OT_pull_image(bpy.types.Operator):
    bl_idname = "entelechy.pull_image"
    bl_label = "Pull picture from Entelechy"
    bl_description = "Read the session's picture into an Image datablock named EntelechyFrame"

    def execute(self, context):
        try:
            client = connect(context)
            image = pull_image(client)
            client.close()
        except Exception as error:
            self.report({"ERROR"}, str(error))
            return {"CANCELED"}
        self.report({"INFO"}, f"pulled {image.name} {image.size[0]}x{image.size[1]}")
        return {"FINISHED"}


class ENTELECHY_OT_param(bpy.types.Operator):
    bl_idname = "entelechy.param"
    bl_label = "Set param"
    bl_description = "Set a graph parameter in the running session"

    def execute(self, context):
        settings = context.scene.entelechy
        try:
            client = connect(context)
            client.call("param", name=settings.param_name, value=parse_value(settings.param_value))
            client.close()
        except Exception as error:
            self.report({"ERROR"}, str(error))
            return {"CANCELED"}
        return {"FINISHED"}


class ENTELECHY_OT_publish(bpy.types.Operator):
    bl_idname = "entelechy.publish"
    bl_label = "Publish text as Kernel"
    bl_description = "Publish the text editor's active text as a Kernel in the running session"

    def execute(self, context):
        text = next((area.spaces.active.text for area in context.screen.areas if area.type == "TEXT_EDITOR" and area.spaces.active.text), None)
        if text is None:
            self.report({"ERROR"}, "open a Kernel in the text editor")
            return {"CANCELED"}
        try:
            client = connect(context)
            published = client.call("publish", source=text.as_string(), expect="any")
            client.close()
        except Exception as error:
            self.report({"ERROR"}, str(error))
            return {"CANCELED"}
        self.report({"INFO"}, f"published {published['id']} generation {published['generation']}")
        return {"FINISHED"}


class ENTELECHY_PT_panel(bpy.types.Panel):
    bl_label = "Entelechy"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Entelechy"

    def draw(self, context):
        settings = context.scene.entelechy
        layout = self.layout
        box = layout.box()
        box.label(text="Files: the scene's inputs")
        box.prop(settings, "assets_dir")
        box.prop(settings, "axes")
        box.operator("entelechy.push")
        box.operator("entelechy.push_rig")
        box.prop(settings, "live")
        box = layout.box()
        box.label(text="Session: a running window")
        row = box.row()
        row.prop(settings, "vertex_node")
        row.prop(settings, "triangle_node")
        box.prop(settings, "weld")
        box.operator("entelechy.pull_mesh")
        row = box.row()
        row.prop(settings, "watch_name")
        row.prop(settings, "watch_interval")
        box.operator("entelechy.watch", text="Stop watching" if settings.watching else "Watch", depress=settings.watching)
        box.operator("entelechy.pull_image")
        row = box.row()
        row.prop(settings, "param_name")
        row.prop(settings, "param_value")
        box.operator("entelechy.param")
        box.operator("entelechy.publish")


_last_push = 0.0


def deformed_by(armature):
    """The mesh objects this armature poses."""
    return [
        obj
        for obj in bpy.data.objects
        if obj.type == "MESH" and armature_of(obj) is armature
    ]


def live_targets(active):
    """What a push should write when `active` is what the person is
    working on, and which ids changing means it is worth writing.

    Posing happens with the armature active, not the mesh — Pose Mode
    selects the armature — so an armature stands for the meshes it
    deforms.
    """
    if active is None:
        return [], set()
    if active.type == "ARMATURE":
        targets = deformed_by(active)
    elif active.type == "MESH":
        targets = [active]
    else:
        return [], set()
    watched = set()
    for obj in targets:
        watched.add(obj)
        if obj.data is not None:
            watched.add(obj.data)
        armature = armature_of(obj)
        if armature is not None:
            watched.add(armature)
            if armature.data is not None:
                watched.add(armature.data)
    return targets, watched


def push_live(obj, assets_dir, axes, depsgraph=None):
    """What Live writes for one object: a rigged one pushes the pose it
    stands in — the mesh and the joints do not change while it is posed,
    so a frame costs one clip of one frame — and anything else pushes
    the mesh it has become."""
    if armature_of(obj) is not None:
        return push_rig(obj, assets_dir, axes, clips=False)
    return push_object(obj, assets_dir, depsgraph, axes)


def _on_depsgraph(scene, depsgraph):
    global _last_push
    settings = getattr(scene, "entelechy", None)
    if settings is None or not settings.live or not settings.assets_dir:
        return
    targets, watched = live_targets(bpy.context.view_layer.objects.active)
    if not targets:
        return
    changed = any(update.id.original in watched for update in depsgraph.updates)
    if not changed or time.monotonic() - _last_push < 0.2:
        return
    _last_push = time.monotonic()
    assets = bpy.path.abspath(settings.assets_dir)
    for obj in targets:
        try:
            push_live(obj, assets, settings.axes, depsgraph)
        except Exception as error:
            print("entelechy bridge: push failed:", error)


classes = (EntelechyPreferences, EntelechySettings, ENTELECHY_OT_push, ENTELECHY_OT_push_rig, ENTELECHY_OT_pull_mesh, ENTELECHY_OT_watch, ENTELECHY_OT_pull_image, ENTELECHY_OT_param, ENTELECHY_OT_publish, ENTELECHY_PT_panel)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.entelechy = bpy.props.PointerProperty(type=EntelechySettings)
    if _on_depsgraph not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(_on_depsgraph)


def unregister():
    if bpy.app.timers.is_registered(_watch_tick):
        bpy.app.timers.unregister(_watch_tick)
    if _on_depsgraph in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(_on_depsgraph)
    del bpy.types.Scene.entelechy
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
