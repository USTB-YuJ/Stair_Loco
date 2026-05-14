import numpy as np
import torch
import transforms3d as t3d
from typing import List
from pytorch3d import transforms
import matplotlib.pyplot as plt
import warp as wp
import cv2

def depth_image_preprocessing(depth_image, near_plane=100, far_plane=1200.0, depth_scale=1000):

    near_mask = abs(depth_image) < near_plane / depth_scale
    far_mask = abs(depth_image) > far_plane / depth_scale

    depth_image[near_mask] = 0
    depth_image[far_mask] = far_plane / depth_scale
    if torch.isnan(depth_image).any() or torch.isinf(depth_image).any():
        raise Exception("nan or inf of depth image detected!")
    return depth_image


def _quat_apply_xyzw(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Apply XYZW quaternion rotation to 3D vectors."""
    q_xyz = quat[..., :3]
    q_w = quat[..., 3:4]
    t = 2.0 * torch.cross(q_xyz, vec, dim=-1)
    return vec + q_w * t + torch.cross(q_xyz, t, dim=-1)


@wp.kernel
def draw_pixels(mesh: wp.uint64, cam_pos: wp.array(dtype=wp.float32), cam_rot: wp.array(dtype=wp.float32), width: int, height: int, pixels: wp.array(dtype=wp.vec3)):
    tid = wp.tid()
    b = tid // (height*width)
    p = tid % (height*width)
    y = p % width
    z = p // width

    sy = -2.0 * float(y) / float(height) + float(width / height) # from the left to the right
    sz = -2.0 * float(z) / float(height) + 1.0 # from the up to the down

    ro = wp.vec3(cam_pos[b*3], cam_pos[b*3+1], cam_pos[b*3+2])
    rot = wp.mat33(cam_rot[b*9], cam_rot[b*9+1], cam_rot[b*9+2],
                   cam_rot[b*9+3], cam_rot[b*9+4], cam_rot[b*9+5],
                   cam_rot[b*9+6], cam_rot[b*9+7], cam_rot[b*9+8])
    rd = wp.normalize(wp.vec3(float(width / height), sy, sz))
    rd = rot @ rd

    color = wp.vec3(0.0, 0.0, 0.0)
    query = wp.mesh_query_ray(mesh, ro, rd, 10.0)

    if query.result:
        color = query.normal * 0.5 + wp.vec3(0.5, 0.5, 0.5)

    pixels[tid] = color


@wp.kernel
def draw_depth(mesh: wp.uint64, cam_pos: wp.array(dtype=wp.float32), cam_rot: wp.array(dtype=wp.float32), width: int, height: int, depth: wp.array(dtype=wp.float32), fovy_dist_offset: wp.array(dtype=wp.float32)):
    tid = wp.tid()
    b = tid // (height*width)
    p = tid % (height*width)
    y = p % width
    z = p // width

    sy = -2.0 * float(y) / float(height) + float(width / height) # from the left to the right
    sz = -2.0 * float(z) / float(height) + 1.0 # from the up to the down

    ro = wp.vec3(cam_pos[b*3], cam_pos[b*3+1], cam_pos[b*3+2])
    rot = wp.mat33(cam_rot[b*9], cam_rot[b*9+1], cam_rot[b*9+2],
                   cam_rot[b*9+3], cam_rot[b*9+4], cam_rot[b*9+5],
                   cam_rot[b*9+6], cam_rot[b*9+7], cam_rot[b*9+8])

    rd = wp.normalize(wp.vec3(float(width / height)+fovy_dist_offset[b], sy, sz))
    d = rd[0]
    rd = rot @ rd

    distance = wp.float32(0.0)
    query = wp.mesh_query_ray(mesh, ro, rd, 5.0)

    if query.result:
        distance = query.t * d
    else:
        distance = 5.0

    depth[tid] = distance


@wp.kernel
def draw_depth_single(mesh: wp.uint64, cam_pos: wp.array(dtype=wp.float32), cam_rot: wp.array(dtype=wp.float32), width: int, height: int, depth: wp.array(dtype=wp.float32), fovy_dist_offset: wp.array(dtype=wp.float32), max_t: float, miss_t: float):
    tid = wp.tid()
    b = tid // (height*width)
    p = tid % (height*width)
    y = p % width
    z = p // width

    sy = -2.0 * float(y) / float(height) + float(width / height)
    sz = -2.0 * float(z) / float(height) + 1.0

    ro = wp.vec3(cam_pos[b*3], cam_pos[b*3+1], cam_pos[b*3+2])
    rot = wp.mat33(cam_rot[b*9], cam_rot[b*9+1], cam_rot[b*9+2],
                   cam_rot[b*9+3], cam_rot[b*9+4], cam_rot[b*9+5],
                   cam_rot[b*9+6], cam_rot[b*9+7], cam_rot[b*9+8])

    rd = wp.normalize(wp.vec3(float(width / height) + fovy_dist_offset[b], sy, sz))
    d = rd[0]
    rd = rot @ rd

    distance = miss_t
    query = wp.mesh_query_ray(mesh, ro, rd, max_t)
    if query.result:
        distance = query.t * d
    depth[tid] = distance


@wp.kernel
def draw_depth_dual(terrain_mesh: wp.uint64, body_meshes: wp.array(dtype=wp.uint64), cam_pos: wp.array(dtype=wp.float32), cam_rot: wp.array(dtype=wp.float32), width: int, height: int, depth: wp.array(dtype=wp.float32), fovy_dist_offset: wp.array(dtype=wp.float32), max_t: float, miss_t: float):
    tid = wp.tid()
    b = tid // (height*width)
    p = tid % (height*width)
    y = p % width
    z = p // width

    sy = -2.0 * float(y) / float(height) + float(width / height)
    sz = -2.0 * float(z) / float(height) + 1.0

    ro = wp.vec3(cam_pos[b*3], cam_pos[b*3+1], cam_pos[b*3+2])
    rot = wp.mat33(cam_rot[b*9], cam_rot[b*9+1], cam_rot[b*9+2],
                   cam_rot[b*9+3], cam_rot[b*9+4], cam_rot[b*9+5],
                   cam_rot[b*9+6], cam_rot[b*9+7], cam_rot[b*9+8])

    rd = wp.normalize(wp.vec3(float(width / height) + fovy_dist_offset[b], sy, sz))
    d = rd[0]
    rd = rot @ rd

    distance = miss_t
    query_t = wp.mesh_query_ray(terrain_mesh, ro, rd, max_t)
    if query_t.result:
        distance = wp.min(distance, query_t.t * d)

    body_mesh = body_meshes[b]
    query_b = wp.mesh_query_ray(body_mesh, ro, rd, max_t)
    if query_b.result:
        distance = wp.min(distance, query_b.t * d)

    depth[tid] = distance


def euler_pos_2_mat(pos: torch.Tensor, euler: torch.Tensor):
    if torch.is_tensor(pos):
        pos = pos.cpu().numpy()
    if torch.is_tensor(euler):
        euler = euler.cpu().numpy()
    
    mats = torch.zeros(pos.shape[0], 4, 4)

    for i in range(pos.shape[0]):
        rot = torch.tensor(t3d.euler.euler2mat(euler[i, 0], euler[i, 1], euler[i, 2], "sxyz"))
        mats[i, :3,:3] = rot
        mats[i, :3,3] = torch.tensor(pos[i])
        mats[i, 3,3] = 1
    return mats


def quat_pos_2_mat_torch(pos: torch.Tensor, quat: torch.Tensor):
    b = pos.shape[0]
    rot = transforms.quaternion_to_matrix(torch.cat((quat[:,3:], quat[:, :3]), dim=1).to(quat.device)) # quat have to be wxyz
    mat = torch.zeros(b, 4, 4).to(pos.device)
    mat[:,:3,:3] = rot
    mat[:,:3,3] = pos
    mat[:,3,3] = 1
    return mat


class DepthRendererWarp:
    def __init__(self, image_params:List, cam2base_xyz: torch.Tensor, cam2base_euler: torch.Tensor, fovy: torch.Tensor, device="cuda:0", num_envs: int = 1, far_t=None, miss_t=None):
        self.image_height = image_params[0]
        self.image_width = image_params[1]
        self.device = device
        self.num_envs = int(num_envs)

        self.cam2base_xyz = cam2base_xyz
        self.cam2base_euler = cam2base_euler

        self.near_plane = 0.1
        self.far_plane = 1.2
        
        self.fovy_dist_offset = 1.0 / torch.tan(torch.deg2rad(fovy)/2) - 1.0
        self.fps = 30

        self.mesh = None
        self.body_link_verts = []
        self.body_link_faces = []

        self.far_t = float(far_t) if far_t is not None else 5.0
        self.miss_t = float(miss_t) if miss_t is not None else self.far_t

        self.robot_meshes = []
        self.robot_mesh_ids = None
        self._robot_mesh_points = []
        self._robot_tris_wp = None
        self._robot_verts = None
        self._template_verts_local = None
        self._vert_to_body = None
        self._refit_stride = 1
        self._refit_counter = 0

        self.robot2cam = euler_pos_2_mat(self.cam2base_xyz, self.cam2base_euler).to(self.device)
        self.warp2gym = torch.tensor([[0, 1, 0, 0],
                                      [0, 0, 1, 0],
                                      [1, 0, 0, 0],
                                      [0, 0, 0, 1]], dtype=torch.float32).to(self.device)

        self.cam_intrinsics_matrix = torch.tensor([[384.77294921875, 0, 324.17236328125],
                                                   [0, 384.77294921875, 236.48226928710938],
                                                   [0, 0, 1]])
        self.depth_scale = 1000

    def load_body_meshes(self, mesh_paths):
        import trimesh
        self.body_link_verts = []
        self.body_link_faces = []
        for p in mesh_paths:
            m = trimesh.load(p)
            self.body_link_verts.append(np.array(m.vertices, dtype=np.float32))
            self.body_link_faces.append(np.array(m.faces, dtype=np.uint32))

    def _build_body_mesh(self, link_pos_w, link_quat_w):
        from legged_gym.utils.math import quat_apply
        nlinks = len(self.body_link_verts)
        verts, faces = [], []
        off = 0
        for i in range(link_pos_w.shape[0]):
            k = i % nlinks
            vl = torch.tensor(self.body_link_verts[k], device=self.device, dtype=torch.float32)
            fl = self.body_link_faces[k]
            vw = quat_apply(link_quat_w[i], vl) + link_pos_w[i]
            vw = vw @ self.warp2gym[:3,:3].T
            verts.append(vw.cpu().numpy())
            faces.append(fl + off)
            off += len(vl)
        verts = np.vstack(verts).astype(np.float32)
        faces = np.concatenate(faces).astype(np.int32).flatten()
        with wp.ScopedDevice(self.device):
            self.body_mesh = wp.Mesh(
                points=wp.array(verts, dtype=wp.vec3, device=self.device),
                indices=wp.array(faces, dtype=wp.int32, device=self.device))

    def init_robot_meshes(self, template_verts_local, template_tris, vert_to_link, body_indices, refit_stride: int = 1):
        """Prebuild per-env robot meshes and cache templates for fast updates."""
        verts_local = torch.tensor(template_verts_local, device=self.device, dtype=torch.float32)
        tris = np.asarray(template_tris, dtype=np.int32).flatten()
        vert_to_link_t = torch.tensor(vert_to_link, device=self.device, dtype=torch.long)
        body_indices_t = torch.tensor(body_indices, device=self.device, dtype=torch.long)

        self._template_verts_local = verts_local
        self._vert_to_body = body_indices_t[vert_to_link_t]

        verts_per_env = verts_local.shape[0]
        self._robot_verts = torch.zeros(self.num_envs * verts_per_env, 3, device=self.device, dtype=torch.float32)

        with wp.ScopedDevice(self.device):
            self._robot_tris_wp = wp.array(tris, dtype=wp.int32, device=self.device)

        self.robot_meshes = []
        self._robot_mesh_points = []
        for i in range(self.num_envs):
            start = i * verts_per_env
            end = (i + 1) * verts_per_env
            points_wp = wp.from_torch(self._robot_verts[start:end], dtype=wp.vec3)
            mesh = wp.Mesh(points=points_wp, indices=self._robot_tris_wp)
            self.robot_meshes.append(mesh)
            self._robot_mesh_points.append(points_wp)

        self.robot_mesh_ids = wp.array([m.id for m in self.robot_meshes], dtype=wp.uint64, device=self.device)
        self._refit_stride = max(1, int(refit_stride))
        self._refit_counter = 0

    def update_robot_meshes(self, rigid_body_states: torch.Tensor):
        """Update per-env robot vertex buffers in place, optionally refitting BVHs."""
        if self._robot_verts is None or self._vert_to_body is None:
            return

        with torch.no_grad():
            rb = rigid_body_states
            num_envs = rb.shape[0]
            verts_per_env = self._template_verts_local.shape[0]
            if num_envs != self.num_envs:
                num_envs = min(num_envs, self.num_envs)
                rb = rb[:num_envs]

            pos = rb[:, self._vert_to_body, 0:3]
            quat = rb[:, self._vert_to_body, 3:7]
            verts_local = self._template_verts_local.unsqueeze(0)
            verts_rot = _quat_apply_xyzw(quat, verts_local)
            verts_world = verts_rot + pos
            verts_warp = verts_world @ self.warp2gym[:3, :3].T

            self._robot_verts.view(num_envs, verts_per_env, 3).copy_(verts_warp)

        self._refit_counter = (self._refit_counter + 1) % self._refit_stride
        if self._refit_counter == 0:
            for mesh in self.robot_meshes:
                mesh.refit()

    def render_mesh(self, vertices: np.ndarray, indices: np.ndarray):
        with wp.ScopedDevice(self.device):
            vertices = np.matmul(self.warp2gym[:3,:3].cpu().numpy(), vertices.transpose(1,0)).transpose(1,0)
            indices = indices.flatten()
            self.mesh = wp.Mesh(points=wp.array(vertices, dtype=wp.vec3, device=self.device), velocities=None, indices=wp.array(indices, dtype=int, device=self.device))

    def render_pixels(self, base_pos: torch.Tensor, base_quat: torch.Tensor) -> torch.Tensor: 
        """
        base_pos: (B, 3)
        base_quat: (B, 4)
        """
        b = base_pos.shape[0]
        pixels = wp.zeros(b * self.image_width * self.image_height, dtype=wp.vec3, device=self.device)

        gym2robot = quat_pos_2_mat_torch(base_pos, base_quat).to(self.device)
        warp2cam = self.warp2gym.unsqueeze(0).repeat([b,1,1]) @ gym2robot @ self.robot2cam
        cam_pos = warp2cam[:,:3,3].reshape(b*3)
        cam_rot = warp2cam[:,:3,:3].reshape(b*9)

        with wp.ScopedDevice(self.device):
            wp.launch(
                kernel = draw_pixels,
                dim = b * self.image_width * self.image_height,
                inputs = [self.mesh.id, wp.array(cam_pos, dtype=wp.float32), wp.array(cam_rot, dtype=wp.float32), self.image_width, self.image_height, pixels],
                device = self.device
            )

        return wp.to_torch(pixels.reshape([b, self.image_height, self.image_width]))
    
    def render_depth(self, base_pos: torch.Tensor, base_quat: torch.Tensor,
                    link_pos_w=None, link_quat_w=None) -> torch.Tensor:
        """
        base_pos: (B, 3)
        base_quat: (B, 4)
        """
        b = base_pos.shape[0]
        depth = wp.zeros(b * self.image_width * self.image_height, dtype=wp.float32, device=self.device)

        gym2robot = quat_pos_2_mat_torch(base_pos, base_quat).to(self.device)
        warp2cam = self.warp2gym.unsqueeze(0).repeat([b,1,1]) @ gym2robot @ self.robot2cam
        cam_pos = warp2cam[:,:3,3].reshape(b*3)
        cam_rot = warp2cam[:,:3,:3].reshape(b*9)

        with wp.ScopedDevice(self.device):
            if self.robot_mesh_ids is not None:
                wp.launch(
                    kernel=draw_depth_dual,
                    dim=b * self.image_width * self.image_height,
                    inputs=[self.mesh.id, self.robot_mesh_ids, wp.array(cam_pos, dtype=wp.float32), wp.array(cam_rot, dtype=wp.float32), self.image_width, self.image_height, depth, self.fovy_dist_offset, self.far_t, self.miss_t],
                    device=self.device,
                )
            else:
                wp.launch(
                    kernel=draw_depth_single,
                    dim=b * self.image_width * self.image_height,
                    inputs=[self.mesh.id, wp.array(cam_pos, dtype=wp.float32), wp.array(cam_rot, dtype=wp.float32), self.image_width, self.image_height, depth, self.fovy_dist_offset, self.far_t, self.miss_t],
                    device=self.device,
                )

        depth = wp.to_torch(depth.reshape([b, self.image_height, self.image_width]))
        return depth

    def show_depth(self, depth: torch.Tensor):
        """
        depth: (B, H, W)
        """
        for i in range(depth.shape[0]):
            plt.imshow(depth[i].cpu().numpy(), cmap="gray")
            plt.title("depth image")
            plt.axis("off")
            plt.show()

    def show_color(self, color: torch.Tensor):
        """
        color: (B, H, W, 3)
        """
        for i in range(color.shape[0]):
            cv2.imshow("color image", color[i].cpu().numpy())
            cv2.waitKey(10000)
            cv2.destroyAllWindows()
    
    def play_show_depth(self, depth: torch.Tensor):
        """
        depth: (H, W)
        """
        # absolutly depth image
        depth_ = depth #(depth/self.depth_scale)
        normalized_depth = (depth_ / self.far_plane * 255).cpu().numpy().astype(np.uint8)

        # relatively depth image
        # depth_max = torch.max(depth_)
        # depth_image = depth_ / depth_max * 255
        # normalized_depth = depth_image.cpu().numpy().astype(np.uint8)
        
        cv2.imshow("Depth Image", normalized_depth)
        cv2.waitKey(10)

if __name__ == "__main__":
    warprender = DepthRendererWarp([60, 106],
                      torch.tensor([0.10043, 0.0222, -0.11]),
                      torch.tensor([0, 60.95 / 180 * np.pi, 0]))