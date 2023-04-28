'''
Code borrowed from
https://github.com/nileshkulkarni/acsm/blob/master/acsm/utils/visdom_render.py
'''

from __future__ import absolute_import, division, print_function
import os
import cv2
import uuid
import os.path as osp
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt
import torch

from acsm.utils.visuals import vis_utils, vis_utils_cmr


class RendererWrapper(object):
    '''
    This is the Renderer Class with the functionalities used from Visdom
    '''
    def __init__(self, renderer, verts, uv_sampler, offset_z, mean_shape_np, model_obj_path, keypoint_cmap, opts):
        self.opts           = opts
        self.uvimgH         = 256
        self.uvimgW         = 256
        self.offset_z       = offset_z
        self.verts_obj      = verts
        self.vis_rend       = renderer
        self.uv_sampler     = uv_sampler
        self.mean_shape_np  = mean_shape_np
        self.keypoint_cmap  = keypoint_cmap
        self.model_obj_path = model_obj_path
        self.Tensor         = torch.cuda.FloatTensor

    def visualize_depth_loss_perpixel(self, per_pixel_losses):
        '''
        Set of per pixel loss for all hypothesis
        '''
        image = per_pixel_losses.unsqueeze(-1).data.cpu().numpy()
        temp  = vis_utils.image_montage(image, nrow=3)
        plt.imshow(temp[:, :, 0], cmap='hot', interpolation='nearest')
        plt.colorbar()
        filename = osp.join('/tmp/', "{}.png".format(str(uuid.uuid4())))
        try:
            plt.savefig(filename)
            image = scipy.misc.imread(filename)
        except Exception as e:
            image = (temp * 0).astype(np.uint8)
        finally:
            os.remove(filename)
            plt.close()
        return image

    def update_verts(self, verts):
        self.verts_obj = verts

    def render_default_bird(self, camera, color=None, tex_size=6):
        image, texture_img = vis_utils_cmr.render_model_default(
            self.vis_rend,
            self.verts_obj,
            self.uvimgH,
            self.uvimgW,
            camera,
            self.uv_sampler,
            color=color,
            tex_size=tex_size
        )
        return image, texture_img

    def render_model_using_nmr(
        self,
        uv_map,
        img,
        mask,
        cam,
        upsample_texture=True,
        other_vps=False,
    ):
        visuals = {}
        uvimg_H = self.uvimgH
        uvimg_W = self.uvimgW
        if upsample_texture:
            img, mask, uv_map = vis_utils_cmr.upsample_img_mask_uv_map(img, mask, uv_map)

        uv_map = uv_map.data.cpu().numpy()
        img = img.unsqueeze(0)
        img = vis_utils.undo_resnet_preprocess(img).squeeze()
        img = img.data.cpu().numpy()
        mask = mask.data.cpu().numpy()
        texture_image = vis_utils_cmr.create_texture_image_from_uv_map(
            uvimg_H,
            uvimg_W,
            uv_map,
            img,
            mask,
        )
        texture_image = torch.from_numpy(texture_image).float().cuda()

        texture_ms, texture_img = self.wrap_texture( texture_image, cam, other_vps=other_vps )
        visuals['texture_ms'] = vis_utils.image_montage(list(texture_ms), nrow=2)
        visuals['texture_img'] = (texture_img * 255).astype(np.uint8)
        return visuals

    def render_gt_kps_heatmap(self, kp3d_uv, camera, kps_vis=None, suffix=''):
        uv_sampler = self.uv_sampler
        tex_size = self.opts.tex_size
        other_vps = False
        uv_H = 256
        uv_W = 256
        all_visuals = []
        default_tex = vis_utils_cmr.create_kp_heat_map_texture(uv_H, uv_W)
        kp_textures = []

        for kpx, kp_uv in enumerate(kp3d_uv):
            visuals = {}
            kp_uv = kp_uv * 255
            uv_cords = [int(kp_uv[0]), int(kp_uv[1])]
            kp_color = self.keypoint_cmap[kpx]
            texture = vis_utils_cmr.create_kp_heat_map_texture(
                uv_H, uv_W, uv_cords[0], uv_cords[1], color=kp_color
            )
            if kps_vis is None or kps_vis[kpx] > 0:
                kp_textures.append(texture)
        # kp_textures =
        kp_textures = np.stack(kp_textures, axis=0)
        default_mask = (0 == np.max(kp_textures[:, 3, None, :, :], axis=0))
        average = np.sum(kp_textures[:, 3, None, :, :], axis=0) + default_mask

        texture = np.sum(kp_textures, axis=0) / average
        texture = default_tex * default_mask + texture * (1 - default_mask)
        texture = texture[0:3, :, :]
        texture = torch.from_numpy(texture).float().cuda()
        # renderer, vert, camera, uv_sampler, texture_image, tex_size, other_vps
        texture_ms, texture_img = self.wrap_texture(
            texture=texture,
            camera=camera,
        )
        visuals = {}
        visuals['texture_kp_zgt' +
                suffix] = vis_utils.image_montage(list(texture_ms), nrow=2)

        visuals['texture_kp_img_zgt' + suffix] = (texture_img *
                                                  255).astype(np.uint8)
        return visuals

    def render_one_camera(self, cameras, verts=None, probs=None, sample_ind=None, gt_cam=None):
        uv_H        = self.uvimgH
        uv_W        = self.uvimgW

        default_tex = vis_utils_cmr.create_monocolor_texture(uv_H, uv_W)[0:3, :, :]
        default_tex = torch.from_numpy(default_tex).float().cuda()
        renderings  = []
        _, max_ind  = torch.max(probs, dim=0)
        max_ind     = max_ind.item()
        max_ind     = sample_ind if sample_ind is not None else max_ind

        from .. import metrics
        cam_errors = np.zeros(len(cameras))
        if gt_cam is not None:
            gt_quat = gt_cam[3:7].data.cpu()
            cam_errors = np.array([ metrics.quat_dist(pred, gt_quat) for pred in cameras[:, 3:].data.cpu() ])

        for cx, camera in enumerate(cameras):
            if verts is not None:
                self.update_verts(verts[cx])
            texture_ms, _ = self.wrap_texture(texture=default_tex, camera=camera, lights=True)
            if gt_cam is not None:
                texture_ms = cv2.putText(
                    texture_ms, "E:{}".format(np.round(cam_errors[cx], 1)), (125, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color=(255, 0, 0), thickness=2
                )

            renderings.append(texture_ms)
        visuals = {}
        visuals['all_hypotheses'] = vis_utils.image_montage(renderings, nrow=3)
        return visuals

    def render_all_hypotheses(self, cameras, verts=None, probs=None, sample_ind=None, gt_cam=None, losses_per_hypo=None):
        uv_H        = self.uvimgH
        uv_W        = self.uvimgW

        default_tex = vis_utils_cmr.create_monocolor_texture(uv_H, uv_W)[0:3, :, :]
        default_tex = torch.from_numpy(default_tex).float().cuda()
        renderings  = []
        _, max_ind  = torch.max(probs, dim=0)
        max_ind     = max_ind.item()
        max_ind     = sample_ind if sample_ind is not None else max_ind

        from .. import metrics
        cam_errors = np.zeros(len(cameras))
        if gt_cam is not None:
            gt_quat = gt_cam[3:7].data.cpu()
            cam_errors = np.array([ metrics.quat_dist(pred, gt_quat) for pred in cameras[:, 3:].data.cpu() ])

        for cx, camera in enumerate(cameras):
            if verts is not None:
                self.update_verts(verts[cx])

            texture_ms, _ = self.wrap_texture(texture=default_tex, camera=camera, lights=True)
            if probs is not None:
                color = (0, 128, 128) if (cx == max_ind) else (255, 0, 0)
                img = cv2.cvtColor(texture_ms, cv2.COLOR_BGR2RGB)
                texture_ms = cv2.putText(img, "P:{}".format(np.round(probs[cx].item(), 2)), (125, 30),
                                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness=2 )
            if gt_cam is not None:
                texture_ms = cv2.putText(
                    texture_ms,
                    "E:{}".format(np.round(cam_errors[cx], 1)), (125, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    thickness=2
                )
            if losses_per_hypo is not None:
                for lx, loss_per_hypo in enumerate(losses_per_hypo):
                    texture_ms = cv2.putText(
                        texture_ms,
                        "E:{}".format(np.round(loss_per_hypo[cx].item(), 2)),
                        (125, 90 + lx * 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        thickness=2
                    )
            renderings.append(texture_ms)
        visuals = {}
        visuals['all_hypotheses'] = vis_utils.image_montage(renderings, nrow=3)
        return visuals

    def wrap_texture(self, texture, camera, lights=True, tex_size=6, other_vps=False):
        img, texture = vis_utils_cmr.wrap_texture_and_render(
            self.vis_rend,
            self.verts_obj,
            camera,
            self.uv_sampler,
            texture,
            tex_size,
            other_vps=other_vps,
            lights=lights
        )
        return img, texture

    def set_mean_shape_verts(self, verts):
        self.verts_obj = verts
        return


    def visualize_contour_img(self, contour, img_size):
        '''
        contour  N x 2 index array
        '''
        contour_img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
        contour_img[contour[:, 1], contour[:, 0]] = 255
        return contour_img

    def render_mask_boundary(self, img, mask):
        import cv2
        img_mask = np.stack([mask, mask, mask], axis=2) * 255
        img_mask = cv2.cvtColor(img_mask, cv2.COLOR_BGR2GRAY)
        (thresh, im_bw) = cv2.threshold(
            img_mask.astype(np.uint8), 127, 255, cv2.THRESH_BINARY
        )
        im_bw = im_bw.astype(np.uint8)
        _, contours, hierarchy = cv2.findContours(
            im_bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        img = np.ascontiguousarray(img, dtype=np.uint8)
        new_img = cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
        return new_img


# helper function

def render_UV_contour(img, uvmap, mask):
    mask = mask.permute(1, 2, 0).numpy().squeeze()
    import os
    import scipy.misc
    import tempfile
    import matplotlib.pyplot as plt
    # levels = [0.1*i for i in range(10)]
    cm1 = plt.get_cmap('jet')
    cm2 = plt.get_cmap('terrain')
    plt.imshow(img)
    plt.contour(
        uvmap[:, :, 0].numpy() * mask,
        10,
        linewidths=1,
        cmap=cm1,
        vmin=0,
        vmax=1
    )
    plt.contour(
        uvmap[:, :, 1].numpy() * mask,
        10,
        linewidths=1,
        cmap=cm2,
        vmin=0,
        vmax=1
    )
    plt.axis('off')
    temp_file = '/tmp/file_{}.jpg'.format(np.random.randint(10000000))
    plt.savefig(temp_file)
    image = scipy.misc.imread(temp_file)
    os.remove(temp_file)
    plt.close()
    return image