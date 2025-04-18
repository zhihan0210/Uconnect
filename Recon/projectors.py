import astra


class ParamProj:
    def __init__(self, n_angles, vox_size, beam_type, det_width, det_count, source_origin, origin_det, proj_type):
        self.n_angles = n_angles
        self.vox_size = vox_size # in mm
        self.beam_type = beam_type  # 'parallel' or 'fanflat' or 'cone'
        self.det_width = det_width  # in mm
        self.det_count = det_count
        self.source_origin = source_origin  # in mm
        self.origin_det = origin_det  # in mm
        self.proj_type = proj_type  # 'cuda' or 'line_fanflat' or 'linear'

    def get_paramProj(self):
        return self


def forward_projection(img, proj_id):
    sino_id, sino = astra.creators.create_sino(img, proj_id)
    astra.data2d.delete(sino_id)
    return sino


def back_projection(sino, proj_id):
    bp_id, img = astra.create_backprojection(sino, proj_id)
    astra.data2d.delete(bp_id)
    return img
