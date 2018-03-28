import dlib
import numpy as np
import skimage.transform as tr
from enum import Enum

class FaceDetector:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()

    def detect_faces(self,
                     image, *,
                     upscale_factor=1,
                     greater_than=None,
                     get_top=None):
        try:
            face_rects = list(self.detector(image, upscale_factor))
        except Exception as e:
            raise FaceDetectorException(e.args)

        if greater_than is not None:
            face_rects = list(filter(lambda r:
                              r.height() > greater_than and r.width() > greater_than,
                              face_rects))

        face_rects.sort(key=lambda r: r.width() * r.height(), reverse=True)

        if get_top is not None:
            face_rects = face_rects[:get_top]

        return face_rects


class FaceAlignMask(Enum):
    INNER_EYES_AND_BOTTOM_LIP = [39, 42, 57]
    OUTER_EYES_AND_NOSE = [36, 45, 33]


class FaceAligner:
    def __init__(self,
                 dlib_predictor_path,
                 face_template_path):
        self.predictor = dlib.shape_predictor(dlib_predictor_path)
        self.face_template = np.load(face_template_path)

    def get_landmarks(self,
                      image,
                      face_rect):
        points = self.predictor(image, face_rect)
        return np.array(list(map(lambda p: [p.x, p.y], points.parts())))

    def align_face(self,
                   image,
                   face_rect, *,
                   dim=96,
                   border=0,
                   mask=FaceAlignMask.INNER_EYES_AND_BOTTOM_LIP):
        mask = np.array(mask.value)

        landmarks = self.get_landmarks(image, face_rect)
        proper_landmarks = border + dim * self.face_template[mask]
        A = np.hstack([landmarks[mask], np.ones((3, 1))]).astype(np.float64)
        B = np.hstack([proper_landmarks, np.ones((3, 1))]).astype(np.float64)
        T = np.linalg.solve(A, B).T

        wrapped = tr.warp(image,
                          tr.AffineTransform(T).inverse,
                          output_shape=(dim + 2 * border, dim + 2 * border),
                          order=3,
                          mode='constant',
                          cval=0,
                          clip=True,
                          preserve_range=True)

        return wrapped

    def align_faces(self,
                    image,
                    face_rects,
                    *args,
                    **kwargs):
        result = []

        for rect in face_rects:
            result.append(self.align_face(image, rect, *args, **kwargs))

        return result


def clip_to_range(img):
    return img / 255.0
