from typing import List, Tuple

import numpy as np
from segmentation import Segment
import config
import copy


def filter_segments(segments: List[List[Segment]]) -> List[List[Segment]]:
    result = []
    mask_segments_list = []
    for segment_list in segments:
        for segment in segment_list:
            if ((config.MIN_W3 <= segment.W3 <= config.MAX_W3) and
                (config.MIN_M1_NORM <= segment.M1_norm <= config.MAX_M1_NORM)
                    and
                (config.MIN_M7_NORM <= segment.M7_norm <= config.MAX_M7_NORM)):
                mask_segments_list.append(segment)

        result.append(copy.deepcopy(mask_segments_list))
        mask_segments_list.clear()

    return result


def try_recognize_logo(segments: List[List[Segment]]) -> List[List[Segment]]:
    logos = []
    for R_square in segments[0]:
        logo = [R_square]
        for G_square in segments[1]:
            if (R_square.bbox[0][0] <= G_square.global_center[0] <=
                    R_square.bbox[1][0]) and (R_square.global_center[1] <
                                              G_square.global_center[1]):
                logo.append(G_square)

        for B_square in segments[1]:
            if (R_square.bbox[0][1] <= B_square.global_center[1] <=
                    R_square.bbox[1][1]) and (R_square.global_center[0] <
                                              B_square.global_center[0]):
                logo.append(B_square)

        if len(logo) != 3:
            continue

        for Y_square in segments[1]:
            if (Y_square.bbox[0][1] <= logo[1].global_center[1] <=
                    Y_square.bbox[1][1]
                    and Y_square.global_center[0] > logo[1].global_center[0]
                ) and (Y_square.bbox[0][0] <= logo[2].global_center[0] <=
                       Y_square.bbox[1][0] and
                       Y_square.global_center[1] > logo[2].global_center[1]):
                logo.append(Y_square)

        if len(logo) == 4:
            logos.append(logo)

    return logos