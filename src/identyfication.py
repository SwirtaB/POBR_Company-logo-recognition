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


# FIXME refactor
def try_recognize_logo(segments: List[List[Segment]]) -> List[List[Segment]]:
    recognized_logos = []
    for R_square in segments[0]:
        logo = [R_square]
        for G_square in segments[1]:
            if (R_square.bbox[0][0] <= G_square.global_center[0] <=
                    R_square.bbox[1][0]) and (
                        R_square.global_center[1] < G_square.global_center[1] <
                        R_square.bbox[1][1] + R_square.local_image.shape[1]):
                logo.append(G_square)

        for B_square in segments[2]:
            if (R_square.bbox[0][1] <= B_square.global_center[1] <=
                    R_square.bbox[1][1]) and (
                        R_square.global_center[0] < B_square.global_center[0] <
                        R_square.bbox[1][0] + R_square.local_image.shape[0]):
                logo.append(B_square)

        if len(logo) != 3:
            continue

        for Y_square in segments[3]:
            if (Y_square.bbox[0][1] <= logo[1].global_center[1] <=
                    Y_square.bbox[1][1]
                    and Y_square.global_center[0] > logo[1].global_center[0]
                ) and (Y_square.bbox[0][0] <= logo[2].global_center[0] <=
                       Y_square.bbox[1][0] and
                       Y_square.global_center[1] > logo[2].global_center[1]):
                logo.append(Y_square)

        if len(logo) == 4:
            recognized_logos.append(logo)

    recognized_logos = _topology_check(segments)
    return recognized_logos


def _topology_check(segments: List[List[Segment]]) -> List[List[Segment]]:
    if len(segments[0]) == 0:
        return []

    recognized_logos = []
    segments_tmp = copy.deepcopy(segments)
    logo: List[Segment | None] = [None, None, None, None]
    search_area = ((-1, -1), (-1, -1))
    empty_flag = False

    while not empty_flag:
        for R_square in segments_tmp[0]:
            search_area = (R_square.bbox[0], (R_square.bbox[1][0] +
                                              R_square.local_image.shape[0],
                                              R_square.bbox[1][1] +
                                              R_square.local_image.shape[1]))
            logo[0] = copy.deepcopy(R_square)
            segments_tmp[0].remove(R_square)
            if len(segments_tmp[0]) == 0:
                empty_flag = True
            break
        for G_square in segments_tmp[1]:
            if (search_area[0][0] < G_square.global_center[0] <
                    search_area[1][0]) and (search_area[0][1] <
                                            G_square.global_center[1] <
                                            search_area[1][1]):
                logo[1] = copy.deepcopy(G_square)
                segments_tmp[1].remove(G_square)
                if len(segments_tmp[1]) == 0:
                    empty_flag = True
                break
        for B_square in segments_tmp[2]:
            if (search_area[0][0] < B_square.global_center[0] <
                    search_area[1][0]) and (search_area[0][1] <
                                            B_square.global_center[1] <
                                            search_area[1][1]):
                logo[2] = copy.deepcopy(B_square)
                segments_tmp[2].remove(B_square)
                if len(segments_tmp[2]) == 0:
                    empty_flag = True
                break
        for Y_square in segments_tmp[3]:
            if (search_area[0][0] < Y_square.global_center[0] <
                    search_area[1][0]) and (
                        search_area[0][1] < Y_square.global_center[1] <
                        search_area[1][1]) and (R_square.bbox[1][0] <
                                                Y_square.global_center[0]):
                logo[3] = copy.deepcopy(Y_square)
                segments_tmp[3].remove(Y_square)
                if len(segments_tmp[3]) == 0:
                    empty_flag = True
                break

        if logo.count(None) == 0:
            if logo[0].global_center[1] < logo[1].global_center[1] and logo[
                    0].global_center[0] < logo[2].global_center[0] and logo[
                        3].global_center[1] > logo[2].global_center[1] and logo[
                            3].global_center[0] > logo[1].global_center[0]:
                recognized_logos.append(copy.deepcopy(logo))

    return recognized_logos