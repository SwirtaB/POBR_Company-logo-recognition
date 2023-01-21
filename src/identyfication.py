from typing import List, Tuple

import numpy as np
from segmentation import Segment
import config
import copy


def filter_segments(segments: List[List[Segment]]) -> List[List[Segment]]:
    result = []
    mask_segments_list = []
    for segment_list in segments:
        for i, segment in enumerate(segment_list):
            if ((config.MIN_W3 <= segment.W3 <= config.MAX_W3) and
                (config.MIN_M1_NORM <= segment.M1_norm <= config.MAX_M1_NORM)
                    and
                (config.MIN_M7_NORM <= segment.M7_norm <= config.MAX_M7_NORM)):
                mask_segments_list.append(segment)

        result.append(copy.deepcopy(mask_segments_list))
        mask_segments_list.clear()

    return result


def try_recognize_logo(segments: List[List[Segment]]) -> List[List[Segment]]:
    return segments