import numpy as np
import cv2

"""
1) Functions for removing a box inside a box
"""

def box_inside_box(coords1, coords2):
    """
    returns True if the second box is inside the first box
    coords are in the form [x1, y1, x2, y2]
    """
    
    if (coords2[1] >= coords1[1]) and (coords2[3] <= coords1[3]) and (coords2[0] >= coords1[0]) and (coords2[2] <= coords1[2]):
        return True
    else:
        return False
    
def remove_box_inside_box(box_list):
    """
    Given a list of boxes with coordinates [x1 y1 x2 y2]
    Check if any boxes are completely inside another box. 
    If there are fewer than 3 boxes inside another box (eg. its not a root with lots of stuff in it)
    Assume these are wrong and remove them
    """
    pop_boxes = []
    for i, box in enumerate(box_list):
        #the list of all the other boxes to check against
        ebox_list = [ebox for ebox in box_list if box != ebox]
        isInside = False
        eq_i = 0 #counter to keep track of how many boxes are inside other boxes
        
        boxes_to_pop = []
        for j,ebox in enumerate(ebox_list):
            #check how many boxes are inside this box
            if box_inside_box(box, ebox) == True:
                isInside = True
                eq_i += 1
                boxes_to_pop.append(ebox)
        
        #add all the boxes to the to pop list
        if (isInside == True) and (eq_i <= 3):
            pop_boxes.extend(boxes_to_pop)

    for pop_box in pop_boxes:
        try:
            box_list.remove(pop_box)
        except:
            pass
    return box_list

"""
2) Functions for merging boxes that might need to be merged
"""
def has_overlap(coords1, coords2):
    """
    check if first box overlaps at least 25% with the second box in x-coordinates
    """
    r_l = [[],[]]
    for i in range(2):
        r_l[i].extend([coords1[i*2], coords2[i*2]])
    
    #the total x-space extended by the two boxes
    totl = max(r_l[1]) - min(r_l[0]) 
    overlap = min(r_l[1]) - max(r_l[0])
    
    xlen_b1 = coords1[2] - coords1[0]
    xlen_b2 = coords2[2] - coords2[0]
    lenlist = [xlen_b1, xlen_b2]
    
    #if one box extends beyond the other on both sides, the overlap should just be 1
    if max([xlen_b1, xlen_b2]) == totl:
        overlap = totl

    if overlap/totl > 0.25:
        return True
    else:
        return False

def merge_boxes(coords1, coords2):
    """
    Take two boxes with coords [x1, y1, x2, y2]
    and merge them into a box spanning both
    """
    new_x1 = min(coords1[0], coords2[0])
    new_y1 = min(coords1[1], coords2[1])
    new_x2 = max(coords1[2], coords2[2])
    new_y2 = max(coords1[3], coords2[3])
    
    return new_x1, new_y1, new_x2, new_y2

def infer_boxes_to_merge(boxl):
    """
    For a list of boxes that are overlapping in x-coordinates, check which box is the closest in distance either above or below the box
    If a box on either side matches certain criteria about the x and ylengths, assume it's two components of an equals sign or ! or i, etc.
    And return the coordinates of the two boxes that should be merged
    """
    
    box_0 = boxl[0]
    y_c0 = (box_0[3] + box_0[1])/2
    
    #find two closest vertical boxes on either side (above, and below)
    ydist_min, ydist_max = 1000, 1000
    box_i_min, box_i_max = 0, 0
    for i,box in enumerate(boxl[1:]):
        yb = (box[3] + box[1])/2
        
        ydist = yb - y_c0
        if ydist < 0:
            if abs(ydist) < ydist_min:
                ydist_min = abs(ydist)
                box_i_min = i+1
        elif ydist >= 0:
            if ydist < ydist_max:
                ydist_max = ydist
                box_i_max = i+1

    #these are the two closest boxes above and below the box
    cbox_down = boxl[box_i_min]
    cbox_up = boxl[box_i_max]
    
    #get lengths in x-coordinates for the box and the two closest boxes
    xlen_box = box_0[2] - box_0[0]
    xlen_cbox_up = cbox_up[2] - cbox_up[0]
    xlen_cbox_down = cbox_down[2] - cbox_down[0]
    
    xlens_up = [xlen_box, xlen_cbox_up]
    xlens_down = [xlen_box, xlen_cbox_down]
    
    #also check if the one of the boxes it not much longer vertically than the other one
    ylen_box = box_0[3] - box_0[1]
    ylen_cbox_up = cbox_up[3] - cbox_up[1]
    ylen_cbox_down = cbox_down[3] - cbox_down[1]
    
    ylens_up = [ylen_box, ylen_cbox_up]
    ylens_down = [ylen_box, ylen_cbox_down]
    
    #if 1) one of the boxes is not more than 1.8 as long (in x-coords) as the other one
    # AND 2) the y-length of one box is not more than 4x that of the other one AND 5x the ylength of the smallest box is larger than the y-distance between the boxes
    # AND 3) the closest box is not the box itself
    # then assume that the boxes should be merged
    
    is_ol_up = max(xlens_up) < 1.7*min(xlens_up) and   \
            (max(ylens_up) < 5*min(ylens_up) and 5*min(ylens_up) > ydist_max) \
            and cbox_up != box_0
    
    is_ol_down = max(xlens_down) < 1.7*min(xlens_down) and \
            (max(ylens_down) < 5*min(ylens_down) and 5*min(ylens_down) > ydist_min ) \
            and cbox_down != box_0

    if is_ol_up == True:
        return [box_0, cbox_up]
    elif is_ol_down == True:
        return [box_0, cbox_down]
    else:
        return [box_0]
        
def create_merged_boxes(box_list):
    
    #now check if any boxes overlap with each other in x coordinates
    tot_overlap_boxes = []
    merged_box_list = []

    for i, box in enumerate(box_list):
        #the list of all the other boxes to check against
        ebox_list = [ebox for ebox in box_list if box != ebox] 
        eq_b = 0
        overlap_boxes = [box]
        for j,ebox in enumerate(ebox_list):
            if has_overlap(box, ebox) == True:
                overlap_boxes.append(ebox)
    
        tot_overlap_boxes.append(overlap_boxes)
        
    pop_boxes = []
    
    #find which of the overlapping boxes might fit the criteria to be merged
    for i in range(len(tot_overlap_boxes)):
        if len(tot_overlap_boxes[i]) > 1:
            boxes_to_merge = infer_boxes_to_merge(tot_overlap_boxes[i])
            if len(boxes_to_merge) == 2:
                #if a pair of boxes to be merged is found, merge them and add the individual boxes to the remove list
                merged_box_list.append(merge_boxes(*boxes_to_merge))
                pop_boxes.extend(boxes_to_merge)
    
    #remove all the boxes that were merged
    for pop_box in pop_boxes:
        try:
            box_list.remove(pop_box)
        except:
            pass
    return box_list, merged_box_list
    
"""
3) Functions for figuring out box order
"""

