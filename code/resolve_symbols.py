import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from box_positions import BoxPositions


"""
Preprocessing 'resolve symbols' pipeline, to create images of individual symbols that the model should make predictions on
See notebook resolve_symbols.ipynb for a more detailed explanation of each step
"""        
    
"""
Step 2) Finding which boxes should be merged
"""
def find_boxes_to_merge(boxl, img_ysize):
    """
    For a list of boxes that are overlapping in x-coordinates, check which box is the closest in distance either above or below the box
    If a box on either side matches certain criteria about the x and ylengths, assume it's two components of an equals sign or ! or i, etc.
    And return the coordinates of the two boxes that should be merged
    Input:
        1) boxl (list) - list of box coordinates, in the form [x1, y1, x2, y2]
        2) img_ysize (int)
    Returns:
        1) list of boxes to be merged. If no merge was found, list will contain a single entry
    """
    
    box_0 = boxl[0]

    #find two closest vertical boxes on either side (above, and below)
    ydist_min, ydist_max = 1000, 1000
    box_i_min, box_i_max = 0, 0
    
    for i,box in enumerate(boxl[1:]):
        ydist =  BoxPositions(box_0, box).ydist 
        if ydist < 0:
            if abs(ydist) < ydist_min:
                ydist_min = abs(ydist)
                box_i_min = i+1
        elif ydist >= 0:
            if ydist < ydist_max:
                ydist_max = ydist
                box_i_max = i+1

    #these are the two closest boxes above and below the box
    box_pos_down = BoxPositions(box_0, boxl[box_i_min])
    box_pos_up = BoxPositions(box_0, boxl[box_i_max])
    
    #if 1) the bigger box is NOT 1.7 bigger than the smaller box in x
    # 2) the bigger box is not NOT 5x bigger than the smaller box in y
    # 3) 5x the length of the smaller box is NOT smaller than the distance between the boxes
    # 4) the boxes aren't the same box
    # 5) one of the boxes is not inside the other box
    # 6) the total size of the combined boxes is not more than 0.2x the image height
    # 7) AND the total number of overlapping boxes is not 3 (in which case it's more likely to be a simple fraction like 1/3)
    # - then return the two boxes to be merged

    is_ol_up = not box_pos_up.isBigger(1.7, axis='x') and not box_pos_up.isBigger(5, axis='y') and 5*min(box_pos_up.ylens) > ydist_max and not box_pos_up.isSame() 
    is_ol_down = not box_pos_down.isBigger(1.7, axis='x') and not box_pos_down.isBigger(5, axis='y') and 5*min(box_pos_down.ylens) > ydist_min and not box_pos_down.isSame() 

    inside_check_up = not box_pos_up.isInside() and not box_pos_up.isInside(invert=True)
    inside_check_down = not box_pos_down.isInside() and not box_pos_down.isInside(invert=True)

    tot_size_check_up = abs(max(box_pos_up.y2s) - min(box_pos_up.y1s)) > 0.18 * img_ysize
    tot_size_check_down = abs(max(box_pos_down.y2s) - min(box_pos_down.y1s)) > 0.18 * img_ysize

    if is_ol_up == True and inside_check_up == True and len(boxl) != 3 and not tot_size_check_up:
        return [box_0, boxl[box_i_max]]
    elif is_ol_down == True and inside_check_down == True and len(boxl) != 3 and not tot_size_check_down:
        return [box_0, boxl[box_i_min]]
    else:
        return [box_0]
    
def create_merged_boxes(box_list, img_ysize):
    """
    Given a list of boxes, find out which ones have enough overlap in x-coordinates to be considered for merging
    Then pass this list to find_boxes_to_merge to determine the two that might be merged
    """
    #now check if any boxes overlap with each other in x coordinates
    tot_overlap_boxes = []
    merged_box_list = []

    for i, box in enumerate(box_list):
        #the list of all the other boxes to check against
        ebox_list = [ebox for ebox in box_list if box != ebox] 
        eq_b = 0
        overlap_boxes = [box]
        for j,ebox in enumerate(ebox_list):
            if BoxPositions(box, ebox).calc_Overlap(axis='x') > 0.25:
                overlap_boxes.append(ebox)
        tot_overlap_boxes.append(overlap_boxes)
        
    rm_boxes = []
    
    #find which of the overlapping boxes might fit the criteria to be merged
    for i in range(len(tot_overlap_boxes)):
        if len(tot_overlap_boxes[i]) > 1:
            boxes_to_merge = find_boxes_to_merge(tot_overlap_boxes[i], img_ysize)
            if len(boxes_to_merge) == 2:
                #if a pair of boxes to be merged is found, merge them and add the individual boxes to the remove list
                merged_box_list.append(BoxPositions(*boxes_to_merge).merge_boxes())
                rm_boxes.extend(boxes_to_merge)
        
    #remove all the boxes that were merged
    for rm_box in rm_boxes:
        try:
            box_list.remove(rm_box)
        except:
            pass

    return box_list, merged_box_list
        
"""
Step 3) Finding the order of symbols
"""
def above_below_box(box, overlap_boxlist):
    """
    Given a box and the list boxes with overlapping x-coordinates, check if the overlapping boxes are above or below
    (or both) the box.
    Returns:
        0 if there are boxes below, 1 if there are boxes both above and below, and 2 if there are boxes above
    """
    ymin_0 = box[1]
    ymins = np.array([overlap_boxlist[j][1] for j in range(len(overlap_boxlist))])
    
    hasAbove = False
    hasBelow= False
    for ymin in ymins:
        if ymin > ymin_0:
            hasBelow = True
        if ymin < ymin_0:
            hasAbove = True
    if hasBelow == True and hasAbove == True:
        return 1
    elif hasBelow == True:
        return 0
    elif hasAbove == True:
        return 2
    else:
        return 0
    
def scan_x_gaps(box_list):
    """
    Given a list of boxes, find which x-coordinates have no boxes overlapping them
    Returns: a list of x-coordinates (at intervals of 5 pixels) that have no overlapping boxes
    """
    #get min and max 
    xmins = np.array([box_list[j][0] for j in range(len(box_list))])
    xmaxs = np.array([box_list[j][2] for j in range(len(box_list))])
    
    xmin_boxes = np.min(xmins)
    xmax_boxes = np.max(xmaxs)
    
    x_gap_list = []
    for x in range(xmin_boxes, xmax_boxes, 5):
        has_overlap = False
        for box in box_list:
            if box[0] <= x <= box[2]:
                has_overlap = True
                break
        if has_overlap == False:
            x_gap_list.append(x)
            
    return x_gap_list
    
def determine_box_level(box_list):
    """
    Given a list of boxes with coordinates [x1 y1 x2 y2],
    Find out which 'level' of the equation the box is in, and whether the box is part of a stack or not
    I define 'level' as a sequence of symbols that can be read left to right in an equation
    And a 'stack' is levels that are on top of each other
    Eg: the equation 1 + 5x + (9-x^2)/(4+x) has 4 levels: 1 + 5x +, (9-x^2), /, and 4+x
    The levels 9-x^2, /, and 4+x are inside a stack that should be read top to bottom
    """
 

    if len(box_list) == 1:
        return box_list, [0], [0]
    
    #first sort all the boxes by their x1 coordinate    
    xmins_l = np.array([box_list[j][0] for j in range(len(box_list))])
    s_box_list = [box for (_, box) in sorted(zip(xmins_l, box_list))]

    #for each box obtain a list of the boxes this box overlaps with, matching certain criteria
    overlap_list = []
    for b, box in enumerate(s_box_list):
        
        #list of all the boxes to check against
        ebox_list = [ebox for ebox in s_box_list if box != ebox] 
        overlap_boxes = []
        overlap_labels = []
        for j,ebox in enumerate(ebox_list):
            box_pos = BoxPositions(box, ebox)
            #here, I will consider a box to be 'overlapping', if 30% of the smaller box is covered in x-coordinates by the bigger box
            #and the boxes overlap lessathan 30% in y-coordinates
            if box_pos.calc_Overlap(axis='x', relative_to='smaller') > 0.3 and box_pos.calc_Overlap(axis='y', relative_to='both') < 0.30:
                overlap_boxes.append(ebox)
                overlap_labels.append(s_box_list.index(ebox))
        overlap_list.append(overlap_boxes)
        
    #now I have the list for each box that this box overlaps with, and I can determine the levels
    level = 0 #initial level
    level_list = []
    stacked_level_list = [] #to keep track of whether the level is part of a stack or not
    in_levels= False #whether we are evaluating boxes inside a stack
    enter_level = False #whether
    middle_symbol = 0 #to keep track of the 'middle symbol' in a stack. There can only be one per stack
    
    
    #x_gaps to figure out when we enter a new stack when two stacks are adjacent to one another
    x_gap_list = scan_x_gaps(s_box_list)
    
    for b, box in enumerate(s_box_list):
        
        #check if a box shares a single overlap with another box. in that case, don't count it as a stack cause its likely to be a super/subscript
        if len(overlap_list[b]) > 0:
            #the index of the box that this box overlaps with
            ol_idx = s_box_list.index(overlap_list[b][0])
            #if both overlap lists have length 1 (it implies their overlapping box is each other) 
            if len(overlap_list[b]) == 1 and len(overlap_list[ol_idx]) == 1:
                single_overlap = True
            else:
                single_overlap = False
        else:
            single_overlap = False
        
        #check if box has overlapping boxes that is not a single overlap
        if len(overlap_list[b]) > 0 and single_overlap == False:
            #if we are in a stack, but the box has crossed an x-gap compared to the previous box , that means we are in a new stack
            for x_gap in x_gap_list:
                if in_levels == True and s_box_list[b-1][2] < x_gap < box[0]:
                    level += 3
                    middle_symbol = 0
                    break
            
            #if we were in an unstacked part and we are entering a stack, add a level
            if enter_level == True:
                level += 1
                enter_level = False
            #are there boxes both above and below this box?
            #0 = below
            #1 = both
            #2 = above
            level_add = above_below_box(box, overlap_list[b])

            #keep track of middle symbol in the stack and find the yc coords
            if level_add == 1 and middle_symbol == 0:
                yc = box[1]*0.5 + box[3]*0.5
            if level_add == 1: 
                middle_symbol += 1
                
            #if a second 'middle symbol' is found in the stack, it's probably part of the upper or lower level instead
            #find out which by comparing ycenter coordinates
            if middle_symbol > 1 and level_add ==1: 
                yc_new = box[1]*0.5 + box[3]*0.5
                if yc_new > yc: level_add = 2
                else: level_add = 0
                
            level_list.append(level + level_add)
            stacked_level_list.append(level_add+1)
            in_levels = True
        else:
            #if we were in a stack previously but not anymore, add 3 to indicate we are in a new level (because I assume hte stack has 3 levels)
            if in_levels == True: 
                middle_symbol = 0
                level +=3 
            in_levels = False
            enter_level = True
            #if there are no overlapping boxes, we can set the level to the same as the previous level and move on
            level_list.append(level)
            stacked_level_list.append(0)
    

    #finally, sort the boxes by level
    s_box_list = [box for (_, box) in sorted(zip(level_list, s_box_list))]
    s_stacked_level_list = [isstack for (_, isstack) in sorted(zip(level_list, stacked_level_list))]
    s_level_list = sorted(level_list)
    
    #return 1) the list of boxes, sorted by level and xmin coords
    #2) the list of levels (sorted)
    #3) a boolean list, which for each box shows if its part of a stack or not
    return s_box_list, s_level_list, s_stacked_level_list

def merge_dots(box_list, stack_list, level_list, img_ysize):
    """
    After the symbols have been sorted, make one final iteration through 
    If there are symbols where one box is entirely above or below the other, AND they are not part of a stack, merge them
    This helps remove some of the individual dots above i's or above other symbols
    """
    
    merged_dots_list = []
    rm_box_list = []
    merge_ind_list = []

    for b,box in enumerate(box_list[:-1]):
        cbox = box_list[b+1]

        #if stack_list[b] == False and stack_list[b+1] == False: 
        if stack_list[b] == stack_list[b+1]:
            box_pos = BoxPositions(box, cbox)
            tot_size_check = abs(max(box_pos.y2s) - min(box_pos.y1s)) > 0.18 * img_ysize
            if (box_pos.calc_Overlap(axis='x', relative_to='smaller') > 0.75) and (box_pos.calc_Overlap(axis='y') < 0.2) and not tot_size_check:
                merged_dots_list.append(box_pos.merge_boxes())
                rm_box_list.append([box, cbox])
                merge_ind_list.append([b, b+1])
            

    #Since the boxes are already sorted, I want to preserve order. That's why I loop backwards over the list
    #For the box_list i'll replace one entry with the merged box and remove the other one
    #For the stacked list and level list, I'll just remove one entry
    for i, rm_boxes in enumerate(rm_box_list[::-1]):
        replace_idx = merge_ind_list[::-1][i][0]
        pop_idx = merge_ind_list[::-1][i][1]
        
        box_list[replace_idx] = merged_dots_list[::-1][i]
        box_list.pop(pop_idx)
        
        stack_list.pop(pop_idx)
        level_list.pop(pop_idx)
        
    return box_list, stack_list, level_list
    
"""
Step 4) Finding the sub/superscript level
"""
def sub_or_superscript_level(coords1, level1, coords2, level2):
    """
    Given two boxes and their levels, determine whether the second box is a subscript or superscript of the first box
    """
    # if the symbols are not the same level, return -10
    if level1 != level2:
        return -10, 0
    
    box_pos = BoxPositions(coords1, coords2)
    
    #calculate the 'superscript score', a combination of the ylen ratio and how far the second box sticks up or down from the first one
    #1) 1 - ylen_ratio
    s1 = 1 - box_pos.ylen_b2/box_pos.ylen_b1
    
    #2) how much does the box stick out?    
    s2 = max([box_pos.calc_box_extends(), 0])

    s2 = s2 
    
    #s3 = box_pos.calc_box_extends_center()

    tot_score = s1*0.8 + s2*3.2 #+ s3*0.45
        
    #check if it's a subscript or superscript
    yc_1, yc_2 = box_pos.ycs
    
    if tot_score > 1 and yc_1 > yc_2:
        return 1, tot_score
    elif tot_score > 1 and yc_1 < yc_2:
        return -1, tot_score
    else:
        return 0, tot_score
        
"""
Step 5) Isolate symbols and put them in square arrays
"""
def isolate_symbols_and_square(box_list, level_list, ind_symbols):
    """
    After all the symbol resolving is done, I will do one final check to see if there are any boxes that have another box contained within them
    This will primarily be the case for roots
    If that is the case, I will use opencvs drawcontours to re-draw the shape
    Finally, I will make sure every symbol is in a square array, to make rescaling easier when calling the model
    """
    
    boxes_checked = []
    extend_list = [] #to figure out how far a square root should extend over subsequent symbols
    for i, box in enumerate(box_list):
        ext_counter = 0
        boxes_checked.append(box)
        #the list of all the other boxes to check against
        ebox_list = [ebox for ebox in box_list if ebox not in boxes_checked]
        eq_i = 0 #counter to keep track of how many boxes are inside other boxes
        boxes_to_rm = []
        for j,ebox in enumerate(ebox_list):
            #check if the second box is completely inside the other one
            if BoxPositions(box, ebox).isInside() == True:
                eq_i += 1
        
        #second loop to figure out how far a root should extend
        #if the next boxes 1) overlap and 2) are on the same level as this one, add +1 to the extension list
        # keep going until we hit a box that is not on this level
        for j,ebox in enumerate(ebox_list):
            b_idx = box_list.index(ebox)
            if level_list[i] != level_list[b_idx]:
                break
            if BoxPositions(box, ebox).isPartInside() == True:
                ext_counter += 1
            
        extend_list.append(ext_counter)

        #if the box has other boxes entirely contained within it, re-draw with drawContours
        if eq_i > 0:
            symbol = ind_symbols[i]
            s_ret, s_thresh=cv2.threshold(symbol,200,255,cv2.THRESH_BINARY)
            s_ctrs, s_ret =cv2.findContours(s_thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
            s_ctrs =sorted(s_ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])
            
            stencil = np.zeros(symbol.shape).astype(symbol.dtype)
            
            #draw the contour
            cv2.drawContours(stencil, [s_ctrs[1]], -1, color=(255, 255, 255), thickness=cv2.FILLED)

            ind_symbols[i] = ~stencil
    
        
        symbol_shape = ind_symbols[i].shape
        
        #some ugly code to center the symbol in a new, square array
        #add about 10% white space on each side
        white_pix = max(symbol_shape) // 10
        if white_pix < 2: white_pix = 2
        if max(symbol_shape)%2 == 0:
            ms = max(symbol_shape) +white_pix
        else:
            ms = max(symbol_shape) + white_pix + 1
        new_symbol = np.zeros((ms, ms)) + 255

        sd_x = ms - symbol_shape[0]
        sd_y = ms - symbol_shape[1]

        new_symbol[sd_x//2:-sd_x//2,sd_y//2:-sd_y//2] = ind_symbols[i]
        ind_symbols[i] = new_symbol
        
        #add slight blur, makes model predictions more reliable
        ind_symbols[i] = cv2.blur(ind_symbols[i],(3,3))
                                     
    return ind_symbols, extend_list

"""
Step 6) Combine everything and resolve symbols
"""

def resolve_symbols_on_img(img_file, plot=True):
    """
    Given an input image file, use opencv's findContours to find the contours related to mathematical symbols,
    and prepare them for model prediction and equation rendering.
    The following steps are performed:
    1) Inner contour boxes are removed
    2) Boxes that are likely to belong together (like the two halves of an = or a !) are merged
    3) The order of the symbols in the equation is determined, as well as whether they are part of a 'stack' (like in a fraction)
    4) The 'sub/superscript' level of each symbol is determined
    5) For symbols with bounding boxes covering other symbols, the symbol is redrawn with opencv using the contours
    6) Each symbol is put inside a square array
    Returns:
        1) A list of square image arrays, ordered by how the symbol should appear in an equation
        2) a 'level list'. Symbols with the same level can be rendered left to right. A new level indicates some change, either 
        3) a 'stack list'. 0 for a symbol not in a stack, 1/2/3 for symbols in the top/middle/bottom of a stack, respectively
        4) a 'script level list'. To determine whether a symbol is a sub/superscript of the previous one. equal script levels means the symbol should be at equal line height
        5) fig and ax objects, if plot=True
    """
    #find contours
    img = cv2.imread(img_file,cv2.IMREAD_GRAYSCALE)
    img_size = img.shape[0] * img.shape[1]

    #if more than 90% of pixels are already at a perfect black/white (0 or 255), just use a simple binary threshholding
    bw_pixs = len(img[(img == 0) | (img == 255)]) / img_size
    if bw_pixs > 0.9:
        ret,thresh=cv2.threshold(img, 230, 255, cv2.THRESH_BINARY)
    #else: use a combination of adaptive threshholding together with a linear cut after to get rid of as many small-scale shadows/dots etc as possible
    else:
        thresh  = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
        blur = cv2.GaussianBlur(thresh,(13,13),0)
        bt = 140
        ret,thresh=cv2.threshold(blur,bt, 255, cv2.THRESH_BINARY)

    ctrs, ret =cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cnt=sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0] + cv2.boundingRect(ctr)[1])

    if plot:
        fig, ax = plt.subplots()
        ax.imshow(img, cmap='gray')

    box_list = []
    
    #get the bounding boxes for all contours
    for i,c in enumerate(cnt[1:]):
        x1,y1,x2,y2= cv2.boundingRect(c)
        #switch to absolute x and y coordinatees (not x1, y1, xlen, ylen)
         
        #step 1) calculate the area of the contour - if negative, the contour is an inner contour and should be excluded
        ctr_ar = cv2.contourArea(c, oriented=True)
     
        
        #only include boxes that are a certain % of the total image area
        if x2*y2 > 2.2e-4 *img_size and ctr_ar > 0:
            box_list.append([x1, y1, x2+x1, y2+y1])
    
    #step 2) find which boxes should be merged, and remove the individual boxes
    box_list, merged_box_list = create_merged_boxes(box_list, img.shape[0])

    #remove non-unique boxes (there might be duplicates in the merged box list)
    tot_boxes = box_list + merged_box_list
    tot_boxes = [list(x) for x in set(tuple(x) for x in tot_boxes)]

    #step 3) of all the boxes that are left, determine the order
    tot_boxes, box_levels, stacked_list = determine_box_level(tot_boxes)
    
    #intermediate step: extra merging
    tot_boxes, stacked_list, box_levels = merge_dots(tot_boxes, stacked_list, box_levels, img.shape[0])

    #step 4) figure out the 'script level' of each symbol (whether it's on the line, or sub/superscript)
    script_level = 0 
    script_level_list = [0]
    if len(tot_boxes) > 1:
        for b, box in enumerate(tot_boxes[:-1]):
            script_add, score = sub_or_superscript_level(tot_boxes[b], box_levels[b], tot_boxes[b+1], box_levels[b+1])
            if script_add == -10:
                script_level = 0
            else:
                script_level += script_add
            script_level_list.append(script_level)

    colors = ['r', 'g', 'b', 'g']
    
    #plot the bounding boxes with some information 
    ind_symbols = []
    for i, box in enumerate(tot_boxes):
        x1 = box[0]
        y1 = box[1]
        x2 = box[2] - box[0]
        y2 = box[3] - box[1]
   
        ind_symbols.append(thresh[y1:y1+y2,x1:x1+x2])
    
        if plot:

            if stacked_list[i] > 0:
                er = 'b'
            else:
                er = 'r'
            
            if script_level_list[i] != 0:
                er = 'g'
            
            rect = patches.Rectangle((x1, y1), x2, y2, linewidth=1, edgecolor=er, facecolor='none')

            ax.add_patch(rect)
            ax.text(x1, y1, str(i))

    if plot:
        ax.plot([], color='b', label='Stacked symbols') #dummies for legend
        ax.plot([], color='r', label='Base level symbols')
        ax.plot([], color='g', label='Super/subscripts')
        ax.legend(frameon=False)
    
    #step 5) make a list for the individual symbols
    ind_symbols, extend_list = isolate_symbols_and_square(tot_boxes, box_levels, ind_symbols)
    
    if plot:
        return ind_symbols, box_levels, stacked_list, script_level_list, extend_list, fig, ax
    else:
        return ind_symbols, box_levels, stacked_list, script_level_list, extend_list

   