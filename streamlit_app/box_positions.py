import numpy as np

class BoxPositions:
    """
    Class that can compare the relative positions of two boxes
    #################
    
    Constructor function:
        coords1, coords2 - box coordinates in the form [x1, y1, x2, y2]
        
    Class attributes:
        x1s, y1s, x2s, y2s: list of respective coordinates for box 1 and box 2
        xlens, ylens: list of respective x/y lengths for box 1 and box 2
        xcs, ycs: list of x/y centers for box 1 and box 1
        ydist: the distance between the boxes, calculated as yc_box2 - yc_box1
        
    Class methods:
        isSame(self):
            True/False check whether box1 = box2
        isInside(self):
            True/False check whether box2 is completely inside box1
        isBigger(self, len_factor, axis='x'):
            True/False check whether the bigger of the two boxes is bigger than the smaller box by some factor, along the specified axis (x or y)
        isPartInside(self)
            True/False check whether the two boxes overlap at all
        calc_Overlap(self, axis='x', relative_to='both')
            calculates how much the boxes overlap, on the specified axis. relative to can be to both boxes, or to the smaller box (-> How much does the smaller box stick out)
        calc_box_extends(self)
            calculates how much the second box extends up/downwards from the first box. output in terms of fraction of the ylength
        merge_boxes(self)
            returns a new box that is the merged version of the two boxes
        boxExtends(self)
            True/False check whether box2 extends beyond box 1 along the specified directon (top/bottom/left/right)
        
    """
    def __init__(self, coords1, coords2):
        self.coords1 = coords1
        self.coords2 = coords2
        
        #calc properties that I might need
        self.x1s = [coords1[0], coords2[0]]
        self.y1s = [coords1[1], coords2[1]]
        self.x2s = [coords1[2], coords2[2]]
        self.y2s = [coords1[3], coords2[3]]
        
        #lengths
        self.xlen_b1 = self.x2s[0] - self.x1s[0]
        self.ylen_b1 = self.y2s[0] - self.y1s[0]
        
        self.xlen_b2 = self.x2s[1] - self.x1s[1]
        self.ylen_b2 = self.y2s[1] - self.y1s[1]
        
        self.xlens  = [self.xlen_b1, self.xlen_b2]
        self.ylens = [self.ylen_b1, self.ylen_b2]
        
        #centers
        self.xcs = [self.x2s[0]*0.5 + self.x1s[0]*0.5, self.x2s[1]*0.5 + self.x1s[1]*0.5]
        self.ycs = [self.y2s[0]*0.5 + self.y1s[0]*0.5, self.y2s[1]*0.5 + self.y1s[1]*0.5]
           
        #ydist
        self.ydist = self.ycs[1] - self.ycs[0]
    
    def isSame(self):
        """
        Check whether the boxes are the same
        """
        if self.coords1 == self.coords2:
            return True
        else:
            return False
        
    def isInside(self, invert=False):
        """
        Check whether the second box is completely inside the first box. If invert=True, check whether b1 is inside b2
        """
        if invert == False:
            if (self.y1s[1] >= self.y1s[0]) and (self.y2s[1] <= self.y2s[0]) and (self.x1s[1] >= self.x1s[0]) and (self.x2s[0] > self.x2s[1]):
                return True
            else:
                return False
        if invert == True:
            if (self.y1s[1] <= self.y1s[0]) and (self.y2s[1] >= self.y2s[0]) and (self.x1s[1] <= self.x1s[0]) and (self.x2s[0] < self.x2s[1]):
                return True
            else:
                return False
            
    def isPartInside(self):
        """
        Check whether the second box is partly inside the first box. The second box does NOT overlap with the first if
        1) the box is above or below the first box
        2) the box is to the right or left of the second box
        If neither of these conditions is satisfied, the boxes must at least partially overlap
        """
        if self.y2s[1] < self.y1s[0] or self.y1s[1] > self.y2s[0]:
            return False
        elif self.x2s[1] < self.x1s[0] or self.x1s[1] > self.x2s[0]:
            return False
        else:
            return True
        
        
    def isBigger(self, len_factor, axis='x'):
        """
        Check whether the bigger box is bigger than the smaller box by some specified factor on the specified axis
        """
        if axis == 'x':
            clen = self.xlens
        if axis == 'y':
            clen = self.ylens
            
        if max(clen) < min(clen) * len_factor:
            return False
        else:
            return True

    def calc_Overlap(self, axis='x', relative_to='both'):
        """
        Check how much overlap the box has along the specified axis
        """
        ext_left= self.boxExtends(direction='left')
        ext_right= self.boxExtends(direction='right')
        ext_top= self.boxExtends(direction='top')
        ext_bottom= self.boxExtends(direction='bottom')

        
        if self.y2s[1] == self.y2s[0]:
            ext_bottom = ext_top
        if self.y1s[1] == self.y1s[0]:
            ext_top = ext_bottom
                
        if axis == 'x':
            #if one of the boxes is longer than the other on both sides, the overlap should be 1
            if (ext_left and ext_right) or (not ext_left and not ext_right):
                overlap = 1
            else:
                abs_overlap = min(self.x2s) - max(self.x1s)
                if relative_to == 'both': #the overlap wrt to the full distance spanned by both boxes
                    totl = max(self.x2s) - min(self.x1s)
                elif relative_to == 'smaller': #how much of the smaller box overlaps with the bigger box
                    totl = min(self.xlens)
                overlap = abs_overlap/totl
                
        if axis == 'y':
            if (ext_top and ext_bottom) or (not ext_top and not ext_bottom):
                overlap = 1
            else:
                abs_overlap = min(self.y2s) - max(self.y1s)
                if relative_to == 'both': #the overlap wrt to the full distance spanned by both boxes
                    totl = max(self.y2s) - min(self.y1s)
                elif relative_to == 'smaller': #how much of the smaller box overlaps with the bigger box
                    totl = min(self.ylens)
                overlap = abs_overlap/totl
        return overlap
    
    def calc_box_extends(self):
        """
        Calculate how much box 2 extends upwards or downwards from box 1, whichever direction is the largest
        """
        extend_down = (self.y2s[1] - self.y2s[0])/self.ylen_b2
        extend_up = (self.y1s[0] - self.y1s[1])/self.ylen_b2
      
        return max([extend_down, extend_up])
    
    def calc_box_extends_center(self):
        """
        Calculate how much box 2 extends upwards vs downwards wrt to the center of box 1
        """
        extend_down = (self.y2s[1] - self.ycs[0])/self.ylen_b2
        extend_up = (self.ycs[0] - self.y1s[1])/self.ylen_b2
        
      
        return abs(extend_down - extend_up)
    
    def merge_boxes(self):
        """
        Take two boxes with coords [x1, y1, x2, y2]
        and merge them into a box spanning both
        """
        new_x1 = min(self.x1s)
        new_y1 = min(self.y1s)
        new_x2 = max(self.x2s)
        new_y2 = max(self.y2s)
    
        return new_x1, new_y1, new_x2, new_y2
            
    def boxExtends(self, direction='left'):
        """
        Check whether box2 extends a certain direction out from of box1
        """
        #print(self.y1s, se
        if direction == 'left':
            if self.x1s[1] < self.x1s[0]:
                return True
            else:
                return False
        if direction == 'right':
            if self.x2s[1] > self.x2s[0]:
                return True
            else:
                return False
        if direction == 'top': #y-coords are inverted
            if self.y1s[1] < self.y1s[0]:
                return True
            else:
                return False
        if direction == 'bottom':
            if self.y2s[1] > self.y2s[0]:
                return True
            else:
                return False
        
        
