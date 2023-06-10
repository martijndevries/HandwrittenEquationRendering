import numpy as np



def turn_into_stack_LOLs(symbol_list, levels, stack, script_levels, extend_list):
    """
    Turn the list of symbols, and script levels into list of lists
    Where each list only contains those entries that are at the same level
    """
    
    l_a = np.array(levels)
    level_lens = [0] +  [len(l_a[l_a==x]) for x in range(max(l_a)+1)]
    level_cum_lens = np.cumsum(level_lens)
    
    symbol_lol = [symbol_list[x_1:x_2] for x_1,x_2 in zip(level_cum_lens[:-1], level_cum_lens[1:])]
    script_lol = [script_levels[x_1:x_2] for x_1,x_2 in zip(level_cum_lens[:-1], level_cum_lens[1:])]
    extend_lol =  [extend_list[x_1:x_2] for x_1,x_2 in zip(level_cum_lens[:-1], level_cum_lens[1:])]
    
    #not lists of lists, just single entries for the stack/level of that list of symbols
    level_lol = [levels[x_1] for x_1,x_2 in zip(level_cum_lens[:-1], level_cum_lens[1:])]
    stack_lol = [stack[x_1] for x_1,x_2 in zip(level_cum_lens[:-1], level_cum_lens[1:])]
    
    return symbol_lol, level_lol, stack_lol, script_lol, extend_lol


def pop_list_idx(script_list, symbol_list, extend_list, idx, label):
    """
    Helper function to remove lest entries when symbols have been combined
    """
    script_list.pop(idx+2)
    script_list.pop(idx+1)
    symbol_list.pop(idx+2)
    symbol_list.pop(idx+1)
    extend_list.pop(idx+2)
    extend_list.pop(idx+1)
    symbol_list[idx] = label
    
    return symbol_list, script_list, extend_list


def search_and_replace_combo(symbol_list, script_list, extend_list, symb_combos, label):
    """
    Helper function to check combination of 3 symbols and replace them with one if necessary
    """
    start_symbs = symb_combos[0]
    mid_symbs = symb_combos[1]
    end_symbs = symb_combos[2]
    
    #search over all possible starting combos
    for start_symb in start_symbs:
        sidx = 0
        ct = symbol_list[:-2].count(start_symb)
        #loop over all occurences of this starting symbol
        for c in range(ct):
            idx = symbol_list[sidx:-2].index(start_symb)
            sidx += idx 
            #if a matching middle and end symbol are find, edit the lists accordingly
            if symbol_list[sidx+1].lower() in mid_symbs and symbol_list[sidx+2].lower() in end_symbs:
                symbol_list, script_list, extend_list = pop_list_idx(script_list, symbol_list, extend_list, sidx, label)
            sidx += 1
    return symbol_list, script_list, extend_list

def check_comb_signs(symbol_list, script_list, extend_list, level):
    """
    Given a list of symbols, check whether there are subsequent symbols that should be interpreted as a single symbol
    SO FAR: checking for sin, cos, tan, log, lim, x, and k
    """
    
    #only bother if there are 3 or more symbols in the list
    if len(symbol_list) < 3: 
        return symbol_list, script_list, extend_list
    
    #sin
    symb_combos = [['s', 'S', '5', '\\lt'], ['i', '1', 't', '!'], ['n']]
    symbol_list, script_list, extend_list = search_and_replace_combo(symbol_list, script_list, extend_list, symb_combos, '\\sin')
    
    #cos 
    symb_combos = [['c', 'C', '('], ['o', '0'], ['s']]
    symbol_list, script_list, extend_list = search_and_replace_combo(symbol_list, script_list, extend_list, symb_combos, '\\cos')

    #tan
    symb_combos = [['t', 'T', '+', '1'], ['a'], ['n']]
    symbol_list, script_list, extend_list = search_and_replace_combo(symbol_list, script_list, extend_list, symb_combos, '\\tan')
                
    #log 
    symb_combos = [['l', 'L', '1'], ['o', '0', 'a'], ['g', ',', ')', '\\gamma', 'y']]
    symbol_list, script_list, extend_list = search_and_replace_combo(symbol_list, script_list, extend_list, symb_combos, '\\log')
                
    #lim
    symb_combos = [['l', 'L'], ['i', '1'], ['m']]
    symbol_list, script_list, extend_list = search_and_replace_combo(symbol_list, script_list, extend_list, symb_combos, '\\lim')

    #x
    idxs = [0, len(symbol_list)-2] #will only try to do the correction at the beginning and end of equatons. Too risky otheriwse
    start_symbs = [')', '7', '2', ']']
    end_symbs = ['(', 'c']

    for idx in idxs:
        try: 
            symbol_list[idx+1]
        except:
            break
        if (symbol_list[idx] in start_symbs) and (symbol_list[idx+1] in end_symbs):
            symbol_list.pop(idx+1)
            script_list.pop(idx+1)
            extend_list.pop(idx+1)
            symbol_list[idx] = 'x'
        
    #if there are two brackets opposite )(, and the next symbol is a superscript or subscript, assume it's an x too
    start_symbs = [')', ']']
    end_symbs = ['(', 'c']
    for start_symb in start_symbs:
        sidx = 0
        ct = symbol_list[:-2].count(start_symb)
        #loop over all occurences of this starting symbol
        for c in range(ct):
            idx = symbol_list[sidx:-2].index(start_symb)
            sidx += idx 
            if symbol_list[sidx+1].lower() in end_symbs and script_list[sidx+2] != script_list[sidx]:
                script_list.pop(sidx+1)
                symbol_list.pop(sidx+1)
                extend_list.pop(sidx+1)
                symbol_list[sidx] = 'x'
            sidx += 1

    #k
    start_symbs = ['1']
    for start_symb in start_symbs:
        sidx = 0
        ct = symbol_list[:-2].count(start_symb)
        for c in range(ct):
            idx = symbol_list[sidx:-1].index(start_symb)
            sidx += idx
            if (symbol_list[sidx+1].lower() == 'c') or (symbol_list[sidx+1].lower() == '('):
                symbol_list.pop(sidx+1)
                script_list.pop(sidx+1)
                extend_list.pop(sidx+1)
                symbol_list[sidx] = 'k'
            sidx +=1
    return symbol_list, script_list, extend_list

def render_equation(symbol_list, levels, stack, script_levels, extend_list):
    """
    Main equation rendering function
    Input: 
        1) list of predicted symbol labels
        2) list of the level of each symbol 
        3) list of the stack of each symbol
        4) list of the script level of each symbol
        4) list of how far each symbol should extend (currently only relevant for square root signs, we need to know how many symbols they cover)
    Returns:
        A string that contains the equation in latex math mode format
    """
    
    #turn the inputs into a list of lists. We will build the equation per level, so iterating over levels is useful
    symbol_lol, level_lol, stack_lol, script_lol, extend_lol = turn_into_stack_LOLs(symbol_list, levels, stack, script_levels, extend_list)
    
    no_ss_p = ['+', '-', '='] #symbols that logically wouldnt be followed by a super/subscript
    no_ss_c = ['=', '\\lt', '\\gt'] #symbols that wouldn't logically be the first symbol in a super/subscript
    
    eq_string = '$'
    
    for sl, symbol_l in enumerate(symbol_lol):
        
        if len(symbol_l) == 0: continue
        
        #if we are in the middle level of a stack, we should have already checked what the symbol is. 
        #add curly brackets where necessary and continue        
        if stack_lol[sl] == 2:
            if symbol_l[0] == '\\sum':
                eq_string += '}_{'
            elif symbol_l[0] == '-': #division
                eq_string += '}{'
            continue
        
        #get rid of the capital_' part of labels
        symbol_l = [symb if 'capital_' not in symb else symb[8:] for symb in symbol_l]
        
        #check whether there are adjacent characters that should be combined into sin,tan,cos etc
        symbol_pp, script_lol[sl], extend_lol[sl] = check_comb_signs(symbol_l, script_lol[sl], extend_lol[sl], level_lol[sl])
        
        #now loop over each symbol in this level
        for s, symbol in enumerate(symbol_pp):
            
            no_close = False
            
            #check script levels
            #if the previous symbol was in no_ss_p OR this symbol is in no_ss_c, and this symbol has a different script level that is not zero
            #apply correction to the script levels
            if s != 0 and (symbol_pp[s-1] in no_ss_p or symbol in no_ss_c) and script_lol[sl][s-1] != script_lol[sl][s] and script_lol[sl][s] != 0:
                no_close = True 
                if script_lol[sl][s] >  script_lol[sl][s-1]:
                    script_lol[sl] = [scrip -1 for scrip in script_lol[sl]]
                elif script_lol[sl][s] <  script_lol[sl][s-1]:
                    script_lol[sl] = [scrip +1 for scrip in script_lol[sl]]
                    
            #assume that whatever follows a log sign should be subscript, even if the preprocessing pipeline didn't pick up on that
            #edit script levels accordingly
            if s > 0:
                if symbol_pp[s-1] == '\\log' and script_lol[sl][s] != script_lol[sl][s-1]-1:
                    script_lol[sl] = [scrip - 1 if scr >= s else scrip for scr,scrip in enumerate(script_lol[sl])]
                    
            #add an opening bracket if we move further from zero
            if script_lol[sl][s] != 0:
                if abs(script_lol[sl][s]) > abs(script_lol[sl][s-1]) and script_lol[sl][s] > 0:
                    eq_string += '^{'
                    no_close = False
                elif abs(script_lol[sl][s]) > abs(script_lol[sl][s-1]) and script_lol[sl][s] < 0:
                    eq_string += '_{'
                else:
                    eq_string += ' '
            
            #add closing bracket if we move closer to zero, except when we have just changed the script level
            if s > 0:
                if abs(script_lol[sl][s]) < abs(script_lol[sl][s-1]) and no_close == False:
                    eq_string += '}'

            #if we entered stack, first check what the middle symbol is
            if stack_lol[sl] == 1 and s == 0:
                #is there a middle symbol?
                if stack_lol[sl+1] == 2:
                    if symbol_lol[sl+1][0] == '\\sum':
                        eq_string += ' \\sum^{'
                    if symbol_lol[sl+1][0] == '-':
                        eq_string += ' \\frac{'
                #if not, is this symbol a lim sign?
                if symbol == '\\lim':
                    eq_string += '\\lim_{'
                    #also break, we will ignore any other chars on this level because a lim should be by itself
                    break
            
            
            if symbol == 'adiv':
                symbol = '\/'
                
            #add symbol
            eq_string += ' ' + symbol 
                   
            #if a sum ended up in this level (it should usually be in stack lvl 2..) add a '_{' anyway
            if symbol == '\\sum':
                eq_string += '_{'
                
            #if the symbol is a square root, add opening bracket 
            if symbol == '\\sqrt':
                eq_string += '{'
                
            #if we are x steps further than a square root symbol that extends x symbols, add a closing bracket for that root
            for q in range(1,s+1):
                if symbol_pp[s-q] == '\\sqrt' and extend_lol[sl][s-q] == q and q!=1:
                    eq_string += '}'
                    
            #closing bracket when we leave the stack
            if stack_lol[sl] == 3 and s == len(symbol_pp) -1:
                eq_string += '}'
                
            #closing bracket when leave a script lvl
            if script_lol[sl][s] != 0 and s == len(symbol_pp) -1:
                eq_string += '}' * np.abs(script_lol[sl][s])
                
    #count how many closing vs opening curly brackets there are. add extra closing ones if necessary
    #(will help to render the equation, even if it's wrong)
    open_c = eq_string.count('{')
    close_c = eq_string.count('}')
    if open_c - close_c > 0:
        eq_string += '}' * (open_c - close_c)
                              
    eq_string += '$'
    
    #lesser than and greater than signs are actually just rendered with < and >, so replace
    eq_string = eq_string.replace('\\lt', '<')
    eq_string = eq_string.replace('\\gt', '>')
    

    return eq_string