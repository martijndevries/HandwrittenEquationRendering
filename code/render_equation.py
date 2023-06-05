import numpy as np



def turn_into_LOLs(symbol_list, levels, stack, script_levels):
    """
    Turn the list of symbols, and script levels into list of lists
    Where each list only contains those entries that are at the same level
    """
    
    l_a = np.array(levels)
    level_lens = [0] +  [len(l_a[l_a==x]) for x in rangae(max(l_a)+1)]
    level_cum_lens = np.cumsum(level_lens)
    
    symbol_lol = [symbol_list[x_1:x_2] for x_1,x_2 in zip(level_cum_lens[:-1], level_cum_lens[1:])]
    script_lol = [script_levels[x_1:x_2] for x_1,x_2 in zip(level_cum_lens[:-1], level_cum_lens[1:])]
    
    #not lists of lists, just single entries for the stack/level of that list of symbols
    level_lol = [levels[x_1] for x_1,x_2 in zip(level_cum_lens[:-1], level_cum_lens[1:])]
    stack_lol = [stack[x_1] for x_1,x_2 in zip(level_cum_lens[:-1], level_cum_lens[1:])]
    
    return symbol_lol, level_lol, stack_lol, script_lol

def combine_signs(symbol_list, script_list):
    """
    Given a list of symbols, check whether there are subsequent symbols that should be interpreted as a single symbol
    SO FAR: checking for sin, cos, tan, log
    """
    
    #sin
    start_symbs = ['s', 'S', '5', '\\lt']

    for start_symb in start_symbs:
        if start_symb in symbol_list[:-2]:
            idx = symbol_list.index(start_symb)
            idx2 = symbol_list.index(start_symb)
            print(start_symb, idx, idx2, symbol_list[idx+1], symbol_list[idx+2])
            if (symbol_list[idx+1].lower() == 'i' or symbol_list[idx+1] == '1') and symbol_list[idx+2].lower()  == ('n'):
                script_list.pop(idx+2)
                script_list.pop(idx+1)
                symbol_list.pop(idx+2)
                symbol_list.pop(idx+1)
                symbol_list[idx] = '\\sin'

    #cos 
    start_symbs = ['c', 'C', '(']
    for start_symb in start_symbs:
        if start_symb in symbol_list[:-2]:
            idx = symbol_list.index(start_symb)
            if (symbol_list[idx+1].lower() == 'o' or symbol_list[idx+1] == '0')  and symbol_list[idx+2].lower()  == ('s'):
                script_list.pop(idx+2)
                script_list.pop(idx+1)
                symbol_list.pop(idx+2)
                symbol_list.pop(idx+1)
                symbol_list[idx] = '\\cos'

    #tan
    start_symbs = ['t', 'T', '+']
    for start_symb in start_symbs:
        if start_symb in symbol_list[:-2]:
            idx = symbol_list.index(start_symb)
            if (symbol_list[idx+1].lower() == 'a')  and symbol_list[idx+2].lower()  == ('n'):
                script_list.pop(idx+2)
                script_list.pop(idx+1)
                symbol_list.pop(idx+2)
                symbol_list.pop(idx+1)
                symbol_list[idx] = ' \\tan' 
                
    #log 
    start_symbs = ['l', 'L']
    for start_symb in start_symbs:
        if start_symb in symbol_list[:-2]:
            if (symbol_list[idx+1].lower() == 'o') or (symbol_list[idx+1].lower() == '0') or s and symbol_list[idx+2].lower()  == 'g':
                script_list.pop(idx+2)
                script_list.pop(idx+1)
                symbol_list.pop(idx+2)
                symbol_list.pop(idx+1)
                symbol_list[idx] = ' \\log' 
    
    return symbol_list, script_list

def render_equation(symbol_list, levels, stack, script_levels):
    
    symbol_lol, level_lol, stack_lol, script_lol = turn_into_stack_LOLs(symbol_list, levels, stack, script_levels)
    
    eq_string = '$'
    no_ss = ['+', '-'] #if at ba
    for sl, symbol_l in enumerate(symbol_lol):
        symbol_l = [symb if 'capital_' not in symb else symb[8:] for symb in symbol_l]
        symbol_pp, script_lol[sl] = check_comb_signs(symbol_l, script_lol[sl])
        slc = 0 #script level correction, for when a sub/superscript doesn't logically make sense even when the preprocessign thinks it is
        no_close = False
        for s, symbol in enumerate(symbol_pp):
            #if we are in the middle level of a stack, we should have already checked what the symbol is. 
            #add curly brackets where necessary and continue
            if stack_lol[sl] == 2:
                eq_string += '}'
                if symbol == '\\sum':
                    eq_string += '_{'
                elif symbol == '-':
                    eq_string += '{'
                continue
                
            #check script levels
            if script_lol[sl][s] != 0:
                if script_lol[sl][s]> script_lol[sl][s-1] and symbol_pp[s-1] not in no_ss :
                    eq_string += '^{'
                    no_close = False
                elif script_lol[sl][s] < script_lol[sl][s-1] and symbol_pp[s-1] not in no_ss:
                    eq_string += '_{'
                    no_close = False
                else:
                    eq_string += ' '
                if (script_lol[sl][s]> script_lol[sl][s-1] or script_lol[sl][s] < script_lol[sl][s-1]) and symbol_pp[s-1] in no_ss:
                    no_close = True
            if script_lol[sl][s] == 0 and s > 0:
                if script_lol[sl][s] != script_lol[sl][s-1]:# and no_close == False:
                    eq_string += '}'
                eq_string += ' '
            
            #if we are not in a stack, simply add the symbol
            if stack_lol[sl] == 0:
                eq_string += symbol
            
            #if we are in a stack, first check what the middle symbol is
            if stack_lol[sl] == 1 and s == 0:
                #is there a middle symbol?
                if stack_lol[sl+1] == 2:
                    if symbol_lol[sl+1][0] == '\\sum':
                        eq_string += ' \\sum^{'
                    if symbol_lol[sl+1][0] == '-':
                        eq_string += ' \\frac{'
                #if not, is this symbol a lim sign?
                elif symbol == '\\lim':
                    eq_string += '\\lim_{'
                    #also break, we will ignore any other chars on this level
                    break
        
            if stack_lol[sl] == 1 or stack_lol[sl] == 3:
                eq_string += ' ' + symbol
                             
            #closing bracket when we leave the stack
            if stack_lol[sl] == 3 and s == len(symbol_pp) -1:
                eq_string += '}'
                
            #closing bracket when leave a script lvl
            if script_lol[sl][s] != 0 and s == len(symbol_pp) -1 and no_close == False:
                eq_string += '}' * np.abs(script_lol[sl][s])

    
    eq_string += '$'
    
    display(Latex(f'' + eqstr))

    return eq_string
                
            