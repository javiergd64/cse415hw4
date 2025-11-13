'''
javiergd_KInARow.py
Authors: Guapilla-Diaz, Javier; Zhang, Ivonne
  Example:  
    Authors: Guapilla-Diaz, Javier; Zhang, Ivonne

An agent for playing "K-in-a-Row with Forbidden Squares" and related games.
CSE 415, University of Washington
'''
from mpmath.functions.zetazeros import count_to

from agent_base import KAgent
from game_types import State, Game_Type
# import game_types # idk why it didnt work, but changed it to random player import
# ^NOTE, it didnt work b/c when I copied make_move() from random_player,
# I forgot to change its argument of:
# news = game_types.State to just State
from random import randint # for the random choosing (before mM or ab)
#GAME_TYPE = None not needed bc of GLOBAL in prepare

AUTHORS = 'Javier Guapilla-Diaz; Ivonne Zhang'
UWNETIDS = ['javiergd', 'yimenz5'] # The first UWNetID here should
# match the one in the file name, e.g., janiesmith99_KInARow.py.

import time

# You'll probably need this to avoid losing a
# game due to exceeding a time limit.

# Create your own type of agent by subclassing KAgent:

class OurAgent(KAgent):  # Keep the class name "OurAgent" so a game master
    # knows how to instantiate your agent class.

    def __init__(self, twin=False):
        self.twin=twin
        self.nickname = 'Beep-Bop'
        if twin: self.nickname += '2'
        self.long_name = 'Beep-Boop-Bop'
        if twin: self.long_name += ' II'
        self.persona = 'bland'
        self.voice_info = {'Chrome': 10, 'Firefox': 2, 'other': 0}
        self.playing = "don't know yet" # e.g., "X" or "O".
        self.alpha_beta_cutoffs_this_turn = -1
        self.num_static_evals_this_turn = -1
        self.zobrist_table_num_entries_this_turn = -1
        self.zobrist_table_num_hits_this_turn = -1
        self.current_game_type = None
        self.playing_mode = KAgent.DEMO

    def introduce(self):
        intro = '\nHey! My name is Beep-Boop-Bop\n'+\
            'Javier (javiergd) & Ivonne (yimenz5) made me\n'+\
            'Hopefully we have a good time playing!\n'
        if self.twin: intro += "By the way, I'm the TWIN.\n"
        return intro

    # Receive and acknowledge information about the game from
    # the game master:
    def prepare(
        self,
        game_type,
        what_side_to_play,
        opponent_nickname,
        expected_time_per_move = 0.1, # Time limits can be
                                      # changed mid-game by the game master.

        utterances_matter=True):      # If False, just return 'OK' for each utterance,
                                      # or something simple and quick to compute
                                      # and do not import any LLM or special APIs.
                                      # During the tournament, this will be False..
        if utterances_matter:
            pass
            # Optionally, import your LLM API here.
            # Then you can use it to help create utterances.
           
        # Write code to save the relevant information in variables
        # local to this instance of the agent.
        # Game-type info can be in global variables.

        ## -- i got this from random player to set up variables and stuff
        # I realized it was missing when the test wasnt working on the game master :,)
        # from randomPlayer
        self.who_i_play = what_side_to_play
        self.opponent_nickname = opponent_nickname
        self.time_limit = expected_time_per_move
        global GAME_TYPE
        GAME_TYPE = game_type
        print("Oh, I love playing randomly at ", game_type.long_name)
        self.my_past_utterances = []
        self.opponent_past_utterances = []
        self.repeat_count = 0
        self.utt_count = 0
        if self.twin: self.utt_count = 5  # Offset the twin's utterances.

       #print("Change this to return 'OK' when ready to test the method.")
        return "OK" #"Not-OK"

    # working on this first -j
    # note for ivonne: I used the random player for layout and initial
    # "test" of randomness
    # The core of your agent's ability should be implemented here:             
    def make_move(self, current_state, current_remark, time_limit=1000,
                  use_alpha_beta=False, #True, uncomment for later
                  use_zobrist_hashing=False, max_ply=3,
                  special_static_eval_fn=None):

        # a list of places we can go, and the move associated
        possible_S, possible_M = successors_and_moves(current_state)
        # S is state and M is move
        
        # successors empty = can't go anywhere, so cant do anything
        if (len(possible_S) == 0):
            utter = "Welp, I'm cornered"
            return [[None, current_state], utter]
        
        # this is to test w/o minimax or ab
        
        new_S, new_M = chooseMove((possible_S, possible_M))
        
        utter = "TEST UTTERANCE" # for now cuz we will use llm?

        return [[new_M, new_S], utter]

        # --- OLD ---
        # Kept it here in case we need to look over it or revert something

        # print("make_move has been called")
        #
        # print("code to compute a good move should go here.")
        # # Here's a placeholder:
        # a_default_move = (0, 0) # This might be legal ONCE in a game,
        # # if the square is not forbidden or already occupied.
        #
        # new_state = current_state # This is not allowed, and even if
        # # it were allowed, the newState should be a deep COPY of the old.
        #
        # new_remark = "I need to think of something appropriate.\n" +\
        # "Well, I guess I can say that this move is probably illegal."
        #
        # print("Returning from make_move")
        # return [[a_default_move, new_state], new_remark]


    # next chunk
    # -------------------------------------------------------------------- #
    # The main adversarial search function:
    def minimax(self,
            state,
            depth_remaining,
            pruning=False,
            alpha=None,
            beta=None):
        print("Calling minimax. We need to implement its body.")

        default_score = 0 # Value of the passed-in state. Needs to be computed.
    
        return [default_score, "my own optional stuff", "more of my stuff"]
        # Only the score is required here but other stuff can be returned
        # in the list, after the score, in case you want to pass info
        # back from recursive calls that might be used in your utterances,
        # etc. 
 
    def static_eval(self, state, game_type=None):
        print('calling static_eval. Its value needs to be computed!')
        # Values should be higher when the states are better for X,
        # lower when better for O.

        board = state.board
        # check the current game type
        if game_type is None:
            k = self.current_game_type.k
        else:
            k = game_type.k

        n = len(board) # row
        m = len(board[0]) # column
        # I changed this to how its seen in the winTesterForK

        score = 0 # initialize score

        # helper for rows and columns checks
        def helper_check(seq):
            nonlocal score
            count = 0
            current = None
            for token in seq + ['$']: # dummie node to end the loop
                if token == current and token in ['X', 'O']:
                    count += 1
                else:
                    if current == 'X':
                        score += 10 ** count
                    elif current == 'O':
                        score -= 10 ** count
                    current = token
                    count = 1 if token in ['X', 'O'] else 0

        # traverse through the rows
        for i in range(n):
            helper_check(board[i])
        # traverse through the columns
        for j in range(m):
            helper_check([board[i, j] for i in range(n)])
        # traverse through the diagonals
        # down-right traverse
        for row in range(n):
            diag = []
            i,j = row,0
            while i < n and j < m:
                diag.append(board[i, j])
                i,j = i+1,j+1
            helper_check(diag)
        for col in range(m):
            diag = []
            i,j = 0,col
            while i < n and j < m:
                diag.append(board[i, j])
                i,j = i+1,j+1
            helper_check(diag)
        # down-left traverse
        for row in range(n):
            diag = []
            i,j = row,m-1
            while i < n and j >= 0:
                diag.append(board[i, j])
                i,j = i+1,j-1
            helper_check(diag)
        for col in range(m-2, -1, -1):
            diag = []
            i,j = 0,col
            while i < n and j >= 0:
                diag.append(board[i, j])
                i,j = i+1,j-1
            helper_check(diag)

        # return 0 if no winner
        return score
 
# OPTIONAL THINGS TO KEEP TRACK OF:

# THESE I GOT FROM RANDOM PLAYER (they seemed useful):
def other(p):
    if p=='X': return 'O'
    return 'X'

# Randomly choose a move.
def chooseMove(statesAndMoves):
    states, moves = statesAndMoves
    if states==[]: return None
    random_index = randint(0, len(states)-1)
    my_choice = [states[random_index], moves[random_index]]
    return my_choice

# The following is a Python "generator" function that creates an
# iterator to provide one move and new state at a time.
# It could be used in a smarter agent to only generate SOME of
# of the possible moves, especially if an alpha cutoff or beta
# cutoff determines that no more moves from this state are needed.
def move_gen(state):
    b = state.board
    p = state.whose_move
    o = other(p)
    mCols = len(b[0])
    nRows = len(b)

    for i in range(nRows):
        for j in range(mCols):
            if b[i][j] != ' ': continue
            news = do_move(state, i, j, o)
            yield [(i, j), news]

# This uses the generator to get all the successors.
def successors_and_moves(state):
    moves = []
    new_states = []
    for item in move_gen(state):
        moves.append(item[0])
        new_states.append(item[1])
    return [new_states, moves]

# Performa a move to get a new state.
def do_move(state, i, j, o):
            #news = game_types.State(old=state)
            news = State(old=state)
            news.board[i][j] = state.whose_move
            news.whose_move = o
            return news

#  WHO_MY_OPPONENT_PLAYS = other(WHO_I_PLAY)
#  MY_PAST_UTTERANCES = []
#  OPPONENT_PAST_UTTERANCES = []
#  UTTERANCE_COUNT = 0
#  REPEAT_COUNT = 0 or a table of these if you are reusing different utterances

