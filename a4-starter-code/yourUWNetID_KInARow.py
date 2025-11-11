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

AUTHORS = 'Javier Guapilla-Diaz; Ivonne Zhang'
UWNETIDS = ['javiergd', 'yimenz5'] # The first UWNetID here should
# match the one in the file name, e.g., janiesmith99_KInARow.py.

import time # You'll probably need this to avoid losing a
 # game due to exceeding a time limit.

# Create your own type of agent by subclassing KAgent:

class OurAgent(KAgent):  # Keep the class name "OurAgent" so a game master
    # knows how to instantiate your agent class.

    def __init__(self, twin=False):
        self.twin=twin
        self.nickname = 'Nic'
        if twin: self.nickname += '2'
        self.long_name = 'Templatus Skeletus'
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
        intro = '\nMy name is Templatus Skeletus.\n'+\
            '"An instructor" made me.\n'+\
            'Somebody please turn me into a real game-playing agent!\n'
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
       print("Change this to return 'OK' when ready to test the method.")
       return "Not-OK"
   
    # The core of your agent's ability should be implemented here:             
    def make_move(self, current_state, current_remark, time_limit=1000,
                  use_alpha_beta=True,
                  use_zobrist_hashing=False, max_ply=3,
                  special_static_eval_fn=None):
        print("make_move has been called")

        print("code to compute a good move should go here.")
        # Here's a placeholder:
        a_default_move = (0, 0) # This might be legal ONCE in a game,
        # if the square is not forbidden or already occupied.
    
        new_state = current_state # This is not allowed, and even if
        # it were allowed, the newState should be a deep COPY of the old.
    
        new_remark = "I need to think of something appropriate.\n" +\
        "Well, I guess I can say that this move is probably illegal."

        print("Returning from make_move")
        return [[a_default_move, new_state], new_remark]

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
        n = board.n # row
        m = board.m # column
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

#  WHO_MY_OPPONENT_PLAYS = other(WHO_I_PLAY)
#  MY_PAST_UTTERANCES = []
#  OPPONENT_PAST_UTTERANCES = []
#  UTTERANCE_COUNT = 0
#  REPEAT_COUNT = 0 or a table of these if you are reusing different utterances

