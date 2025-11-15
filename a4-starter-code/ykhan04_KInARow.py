'''
<yourUWNetID>_KInARow.py
Authors: <Han, Yaokun>

An agent for playing "K-in-a-Row with Forbidden Squares" and related games.
CSE 415, University of Washington

THIS IS A TEMPLATE WITH STUBS FOR THE REQUIRED FUNCTIONS.
YOU CAN ADD WHATEVER ADDITIONAL FUNCTIONS YOU NEED IN ORDER
TO PROVIDE A GOOD STRUCTURE FOR YOUR IMPLEMENTATION.

'''

from agent_base import KAgent
from game_types import State, Game_Type

AUTHORS = 'Yaokun Han' 
UWNETIDS = ['ykhan04'] # The first UWNetID here should
# match the one in the file name, e.g., janiesmith99_KInARow.py.

import time # You'll probably need this to avoid losing a
 # game due to exceeding a time limit.

# Create your own type of agent by subclassing KAgent:

class OurAgent(KAgent):  # Keep the class name "OurAgent" so a game master
    # knows how to instantiate your agent class.

    def __init__(self, twin=False):
        self.twin=twin
        self.nickname = 'SeeUInCU'
        if twin: self.nickname += '2'
        self.long_name = 'AgentCUHK'
        if twin: self.long_name += ' II'
        self.persona = 'instructive'
        self.voice_info = {'Chrome': 10, 'Firefox': 2, 'other': 0}
        self.playing = "don't know yet" # e.g., "X" or "O".
        self.alpha_beta_cutoffs_this_turn = -1
        self.num_static_evals_this_turn = -1
        self.zobrist_table_num_entries_this_turn = -1
        self.zobrist_table_num_hits_this_turn = -1
        self.current_game_type = None
        self.playing_mode = KAgent.DEMO

        # The rest are used for utterance and bonus parts:
        self.deepseek_client = None
        self.game_history = []
        self.move_explanations = []
        self.opponent_remarks = []
        self.utterance_count = 0
        self.last_move_stats = {}

    def introduce(self):
        intro = '\nGreetings! I am AgentCUHK, or you can call me SeeUInCU!\n'+\
            'I\'m created by Yaokun Han (ykhan04) at the University of Washington.\n'+\
            'I\'ll be back to HK in three weeks hahahahaha!\n'
        if self.twin: intro += "Note: I am the enhanced twin version with deeper analysis capabilities.\n"
        return intro

    # Receive and acknowledge information about the game from
    # the game master:
    def prepare(
        self,
        game_type,
        what_side_to_play,
        opponent_nickname,
        expected_time_per_move = 0.1,
        utterances_matter=True):
        
        self.current_game_type = game_type
        self.playing = what_side_to_play
        self.opponent_nickname = opponent_nickname
        self.expected_time_per_move = expected_time_per_move
        self.utterances_matter = utterances_matter
        
        # Initialize game history
        self.game_history = []
        self.move_explanations = []
        self.opponent_remarks = []
        self.utterance_count = 0
        
        if utterances_matter and self.playing_mode == KAgent.DEMO:
            try:
                # Try using OpenAI package with DeepSeek endpoint
                from openai import OpenAI
                self.deepseek_client = OpenAI(
                    api_key="sk-e0bbfc05a3e84e4ca57b051c62835196", 
                    #Deepseek API created. 
                    # (When testing, if running out of balance, contact me and I'll recharge it hahaha.)
                    base_url="https://api.deepseek.com/v1"
                )
                print("DeepSeek API initialized successfully using OpenAI client")
            except ImportError:
                try:
                    from deepseek import DeepSeek
                    self.deepseek_client = DeepSeek(api_key="sk-e0bbfc05a3e84e4ca57b051c62835196")
                    print("DeepSeek API initialized successfully using native client")
                except ImportError:
                    print("DeepSeek API not available, using rule-based instructive responses")
                    self.deepseek_client = None
            except Exception as e:
                print(f"Error initializing DeepSeek: {e}")
                self.deepseek_client = None
        else:
            self.deepseek_client = None
           
        return "OK"

    def generate_instructive_utterance(self, current_state, move, score, opponent_remark=""):
        """Generate instructive utterances about game strategy and theory"""
        
        # This branching condition is created for the bonus part.
        # FIRST: Check for special opponent remarks that require specific responses
        if "tell me how you did that" in opponent_remark.lower():
            return self._explain_computation()
        elif "what's your take on the game so far?" in opponent_remark.lower():
            return self._analyze_game_progress(current_state)
        
        # Then proceed with normal instructive utterance generation
        if self.deepseek_client and self.utterances_matter and self.playing_mode == KAgent.DEMO:
            try:
                prompt = self._create_instructive_prompt(current_state, move, score, opponent_remark)
                
                if hasattr(self.deepseek_client, 'chat'):
                    response = self.deepseek_client.chat.completions.create(
                        model="deepseek-chat",
                        messages=[
                            {"role": "system", "content": "You are Professor AI, an expert game theory instructor. Explain game strategy in an educational, helpful manner. Focus on teaching concepts like minimax, alpha-beta pruning, positional advantage, and strategic patterns. Keep responses concise (1-2 sentences) and focused on instructional value."},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=2000
                    )
                    return response.choices[0].message.content.strip()
                else:
                    response = self.deepseek_client.chat.completions.create(
                        model="deepseek-chat",
                        messages=[
                            {"role": "system", "content": "You are Professor AI, an expert game theory instructor. Explain game strategy in an educational, helpful manner. Focus on teaching concepts like minimax, alpha-beta pruning, positional advantage, and strategic patterns. Keep responses concise (1-2 sentences) and focused on instructional value."},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=2000
                    )
                    return response.choices[0].message.content.strip()
            except Exception as e:
                print(f"DeepSeek API error: {e}, using fallback")
                # Fall back to rule-based if API fails
                return self._rule_based_instructive_response(current_state, move, score, opponent_remark)
        
        return self._rule_based_instructive_response(current_state, move, score, opponent_remark)
           
    def _create_instructive_prompt(self, current_state, move, score, opponent_remark):
        """Create a prompt for DeepSeek to generate instructive responses"""
        
        board_desc = self._describe_board(current_state.board)
        move_desc = f"Moving to position {move}"
        score_desc = f"Evaluation score: {score}"
        
        game_type_name = self.current_game_type.long_name if self.current_game_type else 'K-in-a-Row'
        k_value = self.current_game_type.k if self.current_game_type else 'N/A'
        
        prompt = f"""
        Game Context:
        - Board state: {board_desc}
        - My move: {move_desc}
        - Position evaluation: {score_desc}
        - I am playing as: {self.playing}
        - Game type: {game_type_name}
        - K value: {k_value}
        
        Opponent's remark: "{opponent_remark}"
        
        Please provide a brief, instructive comment about this move that teaches about game strategy. 
        You should using I or my move etc. to refer to yourself.
        Focus on concepts like:
        - Positional advantage
        - Threat creation/defense
        - Strategic patterns
        - Game theory principles
        - Search algorithm insights
        
        Keep it instructive and concise (1-2 sentences mostly. If it is a crucial move, you can use more words).
        """
        return prompt
    
    def _rule_based_instructive_response(self, current_state, move, score, opponent_remark):
        """Generate instructive responses using rule-based system"""
        
        # Analyze the board for teaching points
        teaching_points = self._analyze_teaching_points(current_state, move, score)
        
        # Select from various instructive templates based on context
        instructive_templates = [
            f"This move demonstrates {teaching_points['concept']} by {teaching_points['reason']}.",
            f"From a game theory perspective, this position illustrates {teaching_points['concept']}.",
            f"Notice how this move creates {teaching_points['threats']} while defending against {teaching_points['defenses']}.",
            f"This strategic placement shows the importance of {teaching_points['principle']} in {self.current_game_type.k}-in-a-Row games.",
            f"Using {teaching_points['algorithm']} search, I evaluated this position as favorable due to {teaching_points['reason']}."
        ]
        
        self.utterance_count += 1
        return instructive_templates[self.utterance_count % len(instructive_templates)]
      
    def _analyze_teaching_points(self, current_state, move, score):
        """Analyze the current position for instructional content"""

        board = current_state.board
        i, j = move
        
        # Determine strategic concepts to teach
        # Since most games will not be played in TTT, I focus on FIAR or Cassini instruction.
        concepts = []
        if abs(score) > 500:  # Near winning/losing position
            concepts.append("critical position evaluation")
        elif abs(score) > 100:
            concepts.append("strategic advantage")
        else:
            concepts.append("positional development")
        
        # Analyze threats and defenses
        threats = self._count_potential_threats(board, move, self.playing)
        defenses = self._count_potential_threats(board, move, 'O' if self.playing == 'X' else 'X')
        
        return {
            'concept': concepts[0],
            'reason': f"controlling key lines and creating opportunities",
            'threats': f"{threats} potential winning lines",
            'defenses': f"opponent's attacking options",
            'principle': "center control and line development",
            'algorithm': "minimax with alpha-beta pruning"
        }
    
    def _count_potential_threats(self, board, move, player):
        """Count potential winning threats for a player"""
        # Simplified threat counting
        count = 0
        n_rows = len(board)
        n_cols = len(board[0]) if n_rows > 0 else 0
        k = self.current_game_type.k if self.current_game_type else 3

        # Check rows, columns, and diagonals for potential threats
        return min(3, n_rows * n_cols // (k * 2))  # Rough estimate
    
    def _describe_board(self, board):
        """Create a text description of the board for the AI"""
        n_rows = len(board)
        n_cols = len(board[0]) if n_rows > 0 else 0
        x_count = sum(row.count('X') for row in board)
        o_count = sum(row.count('O') for row in board)
        empty_count = sum(row.count(' ') for row in board)
        
        return f"{n_rows}x{n_cols} board with {x_count} X's, {o_count} O's, and {empty_count} empty spaces"
    
    # The following two methods are used for bonus part 2 task2:
    def _record_move_stats(self, depth_remaining, start_time, final_score):
        end_time = time.time()
        self.last_move_stats = {
            'static_evals': self.num_static_evals_this_turn,
            'cutoffs': self.alpha_beta_cutoffs_this_turn,
            'depth': depth_remaining,
            'time_spent': round(end_time - start_time, 3),
            'final_score': final_score,
            'branches_explored': self.num_static_evals_this_turn + self.alpha_beta_cutoffs_this_turn
        }

    def _explain_computation(self):
        """Explain the computational process for the last move with detailed statistics"""
        if not hasattr(self, 'last_move_stats') or not self.last_move_stats:
            return "I haven't made a move yet, so I can't explain my computation process."
        
        stats = self.last_move_stats
        explanation = (
            f"In my last move, I employed a comprehensive search strategy. "
            f"I evaluated {stats.get('static_evals', self.num_static_evals_this_turn)} positions using static evaluation, "
            f"achieving {stats.get('cutoffs', self.alpha_beta_cutoffs_this_turn)} alpha-beta cutoffs for efficiency. "
            f"The search reached a depth of {stats.get('depth', 'unknown')} ply, "
            f"exploring approximately {stats.get('branches_explored', 'multiple')} branches. "
            f"This took {stats.get('time_spent', 'a brief moment')} seconds. "
            f"My evaluation function scored the position at {stats.get('final_score', 'an unknown value')}, "
            f"considering factors like line control, threat creation, and positional advantage."
        )
        
        # Add search efficiency metrics
        if stats.get('static_evals', 0) > 0:
            efficiency = stats.get('cutoffs', 0) / stats.get('static_evals', 1)
            explanation += f" The pruning efficiency was {efficiency:.1%}, demonstrating effective search space reduction."
        
        return explanation
    
    # The following method is used for bonus part 2 task3:
    def _analyze_game_progress(self, current_state):
        """Provide comprehensive analysis of the game progress using history"""
        board = current_state.board
        n_rows = len(board)
        n_cols = len(board[0]) if n_rows > 0 else 0
        
        # Calculate basic statistics
        x_count = sum(row.count('X') for row in board)
        o_count = sum(row.count('O') for row in board)
        total_moves = x_count + o_count
        total_spaces = n_rows * n_cols
        
        # Build the story from game history
        story = "Let me recount our game's journey:\n"
        
        if len(self.game_history) > 0:
            # Opening phase analysis
            opening_moves = min(4, len(self.move_explanations))
            story += f"- Opening: We began with {opening_moves} strategic placements. "
            
            # Key turning points
            if len(self.move_explanations) > 2:
                story += "Early on, we established our positions. "
            
            # Mid-game developments
            if total_moves >= 4:
                story += f"- Development: By move {total_moves//2}, the battle lines were drawn. "
            
            # Recent developments
            if len(self.move_explanations) >= 2:
                recent_move = self.move_explanations[-1]
                story += f"- Recently: My last move focused on {recent_move.get('explanation', 'strategic positioning')}. "
        
        # Current position analysis
        story += f"\nCurrent state: {total_moves} moves played on our {n_rows}x{n_cols} board. "
        story += f"You control {o_count} positions, I control {x_count}. "
        
        # Position evaluation and prediction
        current_score = self.static_eval(current_state)
        empty_spaces = total_spaces - total_moves
        
        if abs(current_score) > 1000:
            if current_score > 0:
                story += "I've established a commanding position and am likely to win if I play accurately."
            else:
                story += "You've built a strong advantage and have excellent winning chances with precise play."
        elif abs(current_score) > 100:
            if current_score > 0:
                story += f"I have a slight advantage. With {empty_spaces} moves remaining, I'm cautiously optimistic."
            else:
                story += f"You have the initiative. There are {empty_spaces} moves left to capitalize on your position."
        else:
            story += f"The game is balanced. With {empty_spaces} moves remaining, it's anyone's game - precision will decide."
        
        # Strategic assessment based on game phase
        if total_moves < 4:
            story += " We're still in the opening - focus on center control and flexible development."
        elif total_moves < total_spaces * 0.6:
            story += " This is the complex middlegame where tactical awareness is crucial."
        else:
            story += " We're in the endgame where every move carries maximum weight."
        
        # Add specific observations from move history
        if len(self.opponent_remarks) > 1:
            story += f" Our exchange of remarks shows an engaging battle of wits!"
        
        return story
    
    # The core of your agent's ability should be implemented here:             
    def make_move(self, current_state, current_remark, time_limit=1000,
                  use_alpha_beta=True,
                  use_zobrist_hashing=False, max_ply=3,
                  special_static_eval_fn=None):
        
        start_time = time.time()
        # Reset per-turn statistics
        #Also store current remark from opponent
        self.alpha_beta_cutoffs_this_turn = 0
        self.num_static_evals_this_turn = 0
        self.zobrist_table_num_entries_this_turn = -1
        self.zobrist_table_num_hits_this_turn = -1
        self.opponent_remarks.append(current_remark)
        """
        # FIRST: Check if we can win immediately with any move
        winning_move = self._find_winning_move(current_state)
        if winning_move:
            # Create state with winning move
            best_state = State(old=current_state)
            i, j = winning_move
            best_state.board[i][j] = self.playing
            best_state.whose_move = 'O' if self.playing == 'X' else 'X'
            new_remark = f"Game over! I win with this move at position {winning_move}."
            move_info = [winning_move, best_state]
            if self.playing_mode == KAgent.AUTOGRADER:
                move_info.extend([
                    self.alpha_beta_cutoffs_this_turn,
                    self.num_static_evals_this_turn,
                    self.zobrist_table_num_entries_this_turn,
                    self.zobrist_table_num_hits_this_turn
                ])
            return [move_info, new_remark]
        """
        if self.playing_mode == KAgent.AUTOGRADER and special_static_eval_fn:
            static_eval_fn = special_static_eval_fn
        else:
            static_eval_fn = self.static_eval
        
        # Call minimax to get the best move
        if use_alpha_beta:
            best_move, best_score = self.minimax(
                current_state,
                depth_remaining=max_ply,
                pruning=True,
                alpha=float('-inf'),
                beta=float('inf'),
                static_eval_fn=static_eval_fn
            )
        else:
            best_move, best_score = self.minimax(
                current_state,
                depth_remaining=max_ply,
                pruning=False,
                alpha=None,
                beta=None,
                static_eval_fn=static_eval_fn
            )
        self._record_move_stats(max_ply, start_time, best_score)
        # Create new state with the best move and update whose_move 
        #Then generate the utterance explaining the move
        best_state = State(old=current_state)
        i, j = best_move
        best_state.board[i][j] = self.playing
        best_state.whose_move = 'O' if self.playing == 'X' else 'X'
        self.game_history.append(best_state)
        new_remark = self.generate_instructive_utterance(current_state, best_move, best_score, current_remark)
        
        # Store move explanation for history
        self.move_explanations.append({
            'move': best_move,
            'score': best_score,
            'explanation': new_remark
        })
        move_info = [best_move, best_state]
        if self.playing_mode == KAgent.AUTOGRADER:
            move_info.extend([
                self.alpha_beta_cutoffs_this_turn,
                self.num_static_evals_this_turn,
                self.zobrist_table_num_entries_this_turn,
                self.zobrist_table_num_hits_this_turn
            ])
        return [move_info, new_remark]
    """
    def _find_winning_move(self, state):
        n_rows = len(state.board)
        n_cols = len(state.board[0]) if n_rows > 0 else 0
        k = self.current_game_type.k if self.current_game_type else 3
        
        # Check all empty positions
        for i in range(n_rows):
            for j in range(n_cols):
                if state.board[i][j] == ' ':
                    # Test if placing our piece here would win the game
                    if self._would_win(state, i, j, self.playing, k):
                        return (i, j)
        return None

    def _would_win(self, state, row, col, player, k):
        temp_state = State(old=state)
        temp_state.board[row][col] = player
        # Check all directions for k-in-a-row
        directions = [
            (0, 1),   # horizontal
            (1, 0),   # vertical  
            (1, 1),   # diagonal down-right
            (1, -1)   # diagonal down-left
        ]
        #dr->row, dc->col
        for dr, dc in directions:
            count = 1  
            # Check positive direction
            r, c = row + dr, col + dc
            while (0 <= r < len(temp_state.board) and 0 <= c < len(temp_state.board[0]) 
                   and temp_state.board[r][c] == player):
                count += 1
                r += dr
                c += dc
            
            # Check negative direction  
            r, c = row - dr, col - dc
            while (0 <= r < len(temp_state.board) and 0 <= c < len(temp_state.board[0]) 
                   and temp_state.board[r][c] == player):
                count += 1
                r -= dr
                c -= dc
            
            if count >= k:
                return True
        
        return False
        """
    def minimax(self,
            state,
            depth_remaining,
            pruning=False,
            alpha=None,
            beta=None,
            static_eval_fn=None):
        
        # Base case: depth limit reached
        if depth_remaining == 0:
            # Use the provided static evaluation function
            if static_eval_fn:
                score = static_eval_fn(state)
            else:
                score = self.static_eval(state, self.current_game_type)
            self.num_static_evals_this_turn += 1
            return None, score
        
        # Get board dimensions
        n_rows = len(state.board)
        n_cols = len(state.board[0]) if n_rows > 0 else 0
        
        # Determine whose turn it is from the state
        current_player = state.whose_move
        is_maximizing = (current_player == 'X')
        
        # Get all legal moves
        legal_moves = []
        for i in range(n_rows):
            for j in range(n_cols):
                if state.board[i][j] == ' ':
                    legal_moves.append((i, j))
        
        if not legal_moves:
            if static_eval_fn:
                score = static_eval_fn(state)
            else:
                score = self.static_eval(state, self.current_game_type)
            self.num_static_evals_this_turn += 1
            return None, score

        best_move = legal_moves[0]
        best_value = float('-inf') if is_maximizing else float('inf')
        # For autograder mode, use consistent move ordering
        if self.playing_mode == KAgent.AUTOGRADER:
            legal_moves = sorted(legal_moves)
        # Search through moves
        for move in legal_moves:
            # Create new state
            new_state = State(old=state)
            i, j = move
            new_state.board[i][j] = current_player
            # Update whose_move for the next player
            new_state.whose_move = 'O' if current_player == 'X' else 'X'
            
            # Recursive search
            _, score = self.minimax(
                new_state, 
                depth_remaining-1, 
                pruning, 
                alpha, 
                beta, 
                static_eval_fn
            )
            
                # Alpha-beta pruning
            if is_maximizing:
                if score > best_value:
                    best_value = score
                    best_move = move
                if pruning and alpha is not None:
                    if score > alpha:
                        alpha = score
                    if beta is not None and alpha >= beta:
                        self.alpha_beta_cutoffs_this_turn += 1
                        break
            else:
                if score < best_value:
                    best_value = score
                    best_move = move
                if pruning and beta is not None:
                    if score < beta:
                        beta = score
                    if alpha is not None and beta <= alpha:
                        self.alpha_beta_cutoffs_this_turn += 1
                        break
        
        return best_move, best_value

    # I adopted the provided function in our worksheet.
    def static_eval(self, state, game_type=None):
        if game_type is None:
            game_type = self.current_game_type
        
        k = game_type.k if game_type else 3
        f_scores = [0] * (k + 1)  
        
        # Get board dimensions
        n_rows = len(state.board)
        n_cols = len(state.board[0]) if n_rows > 0 else 0
        
        # Evaluate all possible lines of length k
        # Rows
        for i in range(n_rows):
            for j in range(n_cols - k + 1):
                line = [state.board[i][j + offset] for offset in range(k)]
                self._evaluate_line_counters(line, f_scores, k)
        # Columns
        for j in range(n_cols):
            for i in range(n_rows - k + 1):
                line = [state.board[i + offset][j] for offset in range(k)]
                self._evaluate_line_counters(line, f_scores, k)
        # Diagonals (top-left to bottom-right)
        for i in range(n_rows - k + 1):
            for j in range(n_cols - k + 1):
                line = [state.board[i + offset][j + offset] for offset in range(k)]
                self._evaluate_line_counters(line, f_scores, k)
        # Diagonals (top-right to bottom-left)
        for i in range(n_rows - k + 1):
            for j in range(k - 1, n_cols):
                line = [state.board[i + offset][j - offset] for offset in range(k)]
                self._evaluate_line_counters(line, f_scores, k)
        
        # Calculate final score using the formula: e(n) = f1(n) + 10*f2(n) + 100*f3(n) + ...
        total_score = 0
        for i in range(1, k + 1):
            weight = 10 ** (i - 1)  # f1: 1, f2: 10, f3: 100, f4: 1000, etc.
            total_score += f_scores[i] * weight
        
        return total_score

    def _evaluate_line_counters(self, line, f_scores, k):
        """Evaluate a line and update the f_scores counters"""
        x_count = line.count('X')
        o_count = line.count('O')
        
        # If line contains both X and O, it's blocked - no score
        if x_count > 0 and o_count > 0:
            return
        
        # If line contains only X
        if x_count > 0 and o_count == 0:
            f_scores[x_count] += 1
        
        # If line contains only O  
        if o_count > 0 and x_count == 0:
            f_scores[o_count] -= 1

    def evaluate_line_simple(self, line, k):
        """Evaluate a single line using the specified scoring system"""
        x_count = line.count('X')
        o_count = line.count('O')
        
        # If line contains both X and O, it's blocked - no score
        if x_count > 0 and o_count > 0:
            return 0
        
        # If line contains only X
        if x_count > 0 and o_count == 0:
            return 10 ** (x_count - 1)  # f1: 1, f2: 10, f3: 100, etc.
        
        # If line contains only O
        if o_count > 0 and x_count == 0:
            return - (10 ** (o_count - 1))  # negative for O
        
        # Empty line
        return 0
    