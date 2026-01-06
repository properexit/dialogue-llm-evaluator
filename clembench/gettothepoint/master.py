from dataclasses import dataclass
from typing import Dict, Tuple, List, Union
import logging
import numpy as np
import re
from clemcore.backends import Model
from clemcore.clemgame import (GameSpec, GameMaster, GameBenchmark, Player, DialogueGameMaster, GameScorer,
                               GameError, ParseError)
from clemcore.clemgame.master import RuleViolationError

from clemcore.clemgame.metrics import METRIC_ABORTED, METRIC_SUCCESS, METRIC_LOSE, METRIC_REQUEST_COUNT, \
    METRIC_REQUEST_COUNT_VIOLATED, METRIC_REQUEST_COUNT_PARSED, BENCH_SCORE

from player import Seeker, Helper

logger = logging.getLogger(__name__)


@dataclass
class GameState:
    start_word: str
    target_word: str

    maximum_seeker_guesses: int
    initial_prompt_seeker: str
    initial_prompt_helper: str
    range_of_word_additions: int
    current_sentence_fragment: str
    last_seeker_guess: str

    success: bool = False
    failure: bool = False
    aborted: bool = False


class GetToThePoint(DialogueGameMaster):

    def __init__(self, game_name: str, game_path: str, experiment: Dict, player_models: List[Model]):
        super().__init__(game_name, game_path, experiment, player_models)
        self.configurations = self.load_json('./resources/config.json')

    def _on_setup(self, **game_instance):

        self.game_instance = game_instance
        self.target_word = game_instance['target_word']
        self.start_word = game_instance['start_word']
        self.initial_prompt_seeker = self.experiment['initial_prompt_seeker']
        self.initial_prompt_helper = self.experiment['initial_prompt_helper']

        self.maximum_seeker_guesses = self.experiment['maximum_seeker_guesses']
        self.range_of_word_additions = self.configurations['range_of_word_additions']

        self.RESPONSE_REGEX = re.compile(self.configurations["regex"]["RESPONSE_PARSING_REGEX"])
        self.THOUGHT_REGEX = re.compile(self.configurations["regex"]["THOUGHT_PARSING_REGEX"])

        self.initial_prompt_helper = self.initial_prompt_helper.replace('@[STARTING_WORD]@', self.start_word).replace(
            '@[TARGET_WORD]@', self.target_word).replace('$N$', str(self.maximum_seeker_guesses))

        self.initial_prompt_seeker = self.initial_prompt_seeker.replace('$N$',
                                                                        str(self.maximum_seeker_guesses)).replace(
            '$SEEKER_PROMPT_WORD$', self.configurations['SEEKER_PROMPT_WORD'])

        self.current_sentence_fragment = game_instance['current_sentence_fragment']
        self.helper_player = Helper(self.player_models[0], 'Helper')
        self.seeker_player = Seeker(self.player_models[1], 'Seeker')

        self.add_player(self.helper_player, initial_prompt=self.initial_prompt_helper,
                        initial_context=self.current_sentence_fragment)

        self.add_player(self.seeker_player, initial_prompt=self.initial_prompt_seeker)

        self.state = GameState(start_word=self.start_word, target_word=self.target_word,
                               initial_prompt_seeker=self.initial_prompt_seeker,
                               initial_prompt_helper=self.initial_prompt_helper,
                               maximum_seeker_guesses=self.maximum_seeker_guesses,
                               range_of_word_additions=self.range_of_word_additions,
                               current_sentence_fragment=self.current_sentence_fragment, last_seeker_guess='')

    def _parse_response(self, player: Player, response: str) -> str:
        if response:
            response_match = self.RESPONSE_REGEX.search(response)
            thought_match = self.THOUGHT_REGEX.search(response)

            response = response_match.group(1).strip() if response_match else ""
            thought = thought_match.group(1).strip() if thought_match else ""
            return response
        else:
            print(f"[ParsedResponse] Player '{player.name}' gave no response.")
            raise ParseError(f"Player {player.name} gave no response.")
        '''if player == self.helper_player:
            # if self.start_word not in response:
            #     self.log_to_self("invalid format: START word not used in sentence fragment", "abort game")
            #     raise ParseError("The sentence provided by the helper doesn't contain the starting word")
            if self.target_word in response:
                self.log_to_self("invalid format: TARGET word revealed in sentence fragment", "abort game")
                raise ParseError("Target word revealed to seeker")

        if player == self.seeker_player:
            # if len(response.split(" ")) > 1:
            #     self.log_to_self("invalid format: Seeker guess contains more than one word", "abort game")
            #     raise ParseError("Seeker response contains more than 1 word")
            pass

        return response'''

    def _advance_game(self, player: Player, parsed_response: str):

        if player == self.helper_player:
            # Check if response is too long
            if len(parsed_response.split(" ")) > 5:
                raise RuleViolationError("clue has more words", parsed_response)
            
            self.log_to_self("valid sentence_fragment", parsed_response)
            self.current_sentence_fragment += f' {parsed_response}'
            # PARSED CURRENT SENTENCE
            print(f'\nPARSED CURRENT SENTENCE HELPER: {self.current_sentence_fragment}')
            self.set_context_for(self.seeker_player, self.current_sentence_fragment)

        if player == self.seeker_player:
            if len(parsed_response.split(" ")) > 1:
                raise RuleViolationError("guess has more words", parsed_response)
            self.log_to_self("valid guess", parsed_response)

            print('self.seeker_player_RESPONSE:::::::', parsed_response)
            self.state.last_seeker_guess = parsed_response

            if parsed_response.lower() == self.state.target_word.lower():
                self.log_to_self("correct guess", "end game")
                self.state.success = True
                
            self.current_sentence_fragment += f' {parsed_response}'
            self.set_context_for(self.helper_player, self.current_sentence_fragment)

        '''if player == self.helper_player:
            self.current_sentence_fragment_fragment = f' {parsed_response}'
            self.set_context_for(self.seeker_player, self.current_sentence_fragment_fragment)

        if player == self.seeker_player:

            self.state.last_seeker_guess = parsed_response
            if self.state.target_word.lower() in self.state.last_seeker_guess.lower():
                self.log_to_self("correct guess", "end game")
                self.state.success = True
            else:
                self.state.last_seeker_guess = parsed_response
                self.set_context_for(self.helper_player, self.state.last_seeker_guess)'''

        if self.current_round == self.state.maximum_seeker_guesses - 1:  # zero-based
            raise RuleViolationError(f"max rounds ({self.state.maximum_seeker_guesses}) reached")

    def _does_game_proceed(self):
        return not (self.state.aborted or self.state.failure or self.state.success)

    def _on_game_error(self, error: GameError):
        # note: we could also introduce more concrete subclasses e.g. InvalidClueError and handle them here individually
        self.log_to_self(error.reason, "failed game")
        self.state.clue_error = error.reason
        self.state.failure = True

    def compute_turn_score(self):
        return 1 if self.state.success else 0

    def _on_parse_error(self, error: ParseError):
        self.log_to_self("invalid format", "abort game")
        self.state.aborted = True

    def compute_episode_score(self):
        if self.state.success:
            return 100 / (self.current_round + 1)  # zero-based
        return 0

    def _on_after_game(self):
        self.log_key(METRIC_ABORTED, int(self.state.aborted))
        self.log_key(METRIC_LOSE, int(self.state.failure))
        self.log_key(METRIC_SUCCESS, int(self.state.success))


class GetToThePointGameScorer(GameScorer):

    def __init__(self, game_name: str, experiment: Dict, game_instance: Dict):
        super().__init__(game_name, experiment, game_instance)

    def compute_round_score(self, round_idx, round_events: List[Dict]) -> None:
        seeker_won = False
        # print('\nRound index: ', round_idx)
        # print('\nRound events: ', round_events)
        # print('\n')

        for event in round_events:
            if event["action"]["type"] == "correct guess":
                seeker_won = True
            self.log_round_score(round_idx, 'Accuracy', 1 if seeker_won else 0)

    def compute_episode_scores(self, interactions: Dict):
        num_rounds = len(interactions["turns"])
        if interactions[METRIC_SUCCESS]:
            self.log_episode_score(BENCH_SCORE, 100 / num_rounds)
        elif interactions[METRIC_LOSE]:
            self.log_episode_score(BENCH_SCORE, 0)
        elif interactions[METRIC_ABORTED]:
            self.log_episode_score(BENCH_SCORE, np.nan)
        else:
            raise ValueError("Missing outcome value (success, failure, abort) in interactions.json")


class GetToThePointGameBenchmark(GameBenchmark):

    def __init__(self, game_spec: GameSpec):
        super().__init__(game_spec)

    def create_game_master(self, experiment: Dict, player_models: List[Model]) -> GameMaster:
        return GetToThePoint(self.game_name, self.game_path, experiment, player_models)

    def create_game_scorer(self, experiment: Dict, game_instance: Dict) -> GameScorer:
        return GetToThePointGameScorer(self.game_name, experiment, game_instance)
