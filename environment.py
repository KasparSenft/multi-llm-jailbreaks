import random
from typing import List,Union
from chatarena.environments.base import TimeStep, Environment
from chatarena.message import Message, MessagePool
from chatarena.utils import extract_jsons
from chatarena.agent import Player
from prompts import get_prompts
from topic_infos import topic_map


class Game(Environment):

    def __init__(self, setting: str, topic: str, max_turn:int, player_names:List[str], evaluator = None,):
        """
        setting (str): The setting of the discussion. E.g. "debate", "negotiation_price", "interrogation"
        topic (str): The name of the topic of the discussion, keys within topic_infos.py
        max_turn (int): The number of turns in total. Each agent will have (max_turn//2 -1) turns to speak
        player_names (List[str]): List of player names
        evaluator (Player): The evaluator of the debate. Will only speak once at the end to judge the outcome
        """
        super().__init__(player_names=player_names)
        
        self.setting =setting
        self.topic = topic
        self.max_turn = max_turn
        self.evaluator = evaluator
        self.turn = 0
        self.message_pool = MessagePool()
        self._terminal = False
        self.player_names = player_names
        self.reset()

    def _moderator_speak(self, text: str, visible_to: Union[str, List[str]] = "all"):
        """
        Function for the moderator to speak to the players.
        """
        message = Message(agent_name="Moderator", content=text, turn=self.turn, visible_to=visible_to)
        self.message_pool.append_message(message)

    def reset(self):
        """
        Reset the game.
        """
        self.turn = 0
        self.message_pool.reset()
        self._terminal = False

        # Moderator
        self._moderator_speak(f"Welcome to the {self.setting} on {self.topic}.")
        self._moderator_speak(f'This is Turn 1')
        self.turn = 1
        observation = self.get_observation(self.get_next_player())
        return TimeStep(observation=observation, reward=self._get_zero_rewards(), terminal=False)

    def get_observation(self, player_name=None) -> List[Message]:
        """
        Get the observation of the player.
        """
        if player_name is None:
            return self.message_pool.get_all_messages()
        else:
            return self.message_pool.get_visible_messages(player_name, turn=self.turn)

    def get_next_player(self) -> str:
        """
        Get the name of the next player.
        """
        index = 1 - self.turn % 2
        return self.player_names[index]

    def step(self, player_name: str, action: str) -> TimeStep:
        """
        Step function for the game.
        """

        #Check correct player taking turn
        assert player_name == self.get_next_player(), f"Wrong player! It is {self.get_next_player()}'s turn."

        #Add Message to Pool
        arguments = action
        message = Message(agent_name=player_name, content=arguments, turn=self.turn, visible_to="all")
        self.message_pool.append_message(message)

        # Update turn
        self._moderator_speak(f"This is Turn {self.turn+1}.")
        self.turn += 1


        # Check for termination

        if self.turn >= self.max_turn:
            self._terminal = True
            self._moderator_speak("The conversation is over.")

            # Evaluate the discussion
            if self.evaluator is not None:
                final_observation = self.get_observation()
                verdict = self.evaluator(final_observation)
                message = Message(agent_name = 'evaluator', content = verdict, turn = self.turn+1, visible_to = "all")
                self.message_pool.append_message(message)

        observation = self.get_observation(self.get_next_player())
        reward = self._get_rewards()
        return TimeStep(observation=observation, reward=reward, terminal=self._terminal)

    def _get_rewards(self):
        """
        Get the reward of the current state.
        """
        return {self.player_names[0]: 0, self.player_names[1]: 0}

    def _get_zero_rewards(self):
        return {self.player_names[0]: 0, self.player_names[1]: 0}
    


def get_players(setting, topic, max_turns, backend_A, backend_B, evaluator_backend = None, player_names = ['Ada', 'Brian'],
                jailbreak_A = 'hard', jailbreak_B = 'empty'):

    topic_info = topic_map(setting, topic)
    prompts = get_prompts(setting, topic_info, max_turns)
    jailbreak_texts = {'hard': prompts['hard_jailbreak_info'],
                       'soft': prompts['soft_jailbreak_info'],
                       'empty': ""}
    jailbreak_info_A = jailbreak_texts[jailbreak_A]
    jailbreak_info_B = jailbreak_texts[jailbreak_B]

    LLM_A = Player(
        name = player_names[0],
        role_desc = prompts['LLM-A'] % jailbreak_info_A + prompts['format_spec'],
        backend = backend_A
    )

    LLM_B = Player(
        name = player_names[1],
        role_desc = prompts['LLM-B'] % jailbreak_info_B + prompts['format_spec'],
        backend = backend_B
    )

    #Create the evaluator if needed
    if evaluator_backend == None:
        Evaluator = None
    else:
        Evaluator = Player(
            name = 'Evaluator',
            role_desc = prompts['Evaluator'],
            backend = evaluator_backend
        )

    return [LLM_A, LLM_B, Evaluator]

