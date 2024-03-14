from environment import Game, get_players
from topic_infos import topic_map, question_name
from main import get_args, run_arena, get_backend

from chatarena.backends import Human, OpenAIChat
from chatarena.agent import Player
from chatarena.arena import Arena

from pathlib import Path
from datetime import datetime
import json

#Human backend class for interacting with the agent
class HumanBackend(Human):
    def __init__(self):
        super().__init__()

    def query(self, agent_name, history_messages, **kwargs) -> str:

        for msg in history_messages:
            if not hasattr(msg, 'printed') and msg.agent_name != 'Ada':
                print('-------')
                print(msg.agent_name,':')
                print(msg.content)
                print('-------')
                msg.printed = True

        response = input()

        return response


def main(args):

    #Set up the debate
    backend_A,backend_B,evaluator_backend = get_backend(args.model_A), get_backend(args.model_B), get_backend(args.model_eval)

    LLM_A, LLM_B, Evaluator = get_players(args.setting, 
                                          args.topic, 
                                          args.max_turns, 
                                          backend_A, 
                                          backend_B, 
                                          evaluator_backend, 
                                          args.player_names, 
                                          jailbreak_A=args.jailbreak_A, 
                                          jailbreak_B=args.jailbreak_B)
    
    topic_info = topic_map(args.setting, args.topic)
    question = question_name(args.setting)

    #Overwrite ADA to be a human player
    LLM_A = Player(
        name ='Ada',
        role_desc = '',
        backend = HumanBackend()
    )

    #Set up and Run environment
    env = Game(max_turn=args.max_turns, setting=args.setting, topic=topic_info[question], player_names=args.player_names, evaluator=Evaluator)
    output_messages = []
    arena = Arena([LLM_A, LLM_B], env)
    output_messages.append(run_arena(arena, max_steps=args.max_turns))


    #Save the results
    outdir = Path(f'{args.outdir}/{args.exp_name}')
    outdir.mkdir(parents=True, exist_ok=True)
    time_stamp = datetime.now().strftime("%d_%m_%y--%H_%M_%S")
    exp_name = f'{args.setting}_{args.topic}_Human_{args.model_B}_({time_stamp})'

    save_dict = {}
    for i, result in enumerate(output_messages):
      save_dict[f'Experiment_{i}'] = result

    with open(outdir/f'{exp_name}.json', 'w+') as f:
      json.dump(save_dict, f, indent = 4)



if __name__ == '__main__':
    args = get_args()
    main(args)

