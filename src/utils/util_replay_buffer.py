import pathlib
import pickle
class UtilReplayBuffer:
    @staticmethod
    def get_replay_buffer(replay_buffer_path= None):
        if replay_buffer_path:
            replay_buffer_path = pathlib.Path(replay_buffer_path)
            with open(replay_buffer_path, "rb") as f:
                replay_buffer_infos = pickle.load(f)
        else:
            replay_buffer_infos = {}
        return replay_buffer_infos