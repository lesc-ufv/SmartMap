import os
class UtilPath:
    @staticmethod
    def get_last_two_folders(path):
        path = os.path.normpath(path)
        parts = path.split(os.sep)
        if len(parts) >= 2:
            return parts[-2], parts[-1]
        else:
            raise ValueError("Path must have at least two folders")