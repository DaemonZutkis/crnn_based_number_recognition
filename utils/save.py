import os
import pandas as pd

class Save():
    def save_metrix(self, path_to_save, metrix):
        """
        save metrix in path_to_save
        """
        path_metrix = os.path.join(path_to_save)

        metrix = pd.DataFrame(metrix)

        if os.path.isfile(path_metrix):
            metrix_0 = pd.read_csv(path_metrix)
            metrix = pd.concat([metrix_0, metrix], axis=0)

        metrix.to_csv(path_metrix, sep=',', encoding='utf-8', index=None)