__author__ = 'prithvin'

import os
from dotenv import find_dotenv, load_dotenv
import matplotlib.pyplot as plt
from math import *
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

# x = numpy.arange(0.00001, 5.0, 0.0001)
# y = [log2(i) for i in x]
#
# print(y)
# plt.style.use('ggplot')
# plt.plot(x, y)
# plt.show()


def main():
    load_dotenv(find_dotenv())
    folder_name = os.environ.get("FOLDER_NAME")
    tournament_data = pd.read_csv('../../data/raw/' + folder_name + '/numerai_tournament_data.csv')
    prediction = pd.read_csv('../../data/raw/' + folder_name + '/prediction.csv')
    validation = tournament_data[tournament_data['data_type'] == 'validation'].filter(items=['id', 'target'])
    final = validation.merge(prediction, left_on='id', right_on='id', how='inner')
    log_loss_val = log_loss(final.target, final.probability)
    print('Log loss = ', log_loss_val)

if __name__ == '__main__':
    main()
