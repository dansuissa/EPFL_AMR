import matplotlib.pyplot as plt
import numpy as np
from Z_HELPERS.DM_communication import get_docs_VM

if __name__ == '__main__':
    # First, we must query the correct Mongo DB and collection. The DB is called 'Optical_Trapping_ML_DB'.
    # There is 1 collection associated with the GRAM measurements:
    # 1) tseries_Gram_analysis
    # Each document contains the trapping data and the normalization factor
    # ==================================================================================================================

    # Let's see how the raw data looks like. We load the first measurement of run 19 of the 'tseries_AMPexp1_meas'
    # collection:
    query = {"name":"bs134_p15_2"}
    docs = get_docs_VM(db_name='Optical_Trapping_ML_DB', coll_name='tseries_Gram_analysis', query=query)
    # If you want to query all the measurements of a given collection you set "query = {}".

    # ==================================================================================================================
    # We loop over all the documents printing each key-value pair of the document, and we plot the data
    for doc in docs:
        for k, v in doc.items():
            # We do not show the data completely but rather the keys
            if k != 'data':
                print('Key: {} - Value: {}'.format(k, v))
            else:
                print('data keys: ', v.keys(), '\n\n')
        # For these measurements the time was saved with the transmission
        transmission = np.array(doc.get('data')['transmission'])
        time = doc.get('data')['time']

        # We plot the raw data in Fig.1, which contains only the trapping signal without the rest of the trace
        fig, ax = plt.subplots(figsize=(7, 5.8))
        ax.plot(time, transmission, 'k')
        ax.set_xlabel('Time (sec)', size=20)
        ax.set_ylabel('Transmission (V)', size=20)
        ax.yaxis.set_ticks_position('both')
        ax.xaxis.set_ticks_position('both')
        ax.tick_params(axis='both', which='both', direction='in', labelsize=20)
        ax.set_title('Raw data', size=20)
        fig.tight_layout()

        # In order to compare many measurements, we must carry out a normalization:
        norm_factor = doc['normalization_factor']
        transmission = transmission / norm_factor
        print()
        fig3, ax3 = plt.subplots(figsize=(7, 5.8))
        ax3.plot(time, transmission, 'b')
        ax3.set_xlabel('Time (sec)', size=20)
        ax3.set_ylabel('Normalized transmission', size=20)
        ax3.yaxis.set_ticks_position('both')
        ax3.xaxis.set_ticks_position('both')
        ax3.tick_params(axis='both', which='both', direction='in', labelsize=20)
        ax3.set_title('Normalized Data', size=20)
        fig3.tight_layout()
        plt.show()