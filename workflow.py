"""
This is an example workflow
to illustrate some functions of the API

Created by: Martin Sicho
On: 26-11-19, 10:46
"""
import os
import re

import numpy as np
from rdkit import Chem
from tqdm import tqdm

import drugex
from drugex import Voc
from drugex.api.corpus import CorpusCSV, DataProvidingCorpus

from chembl_webresource_client.new_client import new_client
import pandas as pd

from drugex.api.environ.data import ChEMBLCSV
from drugex.api.environ.models import RF
from drugex.api.environ.serialization import FileEnvSerializer, FileEnvDeserializer
from drugex.api.model.callbacks import BasicMonitor
from drugex.api.pretrain.generators import BasicGenerator

DATA_DIR="data" # folder with input data files
OUT_DIR="output/workflow_out" # folder to store the output of this workflow
os.makedirs(OUT_DIR, exist_ok=True) # create the output folder

class CorpusChEMBL(DataProvidingCorpus):

    def __init__(self
                 , gene_names : list
                 , smiles_field="CANONICAL_SMILES"
                 , extracted_fields=(
                    "MOLECULE_CHEMBL_ID"
                    , "CANONICAL_SMILES"
                    , "PCHEMBL_VALUE"
                    , "ACTIVITY_COMMENT"
                    )
                 ):
        super().__init__()
        self.gene_names = gene_names
        self.extracted_fields = extracted_fields
        self.smiles_field=smiles_field
        self.CHEMBL_TARGETS = new_client.target
        self.CHEMBL_COMPOUNDS = new_client.molecule
        self.CHEMBL_ACTIVITIES = new_client.activity
        self.raw_data = pd.DataFrame()

    def _cleanRaw(self):
        subset = set(self.extracted_fields)
        subset.discard("ACTIVITY_COMMENT")
        self.raw_data = self.raw_data.dropna(subset=subset)
        for i, row in self.df.iterrows():
            # replacing the nitrogen electrical group to nitrogen atom "N"
            smile = row['CANONICAL_SMILES'].replace('[NH+]', 'N').replace('[NH2+]', 'N').replace('[NH3+]', 'N')
            # removing the radioactivity of each atom
            smile = re.sub('\[\d+', '[', smile)
            # reserving the largest fragments
            if '.' in smile:
                frags = smile.split('.')
                ix = np.argmax([len(frag) for frag in frags])
                smile = frags[ix]
            # Transforming into canonical SMILES based on the Rdkit built-in algorithm.
            self.df.loc[i, 'CANONICAL_SMILES'] = Chem.CanonSmiles(smile, 0)
            # removing molecule contained metal atom
            if '[Au]' in smile or '[As]' in smile or '[Hg]' in smile or '[Se]' in smile or smile.count('C') + smile.count('c') < 2:
                self.df = self.df.drop(i)
        self.raw_data = self.df

    def updateData(self, update_voc=False, sample=None):
        self.words = set()
        self.raw_data = pd.DataFrame()

        for gene_name in self.gene_names:
            target_chembl_ids = []
            for result in self.CHEMBL_TARGETS.filter(
                    target_synonym__icontains=gene_name
                    , target_type='SINGLE PROTEIN'
                    , organism='Homo sapiens'
            ):
                target_chembl_ids.append(result['target_chembl_id'])
            print("Found following target chembl IDs related to {0}".format(gene_name), target_chembl_ids)

            compound_data = dict()
            for field in self.extracted_fields:
                compound_data[field] = []
            for result in tqdm(self.CHEMBL_ACTIVITIES.filter(
                    target_chembl_id__in=target_chembl_ids
                    # , pchembl_value__isnull=False
            ), desc="Downloading compound data..."):
                for field in self.extracted_fields:
                    compound_data[field].append(result[field.lower()])
            self.raw_data = self.raw_data.append(pd.DataFrame(compound_data))

        # cleanup the raw data in this instance
        self._cleanRaw()

        tokens, canons, self.words = self.fromDataFrame(self.raw_data, self.voc, self.smiles_field, sample)

        # saving the canonical smiles and token sentences as a basis for future transformations
        self.df = pd.DataFrame()
        self.df[self.smiles_field] = canons
        self.df[self.token] = tokens
        self.df.drop_duplicates(subset=self.smiles_field, inplace=True)

        # rewrite the current voc instance if requested
        if update_voc:
            self.voc = Voc(chars=self.words)

        return self.df, self.voc


def data():
    ZINC_CSV=os.path.join(DATA_DIR, "ZINC.txt") # randomly selected sample from the ZINC database

    # Load structure data into a corpus from a CSV file (we assume
    # that we have the structures saved in a csv file in DATA_DIR).
    # This corpus will be used to train the exploitation network later.
    corpus_pre = CorpusCSV(
        update_file=ZINC_CSV
        # The input CSV file we will use to update the data in this corpus.

        , vocabulary=drugex.VOC_DEFAULT
        # The vocabulary object that defines the tokens
        # and other options used to construct and parse SMILES.
        # VOC_DEFAULT is a reasonable "catch all" default.

        , smiles_column="CANONICAL_SMILES"
        # Tells the corpus object what column to look for when
        # extracting SMILES to update the data.

        , sep='\t'
        # The column separator used in the CSV file
    )

    # Now we update the corpus (if we did not do it already).
    # It loads and tokenizes the SMILES it finds in the CSV.
    # In this particular case we request the vocabulary to be updated
    # and that we only take a random sample of 1000 molecules from
    # the file. The tokenized data and updated vocabulary are returned to us.
    corpus_out_zinc = os.path.join(OUT_DIR, "zinc_corpus.txt")
    vocab_out_zinc = os.path.join(OUT_DIR, "zinc_voc.txt")
    if not os.path.exists(corpus_out_zinc):
        df, voc = corpus_pre.updateData(update_voc=True, sample=100000)
        # We don't really use the return values here, but they are
        # still there if we need them for logging purposes or
        # something else.

        # All that remains is to just save our corpus data
        # so that we don't have to recalculate it again.
        # The CorpusCSV class has a method for that so lets use it.
        corpus_pre.saveCorpus(corpus_out_zinc)
        corpus_pre.saveVoc(vocab_out_zinc)
    else:
        # if we already initialized and saved
        # the corpus before, we just overwrite the
        # current one with the saved one
        corpus_pre= CorpusCSV.fromFiles(corpus_out_zinc, vocab_out_zinc)


    # We will also need a corpus for the exploration network.
    # This one we can load from ChEMBL using our customized
    # corpus class above.
    # It can query ChEMBL using a list of gene identifiers
    # and pulls all compounds that have activity data available.
    corpus_out_chembl = os.path.join(OUT_DIR, "chembl_corpus.txt")
    vocab_out_chembl = os.path.join(OUT_DIR, "chembl_voc.txt")
    raw_data_path = os.path.join(OUT_DIR, "ADORA2A.txt")
    if not os.path.exists(corpus_out_chembl):
        corpus_ex = CorpusChEMBL(["ADORA2A"])

        # lets update this corpus and save the results
        # (same procedure as above)
        df, voc = corpus_ex.updateData(update_voc=True)
        corpus_ex.saveCorpus(corpus_out_chembl)
        corpus_ex.saveVoc(vocab_out_chembl)

        # in addition we will also save the raw downloaded data
        corpus_ex.raw_data.to_csv(raw_data_path, sep="\t", index=False)
    else:
        # If we already generated the corpus file,
        # we can load it using the CorpusCSV class
        corpus_ex = CorpusCSV.fromFiles(corpus_out_chembl, vocab_out_chembl)

    # We also need activity data to
    # train the model which will provide the activity
    # values for policy gradient.
    # Luckily, we already have the file to do this:
    environ_data = ChEMBLCSV(raw_data_path, 6.5, id_col='MOLECULE_CHEMBL_ID')
    # This class not only loads the activity data,
    # but also provides access to it to the
    # QSAR learning algorithms (as we will see later).

    return corpus_pre, corpus_ex, environ_data

def environ(environ_data):
    # let's see if we can load the model from disk...
    identifier = 'environ_rf'
    des = FileEnvDeserializer(OUT_DIR, identifier)
    try:
        # The deserializer automatically looks for
        # a model in the given directory with the given identifier
        model = des.getModel()
        print("Model found at:", des.path)
        # return model
    except FileNotFoundError:
        print("Training environment model...")

        # we choose the random forest algorithm
        model = RF(train_provider=environ_data)
        model.fit()
        # we save the model if we need it later
        # we also choose to save the performance data
        ser = FileEnvSerializer(OUT_DIR, identifier, include_perf=True)
        ser.saveModel(model)

    return model

def main():
    # we need data first
    corpus_pre, corpus_ex, environ_data = data()

    # the "easiest" part comes first
    # the environment model for policy gradient
    environ_model = environ(environ_data)

    # Now we can pretrain our exploitation network.
    # This takes a long time and is a quite complex
    # calculation that we would like to monitor
    # as it is going on. We can use the Monitor
    # interface for that. The "BasicMonitor" just
    # saves log files and model checkpoints
    # in the given directory, but we could easily
    # implement our own monitor that could do much more
    # TODO: it would be nice to also have a method in
    # the monitor that would stop the training process
    pr_monitor = BasicMonitor(
        out_dir=OUT_DIR
        , identifier="pr"
    )

    # The monitor actually does more than just monitoring
    # of the process. It also keeps track of the best
    # model built yet and can be used to initialize
    # a generator based on that.
    # We use that feature below if there already is
    # a network state saved somewhere in our output directory.
    if not pr_monitor.getState(): # this will be False if the monitor cannot find an existing state
        pretrained = BasicGenerator(
            monitor=pr_monitor
            , corpus=corpus_pre
            , train_params={
                # these parameters are fed directly to the
                # fit method of the underlying pytorch model
                "epochs" : 3 # lets just make this one quick
            }
        )
        pretrained.pretrain()
    else:
        pretrained = BasicGenerator(
            monitor=pr_monitor
            , initial_state=pr_monitor # the monitor provides initial state
            , corpus=corpus_pre
        )
        # we will not do any training this time,
        # but we could just continue



if __name__ == "__main__":
    main()
