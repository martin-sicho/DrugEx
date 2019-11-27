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
from drugex.api.agent.agents import DrugExAgentTrainer
from drugex.api.agent.callbacks import BasicAgentMonitor
from drugex.api.agent.policy import PG
from drugex.api.corpus import CorpusCSV, DataProvidingCorpus, BasicCorpus

from chembl_webresource_client.new_client import new_client
import pandas as pd

from drugex.api.designer.designers import BasicDesigner, CSVConsumer
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
                 , clean_raw=False
                 , smiles_field="CANONICAL_SMILES"
                 , extracted_fields=(
                    "MOLECULE_CHEMBL_ID"
                    , "CANONICAL_SMILES"
                    , "PCHEMBL_VALUE"
                    , "ACTIVITY_COMMENT"
                    )
                 ):
        super().__init__(smiles_col=smiles_field)
        self.gene_names = gene_names
        self.extracted_fields = extracted_fields
        self.CHEMBL_TARGETS = new_client.target
        self.CHEMBL_COMPOUNDS = new_client.molecule
        self.CHEMBL_ACTIVITIES = new_client.activity
        self.raw_data = pd.DataFrame()
        self.clean_raw = clean_raw

    def _cleanRaw(self):
        subset = set(self.extracted_fields)
        subset.discard("ACTIVITY_COMMENT")
        self.raw_data = self.raw_data.dropna(subset=subset)
        for i, row in self.raw_data.iterrows():
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
            self.raw_data.loc[i, 'CANONICAL_SMILES'] = Chem.CanonSmiles(smile, 0)
            # removing molecule contained metal atom
            if '[Au]' in smile or '[As]' in smile or '[Hg]' in smile or '[Se]' in smile or smile.count('C') + smile.count('c') < 2:
                self.raw_data = self.raw_data.drop(i)
        self.raw_data = self.raw_data.sample(frac=1)

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
        if self.clean_raw:
            self._cleanRaw()

        tokens, canons, self.words = self.fromDataFrame(self.raw_data, self.voc, self.smiles_column, sample)

        self.resetDF(canons, tokens, update_voc)

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
    # The tokenized data and updated vocabulary are returned to us.
    corpus_out_zinc = os.path.join(OUT_DIR, "zinc_corpus.txt")
    vocab_out_zinc = os.path.join(OUT_DIR, "zinc_voc.txt")
    if not os.path.exists(corpus_out_zinc):
        df, voc = corpus_pre.updateData(update_voc=True)
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
    env_data_path = os.path.join(OUT_DIR, "ADORA2A.txt")
    if not os.path.exists(corpus_out_chembl):
        corpus_ex = CorpusChEMBL(["ADORA2A"], clean_raw=True)

        # lets update this corpus and save the results
        # (same procedure as above)
        df, voc = corpus_ex.updateData(update_voc=True)
        corpus_ex.saveCorpus(corpus_out_chembl)
        corpus_ex.saveVoc(vocab_out_chembl)

        # in addition we will also save the raw downloaded data
        corpus_ex.raw_data.to_csv(env_data_path, sep="\t", index=False)
    else:
        # If we already generated the corpus file,
        # we can load it using the CorpusCSV class
        corpus_ex = CorpusCSV.fromFiles(corpus_out_chembl, vocab_out_chembl)

    # Since we requested to update the vocabulary according to
    # tokens found in the underlying smiles for both the zinc
    # and ChEMBL corpus, we now need to unify them. Vocabularies
    # can be combined using the plus operator:
    voc_all = corpus_pre.voc + corpus_ex.voc
    corpus_pre.voc = voc_all
    corpus_ex.voc = voc_all
    # If we did not do this, the exploitation and
    # exploration networks would not be compatible
    # and we would not be able to train the final DrugEx model

    # We also need activity data to
    # train the model which will provide the activity
    # values for policy gradient.
    # Luckily, we already have the file to do this:
    environ_data = ChEMBLCSV(env_data_path, 6.5, id_col='MOLECULE_CHEMBL_ID') # or CMPD_CHEMBLID
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
        print("Pretraining exploitation network...")
        pretrained = BasicGenerator(
            monitor=pr_monitor
            , corpus=corpus_pre
            , train_params={
                # these parameters are fed directly to the
                # fit method of the underlying pytorch model
                "epochs" : 30 # lets just make this one quick
            }
        )
        pretrained.pretrain()
        # This method also has parameters
        # regarding partioning of the training data.
        # We just use the defaults in this case.
    else:
        pretrained = BasicGenerator(
            monitor=pr_monitor
            , initial_state=pr_monitor # the monitor provides initial state
            , corpus=corpus_pre
            # If we are not training this generator,
            # we could also omit this argument entirely.
            # If we did, the default vocabulary would be used
            # (not what we want now).
        )
        # we will not do any training this time,
        # but we could just continue by
        # specifying the training parameters and
        # calling pretrain again
        # TODO: maybe it would be nice if the monitor
        # keeps track of the settings as well

    # We will train the exploration network now.
    # The method is the same, but we use different
    # inputs. We define the monitor first:
    ex_monitor = BasicMonitor(
        out_dir=OUT_DIR
        , identifier="ex"
    )
    # The exploitation network fine-tunes the pretrained
    # one so we have to use the pr_monitor to initialize
    # its starting state:
    corpus_ex.voc = corpus_pre.voc
    if not ex_monitor.getState():
        print("Pretraining exploration network...")
        exploration = BasicGenerator(
            monitor=ex_monitor
            , initial_state=pr_monitor
            , corpus=corpus_ex # We use the target-specific corpus instead.
            , train_params={
                "epochs" : 30 # We will make this one quick too.
            }
        )
        exploration.pretrain(validation_size=512)
        # In this case we want to use a validation set.
        # This set will be used to estimate the
        # loss instead of the training set.
    else:
        exploration = BasicGenerator(
            monitor=ex_monitor
            , initial_state=ex_monitor
            , corpus=corpus_ex
        )

    # We have all ingredients to train
    # the DrugEx agent now. First, we
    # need to define the policy gradient
    # strategy:
    policy = PG( # So far this is the only policy there is in the API
        batch_size=512
        , mc=10
        , epsilon=0.01
        , beta=0.1
    )
    # DrugEx agents have their own monitors.
    # The basic one uses files and the same pattern
    # as we have seen with generators:
    identifier = 'e_%.2f_%.1f_%dx%d' % (policy.epsilon, policy.beta, policy.batch_size, policy.mc)
    agent_monitor = BasicAgentMonitor(OUT_DIR, identifier)
    # Finally, the DrugEx agent itself:
    if not agent_monitor.getState():
        print("Training DrugEx agent...")
        agent = DrugExAgentTrainer(
            agent_monitor # our monitor
            , environ_model # environment for the policy gradient
            , pretrained # the pretrained model
            , policy # our policy gradient implemntation
            , exploration # the fine-tuned model
            , {"n_epochs" : 30}
        )
        agent.train()
    else:
        # The DrugEx agent monitor also provides
        # a generator state -> it is the
        # best model as determined by
        # the agent trainer. We can
        # therefore create a generator based on this initial state
        agent = BasicGenerator(
            initial_state=agent_monitor
            , corpus=BasicCorpus(
                # if we are not training the generator,
                # we can just provide a basic corpus
                # that only provides vocabulary
                # and no corpus data -> we
                # only have to specify the right
                # vocabulary, which is the one of
                # the exploration or exploitation network
                # we choose the exploration network here:
                vocabulary=corpus_pre.voc
            )
        )

    # From a fully trained DrugEx agent,
    # we can create a designer class which
    # will handle sampling of SMILES:
    consumer = CSVConsumer(
        # this is
        os.path.join(OUT_DIR, 'designer_mols.csv')
    )
    designer = BasicDesigner(
        agent=agent # our agent
        , consumer=consumer # use this consumer to return results
        , n_samples=1000 # number of SMILES to sample in total
        , batch_size=512 # number of SMILES to sample in one batch
    )
    designer() # design the molecules
    consumer.save() # save the consumed molecules

if __name__ == "__main__":
    main()
