from enum import Enum

class DatasetWebPaths(Enum):
    transductive = "https://www.dropbox.com/s/6sbhm0rwo4l73jq/wikidata5m_transductive.tar.gz?dl=1"
    corpus = "https://www.dropbox.com/s/7jp4ib8zo3i6m10/wikidata5m_text.txt.gz?dl=1"
    entity_aliases = "https://www.dropbox.com/s/lnbhc8yuhit4wm5/wikidata5m_alias.tar.gz?dl=1"

class DatasetTarNames(Enum):
    transductive = "wikidata5m_transductive.tar.gz"
    corpus = "wikidata5m_text.txt.gz"
    entity_aliases = "wikidata5m_alias.tar.gz"

class DatasetNames(Enum):
    train = "wikidata5m_transductive_train.txt"
    validate = "wikidata5m_transductive_valid.txt"
    test = "wikidata5m_transductive_test.txt"
    corpus = "wikidata5m_text.txt"
