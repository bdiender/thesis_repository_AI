from allennlp.data import Vocabulary
from dataset_readers.dataset_reader import UniversalDependenciesReader


def main():
    datasets = ['de_hdt',
                'nl_lassysmall',
                'sv_talbanken',
                'cs_cac',
                'hu_szeged',
                'data/gsw/gsw_uzh-ud',
                'data/fao/fo_oft-ud',
                'data/hsb/hsb_ufal-ud',
                'data/vep/vep_vwt-ud']
    local = [d for d in datasets if d.startswith('data/')]

    instances = []
    for ds in datasets:
        for split in ['train', 'validation', 'test']:
            if (ds in local) and (split == 'validation'):
                reader = UniversalDependenciesReader(split='dev')
            else:
                reader = UniversalDependenciesReader(split=split)
            instances.extend(reader.read(ds))
    
    vocab = Vocabulary.from_instances(instances)
    print(vocab.save_to_files('master_vocab/'))


if __name__ == '__main__':
    main()