'''
once model trained make tarballs containing weights 
and dictonaries and embeddings for sharing.
Reads config from config.json
'''
import os, argparse, json
import tarfile

def _init_parser():
    parser = argparse.ArgumentParser(
        description="Create tar file containing weight and another containing dictionaries and embeddings."
    )
    parser.add_argument("-r", "--reverse", action="store_true", 
        help="swap translation order of languages in config file.")
    args = parser.parse_args()
    return args

def main():
    with open('config.json') as json_data_file:
        data = json.load(json_data_file)
    langs = data["languages"]
    args = _init_parser()
    work_dir = data["work_dir"]
    if args.reverse:
        langs.reverse()
    print("Configured for {} --> {}".format(langs[0], langs[1]))
    tarballs_dir = "tarballs"
    if not os.path.exists(tarballs_dir):
        os.makedirs(tarballs_dir)
    dct_emb_fn = "{}-{}-dct-embeddings.tar.gz".format(langs[0], langs[1])
    weights_fn = "{}-{}-weights.tar.gz".format(langs[0], langs[1])
    with tarfile.open(os.path.join(tarballs_dir, dct_emb_fn), "w:gz") as tar:
        for fn in os.listdir(work_dir):
            if "embeddings" in fn or "dictionary" in fn:
                tar.add(os.path.join(work_dir, fn))
    weights_dir = "weights/{}-{}".format(langs[0], langs[1])
    with tarfile.open(os.path.join(tarballs_dir, weights_fn), "w:gz") as tar:
        for fn in os.listdir(weights_dir):
            tar.add(os.path.join(weights_dir, fn))

if __name__ == "__main__":
    main()
