import tqdm

batch_start = range(10)

with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=True) as bar:
    print([x for x in bar])