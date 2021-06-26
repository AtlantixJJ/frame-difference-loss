from thirdparty.RAFT.core.datasets import DAVISDataset

ds = DAVISDataset(split="train")
print(ds.image_list[:10])