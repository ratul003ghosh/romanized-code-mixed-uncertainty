from datasets import load_dataset

# streaming=True reads rows one by one, without downloading everything
ds = load_dataset("aplycaebous/BanglaTLit", streaming=True)

print("Splits:", list(ds.keys()))

for split_name in ds.keys():
    print("\n=== Split:", split_name)
    for i, row in enumerate(ds[split_name]):
        if i == 0:
            print("Columns:", list(row.keys()))
        print(row)
        if i >= 4:
            break
