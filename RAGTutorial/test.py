import pandas as pd

df = pd.DataFrame(
    data={
        "id": ["A", "B"],
        "vector": [[1., 1., 1.], [1., 2., 3.]]
    }
)
print(df) 