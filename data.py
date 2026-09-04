import pandas as pd

from pipline import ChronicDiseasePipeline
from visualization import Visualization


url = r"C:\Users\amrrrr\Downloads\Web scraper\data.csv"

df = pd.read_csv(url)

disease_columns = [
    "HeartDiseaseorAttack",
    "Stroke",
    "Diabetes_binary"
]

# Create the composite target before removing its source columns
df["chronic_disease"] = (
    df[disease_columns]
    .eq(1)
    .any(axis=1)
    .astype(int)
)

# Remove the source disease columns and the identifier
df = df.drop(
    columns=["id"] + disease_columns
)

print("Dataset shape:", df.shape)
print("Dataset columns:")
print(df.columns.tolist())

# Optional visualization
# visualizer = Visualization()
# visualizer.visuals(df)

# Create the machine-learning pipeline
pipeline = ChronicDiseasePipeline(
    target="chronic_disease",
    threshold=0.35
)

# Clean, split, remove training outliers, preprocess, and train
pipeline.fit(df)

# Evaluate on the untouched test set
results = pipeline.evaluate()