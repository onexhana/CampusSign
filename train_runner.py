from app.train import train_model

train_model(
    data_dir="npy_data",
    label_map_path="app/label_mapping.json",
    output_model_path="trained_model.pt"
)
