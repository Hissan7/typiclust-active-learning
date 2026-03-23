from simclr.train_simclr import train_simclr
from simclr.extract_embeddings import extract_embeddings


def main():
    model_path = "models/simclr_resnet18.pth"

    train_simclr(
        epochs=5,
        batch_size=128,
        lr=1e-3,
        temperature=0.5,
        save_path=model_path
    )

    dataset, embeddings = extract_embeddings(model_path, train=True)
    print("Embedding shape:", embeddings.shape)
    print("Dataset size:", len(dataset))


if __name__ == "__main__":
    main()