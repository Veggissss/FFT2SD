import model_train

if __name__ == "__main__":
    model_types = ["encoder", "encoder-decoder", "decoder"]
    for model_type in model_types:
        model_train.train(model_type)
