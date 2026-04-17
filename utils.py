def print_val_loss(model):
    print("Val loss:")
    print(float(model.board.data["val_loss"][-1].y))