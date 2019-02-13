import torch


def validation(model, validation_loader, criterion, optimizer, gpu=True):
    test_loss = 0
    accuracy = 0

    for ii, (inputs, labels) in enumerate(validation_loader):
        optimizer.zero_grad()

        if torch.cuda.is_available() and gpu:
            inputs, labels = inputs.to('cuda:0'), labels.to('cuda:0')
            model.to('cuda:0')

        output = model.forward(inputs)
        test_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()

    return test_loss, accuracy


def train(model, criterion, optimizer, train_loader, validation_loader, epochs=6, gpu=True):
    print_every = 5
    steps = 0

    if torch.cuda.is_available() and gpu:
        model.to('cuda')

    for e in range(epochs):
        running_loss = 0
        for ii, (inputs, labels) in enumerate(train_loader):
            steps += 1

            if torch.cuda.is_available() and gpu:
                inputs, labels = inputs.to('cuda'), labels.to('cuda')

            optimizer.zero_grad()

            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()

                # Turn off gradients for validation, will speed up inference
                with torch.no_grad():
                    test_loss, accuracy = validation(model, validation_loader, criterion, optimizer, gpu)

                print(
                    "Epoch: {}/{}... ".format(e + 1, epochs),
                    "Training Loss: {:.3f}".format(running_loss / print_every),
                    "Test Lost {:.3f}".format(test_loss / len(validation_loader)),
                    "Test Accuracy: {:.3f}".format(accuracy / len(validation_loader)),
                )

                running_loss = 0

                # Make sure dropout and grads are on for training
                model.train()
